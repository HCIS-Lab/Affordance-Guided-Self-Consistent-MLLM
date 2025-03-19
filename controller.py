from isaacgym import gymapi

import torch
import inspect
from typing import List

from src.utils import euler_to_quaternion, quaternion_to_euler, orientation_error
from environment import IsaacSim


class Controller:
    def __init__(self, isaac_sim: IsaacSim):
        self.isaac_sim = isaac_sim
        self.device = isaac_sim.device
        self.current_action = 'idle'
        self.action_stage = -1
        self.traj = []
        self._gripper_control_cnt = 0
        
    def _empty_action(self) -> torch.tensor:
        return torch.tensor([[0.], [0.], [0.], [0.], [0.], [0.]], device=self.device)
    
    def low_level_control(self, ctrl, delta=0.05) -> torch.tensor:
        low_level_dict = {
            "up": torch.tensor([[[0.],[0.],[1.],[0.],[0.],[0.]]]),
            "down": torch.tensor([[[0.],[0.],[-1.],[0.],[0.],[0.]]]),
            "left": torch.tensor([[[0.],[-1.],[0.],[0.],[0.],[0.]]]),
            "right": torch.tensor([[[0.],[1.],[0.],[0.],[0.],[0.]]]),
            "backward": torch.tensor([[[-1.],[0.],[0.],[0.],[0.],[0.]]]),
            "forward": torch.tensor([[[1.],[0.],[0.],[0.],[0.],[0.]]]),
            "turn_left": torch.tensor([[[0.],[0.],[0.],[0.],[0.],[-10.]]]),
            "turn_right": torch.tensor([[[0.],[0.],[0.],[0.],[0.],[10.]]]),
            "turn_up": torch.tensor([[[0.],[0.],[0.],[0.],[10.],[0.]]]),
            "turn_down": torch.tensor([[[0.],[0.],[0.],[0.],[-10.],[0.]]]), 
            "rot_right": torch.tensor([[[0.],[0.],[0.],[1.],[0.],[0.]]]),
            "rot_left": torch.tensor([[[0.],[0.],[0.],[-1.],[0.],[0.]]]),
            "idle": self._empty_action()
        }
        if ctrl not in low_level_dict.keys():
            print(f"Unsupported Control: {ctrl}")
            return self._empty_action()
        return low_level_dict[ctrl] * delta
        
    def apply_action(self, action, pos_offset=0.01, axis_offset=0.03, w_offset=0.03) -> torch.tensor:
        # assert action in self.action_list, f"Unsupported Action {action}"
        if action != self.current_action and self.current_action != 'idle':
            print(f'{self.current_action} -> {action}')
            self.action_stage = -1
        self.current_action = action
        
        hand_pos:torch.tensor = self.isaac_sim.rb_state_tensor[self.isaac_sim.franka_hand_indices, :3]
        hand_rot:torch.tensor = self.isaac_sim.rb_state_tensor[self.isaac_sim.franka_hand_indices, 3:7]
        
        if self.action_stage == -1:
            self.traj = self.get_trajectory(self.current_action)
            self.action_stage = 0
            if self.current_action != 'idle':
                print(f"Start {self.current_action}, step: {len(self.traj)}")
            return self._empty_action()
        
        elif self.action_stage == len(self.traj):
            if self.current_action != 'idle':
                print(f'Finish {self.current_action}')
                self.current_action = 'idle'
            self.action_stage = -1
            return self._empty_action()
        
        goal_pose = self.traj[self.action_stage]
        goal_pos = goal_pose[..., :3]
        goal_rot = goal_pose[..., 3:]
        
        to_goal = goal_pos - hand_pos
        pos_dist = torch.norm(to_goal, dim=1).unsqueeze(-1)
        # l2_norm = torch.norm(to_goal, p=2)
        # to_goal = to_goal / l2_norm
        # target_square_sum = 0.005
        # to_goal = to_goal * torch.sqrt(torch.tensor(target_square_sum))
        to_axis = goal_rot[:, :3] - hand_rot[:, :3]
        axis_dist = torch.norm(to_axis, dim=1).unsqueeze(-1)
        w_dist = goal_rot[:, -1] - hand_rot[:, -1]
        w_dist = abs(w_dist)
        pos_err = torch.where(pos_dist > pos_offset, goal_pos - hand_pos, torch.tensor([0., 0., 0.], device=self.device))
        orn_err = torch.where(axis_dist > axis_offset or w_dist > w_offset, orientation_error(goal_rot, hand_rot), torch.tensor([0., 0., 0.], device=self.device))
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
        
        if pos_dist <= pos_offset and axis_dist <= axis_offset and w_dist <= w_offset:
            if self._gripper_control():
                return self._empty_action()
            self.action_stage += 1
            self._gripper_control_cnt = 0
            print(f'{self.current_action}: {self.action_stage}')
        
        return dpose * self.get_delta(self.current_action, self.action_stage)
    
    def _gripper_control(self, wait_time=100):
        gripper_open = self.isaac_sim.franka_dof_upper_limits[7:]
        gripper_close = self.isaac_sim.franka_dof_lower_limits[7:]
        if self.current_action == 'grasp_spoon': 
            if self.action_stage < 1:
                self.isaac_sim.pos_action[:, 7:9] = gripper_open
            elif self.action_stage == 1:
                if self._gripper_control_cnt < wait_time:
                    self._gripper_control_cnt += 1
                    self.isaac_sim.pos_action[:, 7:9] = gripper_close
                    return True
            else:
                self.isaac_sim.pos_action[:, 7:9] = gripper_close
        elif self.current_action == 'put_spoon_back': # open at 4
            if self.action_stage < 3:
                self.isaac_sim.pos_action[:, 7:9] = gripper_close
            elif self.action_stage == 3:
                if self._gripper_control_cnt < wait_time:
                    self._gripper_control_cnt += 1
                    self.isaac_sim.pos_action[:, 7:9] = gripper_open
                    return True
            else:
                self.isaac_sim.pos_action[:, 7:9] = gripper_open
                
        elif self.current_action == 'pull_bowl_closer': 
            if self.action_stage <= 1:
                self.isaac_sim.pos_action[:, 7:9] = gripper_open
            elif self.action_stage == 2:
                if self._gripper_control_cnt < wait_time:
                    self._gripper_control_cnt += 1
                    self.isaac_sim.pos_action[:, 7:9] = gripper_close
                    return True
            elif 2 < self.action_stage < 4:
                self.isaac_sim.pos_action[:, 7:9] = gripper_close
            elif self.action_stage == 4:
                if self._gripper_control_cnt < wait_time:
                    self._gripper_control_cnt += 1
                    self.isaac_sim.pos_action[:, 7:9] = gripper_open
                    return True
            else:
                self.isaac_sim.pos_action[:, 7:9] = gripper_open
                
        elif self.current_action == 'open_dumbwaiter': # close from 2-7
            if self.action_stage < 1 or self.action_stage > 7:
                self.isaac_sim.pos_action[:, 7:9] = gripper_open
            elif self.action_stage == 1:
                if self._gripper_control_cnt < wait_time:
                    self._gripper_control_cnt += 1
                    self.isaac_sim.pos_action[:, 7:9] = gripper_close
                    return True
            else:
                self.isaac_sim.pos_action[:, 7:9] = gripper_close
        elif self.current_action == 'close_dumbwaiter': # close from 1-9
            if self.action_stage < 1 or self.action_stage > 9:
                self.isaac_sim.pos_action[:, 7:9] = gripper_open
            elif self.action_stage == 1:
                if self._gripper_control_cnt < wait_time:
                    self._gripper_control_cnt += 1
                    self.isaac_sim.pos_action[:, 7:9] = gripper_close
                    return True
            else:
                self.isaac_sim.pos_action[:, 7:9] = gripper_close
        elif self.current_action == 'put_bowl_into_dumbwaiter': # close from 2-7 and after 10
            if self.action_stage == 1:
                if self._gripper_control_cnt < wait_time:
                    self._gripper_control_cnt += 1
                    self.isaac_sim.pos_action[:, 7:9] = gripper_open
                    return True
            elif self.action_stage == 2:
                if self._gripper_control_cnt < wait_time:
                    self._gripper_control_cnt += 1
                    self.isaac_sim.pos_action[:, 7:9] = gripper_close
                    return True
            elif self.action_stage < 2 or self.action_stage > 7:
                self.isaac_sim.pos_action[:, 7:9] = gripper_open
            elif self.action_stage > 9:
                self.isaac_sim.pos_action[:, 7:9] = gripper_open
            else:
                self.isaac_sim.pos_action[:, 7:9] = gripper_close
        return False
    
    def get_delta(self, action: str, action_stage) -> float:
        delta = {
            'move': [1],
            'grasp_spoon': [2],
            'put_spoon_back': [0.8],
            'scoop': [3., 3., 3., 3., 6.],
            'drop_food': [1.8, 1.8, 2],
            'open_dumbwaiter': [2, 2, 2, 2, 1, 1, 1, 1, 1, 2],
            'close_dumbwaiter': [2],
            'start_dumbwaiter': [2, 2, 2, 2, 2, 2, 1, 1, 1, 2],
            'pull_bowl_closer': [2, 2, 2, 0.4, 2, 2],
            'put_bowl_into_dumbwaiter': [2, 2, 2, 1],
        }
        if action not in delta.keys():
            return 1
        return delta[action][action_stage if action_stage < len(delta[action]) else -1]
    
    def get_trajectory(self, action) -> List[torch.Tensor]:
        ## TODO
        ## move, take_tool, put_tool, scoop, put_food, open_dumbwaiter, close_dumbwaiter, start_dumbwaiter, (stir)
        """_summary_

        Args:
            action (str): move, grasp_spoon, put_spoon_back, scoop, drop_food, open_dumbwaiter, close_dumbwaiter, start_dumbwaiter, pull_bowl_closer, put_bowl_into_dumbwaiter

        Returns:
            List[torch.Tensor]: _description_
        """
        if action == "idle":
            return []
        
        target_object = None
        if "move" in action:
            target_object = action.replace("move_to_", "").split()[0]
            action = "move"
            
        traj_dict = {
            "move": self._move_traj,
            "grasp_spoon": self._take_tool_traj,
            "put_spoon_back": self._put_tool_traj,
            "scoop": self._scoop_traj,
            "drop_food": self._scoop_put_traj,
            "open_dumbwaiter": self._open_dumbwaiter_traj,
            "close_dumbwaiter": self._close_dumbwaiter_traj,
            "start_dumbwaiter": self._start_dumbwaiter_traj,
            "pull_bowl_closer": self._pull_bowl_traj,
            "put_bowl_into_dumbwaiter": self._put_bowl_into_dumbwaiter_traj,
        }
        
        traj = traj_dict.get(action, None)
        if traj is None:
            print(f"Unsupport action: {action}")
            return []
        hand_pos = self.isaac_sim.rb_state_tensor[self.isaac_sim.franka_hand_indices, :3]
        hand_rot = self.isaac_sim.rb_state_tensor[self.isaac_sim.franka_hand_indices, 3:7]
        params = {
            "hand_pos": hand_pos,  
            "hand_rot": hand_rot,  
            "object": target_object,
            "tool": "spoon",
        }
        param_names = inspect.signature(traj).parameters.keys()
        params = {k: v for k, v in params.items() if k in param_names}
        pos_set, rot_set = traj(**params)
        return [torch.cat([pos, rot], dim=1) for pos, rot in zip(pos_set, rot_set)]
    
    def _move_traj(self, hand_rot, object):
        object_type = "tool" if object in self.isaac_sim.tool_list else "container"
        indices_list = {
            "tool": self.isaac_sim.tool_indices,
            "container": self.isaac_sim.containers_indices
        }
        for key in indices_list[object_type].keys():
            if object in key:
                object_indice = indices_list[object_type][key]
                break
        object_pos = self.isaac_sim.rb_state_tensor[object_indice, :3] + torch.tensor([-0.05, 0, 0.4], device=self.device)
        return [object_pos], [hand_rot]
    
    def _take_tool_traj(self, tool):
        tool_pos = self.isaac_sim.rb_state_tensor[self.isaac_sim.tool_indices[tool], :3]
        tool_rot = self.isaac_sim.rb_state_tensor[self.isaac_sim.tool_indices[tool], 3:7]
        rot = gymapi.Quat(tool_rot[:, 0], tool_rot[:, 1], tool_rot[:, 2], tool_rot[:, 3])
        roll, pitch, yaw = quaternion_to_euler(rot)
        roll += 3.14
        rot = euler_to_quaternion(roll, pitch, yaw)
        hold_hight = 0.002
        
        pos_set = [
            torch.tensor([[tool_pos[:, 0], tool_pos[:, 1], tool_pos[:, 2] + 0.17]], device=self.device),
            torch.tensor([[tool_pos[:, 0], tool_pos[:, 1], tool_pos[:, 2] + 0.095 - hold_hight]], device=self.device),
            torch.tensor([[tool_pos[:, 0], tool_pos[:, 1], tool_pos[:, 2] + 0.13]], device=self.device),
            torch.tensor([[tool_pos[:, 0], tool_pos[:, 1] + 0.05, tool_pos[:, 2] + 0.13]], device=self.device),
            torch.tensor([[tool_pos[:, 0], tool_pos[:, 1] + 0.05, tool_pos[:, 2] + 0.2]], device=self.device)
        ]
        rot_set = [torch.tensor([[rot.x, rot.y, rot.z, rot.w]], device=self.device)] * len(pos_set)
        return pos_set, rot_set
    
    def _put_tool_traj(self, tool):
        tool_pos = self.isaac_sim.tool_pose[tool].p
        tool_rot = self.isaac_sim.tool_pose[tool].r
        rot = tool_rot
        roll, pitch, yaw = quaternion_to_euler(rot)
        roll += 3.14
        rot = euler_to_quaternion(roll, pitch, yaw)
        pos_set = [
            torch.tensor([[tool_pos.x, tool_pos.y + 0.05, tool_pos.z + 0.18]], device=self.device),
            torch.tensor([[tool_pos.x, tool_pos.y + 0.05, tool_pos.z + 0.15]], device=self.device),
            torch.tensor([[tool_pos.x, tool_pos.y - 0.0026, tool_pos.z + 0.13]], device=self.device),
            torch.tensor([[tool_pos.x, tool_pos.y - 0.0026, tool_pos.z + 0.085]], device=self.device),
            torch.tensor([[tool_pos.x, tool_pos.y - 0.0026, tool_pos.z + 0.2]], device=self.device)
        ]
        rot_set = [torch.tensor([[rot.x, rot.y, rot.z, rot.w]], device=self.device)] * len(pos_set)
        return pos_set, rot_set
    
    def _scoop_traj(self, hand_pos):
        init_pos = hand_pos.clone()
        init_pos[:, 2] = 0
        best_tensor = self._find_nearest_container(init_pos)
        init_pos = best_tensor - torch.tensor([[0.01, -0.02, 0.0]], device=self.device)
        init_pos[0][2] = 0.035
        init_pos[0][1] += 0.0127
        #init_pos[0][0] -= 0.02
        # original pos: 0.3871, 0.0877, container pos: 0.43999999999999995, 0.07500000000000001
        pos_set = [
            init_pos + torch.tensor([[0.0000, 0.0000, 0.7836]], device=self.device),
            init_pos + torch.tensor([[-0.0300,  0.0000,  0.7228]], device=self.device),
            init_pos + torch.tensor([[-0.0299,  0.0109,  0.6728]], device=self.device),
            init_pos + torch.tensor([[-0.0271,  0.0050,  0.6808]], device=self.device),
            init_pos + torch.tensor([[-0.0264,  0.0054,  0.6716]], device=self.device),
            init_pos + torch.tensor([[0.0169, 0.0043, 0.6700]], device=self.device),
            init_pos + torch.tensor([[-0.0288,  0.0043,  0.6638]], device=self.device),
            init_pos + torch.tensor([[-0.0295,  0.0043,  0.6638]], device=self.device),
            init_pos + torch.tensor([[-0.0300,  0.0038,  0.6650]], device=self.device),
            init_pos + torch.tensor([[-0.0352,  0.0035,  0.6668]], device=self.device),
            init_pos + torch.tensor([[-0.0505,  0.0031,  0.6698]], device=self.device),
            init_pos + torch.tensor([[-0.0807,  0.0022,  0.6728]], device=self.device),
            init_pos + torch.tensor([[-0.1024,  0.0019,  0.6758]], device=self.device),
            init_pos + torch.tensor([[-0.1016,  0.0021,  0.6708]], device=self.device),
            init_pos + torch.tensor([[-0.1370,  0.0019,  0.6808]], device=self.device),
            init_pos + torch.tensor([[-0.1830,  0.0017,  0.7000]], device=self.device)
        ]

        rot_set = [
            torch.tensor([[ 0.9945,  0.0413, -0.0809,  0.0523]], device=self.device),
            torch.tensor([[ 0.9945,  0.0410, -0.0808,  0.0522]], device=self.device),
            torch.tensor([[ 0.9953,  0.0393, -0.0701,  0.0542]], device=self.device),
            torch.tensor([[ 0.9976,  0.0345, -0.0243,  0.0549]], device=self.device),
            torch.tensor([[ 0.9977,  0.0345, -0.0176,  0.0563]], device=self.device),
            torch.tensor([[0.9975, 0.0310, 0, 0.0575]], device=self.device),
            torch.tensor([[0.9975, 0.0310, 0.0288, 0.0575]], device=self.device),
            torch.tensor([[0.9965, 0.0388, 0.0500, 0.0580]], device=self.device),
            torch.tensor([[0.9952, 0.0278, 0.0736, 0.0586]], device=self.device),
            torch.tensor([[0.9925, 0.0257, 0.1034, 0.0594]], device=self.device),
            torch.tensor([[0.9869, 0.0225, 0.1478, 0.0604]], device=self.device),
            torch.tensor([[0.9621, 0.0144, 0.2648, 0.0626]], device=self.device),
            torch.tensor([[0.9340, 0.0089, 0.3514, 0.0637]], device=self.device),
            torch.tensor([[0.9337, 0.0093, 0.4071, 0.0639]], device=self.device),
            torch.tensor([[0.8848, 0.0023, 0.5116, 0.0641]], device=self.device),
            torch.tensor([[0.8844, 0.0025, 0.5116, 0.0640]], device=self.device)
        ]
        return pos_set, rot_set
    
    def _scoop_put_traj(self, hand_pos):
        init_pos = hand_pos.clone()
        init_pos[:, 2] = 0
        best_tensor = self._find_nearest_container(init_pos)
        init_pos = best_tensor - torch.tensor([[0.01, -0.02, 0.0]], device=self.device)
        init_pos[0][2] = 0.03
        init_pos[0][1] += 0.0127
        #init_pos[0][0] -= 0.02
        # original pos: 0.3871, 0.0877, container pos: 0.43999999999999995, 0.07500000000000001
        x_shift = 0.03
        z_shift = 0.15
        pos_set = [ 
            init_pos + torch.tensor([[-x_shift-0.1024,  0.0019,  0.6758+z_shift]], device=self.device), 
            init_pos + torch.tensor([[-x_shift-0.0807,  0.0022,  0.6728+z_shift]], device=self.device), 
            init_pos + torch.tensor([[-x_shift-0.0505,  0.0031,  0.6698+z_shift]], device=self.device), 
            init_pos + torch.tensor([[-x_shift-0.0352,  0.0035,  0.6668+z_shift]], device=self.device), 
            init_pos + torch.tensor([[-x_shift-0.0300,  0.0038,  0.6650+z_shift]], device=self.device), 
            init_pos + torch.tensor([[-x_shift-0.0295,  0.0043,  0.6638+z_shift]], device=self.device), 
            init_pos + torch.tensor([[-x_shift-0.0288,  0.0043,  0.6638+z_shift]], device=self.device), 
            init_pos + torch.tensor([[-x_shift+0.0169, 0.0043, 0.6700+z_shift]], device=self.device), 
            init_pos + torch.tensor([[-x_shift+0.0000, 0.0043, 0.6700+z_shift]], device=self.device), 
            init_pos + torch.tensor([[-x_shift+0.0000, 0.0000, 0.7836+z_shift]], device=self.device),
        ]
        rot_set = [ 
            torch.tensor([[0.9340, 0.0089, 0.3514, 0.0637]], device=self.device), 
            torch.tensor([[0.9621, 0.0144, 0.2648, 0.0626]], device=self.device), 
            torch.tensor([[0.9869, 0.0225, 0.1478, 0.0604]], device=self.device), 
            torch.tensor([[0.9925, 0.0257, 0.1034, 0.0594]], device=self.device), 
            torch.tensor([[0.9952, 0.0278, 0.0736, 0.0586]], device=self.device), 
            torch.tensor([[0.9965, 0.0388, 0.0500, 0.0580]], device=self.device), 
            torch.tensor([[0.9975, 0.0310, 0.0288, 0.0575]], device=self.device), 
            torch.tensor([[0.9975, 0.0310, 0.0000, 0.0575]], device=self.device),
            torch.tensor([[ 0.9893,  0.0362, -0.1308,  0.0528]], device=self.device),
            torch.tensor([[ 0.9945,  0.0413, -0.0809,  0.0523]], device=self.device),
        ]
        return pos_set, rot_set
    
    def _open_dumbwaiter_traj(self):
        offset = 0.041
        pos_set = [
            torch.tensor([[0.5815, 0.2740, 0.63]], device=self.device),

            torch.tensor([[0.6021, 0.3101 + offset, 0.6262 - 0.015]], device=self.device),
            
            # torch.tensor([[0.6021, 0.321 + offset, 0.6262]], device=self.device),
            
            # torch.tensor([[0.58, 0.3053, 0.6002]], device=self.device)
            
            torch.tensor([[0.5865, 0.2111 + offset, 0.6281 - 0.015]], device=self.device),
            torch.tensor([[0.5528, 0.1524 + offset, 0.6283 - 0.015]], device=self.device),
            torch.tensor([[0.4601, 0.0769 + offset, 0.6280 - 0.015]], device=self.device),
            torch.tensor([[0.2797 - 0.05, 0.0333 + offset - 0.008, 0.6281 - 0.015]], device=self.device),
            torch.tensor([[0.2797 - 0.05, 0.0333 + offset - 0.01, 0.6281 - 0.015]], device=self.device),
            torch.tensor([[0.2851, -0.0098 + offset - 0.01, 0.6229]], device=self.device),
            torch.tensor([[0.3119, 0.0020 + offset - 0.01, 0.8432]], device=self.device)
        ]

        rot_set = [
            torch.tensor([[ 0.5699, 0.4882, 0.4497, -0.4844]], device=self.device),
            torch.tensor([[ 0.5096, 0.5654, 0.4241, -0.4906]], device=self.device),
            torch.tensor([[ 0.5193, 0.5606, 0.4172, -0.4920]], device=self.device),
            torch.tensor([[ 0.5088, 0.5718, 0.4091, -0.4968]], device=self.device),
            torch.tensor([[ 0.5470, 0.5337, 0.4521, -0.4600]], device=self.device),
            torch.tensor([[ 0.5951, 0.4882, 0.4980, -0.3993]], device=self.device),
            torch.tensor([[ 0.5951, 0.4882, 0.4980, -0.3993]], device=self.device),
            torch.tensor([[ 0.5951, 0.4882, 0.4980, -0.3993]], device=self.device),
            torch.tensor([[ 0.9634, 0.1132, 0.2373, -0.0520]], device=self.device)
        ]
        return pos_set, rot_set
    
    def _close_dumbwaiter_traj(self):
        offset = 0.06
        # original pos: 0.3871, 0.0877, container pos: 0.43999999999999995, 0.07500000000000001
        pos_set = [
            torch.tensor([[0.3119, 0.0020 + offset, 0.8432]], device=self.device),
            torch.tensor([[0.2551, -0.0098 + offset, 0.6229]], device=self.device),
            torch.tensor([[0.2851, -0.0098 + offset, 0.6229]], device=self.device),
            torch.tensor([[0.2797, 0.0333 + offset, 0.6281]], device=self.device),
            torch.tensor([[0.2797, 0.0333 + offset, 0.6281]], device=self.device),
            torch.tensor([[0.4601, 0.0769 + offset, 0.6280]], device=self.device),
            torch.tensor([[0.5528, 0.1524 + offset, 0.6283]], device=self.device),
            torch.tensor([[0.5865, 0.2111 + offset, 0.6281]], device=self.device),
            torch.tensor([[0.6081, 0.3025, 0.6262]], device=self.device),
            torch.tensor([[0.5499, 0.2111, 0.6341]], device=self.device)
        ]

        rot_set = [
            torch.tensor([[ 0.9634, 0.1132, 0.2373, -0.0520]], device=self.device),
            torch.tensor([[ 0.5951, 0.4882, 0.4980, -0.3993]], device=self.device),
            torch.tensor([[ 0.5951, 0.4882, 0.4980, -0.3993]], device=self.device),
            torch.tensor([[ 0.5951, 0.4882, 0.4980, -0.3993]], device=self.device),
            torch.tensor([[ 0.5951, 0.4882, 0.4980, -0.3993]], device=self.device),
            torch.tensor([[ 0.5470, 0.5337, 0.4521, -0.4600]], device=self.device),
            torch.tensor([[ 0.5088, 0.5718, 0.4091, -0.4968]], device=self.device),
            torch.tensor([[ 0.5193, 0.5606, 0.4172, -0.4920]], device=self.device),
            torch.tensor([[ 0.5096, 0.5654, 0.4241, -0.4906]], device=self.device),
            torch.tensor([[ 0.5329,  0.5489,  0.4332, -0.4765]], device=self.device),
            torch.tensor([[ 0.9963, 0.0367, 0.0521, 0.0577]], device=self.device)

        ]
        return pos_set, rot_set
    
    def _start_dumbwaiter_traj(self):
        pos_set = [
            torch.tensor([[0.6581, 0.2500, 0.6262]], device=self.device),
            torch.tensor([[0.6581, 0.3025, 0.6341]], device=self.device),
            torch.tensor([[0.6581, 0.2500, 0.6262]], device=self.device)
        ]
        rot_set = [
            torch.tensor([[ 0.5329, 0.5489, 0.4332, -0.4765]], device=self.device),
            torch.tensor([[ 0.5329, 0.5489, 0.4332, -0.4765]], device=self.device),
            torch.tensor([[ 0.5329, 0.5489, 0.4332, -0.4765]], device=self.device)
        ]
        return pos_set, rot_set
    
    def _pull_bowl_traj(self, hand_pos):
        init_pos = hand_pos.clone()
        init_pos[:, 2] = 0
        best_tensor = self._find_nearest_container(init_pos)
        init_pos = best_tensor - torch.tensor([[0.01, -0.02, 0.0]], device=self.device)
        init_pos[0][2] = 0.03
        init_pos[0][1] += 0.0127
        
        pos_set = [
            init_pos + torch.tensor([[-0.07, -0.07, 0.7836]], device=self.device),
            init_pos + torch.tensor([[-0.07, -0.07,  0.57]], device=self.device),
            init_pos + torch.tensor([[-0.07, -0.07,  0.57]], device=self.device),
            torch.tensor([[0.45,0,0.62]], device=self.device),
            torch.tensor([[0.45,0,0.62]], device=self.device),
            torch.tensor([[0.45,0,0.8]], device=self.device),
        ]

        rot_set = [
            torch.tensor([[ 0.8973, -0.4209,  0.1325,  0.0101]], device=self.device),
            torch.tensor([[ 0.8973, -0.4209,  0.1325,  0.0101]], device=self.device),
            torch.tensor([[ 0.8973, -0.4209,  0.1325,  0.0101]], device=self.device),
            torch.tensor([[ 0.8973, -0.4209,  0.1325,  0.0101]], device=self.device),
            torch.tensor([[ 0.8973, -0.4209,  0.1325,  0.0101]], device=self.device),
            torch.tensor([[ 0.8973, -0.4209,  0.1325,  0.0101]], device=self.device)
        ]
        return pos_set, rot_set
    
    def _put_bowl_into_dumbwaiter_traj(self, hand_pos):
        init_pos = hand_pos.clone()
        init_pos[:, 2] = 0
        best_tensor = self._find_nearest_container(init_pos)
        init_pos = best_tensor - torch.tensor([[0.01, -0.02, 0.0]], device=self.device)
        init_pos[0][2] = 0.03
        init_pos[0][1] += 0.0127
        offset = 0.02
        pos_set = [
            init_pos + torch.tensor([[-0.07, -0.07, 0.7836]], device=self.device),
            init_pos + torch.tensor([[-0.07, -0.07,  0.6]], device=self.device),
            init_pos + torch.tensor([[-0.07, -0.07,  0.6]], device=self.device),
            init_pos + torch.tensor([[-0.07, -0.07,  0.6]], device=self.device),
            torch.tensor([[0.4794 - offset, 0.1503 + offset, 0.6256]], device=self.device),
            torch.tensor([[0.4794 - offset, 0.2803 + offset, 0.6277]], device=self.device),
            torch.tensor([[0.4794 - offset, 0.3203 + offset, 0.6277]], device=self.device),
            torch.tensor([[0.4794 - offset, 0.3321 + offset, 0.6277]], device=self.device),
            torch.tensor([[0.4794 - offset, 0.3321 + offset, 0.6700]], device=self.device),
            torch.tensor([[0.4794 - offset, 0.2621 + offset, 0.6700]], device=self.device),
            torch.tensor([[0.4794 - offset, 0.2621 + offset, 0.6277]], device=self.device),
            
        ]

        rot_set = [
            torch.tensor([[ 0.8973, -0.4209,  0.1325,  0.0101]], device=self.device),
            torch.tensor([[ 0.8973, -0.4209,  0.1325,  0.0101]], device=self.device),
            torch.tensor([[ 0.8973, -0.4209,  0.1325,  0.0101]], device=self.device),
            torch.tensor([[ 0.9864,  0.0982,  0.1127, -0.0682]], device=self.device),
            torch.tensor([[ 0.9842, -0.0319,  0.0889, -0.1498]], device=self.device),
            torch.tensor([[ 0.9842, -0.0319,  0.0889, -0.1498]], device=self.device),
            torch.tensor([[ 0.9851,  0.0639,  0.0880, -0.1335]], device=self.device),
            torch.tensor([[ 0.9851,  0.0639,  0.0880, -0.1335]], device=self.device),
            torch.tensor([[ 0.9851,  0.0639,  0.0880, -0.1335]], device=self.device),
            torch.tensor([[ 0.9851,  0.0639,  0.0880, -0.1335]], device=self.device),
            torch.tensor([[ 0.9851,  0.0639,  0.0880, -0.1335]], device=self.device),
        ]
        return pos_set, rot_set
    
    def _find_nearest_container(self, init_pos):
        min_dist = float("inf")
        best_tensor = init_pos
        for indice in self.isaac_sim.containers_indices.values():
            container_p = self.isaac_sim.rb_state_tensor[indice, :3]
            container_p[..., -1] = 0
            # print(f"container pos: {container_p}, init pos: {init_pos}")
            dist = torch.norm(container_p - init_pos, dim=1)
            if dist < min_dist:
                min_dist = dist
                best_tensor = container_p
        return best_tensor