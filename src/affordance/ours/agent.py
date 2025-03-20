import sys
import torch
import numpy as np
import cv2
import open3d as o3d
import pytorch3d.transforms
from typing import List
from isaacgym.torch_utils import quat_mul, quat_conjugate

sys.path.append('/home/hcis-s17/multimodal_manipulation/scoop_env/')
from src.affordance import Affordance_agent
# from src.vild.utils import get_vild_prob

def get_prob(rgb_img_path, prompt_list, params, image_root):
    pass


def pose7d_to_matrix(pose7d: torch.tensor):
    matrix = torch.eye(4, dtype=torch.float32).reshape(1, 4, 4).repeat(pose7d.shape[0], 1, 1)
    matrix[:, :3, :3] = pytorch3d.transforms.quaternion_to_matrix(pose7d[:, [6, 3, 4, 5]])
    matrix[:, :3, 3] = pose7d[:, :3]

    return matrix

def matrix_to_pose_7d(matrix: torch.tensor):
    pose_7d = torch.zeros((matrix.shape[0], 7), dtype=torch.float32)
    pose_7d[:, 3:] = pytorch3d.transforms.matrix_to_quaternion(matrix[:, :3, :3])[:, [1, 2, 3, 0]]
    pose_7d[:, :3] = matrix[:, :3, 3]

    return pose_7d

class Affordance_agent_ours(Affordance_agent):
    def __init__(self, init_object_list, action_list, DH_params, joint_limit, base_pose, device):
        super().__init__(init_object_list, action_list)
        self.image_root = '/home/hcis-s17/multimodal_manipulation/scoop_env/src/affordance/ours/image'
        self.pos_offset = 0.01
        self.axis_offset = 0.03
        self.w_offset = 0.03
        self.DH_params = DH_params
        self.joint_limit = joint_limit
        self.device = device
        self.base_pose = base_pose
        self.affordance_info = None
    
    def get_affordance_info(self):
        return self.affordance_info
    
    def get_affordance(
        self, 
        rgb_img_path,
        gray_scale_img, 
        action_seq, 
        target_container_pos,
        dis_holder,
        dis_dumbwaiter,
        action_candidate=[], 
    ):
        affordance = {action: 0 for action in action_candidate}
        spoon_on_hand = self.spoon_on_hand(action_seq)
        food_on_hand = spoon_on_hand and self.food_on_hand(action_seq)
        dumbwaiter_opened = self.dumbwaiter_opened(action_seq)
        move_to_target = self.move_to_target(action_seq)
        dis_holder_threshold = 0.15
        dis_dumbwaiter_threshold = 0.32
        
        self.affordance_info = ''
        for action in affordance.keys():
            
            if action == 'scoop':
                joint_affordable = target_container_pos[:, 0] < 0.70
            else:
                joint_affordable = True
            if not joint_affordable:
                try:
                    target_color = action_seq[-1].split('_')[2] + ' '
                except:
                    target_color = ''
                self.affordance_info += f'Cannot do {action} because the target {target_color}bowl is too far, please pull it closer. '
                continue
            state_affordable, state_info = self.state_affordable(action, spoon_on_hand, food_on_hand, dumbwaiter_opened, move_to_target, dis_holder < dis_holder_threshold, dis_dumbwaiter < dis_dumbwaiter_threshold)
            if not state_affordable:
                self.affordance_info += f'Cannot do {action} because {state_info}. '
                continue
            
            affordance[action] = 1                


        return affordance
    
    def state_affordable(self, action, spoon_on_hand, food_on_hand, dumbwaiter_opened, move_to_target, obstacle_holder, obstacle_dumbwaiter):
        if action == 'grasp_spoon':
            if spoon_on_hand:
                return False, "spoon is already on hand, please put it back first"
            if obstacle_holder:
                return False, "there is a bowl too close to the holder, please move to the bowl near the holder and pull it to make space"
        elif action == 'put_spoon_back':
            if not spoon_on_hand:
                return False, "spoon is not on hand, please grasp it first"
        elif action == 'scoop':
            if not spoon_on_hand:
                return False, "spoon is not on hand, please grasp it first"
            if food_on_hand:
                return False, "there are already food in the spoon, please drop it first"
            if not move_to_target:
                return False, "the robot did not move to the target bowl first, please move to the target bowl you want to scoop"
        elif action == 'drop_food':
            if not food_on_hand:
                return False, "there is no food in the spoon, please scoop some food first"
            if not move_to_target:
                return False, "the robot is not close to the target bowl, please move to the bowl you want to put food first"
        elif action == 'open_dumbwaiter':
            if spoon_on_hand:
                return False, "spoon is on hand, please put it back first"
            if dumbwaiter_opened:
                return False, "it is already opened, please close it first"
            if obstacle_dumbwaiter:
                return False, "there is a bowl too close to the dumbwaiter, please move to the bowl near the dumbwaiter and pull it to make space"
        elif action == 'close_dumbwaiter':
            if spoon_on_hand:
                return False, "spoon is on hand, please put it back first"
            if not dumbwaiter_opened:
                return False, "it is already closed, please open it first"
        elif action == 'start_dumbwaiter':
            if spoon_on_hand:
                return False, "spoon is on hand, please put it back first"
        elif action == 'put_bowl_into_dumbwaiter':
            if spoon_on_hand:
                return False, "spoon is on hand, please put it back first"
            if not dumbwaiter_opened:
                return False, "the dumbwaiter is not opened, please open it first"
            if not move_to_target:
                return False, "the robot did not move to the target bowl first, please move to the bowl you want to put first" 
        elif action == 'pull_bowl_closer':
            if spoon_on_hand:
                return False, "spoon is on hand, please put it back first"
            if not move_to_target:
                return False, "the robot did not move to the target bowl first, please move to the bowl you want to pull first"
        return True, None
    
    def move_to_target(self, action_seq):
        return len(action_seq) > 0 and action_seq[-1].startswith('move_to')
    
    def spoon_on_hand(self, action_seq):
        """assume the robot has no spoon at the beginning"""
        grasped = False
        for action in action_seq:
            if action == 'grasp_spoon':
                grasped = True
            elif action == 'put_spoon_back':
                grasped = False
        return grasped

    def food_on_hand(self, action_seq):
        """assume the robot has no food at the beginning"""
        food = False
        for action in action_seq:
            if action == 'scoop':
                food = True
            elif action == 'drop_food':
                food = False
        return food
    
    def dumbwaiter_opened(self, action_seq): 
        """assume the dumbwaiter is closed at the beginning"""
        opened = False
        for action in action_seq:
            if action == 'open_dumbwaiter':
                opened = True
            elif action == 'close_dumbwaiter': 
                opened = False
        return opened
    
    def fk_solver(self, q: torch.Tensor):
        # TODO
        # print(self.base_pose) # checked it is the pose of the base in the world frame
        A = pose7d_to_matrix(self.base_pose) # Shape (1, 4, 4)
        j_eef = torch.zeros((1, 7, 6), device=self.device)  # Shape (1, 6, 7)
        T_list = []
        for i in range(len(self.DH_params)):
            a, d, alpha = list(self.DH_params[i].values())
            theta = q[:, i]
            a = torch.tensor(a, device=self.device, dtype=torch.float32)
            d = torch.tensor(d, device=self.device, dtype=torch.float32)
            alpha = torch.tensor(alpha, device=self.device, dtype=torch.float32)
            T = torch.tensor([[
                [torch.cos(theta), -torch.sin(theta), 0, a],
                [torch.sin(theta) * torch.cos(alpha), torch.cos(theta) * torch.cos(alpha), -torch.sin(alpha), -d * torch.sin(alpha)],
                [torch.sin(theta) * torch.sin(alpha), torch.cos(theta) * torch.sin(alpha), torch.cos(alpha), d * torch.cos(alpha)],
                [0, 0, 0, 1]
            ]], device=self.device, dtype=torch.float32)
            T_list.append(T_list[-1] @ T if i != 0 else T)
        
        T_last = torch.tensor([
            [0.7071068, 0.7071068, 0, 0],
            [-0.7071068, 0.7071068, 0, 0],
            [0, 0, 1, 0.107],
            [0, 0, 0, 1]
        ], device=self.device, dtype=torch.float32)
        _ee = torch.tensor([[
            [ 0.70710678,  0.70710678,  0.,          0.        ],
            [-0.70710678,  0.70710678,  0.,          0.        ],
            [ 0.,          0.,          1.,          0.        ],
            [ 0.,          0.,          0.,          1.        ],
        ]], device=self.device, dtype=torch.float32) @ torch.tensor([[
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0.1034],
            [0, 0, 0, 1]
        ]], device=self.device, dtype=torch.float32)
        A = A @ T_list[-1] @ T_last
        
        EE = T_list[-1] @ T_last @ _ee
        O_end = EE[:, :3, 3]
        for i, T in enumerate(T_list):
            Z = T[:, :3, 2]
            O = T[:, :3, 3]
            j_eef[:, i] = torch.cat((torch.cross(Z, (O_end - O)), Z), dim=-1)
        
        pose_7d = matrix_to_pose_7d(A)
        
        return pose_7d, j_eef.transpose(1, 2)

    def ik_solver(self, dpose, j_eef, damping=0.2):
        j_eef_T = torch.transpose(j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (damping ** 2)
        u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(j_eef.shape[0], 7)
        return u
    
    def joint_affordable(self, cur_pose, cur_joint, traj, delta=1, step_num=1, step_size=1/60.):
        ## TODO 
        ## 1. get fix trajectory based on action (might to do refactoring)
        ## 2. calculate each step of joint pose based on ik solver 
        ## 3. check if each step is affordable (check robot joint limit and poses in each step)
        """
        Args:
            cur_pose: torch.tensor, shape=(7,), current pose of the robot
            traj: List[torch.tensor], list of target poses
            cur_joint: torch.tensor, shape=(7,), current joint pose of the robot
            joint_limit: torch.tensor, shape=(7, 2), joint limit of the robot
            j_eef: torch.tensor, shape=(7, 6), jacobian matrix of the end effector
        """
        
        def check_joint_limit(joint_pose):
            return torch.all(joint_pose > self.joint_limit[:, 0]) and torch.all(joint_pose < self.joint_limit[:, 1])
        
        def orientation_error(desired, current):
            cc = quat_conjugate(current)
            q_r = quat_mul(desired, cc)
            return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)
        
        
        _pose, j_eef = self.fk_solver(cur_joint)
        print('FK jeef', j_eef)
        
        cnt = 1
        move = 0
        while len(traj) > 0:
            pose = traj[0]
            diff_pos = torch.norm(pose[:, :3] - cur_pose[:, :3])
            diff_axis, diff_w = self._calculate_quat_diff(pose[:, 3:], cur_pose[:, 3:])
            if diff_pos < self.pos_offset and diff_axis < self.axis_offset and diff_w < self.w_offset:
                traj.pop(0)
                print(cnt)
                cnt += 1
                move = 0
                pass
            pos_err = torch.where(diff_pos > self.pos_offset, pose[:, :3] - cur_pose[:, :3], torch.tensor([0., 0., 0.], device=self.device))
            orn_err = torch.where(diff_axis > self.axis_offset or diff_w > self.w_offset, orientation_error(pose[:, 3:], cur_pose[:, 3:]), torch.tensor([0., 0., 0.], device=self.device))
            dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1) * delta
            for _ in range(step_num):
                if not check_joint_limit(cur_joint):
                    return False
                djoint = self.ik_solver(dpose, j_eef)
                cur_joint += djoint * step_size
                cur_pose, j_eef = self.fk_solver(cur_joint)
                move += 1
                if move > 1000:
                    return False
                # print('Joint', cur_joint)
                # print('Pose', pose)
                # print('Cur pose', cur_pose)
        return True
    
    def _calculate_quat_diff(self, desired_quat, cur_quat):
        to_axis = desired_quat[:, :3] - cur_quat[:, :3]
        axis_dist = torch.norm(to_axis)
        w_dist = desired_quat[:, -1] - cur_quat[:, -1]
        w_dist = abs(w_dist)            
        return axis_dist, w_dist
