from isaacgym import gymapi
from isaacgym import gymtorch

import torch
from PIL import Image

from environment import IsaacSim
from src.controller import Controller
from src.config.utils import read_yaml

def set_keyboard(isaac_sim: IsaacSim, k_mode: str):
    if k_mode == 'collect':
        setting = {
            "UP": "up",
            "DOWN": "down",
            "LEFT": "left",
            "RIGHT": "right",
            "W": "backward",
            "S": "forward",
            "A": "turn_right",
            "D": "turn_left",
            "E": "turn_up",
            "Q": "turn_down",
            "R": "rot_left",
            "T": "rot_right",
            "SPACE": "gripper_close",
        }
        # self.action_list = []
        # ----------------------------------------#
        
    elif k_mode == 'demo':
        setting = {
            "Q": "scoop",
            "W": "drop_food",
            "E": "stir",
            "T": "grasp_spoon",
            "Y": "put_spoon_back",
            "C": "move_around",
            "D": "pull_bowl_closer",
            "M": "open_dumbwaiter",
            "N": "close_dumbwaiter",
            "O": "start_dumbwaiter",
            "K": "put_bowl_into_dumbwaiter",
            "L": "take_bowl_out_dumbwaiter",
        }
        for i, container in enumerate(isaac_sim.containers_indices):
            if i + 1 > 9:
                break
            setting[str(i + 1)] = f"move_to_{container}"
    else:
        raise ValueError(f"Unsupported mode: {k_mode}")
    action_list = list(setting.values())
    
    setting['3'] = 'image'
    setting['X'] = 'save'
    setting['F'] = 'to_file'
    setting['ESCAPE'] = 'quit'
    for key_name, description in setting.items():
        print(f"{key_name: >7} : {description}")
        isaac_sim.gym.subscribe_viewer_keyboard_event(isaac_sim.viewer, getattr(gymapi, f"KEY_{key_name}"), description)
    return action_list

def main(config, k_mode='collect'):
    isaac_sim = IsaacSim(config)
    action_list = set_keyboard(isaac_sim, k_mode)
    isaac_sim.reset()

    action = ""
    controller = Controller(isaac_sim)
    
    if k_mode == 'collect':
        action_ctrl = controller.low_level_control
    elif k_mode == 'demo':
        action_ctrl = controller.apply_action
    
    while not isaac_sim.gym.query_viewer_has_closed(isaac_sim.viewer):        
        # step the physics
        isaac_sim.gym.simulate(isaac_sim.sim)
        isaac_sim.gym.fetch_results(isaac_sim.sim, True)
        isaac_sim.gym.render_all_camera_sensors(isaac_sim.sim)

        isaac_sim.gym.refresh_dof_state_tensor(isaac_sim.sim)
        isaac_sim.gym.refresh_actor_root_state_tensor(isaac_sim.sim)
        isaac_sim.gym.refresh_rigid_body_state_tensor(isaac_sim.sim)
        isaac_sim.gym.refresh_jacobian_tensors(isaac_sim.sim)

        gripper_open = isaac_sim.franka_dof_upper_limits[7:]
        gripper_close = isaac_sim.franka_dof_lower_limits[7:]

        for evt in isaac_sim.gym.query_viewer_action_events(isaac_sim.viewer):
            if (evt.value) > 0:
                action = evt.action
                break
        else:
            action = ""
        
        if action == "quit":
            break
        if action == controller.current_action or action == "":
            action = controller.current_action
        dpose = controller._empty_action()

        if action == "move_around":
            dpose = isaac_sim.move_around()
        elif action == "choose action":
            isaac_sim.saycan_pipeline()
            dpose = torch.tensor([[[0.],[0.],[0.],[0.],[0.],[0.]]])
        elif action == "save":
            hand_pos = isaac_sim.rb_state_tensor[isaac_sim.franka_hand_indices, 0:3]
            hand_rot = isaac_sim.rb_state_tensor[isaac_sim.franka_hand_indices, 3:7]
            dpose = torch.tensor([[[0.],[0.],[0.],[0.],[0.],[0.]]])
            print(hand_pos)
            print(hand_rot)

            isaac_sim.record.append([hand_pos, hand_rot])
        elif action == "to_file":
            pose = [i[0].numpy().squeeze(0) for i in isaac_sim.record]
            rot = [i[1].numpy().squeeze(0) for i in isaac_sim.record]
            first = pose[0]
            first[2] = 0
            pose = [oldpos - first for oldpos in pose]
            with open('./pose.txt', 'w') as f:
                for row in pose:
                    f.write("[" + ", ".join(map(str, row)) + "]" + '\n')
            with open('./rot.txt', 'w') as f:
                for row in rot:
                    f.write("[" + ", ".join(map(str, row)) + "]" + '\n')
        elif action == "image":
            rgb_img = isaac_sim.gym.get_camera_image(isaac_sim.sim, isaac_sim.envs[0], isaac_sim.camera_handles[0], gymapi.IMAGE_COLOR).reshape(1080, 1920, 4)[:,:,:-1]
            rgb_img = Image.fromarray(rgb_img)
            # split text to multiple lines if too long
            rgb_img.save('out.jpg')
        else:
            dpose = action_ctrl(action)
        dpose = dpose.to(isaac_sim.device)
        
        isaac_sim.pos_action[:, :7] = isaac_sim.dof_state[:, isaac_sim.franka_dof_indices, 0].squeeze(-1)[:, :7] + isaac_sim.control_ik(dpose)
    
        test_dof_state = isaac_sim.dof_state[:, :, 0].contiguous()
        test_dof_state[:, isaac_sim.franka_dof_indices] = isaac_sim.pos_action
        franka_actor_indices = isaac_sim.franka_indices.to(dtype=torch.int32)
        isaac_sim.gym.set_dof_position_target_tensor_indexed(
            isaac_sim.sim,
            gymtorch.unwrap_tensor(test_dof_state),
            gymtorch.unwrap_tensor(franka_actor_indices),
            len(franka_actor_indices)
        )

        # update the viewer
        isaac_sim.gym.step_graphics(isaac_sim.sim)
        isaac_sim.gym.draw_viewer(isaac_sim.viewer, isaac_sim.sim, True)

        isaac_sim.gym.sync_frame_time(isaac_sim.sim)

        isaac_sim.frame += 1

    isaac_sim.gym.destroy_viewer(isaac_sim.viewer)
    isaac_sim.gym.destroy_sim(isaac_sim.sim)
    
if __name__ == "__main__":
    config = read_yaml("./src/config/final_task/obstacles.yaml", task_type='obstacles', env_idx=16)
    main(config=config, k_mode='demo')