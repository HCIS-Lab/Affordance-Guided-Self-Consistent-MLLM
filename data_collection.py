from isaacgym import gymapi
from isaacgym import gymtorch

import torch
from PIL import Image

from environment import Isaac
from src.controller import Controller
from src.config.utils import read_yaml

def set_keyboard(isaac: Isaac, k_mode: str):
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
        for i, container in enumerate(isaac.containers_indices):
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
        isaac.gym.subscribe_viewer_keyboard_event(isaac.viewer, getattr(gymapi, f"KEY_{key_name}"), description)
    return action_list

def main(config, k_mode='collect'):
    isaac = Isaac(config)
    action_list = set_keyboard(isaac, k_mode)
    isaac.reset()

    action = ""
    controller = Controller(isaac)
    
    if k_mode == 'collect':
        action_ctrl = controller.low_level_control
    elif k_mode == 'demo':
        action_ctrl = controller.apply_action
    
    while not isaac.gym.query_viewer_has_closed(isaac.viewer):        
        # step the physics
        isaac.gym.simulate(isaac.sim)
        isaac.gym.fetch_results(isaac.sim, True)
        isaac.gym.render_all_camera_sensors(isaac.sim)

        isaac.gym.refresh_dof_state_tensor(isaac.sim)
        isaac.gym.refresh_actor_root_state_tensor(isaac.sim)
        isaac.gym.refresh_rigid_body_state_tensor(isaac.sim)
        isaac.gym.refresh_jacobian_tensors(isaac.sim)

        gripper_open = isaac.franka_dof_upper_limits[7:]
        gripper_close = isaac.franka_dof_lower_limits[7:]

        for evt in isaac.gym.query_viewer_action_events(isaac.viewer):
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
            dpose = isaac.move_around()
        elif action == "choose action":
            isaac.saycan_pipeline()
            dpose = torch.tensor([[[0.],[0.],[0.],[0.],[0.],[0.]]])
        elif action == "save":
            hand_pos = isaac.rb_state_tensor[isaac.franka_hand_indices, 0:3]
            hand_rot = isaac.rb_state_tensor[isaac.franka_hand_indices, 3:7]
            dpose = torch.tensor([[[0.],[0.],[0.],[0.],[0.],[0.]]])
            print(hand_pos)
            print(hand_rot)

            isaac.record.append([hand_pos, hand_rot])
        elif action == "to_file":
            pose = [i[0].numpy().squeeze(0) for i in isaac.record]
            rot = [i[1].numpy().squeeze(0) for i in isaac.record]
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
            rgb_img = isaac.gym.get_camera_image(isaac.sim, isaac.envs[0], isaac.camera_handles[0], gymapi.IMAGE_COLOR).reshape(1080, 1920, 4)[:,:,:-1]
            rgb_img = Image.fromarray(rgb_img)
            # split text to multiple lines if too long
            rgb_img.save('out.jpg')
        else:
            dpose = action_ctrl(action)
        dpose = dpose.to(isaac.device)
        
        isaac.pos_action[:, :7] = isaac.dof_state[:, isaac.franka_dof_indices, 0].squeeze(-1)[:, :7] + isaac.control_ik(dpose)
    
        test_dof_state = isaac.dof_state[:, :, 0].contiguous()
        test_dof_state[:, isaac.franka_dof_indices] = isaac.pos_action
        franka_actor_indices = isaac.franka_indices.to(dtype=torch.int32)
        isaac.gym.set_dof_position_target_tensor_indexed(
            isaac.sim,
            gymtorch.unwrap_tensor(test_dof_state),
            gymtorch.unwrap_tensor(franka_actor_indices),
            len(franka_actor_indices)
        )

        # update the viewer
        isaac.gym.step_graphics(isaac.sim)
        isaac.gym.draw_viewer(isaac.viewer, isaac.sim, True)

        isaac.gym.sync_frame_time(isaac.sim)

        isaac.frame += 1

    isaac.gym.destroy_viewer(isaac.viewer)
    isaac.gym.destroy_sim(isaac.sim)
    
if __name__ == "__main__":
    config = read_yaml("./src/config/final_task/obstacles.yaml", task_type='obstacles', env_idx=16)
    main(config=config, k_mode='demo')