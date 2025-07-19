from isaacgym import gymtorch
from isaacgym import gymapi

import os
import time
import cv2
import numpy as np
import inspect
import torch
from time import time

from environment import Isaac
from src.config.utils import get_task_type_list, get_task_env_num
from src.controller import Controller
from src.decision_pipeline import Decision_pipeline
from src.config.utils import read_yaml

def none_pipeline(decision_pipeline: Decision_pipeline):
    """None pipeline: Do nothing"""
    return {decision_pipeline.random_action(): 1}

def vlm_pipeline(
    decision_pipeline: Decision_pipeline,
    rgb_path, 
    use_vlm=False
):
    """VLM pipeline: Get the best action from the score of VLM

    Args:
        use_vlm (bool, optional): Use VLM (GPT-4o) or not (GPT-3.5). Defaults to False.
    """
    semantic_score = decision_pipeline.get_semantic_score(rgb_path, use_vlm=use_vlm)
    return semantic_score
    
def cot_pipeline(self, decision_pipeline: Decision_pipeline, rgb_path):
    """COT pipeline: Get the best action from the score of Chain of Thought"""
    semantic_score = decision_pipeline.chain_of_thought_baseline(rgb_path)
    return semantic_score

def llmplanner_pipeline(
    decision_pipeline: Decision_pipeline,
    rgb_path, 
    depth_path, 
    use_vlm=False,
    max_replan=5,
    affordance_kwargs={}
):
    pass
        
def our_pipeline(
    decision_pipeline: Decision_pipeline, 
    rgb_path, 
    depth_path, 
    use_vlm=False, 
    max_replan=5, 
    affordance_kwargs={}
):
    for _ in range(max_replan):
        semantic_score = decision_pipeline.get_semantic_score(
            rgb_path, 
            use_affordance_info=True, 
            use_vlm=use_vlm,
            update_record=False,
        )
        best_action = max(semantic_score, key=semantic_score.get)
        action_candidate = [best_action]
        affordance_score = decision_pipeline.get_affordance_score(
            rgb_path, 
            depth_path, 
            action_candidate=action_candidate,
            with_info=True,
            **affordance_kwargs
        )
        if affordance_score[best_action]:
            decision_pipeline.update_record()
            break
        else:
            decision_pipeline.clear_record()
            continue
    else:
        best_action = 'REPLAN_ERROR'
    return {best_action: 1}

def gt_pipeline(gt_action):
    """GT pipeline: Get the best action from the ground truth action

    Args:
        gt_action (str): Ground truth action

    Returns:
        dict: The best action
    """
    gt_action = gt_action if gt_action else 'DONE'
    return {gt_action: 1}

def run_experiment(
    isaac: Isaac, 
    decision_pipeline: Decision_pipeline,
    action_sequence_answer=[], 
    pipeline_name=None, 
    use_vlm=False, 
    record_video=True
):
    """Interface for testing pipeline

    Args:
        action_sequence_answer (List[str], optional): List of ground truth action. Defaults to None.
        pipeline_name (str, optional): Surpported types: lap, saycan, vlm, knowno, cot, uncertainty_cot, our. Defaults to None.
        use_vlm (bool, optional): Set to true to use VLM as planning agent. Defaults to False.
    """
    def write_action_seq():
        filename = os.path.join(decision_pipeline.log_folder, 'result_sequence.txt')
        open(filename, 'w').write('\n'.join(decision_pipeline.action_sequence))
    
    def get_our_pipeline_kwargs():
        # traj_dict = {action: controller.get_trajectory(action) for action in decision_pipeline.action_list}
        holder_pos = torch.tensor([[0.412, -0.36, isaac.default_height / 2 - 0.01]])
        nearest_container_holder = controller._find_nearest_container(holder_pos)
        distance_nearest_container_holder = torch.norm(holder_pos[:, :2] - nearest_container_holder[:, :2])
        
        dumbwaiter_pos = isaac.rb_state_tensor[isaac.dumbwaiter_door_indices_rb, :3]
        nearest_container_dumbwaiter = controller._find_nearest_container(dumbwaiter_pos)
        # print(f"dumbwaiter pos: {dumbwaiter_pos}, nearest container: {nearest_container_dumbwaiter}")
        distance_nearest_dumbwaiter_holder = torch.norm(dumbwaiter_pos[:, :2] - nearest_container_dumbwaiter[:, :2])
        
        kwargs = {
            # 'K': isaac.get_camera_intrinsic(),
            # 'extrinsic': isaac.gym.get_camera_view_matrix(isaac.sim, isaac.envs[0], isaac.camera_handles[0]),
            # 'extrinsic': isaac.get_camera_extrinsic(), 
            # 'traj_dict': traj_dict,
            # 'cur_pose': isaac.rb_state_tensor[isaac.franka_hand_indices, :7],
            # 'cur_joint': isaac.dof_state[:, isaac.franka_dof_indices, 0].squeeze(-1)[:, :7],
            'target_container_pos': controller._find_nearest_container(isaac.rb_state_tensor[isaac.franka_hand_indices, :3]),
            'dis_holder': distance_nearest_container_holder,
            'dis_dumbwaiter': distance_nearest_dumbwaiter_holder,
        }
        return kwargs

    def get_our_affordance_agent_kwargs():
        joint_limit = torch.tensor([x for x in zip(isaac.franka_dof_lower_limits[:7], isaac.franka_dof_upper_limits[:7])])
        kwargs = {
            'DH_params': [
                {'a': 0, 'd': 0.333, 'alpha': 0},
                {'a': 0, 'd': 0, 'alpha': -np.pi/2},
                {'a': 0, 'd': 0.316, 'alpha': np.pi/2},
                {'a': 0.0825, 'd': 0, 'alpha': np.pi/2},
                {'a': -0.0825, 'd': 0.384, 'alpha': -np.pi/2},
                {'a': 0, 'd': 0, 'alpha': np.pi/2},
                {'a': 0.088, 'd': 0, 'alpha': np.pi/2}
            ],
            'joint_limit': joint_limit,
            'base_pose': isaac.rb_state_tensor[isaac.franka_base_indices, :7],
            'device': isaac.device
        }
        return kwargs
    
    isaac.reset()
    controller = Controller(isaac)
    action_time_limit = 120 # sec
    start_wait = 2 # sec
    start = time()
    action_start = time()
    executing = False
    best_action = None
    action_idx = 0
    max_sequence = 20 if len(action_sequence_answer) == 0 else len(action_sequence_answer) + 2
    rgb_path = os.path.join("observation", "rgb.png")
    depth_path = os.path.join("observation", "depth.png")

    pipeline_dict = {
        'none': none_pipeline,
        'vlm': vlm_pipeline,
        'cot': cot_pipeline,
        'our': our_pipeline,
        'gt': gt_pipeline
    }
    affordance_type = 'our'
    kwargs = {}
    if affordance_type == 'our':
        kwargs = get_our_affordance_agent_kwargs()
    decision_pipeline.set_affordance_agent(affordance_type, **kwargs)
    
    if record_video:
        resolution = (1920, 1080)
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 60.0
        video_filename = os.path.join(decision_pipeline.log_folder, "result.mp4")
        out = cv2.VideoWriter(video_filename, codec, fps, resolution)
    
    while not isaac.gym.query_viewer_has_closed(isaac.viewer):
        isaac.gym.simulate(isaac.sim)
        isaac.gym.fetch_results(isaac.sim, True)
        isaac.gym.render_all_camera_sensors(isaac.sim)

        isaac.gym.refresh_dof_state_tensor(isaac.sim)
        isaac.gym.refresh_actor_root_state_tensor(isaac.sim)
        isaac.gym.refresh_rigid_body_state_tensor(isaac.sim)
        isaac.gym.refresh_jacobian_tensors(isaac.sim)
        
        
        if not executing and time() - start > start_wait:
            isaac.get_rgb_image(rgb_path)
            isaac.get_depth_image(depth_path)
            decision_pipeline.set_obs_id()
            if pipeline_name in pipeline_dict:
                pipeline_func = pipeline_dict[pipeline_name]
                params = {
                    'decision_pipeline': decision_pipeline,
                    "rgb_path": rgb_path,
                    "depth_path": depth_path,
                    "use_vlm": use_vlm,  # Fixed parameter
                    "gt_action": action_sequence_answer[action_idx] if action_sequence_answer else None,
                    "affordance_kwargs": get_our_pipeline_kwargs(),
                }
                param_names = inspect.signature(pipeline_func).parameters.keys()
                params = {k: v for k, v in params.items() if k in param_names}
                action_score = pipeline_func(**params)
            else:
                raise ValueError(f"Unsupported test type {pipeline_name}")
            
            best_action = max(action_score, key=action_score.get)
            action_idx += 1
            decision_pipeline.update_action_sequence(best_action)
            executing = True
            
        if record_video:
            img = isaac.gym.get_camera_image(isaac.sim, isaac.envs[0], isaac.camera_handles[0], gymapi.IMAGE_COLOR).reshape(1080, 1920, 4)[:,:,:-1]
            frame = np.array(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.write(frame)

        dpose = controller._empty_action()

        if best_action is None:
            pass
        elif best_action == "DONE" or best_action == 'REPLAN_ERROR' or len(decision_pipeline.action_sequence) >= max_sequence:
            break 
        else:
            dpose = controller.apply_action(best_action)
        if best_action and time() - action_start > action_time_limit or controller.current_action == 'idle':
            executing = False
            controller.reset_action_stage()
            action_start = time()
        
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
    
    write_action_seq()
    if record_video:
        out.release()
    isaac.gym.destroy_viewer(isaac.viewer)
    isaac.gym.destroy_sim(isaac.sim)
            
def main():
    root = os.environ.get('RESULT_DIR', 'experiment_result/test')
    config_file = os.environ.get('CONFIG_FILE', 'src/config/config.yaml')
    pipeline_name = os.environ.get('PIPELINE_NAME', None)
    task_type = os.environ.get('TASK_TYPE', None)
    env_idx = os.environ.get('ENV_IDX', 1)
    env_idx = int(env_idx) if env_idx else 1
    
    all_task = get_task_type_list(config_file)
    if task_type not in all_task:
        raise ValueError(f"Task type {task_type} not found in config file {config_file}. Available tasks: {all_task}")
    log_folder = os.path.join(root, f"{task_type}_{str(env_idx)}")
    os.makedirs("observation", exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)

    config = read_yaml(config_file, task_type=task_type, env_idx=env_idx)
    instruction = config["instruction"] if config["instruction"] != "None" else ""
    action_sequence_answer = config["answer"] if config["answer"] != "None" else []
    isaac = Isaac(config)
    decision_pipeline = Decision_pipeline(instruction, isaac.containers_list, isaac.tool_list, log_folder)
    run_experiment(isaac, decision_pipeline, pipeline_name=pipeline_name, action_sequence_answer=action_sequence_answer, use_vlm=True)
    
if __name__ == "__main__":
    main()