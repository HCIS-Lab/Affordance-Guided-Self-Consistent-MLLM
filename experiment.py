from isaacgym import gymtorch
from isaacgym import gymapi

import os
import time
import cv2
import numpy as np
import inspect
import torch
from time import time

from environment import IsaacSim
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
    isaac_sim: IsaacSim, 
    decision_pipeline: Decision_pipeline,
    action_sequence_answer=[], 
    test_type=None, 
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
        holder_pos = torch.tensor([[0.412, -0.36, isaac_sim.default_height / 2 - 0.01]])
        nearest_container_holder = controller._find_nearest_container(holder_pos)
        distance_nearest_container_holder = torch.norm(holder_pos[:, :2] - nearest_container_holder[:, :2])
        
        dumbwaiter_pos = isaac_sim.rb_state_tensor[isaac_sim.dumbwaiter_door_indices_rb, :3]
        nearest_container_dumbwaiter = controller._find_nearest_container(dumbwaiter_pos)
        # print(f"dumbwaiter pos: {dumbwaiter_pos}, nearest container: {nearest_container_dumbwaiter}")
        distance_nearest_dumbwaiter_holder = torch.norm(dumbwaiter_pos[:, :2] - nearest_container_dumbwaiter[:, :2])
        
        kwargs = {
            # 'K': isaac_sim.get_camera_intrinsic(),
            # 'extrinsic': isaac_sim.gym.get_camera_view_matrix(isaac_sim.sim, isaac_sim.envs[0], isaac_sim.camera_handles[0]),
            # 'extrinsic': isaac_sim.get_camera_extrinsic(), 
            # 'traj_dict': traj_dict,
            # 'cur_pose': isaac_sim.rb_state_tensor[isaac_sim.franka_hand_indices, :7],
            # 'cur_joint': isaac_sim.dof_state[:, isaac_sim.franka_dof_indices, 0].squeeze(-1)[:, :7],
            'target_container_pos': controller._find_nearest_container(isaac_sim.rb_state_tensor[isaac_sim.franka_hand_indices, :3]),
            'dis_holder': distance_nearest_container_holder,
            'dis_dumbwaiter': distance_nearest_dumbwaiter_holder,
        }
        return kwargs

    def get_our_affordance_agent_kwargs():
        joint_limit = torch.tensor([x for x in zip(isaac_sim.franka_dof_lower_limits[:7], isaac_sim.franka_dof_upper_limits[:7])])
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
            'base_pose': isaac_sim.rb_state_tensor[isaac_sim.franka_base_indices, :7],
            'device': isaac_sim.device
        }
        return kwargs
    
    isaac_sim.reset()
    controller = Controller(isaac_sim)
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
    
    while not isaac_sim.gym.query_viewer_has_closed(isaac_sim.viewer):
        isaac_sim.gym.simulate(isaac_sim.sim)
        isaac_sim.gym.fetch_results(isaac_sim.sim, True)
        isaac_sim.gym.render_all_camera_sensors(isaac_sim.sim)

        isaac_sim.gym.refresh_dof_state_tensor(isaac_sim.sim)
        isaac_sim.gym.refresh_actor_root_state_tensor(isaac_sim.sim)
        isaac_sim.gym.refresh_rigid_body_state_tensor(isaac_sim.sim)
        isaac_sim.gym.refresh_jacobian_tensors(isaac_sim.sim)
        
        
        if not executing and time() - start > start_wait:
            isaac_sim.get_rgb_image(rgb_path)
            isaac_sim.get_depth_image(depth_path)
            decision_pipeline.set_obs_id()
            pipeline_name = test_type
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
                raise ValueError(f"Unsupported test type {test_type}")
            
            best_action = max(action_score, key=action_score.get)
            action_idx += 1
            decision_pipeline.update_action_sequence(best_action)
            executing = True
            
        if record_video:
            img = isaac_sim.gym.get_camera_image(isaac_sim.sim, isaac_sim.envs[0], isaac_sim.camera_handles[0], gymapi.IMAGE_COLOR).reshape(1080, 1920, 4)[:,:,:-1]
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
    
    write_action_seq()
    if record_video:
        out.release()
    isaac_sim.gym.destroy_viewer(isaac_sim.viewer)
    isaac_sim.gym.destroy_sim(isaac_sim.sim)

def prepare_experiment(root, config_file):
    print(f"Prepare experiment in {root}")
    for task_type in get_task_type_list(config_file):
        for env_idx in range(1, get_task_env_num(config_file, task_type)+1):
            os.makedirs(os.path.join(root, f"{task_type}_{str(env_idx)}"), exist_ok=True)
            
def main():
    root = os.environ.get('RESULT_DIR', 'experiment_result/test')
    config_file = os.environ.get('CONFIG_FILE', 'src/config/config.yaml')
    test_type = os.environ.get('TEST_TYPE', None)
    task_type = os.environ.get('TASK_TYPE', None)
    env_idx = os.environ.get('ENV_IDX', 1)
    env_idx = int(env_idx) if env_idx else 1
    
    all_task = get_task_type_list(config_file)
    excepted_task = []
    specific_task = list(set(all_task) - set(excepted_task))
    
    if not os.path.exists(root):
        prepare_experiment(root, config_file)
        
    if not specific_task or (task_type, env_idx) in specific_task or task_type in specific_task:
        config = read_yaml(config_file, task_type='obstacles', env_idx=16)
        instruction = config["instruction"] if config["instruction"] != "None" else ""
        action_sequence_answer = config["answer"] if config["answer"] != "None" else []
        isaac_sim = IsaacSim(config)
        log_folder = os.path.join(root, f"{task_type}_{str(env_idx)}")
        decision_pipeline = Decision_pipeline(instruction, isaac_sim.containers_list, isaac_sim.tool_list, log_folder)
        run_experiment(isaac_sim, decision_pipeline, test_type=test_type, action_sequence_answer=action_sequence_answer, use_vlm=True)
    
if __name__ == "__main__":
    main()