"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


DOF control methods example
---------------------------
An example that demonstrates various DOF control methods:
- Load cartpole asset from an urdf
- Get/set DOF properties
- Set DOF position and velocity targets
- Get DOF positions
- Apply DOF efforts
"""
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym.torch_utils import to_torch, quat_mul, quat_conjugate

import os
import time
import random
import torch
import numpy as np
import math
import cv2
import inspect

from time import time, sleep
from PIL import Image, ImageDraw, ImageFont
torch.pi = math.pi

from src import Decision_pipeline, Ball_generator
from src.utils import euler_to_quaternion
from src.config import read_yaml

class IsaacSim():
    def __init__(self, env_cfg_dict, log_folder=None, record_video=False):
        args = gymutil.parse_arguments(description="Joint control Methods Example")
        args.use_gpu = False
        self.device = 'cuda:0' if args.use_gpu and torch.cuda.is_available() else "cpu"
        self.env_cfg_dict = env_cfg_dict
        
        # initialize gym
        self.gym = gymapi.acquire_gym()

        # create simulator
        self.env_spacing = 1.5
        self.max_episode_length = 195
        self.asset_root = "urdf"
        self.gravity = -9.8 # -1
        self.use_container_pos = True
        self.gripper_action_offset = 100
        
        # if there is already one sim, close it
        self.create_sim(args)
        
        # create viewer using the default camera properties
        viewer_props = gymapi.CameraProperties()
        viewer_props.width = 1080
        viewer_props.height = 720
        self.viewer = self.gym.create_viewer(self.sim, viewer_props)
        
        # Look at the first env
        self.cam_pos = gymapi.Vec3(1., 0, 1.5)
        cam_target = gymapi.Vec3(0, 0, 0)
        self.gym.viewer_camera_look_at(self.viewer, None, self.cam_pos, cam_target)

        
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(self.num_envs, self.num_dofs, 2)

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        _rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_state_tensor = gymtorch.wrap_tensor(_rb_state_tensor).view(-1, 13)

        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)

        # jacobian entries corresponding to franka hand
        self.j_eef = jacobian[:, self.hand_joint_index, :, :7]
        
        # for trajectory collection
        self.record = []
        
        self.decision_pipeline = Decision_pipeline(self.containers_list, self.tool_list, log_folder=log_folder)
        self.record_video = record_video 
        self.log_folder = log_folder if log_folder is not None else "temp"

    
    def create_sim(self, args):
        # parse arguments
        
        args.use_gpu_pipeline = args.use_gpu
        self.num_envs = 1
        self.action_sequence = []
        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, self.gravity)

        # viscosity
        sim_params.dt = 1.0 / 60.0 # 1.0 / 60 

        sim_params.substeps = 8 # 10

        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 4

        sim_params.physx.friction_offset_threshold = 0.01
        sim_params.physx.friction_correlation_distance = 5
        sim_params.physx.contact_offset = 0.001
        sim_params.physx.rest_offset = 0.000001
        sim_params.physx.max_depenetration_velocity = 1
        

        sim_params.use_gpu_pipeline = args.use_gpu_pipeline
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu
        
        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
        
        
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.env_spacing, int(np.sqrt(self.num_envs)))
        self.gym.prepare_sim(self.sim)
        
    def add_camera(self, env_ptr):
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1920
        camera_props.height = 1080
        camera_props.enable_tensors = True
        self.camera_props = camera_props
        cam_handle = self.gym.create_camera_sensor(env_ptr, camera_props)
        self.gym.set_camera_location(cam_handle, env_ptr, gymapi.Vec3(1., 0, 1.5), gymapi.Vec3(0, 0, 0))
        self.camera_handles.append(cam_handle)
    
    def get_camera_intrinsic(self):
        width, height = self.camera_props.width, self.camera_props.height
        horizontal_fov = self.camera_props.horizontal_fov
        fx = width / (2 * math.tan(horizontal_fov / 2))
        fy = fx
        cx = width / 2
        cy = height / 2
        return torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
    
    def get_camera_extrinsic(self):
        cam_pos = torch.tensor([self.cam_pos.x, self.cam_pos.y, self.cam_pos.z], dtype=torch.float32)
        cam_target = torch.tensor([0, 0, 0], dtype=torch.float32)
        cam_up = torch.tensor([0, 0, 1], dtype=torch.float32)
        cam_x = (cam_target - cam_pos) / torch.norm(cam_target - cam_pos)
        cam_y = torch.cross(cam_up, cam_x) / torch.norm(torch.cross(cam_up, cam_x))
        cam_z = torch.cross(cam_x, cam_y)
        cam_R = torch.stack([cam_x, cam_y, cam_z], dim=1)
        cam_T = -cam_R @ cam_pos
        cam_extrinsic = torch.eye(4, dtype=torch.float32)
        cam_extrinsic[:3, :3] = cam_R
        cam_extrinsic[:3, 3] = cam_T
        return cam_extrinsic
    
    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
        
    def create_table(self):

        # create table asset
        file_name = 'holder/holder_table.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.resolution = 300000
        self.table_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)
        # self.table_dims = gymapi.Vec3(0.8, 1, self.default_height)
        # self.table_asset = self.gym.create_box(self.sim, self.table_dims.x, self.table_dims.y, self.table_dims.z, asset_options)
        self.table_pose = gymapi.Transform()
        self.table_pose.p = gymapi.Vec3(0.5, 0, 0)

    # create bowls & plates
    
    def create_container(self):
        from matplotlib.colors import to_rgba
        def calculate_dist(pose1, pose2):
            return np.sqrt((pose1.p.x - pose2.p.x) ** 2 + (pose1.p.y - pose2.p.y) ** 2)
        def get_random_position(min_x, max_x, min_y, max_y, height_offset):
            return gymapi.Vec3(
                min_x + (max_x - min_x) * random.random(), 
                min_y + (max_y - min_y) * random.random(), 
                self.default_height / 2 + height_offset
            )
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = False
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.resolution = 500000
        if self.container_config_list is not None:
            self.container_num = len(self.container_config_list) if self.container_config_list is not None else 0
            file_name_list = [f'container/{x["type"]}.urdf' if x["food"]["type"] != "None" else 'container/bowl_tofu.urdf' for x in self.container_config_list]
            self.container_asset = [self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options) for file_name in file_name_list]
        else:
            self.container_num = 0
            file_name_list = []
            self.container_asset = []
        self.containers_indices = {}
        self.containers_pose = []
        self.containers_color = []
        self.containers_list = []
        print(file_name_list)
        self.min_container_dist = 0.2
        min_x = 0.35
        max_x = 0.55
        min_y = -0.15
        max_y = 0.25
        for i in range(self.container_num):
            stop_sample = False
            container_config = self.container_config_list[i]
            container_type = container_config["type"]
            
            container_pose = gymapi.Transform()
            container_pose.r = gymapi.Quat(1, 0, 0, 1) if 'plate' not in container_type else gymapi.Quat(0, 0, 0, 1)
            height_offset = 0.02 if container_type == "bowl" else 0
            print(type(container_config["x"]))
            if type(container_config["x"]) == float:
                container_pose.p = gymapi.Vec3(container_config["x"], container_config["y"], self.default_height / 2 + height_offset)
            elif type(container_config["x"]) == list:
                print(f'random sample in range {container_config["x"]}, {container_config["y"]}')
                temp_min_x, temp_max_x = container_config["x"]
                temp_min_y, temp_max_y = container_config["y"]
                while not stop_sample:
                    container_pose.p = get_random_position(temp_min_x, temp_max_x, temp_min_y, temp_max_y, height_offset)
                    stop_sample = True
                    for pose in self.containers_pose:
                        if calculate_dist(pose, container_pose) < self.min_container_dist:
                            stop_sample = False
                            break
            else:
                while not stop_sample:
                    container_pose.p = get_random_position(min_x, max_x, min_y, max_y, height_offset)
                    stop_sample = True
                    for pose in self.containers_pose:
                        if calculate_dist(pose, container_pose) < self.min_container_dist:
                            stop_sample = False
                            break
                print(container_pose.p)
            
            c = None
            color_code = None
            if container_config["color"] == None:
                c = "white"
                color_code = "white"
            else:
                c = container_config["color"]
                color_code = container_config["colorcode"]
            
            food = None
            food_config = container_config["food"]
            if food_config['type'] == "None":
                food = "(with tofu pudding)"
            elif food_config['type'] == "ball":
                food_colors = " and ".join(container_config["food"]["foodname"])
                food = f"(with {food_colors})"                
            else:
                food = f"(with {self.env_cfg_dict['containers'][i]['food']})"
                
            rgba = to_rgba(color_code)
            color = gymapi.Vec3(rgba[0], rgba[1], rgba[2])
            self.containers_indices[f"{c}_{container_type} {food}"] = []
            self.containers_list.append(f"{c}_{container_type} {food}")
            self.containers_pose.append(container_pose)
            self.containers_color.append(color)
        print(self.containers_list)
                
    def add_container(self, env_ptr):
        for i in range(self.container_num):
            container_handle = self.gym.create_actor(env_ptr, self.container_asset[i], self.containers_pose[i], self.containers_list[i], 0, 0)
            self.gym.set_actor_scale(env_ptr, container_handle, 0.5)
            self.gym.set_rigid_body_color(env_ptr, container_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, self.containers_color[i])
            container_idx = self.gym.get_actor_rigid_body_index(env_ptr, container_handle, 0, gymapi.DOMAIN_SIM)
            self.containers_indices[self.containers_list[i]].append(container_idx)

    def create_butter(self):
        file_name = 'food/butter.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.resolution = 300000
        self.butter_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)
        self.butter_poses = []
        butter_pose = gymapi.Transform()
        butter_pose.r = gymapi.Quat(1, 0, 0, 1)
        butter_pose.p = gymapi.Vec3(0.4, 0., self.default_height/2)
        self.butter_poses.append(butter_pose)
        butter_pose.p = gymapi.Vec3(0.4, 0.05, self.default_height/2)
        self.butter_poses.append(butter_pose)
                  
    def create_holder(self):
        file_name = 'holder/holder.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.resolution = 300000
        self.holder_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)
        self.holder_pose = gymapi.Transform()
        self.holder_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.holder_pose.p = gymapi.Vec3(0.6, 0, self.default_height/2)
        
    def create_tool(self):
        
        # Load spoon asset
        self.tool_asset = {}
        self.tool_pose = {}
        self.tool_handle = {}
        self.tool_indices = {}
        tool_x = 0.412
        tool_x_between = 0.086
        tool_y = -0.36
        tool_z = self.default_height / 2 - 0.01
        height_offset = 0.19
        for i, tool in enumerate(self.tool_list):
            file_name = f'grab_tool/{tool}.urdf'
            self.tool_handle[tool] = []
            self.tool_indices[tool] = []
            asset_options = gymapi.AssetOptions()
            asset_options.armature = 0.01
            asset_options.vhacd_enabled = True
            asset_options.fix_base_link = False
            asset_options.disable_gravity = False    
            asset_options.vhacd_params.resolution = 500000
            self.tool_asset[tool] = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)
            pose = gymapi.Transform()
            pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 1.57)
            pose.p = gymapi.Vec3(tool_x,  tool_y, tool_z + height_offset)
            tool_x += tool_x_between
            self.tool_pose[tool] = pose

    def add_tool(self, env_ptr):
        for i, tool in enumerate(self.tool_list):
            handle = self.gym.create_actor(env_ptr, self.tool_asset[tool], self.tool_pose[tool], tool, 0, 0)
            if tool == 'spoon':
                self.gym.set_actor_scale(env_ptr, handle, 0.75)
            cube_idx = self.gym.find_actor_rigid_body_index(env_ptr, handle, f"cube_{tool}", gymapi.DOMAIN_SIM)
            tool_idx = self.gym.get_actor_index(env_ptr, handle, gymapi.DOMAIN_SIM)
            self.tool_handle[tool].append(handle)
            # self.tool_indices[tool].append(tool_idx)
            self.tool_indices[tool].append(cube_idx)
            
            
            body_shape_prop = self.gym.get_actor_rigid_shape_properties(env_ptr, handle)
            body_shape_prop[0].thickness = 10
            body_shape_prop[0].friction = 500
            body_shape_prop[0].contact_offset = 0.
            body_shape_prop[1].friction = 0.01
            self.gym.set_actor_rigid_shape_properties(env_ptr, handle, body_shape_prop)
                
    def create_ball(self):
        self.ball_radius, self.ball_mass, self.ball_friction = 0.0035, 1e-4, 0.001
        self.between_ball_space = 0.05
        ball_generator = Ball_generator()
        file_name = f'ball/BallHLS.urdf'
        ball_generator.generate(root=self.asset_root, file_name=file_name, ball_radius=self.ball_radius, ball_mass=self.ball_mass)
        self.ball_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, gymapi.AssetOptions())
    
    def set_ball_property(self, env_ptr, ball_pose, color):

        ball_friction = self.ball_friction
        ball_rolling_friction = 0.001
        ball_torsion_friction = 0.001
        ball_handle = self.gym.create_actor(env_ptr, self.ball_asset, ball_pose, f"{color} grain", 0, 0)
        body_shape_prop = self.gym.get_actor_rigid_shape_properties(env_ptr, ball_handle)
        body_shape_prop[0].friction = ball_friction
        # body_shape_prop[0].rolling_friction = ball_rolling_friction
        # body_shape_prop[0].torsion_friction = ball_torsion_friction
        body_shape_prop[0].contact_offset = 0.0001  # Distance at which contacts are generated
        body_shape_prop[0].rest_offset = 0.000001   # How far objects should come to rest from the surface of this body 
        body_shape_prop[0].restitution = 0.         # when two objects hit or collide, the speed at which they move after the collision
        body_shape_prop[0].thickness = 0.           # the ratio of the final to initial velocity after the rigid body collides. 
        

        self.gym.set_actor_rigid_shape_properties(env_ptr, ball_handle, body_shape_prop)
        return ball_handle
    
    def create_food(self):
        file_name = 'food/butter.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.resolution = 300000
        self.butter_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)
        
        file_name = 'food/forked_food.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.resolution = 300000
        self.forked_food_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)
        
    def add_food(self, env_ptr):
        # add balls
        from matplotlib.colors import to_rgba
        
        self.ball_handles = []
        ball_pose = gymapi.Transform()
        ball_pose.r = gymapi.Quat(0, 0, 0, 1)
        ball_spacing = self.between_ball_space
        for i, containers_pose in enumerate(self.containers_pose):
            food_config = self.container_config_list[i]["food"]
            if food_config['type'] == "ball":
                total_amount = int(food_config['amount'])
                color_num = len(food_config["foodname"])
                x_config, y_config = food_config.get('position', [None, None])
                ball_amount_max = min(10, total_amount)
                ran = ball_spacing * 0.18 * ball_amount_max
                range_offset = 0
                ball_range = (
                    (containers_pose.p.x - ran / 2 + range_offset, containers_pose.p.x + ran / 2 - range_offset), 
                    (containers_pose.p.y - ran / 2 + range_offset, containers_pose.p.y + ran / 2 - range_offset)
                ) # area: 0.09 * 0.09
                central_x = np.random.uniform(ball_range[0][0] + x_config * ran / 3, ball_range[0][1] - (2 - x_config) * ran / 3) if x_config else np.mean(ball_range[0])
                central_y = np.random.uniform(ball_range[1][0] + y_config * ran / 3, ball_range[1][1] - (2 - y_config) * ran / 3) if y_config else np.mean(ball_range[1])
                print(self.container_config_list[i]["color"].upper())
                print(('north', 'central', 'south')[int((central_x - ball_range[0][0]) // 0.03)], end='-')
                print(('west', 'central', 'east')[int((central_y - ball_range[1][0]) // 0.03)])
                print('=' * 20)
                ran_x = min(abs(central_x - ball_range[0][0]), abs(central_x - ball_range[0][1])) * 2
                ran_y = min(abs(central_y - ball_range[1][0]), abs(central_y - ball_range[1][1])) * 2
                if x_config != 1:
                    ran_x = min(0.03, ran_x)
                if y_config != 1:
                    ran_y = min(0.03, ran_y)
                x_start, y_start = central_x - ran_x / 2, central_y - ran_y / 2
                for cnt, c in enumerate(food_config["colorcode"]):
                    ball_amount = int(total_amount / color_num)
                    rgba = to_rgba(c)
                    color = gymapi.Vec3(rgba[0], rgba[1], rgba[2])
                    sub_ran = ball_amount / total_amount * ran_y
                    x, y, z = x_start, y_start, self.default_height / 2 + 0.02
                    while ball_amount > 0:
                        y = y_start
                        while y < y_start + sub_ran and ball_amount > 0:
                            x = x_start
                            while x < x_start + ran_x and ball_amount > 0:
                                ball_pose.p = gymapi.Vec3(x + (random.random() - 0.5) * 0.01, y + (random.random() - 0.5) * 0.01, z)
                                ball_handle = self.set_ball_property(env_ptr, ball_pose, c)
                                self.ball_handles.append(ball_handle)
                                self.gym.set_rigid_body_color(env_ptr, ball_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
                                x += ball_spacing * 0.18
                                ball_amount -= 1
                            y += ball_spacing * 0.18
                        z += ball_spacing * 0.18
                    y_start += sub_ran
                        
            elif food_config['type'] == "cuttable_food":
                rot = gymapi.Quat(1, 0, 0, 1)
                x, y, z = containers_pose.p.x, containers_pose.p.y, containers_pose.p.z + 0.03
                for _ in range(2):
                    pose = gymapi.Transform()
                    pose.r = rot
                    pose.p = gymapi.Vec3(x, y, z)
                    y += 0.05
                    self.butter_handle = self.gym.create_actor(env_ptr, self.forked_food_asset, pose, "cuttable_food", 0, 0)
                    butter_idx = self.gym.get_actor_index(env_ptr, self.butter_handle, gymapi.DOMAIN_SIM)
                    self.butter_indices.append(butter_idx)    
                    self.forked_food_indices.append(butter_idx)    
            
            elif food_config['type'] == "forked_food":
                pose = gymapi.Transform()
                pose.r = gymapi.Quat(1, 0, 0, 1)
                pose.p = gymapi.Vec3(containers_pose.p.x, containers_pose.p.y, self.default_height/2 + 0.03)
                self.forked_food_handle = self.gym.create_actor(env_ptr, self.forked_food_asset, pose, "forked_food", 0, 0)
                idx = self.gym.get_actor_index(env_ptr, self.forked_food_handle, gymapi.DOMAIN_SIM)
                self.forked_food_indices.append(idx)

    def create_dumbwaiter(self):
        file_name = 'dumbwaiter/mobility.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.resolution = 300000
        self.dumbwaiter_asset = self.gym.load_asset(self.sim, self.asset_root, file_name, asset_options)
        self.num_dofs += self.gym.get_asset_dof_count(self.dumbwaiter_asset)
        self.dumbwaiter_dof_props = self.gym.get_asset_dof_properties(self.dumbwaiter_asset)
        self.dumbwaiter_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        
        self.dumbwaiter_pose = gymapi.Transform()
        quat = euler_to_quaternion(0, 0, math.pi / 2)
        self.dumbwaiter_pose.r = quat
        self.dumbwaiter_pose.p = gymapi.Vec3(0.5, 0.62, self.default_height / 2 + 0.08)
    
    def add_dumbwaiter(self, env_ptr):
        dumbwaiter_handle = self.gym.create_actor(env_ptr, self.dumbwaiter_asset, self.dumbwaiter_pose, 'dumbwaiter', 0, 8)
        self.gym.set_actor_scale(env_ptr, dumbwaiter_handle, 0.28)
        self.gym.set_actor_dof_properties(env_ptr, dumbwaiter_handle, self.dumbwaiter_dof_props)
        self.dumbwaiter_door_indices = self.gym.find_actor_dof_index(env_ptr, dumbwaiter_handle, 'door', gymapi.DOMAIN_SIM)
        self.dumbwaiter_door_indices = to_torch(self.dumbwaiter_door_indices, dtype=torch.long, device=self.device)
        self.dumbwaiter_door_indices_rb.append(self.gym.find_actor_rigid_body_index(env_ptr, dumbwaiter_handle, 'link_1', gymapi.DOMAIN_SIM))
        dumbwaiter_idx = self.gym.get_actor_index(env_ptr, dumbwaiter_handle, gymapi.DOMAIN_SIM)
        self.dumbwaiter_indices.append(dumbwaiter_idx)
        
    def create_franka(self, reload=None):
        # create franka asset
        self.num_dofs = 0
        asset_file_franka = "franka_description/robots/original_franka.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True
        asset_options.disable_gravity = True
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params.resolution = 1000000
        self.franka_asset = self.gym.load_asset(self.sim, self.asset_root, asset_file_franka, asset_options)
        self.franka_dof_names = self.gym.get_asset_dof_names(self.franka_asset)
        self.num_dofs += self.gym.get_asset_dof_count(self.franka_asset)

        self.hand_joint_index = self.gym.get_asset_joint_dict(self.franka_asset)["panda_hand_joint"]

        # # set franka dof properties
        self.franka_dof_props = self.gym.get_asset_dof_properties(self.franka_asset)
        self.franka_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        
        self.franka_dof_props["stiffness"][:7].fill(3000.0)
        self.franka_dof_props["stiffness"][7:9].fill(3000.0)
        self.franka_dof_props["damping"][:7].fill(2000.0)
        self.franka_dof_props["damping"][7:9].fill(2000.0)
        self.franka_dof_props["effort"][:7].fill(float('inf'))
        self.franka_dof_props["effort"][7:9].fill(float('inf'))
        
        self.franka_dof_lower_limits = self.franka_dof_props['lower']
        self.franka_dof_upper_limits = self.franka_dof_props['upper']
        self.franka_dof_lower_limits = to_torch(self.franka_dof_lower_limits, device=self.device)
        self.franka_dof_upper_limits = to_torch(self.franka_dof_upper_limits, device=self.device)

        # set default pose
        self.franka_start_pose = gymapi.Transform()
        self.franka_start_pose.p = gymapi.Vec3(0, 0.0, 0.2) if reload == None else gymapi.Vec3(reload[0], reload[1], reload[2])
        self.franka_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0) if reload == None else gymapi.Quat(reload[3], reload[4], reload[5], reload[6])
    
    def add_franka(self, i, env_ptr):
        
        self.franka_indices = []
        self.franka_dof_indices = []
        self.franka_hand_indices = []
        self.franka_base_indices = []
        # create franka and set properties
        franka_handle = self.gym.create_actor(env_ptr, self.franka_asset, self.franka_start_pose, "franka", i, 4, 2)
        
        franka_sim_index = self.gym.get_actor_index(env_ptr, franka_handle, gymapi.DOMAIN_SIM)
        self.franka_indices.append(franka_sim_index)
        franka_dof_index = [
            self.gym.find_actor_dof_index(env_ptr, franka_handle, dof_name, gymapi.DOMAIN_SIM)
            for dof_name in self.franka_dof_names
        ]
        self.franka_dof_indices.extend(franka_dof_index)
        
        franka_hand_sim_idx = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
        franka_base_sim_idx = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, "panda_link0", gymapi.DOMAIN_SIM)
        self.franka_hand_indices.append(franka_hand_sim_idx)
        self.franka_base_indices.append(franka_base_sim_idx)
        self.gym.set_actor_dof_properties(env_ptr, franka_handle, self.franka_dof_props)
        
        body_shape_prop = self.gym.get_actor_rigid_shape_properties(env_ptr, franka_handle)
        for k in range(11):
            body_shape_prop[k].thickness = 8
            body_shape_prop[k].friction = 55
        
        self.gym.set_actor_rigid_shape_properties(env_ptr, franka_handle, body_shape_prop)
        self.franka_indices = to_torch(self.franka_indices, dtype=torch.long, device=self.device)
        self.franka_dof_indices = to_torch(self.franka_dof_indices, dtype=torch.long, device=self.device)
        self.franka_hand_indices = to_torch(self.franka_hand_indices, dtype=torch.long, device=self.device)
        self.franka_base_indices = to_torch(self.franka_base_indices, dtype=torch.long, device=self.device)
        

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, 0.75 * -spacing, 0.0)
        upper = gymapi.Vec3(spacing, 0.75 * spacing, spacing)
        self.default_height = 0.9
        self.create_table()
        self.create_container()
        self.create_ball()
        self.create_franka()
        self.create_tool()
        self.create_food()
        self.create_dumbwaiter()
        self.env_ptr_list = []
        # cache some common handles for later use
        self.camera_handles = []
        self.urdf_link_indices = []
        self.butter_indices = []
        self.forked_food_indices = []
        self.dumbwaiter_indices = []
        self.dumbwaiter_door_indices_rb = []
        self.envs = []
        self.is_acting = {}
        self.action_stage = {}
        self.delta = {}

        # create and populate the environments
        for env_i in range(num_envs):
            # create env
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.env_ptr_list.append(env_ptr)
            self.envs.append(env_ptr)
            self.add_camera(env_ptr)
            self.add_container(env_ptr)
            self.table = self.gym.create_actor(env_ptr, self.table_asset, self.table_pose, "table", 0, 8)
            self.add_food(env_ptr)
            self.add_franka(env_i, env_ptr)
            self.add_tool(env_ptr)
            self.add_dumbwaiter(env_ptr)

        self.urdf_link_indices = to_torch(self.urdf_link_indices, dtype=torch.long, device=self.device)
        self.butter_indices = to_torch(self.butter_indices, dtype=torch.long, device=self.device)
        self.forked_food_indices = to_torch(self.forked_food_indices, dtype=torch.long, device=self.device)
        self.dumbwaiter_indices = to_torch(self.dumbwaiter_indices, dtype=torch.long, device=self.device)
        self.dumbwaiter_door_indices_rb = to_torch(self.dumbwaiter_door_indices_rb, dtype=torch.long, device=self.device)
        for container in self.containers_list:
            if len(self.containers_indices[container]) > 0:
                self.containers_indices[container] = to_torch(self.containers_indices[container], dtype=torch.long, device=self.device) 
            else:
                self.containers_indices.pop(container)
                
        for tool in self.tool_list:
            if len(self.tool_indices[tool]) > 0:
                self.tool_indices[tool] = to_torch(self.tool_indices[tool], dtype=torch.long, device=self.device) 
            else:
                self.tool_indices.pop(tool)

    def reset(self):
        self.franka_init_pose = torch.tensor([-0.4969, -0.5425,  0.3321, -2.0888,  0.0806,  1.6983,  0.5075,  0.0400, 0.0400], dtype=torch.float32, device=self.device)
        self.dof_state[:, self.franka_dof_indices, 0] = self.franka_init_pose 
        self.dof_state[:, self.franka_dof_indices, 1] = 0
        self.dof_state[:, self.dumbwaiter_door_indices, 0] = 0 # math.pi # math.pi / 18
        self.dof_state[:, self.dumbwaiter_door_indices, 1] = 0
            
        target_tesnsor = self.dof_state[:, :, 0].contiguous()

        self.pos_action = torch.zeros((self.num_envs, 9), dtype=torch.float32, device=self.device)
        self.pos_action[:, 0:9] = target_tesnsor[:, self.franka_dof_indices[0:9]]

        franka_actor_indices = self.franka_indices.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state)
        )

        self.gym.set_dof_position_target_tensor(
            self.sim,
            gymtorch.unwrap_tensor(target_tesnsor)
        )
        self.frame = 0
        
    def control_ik(self, dpose, damping=0.05):
        # solve damped least squares
        j_eef_T = torch.transpose(self.j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (damping ** 2)
        u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 7)
        return u

    def none_pipeline(self):
        """None pipeline: Do nothing"""
        return {self.decision_pipeline.random_action(): 1}
    
    def vlm_pipeline(self, use_vlm=False):
        """VLM pipeline: Get the best action from the score of VLM

        Args:
            use_vlm (bool, optional): Use VLM (GPT-4o) or not (GPT-3.5). Defaults to False.
        """
        rgb_path = os.path.join("observation", "rgb.png")
        rgb_image = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handles[0], gymapi.IMAGE_COLOR).reshape(1080, 1920, 4)[:,:,:-1]
        Image.fromarray(rgb_image).save(rgb_path)
        semantic_score = self.decision_pipeline.get_semantic_score(self.instruction, rgb_path, self.action_sequence, use_vlm=use_vlm)
        return semantic_score
        
    def cot_pipeline(self):
        """COT pipeline: Get the best action from the score of Chain of Thought"""
        
        rgb_path = os.path.join("observation", "rgb.png")
        rgb_image = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handles[0], gymapi.IMAGE_COLOR).reshape(1080, 1920, 4)[:,:,:-1]
        Image.fromarray(rgb_image).save(rgb_path)
        semantic_score = self.decision_pipeline.chain_of_thought_baseline(self.instruction, rgb_path, self.action_sequence, [])
        return semantic_score
    
    def get_our_pipeline_kwargs(self):
        traj_dict = {action: self.get_trajectory(action) for action in self.decision_pipeline.action_list}
        holder_pos = torch.tensor([[0.412, -0.36, self.default_height / 2 - 0.01]])
        nearest_container_holder = self.find_nearest_container(holder_pos)
        distance_nearest_container_holder = torch.norm(holder_pos[:, :2] - nearest_container_holder[:, :2])
        
        dumbwaiter_pos = self.rb_state_tensor[self.dumbwaiter_door_indices_rb, :3]
        nearest_container_dumbwaiter = self.find_nearest_container(dumbwaiter_pos)
        print(f"dumbwaiter pos: {dumbwaiter_pos}, nearest container: {nearest_container_dumbwaiter}")
        distance_nearest_dumbwaiter_holder = torch.norm(dumbwaiter_pos[:, :2] - nearest_container_dumbwaiter[:, :2])
        
        kwargs = {
            # 'K': self.get_camera_intrinsic(),
            # 'extrinsic': self.gym.get_camera_view_matrix(self.sim, self.envs[0], self.camera_handles[0]),
            # 'extrinsic': self.get_camera_extrinsic(), 
            # 'traj_dict': traj_dict,
            # 'cur_pose': self.rb_state_tensor[self.franka_hand_indices, :7],
            # 'cur_joint': self.dof_state[:, self.franka_dof_indices, 0].squeeze(-1)[:, :7],
            'target_container_pos': self.find_nearest_container(self.rb_state_tensor[self.franka_hand_indices, :3]),
            'dis_holder': distance_nearest_container_holder,
            'dis_dumbwaiter': distance_nearest_dumbwaiter_holder,
        }
        return kwargs
    
    def get_our_affordance_agent_kwargs(self):
        joint_limit = torch.tensor([x for x in zip(self.franka_dof_lower_limits[:7], self.franka_dof_upper_limits[:7])])
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
            'base_pose': self.rb_state_tensor[self.franka_base_indices, :7],
            'device': self.device
        }
        
        return kwargs
            
    def our_pipeline(self, use_vlm=False, max_replan=5):
        # self.instruction += f" {len(self.action_sequence) + 1}. "
        rgb_path = os.path.join("observation", "rgb.png")
        depth_path = os.path.join("observation", "depth.png")
        rgb_image = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handles[0], gymapi.IMAGE_COLOR).reshape(1080, 1920, 4)[:,:,:-1]
        depth_image = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handles[0], gymapi.IMAGE_DEPTH)
        depth_image = np.clip(depth_image, -1.8, 0)
        depth_image = ((depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image)) * 255).astype(np.uint8)
        Image.fromarray(rgb_image).save(rgb_path)
        Image.fromarray(depth_image).save(depth_path)
        kwargs = self.get_our_pipeline_kwargs()
        for _ in range(max_replan):
            semantic_score = self.decision_pipeline.get_semantic_score(
                self.instruction, 
                rgb_path, 
                self.action_sequence, 
                use_affordance_info=True, 
                use_vlm=use_vlm,
                update_record=False,
            )
            best_action = max(semantic_score, key=semantic_score.get)
            action_candidate = [best_action]
            affordance_score = self.decision_pipeline.get_affordance_score(
                rgb_path, 
                depth_path, 
                action_sequence=self.action_sequence, 
                action_candidate=action_candidate,
                with_info=True,
                **kwargs
            )
            if affordance_score[best_action]:
                self.decision_pipeline.update_record()
                break
            else:
                self.decision_pipeline.clear_record()
                continue
        else:
            best_action = 'REPLAN_ERROR'
        return {best_action: 1}
    
    def gt_pipeline(self, gt_action):
        """GT pipeline: Get the best action from the ground truth action

        Args:
            gt_action (str): Ground truth action

        Returns:
            dict: The best action
        """
        print(self.decision_pipeline.init_object_list)
        gt_action = gt_action if gt_action else 'DONE'
        return {gt_action: 1}
    
    def test_pipeline(
        self, 
        action_sequence_answer=[], 
        calibration_collect=False, 
        test_type=None, 
        threshold=None, 
        use_vlm=False, 
        uncertainty_threshold=0.
    ):
        """Interface for testing pipeline

        Args:
            action_sequence_answer (List[str], optional): List of ground truth action. Defaults to None.
            calibration_collect (bool, optional): Set to true if need to collect calibration dataset. Defaults to False.
            test_type (str, optional): Surpported types: lap, saycan, vlm, knowno, cot, uncertainty_cot, our. Defaults to None.
            threshold (float, optional): Uncertainty confidence level. Defaults to 0.
            use_vlm (bool, optional): Set to true to use VLM as planning agent. Defaults to False.
            uncertainty_threshold (float, optional): Uncertainty confidence level for COT. Defaults to 0.
        """
        def write_score(action_idx, total_action, human_help):
            filename = os.path.join(self.log_folder, 'result.txt')
            assert not os.path.exists(filename), FileExistsError(filename)
            open(filename, 'w').write(f"{action_idx} / {total_action} = {float(action_idx) / float(total_action): .4f}\nHuman Help: {human_help}")
        def write_action_seq():
            filename = os.path.join(self.log_folder, 'result_sequence.txt')
            open(filename, 'w').write('\n'.join(self.action_sequence))
        
        assert not calibration_collect or action_sequence_answer, "Please provide sequence answer when collecting calibration data"
        assert bool(threshold) ^ (test_type not in ['lap', 'knowno']), "Please provide threshold when using LAP or KnowNo"
        self.reset()
        action_time_limit = 120 # sec
        start_wait = 2 # sec
        start = time()
        self.action_start = time()
        self.executing = False
        best_action = None
        action_idx = 0
        total_action = len(action_sequence_answer) if action_sequence_answer else 0
        max_sequence = 10 if len(action_sequence_answer) == 0 else len(action_sequence_answer) + 2
        human_help = 0
        affordance_list = {
            'lap': 'lap',
            'saycan': 'classifier',
            'our': 'our',
            'affordance': 'our'
        }
        pipeline_list = {
            'lap': 'saycan',
            'knowno': 'vlm',
        }
        
        affordance_type = affordance_list.get(test_type, None)
        kwargs = {}
        if affordance_type == 'our':
            kwargs = self.get_our_affordance_agent_kwargs()
        self.decision_pipeline.set_affordance_agent(affordance_type, **kwargs)
        
        if self.record_video:
            resolution = (1920, 1080)
            codec = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 60.0
            video_filename = os.path.join(self.log_folder, "result.mp4")
            out = cv2.VideoWriter(video_filename, codec, fps, resolution)
        
        while not self.gym.query_viewer_has_closed(self.viewer):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.render_all_camera_sensors(self.sim)

            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)
            
            
            if not self.executing and time() - start > start_wait:
                self.decision_pipeline.set_obs_id()
                if calibration_collect:
                    best_action = action_sequence_answer[action_idx]
                    self.calibration_data_collection(best_action, use_vlm=use_vlm)
                    if action_idx == total_action:
                        break
                else:
                    pipeline_name = pipeline_list.get(test_type, test_type)
                    if hasattr(self, f"{pipeline_name}_pipeline"):
                        pipeline_func = getattr(self, f"{pipeline_name}_pipeline")
                        params = {
                            "use_vlm": use_vlm,  # Fixed parameter
                            "uncertainty_threshold": uncertainty_threshold,  # Flexible parameter
                            "gt_action": action_sequence_answer[action_idx] if action_sequence_answer else None
                        }
                        param_names = inspect.signature(pipeline_func).parameters.keys()
                        params = {k: v for k, v in params.items() if k in param_names}
                        action_score = pipeline_func(**params)
                    else:
                        raise ValueError(f"Unsupported test type {test_type}")
                
                if threshold is not None:
                    # allow evaluation about human help
                    best_action = [action for action, score in action_score.items() if score > threshold]
                    if len(best_action) == 0:
                        # LLM have no confidence in any action
                        best_action = [max(action_score, key=action_score.get)]
                    if action_sequence_answer[action_idx] in best_action:
                        best_action = action_sequence_answer[action_idx]
                        if len(best_action) > 1:
                            human_help += 1
                    else:
                        write_score(action_idx, total_action, human_help)
                        break
                else:
                    best_action = max(action_score, key=action_score.get)
                action_idx += 1
                print(best_action)
                self.action_sequence.append(best_action)
                self.executing = True
                
            if self.record_video:
                img = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handles[0], gymapi.IMAGE_COLOR).reshape(1080, 1920, 4)[:,:,:-1]
                frame = np.array(img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                out.write(frame)

            dpose = torch.tensor([[[0.],[0.],[0.],[0.],[0.],[0.]]], device=self.device)

            if best_action is None:
                pass
            elif best_action == "scoop":
                dpose = self.scoop()
            elif best_action == "stir":
                dpose = self.stir()
            elif best_action == "fork":
                dpose = self.fork()
            elif best_action == "cut":
                dpose = self.cut()
            elif best_action == "drop_food":
                dpose = self.scoop_put()
            elif best_action == "pull_bowl_closer":
                dpose = self.pull_bowl_closer()
            elif best_action == 'grasp_spoon':
                dpose = self.take_tool('spoon')
            elif best_action == 'put_spoon_back':
                dpose = self.put_tool('spoon')
            elif best_action == 'open_dumbwaiter':
                dpose = self.open_dumbwaiter()
            elif best_action == 'close_dumbwaiter':
                dpose = self.close_dumbwaiter()
            elif best_action == 'start_dumbwaiter':
                dpose = self.start_dumbwaiter()
            elif best_action == 'put_bowl_into_dumbwaiter':
                dpose = self.put_bowl_into_dumbwaiter()
            elif best_action == 'take_bowl_out_dumbwaiter':
                raise NotImplementedError("Not implemented yet")
                dpose = self.take_bowl_out_dumbwaiter()
            elif best_action == "DONE" or best_action == 'REPLAN_ERROR' or len(self.action_sequence) >= max_sequence:
                break 
            elif "move" in best_action:
                for object in self.containers_list:
                    if object.split()[0] in best_action:
                        dpose = self.move(object, slow=True)
                        break
            if best_action and time() - self.action_start > action_time_limit or not (True in self.is_acting.values()):
                # print(f"{best_action} done")s
                self.executing = False
                self.action_state_reset()
                self.action_start = time()
            
            
            dpose = dpose.to(self.device)
            
            
            self.pos_action[:, :7] = self.dof_state[:, self.franka_dof_indices, 0].squeeze(-1)[:, :7] + self.control_ik(dpose)
       
            test_dof_state = self.dof_state[:, :, 0].contiguous()
            test_dof_state[:, self.franka_dof_indices] = self.pos_action

            franka_actor_indices = self.franka_indices.to(dtype=torch.int32)
            self.gym.set_dof_position_target_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(test_dof_state),
                gymtorch.unwrap_tensor(franka_actor_indices),
                len(franka_actor_indices)
            )

            # update the viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)

            self.gym.sync_frame_time(self.sim)

            self.frame += 1
        
        write_action_seq()
        if self.record_video:
            out.release()
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
        
    def render_config(self, file_path, text):
        self.reset()
        start = time()
        start_wait = 2
        while not self.gym.query_viewer_has_closed(self.viewer):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.render_all_camera_sensors(self.sim)

            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)
            
            
            if  time() - start > start_wait:
                rgb_img = self.gym.get_camera_image(self.sim, self.envs[0], self.camera_handles[0], gymapi.IMAGE_COLOR).reshape(1080, 1920, 4)[:,:,:-1]
                rgb_img = Image.fromarray(rgb_img)
                myFont = ImageFont.truetype('FreeMono.ttf', 36) 
                I1 = ImageDraw.Draw(rgb_img)
                # split text to multiple lines if too long
                I1.text((28, 36), text, font=myFont, fill=(0, 0, 128))
                rgb_img.save(file_path)
                break
        
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)

            self.gym.sync_frame_time(self.sim)

            self.frame += 1
            
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
           
    def reinit(self, env_cfg_dict):
        self.env_cfg_dict = env_cfg_dict
        #tool_type : spoon, knife, stir, fork
        self.tool = env_cfg_dict["tool"]
        self.tool_list = np.array(["spoon", "fork", "knife"]) if self.env_cfg_dict["tool"] == "None" else self.env_cfg_dict["tool"]
        # assert self.tool in ["spoon", "knife", "stir", "fork"]
        
        
        # initialize gym
        self.gym = gymapi.acquire_gym()

        # create simulator
        self.env_spacing = 1.5
        self.max_episode_length = 195
        self.asset_root = "urdf"
        self.gravity = -9.8
        
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(self.num_envs, self.num_dofs, 2)

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        _rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_state_tensor = gymtorch.wrap_tensor(_rb_state_tensor).view(-1, 13)

        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        jacobian = gymtorch.wrap_tensor(_jacobian)

        # jacobian entries corresponding to franka hand
        self.j_eef = jacobian[:, self.hand_joint_index, :, :7]
        
        # for trajectory collection
        self.record = []
    
    @property
    def instruction(self):
        return self.env_cfg_dict["instruction"] if self.env_cfg_dict["instruction"] != "None" else ""
    
    @property
    def tool_list(self):
        return self.env_cfg_dict["tool"] if self.env_cfg_dict["tool"] != "None" else np.array(["spoon"])

    @property
    def container_config_list(self):
        return self.env_cfg_dict["containers"]

if __name__ == "__main__":
    config = read_yaml("./src/config/final_task/obstacles.yaml", task_type='obstacles', env_idx=16)
    os.makedirs('temp', exist_ok=True)
    isaac = IsaacSim(config, log_folder='temp')
    isaac.data_collection()
    # isaac.simulate()
