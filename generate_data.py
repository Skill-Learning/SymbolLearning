import argparse
import gym
import numpy as np
from autolab_core import YamlConfig, RigidTransform
import matplotlib.pyplot as plt
from datetime import datetime
import os
import csv
from isaacgym import gymapi
from isaacgym_utils.scene import GymScene
from isaacgym_utils.assets import GymFranka, GymBoxAsset
from isaacgym_utils.camera import GymCamera
from isaacgym_utils.math_utils import RigidTransform_to_transform
from policy import *
from isaacgym_utils.draw import draw_transforms, draw_contacts, draw_camera
import time 
from visualization.visualizer3d import Visualizer3D as vis3d
import torch
from utils import *

# TODO: Policy Class 
# TODO: utils.py -> Observation Collection Class -> this collects pcd or image 
# TODO: utils.py -> loggin class 

def vis_cam_images(image_list):
    for i in range(0, len(image_list)):
        plt.figure()
        im = image_list[i].data
        # for showing normal map
        if im.min() < 0:
            im = im / 2 + 0.5
        plt.imshow(im)
    plt.show()


def subsample(pts, rate):
    n = int(rate * len(pts))
    idxs = np.arange(len(pts))
    np.random.shuffle(idxs)
    return pts[idxs[:n]]

class GenerateData():
    def __init__(self, cfg, object_type="std_cube"):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.object_type = object_type
    # Create Environment 
        self.scene = GymScene(cfg['scene'])
        self.franka = GymFranka(cfg['franka'], self.scene, actuation_mode='torques')
        self.table = GymBoxAsset(self.scene, **cfg['table']['dims'], shape_props = cfg['table']['shape_props'], asset_options = cfg['table']['asset_options'])
        self.franka_name, self.table_name, self.block_name = 'franka', 'table', 'block'

        # TODO - low priority: Replace name block with object

        if self.object_type == "std_cube":
            self.block = GymBoxAsset(self.scene, **cfg['block']['dims'], shape_props = cfg['block']['shape_props'], asset_options = cfg['block']['asset_options'], rb_props=cfg['block']['rb_props'])
        
        # TODO: Add other objects
        elif self.object_type == "std_cylinder":
            raise NotImplementedError
        elif self.object_type == "std_sphere":
            raise NotImplementedError
        elif self.object_type == "high_cube":
            self.block = GymBoxAsset(self.scene, **cfg['block']['high_dims'], shape_props = cfg['block']['shape_props'], asset_options = cfg['block']['asset_options'], rb_props=cfg['block']['rb_props'])
        elif self.object_type == "long_cube":
            self.block = GymBoxAsset(self.scene, **cfg['block']['long_dims'], shape_props = cfg['block']['shape_props'], asset_options = cfg['block']['asset_options'], rb_props=cfg['block']['rb_props'])
        elif self.object_type == "wide_cube":
            self.block = GymBoxAsset(self.scene, **cfg['block']['wide_dims'], shape_props = cfg['block']['shape_props'], asset_options = cfg['block']['asset_options'], rb_props=cfg['block']['rb_props'])
        # Add transforms to scene
        self.table_transform = gymapi.Transform(p=gymapi.Vec3(cfg['table']['dims']['sx']/3, 0, cfg['table']['dims']['sz']/2))
        self.franka_transform = gymapi.Transform(p=gymapi.Vec3(0, 0, cfg['table']['dims']['sz'] + 0.01))

        
        # Add Cameras 
        self.camera = GymCamera(self.scene, cfg['camera'])
        self.camera_names  = [f"cam{i}" for i in range(cfg['num_cameras'])] #num cameras = 3

        self.camera_transforms = [
            RigidTransform_to_transform(
                RigidTransform(
                    translation=[1.38, 0, 1],
                    rotation=np.array([
                        [0, 0, -1],
                        [1, 0, 0],
                        [0, -1, 0]
                    ]) @ RigidTransform.x_axis_rotation(np.deg2rad(-30))
            )),
            # # left
            # RigidTransform_to_transform(
            #     RigidTransform(
            #         translation=[0.5, -0.8, 1],
            #         rotation=np.array([
            #             [1, 0, 0],
            #             [0, 0, 1],
            #             [0, -1, 0]
            #         ]) @ RigidTransform.x_axis_rotation(np.deg2rad(-30))
            # )),
            # # right
            # RigidTransform_to_transform(
            #     RigidTransform(
            #         translation=[0.5, 0.8, 1],
            #         rotation=np.array([
            #             [-1, 0, 0],
            #             [0, 0, -1],
            #             [0, -1, 0]
            #         ]) @ RigidTransform.x_axis_rotation(np.deg2rad(-30))
            # ))
        ]
        assert len(self.camera_transforms) == cfg['num_cameras'], "Number of camera transforms must match number of cameras"
        def setup(scene, _):
            self.scene.add_asset('table', self.table, self.table_transform)
            self.scene.add_asset('franka', self.franka, self.franka_transform, collision_filter = 1)
            self.scene.add_asset('block', self.block, gymapi.Transform())
            for i in range(cfg['num_cameras']):
                self.scene.add_standalone_camera(self.camera_names[i], self.camera, self.camera_transforms[i])

        self.scene.setup_all_envs(setup)
        self.scene.render_cameras()

        # for drawing stuff in the scene
    def custom_draws(self,scene):
        for env_idx in scene.env_idxs:
            ee_transform = self.franka.get_ee_transform(env_idx, self.block_name)
            transforms = [ee_transform]
            draw_transforms(scene,[env_idx], transforms)
            for i in range(cfg['num_cameras']):
                draw_camera(scene, [env_idx], self.camera_transforms[i], length = 0.04)
        draw_contacts(scene, scene.env_idxs)
    

    def run_episode(self):
        # sample block poses
        block_dims = [self.block.sx,self.block.sy,self.block.sz]
        # actions = ['PokeX', 'PokeY', 'PokeTop' ,'GraspTop', 'GraspFront', 'GraspSide', 'Testing']
        # actions = ['PokeTop', 'PokeX', 'PokeY', 'PokeFrontRE', 'PokeFrontLE']
        actions = ['PokeFrontRE']
        action = actions[np.random.randint(0, len(actions))]
        if action == 'PokeX':
            policy = PokeFrontPolicy(self.franka, self.franka_name, self.block, self.block_name)
            action_vec = torch.tensor([1, 0, 0, 0, 0, 0])
        elif action == 'PokeY':
            policy = PokeSidePolicy(self.franka, self.franka_name, self.block, self.block_name, block_dims)
            action_vec = torch.tensor([0, 1, 0, 0, 0, 0])
        elif action == 'PokeFrontRE':
            policy = PushFrontREPolicy(self.franka, self.franka_name, self.block, self.block_name, block_dims)
            action_vec = torch.tensor([0, 0, 1, 0, 0, 0])
        elif action == 'PokeFrontLE':
            policy = PushFrontLEPolicy(self.franka, self.franka_name, self.block, self.block_name, block_dims)
            action_vec = torch.tensor([0, 0, 0, 1, 0, 0])
        elif action == 'PokeTop':
            policy = PokeFrontPolicy(self.franka, self.franka_name, self.block, self.block_name)
            if(self.object_type == "high_cube"):
                policy = TopplePolicy(self.franka, self.franka_name, self.block, self.block_name, block_dims)
            action_vec = torch.tensor([0, 0, 0, 0, 1, 0])
        # elif action == 'GraspSide':
        #     policy = GraspSidePolicy(self.franka, self.franka_name, self.block, self.block_name)
        #     action_vec = torch.tensor([0, 0, 0, 0, 1, 0])
        elif action == 'Testing':
            '''This is just a random condition to test the environment, move to the hardcoded position'''
            policy = PushFrontLEPolicy(self.franka, self.franka_name, self.block, self.block_name, block_dims)
            action_vec = torch.tensor([0, 0, 0, 0, 1, 0])
        else:
            raise ValueError(f"Invalid action {action}")

        if(self.object_type == "high_cube"):
            z = cfg['table']['dims']['sz'] + cfg['block']['high_dims']['sz'] / 2 + 0.1
        else:
            z =cfg['table']['dims']['sz'] + cfg['block']['dims']['sz'] / 2 + 0.1
        block_transforms = [gymapi.Transform(p=gymapi.Vec3(
            (np.random.rand()*2 - 1) * 0.1 + 0.4, 
            (np.random.rand()*2 - 1) * 0.2,
            z
        )) for _ in range(self.scene.n_envs)]

        # set block poses
        for env_idx in self.scene.env_idxs:
            self.block.set_rb_transforms(env_idx, self.block_name, [block_transforms[env_idx]])

        
        # Collect Object data 
        
        # * Pose: [p[x,y,z] r[x,y,z,w]]

        initial_poses = []
        initial_images = []
        policy.reset()
        for _ in range(100):
            self.scene.step()
        self.scene.render_cameras()
        for env_idx in self.scene.env_idxs:
            initial_poses.append(self.block.get_rb_poses_as_np_array(env_idx, self.block_name))
            img = self.camera.frames(env_idx, self.camera_names[0], True, False, False, False)['color'].raw_data
            img = torch.from_numpy(img).permute(2, 0, 1)/float(255.0)
            initial_images.append(img)
        initial_poses = np.array(initial_poses)
        initial_poses = torch.tensor(initial_poses)

        self.scene.run(time_horizon=policy.time_horizon, policy=policy, custom_draws=self.custom_draws)
        
        final_poses = []
        final_images = []
        self.scene.render_cameras()
        for env_idx in self.scene.env_idxs:
            final_poses.append(self.block.get_rb_poses_as_np_array(env_idx, self.block_name))
            img = self.camera.frames(env_idx, self.camera_names[0], True, False, False, False)['color'].raw_data
            img = torch.from_numpy(img).permute(2, 0, 1)/float(255.0)
            final_images.append(img)
        final_poses = np.array(final_poses)
        final_poses = torch.tensor(final_poses)
        return initial_images, initial_poses, final_images, final_poses, action_vec, self.object_type
        
    def generate_data(self, num_episodes, csv_path, data_dir):
        for i in range(num_episodes):
            initial_images, initial_poses, final_images, final_poses, action_vec, obj_type= self.run_episode()
            # for env_idx in self.scene.env_idxs:
            #     row = make_data_row(i, action_vec, initial_poses[env_idx], initial_images[env_idx], final_poses[env_idx], final_images[env_idx], data_dir, env_idx, obj_type= obj_type)
            #     with open(csv_path, 'a') as f:
            #         writer = csv.writer(f)
            #         writer.writerow(row)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()
    cfg = YamlConfig(args.config)
    header = ['timestamp', 'episode', 'object_type', 'action_vector',
            'initial_pose_position_x', 'initial_pose_position_y', 'initial_pose_position_z',
            'initial_pose_orientation_x', 'initial_pose_orientation_y', 'initial_pose_orientation_z', 'initial_pose_orientation_w',
            'final_pose_position_x', 'final_pose_position_y', 'final_pose_position_z',
            'final_pose_orientation_x', 'final_pose_orientation_y', 'final_pose_orientation_z', 'final_pose_orientation_w',
            'initial_image_path', 'final_image_path']


    curr_date = datetime.now().strftime("%Y%m%d")
    csv_path = f"data/{curr_date}/data.csv"
    data_dir = os.getcwd() + f"/data/{curr_date}"
    if(not os.path.exists(f"data/{curr_date}")):
        try:
            os.mkdir(f"data/{curr_date}")
            with open(csv_path, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)
            os.mkdir(f"data/{curr_date}/images")
        except OSError:
            print (f"Creation of the directory data/{curr_date} failed")

    # data_generater = GenerateData(cfg)

    # data_generater.generate_data(cfg['data']['num_episodes'], csv_path, data_dir)

    # data_generater = GenerateData(cfg,object_type='std_cube')
    # data_generater.generate_data(cfg['data']['num_episodes'], csv_path, data_dir)

    data_generater = GenerateData(cfg,object_type='high_cube')
    data_generater.generate_data(cfg['data']['num_episodes'], csv_path, data_dir)



