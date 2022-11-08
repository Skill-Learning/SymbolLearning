import argparse

import numpy as np
from autolab_core import YamlConfig, RigidTransform
import matplotlib.pyplot as plt

from isaacgym import gymapi
from isaacgym_utils.scene import GymScene
from isaacgym_utils.assets import GymFranka, GymBoxAsset
from isaacgym_utils.camera import GymCamera
from isaacgym_utils.math_utils import RigidTransform_to_transform
from policy import GraspFrontPolicy, GraspTopPolicy, PokeFrontPolicy, PokeSidePolicy, GraspBlockPolicy
from isaacgym_utils.draw import draw_transforms, draw_contacts, draw_camera

from visualization.visualizer3d import Visualizer3D as vis3d
import torch

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
    def __init__(self, cfg):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create Environment 
        self.scene = GymScene(cfg['scene'])
        self.franka = GymFranka(cfg['franka'], self.scene, actuation_mode='torques')
        self.table = GymBoxAsset(self.scene, **cfg['table']['dims'], shape_props = cfg['table']['shape_props'], asset_options = cfg['table']['asset_options'])
        # TODO: Sample block sizes from a distribution later
        # TODO: Add more shapes to sample from in the train function 
        self.block = GymBoxAsset(self.scene, **cfg['block']['dims'], shape_props = cfg['block']['shape_props'], asset_options = cfg['block']['asset_options'])

        self.franka_name, self.table_name, self.block_name = 'franka', 'table', 'block'

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
            # left
            RigidTransform_to_transform(
                RigidTransform(
                    translation=[0.5, -0.8, 1],
                    rotation=np.array([
                        [1, 0, 0],
                        [0, 0, 1],
                        [0, -1, 0]
                    ]) @ RigidTransform.x_axis_rotation(np.deg2rad(-30))
            )),
            # right
            RigidTransform_to_transform(
                RigidTransform(
                    translation=[0.5, 0.8, 1],
                    rotation=np.array([
                        [-1, 0, 0],
                        [0, 0, -1],
                        [0, -1, 0]
                    ]) @ RigidTransform.x_axis_rotation(np.deg2rad(-30))
            ))
        ]
        assert len(self.camera_transforms) == cfg['num_cameras'], "Number of camera transforms must match number of cameras"
        scene = self.scene
        def setup(scene, _):
            self.scene.add_asset('table', self.table, self.table_transform)
            self.scene.add_asset('franka', self.franka, self.franka_transform, collision_filter = 1)
            self.scene.add_asset('block', self.block, gymapi.Transform())
            for i in range(cfg['num_cameras']):
                self.scene.add_standalone_camera(self.camera_names[i], self.camera, self.camera_transforms[i])
            
        self.scene.setup_all_envs(setup)

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

        # actions = ['PokeX', 'PokeY', 'GraspTop', 'GraspFront', 'GraspSide', 'Testing']
        actions = ['Testing']
        action = actions[np.random.randint(0, len(actions))]
        # TODO: Create a new policy class for an ensemble of policies and the function selects the policy depending 
        # TODO: on the action selected from the draw
        if action == 'PokeX':
            policy = PokeFrontPolicy(self.franka, self.franka_name, self.block, self.block_name)
            action_vec = torch.tensor([1, 0, 0, 0, 0, 0])
        elif action == 'PokeY':
            policy = PokeSidePolicy(self.franka, self.franka_name, self.block, self.block_name)
            action_vec = torch.tensor([0, 1, 0, 0, 0, 0])
        elif action == 'GraspTop':
            policy = GraspTopPolicy(self.franka, self.franka_name, self.block, self.block_name)
            action_vec = torch.tensor([0, 0, 1, 0, 0, 0])
        elif action == 'GraspFront':
            policy = GraspFrontPolicy(self.franka, self.franka_name, self.block, self.block_name)
            action_vec = torch.tensor([0, 0, 0, 1, 0, 0])
        # elif action == 'GraspSide':
        #     policy = GraspSidePolicy(self.franka, self.franka_name, self.block, self.block_name)
        #     action_vec = torch.tensor([0, 0, 0, 0, 1, 0])
        elif action == 'Testing':
            '''This is just a random condition to test the environment, move to the hardcoded position'''
            ee_pose = self.franka.get_ee_transform(0, 'franka')
            # pose = gymapi.Transform(p = gymapi.Vec3(0.4, 0.4, 0.5), r = ee_pose.r)
            policy = GraspTopPolicy(self.franka, self.franka_name, self.block, self.block_name)
            action_vec = torch.tensor([0, 0, 0, 0, 0, 1])
        else:
            raise ValueError(f"Invalid action {action}")

        block_transforms = [gymapi.Transform(p=gymapi.Vec3(
            (np.random.rand()*2 - 1) * 0.1 + 0.5, 
            (np.random.rand()*2 - 1) * 0.2,
            cfg['table']['dims']['sz'] + cfg['block']['dims']['sz'] / 2 + 0.1
        )) for _ in range(self.scene.n_envs)]

        # set block poses
        for env_idx in self.scene.env_idxs:
            self.block.set_rb_transforms(env_idx, self.block_name, [block_transforms[env_idx]])

        
        # Collect Object data 
        # TODO: Collect object data 
        # TODO: This image/ pcd also need to have 2 extra channels for positional encoding 
        # ! need to find a smarter way to encode position, like transformers do

        # TODO(mj): Collect scene before running policy
        # observation_initial = torch.zeros(1, 3, 224, 224)
        import ipdb; ipdb.set_trace()
        obs_init_img = self.camera.frames(0, self.camera_names[0], True, False, False, False)['color'].raw_data
        observation_initial = torch.from_numpy(obs_init_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        # print(observation_initial.data)
        # imgplot = plt.imshow(observation_initial)
        # plt.show()
        policy.reset()
        self.scene.run(time_horizon=policy.time_horizon, policy=policy, custom_draws=self.custom_draws)
        
        # TODO(mj): Collect scene after running policy
        # observation_final = torch.zeros(1, 3, 224, 224)
        obs_final_img = self.camera.frames(0, self.camera_names[1], True, False, False, False)['color'].raw_data
        observation_final = torch.from_numpy(obs_final_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        # print(observation_final.raw_data)
        # import pdb; pdb.set_trace()
        # imgplot = plt.imshow(observation_final)
        # plt.show()
        return observation_initial, action_vec, observation_final
        
    def generate_data(self, num_episodes):
        obs_initial = []
        actions = []
        obs_final = []
        for _ in range(num_episodes):
            observation_initial, action_vec, observation_final = self.run_episode()
            obs_initial.append(observation_initial)
            print(observation_initial.shape)
            actions.append(action_vec)
            obs_final.append(observation_final)
            print(observation_final.shape)
        return obs_initial, actions, obs_final


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()
    cfg = YamlConfig(args.config)

    data_generater = GenerateData(cfg)
    # import pdb; pdb.set_trace()

    obs_initial, actions, obs_final = data_generater.generate_data(cfg['num_episodes'])

    # TODO: Save the data in a torch.pth file