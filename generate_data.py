import argparse
import gym
import numpy as np
from autolab_core import YamlConfig, RigidTransform, PointCloud
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
import open3d as o3d
from matplotlib import cm


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

        if(self.object_type == "high_cube"):
            z = cfg['table']['dims']['sz'] + cfg['block']['high_dims']['sz'] / 2 + 0.1
        else:
            z =cfg['table']['dims']['sz'] + cfg['block']['dims']['sz'] / 2 + 0.1
        
        self.block_transforms = [gymapi.Transform(p=gymapi.Vec3(
            (np.random.rand()*2 - 1) * 0.1 + 0.3, 
            (np.random.rand()*2 - 1) * 0.2 + 0.1,
            z
        )) for _ in range(self.scene.n_envs)]
        # Add Cameras 
        self.camera = GymCamera(self.scene, cfg['camera'])
        self.camera_names  = [f"cam{i}" for i in range(cfg['num_cameras'])] #num cameras = 3
        self.camera_transforms = [
        # front
            RigidTransform_to_transform(
                RigidTransform(
                    translation=[1.38, 0, 0.725],
                    rotation=np.array([
                        [0, 0, -1],
                        [1, 0, 0],
                        [0, -1, 0]
                    ]) @ RigidTransform.x_axis_rotation(np.deg2rad(0))
            )),
            # left
            RigidTransform_to_transform(
                RigidTransform(
                    translation=[0.5, -0.8, self.franka_transform.p.z+0.05],
                    rotation=np.array([
                        [1, 0, 0],
                        [0, 0, 1],
                        [0, -1, 0]
                    ]) @ RigidTransform.x_axis_rotation(np.deg2rad(0))
            )),
            # right
            RigidTransform_to_transform(
                RigidTransform(
                    translation=[0.5, 0.8, self.franka_transform.p.z+0.05],
                    rotation=np.array([
                        [-1, 0, 0],
                        [0, 0, -1],
                        [0, -1, 0]
                    ]) @ RigidTransform.x_axis_rotation(np.deg2rad(0))
            )),
            #rearNegY(right)
            RigidTransform_to_transform(
                RigidTransform(
                    translation=[self.franka_transform.p.x,-0.075-0.2, 0.725+0.01],
                    rotation=np.array([
                        [-1, 0, 0],
                        [0, 0, -1],
                        [0, -1, 0]
                    ]) @ RigidTransform.y_axis_rotation(np.deg2rad(-135))
            )),
            #rearPosY(left)
            RigidTransform_to_transform(
                RigidTransform(
                    translation=[self.franka_transform.p.x,-0.075+0.6, 0.725+0.01],
                    rotation=np.array([
                        [-1, 0, 0],
                        [0, 0, -1],
                        [0, -1, 0]
                    ]) @ RigidTransform.y_axis_rotation(np.deg2rad(-45))
            )),
            #top
            RigidTransform_to_transform(
                RigidTransform(
                    translation=[0.4,-0.075, 0.725+0.4],
                    rotation=np.array([
                        [-1, 0, 0],
                        [0, 0, -1],
                        [0, -1, 0]
                    ]) @ RigidTransform.x_axis_rotation(np.deg2rad(-90))
            ))
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
    
    def generate_point_cloud(self,block_transform,visualize=True):
    
        color_list, depth_list, seg_list, normal_list = [], [], [], []
        env_idx = 0
        for cam_name in self.camera_names:
            # get images of cameras in first env 
            frames = self.camera.frames(env_idx, cam_name)
            color_list.append(frames['color'])
            depth_list.append(frames['depth'])
            seg_list.append(frames['seg'])
            normal_list.append(frames['normal'])
        
        intrs = [self.camera.get_intrinsics(cam_name) for cam_name in self.camera_names]

        # Deproject to point clouds
        pcs_cam = []
        for i, depth in enumerate(depth_list):
            pc_raw = intrs[i].deproject(depth)
            points_filtered = pc_raw.data[:, np.logical_not(np.any(pc_raw.data > 5, axis=0))]
            pcs_cam.append(PointCloud(points_filtered, pc_raw.frame))

        # Get camera poses
        camera_poses = [
            self.camera.get_extrinsics(env_idx, cam_name)
            for cam_name in self.camera_names
        ]

        # Transform point clouds from camera frame into world frame
        pcs_world = [camera_poses[i] * pc for i, pc in enumerate(pcs_cam)]

        point_cloud=[]
        for i, pc in enumerate(pcs_world):

            points=pc.data.T
            loc_z=np.where(points[:,2]>0.508 ,True,False)
            points=points[loc_z]

            loc_z=np.where(points[:,2]<block_transform.p.z+0.125 ,True,False)
            points=points[loc_z]


            loc_x=np.where(points[:,0]>block_transform.p.x-0.025,True,False)
            points=points[loc_x]

            loc_x=np.where(points[:,0]<0.9,True,False)
            points=points[loc_x]

            loc_y=np.where(points[:,1]>-0.075-0.2,True,False)
            points=points[loc_y]

            loc_y=np.where(points[:,1]<-0.075+0.5,True,False)
            points=points[loc_y]

            point_cloud.append(points)
            
        
        point_cloud=np.concatenate(point_cloud)
        pcd=o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        pcd = pcd.voxel_down_sample(voxel_size=cfg['point_cloud']['vox_size'])
        pcd=pcd.farthest_point_down_sample(cfg['point_cloud']['num_points'])

        if visualize:
            for camera_pose in camera_poses:
                vis3d.pose(camera_pose)

            vis3d.points(
                    pcd.points, 
                    color=cm.tab10.colors[i],
                    scale=0.005
                )

            vis3d.show()

        return pcd

    def run_episode(self):
        # sample block poses
        block_dims = [self.block.sx,self.block.sy,self.block.sz]
        # actions = ['PokeX', 'PokeY', 'PokeTop' ,'GraspTop', 'GraspFront', 'GraspSide', 'Testing']
        # actions = ['PokeFrontRE', 'PokeFrontLE']
        # actions = ['PokeTop', 'PokeX', 'PokeY']
        actions = ['PokeTop']
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

        # set block poses
        for env_idx in self.scene.env_idxs:
            self.block.set_rb_transforms(env_idx, self.block_name, [self.block_transforms[env_idx]])

        
        # Collect Object data 
        
        # * Pose: [p[x,y,z] r[x,y,z,w]]
        # import ipdb;ipdb.set_trace()

        initial_poses = []
        point_clouds = []
        policy.reset()
        for _ in range(100):
            self.scene.step()
        self.scene.render_cameras()
        for i,env_idx in enumerate(self.scene.env_idxs):
            initial_poses.append(self.block.get_rb_poses_as_np_array(env_idx, self.block_name))
            self.scene.render_cameras()
            pc = self.generate_point_cloud(self.block_transforms[i], visualize=cfg['flags']['visualize'])
            point_clouds.append(pc)

        initial_poses = np.array(initial_poses)
        initial_poses = torch.tensor(initial_poses)


        self.scene.run(time_horizon=policy.time_horizon, policy=policy, custom_draws=self.custom_draws)
        
        final_poses = []
        self.scene.render_cameras()
        for env_idx in self.scene.env_idxs:
            final_poses.append(self.block.get_rb_poses_as_np_array(env_idx, self.block_name))

        final_poses = np.array(final_poses)
        final_poses = torch.tensor(final_poses)
        return  initial_poses, point_clouds, final_poses, action_vec, self.object_type
        
    def generate_data(self, num_episodes, csv_writer, save_data=False):
        for i in range(num_episodes):
            initial_poses, point_clouds, final_poses, action_vec, obj_type= self.run_episode()
            
            if save_data:
                for env_idx in self.scene.env_idxs:
                    row = make_data_row(i, action_vec, initial_poses[env_idx], point_clouds[env_idx], final_poses[env_idx], data_dir, env_idx, obj_type= obj_type)
                    csv_writer.writerow(row)


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
    data_dir = f"./data/{curr_date}"
    csv_path = f"./data/{curr_date}/data.csv"

    # import ipdb; ipdb.set_trace()
    if not os.path.exists('./data'):
        os.mkdir('./data')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        
    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        data_generater = GenerateData(cfg,object_type='high_cube')
        data_generater.generate_data(cfg['data']['num_episodes'], csv_writer=writer, save_data=cfg['flags']['save_data'])
    
    f.close()

    

