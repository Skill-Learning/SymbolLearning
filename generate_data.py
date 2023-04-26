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
from isaacgym_utils.assets import GymFranka, GymBoxAsset, GymURDFAsset
from isaacgym_utils.camera import GymCamera
from isaacgym_utils.math_utils import RigidTransform_to_transform
from policy import GraspPointYPolicy
from isaacgym_utils.draw import draw_transforms, draw_contacts, draw_camera
import time 
from visualization.visualizer3d import Visualizer3D as vis3d
import torch
from utils import *
import open3d as o3d
from matplotlib import cm
from tqdm import tqdm 
# from pytorch3d.ops import sample_farthest_points


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
        self.franka = GymFranka(cfg['franka'], self.scene, actuation_mode='attractors')
        self.table = GymBoxAsset(self.scene, **cfg['table']['dims'], shape_props = cfg['table']['shape_props'], asset_options = cfg['table']['asset_options'])
        self.podium = GymBoxAsset(self.scene, **cfg['podium']['dims'], shape_props = cfg['podium']['shape_props'], asset_options = cfg['podium']['asset_options'])
        self.franka_name, self.table_name, self.block_name, self.podium_name = 'franka', 'table', 'block', 'podium'

        self.block = GymURDFAsset(
                        cfg['urdf']['urdf_path'],
                        self.scene, 
                        shape_props=cfg['urdf']['shape_props'],
                        rb_props=cfg['urdf']['rb_props'],
                    )
        # Add transforms to scene
        self.table_transform = gymapi.Transform(p=gymapi.Vec3(cfg['table']['dims']['sx']/3, 0, cfg['table']['dims']['sz']/2))
        self.franka_transform = gymapi.Transform(p=gymapi.Vec3(0, 0, cfg['table']['dims']['sz'] + 0.01))

        z =cfg['table']['dims']['sz'] + cfg['block']['dims']['sz'] / 2 + 0.1 + 0.3
        
        # self.block_transforms = [gymapi.Transform(p=gymapi.Vec3(
        #     (np.random.rand()*2 - 1) * 0.001 + 0.45, 
        #     (np.random.rand()*2 - 1) * 0.001 + 0.0,
        #     z),
        #     r=gymapi.Quat(0, 0, 0, 1) 
        # ) for _ in range(self.scene.n_envs)]
        self.block_transforms = [
            RigidTransform_to_transform(
                RigidTransform(
            translation = np.array([(np.random.rand()*2 - 1)*0.05 + 0.45, 
                                    (np.random.rand()*2 - 1)*0.05 + 0.0,
                                    z]),
            rotation = np.eye(3) @ RigidTransform.z_axis_rotation(np.random.rand())) 
            ) for _ in range(self.scene.n_envs)]



        self.podium_transforms = [gymapi.Transform(
            p=gymapi.Vec3(self.block_transforms[i].p.x,
                            self.block_transforms[i].p.y,
                            z - 0.3)
                            ) for i in range(self.scene.n_envs)]

        # Add Cameras 
        self.camera = GymCamera(self.scene, cfg['camera'])
        self.camera_names  = [f"cam{i}" for i in range(cfg['num_cameras'])] #num cameras = 3
        self.camera_transforms = [
        # front
            RigidTransform_to_transform(
                RigidTransform(
                    translation=[1.38, 0, self.block_transforms[0].p.z],
                    rotation=np.array([
                        [0, 0, -1],
                        [1, 0, 0],
                        [0, -1, 0]
                    ]) @ RigidTransform.x_axis_rotation(np.deg2rad(0))
            )),
            # left
            RigidTransform_to_transform(
                RigidTransform(
                    translation=[0.5, -0.8, self.block_transforms[0].p.z],
                    rotation=np.array([
                        [1, 0, 0],
                        [0, 0, 1],
                        [0, -1, 0]
                    ]) @ RigidTransform.x_axis_rotation(np.deg2rad(0))
            )),
            # right
            RigidTransform_to_transform(
                RigidTransform(
                    translation=[0.5, 0.8, self.block_transforms[0].p.z],
                    rotation=np.array([
                        [-1, 0, 0],
                        [0, 0, -1],
                        [0, -1, 0]
                    ]) @ RigidTransform.x_axis_rotation(np.deg2rad(0))
            )),
            #rearNegY(right)
            RigidTransform_to_transform(
                RigidTransform(
                    translation=[self.franka_transform.p.x-0.2,-0.075-0.2, self.block_transforms[0].p.z],
                    rotation=np.array([
                        [-1, 0, 0],
                        [0, 0, -1],
                        [0, -1, 0]
                    ]) @ RigidTransform.y_axis_rotation(np.deg2rad(-125))
            )),
            #rearPosY(left)
            RigidTransform_to_transform(
                RigidTransform(
                    translation=[self.franka_transform.p.x-0.2,-0.075+0.6, self.block_transforms[0].p.z],
                    rotation=np.array(([
                        [-1, 0, 0],
                        [0, 0, -1],
                        [0, -1, 0]
                    ]) @ RigidTransform.y_axis_rotation(np.deg2rad(-45))@RigidTransform.x_axis_rotation(np.deg2rad(-10)))
            )),
            #top1
            # RigidTransform_to_transform(
            #     RigidTransform(
            #         translation=[0.35,-0.075-0.3, 0.725+0.9],
            #         rotation=np.array([
            #             [-1, 0, 0],
            #             [0, 0, -1],
            #             [0, -1, 0]
            #         ]) @ RigidTransform.x_axis_rotation(np.deg2rad(-110))
            # )),
            # #top2
            # RigidTransform_to_transform(
            #     RigidTransform(
            #         translation=[0.35,-0.075+0.4, 0.725+0.9],
            #         rotation=np.array([
            #             [-1, 0, 0],
            #             [0, 0, -1],
            #             [0, -1, 0]
            #         ]) @ RigidTransform.x_axis_rotation(np.deg2rad(-70))
            # ))
        ]
        
        assert len(self.camera_transforms) == cfg['num_cameras'], "Number of camera transforms must match number of cameras"
        def setup(scene, _):
            self.scene.add_asset('table', self.table, self.table_transform)
            self.scene.add_asset('franka', self.franka, self.franka_transform, collision_filter = 1)
            self.scene.add_asset('podium', self.podium, gymapi.Transform())
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
    
    def vis_cam_images(self,image_list):
        for i in range(0, len(image_list)):
            plt.figure()
            im = image_list[i].data
            # for showing normal map
            if im.min() < 0:
                im = im / 2 + 0.5
            plt.imshow(im)
        plt.show()

    
    def generate_point_cloud(self,block_transform,visualize=True, debug=False):
    
        color_list, depth_list, seg_list, normal_list = [], [], [], []
        env_idx = 0
        for cam_name in self.camera_names:
            # get images of cameras in first env 
            frames = self.camera.frames(env_idx, cam_name)
            color_list.append(frames['color'])
            depth_list.append(frames['depth'])
            seg_list.append(frames['seg'])
            normal_list.append(frames['normal'])

        if debug:
            self.vis_cam_images(color_list)
            self.vis_cam_images(depth_list)
            self.vis_cam_images(seg_list)
            self.vis_cam_images(normal_list) 
            import ipdb; ipdb.set_trace()
        
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
            loc_z=np.where(points[:,2]>0.81,True,False)
            points=points[loc_z]
            
            loc_z=np.where(points[:,2]<block_transform.p.z+0.01 ,True,False)
            points=points[loc_z]


            loc_x=np.where(points[:,0]>0.35,True,False)
            points=points[loc_x]

            loc_x=np.where(points[:,0]<0.9,True,False)
            points=points[loc_x]

            loc_y=np.where(points[:,1]>-0.075-0.2,True,False)
            points=points[loc_y]

            loc_y=np.where(points[:,1]<-0.075+0.5,True,False)
            points=points[loc_y]

            point_cloud.append(points)
            
        
        point_cloud=np.concatenate(point_cloud)
        import ipdb; ipdb.set_trace()
        #Change Normals
        min_z = np.min(point_cloud[:,2])
        max_z = np.max(point_cloud[:,2])

        min_bol=np.where(point_cloud[:,2]>=1.01*min_z,True, False)
        point_cloud=point_cloud[min_bol]

        top_bol=np.where(point_cloud[:,2]<=0.98*max_z,False, True)
        point_cloud_top=point_cloud[top_bol]
        point_cloud_bottom=point_cloud[~top_bol]

        #Centroid calculation
        top_bol=np.where(point_cloud[:,2]<=0.991*max_z,False, True)
        point_cloud_centroid=point_cloud[top_bol]
        centroid=np.mean(point_cloud_centroid,axis=0)
        
        
        #Processing top portion of point cloud
        pcd_top=o3d.geometry.PointCloud()
        pcd_top.points = o3d.utility.Vector3dVector(point_cloud_top)
        pcd_top = pcd_top.voxel_down_sample(voxel_size=cfg['point_cloud']['vox_size'])
        normals = np.array([0,0,1])
        normals = np.tile(normals, (np.asarray(pcd_top.points).shape[0], 1))
        pcd_top.normals = o3d.utility.Vector3dVector(normals)
        # print(np.asarray(pcd_top.points).shape)

        #Processing bottom portion of point cloud
        pcd_rest=o3d.geometry.PointCloud()
        pcd_rest.points = o3d.utility.Vector3dVector(point_cloud_bottom)
        pcd_rest = pcd_rest.voxel_down_sample(voxel_size=cfg['point_cloud']['vox_size'])
        pcd_rest.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        #Merging point cloud
        pcd=pcd_rest + pcd_top

        #Compute farthest point sampling
        points=np.asarray(pcd.points)
        # import ipdb; ipdb.set_trace()
        indices,_=farthest_point_sampling(points,k=cfg['point_cloud']['num_points'])

        # pcd.estimate_normals()
        if visualize:
            for camera_pose in camera_poses:
                vis3d.pose(camera_pose)

            # vis3d.points(
            #         points[indices[0]], 
            #         color=cm.tab10.colors[i],
            #         scale=0.005
            #     )

            # vis3d.points(
            #         centroid, 
            #         color=[0.5,0.5,0.5],
            #         scale=0.005
            #     )

            o3d.visualization.draw_geometries([pcd], point_show_normal=True)

            # vis3d.show()
        return {'full_point':points, 'downsampled_point':points[indices], 'normals':np.asarray(pcd.normals)[indices[0]], 'centoid':centroid}

    def run_episode(self):

        # Collect Object data 
        
        # * Pose: [p[x,y,z] r[x,y,z,w]]
        # import ipdb;ipdb.set_trace()
        # set block poses
        for env_idx in self.scene.env_idxs:
            self.block.set_rb_transforms(env_idx, self.block_name, [self.block_transforms[env_idx]])
            self.podium.set_rb_transforms(env_idx, self.podium_name, [self.podium_transforms[env_idx]])
        initial_poses = []
        final_poses = np.zeros((len(self.scene.env_idxs), 7))
        grasping_points = np.zeros((len(self.scene.env_idxs), 3))
        point_clouds = []
        normals=[]
        point_cloud_downsampled=[]
        centroids = []
        for _ in range(100):
            self.scene.step()
        self.scene.render_cameras()
        for i,env_idx in enumerate(self.scene.env_idxs):
            initial_poses.append(self.block.get_rb_poses_as_np_array(env_idx, self.block_name))
            self.scene.render_cameras()
            data = self.generate_point_cloud(self.block_transforms[i], visualize=cfg['flags']['visualize'])
            point_clouds.append(data['full_point'])
            normals.append(data['normals'])
            point_cloud_downsampled.append(data['downsampled_point'])
            centroids.append(data['centoid'])

        actions = ['GraspPointX']
        action = actions[np.random.randint(0, len(actions))]
        if action == 'GraspPointX':
            policy = GraspPointYPolicy(self.franka, self.franka_name, self.block, self.block_name, point_cloud_downsampled, normals, cfg['scene']['n_envs'], centroid_point=centroids, final_poses = final_poses, grasping_points = grasping_points)
        else:
            raise ValueError(f"Invalid action {action}")
        
        initial_poses = np.array(initial_poses)
        initial_poses = torch.tensor(initial_poses)

        self.scene.run(time_horizon=policy.time_horizon, policy=policy, custom_draws=self.custom_draws)
        self.scene.render_cameras()
        final_poses = np.array(final_poses)
        final_poses = torch.tensor(final_poses)
        final_poses = final_poses.unsqueeze(0)

        grasping_points = np.array(grasping_points)
        grasping_points = torch.tensor(grasping_points)


        action_vec = torch.tensor([1, 0, 0, 0, 0, 0])
        return  initial_poses, point_clouds, point_cloud_downsampled, final_poses, action_vec, self.object_type, normals, grasping_points
        
    def generate_data(self, num_episodes, csv_writer, save_data=False):
        for i in tqdm(range(num_episodes)):
            initial_poses, point_clouds, point_cloud_downsampled, final_poses, action_vec, obj_type, normals, grasping_points = self.run_episode()
            
            if save_data:
                for env_idx in self.scene.env_idxs:
                    row = make_data_row(i, action_vec, initial_poses[env_idx], point_clouds[env_idx],final_poses[env_idx], data_dir, env_idx, grasping_points,obj_type= obj_type)
                    csv_writer.writerow(row)
                    # import ipdb; ipdb.set_trace()


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
            'entire_pc_path','grasping_pointx','grasping_pointy','grasping_pointz']


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

    

