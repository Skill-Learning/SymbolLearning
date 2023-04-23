from abc import ABC, abstractmethod
import numpy as np
from autolab_core import RigidTransform, transformations
from isaacgym import gymapi
from math_utils import min_jerk, slerp_quat, vec3_to_np, np_to_vec3, \
                    project_to_line, compute_task_space_impedance_control, transform_to_RigidTransform,\
                        RigidTransform_to_transform, rotation_between_axes




class Policy(ABC):

    def __init__(self):
        self._time_horizon = -1

    @abstractmethod
    def __call__(self, scene, env_idx, t_step, t_sim):
        pass

    def reset(self):
        pass

    @property
    def time_horizon(self):
        return self._time_horizon


class RandomDeltaJointPolicy(Policy):

    def __init__(self, franka, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._franka = franka
        self._name = name

    def __call__(self, scene, env_idx, _, __):
        delta_joints = (np.random.random(self._franka.n_dofs) * 2 - 1) * ([0.05] * 7 + [0.005] * 2)
        self._franka.apply_delta_joint_targets(env_idx, self._name, delta_joints)


class GraspBlockPolicy(Policy):

    def __init__(self, franka, franka_name, block, block_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._franka = franka
        self._franka_name = franka_name
        self._block = block
        self._block_name = block_name

        self._time_horizon = 1000

        self.reset()

    def reset(self):
        self._pre_grasp_transforms = []
        self._grasp_transforms = []
        self._init_ee_transforms = []
        self._ee_waypoint_policies = []

    def __call__(self, scene, env_idx, t_step, t_sim):
        ee_transform = self._franka.get_ee_transform(env_idx, self._franka_name)

        if t_step == 0:
            self._init_ee_transforms.append(ee_transform)
            self._ee_waypoint_policies.append(
                EEImpedanceWaypointPolicy(self._franka, self._franka_name, ee_transform, ee_transform, T=20)
            )

        if t_step == 20:
            block_transform = self._block.get_rb_transforms(env_idx, self._block_name)[0]
            grasp_transform = gymapi.Transform(p=block_transform.p, r=self._init_ee_transforms[env_idx].r)
            pre_grasp_transfrom = gymapi.Transform(p=grasp_transform.p + gymapi.Vec3(0, 0, 0.2), r=grasp_transform.r)

            self._grasp_transforms.append(grasp_transform)
            self._pre_grasp_transforms.append(pre_grasp_transfrom)

            self._ee_waypoint_policies[env_idx] = \
                EEImpedanceWaypointPolicy(
                    self._franka, self._franka_name, ee_transform, self._pre_grasp_transforms[env_idx], T=180
                )

        if t_step == 200:
            self._ee_waypoint_policies[env_idx] = \
                EEImpedanceWaypointPolicy(
                    self._franka, self._franka_name, self._pre_grasp_transforms[env_idx], self._grasp_transforms[env_idx], T=100
                )

        if t_step == 300:
            self._franka.close_grippers(env_idx, self._franka_name)
        
        if t_step == 400:
            self._ee_waypoint_policies[env_idx] = \
                EEImpedanceWaypointPolicy(
                    self._franka, self._franka_name, self._grasp_transforms[env_idx], self._pre_grasp_transforms[env_idx], T=100
                )

        if t_step == 500:
            self._ee_waypoint_policies[env_idx] = \
                EEImpedanceWaypointPolicy(
                    self._franka, self._franka_name, self._pre_grasp_transforms[env_idx], self._grasp_transforms[env_idx], T=100
                )

        if t_step == 600:
            self._franka.open_grippers(env_idx, self._franka_name)

        if t_step == 700:
            self._ee_waypoint_policies[env_idx] = \
                EEImpedanceWaypointPolicy(
                    self._franka, self._franka_name, self._grasp_transforms[env_idx], self._pre_grasp_transforms[env_idx], T=100
                )

        if t_step == 800:
            self._ee_waypoint_policies[env_idx] = \
                EEImpedanceWaypointPolicy(
                    self._franka, self._franka_name, self._pre_grasp_transforms[env_idx], self._init_ee_transforms[env_idx], T=100
                )

        self._ee_waypoint_policies[env_idx](scene, env_idx, t_step, t_sim)


class GraspPointPolicy(Policy):

    def __init__(self, franka, franka_name, grasp_transform, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._franka = franka
        self._franka_name = franka_name
        self._grasp_transform = grasp_transform

        self._time_horizon = 710

        self.reset()

    def reset(self):
        self._pre_grasp_transforms = []
        self._grasp_transforms = []
        self._init_ee_transforms = []

    def __call__(self, scene, env_idx, t_step, _):
        t_step = t_step % self._time_horizon

        if t_step == 0:
            self._init_joints = self._franka.get_joints(env_idx, self._franka_name)
            self._init_rbs = self._franka.get_rb_states(env_idx, self._franka_name)

        if t_step == 20:
            ee_transform = self._franka.get_ee_transform(env_idx, self._franka_name)
            self._init_ee_transforms.append(ee_transform)

            pre_grasp_transfrom = gymapi.Transform(p=self._grasp_transform.p, r=self._grasp_transform.r)
            pre_grasp_transfrom.p.z += 0.2

            self._grasp_transforms.append(self._grasp_transform)
            self._pre_grasp_transforms.append(pre_grasp_transfrom)

            self._franka.set_ee_transform(env_idx, self._franka_name, self._pre_grasp_transforms[env_idx])

        if t_step == 100:
            self._franka.set_ee_transform(env_idx, self._franka_name, self._grasp_transforms[env_idx])

        if t_step == 150:
            self._franka.close_grippers(env_idx, self._franka_name)
        
        if t_step == 250:
            self._franka.set_ee_transform(env_idx, self._franka_name, self._pre_grasp_transforms[env_idx])

        if t_step == 350:
            self._franka.set_ee_transform(env_idx, self._franka_name, self._grasp_transforms[env_idx])

        if t_step == 500:
            self._franka.open_grippers(env_idx, self._franka_name)

        if t_step == 550:
            self._franka.set_ee_transform(env_idx, self._franka_name, self._pre_grasp_transforms[env_idx])

        if t_step == 600:
            self._franka.set_ee_transform(env_idx, self._franka_name, self._init_ee_transforms[env_idx])

        if t_step == 700:
            self._franka.set_joints(env_idx, self._franka_name, self._init_joints)
            self._franka.set_rb_states(env_idx, self._franka_name, self._init_rbs)


class FrankaEEImpedanceController:

    def __init__(self, franka, franka_name):
        self._franka = franka
        self._franka_name = franka_name
        self._elbow_joint = 3

        Kp_0, Kr_0 = 200, 8
        Kp_1, Kr_1 = 200, 5
        self._Ks_0 = np.diag([Kp_0] * 3 + [Kr_0] * 3)
        self._Ds_0 = np.diag([4 * np.sqrt(Kp_0)] * 3 + [2 * np.sqrt(Kr_0)] * 3)
        self._Ks_1 = np.diag([Kp_1] * 3 + [Kr_1] * 3)
        self._Ds_1 = np.diag([4 * np.sqrt(Kp_1)] * 3 + [2 * np.sqrt(Kr_1)] * 3)

    def compute_tau(self, env_idx, target_transform):
        # primary task - ee control
        ee_transform = self._franka.get_ee_transform(env_idx, self._franka_name)

        J = self._franka.get_jacobian(env_idx, self._franka_name)
        q_dot = self._franka.get_joints_velocity(env_idx, self._franka_name)[:7]
        x_vel = J @ q_dot

        tau_0 = compute_task_space_impedance_control(J, ee_transform, target_transform, x_vel, self._Ks_0, self._Ds_0)

        # secondary task - elbow straight
        link_transforms = self._franka.get_links_transforms(env_idx, self._franka_name)
        elbow_transform = link_transforms[self._elbow_joint]

        u0 = vec3_to_np(link_transforms[0].p)[:2]
        u1 = vec3_to_np(link_transforms[-1].p)[:2]
        curr_elbow_xyz = vec3_to_np(elbow_transform.p)
        goal_elbow_xy = project_to_line(curr_elbow_xyz[:2], u0, u1)
        elbow_target_transform = gymapi.Transform(
            p=gymapi.Vec3(goal_elbow_xy[0], goal_elbow_xy[1], curr_elbow_xyz[2] + 0.2),
            r=elbow_transform.r
        )

        J_elb = self._franka.get_jacobian(env_idx, self._franka_name, target_joint=self._elbow_joint)
        x_vel_elb = J_elb @ q_dot

        tau_1 = compute_task_space_impedance_control(J_elb, elbow_transform, elbow_target_transform, x_vel_elb, self._Ks_1, self._Ds_1)
        
        # nullspace projection
        JT_inv = np.linalg.pinv(J.T)
        Null = np.eye(7) - J.T @ (JT_inv)
        tau = tau_0 + Null @ tau_1

        return tau


class EEImpedanceWaypointPolicy(Policy):

    def __init__(self, franka, franka_name, init_ee_transform, goal_ee_transform, T=300):
        self._franka = franka
        self._franka_name = franka_name

        self._T = T
        self._ee_impedance_ctrlr = FrankaEEImpedanceController(franka, franka_name)

        init_ee_pos = vec3_to_np(init_ee_transform.p)
        goal_ee_pos = vec3_to_np(goal_ee_transform.p)
        self._traj = [
            gymapi.Transform(
                p=np_to_vec3(min_jerk(init_ee_pos, goal_ee_pos, t, self._T)),
                r=slerp_quat(init_ee_transform.r, goal_ee_transform.r, t, self._T),
            )
            for t in range(self._T)
        ]

    @property
    def horizon(self):
        return self._T

    def __call__(self, scene, env_idx, t_step, t_sim):
        target_transform = self._traj[min(t_step, self._T - 1)]
        tau = self._ee_impedance_ctrlr.compute_tau(env_idx, target_transform)
        self._franka.apply_torque(env_idx, self._franka_name, tau)


class GraspPolicy(Policy):
    def __init__(self, franka, franka_name, block, block_name, points, point_normals, n_envs, *args, **kwargs):
        super().__init__(*args, **kwargs)

        """
        :param franka: Franka object
        :param franka_name: name of the franka
        :param block: block object
        :param block_name: name of the block
        :param points: list of points to grasp [o3d.PointCloud]*n
        :param point_normals: list of point normals [np.array([x,y,z])]*n
        Note: n should be equal to the n_envs from scene
        # TODO MS: add an assert for the above check in the generate_data.py
        """

        self._franka = franka
        self._franka_name = franka_name
        self._time_horizon = 710
        self._block = block
        self._block_name = block_name
        self._points = points
        self._point_normals = point_normals
        self._n_envs = n_envs
        self.reset()

    def reset(self):
        self._pre_grasp_transforms = []
        self._grasp_transforms = []
        self._init_ee_transforms = []
        self._ee_waypoint_policies = []
        self._post_grasp_transforms = []

    @abstractmethod
    def _get_grasp_transform(self, env_idx):
        raise NotImplementedError

    @abstractmethod
    def _get_pre_grasp_transform(self, env_idx, grasp_transform):
        raise NotImplementedError        

    def __call__(self, scene, env_idx, t_step, t_sim):
        ee_transform = self._franka.get_ee_transform(env_idx, self._franka_name)

        if t_step == 0:
            self._init_joints = self._franka.get_joints(env_idx, self._franka_name)
            self._init_rbs = self._franka.get_rb_states(env_idx, self._franka_name)

        if t_step == 20:
            ee_transform = self._franka.get_ee_transform(env_idx, self._franka_name)
            self._init_ee_transforms.append(ee_transform)

            grasp_transform = self._get_grasp_transform(env_idx, ee_transform)
            self._grasp_transforms.append(grasp_transform)
            self._pre_grasp_transforms.append(self._get_pre_grasp_transform(env_idx, grasp_transform))
            post_grasp_transform = gymapi.Transform(p=grasp_transform.p + gymapi.Vec3(0, 0, 0.2), r=grasp_transform.r)
            self._post_grasp_transforms.append(post_grasp_transform)

            self._franka.set_ee_transform(env_idx, self._franka_name, self._pre_grasp_transforms[env_idx])

        if t_step == 100:
            self._franka.set_ee_transform(env_idx, self._franka_name, self._grasp_transforms[env_idx])

        if t_step == 150:
            self._franka.close_grippers(env_idx, self._franka_name)
        
        if t_step == 250:
            self._franka.set_ee_transform(env_idx, self._franka_name, self._post_grasp_transforms[env_idx])

        if t_step == 350:
            self._franka.set_ee_transform(env_idx, self._franka_name, self._grasp_transforms[env_idx])

        if t_step == 500:
            self._franka.open_grippers(env_idx, self._franka_name)


        if t_step == 550:
            self._franka.set_ee_transform(env_idx, self._franka_name, self._pre_grasp_transforms[env_idx])

        if t_step == 600:
            self._franka.set_ee_transform(env_idx, self._franka_name, self._init_ee_transforms[env_idx])

        if t_step == 700:
            self._franka.set_joints(env_idx, self._franka_name, self._init_joints)
            self._franka.set_rb_states(env_idx, self._franka_name, self._init_rbs)
    
class GraspPointXPolicy(GraspPolicy):

    def _get_grasp_transform(self, env_idx, ee_transform):
        # pc = np.asarray(self._points[env_idx].points)
        # max_z_idx = np.argmax(pc[:,2])
        random_index = np.random.randint(0, self._n_envs)
        random_index = 0

        # import ipdb; ipdb.set_trace()
        grasping_location = np.asarray(self._points[env_idx].points)[random_index]

        grasping_normal = -self._point_normals[env_idx][random_index]

        # align the z-axis of the gripper with the normal of the point
        gripper_transform = transform_to_RigidTransform(ee_transform)
        gripper_z = gripper_transform.z_axis

        rotation_matrix = rotation_between_axes(gripper_z, grasping_normal)

        # rotate the ee_transform with the rotation matrix
        grasp_transform = RigidTransform(
            translation=grasping_location,
            rotation = gripper_transform.rotation @ rotation_matrix,
        )

        # grasp_transform = RigidTransform(
        #     translation=grasping_location,
        #     rotation = gripper_transform.rotation,
        # )

        return RigidTransform_to_transform(grasp_transform)

    
    def _get_pre_grasp_transform(self, env_idx, grasp_transform):
        
        grasp_rigid_transform = transform_to_RigidTransform(grasp_transform)

        grasp_z = grasp_rigid_transform.z_axis

        pre_grasp_transform = RigidTransform(
            translation=grasp_rigid_transform.translation + 0.1 * np.array([0, 0, 1]),
            rotation=grasp_rigid_transform.rotation,
        )

        return RigidTransform_to_transform(pre_grasp_transform)



class GraspPointYPolicy(GraspPolicy):

    def _get_grasp_transform(self, env_idx, ee_transform):
        random_index = np.random.randint(0, self._n_envs)
        random_index = 0
        grasping_location = np.asarray(self._points[env_idx].points)[random_index]
        grasping_normal = -self._point_normals[env_idx][random_index]

        # align the z-axis of the gripper with the normal of the point
        gripper_transform = transform_to_RigidTransform(ee_transform)
        gripper_z = gripper_transform.z_axis

        rotation_matrix = rotation_between_axes(gripper_z, grasping_normal)

        # rotate the ee_transform with the rotation matrix
        grasp_transform = RigidTransform(
            translation=grasping_location,
            rotation = gripper_transform.rotation @ rotation_matrix @ RigidTransform.z_axis_rotation(np.pi/2),
        )

        return RigidTransform_to_transform(grasp_transform)

    
    def _get_pre_grasp_transform(self, env_idx, grasp_transform):
        
        grasp_rigid_transform = transform_to_RigidTransform(grasp_transform)

        grasp_z = grasp_rigid_transform.z_axis

        pre_grasp_transform = RigidTransform(
            translation=grasp_rigid_transform.translation - 0.1 * grasp_z,
            rotation=grasp_rigid_transform.rotation,
        )

        return RigidTransform_to_transform(pre_grasp_transform)

class GraspPointBehindPolicy(GraspPolicy):

    def _get_grasp_transform(self, env_idx, ee_transform):

        pcd = np.asarray(self._points[env_idx].points)
        min_x = np.min(pcd[:,0])
        max_x = np.max(pcd[:,0])
        rear_point_range = min_x + (max_x - min_x) * 0.1
        rear_points = pcd[pcd[:,0] < rear_point_range]

        rear_points = rear_points[rear_points[:,2].argsort()]
        grasping_location = rear_points[len(rear_points)//2]

        # align the z-axis of the gripper with the normal of the point
        gripper_transform = transform_to_RigidTransform(ee_transform)

        # rotate the ee_transform with the rotation matrix
        grasp_transform = RigidTransform(
            translation=grasping_location,
            rotation = gripper_transform.rotation,
        )

        return RigidTransform_to_transform(grasp_transform)

    
    def _get_pre_grasp_transform(self, env_idx, grasp_transform):
        
        grasp_rigid_transform = transform_to_RigidTransform(grasp_transform)

        pre_grasp_transform = RigidTransform(
            translation=grasp_rigid_transform.translation - 0.1 * np.array([1,0,0]),
            rotation=grasp_rigid_transform.rotation,
        )

        return RigidTransform_to_transform(pre_grasp_transform)
    

class GraspPointRightPolicy(GraspPolicy):

    def _get_grasp_transform(self, env_idx, ee_transform):

        pcd = np.asarray(self._points[env_idx].points)
        min_y = np.min(pcd[:,1])
        max_y = np.max(pcd[:,1])
        left_point_range = min_y + (max_y - min_y) * 0.1
        left_points = pcd[pcd[:,0] < left_point_range]

        left_points = left_points[left_points[:,2].argsort()]
        grasping_location = left_points[len(left_points)//2]

        # align the z-axis of the gripper with the normal of the point
        gripper_transform = transform_to_RigidTransform(ee_transform)

        # rotate the ee_transform with the rotation matrix
        grasp_transform = RigidTransform(
            translation=grasping_location,
            rotation = gripper_transform.rotation,
        )

        return RigidTransform_to_transform(grasp_transform)

    
    def _get_pre_grasp_transform(self, env_idx, grasp_transform):
        
        grasp_rigid_transform = transform_to_RigidTransform(grasp_transform)

        pre_grasp_transform = RigidTransform(
            translation=grasp_rigid_transform.translation - 0.1 * np.array([0,1,0]),
            rotation=grasp_rigid_transform.rotation,
        )

        return RigidTransform_to_transform(pre_grasp_transform)


class PokePolicy(Policy):
    def __init__(self, franka, franka_name, block, block_name, points, point_normals, n_envs, *args, **kwargs):
        super().__init__(*args, **kwargs)

        """
        :param franka: Franka object
        :param franka_name: name of the franka
        :param block: block object
        :param block_name: name of the block
        :param points: list of points to grasp [o3d.PointCloud]*n
        :param point_normals: list of point normals [np.array([x,y,z])]*n
        Note: n should be equal to the n_envs from scene
        # TODO MS: add an assert for the above check in the generate_data.py
        """

        self._franka = franka
        self._franka_name = franka_name
        self._time_horizon = 710
        self._block = block
        self._block_name = block_name
        self._points = points
        self._point_normals = point_normals
        self._n_envs = n_envs
        self.reset()

    def reset(self):
        self._pre_poke_transforms = []
        self._poke_transforms = []
        self._init_ee_transforms = []
        self._ee_waypoint_policies = []
        self._post_poke_transforms = []

    @abstractmethod
    def _get_poke_transform(self, env_idx):
        raise NotImplementedError

    @abstractmethod
    def _get_pre_poke_transform(self, env_idx, poke_transform):
        raise NotImplementedError  

    @abstractmethod
    def _get_post_poke_transform(self, env_idx, poke_transform):
        raise NotImplementedError        

    def __call__(self, scene, env_idx, t_step, t_sim):
        ee_transform = self._franka.get_ee_transform(env_idx, self._franka_name)

        if t_step == 0:
            self._init_joints = self._franka.get_joints(env_idx, self._franka_name)
            self._init_rbs = self._franka.get_rb_states(env_idx, self._franka_name)

        if t_step == 20:
            ee_transform = self._franka.get_ee_transform(env_idx, self._franka_name)
            self._init_ee_transforms.append(ee_transform)

            poke_transform = self._get_poke_transform(env_idx, ee_transform)
            self._poke_transforms.append(poke_transform)
            self._pre_poke_transforms.append(self._get_pre_poke_transform(env_idx, poke_transform))
            self._post_poke_transforms.append(self._get_post_poke_transform(env_idx, poke_transform))

            self._franka.set_ee_transform(env_idx, self._franka_name, self._pre_poke_transforms[env_idx])

        if t_step == 100:
            self._franka.set_ee_transform(env_idx, self._franka_name, self._poke_transforms[env_idx])

        if t_step == 150:
            self._franka.close_grippers(env_idx, self._franka_name)
        
        if t_step == 250:
            self._franka.set_ee_transform(env_idx, self._franka_name, self._post_poke_transforms[env_idx])

        if t_step == 350:
            self._franka.set_ee_transform(env_idx, self._franka_name, self._poke_transforms[env_idx])

        if t_step == 500:
            self._franka.open_grippers(env_idx, self._franka_name)


        if t_step == 550:
            self._franka.set_ee_transform(env_idx, self._franka_name, self._pre_poke_transforms[env_idx])

        if t_step == 600:
            self._franka.set_ee_transform(env_idx, self._franka_name, self._init_ee_transforms[env_idx])

        if t_step == 700:
            self._franka.set_joints(env_idx, self._franka_name, self._init_joints)
            self._franka.set_rb_states(env_idx, self._franka_name, self._init_rbs)


class PokePointXPolicy(PokePolicy):

    def _get_poke_transform(self, env_idx, ee_transform):
        # pc = np.asarray(self._points[env_idx].points)
        # max_z_idx = np.argmax(pc[:,2])
        random_index = np.random.randint(0, 50)
        random_index = 0

        # import ipdb; ipdb.set_trace()
        pokeing_location = np.asarray(self._points[env_idx].points)[random_index]

        gripper_transform = transform_to_RigidTransform(ee_transform)

        poke_transform = RigidTransform(
            translation=pokeing_location,
            rotation = gripper_transform.rotation,
        )


        return RigidTransform_to_transform(poke_transform)

    
    def _get_pre_poke_transform(self, env_idx, poke_transform):
        
        poke_rigid_transform = transform_to_RigidTransform(poke_transform)

        pre_poke_transform = RigidTransform(
            translation=poke_rigid_transform.translation - 0.15 * np.array([1, 0, 0]),
            rotation=poke_rigid_transform.rotation,
        )

        return RigidTransform_to_transform(pre_poke_transform)
    
    def _get_post_poke_transform(self, env_idx, poke_transform):
        
        poke_rigid_transform = transform_to_RigidTransform(poke_transform)

        pre_poke_transform = RigidTransform(
            translation=poke_rigid_transform.translation + 0.15 * np.array([1, 0, 0]),
            rotation=poke_rigid_transform.rotation,
        )

        return RigidTransform_to_transform(pre_poke_transform)