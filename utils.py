import torch 
import torchvision
import numpy as np 
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from datetime import datetime
import os
import csv
from labels_dict import *
import open3d as o3d
import torchvision.transforms as T

def init_weights_and_freeze(model_tgt, model_src, freeze_layers = []):
    '''
        copy weights to existing layers in model_tgt from model_src
    '''
    for w_tgt, w_src in zip(model_tgt.named_parameters(), model_src.named_parameters()):
        if 'layer' in w_tgt[0]:
            w_tgt[1].data = w_src[1].data
            w_tgt[1].requires_grad = True

        # check for layer freeze 
        for unfreeze_layer in freeze_layers:
            if unfreeze_layer in w_tgt[0]:
                w_tgt[1].requires_grad = False

def train_val_split(dataset, val_split=0.2):
    '''
        split dataset into train and validation set
    '''
    dataset_size = len(dataset)
    train_idx, val_idx = train_test_split(list(range(dataset_size)), test_size=val_split)
    dataset = {}
    dataset['train'] = Subset(dataset, train_idx)
    dataset['val'] = Subset(dataset, val_idx)
    return dataset

def load_data_dict(filename, data_split=0.75):
    data_dict = torch.load(f"{os.getcwd()}/training_data/{filename}.pt")
    
    train_data, test_data = torch.utils.data.random_split(
                                data_dict,cd_rest=o3d.geometry.PointCloud()
                                 [int(data_split*len(data_dict)), len(data_dict)-int(data_split*len(data_dict))], 
                                 generator=torch.Generator().manual_seed(42))
    return train_data, test_data

def load_data_without_split(filename):
    data_dict = torch.load(f"{os.getcwd()}/training_data/{filename}.pt")
    return data_dict

def make_data_row(episode_num, action_vector, 
                    init_pose, point_cloud, final_pose, data_dir, env_idx, grasping_point,obj_type="std_cube"):
    '''
        make a row of data for training
    '''
    row = []
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    row.append(timestamp)
    row.append(episode_num)
    row.append(obj_type)
    action_str = ""
    action_vector = action_vector.tolist()
    for i in range(len(action_vector)):
        action_str += str(action_vector[i])
    row.append(action_str)
    assert len(init_pose[0]) == 7
    assert len(final_pose[0]) == 7
    for i in range(7):
        row.append(init_pose[0][i].item())
    for i in range(7):
        row.append(final_pose[0][i].item())
    
    if not os.path.exists(data_dir+'/point_clouds'):
        os.mkdir(data_dir+'/point_clouds')

    pcd_rest=o3d.geometry.PointCloud()
    pcd_rest.points = o3d.utility.Vector3dVector(point_cloud)
    point_cloud_path = f"{data_dir}/point_clouds/{episode_num}_{env_idx}_{timestamp}.pcd"
    o3d.io.write_point_cloud(point_cloud_path, pcd_rest)
    row.append(point_cloud_path)
    for i in range(3):
        row.append(grasping_point[0][i].item())

    return row

def make_inference_row(action_vector, 
                    init_pose, gt_pose, predicted_pose):
    '''
        make a row of data for training
    '''
    row = []
    action_str = ""
    action_vector = action_vector.tolist()
    for i in range(len(action_vector)):
        action_str += str(action_vector[i])
    row.append(action_str)
    for i in range(7):
        row.append(init_pose[i].item())
    for i in range(7):
        row.append(gt_pose[i].item())
    for i in range(7):
        row.append(predicted_pose[i].item())
    return row

#  header = ['timestamp', 'episode', 'object_type', 'action_vector',
#             'initial_pose_position_x', 'initial_pose_position_y', 'initial_pose_position_z',
#             'initial_pose_orientation_x', 'initial_pose_orientation_y', 'initial_pose_orientation_z', 'initial_pose_orientation_w',
#             'final_pose_position_x', 'final_pose_position_y', 'final_pose_position_z',
#             'final_pose_orientation_x', 'final_pose_orientation_y', 'final_pose_orientation_z', 'final_pose_orientation_w',
#             'entire_pc_path','grasping_pointx','grasping_pointy','grasping_pointz']

def process_row(row, debug=False):
    '''
        row: list of strings
        return obj, action, init_pose, final_pose, point_cloud
    '''
    if debug:
        import ipdb; ipdb.set_trace()

    obj = row[2]
    # action = row[3]
    # action_str = ACTION_DICT[action]
    # label_str = f"{action_str}+{obj}"

    init_pose = []
    for i in range(7):
        init_pose.append(float(row[4+i]))
    init_pose = torch.tensor(init_pose)

    final_pose = []
    for i in range(7):
        final_pose.append(float(row[11+i]))
    final_pose = torch.tensor(final_pose)

    grasping_point=[]
    for i in range(3):
        grasping_point.append(float(row[-3+i]))
    grasping_point=torch.tensor(grasping_point)

    pc_np = np.asarray(o3d.io.read_point_cloud(row[-4]).points)
    pc_ten=torch.tensor(pc_np)

    #label calculation
    ''' 
    [0,1]: Top grasp
    [1,0]: Wrap grasp
    '''
    max_z=0.8665
    label=torch.randint(1,3,(len(pc_ten),2))
    label_bol=torch.where(pc_ten[:,2]>0.98*max_z,True,False)
    label[label_bol]=torch.tensor([0,1])
    label[~label_bol]=torch.tensor([1,0])

    return label, init_pose, final_pose, pc_ten, grasping_point



def make_data_dict(list_dirs, save_filename ,debug=False, save_data=True):
    if(debug):
        import ipdb; ipdb.set_trace()
    data_dict = []
    for dir in list_dirs:
        if(debug):
            print(f"Processing {dir}")
        dir_path = os.path.join(os.getcwd(), f"data/{dir}")
        csv_path = f"{dir_path}/data.csv"
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                label, init_pose, final_pose, pc, grasping_point = process_row(row,debug)
                normalized_pc= normalize_point_cloud(pc, grasping_point)
                ideal_pose = init_pose
                ideal_pose[2] +=0.2
                final_pose = final_pose.unsqueeze(0)
                ideal_pose = ideal_pose.unsqueeze(0)
                pose_dist = pose_dist_metric(final_pose, ideal_pose)
                data_dict.append({
                                "label": label,
                                "pose_dist": pose_dist,
                                "point_cloud":normalized_pc,
                                "grasping_point":grasping_point
                                })
    if save_data:
        if not os.path.exists('./training_data'):
            os.mkdir('./training_data')
        save_location = f"./training_data/{save_filename}.pt"
        print(f"Saving data to {save_location}")
        torch.save(data_dict, save_location)
    return data_dict

def load_point_cloud_dict(filename):
    save_location = f"{os.getcwd()}/training_data/{filename}.pt"
    print(f"Loading data from {save_location}")
    data_dict = torch.load(save_location)

    pose_dist = torch.tensor([])
    for i in range(len(data_dict)):
        pose_dist = torch.cat((pose_dist, data_dict[i]["pose_dist"]), dim=0)
    mean = torch.mean(pose_dist, dim=0)
    std = torch.std(pose_dist, dim=0)

    for i in range(len(data_dict)):
        data_dict[i]["pose_dist"] = (data_dict[i]["pose_dist"] - mean)/std

    return data_dict

def quat2rot(quat):
    '''
    quat: [B, 4]
    
    '''    
    w = quat[:, 3]
    x = quat[:, 0]
    y = quat[:, 1]
    z = quat[:, 2]

    A = torch.stack([
        torch.vstack([w, -z, y, x]),
        torch.vstack([z, w, -x, y]),
        torch.vstack([-y, x, w, z]),
        torch.vstack([-x, -y, -z, w]),
    ], dim=1)
    B = torch.stack([
        torch.vstack([w, -z, y, -x]),
        torch.vstack([z, w, -x, -y]),
        torch.vstack([-y, x, w, -z]),
        torch.vstack([x, y, z, w]),
    ], dim=1)
    A = A.view(-1, 4, 4)
    B = B.view(-1, 4, 4)
    mat = torch.bmm(A, B.transpose(1, 2))
    return mat[:, :3, :3]

def pose_dist_metric(pose1, pose2):
    delta_t = torch.norm(pose1[:, :3] - pose2[:, :3], p=2, dim=1)
    rot1 = quat2rot(pose1[:, 3:])
    rot2 = quat2rot(pose2[:, 3:])
    delta_rot = torch.norm(torch.bmm(rot1, rot2.transpose(1, 2)) - torch.eye(3), p=2, dim=(1, 2))
    return delta_t + delta_rot

def l2_norm(x, y):
    """Calculate l2 norm (distance) of `x` and `y`.
    Args:
        x (numpy.ndarray or cupy): (batch_size, num_point, coord_dim)
        y (numpy.ndarray): (batch_size, num_point, coord_dim)
    Returns (numpy.ndarray): (batch_size, num_point,)
    """
    return ((x - y) ** 2).sum(axis=2)

def farthest_point_sampling(pts, k, initial_idx=None, metrics=l2_norm,
                            skip_initial=False, indices_dtype=np.int32,
                            distances_dtype=np.float32):
    """Batch operation of farthest point sampling
    Code referenced from below link by @Graipher
    https://codereview.stackexchange.com/questions/179561/farthest-point-algorithm-in-python
    Args:
        pts (numpy.ndarray or cupy.ndarray): 2-dim array (num_point, coord_dim)
            or 3-dim array (batch_size, num_point, coord_dim)
            When input is 2-dim array, it is treated as 3-dim array with
            `batch_size=1`.
        k (int): number of points to sample
        initial_idx (int): initial index to start farthest point sampling.
            `None` indicates to sample from random index,
            in this case the returned value is not deterministic.
        metrics (callable): metrics function, indicates how to calc distance.
        skip_initial (bool): If True, initial point is skipped to store as
            farthest point. It stabilizes the function output.
        xp (numpy or cupy):
        indices_dtype (): dtype of output `indices`
        distances_dtype (): dtype of output `distances`
    Returns (tuple): `indices` and `distances`.
        indices (numpy.ndarray or cupy.ndarray): 2-dim array (batch_size, k, )
            indices of sampled farthest points.
            `pts[indices[i, j]]` represents `i-th` batch element of `j-th`
            farthest point.
        distances (numpy.ndarray or cupy.ndarray): 3-dim array
            (batch_size, k, num_point)
    """
    if pts.ndim == 2:
        # insert batch_size axis
        pts = pts[None, ...]
    assert pts.ndim == 3
    # xp = cuda.get_array_module(pts)
    batch_size, num_point, coord_dim = pts.shape
    indices = np.zeros((batch_size, k, ), dtype=indices_dtype)

    # distances[bs, i, j] is distance between i-th farthest point `pts[bs, i]`
    # and j-th input point `pts[bs, j]`.
    distances = np.zeros((batch_size, k, num_point), dtype=distances_dtype)
    if initial_idx is None:
        indices[:, 0] = np.random.randint(len(pts))
    else:
        indices[:, 0] = initial_idx

    batch_indices = np.arange(batch_size)
    farthest_point = pts[batch_indices, indices[:, 0]]
    # minimum distances to the sampled farthest point
    try:
        min_distances = metrics(farthest_point[:, None, :], pts)
    except Exception as e:
        import IPython; IPython.embed()

    if skip_initial:
        # Override 0-th `indices` by the farthest point of `initial_idx`
        indices[:, 0] = np.argmax(min_distances, axis=1)
        farthest_point = pts[batch_indices, indices[:, 0]]
        min_distances = metrics(farthest_point[:, None, :], pts)

    distances[:, 0, :] = min_distances
    for i in range(1, k):
        indices[:, i] = np.argmax(min_distances, axis=1)
        farthest_point = pts[batch_indices, indices[:, i]]
        dist = metrics(farthest_point[:, None, :], pts)
        distances[:, i, :] = dist
        min_distances = np.minimum(min_distances, dist)
    return indices, distances



def normalize_point_cloud(point_cloud, grasping_point = None):
    """
    :param point_cloud: tensor (N, 3)
    :param grasping_point: tensor (3)

    return: tensor (N, 3)
    return: tensor (3)
    """

    if point_cloud.shape[0] == 1:
        return point_cloud, grasping_point
    
    centroid = torch.mean(point_cloud, dim=0, keepdim=True)
    point_cloud = point_cloud - centroid
    furthest_distance = torch.max(torch.sqrt(torch.sum(point_cloud ** 2, dim=-1, keepdim=True)), dim=0, keepdim=True)[0]
    point_cloud = point_cloud / furthest_distance
    if grasping_point is None:
        return point_cloud
    
    grasping_point = (grasping_point - centroid) / furthest_distance

    return point_cloud, grasping_point