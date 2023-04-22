import torch 
import torchvision
import numpy as np 
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from datetime import datetime
import os
import csv
from labels_dict import *
import open3d
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
                                data_dict,
                                 [int(data_split*len(data_dict)), len(data_dict)-int(data_split*len(data_dict))], 
                                 generator=torch.Generator().manual_seed(42))
    return train_data, test_data

def load_data_without_split(filename):
    data_dict = torch.load(f"{os.getcwd()}/training_data/{filename}.pt")
    return data_dict

def make_data_row(episode_num, action_vector, 
                    init_pose, point_cloud, point_cloud_downsampled, final_pose, data_dir, env_idx, obj_type="std_cube"):
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
    if not os.path.exists(data_dir+'/point_cloud_downsampled'):
        os.mkdir(data_dir+'/point_cloud_downsampled')

    point_cloud_path = f"{data_dir}/point_cloud_downsampled/{episode_num}_{env_idx}_init_image_{timestamp}.ply"
    open3d.io.write_point_cloud(point_cloud_path, point_cloud_downsampled)
    row.append(point_cloud_path)

    point_cloud_path = f"{data_dir}/point_clouds/{episode_num}_{env_idx}_init_image_{timestamp}.ply"
    open3d.io.write_point_cloud(point_cloud_path, point_cloud)
    row.append(point_cloud_path)
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

def process_row(row, debug=False):
    '''
        row: list of strings
        return obj, action, init_pose, final_pose, init_img, final_img
    '''
    obj = row[2]
    action = row[3]
    action_str = ACTION_DICT[action]
    label_str = f"{action_str}+{obj}"
    label = LABELS_DICT[label_str]
    action_vector = []
    for i in action:
        action_vector.append(int(i))
    action_vector = torch.tensor(action_vector)
    init_pose = []
    for i in range(7):
        init_pose.append(float(row[4+i]))
    init_pose = torch.tensor(init_pose)

    final_pose = []
    for i in range(7):
        final_pose.append(float(row[11+i]))
    final_pose = torch.tensor(final_pose)
    path = f"data/REAL_DATA"
    init_img = torchvision.io.read_image(path + row[18])

    final_img = torchvision.io.read_image(path + row[19])

    return label, action_vector, init_pose, final_pose, init_img, final_img



def make_data_dict(list_dirs, save_filename ,debug=False):
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
                label, action_vector, init_pose, final_pose, init_img, final_img = process_row(row,debug)
                data_dict.append({
                                "label": label,
                                "action_vector": action_vector,
                                "init_pose":init_pose,
                                "final_pose":final_pose,
                                "init_img":init_img,
                                "final_img":final_img})

    save_location = f"{os.getcwd()}/training_data/{save_filename}.pt"
    print(f"Saving data to {save_location}")
    torch.save(data_dict, save_location)
    return data_dict

def make_normalized_data_dict(list_dirs, save_filename ,debug=False):
    if(debug):
        import ipdb; ipdb.set_trace()
    transform = T.Resize(size = (320, 240))
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
                label, action_vector, init_pose, final_pose, init_img, final_img = process_row(row,debug)
                init_img = transform(init_img)
                data_dict.append({
                                "label": label,
                                "action_vector": action_vector,
                                "init_pose":init_pose,
                                "final_pose":final_pose,
                                "init_img":init_img,
                                "final_img":final_img, 
                                "delta_gt": final_pose - init_pose})
        
    

    save_location = f"{os.getcwd()}/training_data/{save_filename}.pt"
    print(f"Saving data to {save_location}")
    torch.save(data_dict, save_location)
    return data_dict

def load_normalized_data_dict(filename):
    save_location = f"{os.getcwd()}/training_data/{filename}.pt"
    print(f"Loading data from {save_location}")
    data_dict = torch.load(save_location)

    delta_gt = torch.tensor([])
    for i in range(len(data_dict)):
        delta_gt = torch.cat((delta_gt, data_dict[i]["delta_gt"]), dim=0)
    mean = torch.mean(delta_gt, dim=0)
    std = torch.std(delta_gt, dim=0)

    for i in range(len(data_dict)):
        data_dict[i]["delta_gt"] = (data_dict[i]["delta_gt"] - mean)/std

    return data_dict, mean, std



def load_normalized_coarse_data_dict(filename):
    save_location = f"{os.getcwd()}/training_data/{filename}.pt"
    print(f"Loading data from {save_location}")
    data_dict = torch.load(save_location)

    delta_gt = torch.tensor([])
    init_poses = torch.tensor([])
    for i in range(len(data_dict)):
        init_poses = torch.cat((init_poses, data_dict[i]["init_pose"]), dim=0)
        delta_gt = torch.cat((delta_gt, data_dict[i]["delta_gt"]), dim=0)
    mean_delta_gt = torch.mean(delta_gt, dim=0)
    std_delta_gt = torch.std(delta_gt, dim=0)

    mean_init_pose = torch.mean(init_poses, dim=0)
    std_init_pose = torch.std(init_poses, dim=0)


    for i in range(len(data_dict)):
        data_dict[i]["init_pose"] = (data_dict[i]["init_pose"] - mean_init_pose)/std_init_pose
        data_dict[i]["delta_gt"] = (data_dict[i]["delta_gt"] - mean_delta_gt)/std_delta_gt

    return data_dict, mean_init_pose, std_init_pose, mean_delta_gt, std_delta_gt

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
    delta_rot = torch.norm(torch.bmm(rot1, rot2.transpose(1, 2)) - torch.eye(3).cuda(), p=2, dim=(1, 2))
    return delta_t + delta_rot