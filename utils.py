import torch 
import torchvision
import numpy as np 
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from datetime import datetime
import os
import csv
from labels_dict import *
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

def make_data_row(episode_num, action_vector, 
                    init_pose, init_img, final_pose, final_img, data_dir, env_idx, obj_type="std_cube"):
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
    init_img_path = f"{data_dir}/images/{episode_num}_{env_idx}_init_image_{timestamp}.png"
    torchvision.utils.save_image(init_img, init_img_path)
    row.append(init_img_path)
    final_img_path = f"{data_dir}/images/{episode_num}_{env_idx}_final_image_{timestamp}.png"
    torchvision.utils.save_image(final_img, final_img_path)
    row.append(final_img_path)
    return row

def process_row(row):
    '''
        row: list of strings
        return obj, action, init_pose, final_pose, init_img, final_img
    '''
    obj = row[2]
    action = row[3]
    label_str = f"{obj}+{action}"
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

    init_img = torchvision.io.read_image(row[18])

    final_img = torchvision.io.read_image(row[19])

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
            for row in reader:
                label, action_vector, init_pose, final_pose, init_img, final_img = process_row(row)
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


