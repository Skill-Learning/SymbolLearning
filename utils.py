import torch 
import torchvision
import numpy as np 
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from datetime import datetime

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

def make_data_row(episode_num, action_vector, 
                    init_pose, init_img, final_pose, final_img, obj_type="std_cube"):
    '''
        make a row of data for training
    '''
    row = []
    timestamp = datetime.now().strftime("%d/%m/%Y_%H:%M:%S")
    row.append(timestamp)
    row.append(episode_num)
    row.append(obj_type)
    action_str = ""
    for i in range(len(action_vector)):
        action_str += str(action_vector[i])
    row.append(action_str)
    assert len(init_pose) == 7
    assert len(final_pose) == 7
    row.append(init_pose)
    row.append(final_pose)
    init_img_path = f"{episode_num}_init_image_{timestamp}.png"
    torchvision.utils.save_image(init_img, init_img_path)
    row.append(init_img_path)
    final_img_path = f"{episode_num}_final_image_{timestamp}.png"
    torchvision.utils.save_image(final_img, final_img_path)
    row.append(final_img_path)
    return row

