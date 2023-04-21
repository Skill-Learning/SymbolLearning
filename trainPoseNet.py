import argparse 
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from autolab_core import YamlConfig
from tqdm import tqdm
from torch.utils.data import DataLoader
from model.encoder import Encoder
from loss import SupervisedContrastiveLoss, PoseLoss
from tensorboardX import SummaryWriter
from model.poseNet import PoseNet

dir_list = [
            'REAL_DATA',
            ]
    

def val(net, pose_net, val_data, loss_fn, cfg, writer, dir_path, step_number, mean, std):
    batch_size = cfg['fine_training']['batch_size']['val']
    device = cfg['device']
    _len_val = len(val_data)
    val_data = DataLoader(val_data, batch_size=batch_size, shuffle=True, drop_last=True)
    pose_net.eval()
    net.eval()
    with torch.no_grad():
        with tqdm(total=_len_val, desc=f'Validation', unit='epoch') as pbar:
            for i, data in enumerate(val_data):
                # [B, C, H, W]
                init_img = data['init_img'].float().to(device)
                # final_img = data['final_img'].to(device)
                # [B, 7]
                init_pose = data['init_pose'].to(device)
                final_pose = data['final_pose'].to(device)

                # [B, 6]
                action_vector = data['action_vector'].to(device)

                embeddings = net(init_img, action_vector, init_pose)
                predicted_pose = pose_net(embeddings)
                delta_gt = data['delta_gt'].to(device)
                delta_gt = delta_gt[:,:3]
                delta_pred = predicted_pose[:,:3]
                loss = loss_fn(delta_pred, delta_gt)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(init_img.shape[0])
    if(cfg['logging']['log_enable']):
        writer.add_scalar('Loss/val', loss.item(), step_number)
        writer.close()


def train(net, pose_net, train_data, loss_fn, optimizer, scheduler, cfg, writer, dir_path, test_data, mean, std):

    if(cfg['debug']):
        print("DEBUG MODE")
        import ipdb; ipdb.set_trace()

    batch_size = cfg['fine_training']['batch_size']['train']
    epochs = cfg['fine_training']['epochs']
    device = cfg['device']
    lr = cfg['fine_training']['lr']
    _len_train = len(train_data)
    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    step_number = 0 
    for epoch in range(epochs):
        
        pose_net.train()
        net.eval()
        with tqdm(total=_len_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='epoch') as pbar:
            for i, data in enumerate(train_data):
                # [B, C, H, W]
                init_img = data['init_img'].float().to(device)
                # final_img = data['final_img'].to(device)
                # [B, 7]
                init_pose = data['init_pose'].to(device)
                final_pose = data['final_pose'].to(device)

                # [B, 6]
                action_vector = data['action_vector'].to(device)

                embeddings = net(init_img, action_vector, init_pose)
                predicted_pose = pose_net(embeddings)
                optimizer.zero_grad()
                # delta_pred = predicted_pose - init_pose
                # delta_pred = (delta_pred - mean)/std
                # delta_pred = delta_pred[:,:3]
                delta_gt = data['delta_gt'].to(device)
                delta_gt = delta_gt[:,:3]
                delta_pred = predicted_pose[:,:3]
                loss = loss_fn(delta_pred, delta_gt)
                loss.backward()
                optimizer.step()
                step_number += 1
                if(cfg['logging']['log_enable']):
                    writer.add_scalar('Loss/train', loss.item(), step_number)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(init_img.shape[0])
        
        print(f"Epoch {epoch + 1}/{epochs} Loss: {loss.item()}")
        scheduler.step()
        # TODO: Add a validation step
        torch.save(pose_net.state_dict(), f'{dir_path}/{epoch}.pth')
        print(f"Checkpoint {epoch} saved !")
        print("Validation")
        val(net, pose_net, test_data, loss_fn, cfg, writer, dir_path, step_number, mean, std)
    if(cfg['logging']['log_enable']):
        writer.close()


if __name__ == "__main__":
    
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='config/config.yaml')
    args = parser.parse_args()
    cfg = YamlConfig(args.cfg)

    if cfg['debug']:
        import ipdb; ipdb.set_trace()

    device = torch.device(cfg['device']) 

    if not cfg['fine_training']['use_saved_data']:
        if cfg['verbose']:
            print("Preprocessing data...")
        make_normalized_data_dict(dir_list, cfg['fine_training']['save_filename'], cfg['debug'])
        if cfg['verbose']:
            print("Data preprocessing done !")
    
    print("Loading data...")
    # import ipdb; ipdb.set_trace()
    data_dict, init_mean, init_std, delta_gt_mean, delta_gt_std = load_normalized_coarse_data_dict(cfg['fine_training']['save_filename'])
    data_split = cfg['data']['train_split']
    train_data, test_data = torch.utils.data.random_split(
                            data_dict,
                                [int(data_split*len(data_dict)), len(data_dict)-int(data_split*len(data_dict))], 
                                generator=torch.Generator().manual_seed(42))
    if cfg['verbose']:
        print("Data loaded !")
    
    net = Encoder(cfg).to(device)
    # load the weights from the encoder
    net.load_state_dict(torch.load(cfg['fine_training']['encoder_weights']))
    net.eval()
    pose_net = PoseNet(cfg).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(pose_net.parameters(), lr=cfg['fine_training']['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['fine_training']['epochs'])

    writer = None
    if(cfg['logging']['log_enable']):
        writer = SummaryWriter(comment=f"_fine_LR_{cfg['fine_training']['lr']}_BS_{cfg['fine_training']['batch_size']['train']}_{cfg['logging']['comment']}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_path = f"checkpoints/fine_training/{cfg['fine_training']['save_filename']}_{timestamp}"
    os.makedirs(dir_path, exist_ok=True)
    train(net, pose_net, train_data, loss_fn, optimizer, scheduler, cfg, writer, dir_path, test_data, delta_gt_mean, delta_gt_std)
