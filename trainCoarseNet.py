import argparse 
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from autolab_core import YamlConfig
import tqdm
from torch.utils.data import DataLoader
from model.encoder import Encoder
from loss import SupervisedContrastiveLoss

dir_list = ['20221129']


def train(net, train_data, loss_fn, optimizer, scheduler, cfg):

    if(cfg['debug']):
        print("DEBUG MODE")
        import ipdb; ipdb.set_trace()

    batch_size = cfg['coarse_training']['batch_size']['train']
    epochs = cfg['coarse_training']['epochs']
    device = cfg['device']
    lr = cfg['coarse_training']['lr']

    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(epochs):
        net.train()
        with tqdm(total=batch_size, desc=f'Epoch {epoch + 1}/{epochs}', unit='epoch') as pbar:
            for i, data in enumerate(train_data):
                # [B, C, H, W]
                init_img = data['init_img'].to(device)
                # final_img = data['final_img'].to(device)
                # [B, 7]
                init_pose = data['init_pose'].to(device)
                final_pose = data['final_pose'].to(device)
                # [B,]
                labels = data['labels'].to(device)

                # [B, 6]
                action_vector = data['action_vector'].to(device)

                embeddings = net(init_img, action_vector, init_pose)
                pose_change = final_pose - init_pose

                optimizer.zero_grad()
                output = net()
                loss = loss_fn(embeddings, pose_change, labels)
                loss.backward()
                optimizer.step()

                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(init_img.shape[0])
        
        print(f"Epoch {epoch + 1}/{epochs} Loss: {loss.item()}")
        scheduler.step()
        # TODO: Add a validation step
        torch.save(net.state_dict(), f'checkpoints/{epoch}.pth')
        print(f"Checkpoint {epoch} saved !")


if __name__ == "__main__":
    
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', '-c', type=str, default='cfg/config.yaml')
    args = parser.parse_args()
    cfg = YamlConfig(args.cfg)

    if cfg['debug']:
        import ipdb; ipdb.set_trace()

    device = torch.device(cfg['device'])

    if not cfg['data']['use_saved_data']:
        if cfg['verbose']:
            print("Preprocessing data...")
        make_data_dict(dir_list, cfg['data']['filename'], cfg['debug'])
        if cfg['verbose']:
            print("Data preprocessing done !")
    
    print("Loading data...")
    train_data, test_data = load_data_dict(cfg['data']['filename'], cfg['data']['train_split'])
    if cfg['verbose']:
        print("Data loaded !")
    
    net = Encoder(cfg).to(device)
    loss_fn = SupervisedContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr=cfg['coarse_training']['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['coarse_training']['epochs'])

    train(net, train_data, loss_fn, optimizer, scheduler, cfg)

    # MS: 
    # TODO: add linear warmup
    # TODO: maybe use LARS optimizer: https://github.com/kakaobrain/torchlars
