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
from tensorboardX import SummaryWriter
from model.affordanceNet import AffordanceNet


dir_list = [
    20230424
            ]

def val(net, val_data, quality_loss_fn,action_loss_fn, cfg, writer, step_number):
    batch_size = cfg['anet_training']['batch_size']['val']
    device = cfg['device']
    _len_val = len(val_data)
    val_data = DataLoader(val_data, batch_size=batch_size, shuffle=True, drop_last=True)
    net.eval()
    with torch.no_grad():
        with tqdm(total=_len_val, desc=f'Validation', unit='epoch') as pbar:
            for i, data in enumerate(val_data):

                point_cloud = data['point_cloud'].float().to(device)
                grasping_point=data['grasping_point'].float().to(device)
                action_label = data['label'].float().to(device)
                quality_pred, action_pred = net(point_cloud, grasping_point)
                gt_quality=torch.zeros_like(quality_pred)
                optimizer.zero_grad()
                
                loss_quality = quality_loss_fn(quality_pred, gt_quality)
                loss_action=action_loss_fn(action_pred, action_label)
                loss=loss_quality+loss_action

                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(point_cloud.shape[0])

    if(cfg['logging']['log_enable']):
        writer.add_scalar('Loss/val', loss.item(), step_number)
        writer.close()


def train(net, train_data,quality_loss_fn,action_loss_fn, optimizer, scheduler, cfg, writer, dir_path, test_data):

    if(cfg['debug']):
        print("DEBUG MODE")
        import ipdb; ipdb.set_trace()

    batch_size = cfg['anet_training']['batch_size']['train']
    epochs = cfg['anet_training']['epochs']
    device = cfg['device']
    lr = cfg['anet_training']['lr']
    _len_train = len(train_data)
    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    step_number = 0
    for epoch in range(epochs):
        net.train()
        with tqdm(total=_len_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='epoch') as pbar:
            for i, data in enumerate(train_data):
                optimizer.zero_grad()
                
                point_cloud = data['point_cloud'].float().to(device)
                
                grasping_point=data['grasping_point'].float().to(device)
                action_label = data['label'].float().to(device)
                quality_pred, action_pred = net(point_cloud, grasping_point)
                gt_quality=torch.zeros_like(quality_pred)
                loss_quality = quality_loss_fn(quality_pred, gt_quality)
                loss_action=action_loss_fn(action_pred, action_label)
                loss=loss_quality+loss_action
                loss.backward()

                optimizer.step()
                step_number += 1
                if(cfg['logging']['log_enable']):
                    writer.add_scalar('Loss/train', loss.item(), step_number)
                    
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(point_cloud.shape[0])
        # tsne 
        print(f"Epoch {epoch + 1}/{epochs} Loss: {loss.item()}")
        scheduler.step()
        # TODO: Add a validation step
        torch.save(net.state_dict(), f'{dir_path}/{epoch}.pth')
        print(f"Checkpoint {epoch} saved !")
        print("Validation")
        val(net, test_data, quality_loss_fn,action_loss_fn, cfg, writer, step_number)
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

    if not cfg['data']['use_saved_data']:
        if cfg['verbose']:
            print("Preprocessing data...")
        make_data_dict(dir_list, cfg['data']['filename'], cfg['debug'])
        if cfg['verbose']:
            print("Data preprocessing done !")
    
    print("Loading data...")

    data_dict= load_point_cloud_dict(cfg['data']['filename'])
    data_split = cfg['data']['train_split']
    train_data, test_data = torch.utils.data.random_split(
                                data_dict,
                                 [int(data_split*len(data_dict)), len(data_dict)-int(data_split*len(data_dict))], 
                                 generator=torch.Generator().manual_seed(42))

    if cfg['verbose']:
        print("Data loaded !")

    num_points = cfg['data']['num_points']

    net = AffordanceNet(cfg).to(device)
    quality_loss = nn.MSELoss(reduction='sum')
    action_loss=nn.BCEWithLogitsLoss(reduction='sum')
    optimizer = optim.Adam(net.parameters(), lr=cfg['anet_training']['lr'], betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['anet_training']['epochs'])

    writer = None
    if(cfg['logging']['log_enable']):
        writer = SummaryWriter(comment=f"_LR_{cfg['anet_training']['lr']}_BS_{cfg['anet_training']['batch_size']['train']}_{cfg['logging']['comment']}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_path = f"checkpoints/{cfg['data']['filename']}_{timestamp}"
    os.makedirs(dir_path, exist_ok=True)
    train(net, train_data, quality_loss,action_loss, optimizer, scheduler, cfg, writer, dir_path, test_data)

    # MS: 
    # TODO: add linear warmup
    # TODO: maybe use LARS optimizer: https://github.com/kakaobrain/torchlars
