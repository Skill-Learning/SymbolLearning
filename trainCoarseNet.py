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
from loss import SupervisedContrastiveLoss
from tensorboardX import SummaryWriter
from tsne_torch import TorchTSNE as TSNE

dir_list = [
            # '20221203',
            '20221204_REAL',
            ]

def val(net, val_data, loss_fn, cfg, writer, dir_path, init_mean, init_std, step_number):
    batch_size = cfg['coarse_training']['batch_size']['val']
    device = cfg['device']
    _len_val = len(val_data)
    val_data = DataLoader(val_data, batch_size=batch_size, shuffle=True, drop_last=True)
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
                # [B,]
                labels = data['label'].to(device)

                # [B, 6]
                action_vector = data['action_vector'].to(device)

                embeddings = net(init_img, action_vector, init_pose)
                init_pose = init_pose*init_std + init_mean
                loss = loss_fn(embeddings, init_pose, final_pose, labels)

                    # if(i % 500 == 0):
                    #     writer.add_embedding(embeddings, metadata=labels, global_step=i)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(init_img.shape[0])

    if(cfg['logging']['log_enable']):
        writer.add_scalar('Loss/val', loss.item(), step_number)
        writer.close()


def train(net, train_data, loss_fn, optimizer, scheduler, cfg, writer, dir_path, test_data, init_mean, init_std):

    if(cfg['debug']):
        print("DEBUG MODE")
        import ipdb; ipdb.set_trace()

    batch_size = cfg['coarse_training']['batch_size']['train']
    epochs = cfg['coarse_training']['epochs']
    device = cfg['device']
    lr = cfg['coarse_training']['lr']
    _len_train = len(train_data)
    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    step_number = 0
    for epoch in range(epochs):
        net.train()
        with tqdm(total=_len_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='epoch') as pbar:
            for i, data in enumerate(train_data):
                # [B, C, H, W]
                init_img = data['init_img'].float().to(device)
                # final_img = data['final_img'].to(device)
                # [B, 7]
                init_pose = data['init_pose'].to(device)
                final_pose = data['final_pose'].to(device)
                # [B,]
                labels = data['label'].to(device)

                # [B, 6]
                action_vector = data['action_vector'].to(device)

                embeddings = net(init_img, action_vector, init_pose)

                optimizer.zero_grad()
                init_pose = init_pose*init_std + init_mean
                loss = loss_fn(embeddings, init_pose, final_pose, labels)
                loss.backward()
                optimizer.step()
                step_number += 1
                if(cfg['logging']['log_enable']):
                    writer.add_scalar('Loss/train', loss.item(), step_number)
                    if(step_number % 50 == 0):
                        writer.add_embedding(embeddings, metadata=labels, global_step = step_number)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(init_img.shape[0])
        # tsne 
        print(f"Epoch {epoch + 1}/{epochs} Loss: {loss.item()}")
        scheduler.step()
        # TODO: Add a validation step
        torch.save(net.state_dict(), f'{dir_path}/{epoch}.pth')
        print(f"Checkpoint {epoch} saved !")
        print("Validation")
        val(net, test_data, loss_fn, cfg, writer, dir_path, init_mean, init_std, step_number)
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
        make_normalized_data_dict(dir_list, cfg['data']['filename'], cfg['debug'])
        if cfg['verbose']:
            print("Data preprocessing done !")
    
    print("Loading data...")

    data_dict, init_mean, init_std, delta_gt_mean, delta_gt_std = load_normalized_coarse_data_dict(cfg['data']['filename'])
    data_split = cfg['data']['train_split']
    train_data, test_data = torch.utils.data.random_split(
                                data_dict,
                                 [int(data_split*len(data_dict)), len(data_dict)-int(data_split*len(data_dict))], 
                                 generator=torch.Generator().manual_seed(42))

    if cfg['verbose']:
        print("Data loaded !")
    
    net = Encoder(cfg).to(device)
    loss_fn = SupervisedContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr=cfg['coarse_training']['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['coarse_training']['epochs'])

    writer = None
    if(cfg['logging']['log_enable']):
        writer = SummaryWriter(comment=f"_LR_{cfg['coarse_training']['lr']}_BS_{cfg['coarse_training']['batch_size']['train']}_{cfg['logging']['comment']}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_path = f"checkpoints/{cfg['data']['filename']}_{timestamp}"
    os.makedirs(dir_path, exist_ok=True)
    train(net, train_data, loss_fn, optimizer, scheduler, cfg, writer, dir_path, test_data, init_mean, init_std)

    # MS: 
    # TODO: add linear warmup
    # TODO: maybe use LARS optimizer: https://github.com/kakaobrain/torchlars
