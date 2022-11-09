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



def train(net, train_data, loss_fn, optimizer, scheduler, cfg):
    batch_size = cfg['coarse_training']['batch_size']['train']
    epochs = cfg['coarse_training']['epochs']
    device = cfg['device']
    lr = cfg['coarse_training']['lr']

    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)


    for epoch in range(epochs):
        net.train()
        with tqdm(total=batch_size, desc=f'Epoch {epoch + 1}/{epochs}', unit='epoch') as pbar:
            for i, data in enumerate(train_data):
                initial_obs, action_vec, final_obs = data
                initial_obs = initial_obs.to(device=device, dtype=torch.float32)
                action_vec = action_vec.to(device=device, dtype=torch.float32)
                final_obs = final_obs.to(device=device, dtype=torch.float32)

                optimizer.zero_grad()
                output = net(initial_obs, action_vec)
                loss = loss_fn(output, final_obs)
                loss.backward()
                optimizer.step()

                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(initial_obs.shape[0])
        
        print(f"Epoch {epoch + 1}/{epochs} Loss: {loss.item()}")
        scheduler.step()
        torch.save(net.state_dict(), f'checkpoints/{epoch}.pth')
        print(f"Checkpoint {epoch} saved !")