import argparse 
import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from autolab_core import YamlConfig
import tqdm 

from model.symbol_learning_net import SymbolLearningNet

# TODO: Dataoader Class and Logger class in utils.py
# from utils import logger


def train(net, train_data, cfg):
    batch_size = cfg['batch_size']
    epochs = cfg['epochs']
    lr = cfg['lr']
    device = cfg['device']

    optimizer = optim.Adam(net.parameters(), lr=lr)
    # TODO: Verify the loss function for point clouds 
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        net.train()
        with tqdm(total=batch_size, desc=f'Epoch {epoch + 1}/{args.epochs}', unit='img') as pbar:
            for i, data in enumerate(train_data):
                inital_obs, action_vec, final_obs = data
                inital_obs = inital_obs.to(device=device, dtype=torch.float32)
                action_vec = action_vec.to(device=device, dtype=torch.float32)
                final_obs = final_obs.to(device=device, dtype=torch.float32)

                optimizer.zero_grad()
                output = net(inital_obs, action_vec)
                loss = loss_fn(output, final_obs)
                loss.backward()
                optimizer.step()

                logger.log_scalar('loss', loss.item(), epoch * len(train_data) + i)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(inital_obs.shape[0])
        
        print(f"Epoch {epoch + 1}/{args.epochs} Loss: {loss.item()}")
        logger.log_scalar('loss', loss.item(), epoch * len(train_data) + i)
        torch.save(net.state_dict(), f'checkpoints/{epoch}.pth')
        print(f"Checkpoint {epoch} saved !")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()
    cfg = YamlConfig(args.config)

    # Pass this  to trainer with the cfg['training]
    train_data = DataLoader(cfg['training'])
    net = SymbolLearningNet(cfg['model'])
    train(net, cfg['training'])

