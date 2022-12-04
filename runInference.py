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
from model.poseNet import PoseNet

dir_list = [
            '20221203',
            ]
    

def test(net, pose_net, val_data, loss_fn, cfg, csv_writer):
    batch_size = cfg['fine_training']['batch_size']['val']
    device = cfg['device']
    _len_val = len(val_data)
    val_data = DataLoader(val_data, batch_size=batch_size, shuffle=True, drop_last=True)
    pose_net.eval()
    net.eval()
    with torch.no_grad():
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
            for i in range(predicted_pose.shape[0]):
                row = make_inference_row(action_vector[i], init_pose[i], final_pose[i], predicted_pose[i])
                csv_writer.writerow(row)




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
        make_data_dict(dir_list, cfg['fine_training']['save_filename'], cfg['debug'])
        if cfg['verbose']:
            print("Data preprocessing done !")
    
    print("Loading data...")
    train_data, test_data = load_data_dict(cfg['fine_training']['save_filename'], cfg['data']['train_split'])
    if cfg['verbose']:
        print("Data loaded !")
    
    net = Encoder(cfg).to(device)
    # load the weights from the encoder
    net.load_state_dict(torch.load(cfg['fine_training']['encoder_weights']))
    net.eval()

    pose_net = PoseNet(cfg).to(device)
    pose_net.load_state_dict(torch.load(cfg['fine_training']['pose_weights']))
    loss_fn = torch.nn.MSELoss()
    optimizer = optim.Adam(pose_net.parameters(), lr=cfg['fine_training']['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['fine_training']['epochs'])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    inference_file_path = f"{cfg['fine_training']['inference_dir']}/inference_{timestamp}.csv"
    with open(inference_file_path, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['action_vector', 'init_pose', 'final_pose', 'predicted_pose'])
        test(net, pose_net, test_data, loss_fn, cfg, csv_writer)

