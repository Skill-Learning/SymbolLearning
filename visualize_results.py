import numpy as np
import torch
from model.affordanceNet import AffordanceNet
import pytorch3d
import open3d as o3d
from pytorch3d.renderer import (
    AlphaCompositor,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
)
import imageio
from autolab_core import YamlConfig
import argparse 
from pytorch3d.vis.plotly_vis import plot_scene


def get_points_renderer(
    image_size=256, device=None, radius=0.01, background_color=(1, 1, 1)
):
    """
    Returns a Pytorch3D renderer for point clouds.

    Args:
        image_size (int): The rendered image size.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
        radius (float): The radius of the rendered point in NDC.
        background_color (tuple): The background color of the rendered image.
    
    Returns:
        PointsRenderer.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(image_size=image_size, radius=radius,)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer

def visualize(model,point_cloud, path='./results.gif', device='cuda:0'):

    '''
    Args:
        point_cloud: (N,3) tensor
        action_pred: (1,2) tensor
        quality_pred: (1,1) tensor
    '''
    point_cloud=point_cloud.float()@torch.tensor([
                        [1,0,0],
                        [0,0,-1],
                        [0,1,0]
        
    ]).float().to(device)
    image_size=1024
    background_color=(1, 1, 1)
    colors = [np.array([0.0,0.0,1.0]), np.array([0.0,1.0,0.0])]

     # Construct various camera viewpoints
    dist = 5
    elev = -30
    # azim = [180 - 12*i for i in range(30)]
    azim = torch.linspace(0, 360, 30)
    R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist=dist, elev=elev, azim=azim, device=device)
    # import ipdb; ipdb.set_trace()
    c = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)

    visualize_colors=[]
    # import ipdb; ipdb.set_trace()
    labels = []
    action_pred_list = []
    point_cloud=point_cloud.unsqueeze(0).float().to(device)
    quality=[]
    for point in point_cloud[0]:
        quality_pred, action_pred=model(point_cloud, point.unsqueeze(0).unsqueeze(0).float())
        action_pred=action_pred
        # import ipdb; ipdb.set_trace()
        action_pred_list.append(action_pred)
        labels.append(action_pred.argmax().item())
        point_color=colors[action_pred.argmax().item()]*(1-quality_pred.item())
        visualize_colors.append(point_color)
    

    visualize_colors = torch.tensor(np.array(visualize_colors)).float().to(point_cloud.device)
    visualize_colors=visualize_colors.unsqueeze(0)
    # colors = torch.tensor(colors).to(sample_verts.device)
    # Colorize points based on segmentation labels
    

    # sample_colors = sample_colors.repeat(num_points,1,1).to(torch.float)


    point_cloud_ob = pytorch3d.structures.Pointclouds(points=point_cloud, features=visualize_colors).to(device)

    renderer = get_points_renderer(image_size=image_size, background_color=background_color, device=device)
    # import ipdb; ipdb.set_trace()
    rend = renderer(point_cloud_ob.extend(30), cameras=c).cpu().numpy() # (30, 256, 256, 3)
    # convert to uint8

    rend = (rend*255).astype('uint8')
    fig=plot_scene({
        "figure": {
            "Pointcloud": point_cloud_ob,
            "Camera": c,
        }
        })
    fig.show()
    imageio.mimsave(path, rend, fps=5)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    args = parser.parse_args()
    cfg = YamlConfig(args.config)

    device='cuda:0'
    model= AffordanceNet(cfg).to(device)
    model.load_state_dict(torch.load('/home/manpreet-pc/sim_envs/isaacgym-utils/l4m_sim/SymbolLearning/checkpoints/point_clouds_good_20230426_012241/149.pth',map_location=device))
    data=torch.load('/home/manpreet-pc/sim_envs/isaacgym-utils/l4m_sim/SymbolLearning/training_data/point_clouds_good.pt',map_location=device)
    label_list = []
    for i in range(len(data)):
        label_list.append(data[i]['label'])

    # import ipdb; ipdb.set_trace()
    point_cloud=data[0]['point_cloud']
    visualize(model,point_cloud, path='./results_plus90.gif', device=device)