import torch 
from utils import make_data_dict
import numpy as np


list_dirs = [20230424]
save_filename = f"point_clouds_correctted"

data_dict = make_data_dict(list_dirs, save_filename)

length=np.zeros((len(data_dict)))
for i in range(len(data_dict)):

    pc=data_dict[i]['point_cloud']
    length[i]=pc.shape[0]


import ipdb; ipdb.set_trace()
print(length.min())
# print(length)