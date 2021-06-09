import torch
import h5py
import numpy as np
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = './ckpt/ckpt_best.pth'
ckpt = torch.load(model_path, map_location=device)
tensor_para = ckpt['net_state_dict']

f = h5py.File('./ckpt/ckpt_best.h5','w')

# pth file to h5 file
for key in tensor_para.keys():
    if key.split('.')[-1] == 'weight' or key.split('.')[-1] == 'bias' \
    or key.split('.')[-1] == 'running_mean' or key.split('.')[-1] == 'running_var'\
    or key.split('.')[-1] == 'beta':
        layer = f.create_dataset(key,data=tensor_para[key].detach().cpu().numpy())
        # for debug
        print(key,'' ,tensor_para[key].detach().cpu().numpy().shape)

#turn hf file to csv file for visualization
keys = list(f.keys())
values = [ f[key] for key in keys]
pd.DataFrame(values).to_csv("./ckpt/ckpt_best.csv")

f.close()