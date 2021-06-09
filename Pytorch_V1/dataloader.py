#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np
import torch
import torch.utils.data as data
import os.path as ops
import glob
from matplotlib import pyplot as plt


# In[52]:


class SNYData(data.Dataset):
    def __init__(self,data_path):
        self.data_path = data_path
        clean_data_dir = ops.join(self.data_path,'labels')
        noise_data_dir = ops.join(self.data_path,'data')
        self.noised_list = glob.glob(noise_data_dir+'/*')
        self.clean_list = glob.glob(clean_data_dir+'/*')
        assert len(self.noised_list) == len(self.clean_list),         'clean data should have same amount as the noise data!'
    
    def process_single_data(self,file_name):
        with open(file_name,'r') as f:
            lines = f.readlines() 
        f.close()
        data = np.array([float(x) for x in lines],dtype=np.float32)
        return data
        
    def __getitem__(self,idx):
        noise_data = self.process_single_data(self.noised_list[idx])
        clean_data = self.process_single_data(self.clean_list[idx])
        noise_data,clean_data = torch.from_numpy(noise_data), torch.from_numpy(clean_data)
        return torch.unsqueeze(noise_data, 0), torch.unsqueeze(clean_data,0) 
    # expand the 1D signals (Batch_Size,Signal_Length) into 1D tensor (Batch_Size,Channels,Signal_Length) for torch processing
        
    
    def __len__(self):
        return len(self.clean_list)


# In[53]:


if __name__=='__main__':
    path = '/home/vincent/Documents/Projects/DeepDenoising/SYN_DATA/Time_Disorder_data'
    dataset = SNYData(path)
    from torch.utils.data import DataLoader
    trainloader = DataLoader(dataset,batch_size=4,shuffle=True)
    data_ITR = trainloader.__iter__()
    data = data_ITR.next()
    print(data[0].shape)

# In[ ]:





# In[ ]:




