#!/usr/bin/env python
# coding: utf-8

# In[25]:


# Libraries
import argparse
import sys
import os.path as ops
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from pdb import set_trace

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch import functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from dataloader import SNYData
from models.UNets import UNet1D
from models.BaseModels import EncoderDecoder, EncoderDecoder_ss, EncoderDecoder_s, EncoderDecoder_m
from models.Models import ICNN_s,ICNN_m,ICNN_l
from utils import figure_plot,rmse,psnr,seed_everything
import pdb
# In[26]:


def get_args():
    parser = argparse.ArgumentParser(description='Train Network to Denoise', 
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e','--epochs',type=int,default=200,help='Number of epochs',dest='epochs')
    parser.add_argument('-b','--batch_size',type=int,default=1,help='Batch size')
    parser.add_argument('-l','--lr',type=float,default=1e-4,help='Learn rate',dest='lr')
    parser.add_argument('-dp','--data_path',type=str,default='../TrainSample',help='Data path')
    parser.add_argument('-mp','--model_path',type=str,default='../ckpt',help='Model path')
    parser.add_argument('-lp','--log_path',type=str,default=None,help='Log path')
    parser.add_argument('-vr','--validation_rate',type=float,default=0.3,help='Validation rate')
    parser.add_argument('-r','--resume',type=bool,default=False,help='Wheter resume previous model')
    parser.add_argument('-d', '--log_dir', dest='log_dir', type=str, default='./runs', help='The directory of log file')
    parser.add_argument('-fd','--fig_dir',dest='fig_dir',type=str,default='./Fig',help='The directory of figures or images')
    return parser.parse_args()


def train(net,device,argparse=None):
    writer = SummaryWriter(comment=f'LR_{argparse.lr}_BS_{argparse.batch_size}', log_dir=argparse.log_dir)

    dataset=SNYData(argparse.data_path)
    train_len = int(len(dataset) * (1-argparse.validation_rate))
    val_len = int(len(dataset) - train_len)
    trainset,valset = random_split(dataset,[train_len,val_len])
                                      #generator=torch.Generator().manual_seed(42))
    trainloader = DataLoader(trainset,batch_size=argparse.batch_size,shuffle=True)
    valoader = DataLoader(valset,batch_size=argparse.batch_size,shuffle=True)
    criterion = nn.SmoothL1Loss()
    scaler = GradScaler()
    schd_batch_update = False
    optimizer = optim.SGD(net.parameters(),lr=argparse.lr,momentum=0.9,weight_decay=1e-6)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = 10,
            eta_min=1e-6,
            last_epoch=-1
       )

    net = net.to(device)
    if argparse.resume:
        save_dict = torch.load(ops.join(argparse.model_path,'ckpt_240.pth'),map_location=device)
        net.load_state_dict(save_dict['net_state_dict'])
        optimizer.load_state_dict(save_dict['optim_state_dict'])
        loss = save_dict['loss']
        epoch = save_dict['epoch']
    
    score = 0.0
    global_step = 0

    
    for epoch in range(1,argparse.epochs+1):
        with tqdm(total=len(trainloader), desc=f'Epoch {epoch}/{argparse.epochs}') as pbar:
            
            epoch_loss = None
            
            for index,data in enumerate(trainloader):
                net.train()
                inputs, labels = data
                inputs,labels = inputs.to(device),labels.to(device)
               
                with autocast():
                    preds = net(inputs)
                    loss = criterion(preds,labels)
                
                
                #epoch_loss = epoch_loss  loss.item()
                if epoch_loss is None:
                    epoch_loss = loss.item()
                else:
                    epoch_loss = epoch_loss * 0.99 + loss.item() * 0.01 # regularze the loss
                
                # loss backward propagation
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() 

                # # optimizer without scaler
                # loss.backward()
                # optimizer.zero_grad()
                # optimizer.step()

                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(1)


            writer.add_scalar('Loss/train',loss,global_step)

            if epoch % 10 ==0:
                average_loss,fig = val(valoader,criterion,net,device,writer,global_step)
                scheduler.step(average_loss)
                print(' Testing loss: '+str(average_loss))
                if epoch % 10 == 0:
                    fig.savefig('./Fig/Evaluation_Example_{}'.format(epoch),bbox_inches='tight')
                    plt.close('all')

            if epoch % 10 == 0:
                fig = figure_plot(inputs[0].detach().cpu().numpy().squeeze(),\
                    labels[0].detach().cpu().numpy().squeeze(),\
                    preds[0].detach().cpu().numpy().squeeze()) # Noised signal, clean, prediction
                fig.savefig('./Fig/Training_Example_{}.png'.format(epoch), bbox_inches='tight')
                plt.close('all')
            global_step += 1

            if epoch % 10 == 0:
                torch.save({
                'epoch': epoch,
                'net_state_dict': net.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'loss': loss},'{}/ckpt_{}.pth'.format(argparse.model_path,str(epoch)))
    
    print('After {} iterations, training finishes!'.format(global_step))
    writer.close()

def val(valoader,criterion,net,device,writer,global_step):
    with torch.no_grad():
        net.eval()
        val_loss = 0.0

        with tqdm(total=len(valoader), desc='Validation round', unit='batch', leave=False) as pbar:

            for index,data in enumerate(valoader):
                    
                    inputs, labels = data
                    inputs,labels = inputs.to(device),labels.to(device)
                   
                    preds = net(inputs)
                    loss = criterion(preds,labels)
                    val_loss += loss.item()
                    
                    pbar.set_postfix(**{'loss (batch)': loss.item()})
                    pbar.update(1)
            writer.add_scalar('Loss/val',loss,global_step)
            fig = figure_plot(inputs[0].detach().cpu().numpy().squeeze(),\
                        labels[0].detach().cpu().numpy().squeeze(),\
                        preds[0].detach().cpu().numpy().squeeze()) # Noised signal, clean, prediction
            
            return val_loss/len(valoader),fig
    

def main():
    net = ICNN_s(n_channels=1,n_classes=1)
    argparse = get_args()
    epochs = argparse.epochs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start = time.time()
    train(net,device,argparse)
    end = time.time()
    period = (end-start)
    print('Total training time for '+str(epochs)+' epochs :'+str(period)+'s. The average training time for each epoch:'+str(float(period/epochs))+'s.')

if __name__=='__main__':
    seed_everything(666)
    main()




