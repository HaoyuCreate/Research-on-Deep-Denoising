import argparse
import sys
import time
import os.path as ops
import matplotlib.pyplot as plt
from pdb import set_trace

import numpy as np
from numpy.random import randn, randint

import torch
import torch.nn as nn
from torch import functional as F
from models.UNets import UNet1D
from models.BaseModels import EncoderDecoder,EncoderDecoder_ss, EncoderDecoder_s, EncoderDecoder_m
from models.Models import ICNN_s,ICNN_m,ICNN_l,Wiener_Filter_Tensor
from models.srls import process_L1,process_GMC
from utils import *

def rmse (x): return np.sqrt( np.mean( np.abs(x)**2 ) )
def psnr (x): return 20 * np.log10(10 /np.mean( np.abs(x)**2))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Single_Time_disorder_data_creator(sigma,length=100,sig_rate=0.2,seed=None):
    # clean data initial
    label = np.zeros(length)
    
    np.random.seed(seed) 
    number = int(sig_rate * length)
    signal = randint(2,11,number)
    index = randint(0,length,number)
    
    assert len(signal)==len(index), 'number should be equally long to index'
    label[index] = signal.astype(np.float)
    
    noise = randn(length)# noise : white Gaussian noise
    sample = label + sigma * noise # signal plus noise
    
    return label,sample


def get_args():
    parser = argparse.ArgumentParser(description='Test Network of Denoising', 
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b','--batch_size',type=int,default=1,help='Batch size')
    parser.add_argument('-mp','--model_path',type=str,default='../ckpt/ckpt_best.pth',help='Model path')
    parser.add_argument('-lp','--log_path',type=str,default=None,help='Log path')
    parser.add_argument('-fd','--fig_dir',dest='fig_dir',type=str,default='./Figs',help='The directory of figures or images')
    return parser.parse_args()

def weight_loading(argparse):
    ckpt = torch.load(argparse.model_path, map_location=device)
    net_weights = ckpt['net_state_dict']
    net_para = net_weights['wienernet.beta']
    net_weights.pop('wienernet.beta')
    return net_weights, net_para
    

def denoising(net,data):
    mask_dtype = torch.float32 if net.n_classes == 1 else torch.long
    net.to(device)
    data = data.to(device,dtype=mask_dtype)
    preds = net(data)
    return preds.detach().cpu().numpy().squeeze(),data.detach().cpu().numpy().squeeze()

def evaluation(preds,labels,data,argparse,Info='Test'):
    assert preds.shape == labels.shape == data.shape, 'Data sizes are not compatiable!'
    if data.ndim <= 1:
        i = 'target'
        # fig = figure_plot(data,labels,preds)
        # fig.savefig(ops.join(argparse.fig_dir,'best_{}_Example_{}_eval.png'.format(Info,i)), bbox_inches='tight')
    else:
        for i in range(preds.shape[0]):
            # fig = figure_plot(data[i],labels[i],preds[i])
            # fig.savefig(ops.join(argparse.fig_dir,'best_{}_Example_{}_eval.png'.format(Info,i)), bbox_inches='tight')
            pass

def main(net,net_weights):
    '''
        For random input
    '''
    net.load_state_dict(net_weights)
    net.train()
    #net.eval()

    _labels,_data = list(),list()
    for i in range(argparse.batch_size):
        label,sample = Single_Time_disorder_data_creator(sigma=3)
        _labels.append(label)
        _data.append(sample)
    
    _data = torch.from_numpy(np.expand_dims(np.array(_data), axis=1))
    labels = np.array(_labels)

    start = time.time()
    preds,data = denoising(net,_data)
    end = time.time()
    period = (end-start)
    
    evaluation(preds,labels.squeeze(),data,argparse)
    print('Average running time of the network for a signal: '+str(float(period/argparse.batch_size))+'s.')

    start1 = time.time()
    if _data.numpy().squeeze().ndim <= 1:
        wiener_preds = np.array(wiener_filter(_data.numpy().squeeze())[0])
    else:
        wiener_preds = np.array([wiener_filter(sample)[0] for sample in _data.numpy().squeeze()])
    end = time.time()
    period1 = (end-start)
    
    evaluation(wiener_preds,labels.squeeze(),data,argparse,Info='Wiener_Filter')
    print('Average running time of the Wiener filter for a signal: '+str(float(period1/argparse.batch_size))+'s.')

    return preds,wiener_preds,labels.squeeze(),_data.numpy().squeeze()

def loading_data(file_name):
    with open(file_name,'r') as f:
        lines = f.readlines() 
    f.close()
    data = np.array([float(x) for x in lines],dtype=np.float32)
    return data

def main2(net,name,net_weights):
    '''
        for a given input (one batch),e.g. Test1_label.txt
    '''
    net.load_state_dict(net_weights)
    ##net.eval()
    net.train()
    _labels,_data = list(),list()
    
    for _ in range(argparse.batch_size):_labels.append(loading_data('../TestSamples/{}_label.txt'.format(name)))
    for _ in range(argparse.batch_size):_data.append(loading_data('../TestSamples/{}_data.txt'.format(name)))

    _data = torch.from_numpy(np.expand_dims(np.array(_data), axis=1))
    labels = np.array(_labels).squeeze()

    start = time.time()
    preds,data = denoising(net,_data)
    end = time.time()
    period = (end-start)

    evaluation(preds,labels,data,argparse)
    print('Average running time of the network for a signal: '+str(float(period/argparse.batch_size))+'s.')

    start1 = time.time()
    if _data.numpy().squeeze().ndim <= 1:
        wiener_preds = np.array(wiener_filter(_data.numpy().squeeze())[0])
    else:
        wiener_preds = np.array([wiener_filter(sample)[0] for sample in _data.numpy().squeeze()])
    end = time.time()
    period1 = (end-start)
    
    evaluation(wiener_preds,labels,_data.numpy().squeeze(),argparse,Info='Wiener_Filter')
    print('Average running time of the Wiener filter for a signal: '+str(float(period1/argparse.batch_size))+'s.')

    return preds, wiener_preds,labels,data



if __name__=='__main__':
    argparse = get_args()
    net_weights,net_para = weight_loading(argparse)
    
    
    # #main2
    # testing on given samples
    net = ICNN_s(n_channels=1,n_classes=1,pre_train_para=net_para)
    name_list = ['Test0','Test1','Test2','Test3','Test4','Test5']
    for name in name_list:
        fig_pred, fig_weiner_pred, fig_label, fig_data = main2(net,name,net_weights)
        import matplotlib.pyplot as plt

        figpred = plt.figure(figsize=(50,40))
        plt.plot(fig_pred)
        plt.title('Denoised data (Our Pytorch\'s)',fontsize=70)
        plt.annotate('RMSE %0.2f'%rmse(fig_pred-fig_label),xy=(300,1900),fontsize=90,xycoords='figure points')
        plt.annotate('PSNR %0.2f'%psnr(fig_pred-fig_label),xy=(300,1800),fontsize=90,xycoords='figure points')
        figpred.savefig(ops.join(argparse.fig_dir,'{}_Torch_pred_Unfixed_batch.png'.format(name)),bbox_inches='tight',pad_inches=0.0)

        figlabel = plt.figure(figsize=(50,40))
        plt.plot(fig_label)
        plt.title('Noise-free data',fontsize=70)
        figlabel.savefig(ops.join(argparse.fig_dir,'{}_label.png'.format(name)),bbox_inches='tight',pad_inches=0.0)

        figdata = plt.figure(figsize=(50,40))
        plt.plot(fig_data)
        plt.title('Noisy data',fontsize=70)
        figdata.savefig(ops.join(argparse.fig_dir,'{}_data.png'.format(name)),bbox_inches='tight',pad_inches=0.0)

        figweinerpred = plt.figure(figsize=(50,40))
        plt.plot(fig_weiner_pred)
        plt.title('Denoised data (Our Pytorch\'s)',fontsize=70)
        plt.annotate('RMSE %0.2f'%rmse(fig_weiner_pred-fig_label),xy=(300,1900),fontsize=90,xycoords='figure points')
        plt.annotate('PSNR %0.2f'%psnr(fig_weiner_pred-fig_label),xy=(300,1800),fontsize=90,xycoords='figure points')
        figweinerpred.savefig(ops.join(argparse.fig_dir,'{}_Weiner_pred.png'.format(name)),bbox_inches='tight',pad_inches=0.0)



    # main
    # testing on random samples and save the samples as testing sample
    # name = 'Test4'
    # net = ICNN_s(n_channels=1,n_classes=1,pre_train_para=net_para)
    # preds,weinerPreds,labels,data = main(net)
    # import matplotlib.pyplot as plt
    
    # figpred = plt.figure(figsize=(50,40))
    # plt.plot(preds)
    # plt.title('Denoised data (Our Pytorch\'s)',fontsize=70)
    # plt.annotate('RMSE %0.2f'%rmse(preds-labels),xy=(300,1800),fontsize=50,xycoords='figure points')
    # plt.annotate('PSNR %0.2f'%psnr(preds-labels),xy=(300,1850),fontsize=50,xycoords='figure points')
    # figpred.savefig(ops.join(argparse.fig_dir,'{}_Torch_pred_Fixed_batch.png'.format(name)),bbox_inches='tight',pad_inches=0.0)

    # figweinerpred = plt.figure(figsize=(50,40))
    # plt.plot(weinerPreds)
    # plt.title('Denoised data (Our Pytorch\'s)',fontsize=70)
    # plt.annotate('RMSE %0.2f'%rmse(weinerPreds-labels),xy=(300,1800),fontsize=50,xycoords='figure points')
    # plt.annotate('PSNR %0.2f'%psnr(weinerPreds-labels),xy=(300,1850),fontsize=50,xycoords='figure points')
    # figweinerpred.savefig(ops.join(argparse.fig_dir,'{}_Weiner_pred.png'.format(name)),bbox_inches='tight',pad_inches=0.0)


    # figlabel = plt.figure(figsize=(50,40))
    # plt.plot(labels)
    # plt.title('Noise-free data',fontsize=70)
    # figlabel.savefig(ops.join(argparse.fig_dir,'{}_label.png'.format(name)),bbox_inches='tight',pad_inches=0.0)


    # figdata = plt.figure(figsize=(50,40))
    # plt.plot(data)
    # plt.title('Noisy data',fontsize=70)
    # figdata.savefig(ops.join(argparse.fig_dir,'{}_data.png'.format(name)),bbox_inches='tight',pad_inches=0.0)


    # label_name = r'../TestSamples/{}_label.txt'.format(name)
    # file_name = r'../TestSamples/{}_data.txt'.format(name)

    # with open(label_name,'w') as f:
    #     for number in labels:
    #         f.write(str(number))
    #         f.write('\n')
    # f.close()

    # with open(file_name,'w') as f:
    #     for number in data:
    #         f.write(str(number))
    #         f.write('\n')
    # f.close()


    
    # # main3 # calculate the average value
    # net = ICNN_s(n_channels=1,n_classes=1,pre_train_para=net_para)
    
    # model_RMSE = list()
    # model_PSNR = list()
    # wiener_RMSE = list()
    # wiener_PSNR = list()

    
    # for _ in range(1000):
    #     preds,weinerPreds,labels,data = main(net)
    #     model_RMSE.append(rmse(preds-labels))
    #     model_PSNR.append(psnr(preds-labels))

    #     wiener_RMSE.append(rmse(weinerPreds-labels))
    #     wiener_PSNR.append(psnr(weinerPreds-labels))
    
    # average_model_RMSE = float(sum(model_RMSE)/1000)
    # average_model_PSNR = float(sum(model_PSNR)/1000)

    # average_wiener_RMSE = float(sum(wiener_RMSE)/1000)
    # average_wiener_PSNR = float(sum(wiener_PSNR)/1000)


    # print('network with fixed parameters-- RMSE:{}, PSNR:{}'.format(average_model_RMSE,average_model_PSNR) )
    # print('network with weiner filter-- RMSE:{}, PSNR:{}'.format(average_wiener_RMSE,average_wiener_PSNR) )


    # log_name = r'./log.txt'
    # with open(log_name,'a') as f:
    #     f.write('\n')
    #     f.write('network with fixed parameters-- RMSE:{}, PSNR:{} \n'.format(average_model_RMSE,average_model_PSNR))
    #     f.write('network with weiner filter-- RMSE:{}, PSNR:{}'.format(average_wiener_RMSE,average_wiener_PSNR) )
    # f.close()

    # #main 4 calculate other method
    # net = ICNN_s(n_channels=1,n_classes=1,pre_train_para=netpara)
    # L1_RMSE = list()
    # L1_PSNR = list()
    # GMC_RMSE = list()
    # GMC_PSNR = list()

    # for _ in range(1000):
    #     _,_,labels,data = main(net)
    #     L1Pred = process_L1(data.squeeze())
    #     GMCPred = process_GMC(data.squeeze())
        
    #     L1_RMSE.append(rmse(L1Pred-labels))
    #     L1_PSNR.append(psnr(L1Pred-labels))

    #     GMC_RMSE.append(rmse(GMCPred-labels))
    #     GMC_PSNR.append(psnr(GMCPred-labels))

    # average_L1_RMSE = float(sum(L1_RMSE)/1000)
    # average_L1_PSNR = float(sum(L1_PSNR)/1000)

    # average_GMC_RMSE = float(sum(GMC_RMSE)/1000)
    # average_GMC_PSNR = float(sum(GMC_PSNR)/1000)
    

    # print('network with L1 norm-- RMSE:{}, PSNR:{}'.format(average_L1_RMSE,average_L1_PSNR) )
    # print('network with GMC norm-- RMSE:{}, PSNR:{}'.format(average_GMC_RMSE,average_GMC_PSNR) )


    # log_name = r'./log.txt'
    # with open(log_name,'a') as f:
    #     f.write('\n')
    #     f.write('network with L1 norm-- RMSE:{}, PSNR:{} \n'.format(average_L1_RMSE,average_L1_PSNR))
    #     f.write('network with GMC norm-- RMSE:{}, PSNR:{}'.format(average_GMC_RMSE,average_GMC_PSNR) )
    # f.close()
