import numpy as np
import h5py
import skimage.measure
from NN1D_parts_v1 import _conv1d,_batchnorm1d,_relu,_leaky_relu,_pooling_1d,_upsample1d
import pdb

fix_batch_norm = False

def doubleConv(x,weights,bias,means,Vars):
    weight1,weight2,weight3,weight4 = weights
    bias1,bias2,bias3,bias4 = bias
    mean1,mean2 = means
    var1,var2 = Vars
    x = _conv1d(x,weight1,bias1)
    x = _batchnorm1d(x,weight2,bias2,mean1,var1)
    x = _relu(x)
    x = _conv1d(x,weight3,bias3)
    x = _batchnorm1d(x,weight4,bias4,mean2,var2)
    out = _relu(x)
    return out

def maxpoolConv(x,weights,bias,means,Vars):
    weight1,weight2,weight3,weight4 = weights
    bias1,bias2,bias3,bias4 = bias
    mean1,mean2 = means
    var1,var2 = Vars
    x = _pooling_1d(x,ksize=2)
    x = _conv1d(x,weight1,bias1)
    x = _batchnorm1d(x,weight2,bias2,mean1,var1)
    x = _relu(x)
    x = _conv1d(x,weight3,bias3)
    x = _batchnorm1d(x,weight4,bias4,mean2,var2)
    out = _relu(x)
    return out

def upConv(x1,x2,weights,bias,means,Vars):
    weight1,weight2,weight3,weight4 = weights
    bias1,bias2,bias3,bias4 = bias
    mean1,mean2 = means
    var1,var2 = Vars

    x1 = _upsample1d(x1,scale=2)
    if x1.shape[-1]<=x2.shape[-1]:
        diff = x2.shape[-1] - x1.shape[-1]
        x1 = np.pad(x1,((0,0),(0,0),(diff//2,diff - diff//2)),'constant',constant_values=(0,0))
    else:
        diff = x1.shape[-1] - x2.shape[-1]
        x2 = np.pad(x2,((0,0),(0,0),(diff//2,diff - diff//2)),'constant',constant_values=(0,0))
    x = np.concatenate((x2,x1),axis=1)
    x = _conv1d(x,weight1,bias1)
    x = _batchnorm1d(x,weight2,bias2,mean1,var1)
    x = _relu(x)
    x = _conv1d(x,weight3,bias3)
    x = _batchnorm1d(x,weight4,bias4,mean2,var2)
    out = _relu(x)
    return out

def outConv(x,weights,bias):
    weight1= weights[0]
    bias1 = bias[0]
    out = _conv1d(x,weight1,bias1)
    return out

class ICNN_s_v1:
    def __init__(self,model_path):
        self.model_path = model_path
        self.inc = doubleConv # 1--> 2
        self.donw1 = maxpoolConv # 2--> 4
        self.donw2 = maxpoolConv # 4 --> 8
        self.donw3 = maxpoolConv # 8 --> 16
        self.donw4 = maxpoolConv # 16 --> 16
        
        self.up1 = upConv # 16+16 --> 16
        self.up2 = upConv # 16 --> 8
        self.up3 = upConv # 8 --> 4
        self.up4 = upConv # 4 --> 2
        self.outc = outConv # 2 --> 1
        self.weights_load()

    def weights_load(self):
        # loading weights and bias
        f = h5py.File(self.model_path, 'r')
        
        weights_list = list(f.keys())
        self.model_weights_bn = {
        'inc':list(),
        'down1':list(),
        'down2':list(),
        'down3':list(),
        'down4':list(),
        'up1':list(),
        'up2':list(),
        'up3':list(),
        'up4':list(),
        'outc':list()   
        }
        self.model_bias_bn = {
        'inc':list(),
        'down1':list(),
        'down2':list(),
        'down3':list(),
        'down4':list(),
        'up1':list(),
        'up2':list(),
        'up3':list(),
        'up4':list(),
        'outc':list()   
        }
        self.model_means_bn = {
        'inc':list(),
        'down1':list(),
        'down2':list(),
        'down3':list(),
        'down4':list(),
        'up1':list(),
        'up2':list(),
        'up3':list(),
        'up4':list(),
        'outc':list()   
        }
        self.model_vars_bn = {
        'inc':list(),
        'down1':list(),
        'down2':list(),
        'down3':list(),
        'down4':list(),
        'up1':list(),
        'up2':list(),
        'up3':list(),
        'up4':list(),
        'outc':list()   
        }
        for item in weights_list:
            if item.split('.')[-1] == 'weight':
                self.model_weights_bn[item.split('.')[0]].append(f[item])
            if item.split('.')[-1] == 'bias':
                self.model_bias_bn[item.split('.')[0]].append(f[item])
            if item.split('.')[-1] == 'running_mean':
                if fix_batch_norm:
                    self.model_means_bn[item.split('.')[0]].append(f[item])
                else:
                    self.model_means_bn[item.split('.')[0]].append(None) # Not fix batch norm means
            if item.split('.')[-1] == 'running_var':
                if fix_batch_norm:
                    self.model_vars_bn[item.split('.')[0]].append(f[item])
                else : 
                    self.model_vars_bn[item.split('.')[0]].append(None) # Not fix batch norm vars

    def compute(self,input_signal):
        '''
        denoise the input signal by our ICNN_s model
        input_siganl: array (batch,channel, length)
        '''
        x1 = self.inc(input_signal,\
            self.model_weights_bn['inc'],\
            self.model_bias_bn['inc'],
            self.model_means_bn['inc'],
            self.model_vars_bn['inc']
            )
        x2 = self.donw1(x1,\
            self.model_weights_bn['down1'],\
            self.model_bias_bn['down1'],
            self.model_means_bn['down1'],
            self.model_vars_bn['down1']
            )
        x3 = self.donw2(x2,\
            self.model_weights_bn['down2'],\
            self.model_bias_bn['down2'],
            self.model_means_bn['down2'],
            self.model_vars_bn['down2']
            )
        x4 = self.donw3(x3,\
            self.model_weights_bn['down3'],\
            self.model_bias_bn['down3'],
            self.model_means_bn['down3'],
            self.model_vars_bn['down3']
            )
        x5 = self.donw4(x4,\
            self.model_weights_bn['down4'],\
            self.model_bias_bn['down4'],
            self.model_means_bn['down4'],
            self.model_vars_bn['down4']
            )
        x = self.up1(x5,x4,\
            self.model_weights_bn['up1'],\
            self.model_bias_bn['up1'],
            self.model_means_bn['up1'],
            self.model_vars_bn['up1']
            )
        x = self.up2(x,x3,\
            self.model_weights_bn['up2'],\
            self.model_bias_bn['up2'],
            self.model_means_bn['up2'],
            self.model_vars_bn['up2']
            )
        x = self.up3(x,x2,\
            self.model_weights_bn['up3'],\
            self.model_bias_bn['up3'],
            self.model_means_bn['up3'],
            self.model_vars_bn['up3']
            )
        x = self.up4(x,x1,\
            self.model_weights_bn['up4'],\
            self.model_bias_bn['up4'],
            self.model_means_bn['up4'],
            self.model_vars_bn['up4']
            )
        out = self.outc(x,\
            self.model_weights_bn['outc'],\
            self.model_bias_bn['outc'])
        return out


if __name__=='__main__':
    model_path = r'../ckpt/ckpt_best.h5'
    name ='Test4'
    net = ICNN_s_v1(model_path)
    net.weights_load()
    
    # label: noise-free signal
    label_name = r'/home/vincent/Documents/Projects/DeepDenoising/Code1/TestSamples/{}_label.txt'.format(name)
    with open(label_name,'r') as f:
        line = f.readlines() 
    f.close()
    label = np.array([float(x) for x in line],dtype=np.float32)

    # data: noised signal
    file_name = r'/home/vincent/Documents/Projects/DeepDenoising/Code1/TestSamples/{}_data.txt'.format(name)
    with open(file_name,'r') as f:
        lines = f.readlines() 
    f.close()
    sig = np.array([float(x) for x in lines],dtype=np.float32)
    sig = np.expand_dims(np.expand_dims(sig,0),0) #epand the signal into (1,1,L) array to fit the trained parameters
    pred = net.compute(sig)
    
    # output: denoised signal
    pred = pred.squeeze()

    import matplotlib.pyplot as plt
    from utils import rmse,psnr

    # fig = plt.figure(figsize=(50,40))
    # ax1 = fig.add_subplot(311)
    # ax1.set_title('Noise-free data',fontsize=20)
    # ax1.plot(label)
    # ax2 = fig.add_subplot(312)
    # ax2.plot(sig.squeeze())
    # ax2.set_title('Noisy data',fontsize=20)
    # ax3 = fig.add_subplot(313)
    # ax3.set_title('Denoised data',fontsize=20)
    # ax3.plot(pred[1:100])
    # ax3.text(2,0,'RMSE %0.2f'%rmse(pred[1:101]-label),fontsize=15)
    # ax3.text(2,1.5,'PSNR %0.2f'%psnr(pred[1:101]-label),fontsize=15)
    # plt.show()
    # if fix_batch_norm:
    #     figpred.savefig('../Fig/useful/Tset1_data_fixed_batch.png')
    # else:
    #     figpred.savefig('../Fig/useful/Tset1_data_unfixed_batch.png')
    
    figlabel = plt.figure(figsize=(50,40))
    plt.plot(label)
    plt.title('Noise-free data',fontsize=70)
    figlabel.savefig('../Fig/useful/final/{}_label.png'.format(name),bbox_inches='tight',pad_inches=0.0)
    
    fignoise = plt.figure(figsize=(50,40))
    plt.plot(sig.squeeze())
    plt.title('Noisy data',fontsize=70)
    fignoise.savefig('../Fig/useful/final/{}_noised_data.png'.format(name),bbox_inches='tight',pad_inches=0.0)

    figpred = plt.figure(figsize=(50,40))
    plt.plot(pred[1:100])
    plt.title('Denoised data (Our Numpy\'s)',fontsize=70)
    plt.annotate('RMSE %0.2f'%rmse(pred[1:101]-label),xy=(300,1800),fontsize=50,xycoords='figure points')
    plt.annotate('PSNR %0.2f'%psnr(pred[1:101]-label),xy=(300,1850),fontsize=50,xycoords='figure points')

    if fix_batch_norm:
        figpred.savefig('../Fig/useful/final/{}_Numpy_data_Fixed_batch.png'.format(name),bbox_inches='tight',pad_inches=0.0,)
    else:
        figpred.savefig('../Fig/useful/final/{}_Numpy_data_unfixed_batch.png'.format(name),bbox_inches='tight',pad_inches=0.0)


    from scipy.signal.signaltools import wiener
    weinerPred = wiener(sig.squeeze())

    figwinner = plt.figure(figsize=(50,40))
    plt.plot(weinerPred)
    plt.title('Denoised data (Winer Filter)',fontsize=70)
    plt.annotate('RMSE %0.2f'%rmse(weinerPred-label),xy=(300,1800),fontsize=50,xycoords='figure points')
    plt.annotate('PSNR %0.2f'%psnr(weinerPred-label),xy=(300,1850),fontsize=50,xycoords='figure points')
    figwinner.savefig('../Fig/useful/final/{}_Numpy_weiner_denoised_data.png'.format(name),bbox_inches='tight',pad_inches=0.0)