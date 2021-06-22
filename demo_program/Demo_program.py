import numpy as np
import h5py
import skimage.measure
from models.NN1D_parts_v1 import _conv1d,_batchnorm1d,_relu,_leaky_relu,_pooling_1d,_upsample1d,Wiener_Filter
from models.srls import process_L1,process_GMC
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

class weinerConv_s_v1:
    def __init__(self,model_path):
        self.model_path = model_path
        self.winc = doubleConv 
        self.wdonw1 = maxpoolConv 
        self.wdonw2 = maxpoolConv 
        self.wdonw3 = maxpoolConv 

    def weights_load(self):
        # loading weights and bias
        f = h5py.File(self.model_path, 'r')
        
        weights_list = list()
        for item in f.keys():
            if item.split('.')[0] == 'wienernet':
                weights_list.append(item)
        
        self.model_weights_bn = {
        'inc':list(),
        'down1':list(),
        'down2':list(),
        'down3':list()
        }
        self.model_bias_bn = {
        'inc':list(),
        'down1':list(),
        'down2':list(),
        'down3':list()
        }
        self.model_means_bn = {
        'inc':list(),
        'down1':list(),
        'down2':list(),
        'down3':list()
        }
        self.model_vars_bn = {
        'inc':list(),
        'down1':list(),
        'down2':list(),
        'down3':list()
        }
        self.parameters = list() # beta & lamda
        for item in weights_list:
            if item.split('.')[-1] == 'weight':
                self.model_weights_bn[item.split('.')[1]].append(f[item])
            if item.split('.')[-1] == 'bias':
                self.model_bias_bn[item.split('.')[1]].append(f[item])
            if item.split('.')[-1] == 'running_mean':
                if fix_batch_norm:
                    self.model_means_bn[item.split('.')[1]].append(f[item])
                else:
                    self.model_means_bn[item.split('.')[1]].append(None) # Not fix batch norm means
            if item.split('.')[-1] == 'running_var':
                if fix_batch_norm:
                    self.model_vars_bn[item.split('.')[1]].append(f[item])
                else : 
                    self.model_vars_bn[item.split('.')[1]].append(None) # Not fix batch norm vars
            if item.split('.')[-1] == 'beta':
                self.parameters.append(f[item])

    def compute(self,input_system):
        h1 = self.winc(input_system,\
            self.model_weights_bn['inc'],\
            self.model_bias_bn['inc'],
            self.model_means_bn['inc'],
            self.model_vars_bn['inc']
            )
        h2 = self.wdonw1(h1,\
            self.model_weights_bn['down1'],\
            self.model_bias_bn['down1'],
            self.model_means_bn['down1'],
            self.model_vars_bn['down1']
            )
        h3 = self.wdonw2(h2,\
            self.model_weights_bn['down2'],\
            self.model_bias_bn['down2'],
            self.model_means_bn['down2'],
            self.model_vars_bn['down2']
            )
        h = self.wdonw3(h3,\
            self.model_weights_bn['down3'],\
            self.model_bias_bn['down3'],
            self.model_means_bn['down3'],
            self.model_vars_bn['down3']
            )
        h = np.transpose(_pooling_1d(h,ksize=4),(1,0,2))
        return h, self.parameters


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
        self.wienerNet = weinerConv_s_v1(self.model_path)
        self.wienerNet.weights_load()

    def weights_load(self):
        # loading weights and bias
        f = h5py.File(self.model_path, 'r')
        
        weights_list = list()
        for item in f.keys():
            if item.split('.')[0] != 'wienernet':
                weights_list.append(item)
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

        k = self.model_weights_bn['down4'][0]
        _,h0 = Wiener_Filter(input_signal.squeeze())

        h,parameters = self.wienerNet.compute(np.expand_dims(h0,axis=(0,1)))
        h = np.repeat(h,k.shape[1],axis=1) # one batch, so no average operation
        self.model_weights_bn['down4'][0] = k + parameters[0] * h * k
        
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

######################################################### Main algorithm ################################################
if __name__=='__main__':
    ########---* loading models *---#######
    model_path = r'./ckpt_best.h5'
    name_list = ['Test0','Test1','Test2','Test3','Test4','Test5']
    for name in name_list:
        ########---* initial network
        net = ICNN_s_v1(model_path)
        net.weights_load()
        
        ########---* loading data and labels *---#######
        # label: noise-free signal
        label_name = r'./TestSamples/{}_label.txt'.format(name)
        with open(label_name,'r') as f:
            line = f.readlines() 
        f.close()
        label = np.array([float(x) for x in line],dtype=np.float32)
        # data: noised signal
        file_name = r'./TestSamples/{}_data.txt'.format(name)
        with open(file_name,'r') as f:
            lines = f.readlines() 
        f.close()
        sig = np.array([float(x) for x in lines],dtype=np.float32)
        sig = np.expand_dims(np.expand_dims(sig,0),0) #epand the signal into (1,1,L) array to fit the trained parameters
        

        ########---* denoise the signals *---#######
        pred = net.compute(sig)
        pred = pred.squeeze()

        ########---* Wiener filter desnoises the signals *---#######
        from scipy.signal.signaltools import wiener
        weinerPred = wiener(sig.squeeze())

        ########---* plot results *---#######
        import matplotlib.pyplot as plt
        from utils import rmse,psnr

        fig = plt.figure(figsize=(50,40))
        ax1 = fig.add_subplot(221)
        ax1.set_title('Noise-free data',fontsize=30)
        ax1.plot(label)
        ax2 = fig.add_subplot(222)
        ax2.plot(sig.squeeze())
        ax2.set_title('Noisy data',fontsize=30)
        ax3 = fig.add_subplot(223)
        ax3.plot(weinerPred)
        plt.title('Denoised data (Winer filter\'s)',fontsize=30)
        ax3.annotate('RMSE %0.2f'%rmse(weinerPred-label),xy=(350,550),fontsize=25,xycoords='figure points')
        ax3.annotate('PSNR %0.2f'%psnr(weinerPred-label),xy=(350,580),fontsize=25,xycoords='figure points')
        ax4 = fig.add_subplot(224)
        ax4.set_title('Denoised data (Our numpy version\'s)',fontsize=30)
        ax4.plot(pred[1:101])
        ax4.annotate('RMSE %0.2f'%rmse(pred[1:101]-label),xy=(1470,550),fontsize=25,xycoords='figure points')
        ax4.annotate('PSNR %0.2f'%psnr(pred[1:101]-label),xy=(1470,580),fontsize=25,xycoords='figure points')
        plt.show()
        if fix_batch_norm:
            fig.savefig('./Figs/{}_fixed_mode_results.png'.format(name),bbox_inches='tight',pad_inches=0.0)
        else:
            fig.savefig('./Figs/{}_unfixed_mode_results.png'.format(name),bbox_inches='tight',pad_inches=0.0)

###########################################################################################################################