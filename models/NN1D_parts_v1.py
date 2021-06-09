import numpy as np
from numpy.fft import fft,ifft
import sys
import pdb
from scipy.ndimage import zoom
from skimage.measure import block_reduce
# for debugguing
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

def _conv1d(series, kernel, bias, striding=1, padding=1):
    '''
        series: array [b,C_in,L]
        kenerl: array [C_out,C_in,L_k] or [C_out,]
        bias : array [C_out,]
    '''
    in_chanels = series.shape[1] # channels of the input
    out_chanels = kernel.shape[0] # number of kernels

    try:
        res = np.zeros(shape=(1, out_chanels, 
                              (int((series.shape[2] + 2 * padding-kernel.shape[2])/striding))+1
                              ))
    except ZeroDivisionError:
        print("\nException: striding cannot be zero\n", file=sys.stderr)
        sys.exit(1)

    series = np.pad(series,((0,0),(0,0),(padding,padding)),'constant',constant_values=(0,0))

    for j in range(out_chanels):
        # kenel j convolve with all features in series
        # K = kernel[j,:,:].squeeze()
        for i in range(in_chanels):
            # kernel j convolve with feature map i 
            K = kernel[j,i,:]
            X = series[:,i,:].squeeze()

            m = 0
            while (m*striding)+kernel.shape[-1] <= series.shape[-1]:
                res[0,j,m] += np.sum(np.multiply(X[m*striding:(m*striding)+K.shape[-1]] , K))
                m += 1

    bias = np.expand_dims(bias,axis=1)
    res = np.add(res, bias)
    return res

def _batchnorm1d(x,weight,bias,mean=None,var=None,eps=1e-5):
    '''
    weight: gamma
    bias: beta
    '''
    gamma = np.expand_dims(weight,(0,2))
    beta = np.expand_dims(bias,(0,2))

    n_channels = x.shape[1] #0:batch_size,1:number of channel,2:length of input

    if mean is None: # Not fixed Var and Mean, using for training
        sample_mean =  np.repeat(
        np.expand_dims(x.mean(axis=-1),axis=2),\
        x.shape[-1],
        axis = 2
        )
    else:
        sample_mean = np.repeat(np.expand_dims(mean,(0,2)),x.shape[-1],axis=2)
    
    if var is None: # fixed mean and var, using for testing
        sample_var =  np.repeat(
        np.expand_dims(x.var(axis=-1),axis=2),\
        x.shape[-1],
        axis = 2
        )
    else:
        sample_var =  np.repeat(np.expand_dims(var,(0,2)),x.shape[-1],axis=2)

    std = np.sqrt(sample_var + eps)

    x_centered = x - sample_mean
    x_norm = x_centered / std
    #latent = np.multiply(gamma , x_norm)
    out = gamma * x_norm + beta
    cache = (x_norm, x_centered, std, gamma)

    return out

def _relu(z):
    return np.where(z > 0, z, 0.)


def _leaky_relu(z):
    return np.where(z > 0, z, z * 0.01)


def _pooling_1d(series,ksize,striding=1,method='max',padding=0):

    if method == 'average':
        pool_method = np.average
    else:
        pool_method = np.max

    if striding is None or striding==0:
        return series

    if series.shape[-1] % ksize==0:
        res = block_reduce(series,(1,1,ksize),func=np.max)
    else:
        corped_series = np.copy(series)
        corped_series = np.delete(corped_series,tuple(range(-ksize+1,0)),axis=-1)
        res = block_reduce(corped_series,(1,1,ksize),func=np.max)
    return res

def _upsample1d(x, scale,mode='linear'):
    out = zoom(x,[1,1,2],order=1)
    return out

def Wiener_Filter(im,mysize=None, noise=None):
    '''
    A durable implement of wiener filter and calcualte the system function
    '''
    im = np.asarray(im)
    if mysize is None:
        mysize = [3] * im.ndim #kernel size = 3
    mysize = np.asarray(mysize)
    
    # Estimate the local mean
    lMean = np.correlate(im, np.ones(mysize), 'same') / np.prod(mysize, axis=0)
#     print('Mean',lMean)
    # Estimate the local variance
    lVar = (np.correlate(im ** 2, np.ones(mysize), 'same') /
            np.prod(mysize, axis=0) - lMean ** 2)
#     print('Var',lVar)
    
    # Estimate the noise power if needed.
    if noise is None:
        noise = np.mean(np.ravel(lVar), axis=0)
#     print('Noise',noise)

    res = (im - lMean)
#     print('1',res)
    res *= (1 - noise / lVar)
#     print('2',res)
#     print(noise/lVar)
    res += lMean
#     print('3',res)
    out = np.where(lVar < noise, lMean, res)
#     print('4',out)
    h_sys_inverse = np.abs(ifft(fft(out)/fft(im)))
#     print(h_sys_inverse)
    return out,h_sys_inverse


if __name__=='__main__':
    import h5py
    import numpy as np
    f = h5py.File('../ckpt/ckpt_80.h5', 'r')

    input_shape = (1,1,10)
    # input_shape2 = (1,16,10)
    # sig = np.ones(input_shape)
    # np.random.seed(0)
    # sig = np.random.rand(1,1,10)
    # sig2 = np.ones(input_shape2)
    # data: noised signal
    file_name = r'/home/vincent/Documents/Projects/DeepDenoising/Code1/TestSamples/Test1_data.txt'
    with open(file_name,'r') as ftxt:
        lines = ftxt.readlines() 
    ftxt.close()
    sig = np.array([float(x) for x in lines],dtype=np.float32)
    sig = np.expand_dims(np.expand_dims(sig,0),0)
    print(sig)

    
    weight1 = f['inc.double_conv.0.weight']
    bias1 = f['inc.double_conv.0.bias']
    weight2 = f['inc.double_conv.1.weight']
    bias2 = f['inc.double_conv.1.bias']
    mean1 = f['inc.double_conv.1.running_mean']
    var1 = f['inc.double_conv.1.running_var']

    weight3 = f['inc.double_conv.3.weight']
    bias3 = f['inc.double_conv.3.bias']
    weight4 = f['inc.double_conv.4.weight']
    bias4 = f['inc.double_conv.4.bias']
    mean2 = f['inc.double_conv.4.running_mean']
    var2 = f['inc.double_conv.4.running_var']
    

    x1 = _conv1d(sig,weight1,bias1)
    x2 = _batchnorm1d(x1,weight2,bias2,mean1,var1)
    x3 = _relu(x2)
    x4 = _conv1d(x3,weight3,bias3)
    x5 = _batchnorm1d(x4,weight4,bias4,mean2,var2)
    out = _relu(x5)

    print(out.shape)
    print(out)

    print('start pytorch version now')
    print('start pytorch version now')
    print('start pytorch version now \n')

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Tsig = torch.from_numpy(sig).float().to(device)
    # ckpt = torch.load('../ckpt/ckpt_80.pth', map_location=device)
    # tensor_para = ckpt['net_state_dict']
    
    # Tweights1 = tensor_para['inc.double_conv.0.weight']
    # Tbias1 = tensor_para['inc.double_conv.0.bias']
    # Tweights2 = tensor_para['inc.double_conv.1.weight']
    # Tbias2 = tensor_para['inc.double_conv.1.bias']
    # Tmean1 = tensor_para['inc.double_conv.1.running_mean']
    # Tvar1 = tensor_para['inc.double_conv.1.running_var']
    
    # Tweights3 = tensor_para['inc.double_conv.3.weight']
    # Tbias3 = tensor_para['inc.double_conv.3.bias']
    # Tweights4 = tensor_para['inc.double_conv.4.weight']
    # Tbias4 = tensor_para['inc.double_conv.4.bias']
    # Tmean2 = tensor_para['inc.double_conv.4.running_mean']
    # Tvar2 = tensor_para['inc.double_conv.4.running_var']
    
    # net = DoubleConv1D(1,2).to(device)
    # net.eval()

    # net.double_conv[0].weight = torch.nn.Parameter(Tweights1)
    # net.double_conv[0].bias = torch.nn.Parameter(Tbias1)
    # net.double_conv[1].weight = torch.nn.Parameter(Tweights2)
    # net.double_conv[1].bias = torch.nn.Parameter(Tbias2)
    # net.double_conv[1].running_mean = torch.tensor(Tmean1)
    # net.double_conv[1].running_var = torch.tensor(Tvar1)

    # net.double_conv[3].weight = torch.nn.Parameter(Tweights3)
    # net.double_conv[3].bias = torch.nn.Parameter(Tbias3)
    # net.double_conv[4].weight = torch.nn.Parameter(Tweights4)
    # net.double_conv[4].bias = torch.nn.Parameter(Tbias4)
    # net.double_conv[4].running_mean = torch.tensor(Tmean2)
    # net.double_conv[4].running_var = torch.tensor(Tvar2)

    # Tx1 = net.double_conv[0](Tsig)
    # Tx2 = net.double_conv[1](Tx1)
    # Tx3 = net.double_conv[2](Tx2)
    # Tx4 = net.double_conv[3](Tx3)
    # Tx5 = net.double_conv[4](Tx4)
    # out2 = net.double_conv[5](Tx5)
    # print(Tx5.size())
    # print(Tx5)



