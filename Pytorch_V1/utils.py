import os
import cv2
import math
import numpy as np
import random
import torch
from numpy.fft import fft,ifft
import matplotlib.pyplot as plt
import pdb


_modedict = {'valid': 0, 'same': 1, 'full': 2}

def seed_everything(seed:int):
    '''

    Args:
        seed: int
    '''
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    torch.manual_seed(seed)            # fix torch seed on CPU
    torch.cuda.manual_seed(seed)       # fix torch seed on GPU
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = True

def wiener_filter(im,mysize=None, noise=None):
    '''
    A durable implement of wiener filter and calcualte the system function
    '''
    im = np.asarray(im)
    if mysize is None:
        mysize = [3] * im.ndim #kernel size = 3
#     print('im',im)
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
    res *= (1 - noise / (lVar+1e-6))
#     print('2',res)
#     print(noise/lVar)
    res += lMean
#     print('3',res)
    out = np.where(lVar < noise, lMean, res)
#     print('4',out)
    h_sys_inverse = np.abs(ifft(fft(out)/fft(im)))
#     print(h_sys_inverse)
    return out,h_sys_inverse


def figure_plot(sig,label,pred):
    assert sig.shape==label.shape==pred.shape, 'Incompetible inputs and predictions'
    fig = plt.figure(figsize=(50,30))
    ax1 = fig.add_subplot(311)
    ax1.set_title('Noise-free data',fontsize=70)
    ax1.plot(label)
    ax2 = fig.add_subplot(312)
    ax2.plot(sig)
    ax2.set_title('Noisy data',fontsize=70)
    ax3 = fig.add_subplot(313)
    ax3.set_title('Denoised data',fontsize=70)
    ax3.plot(pred)
    # ax3.text(2,0.5,'RMSE %0.2f'%rmse(pred-label),fontsize=40)
    # ax3.text(2,1.5,'PSNR %0.2f'%psnr(pred-label),fontsize=40)
    ax3.annotate('PSNR %0.2f'%psnr(pred-label),xy=(300,1800),fontsize=50,xycoords='figure points')
    ax3.annotate('RMSE %0.2f'%rmse(pred-label),xy=(300,1850),fontsize=50,xycoords='figure points')
    return fig

def rmse (x): return np.sqrt( np.mean( np.abs(x)**2 ) )

def psnr (x): return 20 * np.log10(10 /np.mean( np.abs(x)**2))

def _np_conv_ok(volume, kernel, mode):
    if volume.ndim == kernel.ndim == 1:
        if mode in ('full', 'valid'):
            return True
        elif mode == 'same':
            return volume.size >= kernel.size
    else:
        return False

def _reverse_and_conj(x):
    """
    Reverse array `x` in all dimensions and perform the complex conjugate
    """
    reverse = (slice(None, None, -1),) * x.ndim
    return x[reverse].conj()

def _np_conv_ok(volume, kernel, mode):
    if volume.ndim == kernel.ndim == 1:
        if mode in ('full', 'valid'):
            return True
        elif mode == 'same':
            return volume.size >= kernel.size
    else:
        return False

def _inputs_swap_needed(mode, shape1, shape2, axes=None):
    if mode != 'valid':
        return False

    if not shape1:
        return False

    if axes is None:
        axes = range(len(shape1))

    ok1 = all(shape1[i] >= shape2[i] for i in axes)
    ok2 = all(shape2[i] >= shape1[i] for i in axes)

    if not (ok1 or ok2):
        raise ValueError("For 'valid' mode, one must be at least "
                         "as large as the other in every dimension")

    return not ok1

def correlate(in1, in2, mode='full', method='auto'):
    in1 = np.asarray(in1)
    in2 = np.asarray(in2)

    if in1.ndim == in2.ndim == 0:
        return in1 * in2.conj()
    elif in1.ndim != in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")

    # Don't use _valfrommode, since correlate should not accept numeric modes
    try:
        val = _modedict[mode]
    except KeyError as e:
        raise ValueError("Acceptable mode flags are 'valid',"
                         " 'same', or 'full'.") from e

    # this either calls fftconvolve or this function with method=='direct'
    if method in ('fft', 'auto'):
        return convolve(in1, _reverse_and_conj(in2), mode, method)

    elif method == 'direct':
        # fastpath to faster numpy.correlate for 1d inputs when possible
        if _np_conv_ok(in1, in2, mode):
            return np.correlate(in1, in2, mode)

        swapped_inputs = ((mode == 'full') and (in2.size > in1.size) or
                          _inputs_swap_needed(mode, in1.shape, in2.shape))

        if swapped_inputs:
            in1, in2 = in2, in1

        if mode == 'valid':
            ps = [i - j + 1 for i, j in zip(in1.shape, in2.shape)]
            out = np.empty(ps, in1.dtype)

            z = sigtools._correlateND(in1, in2, out, val)

        else:
            ps = [i + j - 1 for i, j in zip(in1.shape, in2.shape)]

            # zero pad input
            in1zpadded = np.zeros(ps, in1.dtype)
            sc = tuple(slice(0, i) for i in in1.shape)
            in1zpadded[sc] = in1.copy()

            if mode == 'full':
                out = np.empty(ps, in1.dtype)
            elif mode == 'same':
                out = np.empty(in1.shape, in1.dtype)

            z = sigtools._correlateND(in1zpadded, in2, out, val)

        if swapped_inputs:
            # Reverse and conjugate to undo the effect of swapping inputs
            z = _reverse_and_conj(z)

        return z

    else:
        raise ValueError("Acceptable method flags are 'auto',"
                         " 'direct', or 'fft'.")


def convolve(in1, in2, mode='full', method='auto'):
    volume = np.asarray(in1)
    kernel = np.asarray(in2)

    if volume.ndim == kernel.ndim == 0:
        return volume * kernel
    elif volume.ndim != kernel.ndim:
        raise ValueError("volume and kernel should have the same "
                         "dimensionality")

    if _inputs_swap_needed(mode, volume.shape, kernel.shape):
        # Convolution is commutative; order doesn't have any effect on output
        volume, kernel = kernel, volume

    if method == 'auto':
        method = choose_conv_method(volume, kernel, mode=mode)

    if method == 'fft':
        out = fftconvolve(volume, kernel, mode=mode)
        result_type = np.result_type(volume, kernel)
        if result_type.kind in {'u', 'i'}:
            out = np.around(out)
        return out.astype(result_type)
    elif method == 'direct':
        # fastpath to faster numpy.convolve for 1d inputs when possible
        if _np_conv_ok(volume, kernel, mode):
            return np.convolve(volume, kernel, mode)

        return correlate(volume, _reverse_and_conj(kernel), mode, 'direct')
    else:
        raise ValueError("Acceptable method flags are 'auto',"
                         " 'direct', or 'fft'.")


def choose_conv_method(in1, in2, mode='full', measure=False):
    volume = np.asarray(in1)
    kernel = np.asarray(in2)

    if measure:
        times = {}
        for method in ['fft', 'direct']:
            times[method] = _timeit_fast(lambda: convolve(volume, kernel,
                                         mode=mode, method=method))

        chosen_method = 'fft' if times['fft'] < times['direct'] else 'direct'
        return chosen_method, times

    # for integer input,
    # catch when more precision required than float provides (representing an
    # integer as float can lose precision in fftconvolve if larger than 2**52)
    if any([_numeric_arrays([x], kinds='ui') for x in [volume, kernel]]):
        max_value = int(np.abs(volume).max()) * int(np.abs(kernel).max())
        max_value *= int(min(volume.size, kernel.size))
        if max_value > 2**np.finfo('float').nmant - 1:
            return 'direct'

    if _numeric_arrays([volume, kernel], kinds='b'):
        return 'direct'

    if _numeric_arrays([volume, kernel]):
        if _fftconv_faster(volume, kernel, mode):
            return 'fft'

    return 'direct'

def _numeric_arrays(arrays, kinds='buifc'):
    if type(arrays) == np.ndarray:
        return arrays.dtype.kind in kinds
    for array_ in arrays:
        if array_.dtype.kind not in kinds:
            return False
    return True


def _conv_ops(x_shape, h_shape, mode):
    if mode == "full":
        out_shape = [n + k - 1 for n, k in zip(x_shape, h_shape)]
    elif mode == "valid":
        out_shape = [abs(n - k) + 1 for n, k in zip(x_shape, h_shape)]
    elif mode == "same":
        out_shape = x_shape
    else:
        raise ValueError("Acceptable mode flags are 'valid',"
                         " 'same', or 'full', not mode={}".format(mode))

    s1, s2 = x_shape, h_shape
    if len(x_shape) == 1:
        s1, s2 = s1[0], s2[0]
        if mode == "full":
            direct_ops = s1 * s2
        elif mode == "valid":
            direct_ops = (s2 - s1 + 1) * s1 if s2 >= s1 else (s1 - s2 + 1) * s2
        elif mode == "same":
            direct_ops = (s1 * s2 if s1 < s2 else
                          s1 * s2 - (s2 // 2) * ((s2 + 1) // 2))
    else:
        if mode == "full":
            direct_ops = min(_prod(s1), _prod(s2)) * _prod(out_shape)
        elif mode == "valid":
            direct_ops = min(_prod(s1), _prod(s2)) * _prod(out_shape)
        elif mode == "same":
            direct_ops = _prod(s1) * _prod(s2)

    full_out_shape = [n + k - 1 for n, k in zip(x_shape, h_shape)]
    N = _prod(full_out_shape)
    fft_ops = 3 * N * np.log(N)  # 3 separate FFTs of size full_out_shape
    return fft_ops, direct_ops


def _fftconv_faster(x, h, mode):
    fft_ops, direct_ops = _conv_ops(x.shape, h.shape, mode)
    offset = -1e-3 if x.ndim == 1 else -1e-4
    constants = {
            "valid": (1.89095737e-9, 2.1364985e-10, offset),
            "full": (1.7649070e-9, 2.1414831e-10, offset),
            "same": (3.2646654e-9, 2.8478277e-10, offset)
            if h.size <= x.size
            else (3.21635404e-9, 1.1773253e-8, -1e-5),
    } if x.ndim == 1 else {
            "valid": (1.85927e-9, 2.11242e-8, offset),
            "full": (1.99817e-9, 1.66174e-8, offset),
            "same": (2.04735e-9, 1.55367e-8, offset),
    }
    O_fft, O_direct, O_offset = constants[mode]
    return O_fft * fft_ops < O_direct * direct_ops + O_offset

def _prod(iterable):
    product = 1
    for x in iterable:
        product *= x
    return product



if __name__=='__main__':
    import matplotlib.pyplot as plt
    sig = np.repeat([0., 1., 1., 0., 1., 0., 0., 1.], 128)
    sig_noise = sig + np.random.randn(len(sig))
    corr = correlate(sig_noise, np.ones(128), mode='same') / 128
    print(corr.shape)

    from scipy import signal
    sig = np.repeat([0., 1., 0.], 100)
    win = signal.windows.hann(50)
    filtered = convolve(sig, win, mode='same') / sum(win)
    print(filtered.shape)