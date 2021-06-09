# GMC_demo
# 
# Demonstration of the generalized MC (GMC) penalty for sparse-regularized
# linear least squares. The GMC penalty is designed to maintain the convexity of
# the cost function to be minimized. In this example, sparse regularization
# is used to estimate a signal (with frequency-domain sparsity) from a
# noisy observation.
#  
# Demonstration software for the paper:
# I. Selesnick, 'Sparse Regularization via Convex Analysis'
# IEEE Transactions on Signal Processing, 2017.

# Ivan Selesnick
# May 2016 - MATLAB version
# May 2017 - Python version

import matplotlib
matplotlib.use('Agg')

# import numerical array library
import numpy as np
from numpy.random import randn, randint


# import sparse-regularized least squares functions
from models.srls import process_L1,process_GMC


# Define root-mean-square-error (RMSE)
def rmse (x): return np.sqrt( np.mean( np.abs(x)**2 ) )
def psnr (x): return 20 * np.log10(10 /np.mean( np.abs(x)**2))


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



label,data = Single_Time_disorder_data_creator(3)

data_L1 = process_L1(data)

data_GMC = process_GMC(data)

print(' RMSE (L1 ) = %f' % rmse(data_L1 - label))
print(' RMSE (GMC) = %f' % rmse(data_GMC - label))
