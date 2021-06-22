# srls: Sparse-Regularized Least Squares

# Ivan Selesnick
# NYU Tandon
# selesi@nyu.edu
# March 2016 (Matlab)
# May 2017 (Python)

import numpy as np

# soft-threshold funciton
def soft(x, T):
    EPS = 1e-10
    return np.maximum(1 - T/(np.abs(x) + EPS), 0) * x
    # return np.maximum(1 - T/np.abs(x), 0) * x


def truncate (x,M): return x[:M]
def AH (x): return np.fft.fft(x, n = len(x)) / np.sqrt(len(x))
def A (x): return truncate(np.fft.ifft(x) * np.sqrt(len(x)),100)


def srls_L1(y, A, AH, rho, lam):
    """
    srls_L1: Sparse-Regularized Least Squares with L1 norm penalty

    Minimize_x ||y - A x||_2^2 + lam ||x||_1

    INPUT
    y      : data
    A, AH  : function handles for A and its conjugate transpose
    rho    : rho >= maximum eigenvalue of A'A
    lam    : regularization parameter, lam > 0

    Motified from Prof. Ivan Selesnick's version
    """



    MAX_ITER = 10000
    TOL_STOP = 1e-4

    cost = np.zeros(MAX_ITER)       # cost function history

    mu = 1.9 / rho

    # Initialization
    AHy = AH(y)                # A'*y
    x = AH(np.zeros(np.shape(y)))
    Ax = A(x)

    iter = 0
    old_x = x

    delta_x = np.inf

    while (delta_x > TOL_STOP) and (iter < MAX_ITER):
        z = x - mu * ( AH(Ax) - AHy )
        x = soft(z, lam * mu)
        Ax = A(x)

        # cost function history
        residual = y - Ax
        cost[iter] = 0.5 * np.sum(np.abs(residual)**2) + lam * np.sum(np.abs(x))

        delta_x = np.max(np.abs( x - old_x )) / np.max(np.abs(x))
        old_x = x

        iter = iter + 1
    return x


def srls_GMC(y, A, AH, rho, lam, gamma):
    """
    x = srls_GMC(y, A, AH, rho, lam, gamma)

    srls_GMC: Sparse-Regularized Least Squares with generalized MC (GMC) penalty

    argmin_x  argmax_v { F(x,v) =
        1/2 ||y - A x||^2 + lam ||x||_1 - gamma/2 ||A(x-v)||_2^2 - lam ||v||_1 }

    INPUT
        y       data
        A, AH   operators for A and A^H
        rho     rho >= maximum eigenvalue of A^H A
        lam     regularization parameter, lam > 0
        gamma   0 <= gamma < 1

    Motified from Prof. Ivan Selesnick's version
    """

    MAX_ITER = 10000
    TOL_STOP = 1e-4

    mu = 1.9 / ( rho * max( 1.0,  gamma / (1.0-gamma) ) )

    # Initialization
    AHy = AH(y)
    x = AH(np.zeros(np.shape(y)))
    v = AH(np.zeros(np.shape(y)))
    Ax = A(x)

    iter = 0
    old_x = x

    delta_x = np.inf

    while (delta_x > TOL_STOP) and (iter < MAX_ITER):
        # update x
        zx = x - mu * ( AH(A(x + gamma*(v-x))) - AHy );
        zv = v - mu * ( gamma * AH(A(v-x)) );
    
        # update v
        x = soft(zx, mu * lam);
        v = soft(zv, mu * lam);

        delta_x = np.max(np.abs( x - old_x )) / np.max(np.abs(x))
        old_x = x

        iter = iter + 1
    return x


def process_L1(data,lam_L1=1.0):
    c_L1 = srls_L1(data, A, AH, 1, lam_L1)
    data_L1 = A(c_L1)
    return data_L1

def process_GMC(data,lam_GMC = 3.0,gamma = 0.8):
    c_GMC = srls_GMC(data, A, AH, 1, lam_GMC, gamma)
    data_GMC = A(c_GMC)
    return data_GMC
