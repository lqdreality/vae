import keras.backend as K
from keras import metrics
import numpy as np


def GaussianGaussianLoss(mu_kl, log_var_kl, const_var=None) :
    """
    -L(Q) when Q is Gaussian and log P(X|z) is Gaussian
    
    Has a homoscedastic mode or a heteroscedastic mode
    
    Assumes P(z) ~ N(0,I)
    
    Parameters
    ----------
    mu_kl: mean parameters of Q(z)
    log_var_kl: log variance of Q(z)
    const_var: Controls whether z is homoscedastic or heteroscedastic
    """
    if const_var is None : # Heteroscedastic
        def ggl(x, mu_log_var) :
            N = K.int_shape(mu_log_var)[1]
            mu = mu_log_var[:,:,0]
            log_var = mu_log_var[:,:,1]
            mu = mu[:,:,np.newaxis]
            log_var = log_var[:,:,np.newaxis]
            recon = -1.*K.sum(-0.5*log_var - 0.5*K.exp(-1.*log_var)*K.square(x - mu), axis=1)
            dkl = -0.5 * K.sum(-K.exp(log_var_kl) - K.square(mu_kl) + 1. + log_var_kl, axis=-1)
            return dkl
        return ggl
    else : # Homoscedastic
        const_var = float(const_var)
        def ggl(x, mu) :
            recon = -1.*K.sum(-0.5*const_var - 0.5*K.exp(-1.*const_var)*K.square(x - mu), axis=1)
            dkl = -0.5 * K.sum(-K.exp(log_var_kl) - K.square(mu_kl) + 1. + log_var_kl, axis=-1)
            return dkl + recon
        return ggl

def PoissonGaussianLoss(mu_kl, log_var_kl) :
    """
    -L(Q) when Q is Gaussian and log P(X|z) is Poisson
    Assumes P(z) ~ N(0,I)
    
    Parameters
    ----------
    mu_kl: mean parameters of Q(z)
    log_var_kl: log variance of Q(z)
    """
    def pgl(x, lambdas) :
        N = K.int_shape(lambdas)[1]
        recon = -1.*K.sum(x*lambdas - K.exp(lambdas), axis=1)
        dkl = -0.5 * K.sum(-K.exp(log_var_kl) - K.square(mu_kl) + 1. + log_var_kl, axis=-1)
        return recon + dkl
    return pgl

def BernoulliGaussianLoss(mu_kl, log_var_kl) :
    """
    -L(Q) when Q is Gaussian and log P(X|z) is Bernoulli
    Assumes P(z) ~ N(0,I)
    
    Parameters
    ----------
    mu_kl: mean parameters of Q(z)
    log_var_kl: log variance of Q(z)
    """
    def bgl(x, p) :
        N = K.int_shape(p)[1]
        recon = N*metrics.binary_crossentropy(x, p)
        dkl = -0.5 * K.sum(-K.exp(log_var_kl) - K.square(mu_kl) + 1. + log_var_kl, axis=-1)
        return dkl + recon
    return bgl

def BernoulliExponentialLoss(lamb) :
    """
    -L(Q) when Q is exponential and log P(X|z) is Bernoulli
    Assumes P(z) ~ Exp(1)
    
    Parameters
    ----------
    lambda: parameters of Q(z)
    """
    def bexl(x, p) :
        N = K.int_shape(p)[1]
        recon = N*metrics.binary_crossentropy(x, p)
        dkl = K.sum((-1./lamb) + K.log(lamb) - 1, axis=-1)
        return recon+dkl
    return bexl


