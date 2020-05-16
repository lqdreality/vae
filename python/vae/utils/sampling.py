import keras.backend as K

def GaussianPosteriorSample(bs, ls) :
    """
    Defines the Reparameterization Trick for a Gaussian Latent Variable
    z = g(e) = mu + sigma*e, e ~ N(0,1)
    
    Parameters
    ----------
    bs: Batch Size
    ls: Latent Size, i.e., the dimensionality of the latent space
    """
    def gps(args) :
        mu, log_var = args
        eps = K.random_normal(shape=(bs, ls), mean=0.0, stddev=1.0) # 10 x 2
        return mu + K.exp(log_var / 2.) * eps
    return gps

def ExponentialPosteriorSample(bs, ls) :
    """
    Defines the Reparameterization Trick for an Exponential Latent Variable
    z = g(e) = (1/lambda)*log(-e+1), e ~ Uni(0,1)
    
    Parameters
    ----------
    bs: Batch Size
    ls: Latent Size, i.e., the dimensionality of the latent space
    """
    def exps(args) :
        lamb = args
        eps = K.random_uniform(shape=(bs, ls))
        ans = (-1./lamb) * K.log(-eps + 1)
        return ans
    return exps
