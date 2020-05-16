import numpy as np

"""
The following class implements the EM algorith for Multinomial Mixture Model
termed MultinoullioMixture
"""

class MultinoulliMixture :
    def __init__(self,
                 num_latent=10) :
        self.K = num_latent # number of latent variables
        self.pk = np.ones(self.K,)/self.K # will be K x 1
        self.mu = None # will be K x D
        self.eps = np.finfo('float64').eps

    def fit(self, X, num_iters=100) :
        N,D = X.shape
        if self.mu is None :
            print 'Initializing mu'
            self.mu = 0.5*np.random.rand(self.K, D)+0.25 # K x D
        pxn = np.zeros((self.K, N)) # K x N
        
        for i in range(0, num_iters) :
            for k in range(0, self.K) :
                tmp = X*self.mu[k,:] + np.logical_not(X)*(1-self.mu[k,:]) # N x D
                if np.any(tmp == 0) :
                    pass
                pxn[k,:] = np.prod(tmp, axis=1) # p(x|mu), N x 1
                if np.any(pxn[k,:] == 0) :
                    pass
            pxn[pxn == 0] = self.eps
            pxz = (pxn.transpose()*self.pk).transpose() # p(x|mu)p(k), K x N
            if np.any(pxz == 0) :
                pass
                pxz[pxz == 0] = self.eps
            gamma = pxz/np.sum(pxz, axis=0) # K x N
            nks = np.sum(gamma, axis=1) # K x 1
            if np.sum(nks == 0) > 0 :
                pass
            self.mu = (gamma.dot(X).transpose()/nks).transpose() # K x D
            self.pk = nks/N # K x 1