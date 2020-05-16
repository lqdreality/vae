#from .params.VAEParams import VAEParams
from .params import VAEParams
from . import losses as l
from .layers import SampleLayer
from .utils.sampling import GaussianPosteriorSample,\
                            ExponentialPosteriorSample

from keras.models import Model
from keras.layers import Input,\
                         Dense,\
                         Lambda,\
                         Reshape,\
                         Concatenate
from keras.optimizers import Adagrad, SGD

import re
import numpy as np

BERGAUSS = 0
GAUSSGAUSS = 1
HOMOSCEDGAUSSGAUSS = 2
POISSONGAUSS = 3
BEREXP = 4

class VAE :
    def __init__(self,
                 params=None,
                 param_file=None) :
        """
        Variational Auto-Encoder Constructor.
        
        Parameters
        ----------
        params: A VAEParams Object
        param_file: A .json file describing the VAE
        """
        
        self.params = None
        self.type = None
        self.model = None
        
        if params is not None :
            self.params = params
            self._set_type()
        elif isinstance(param_file, str) :
            self.load_params(param_file)
        
    def _set_type(self) :
        """
        Sets the type of VAE
        
        Options are:
        Bernoulli-Gaussian
        Homoscedastic-Gaussian-Gaussian
        Heteroscedastic-Gaussian-Gaussian
        Poisson-Gaussian
        Bernoulli-Exponential
        """
        if self.params.vae_type.lower() == 'bernoulligaussian' :
            self.type = BERGAUSS
        elif self.params.vae_type.lower() == 'gaussiangaussian' :
            self.type = GAUSSGAUSS
        elif self.params.vae_type.lower() == 'homoscedasticgaussiangaussian' :
            self.type = HOMOSCEDGAUSSGAUSS
        elif self.params.vae_type.lower() == 'poissongaussian' :
            self.type = POISSONGAUSS
        elif self.params.vae_type.lower() == 'bernoulliexponential' :
            self.type = BEREXP
        else :
            raise
        
    def load_params(self, param_file) :
        """
        Loads Params from an appropriate .json file
        """
        self.params = VAEParams()
        if param_file :
            self.params.load(param_file)
        else :
            print('No Parameter file specified, using default')
        self._set_type()
        
    def get_loss(self, loss_inputs) :
        """
        Returns the appropriate loss given the type
        Private Method
        """
        r = None

        if self.type == BEREXP :
            lambda_kl = loss_inputs[0]
        else :
            mu_dkl = loss_inputs[0]
            log_var_dkl = loss_inputs[1]

        if self.type == BERGAUSS :
            r = l.BernoulliGaussianLoss(mu_dkl, log_var_dkl)
        elif self.type == GAUSSGAUSS :
            r = l.GaussianGaussianLoss(mu_dkl, log_var_dkl)
        elif self.type == HOMOSCEDGAUSSGAUSS :
            if self.params.variance is not None :
                log_var = np.log(self.params.variance).astype('float32')
            else :
                log_var = 0.0
            r = l.GaussianGaussianLoss(mu_dkl, log_var_dkl,
                                     const_var=log_var)
        elif self.type == POISSONGAUSS :
            r = l.PoissonGaussianLoss(mu_dkl, log_var_dkl)
        elif self.type == BEREXP :
            r = l.BernoulliExponentialLoss(lambda_kl)
        else :
            raise
        return r
        
    def construct(self) :
        """
        Constructs the VAE in Keras according the params
        """
        lp = self.params.layer_params
        shape, rest = lp[0].get_keras_params()
        input_layer = Input(shape=shape, **rest)
        layer = input_layer
        
        loss_inputs = []
        concatenate_inputs = []
        for i in range(1, len(lp)) :
            if lp[i].type.lower() == 'densekd' : # Switch between encoding and decoding
                tmp = []
                for k in range(0, lp[i].K) :
                    size, rest = lp[i].layer_list[k].get_keras_params()
                    if lp[i].layer_list[k].reshape is not None :
                        dlayer = Dense(size, **rest)(layer)
                        relayer = Reshape(lp[i].layer_list[k].reshape,
                                          name='Reshape_'+lp[i].layer_list[k].name)(dlayer)
                        tmp.append(relayer)
                    else :
                        tmp.append(Dense(size, **rest)(layer))
                layer = tmp
                if lp[i].to_loss :
                    loss_inputs = tmp
                if lp[i].concat :
                    concatenate_inputs = tmp
            elif lp[i].type.lower() == 'sample' :
                size, rest = lp[i].get_keras_params()
                layer = SampleLayer(size, **rest)(layer)
            elif lp[i].type.lower() == 'concatenation' :
                if len(concatenate_inputs) == 0 :
                    print('Concatenate Layer Requested, but no layer to concat')
                    continue
                rest = lp[i].get_keras_params()
                layer = Concatenate(**rest)(concatenate_inputs)
                concatenate_inputs = []
            else :
                size, rest = lp[i].get_keras_params()
                layer = Dense(size, **rest)(layer)
                if lp[i].to_loss :
                    loss_inputs.append(layer)
                if lp[i].reshape is not None :
                    layer = Reshape(lp[i].reshape, name='Reshape_'+lp[i].name)(layer)
        
        self.model = Model(input_layer, layer)
        
        loss = self.get_loss(loss_inputs)
        
        optimizer = self.construct_optimizer()
        
        self.model.compile(optimizer=optimizer, loss=loss)
        
    def construct_optimizer(self) :
        """
        Constructs the optimizer
        """
        if self.params.optimizer.name == 'Adagrad' :
            optimizer = Adagrad(lr=self.params.optimizer.learning_rate)
        elif self.params.optimizer.name == 'SGD' :
            optimizer = SGD(lr=self.params.optimizer.learning_rate,
                                 momentum=self.params.optimizer.momentum,
                                 nesterov=self.params.optimizer.nesterov)
        else :
            raise
        return optimizer
    
    def fit(self, X, shuffle=False, add_axis=False) :
        """
        Fit the model
        
        Parameters
        ----------
        X: data to fit
        shuffle: Randomly shuffle the dataset
        add_axis: Sometimes if there are more than one output layer an axis should be added
        """
        if add_axis :
            ndim = X.ndim
            if ndim == 2 :
                X_test = X[:,:,np.newaxis]
            elif ndim == 3 :
                X_test = X[:,:,:,np.newaxis]
        else :
            X_test = X.copy()
        self.model.fit(X, X_test,
                       batch_size=self.params.batch_size,
                       epochs=self.params.num_epochs,
                       shuffle=shuffle)
        
    def fit_generator(self, X) :
        pass
        
    def encode(self, X, encoding_layer, skip=[]) :
        """
        Encode X
        
        Parameters
        ----------
        encoding_layer: The name of the output layer of the encode step
        skip: Sometimes layers are linked and need to be skipped
        """
        input_layer = self.model.inputs[0]
        layer = input_layer
        for i in range(1,len(self.model.layers)) :
            if self.model.layers[i].name in skip :
                continue
            l = self.model.layers[i]
            layer = l(layer)
            if self.model.layers[i].name == encoding_layer :
                break
        encoder = Model(input_layer, layer)
        return encoder.predict(X, batch_size=self.params.batch_size)
    
    def decode(self, Z, decoding_layer_start) :
        """
        Decode Z
        
        Parameters
        ----------
        decoding_layer_start: the name of the layer to start the decoding
        """
        input_layer = Input(shape=(Z.shape[1],))
        layer = input_layer
        
        decode_found = False
        for i in range(1,len(self.model.layers)) :
            if self.model.layers[i].name == decoding_layer_start :
                decode_found = True
                
            if decode_found :
                l = self.model.layers[i]
                layer = l(layer)
                
        decoder = Model(input_layer,layer)
        return decoder.predict(Z, batch_size=Z.shape[0])
        
        
        
        
        
        
        
        
        
        
        
