from .utils.sampling import GaussianPosteriorSample,\
                            ExponentialPosteriorSample

import keras.backend as K
from keras.layers import Lambda

class SampleLayer :
    def __init__(self,
                 size,
                 name=None,
                 distribution=None) :
        """ Constructs a new layer, the sample layer which will act as the Monte Carlo
        Expectation estimation
        """
        self.size = size
        self.name = name
        self.distribution = distribution
    
    def __call__(self, inputs) :
        if self.distribution.lower() == 'gaussian' :
            batch_size = K.shape(inputs[0])[0]
            sample_foo = GaussianPosteriorSample(batch_size, self.size)
        elif self.distribution.lower() == 'exponential' :
            batch_size = K.shape(inputs)[0]
            sample_foo = ExponentialPosteriorSample(batch_size, self.size)
        else :
            raise
        r = Lambda(sample_foo, name=self.name, output_shape=(self.size,))(inputs)
        return r
