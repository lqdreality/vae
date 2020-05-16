from .OptimizerParams import AdagradParams, SGDParams
from .LayerParams import InputLayerParams,\
                        DenseLayerParams,\
                        DenseKDLayerParams,\
                        SampleLayerParams,\
                        ConcatenationLayerParams
import json
from collections import OrderedDict
        
class VAEParams :
    def __init__(self, 
                 layer_params=[],
                 batch_size=10,
                 num_epochs=10,
                 num_samples=100,
                 opt=None,
                 vae_type='BernoulliGaussian') :
        """
        VAE Params Constructor
        
        This class contains the entire parameterization for a VAE
        
        Parameters
        ----------
        batch_size: Size of each batch to be trained
        num_epochs: The number of epochs to use with training
        num_samples: The number of samples to be considered at one time
        vae_type: the distribution of X|z and Q(z)
        """
        self.layer_params = layer_params
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_samples = num_samples
        self.optimizer = opt
        self.vae_type = vae_type
        self.variance = None
        
    def __str__(self) :
        s = '\n'
        s += 'Batch Size: ' + str(self.batch_size) + '\n'
        s += 'Num Epochs: ' + str(self.num_epochs) + '\n'
        s += 'Number of Samples: ' + str(self.num_samples) + '\n\n'
        s += 'VAE Type: ' + self.vae_type + '\n\n'
        s += self.optimizer.__str__() +'\n\n'
        for p in self.layer_params :
            s += p.__str__() + '\n\n'
        return s
        
    def load(self, param_file) :
        """
        Loads parameters from the appropriate .json parameter file
        
        Parameters
        ----------
        param_file: file name/location of .json file
        """
        f = open(param_file, 'r')
        j = json.loads(f.read(), object_pairs_hook=OrderedDict)
        f.close()
        
        self.variance = j.pop('Variance', None)
        
        batch_size = j.pop('Batch Size', None)
        if batch_size is None :
            print('No Batch Size appearing in ' + param_file +\
                  ' setting Batch Size = ' + str(self.batch_size))
        else :
            self.batch_size = batch_size

        num_epochs = j.pop('Num Epochs', None)
        if num_epochs is None :
            print('No Num Epochs appearing in ' + param_file +\
                  ' setting Num Epochs = ' + str(self.num_epochs))
        else :
            self.num_epochs = num_epochs

        num_samples = j.pop('Num Samples', None)
        if num_samples is None :
            print('No Num Samples appearing in ' + param_file +\
                  ' setting Num Samples = ' + str(self.num_samples))
        else :
            self.num_samples = num_samples
        
        # Get auto-encoder type
        vae_type = j.pop('VAE Type', None)
        if vae_type is None :
            print('No VAE Type appearing in ' + param_file +\
                  ' setting VAE Type = ' + self.vae_type)
        else :
            self.vae_type = vae_type
        
        # Get input size
        input_size = j.pop('Input Size', None)
        
        # Get Optimizer Params
        opt = j.pop('Optimizer', None)
        if opt is None :
            print('No Optimizer appearing in ' + param_file +\
                  ' setting Optimizer to = ' + self.optimizer.__str__())
            self.optimizer = AdagradParams(learning_rate=1e-3)
        elif opt['type'].lower() == 'adagrad' :
            self.optimizer = AdagradParams(**opt['meta'])
        elif opt['type'].lower() == 'sgd' :
            self.optimizer = SGDParams(**opt['meta'])
        else :
            raise
        
        if input_size is None :
            raise
        elif isinstance(input_size, list) :
            self.layer_params.append(InputLayerParams(shape=tuple(input_size)))
        else :
            self.layer_params.append(InputLayerParams(shape=(input_size,)))
        
        keys = j.keys()
        for k in keys :
            type = j[k].get('type', None)
            if type is not None :
                if type.lower() == 'dense' :
                    self.layer_params.append(DenseLayerParams(**j[k]['meta']))
                elif type.lower() == 'sample' :
                    self.layer_params.append(SampleLayerParams(**j[k]['meta']))
                elif type.lower() == 'concatenation' :
                    self.layer_params.append(ConcatenationLayerParams(**j[k]['meta']))
                elif type.lower() == 'densekd' :
                    in_dict = j[k]['meta'].copy()
                    name = in_dict.pop('name', None)
                    K = in_dict.pop('K', 1)
                    to_loss = in_dict.pop('to_loss', False)
                    concat = in_dict.pop('concat', False)
                    self.layer_params.append(DenseKDLayerParams(in_dict, 
                                                                name=name, 
                                                                to_loss=to_loss,
                                                                K=K,
                                                                concat=concat))
            else :
                self.layer_params.append(DenseLayerParams(**j[k]['meta']))
            
