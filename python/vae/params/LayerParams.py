class LayerParams :
    def __init__(self, name=None, type=None, to_loss=False) :
        self.name = name
        self.type = type
        self.to_loss = to_loss
        
    def __str__(self) :
        s = ''
        s += 'Name: ' + self.name + '\n'
        s += 'Type: ' + self.type + '\n'
        s += 'Output to Loss: ' + str(self.to_loss) + '\n'
        return s

class InputLayerParams(LayerParams) :
    def __init__(self, **kwargs) :
        LayerParams.__init__(self, 
                             name=kwargs.get('name', 'Input'),
                             type='Input',
                             to_loss=kwargs.get('to_loss', False))
        self.shape = kwargs.get('shape', None)
        
    def __str__(self) :
        s = ''
        s += LayerParams.__str__(self)
        s += 'Shape: ' + str(self.shape) + '\n\n'
        return s
        
    def get_keras_params(self) :
        d = {'name': self.name}
        return (self.shape, d)
    
class SampleLayerParams(LayerParams) :
    def __init__(self, **kwargs) :
        LayerParams.__init__(self, 
                             name=kwargs.get('name', None),
                             type='Sample',
                             to_loss=kwargs.get('to_loss', False))
        self.size = kwargs.get('size', None)
        self.distribution = kwargs.get('distribution', None)
        
    def __str__(self) :
        s = ''
        s += LayerParams.__str__(self)
        s += 'Size: ' + str(self.size) + '\n'
        s += 'Distribution: ' + self.distribution + '\n\n'
        return s
    
    def get_keras_params(self) :
        d = {'distribution': self.distribution, 'name': self.name}
        return (self.size, d)

class ConcatenationLayerParams(LayerParams) :
    def __init__(self, **kwargs) :
        LayerParams.__init__(self,
                             name=kwargs.get('name', None),
                             type='Concatenation',
                             to_loss=kwargs.get('to_loss', False))
        self.axis = kwargs.get('axis', -1)

    def __str__(self) :
        s = ''
        s += LayerParams.__str__(self)
        s += 'Axis: ' + str(self.axis) + '\n\n'
        return s

    def get_keras_params(self) :
        d = {'axis': self.axis, 'name': self.name}
        return d
        
class DenseLayerParams(LayerParams) :
    def __init__(self, **kwargs) :
        LayerParams.__init__(self, 
                             name=kwargs.get('name', None),
                             type='Dense',
                             to_loss=kwargs.get('to_loss', False))
        self.size = kwargs.get('size', None)
        self.activation = kwargs.get('activation', 'relu')
        self.reshape = kwargs.get('reshape', None)
        if self.reshape is not None :
            self.reshape = tuple(self.reshape)
        
    def __str__(self) :
        s = ''
        s += LayerParams.__str__(self)
        s += 'Size: ' + str(self.size) + '\n'
        s += 'Activation: ' + self.activation + '\n'
        if self.reshape is None :
            s += 'Reshape: None \n\n'
        else :
            s += 'Reshape: ' + str(self.reshape) + '\n\n'
        return s
        
    def get_keras_params(self) :
        d = {'activation': self.activation, 'name': self.name}
        return (self.size, d)
    
class DenseKDLayerParams(LayerParams) :
    def __init__(self, in_dict, name=None, to_loss=False, K=1, concat=False) :
        LayerParams.__init__(self, 
                             name=name,
                             type='DenseKD',
                             to_loss=to_loss)
        self.K = K
        self.concat = concat
        self.layer_list = []
        for i in range(0, self.K) :
            self.layer_list.append(DenseLayerParams(**in_dict[str(i)]))
            
    def __str__(self) :
        s = ''
        s += LayerParams.__str__(self)
        s += 'K: ' + str(self.K) + '\n'
        s += 'Concat: ' + str(self.concat) + '\n\n'
        for i in range(0, self.K) :
            s += '-----\n'
            s += self.layer_list[i].__str__()
            s += '-----\n\n'
        return s
