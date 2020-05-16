class InitializerParams :
    def __init__(self, name=None) :
        self.name = name

class ConstantInitializerParams(InitializerParams) :
    def __init__(self, **kwargs) :
        InitializerParams.__init__(self, name='Constant')
        self.val = kwargs.get('val', 0.0)

    def __str__(self) :
        s = ''
        s += 'Initializer: ' + self.name + '\n'
        s += '>  Value: ' + str(self.val) + '\n'
        return s

    def __repr__(self) :
        return self.__str__()

    def __get__(self, key) :
        #

class RandomNormalInitializerParams(InitializerParams) :
    def __init__(self, **kwargs) :
        InitializerParams.__init__(self, name='Random_Normal')
        self.mean = kwargs.get('mean', 0.0)
        self.stddev = kwargs.get('stddev', 0.05)
        self.seed = kwargs.get('seed', None)

    def __str__(self) :
        s = ''
        s += 'Initializer: ' + self.name + '\n'
        s += '>  Mean: ' + str(self.mean) + '\n'
        s += '>  Stddev: ' + str(self.stddev) + '\n'
        if self.seed is not None :
            s += '>  Seed: ' + str(self.seed) + '\n'
        else :
            s += '>  Seed: None\n'
        return s
