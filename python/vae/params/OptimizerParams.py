class OptimizerParams :                                                         
    def __init__(self, name=None) :                                             
        self.name = name                                                        
                                                                                
class SGDParams(OptimizerParams) :                                              
    def __init__(self, **kwargs) :                                              
        OptimizerParams.__init__(self, name='SGD')                              
        self.learning_rate = kwargs.get('learning_rate', 10e-4)                 
        self.momentum = kwargs.get('momentum', .9)                              
        self.nesterov = kwargs.get('nesterov', False)                           
        self.decay = kwargs.get('decay', 0.0)
                                                                                
    def __str__(self) :                                                         
        s = ''                                                                  
        s += 'Optimizer: ' + self.name + '\n'                                   
        s += 'Learning Rate: ' + str(self.learning_rate) + '\n'                 
        s += 'Momentum: ' + str(self.momentum) + '\n'                           
        s += 'Nesterov: ' + str(self.nesterov) + '\n'                            
        s += 'Decay: ' + str(self.decay) + '\n'
        return s                                                                

    def __repr__(self) :
        return self.__str__()
                                                                                
class AdagradParams(OptimizerParams) :                                          
    def __init__(self, **kwargs) :                                              
        OptimizerParams.__init__(self,name='Adagrad')                           
        self.learning_rate = kwargs.get('learning_rate', 10e-4)                 
                                                                                
    def __str__(self) :                                                         
        s = ''                                                                  
        s += 'Optimizer: ' + self.name + '\n'                                   
        s += 'Learning Rate: ' + str(self.learning_rate) + '\n'                 
        return s

    def __repr__(self) :
        return self.__str__()
