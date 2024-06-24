class Configurable:
    """ Object related to a configuration """
    def __init__(self):
        self.config = self.config_class()
    @property
    def config_class(self):
        raise NotImplementedError
    def set_config(self,config):
        self.config = config
    def __getitem__(self,k):
        if isinstance(k,str):
            return self.config[k]
    def set(self,k,v):
        if isinstance(k,str):
            self.config[k] = v
    def __setitem__(self,k,v):
        self.set(k,v)
    def __str__(self):
        return str(self.__class__)+str(self.config)
    def __repr__(self):
        return str(self)
        
    def keys(self):
        return self.config.keys()