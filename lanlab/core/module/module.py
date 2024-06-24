class Module:
    def run(self,struct):
        raise NotImplementedError
    def __call__(self,struct,*args,**kwargs):
        return self.run(struct,*args,**kwargs)
     