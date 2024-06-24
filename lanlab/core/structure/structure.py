class Structure:
    def __call__(self,struct):
        return struct+self
    def save(self,path):
        from lanlab.file.loader import save #To avoid circular import
        return save(self,path)