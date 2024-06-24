from lanlab.core.module import Module
from lanlab.core.structure.segment import Segment
from lanlab.file.loader import load,save
from lanlab.core.structure.structure import Structure

import copy
import os

class Sequential(Module):
    def __init__(self,*modules):
        super().__init__()
        self.list_modules = list(modules)

        #Verify all elements are modules and create a segment if it is a string
        for i in range(len(self.list_modules)):
            if isinstance(self.list_modules[i],str):
                self.list_modules[i] = Segment({'text':self.list_modules[i],"origin":"user"})
            elif not(isinstance(self.list_modules[i],Module)) and not isinstance(self.list_modules[i],Structure):
                raise TypeError("Sequential modules must be either strings,  modules or Structures")

    def run(self,struct,path=None):
        if path is not None:
            if os.path.exists(path+'.npy'):
                return load(path)
            else:
                out = self._run(struct)
                out.save(path)
                return out
        else:
            return self._run(struct)
        
    def _run(self,struct):
        print('run')
        data = copy.deepcopy(struct)
        for module in self.list_modules:
            data = module(data)
        return data