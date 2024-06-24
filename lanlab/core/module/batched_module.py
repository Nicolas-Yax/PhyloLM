from lanlab.core.module.module import Module
from lanlab.core.structure.batch import Batch

class BatchedModule(Module):
    def run(self,struct):
        if isinstance(struct,Batch):
            return struct + struct.map(self._run)
        else:
            return self._run(struct)