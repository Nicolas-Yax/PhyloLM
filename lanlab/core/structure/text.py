from lanlab.core.structure.sequence import Sequence
from lanlab.core.structure.segment import Segment

class Text(Sequence):
    def __init__(self,text):
        seg = Segment({'text':text,"origin":"user"})
        super().__init__(l=[seg])