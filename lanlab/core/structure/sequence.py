from lanlab.core.structure.segment import Segment
from lanlab.core.structure.structure import Structure

class Sequence(Structure):
    """A sequence is a chain of segments (a batch of segment objects) that represent a dialog with a language model """
    def __init__(self,l=None):
        if l is None:
            l = []
        #Verify that all elements in the list are segments or loadable as Segments
        ll = []
        for k in l:
            if not(isinstance(k,Segment)):
                ll.append(Segment(k))
            else:
                ll.append(k)
        self.l = ll

        self.serialize = self.to_dict
    
    def __add__(self,s1):
        """ Returns a new sequence that is the concatenation of two sequences or a sequence and a segment"""
        s = Sequence(l=self.l)
        s += s1
        return s

    def __iadd__(self,s):
        """ Adds a sequence or a segment to the current sequence"""
        if isinstance(s,Sequence):
            self.l += s.l
        elif isinstance(s,Segment):
            self.l.append(s)
        elif isinstance(s,str):
            self.l.append(Segment({'text':s,'origin':'user'}))
        else:
            print("Sequence: cannot add object of type "+str(type(s))+" to a sequence")
            assert False #Cannot add this type of object to a sequence
        return self
    
    def serialize(self):
        return [k.serialize() for k in self]
    
    def __str__(self):
        """ Returns the concatenation of all segments' text """
        s = "Sequence(\n"
        for segment in self:
            s += segment['text'] + '\n'
        s += ")"
        return s

    def __getitem__(self, i):
        return self.l[i]
    
    def __setitem__(self, i, value):
        self.l[i] = value

    def __delitem__(self, i):
        del self.l[i]
    
    def __len__(self):
        return len(self.l)
    
    def __iter__(self):
        return iter(self.l)
    
    def to_dict(self):
        return [k.to_dict() for k in self]
    
    def format(self,type='completion'):
        """ Returns the concatenation of all segments' text"""
        s = ""
        for segment in self:
            s += segment.format(type=type)
        return s
    
    def show(self,type='completion'):
        """ Prints the concatenation of all segments' text"""
        s = ""
        for segment in self:
            s += segment.show(type=type) + '\n'
        return s
    
    