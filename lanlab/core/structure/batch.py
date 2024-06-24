
from lanlab.core.structure.sequence import Sequence
from lanlab.core.structure.structure import Structure
from lanlab.core.structure.text import Text

import numpy as np
import copy

class Batch(Structure):
    """ Set of sequences of given shape """
    def __init__(self,obj):
        """ obj is either a tuple representing a shape or a numpy array"""
        if isinstance(obj,tuple):
            self.reset(obj)
        elif isinstance(obj,np.ndarray):
            self.init(obj)
        else:
            raise Exception("Batch: cannot initialize with given object. It needs to be a tuple or a numpy array")

    def reset(self,shape):
        self.array = np.empty(shape, dtype=object)
        for index in np.ndindex(shape):
            self.array[index] = Sequence(l=[])

    def init(self,array):
        self.array = array

    def repeat(self,nb,axis=-1):
        """ Repeat the batch n times along the given axis """
        if axis == -1:
            axis = len(self.shape)-1
        array = np.empty(self.shape[:axis] + (nb,) + self.shape[axis+1:],dtype=object)
        for index in np.ndindex(self.shape[:axis] + (nb,) + self.shape[axis+1:]):
            array[index] = copy.deepcopy(self.array[index[:axis] + index[axis+1:]]) #Copy the sequence
        return Batch(array)

    def serialize(self):
        array = copy.deepcopy(self.array)
        for index in np.ndindex(self.shape):
            array[index] = array[index].serialize()
        serialized_array = array.tolist()
        return serialized_array

    @property
    def shape(self):
        return self.array.shape

    def __getitem__(self, i):
        return self.array[i]
    
    def __setitem__(self, i, value):
        if isinstance(value,str):
            value = Text(str)
        self.array[i] = value
    
    def __len__(self):
        return len(self.array)
    
    def __add__(self,other):
        if isinstance(other,Batch):
            if self.shape != other.shape:
                raise Exception("Batch: cannot add two arrays of different shapes")
            return Batch(self.array + other.array)
        else:
            return Batch(self.array + other)
    
    def __iadd__(self,other):
        if isinstance(other,Batch):
            self.array += other.array
        else:
            self.array += other
        return self
    
    def map(self,f):
        array = np.empty(self.shape,dtype=object)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                array[i,j] = f(self.array[i,j])
        out = Batch(array)
        return out
    
    def to_dict(self):
        #Return a dictionary with the 2D data in json format
        return {'name':self.name,'data':[[k.serialize() for k in row] for row in self.array]}
    
    def from_dict(self,d):
        #Load a dictionary with the 2D data in json format
        self.array = np.array([[Sequence(l=k) for k in row] for row in d['data']])
        self.shape = self.array.shape
        self.name = d['name']

    def pop_data(self):
        a = self.array
        self.array = np.empty(self.shape,dtype=object)
        return a

    def push_data(self,data):
        self.array = data