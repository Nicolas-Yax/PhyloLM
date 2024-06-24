import os
import numpy as np
import json

from lanlab.core.structure.sequence import Sequence
from lanlab.core.structure.segment import Segment
from lanlab.core.structure.batch import Batch

def load_segment(dict):
    return Segment(dict)

def load_sequence(list):
    return Sequence(list)

def load_batch(list):
    #list should represent a numpy array of Sequence (represented as lists of dicts)
    #Step 1 : get the shape of the batch
    a = np.array(list)
    shape = a.shape[:-1]
    #Step 2 : create the batch
    batch = Batch(shape)
    #Step 3 : fill the batch
    for index in np.ndindex(shape):
        batch[index] = Sequence(a[index])
    return batch

def load(path):
    """ Load a file from a path. The file can be a segment, a sequence or a batch. It may not work correctly if sequences in batches do not have the same length."""
    data = np.load(path+'.npy',allow_pickle=True)
    if isinstance(data,dict):
        return load_segment(data)
    else:
        a = np.array(data)
        if len(a.shape) == 1:
            return load_sequence(data)
        elif len(a.shape) >= 1:
            return load_batch(data)
        else:
            raise Exception("Cannot find the shape of the data to load")

def save_segment(segment,path):
    d = segment.serialize()
    np.save(path+'.npy',d)

def save_sequence(sequence,path):
    d = sequence.serialize()
    np.save(path+'.npy',d)

def save_batch(batch,path):
    d = batch.serialize()
    np.save(path+'.npy',d)

def save(data,path):
    #Create the directory if it does not exist
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if isinstance(data,Segment):
        return save_segment(data,path)
    elif isinstance(data,Sequence):
        return save_sequence(data,path)
    elif isinstance(data,Batch):
        return save_batch(data,path)
    else:
        raise Exception("Cannot save data of type {}".format(type(data)))
    
#Legacy code

def load_segment_json(dict):
    return Segment(dict)

def load_sequence_json(list):
    return Sequence(list)

def load_batch_json(list):
    #list should represent a numpy array of Sequence (represented as lists of dicts)
    #Step 1 : get the shape of the batch
    a = np.array(list)
    shape = a.shape[:-1]
    #Step 2 : create the batch
    batch = Batch(shape)
    #Step 3 : fill the batch
    for index in np.ndindex(shape):
        batch[index] = Sequence(a[index])
    return batch

def load_json(path):
    """ Load a file from a path. The file can be a segment, a sequence or a batch. It may not work correctly if sequences in batches do not have the same length."""
    data = json.load(open(path,'r'))
    if isinstance(data,dict):
        return load_segment_json(data)
    else:
        a = np.array(data)
        if len(a.shape) == 1:
            return load_sequence_json(data)
        elif len(a.shape) >= 1:
            return load_batch_json(data)
        else:
            raise Exception("Cannot find the shape of the data to load")

