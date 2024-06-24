import logging

def to_dict(d):
    """ Transform a SafeDict into a normal dictionary"""
    #If the object is a list, transform all the elements
    if isinstance(d,list):
        for i in range(len(d)):
            d[i] = to_dict(d[i])
    #If the object is a dictionary, transform all the elements
    elif isinstance(d,dict):
        for k in d:
            d[k] = to_dict(d[k])
    #If the object is a SafeDict, transform it into a normal dictionary
    elif isinstance(d,SafeDict):
        return d.to_dict()
    return d

class SafeDict:
    """ Dictionary for which an error is returned if the user tries to assign an unknown variable (useful to be sure that there aren't spelling mistakes in the config parameters) """
    def __init__(self,d=None):
        #Initialize the dictionary
        if d is None:
            d = {}
        self.d = d

    def had_key(self,k):
        """Return True if the key k is in the dictionary"""
        return k in self.d
    
    def keys(self):
        """Return the keys of the dictionary"""
        return list(self.d.keys())

    def add_key(self,k,v=None):
        """Add an attribute to the dictionary"""
        try:
            self.d[k]
        except KeyError:
            self.d[k] = v

    def load_from(self,d):
        """Load the dictionary from another dictionary"""
        if not(d is None):
            for k in d:
                self[k] = d[k]

    def __getitem__(self,k):
        """Return the value of the key k"""
        return self.d[k]
    
    def set(self,k,v):
        """Set the value of the key k to v"""
        try:
            self.d[k]
            self.d[k] = v
        except KeyError:
            logging.error("Key '"+k+"' not in SafeDict "+str(self.__class__)+" with keys "+str(list(self.d.keys())))
            raise KeyError #"Key not in SafeDict"
        
    def __setitem__(self,k,v):
        """Set the value of the key k to v"""
        self.set(k,v)

    def __iter__(self):
        """Return an iterator over the dictionary"""
        return iter(self.d)

    def set(self,k,v):
        """Set the value of the key k to v"""
        self.d[k] = v

    def copy(self):
        """Return a copy of the dictionary"""
        return SafeDict(self.d.copy())

    def to_dict(self):
        """ Transform the SafeDict into a normal dictionary """
        return self.d
    
    def from_dict(self,d):
        """ alias of load_from """
        self.load_from(d)

    def __str__(self):
        return str(self.d)
    
    def __repr__(self):
        return str(self)

    def __eq__(self,o):
        return self.d == o.d
    
    def __ne__(self,o):
        return self.d != o.d
    
    def __contains__(self,k):
        return k in self.d