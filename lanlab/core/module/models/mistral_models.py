import time
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import mistralai
import logging
import os
import requests

from lanlab.core.module.models.model import Model, ModelConfig
from lanlab.core.tools.dict_tools import SafeDict
from lanlab.core.structure.sequence import Sequence
from lanlab.core.structure.segment import Segment

class UnsplittableQuery(Exception):
    def __init__(self):
        super().__init__()
        self.message = "The query cannot be found in the token list. This could either be due to a bug so that the query and the list of tokens mismatch or it could be due to a retokenization issue (if A is a list of token : tokenize(detokenize(A)) != A)."

def split_token_list(l,query):
    #Split the token list l into two lists, the first one containing the tokens that are in the query and the second one containing the rest of the tokens
    s = ""
    if s == query:
        return l
    for i in range(len(l)):
        s += l[i]
        if s == query:
            return l[:i+1],l[i+1:]
    raise UnsplittableQuery

def chat_to_str(l):
    #Convert a chat completion from openai to a string
    s = ""
    for i,d in enumerate(l):
        k,v = d['role'],d['content']
        s += k.upper()+": "+v
        if i < len(l)-1:
            s += '\n'
    return s

def segment_from_MISTRALChat(answer):
    seg = Segment()
    seg["model"] = answer['model']
    seg['text'] = answer['choices'][0]['message']['content']
    seg['origin'] = answer['choices'][0]['message']['role']
    seg['finish_reason'] = answer['choices'][0]['finish_reason']
    return seg

def sequence_to_MISTRALChat(seq):
    #Convert a sequence to a list of dictionaries that can be used as input for the chat API
    if seq is None or len(seq) == 0:
        return [ChatMessage(role='user',content='')]
    l = []
    for segment in seq:
        d = {}
        d['role'] = segment['origin']#.format(type='chat')
        d['content'] = segment.format(type='chat')
        l.append(ChatMessage(**d))
    return l

ERROR_COUNT = 0
ERROR_COUNT_MAX = 100

def Mistralretry(f):
    """This decorator is used to retry a function if it raises an APIError or a RateLimitError."""
    def g(*args,**kwargs):
        global ERROR_COUNT, ERROR_COUNT_MAX
        try:
            return f(*args,**kwargs)
        except requests.exceptions.Timeout:
            logging.info('[requests Timeout] : Retrying in 15 sec')
            ERROR_COUNT += 1
            if ERROR_COUNT > ERROR_COUNT_MAX:
                logging.error('Too many APIErrors : the process will stop')
                assert False
            time.sleep(15)
            return g(*args,**kwargs)
        except mistralai.exceptions.MistralException as e:
            if e.message == "Unexpected response: {'message': 'Requests rate limit exceeded'}":
                logging.info('[RateLimitError] : Retrying in 15 sec')
                #ERROR_COUNT += 1
                if ERROR_COUNT > ERROR_COUNT_MAX:
                    logging.error('Too many APIErrors : the process will stop')
                    assert False
                time.sleep(15)
                return g(*args,**kwargs)
            else:
                raise e
    return g

class MistralGPT:
    pass
        
class MistralGPTModelLister(type):
    # This list will store subclasses of A
    subclasses = []

    def __new__(cls, name, bases, dct):
        # Create the new class
        new_class = super().__new__(cls, name, bases, dct)
        # Check if it's a subclass of A (but not A itself)
        if cls._is_subclass_of_MistralGPT(bases):
            cls.subclasses.append(new_class)
        return new_class
    
    @classmethod
    def _is_subclass_of_MistralGPT(cls, bases):
        for base in bases:
            # Directly check if this is A
            if base.__name__ == 'MistralGPT':
                return False
            # Recursively check for subclasses of A
            if issubclass(base, MistralGPT) and base is not MistralGPT:
                return True
        return False
    
MISTRAL_CLIENT = MistralClient(api_key=open('.api_mistral', 'r').read())
class MistralGPT(Model,metaclass=MistralGPTModelLister):
    """This class is a wrapper for the OpenAI GPT-3 API. It is used to send requests to the API and to get the responses."""
    def __init__(self):
        super().__init__()
        self._engine = None

        self._mode = 'complete'
        self.abc = 3

    @property
    def name(self):
        raise NotImplementedError

    @property
    def engine(self):
        return NotImplementedError

    @Mistralretry
    def complete(self,sequence):
        raise NotImplementedError
    
    @Mistralretry
    def read(self,sequence):
        """ Reads the given sequence, do not generate an additional segment but updates the logp and top_logp of all the segments in the sequence"""
        raise NotImplementedError
    
    @property
    def url(self):
        return None

class MistralChatGPTConfig(ModelConfig):
    pass

class MistralChatGPT(MistralGPT):
    """This class is used to interact with the chat models of OpenAI.'"""
    @property
    def config_class(self):
        return MistralChatGPTConfig
    @Mistralretry
    def complete(self,sequence,config=None):
        """ Complete the given sequence adding a segment generated by the LLM to it"""
        global MISTRAL_CLIENT
        if config is None:
            config = self.config
        messages = sequence_to_MISTRALChat(sequence)
        answer = MISTRAL_CLIENT.chat(
            messages=messages,
            **config.to_dict(),
            model=self.engine).model_dump()
        
        new_segment = segment_from_MISTRALChat(answer)
        return sequence+new_segment
    @Mistralretry
    def read(self,sequence,config=None):
        raise NotImplementedError #Not available on this family of models due to API limitations
        
    
#--------------------------------------------
# MistralChatGPT
#--------------------------------------------

class MistralTiny2312(MistralChatGPT):
    @property
    def engine(self):
        return 'mistral-tiny-2312'
    @property
    def name(self):
        return 'MISTiny2312'
    
class MistralSmall2312(MistralChatGPT):
    @property
    def engine(self):
        return 'mistral-small-2312'
    @property
    def name(self):
        return 'MISSmall2312'
    
class MistralSmall2402(MistralChatGPT):
    @property
    def engine(self):
        return 'mistral-small-2402'
    @property
    def name(self):
        return 'MISSmall2402'

class MistralMedium2312(MistralChatGPT):
    @property
    def engine(self):
        return 'mistral-medium-2312'
    @property
    def name(self):
        return 'MISMedium2312'
    
class MistralLarge2402(MistralChatGPT):
    @property
    def engine(self):
        return 'mistral-large-2402'
    @property
    def name(self):
        return 'MISLarge2402'
    
def get_mistral_model_classes():
    return MistralGPT.subclasses