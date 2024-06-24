import time
import anthropic
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

def segment_from_ClaudeChat(answer):
    seg = Segment()
    seg["model"] = answer.model
    seg['text'] = answer.content[0].text
    seg['origin'] = answer.role
    seg['finish_reason'] = answer.stop_reason
    return seg

def sequence_to_ClaudeChat(seq):
    #Convert a sequence to a list of dictionaries that can be used as input for the chat API
    if seq is None or len(seq) == 0:
        return [dict(role='user',content='')]
    l = []
    for segment in seq:
        d = {}
        d['role'] = segment['origin']#.format(type='chat')
        d['content'] = segment.format(type='chat')
        l.append(d)
    return l

ERROR_COUNT = 0
ERROR_COUNT_MAX = 100

def Clauderetry(f):
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
        except anthropic.RateLimitError as e:
            logging.info('[RateLimitError] : Retrying in 15 sec')
            #ERROR_COUNT += 1
            if ERROR_COUNT > ERROR_COUNT_MAX:
                logging.error('Too many APIErrors : the process will stop')
                assert False
            time.sleep(15)
            return g(*args,**kwargs)
        except anthropic.InternalServerError as e:
            logging.info('[InternalServerError] : Retrying in 15 sec')
            ERROR_COUNT += 1
            if ERROR_COUNT > ERROR_COUNT_MAX:
                logging.error('Too many APIErrors : the process will stop')
                assert False
            time.sleep(15)
            return g(*args,**kwargs)
    return g

class ClaudeGPT:
    pass
        
class ClaudeGPTModelLister(type):
    # This list will store subclasses of A
    subclasses = []

    def __new__(cls, name, bases, dct):
        # Create the new class
        new_class = super().__new__(cls, name, bases, dct)
        # Check if it's a subclass of A (but not A itself)
        if cls._is_subclass_of_ClaudeGPT(bases):
            cls.subclasses.append(new_class)
        return new_class
    
    @classmethod
    def _is_subclass_of_ClaudeGPT(cls, bases):
        for base in bases:
            # Directly check if this is A
            if base.__name__ == 'ClaudeGPT':
                return False
            # Recursively check for subclasses of A
            if issubclass(base, ClaudeGPT) and base is not ClaudeGPT:
                return True
        return False
    
CLAUDE_CLIENT = anthropic.Anthropic(
    api_key=open('.api_claude', 'r').read(),
)
class ClaudeGPT(Model,metaclass=ClaudeGPTModelLister):
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

    @Clauderetry
    def complete(self,sequence):
        raise NotImplementedError
    
    @Clauderetry
    def read(self,sequence):
        """ Reads the given sequence, do not generate an additional segment but updates the logp and top_logp of all the segments in the sequence"""
        raise NotImplementedError
    
    @property
    def timeout(self):
        return 60*3
    
    @property
    def url(self):
        return None

class ClaudeChatGPTConfig(ModelConfig):
    pass

class ClaudeChatGPT(ClaudeGPT):
    """This class is used to interact with the chat models of OpenAI.'"""
    @property
    def config_class(self):
        return ClaudeChatGPTConfig
    @Clauderetry
    def complete(self,sequence,config=None):
        """ Complete the given sequence adding a segment generated by the LLM to it"""
        global CLAUDE_CLIENT
        if config is None:
            config = self.config
        messages = sequence_to_ClaudeChat(sequence)
        answer = CLAUDE_CLIENT.messages.create(
            messages=messages,
            **config.to_dict(),
            model=self.engine)
        
        new_segment = segment_from_ClaudeChat(answer)
        return sequence+new_segment
    @Clauderetry
    def read(self,sequence,config=None):
        raise NotImplementedError #Not available on this family of models due to API limitations
        
    
#--------------------------------------------
# ClaudeChatGPT
#--------------------------------------------
class ClaudeInstant12(ClaudeChatGPT):
    @property
    def engine(self):
        return 'claude-instant-1.2'
    @property
    def name(self):
        return 'CLI12'

class Claude2(ClaudeChatGPT):
    @property
    def engine(self):
        return 'claude-2.0'
    @property
    def name(self):
        return 'CL20'
    
class Claude21(ClaudeChatGPT):
    @property
    def engine(self):
        return 'claude-2.1'
    @property
    def name(self):
        return 'CL21'
    
class Claude3Haiku(ClaudeChatGPT):
    @property
    def engine(self):
        return 'claude-3-haiku-20240307'
    @property
    def name(self):
        return 'CL3HAIKU240307'

class Claude3Sonnet(ClaudeChatGPT):
    @property
    def engine(self):
        return 'claude-3-sonnet-20240229'
    @property
    def name(self):
        return 'CL3SONNET240229'
    
class Claude3Opus(ClaudeChatGPT):
    @property
    def engine(self):
        return 'claude-3-opus-20240229'
    @property
    def name(self):
        return 'CL3OPUS240229'
    
def get_claude_model_classes():
    return ClaudeGPT.subclasses