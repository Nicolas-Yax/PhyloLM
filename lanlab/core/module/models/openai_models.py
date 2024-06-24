import time
import openai
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

def segment_from_OPENAICompletion(answer):
    seg = Segment()
    seg["model"] = answer['model']
    seg['text'] = answer['choices'][0]['text']
    seg['tokens'] = answer['choices'][0]['logprobs']['tokens']
    seg['logp'] = answer['choices'][0]['logprobs']['token_logprobs']
    seg['top_logp'] = [dict(e) if not(e is None) else None for e in answer['choices'][0]['logprobs']['top_logprobs']]
    seg['finish_reason'] = answer['choices'][0]['finish_reason']
    seg['origin'] = 'assistant'
    if 'logits' in answer['choices'][0]['logprobs']:
        seg['logits'] = answer['choices'][0]['logprobs']['logits']
    return seg

def segment_from_OPENAIChat(answer):
    seg = Segment()
    seg["model"] = answer['model']
    seg['text'] = answer['choices'][0]['message']['content']
    seg['origin'] = answer['choices'][0]['message']['role']
    seg['finish_reason'] = answer['choices'][0]['finish_reason']
    return seg

def sequence_to_OPENAIChat(seq):
    #Convert a sequence to a list of dictionaries that can be used as input for the chat API
    if seq is None or len(seq) == 0:
        return [{'role':'user','content':''}]
    l = []
    for segment in seq:
        d = {}
        d['role'] = segment['origin']#.format(type='chat')
        d['content'] = segment.format(type='chat')
        l.append(d)
    return l

ERROR_COUNT = 0
ERROR_COUNT_MAX = 100

def OPENAIretry(f):
    """This decorator is used to retry a function if it raises an APIError or a RateLimitError."""
    def g(*args,**kwargs):
        global ERROR_COUNT, ERROR_COUNT_MAX
        try:
            return f(*args,**kwargs)
        except openai.APIError:
            logging.error('[openai API Error] : Retrying in 1 sec')
            ERROR_COUNT += 1
            if ERROR_COUNT > ERROR_COUNT_MAX:
                logging.error('Too many APIErrors : the process will stop')
                assert False
            time.sleep(1)
            return g(*args,**kwargs)
        except openai.RateLimitError:
            logging.error('[openai RateLimitError] : Retrying in 15 sec')
            #ERROR_COUNT += 1
            if ERROR_COUNT > ERROR_COUNT_MAX:
                logging.error('Too many APIErrors : the process will stop')
                assert False
            time.sleep(15)
            return g(*args,**kwargs)
        except requests.exceptions.Timeout:
            logging.error('[requests Timeout] : Retrying in 15 sec')
            ERROR_COUNT += 1
            if ERROR_COUNT > ERROR_COUNT_MAX:
                logging.error('Too many APIErrors : the process will stop')
                assert False
            time.sleep(15)
            return g(*args,**kwargs)
        """except openai.Timeout:
            logging.info('[openai Timeout] : Retrying in 15 sec')
            ERROR_COUNT += 1
            if ERROR_COUNT > ERROR_COUNT_MAX:
                logging.error('Too many APIErrors : the process will stop')
                assert False
            time.sleep(15)
        except openai.ServiceUnavailableError:
            logging.info('[openai ServiceUnavailableError] : Retrying in 15 sec')
            ERROR_COUNT += 1
            if ERROR_COUNT > ERROR_COUNT_MAX:
                logging.error('Too many APIErrors : the process will stop')
                assert False
            time.sleep(15)"""
    return g

class OPENAIGPT:
    pass
        
class OPENAIGPTModelLister(type):
    # This list will store subclasses of A
    subclasses = []

    def __new__(cls, name, bases, dct):
        # Create the new class
        new_class = super().__new__(cls, name, bases, dct)
        # Check if it's a subclass of A (but not A itself)
        if cls._is_subclass_of_OPENAIGPT(bases):
            cls.subclasses.append(new_class)
        return new_class
    
    @classmethod
    def _is_subclass_of_OPENAIGPT(cls, bases):
        for base in bases:
            # Directly check if this is A
            if base.__name__ == 'OPENAIGPT':
                return False
            # Recursively check for subclasses of A
            if issubclass(base, OPENAIGPT) and base is not OPENAIGPT:
                return True
        return False
    
OPENAI_CLIENT = openai.OpenAI(api_key=open('.api_openai', 'r').read())
class OPENAIGPT(Model,metaclass=OPENAIGPTModelLister):
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

    @OPENAIretry
    def complete(self,sequence):
        raise NotImplementedError
    
    @OPENAIretry
    def read(self,sequence):
        """ Reads the given sequence, do not generate an additional segment but updates the logp and top_logp of all the segments in the sequence"""
        raise NotImplementedError
    
    @property
    def url(self):
        return None

class ChatGPTConfig(ModelConfig):
    def __init__(self):
        super().__init__()
        self.add_key('stop',None)
        self.add_key('logit_bias',{})

class ChatGPT(OPENAIGPT):
    """This class is used to interact with the chat models of OpenAI.'"""
    @property
    def config_class(self):
        return ChatGPTConfig
    #@OPENAIretry
    def complete(self,sequence,config=None):
        """ Complete the given sequence adding a segment generated by the LLM to it"""
        global OPENAI_CLIENT
        if config is None:
            config = self.config
        messages = sequence_to_OPENAIChat(sequence)
        answer = OPENAI_CLIENT.chat.completions.create(
            messages=messages,
            **config.to_dict(),
            model=self.engine).model_dump()
        new_segment = segment_from_OPENAIChat(answer)
        return sequence+new_segment
    @OPENAIretry
    def read(self,sequence,config=None):
        raise NotImplementedError #Not available on this family of models due to API limitations

class CompletionGPTConfig(ModelConfig):
    """ Class that implements top_logprobs parameter """
    def __init__(self):
        super().__init__()
        self.add_key('logprobs',5)
        self.add_key('stop',None)
        self.add_key('logit_bias',{})

class CompletionGPT(OPENAIGPT):
    """This class is used to interact with the completion models of OpenAI from the completion module. These models do not take a sequence as input but the prompt (str) to be completed directly.'"""
    @property
    def config_class(self):
        return CompletionGPTConfig
    @OPENAIretry
    def complete(self,sequence,config=None):
        global OPENAI_CLIENT
        if config is None:
            config = self.config
        prompt = sequence.format(type='completion')
        answer = OPENAI_CLIENT.completions.create(
            prompt=prompt,
            model = self.engine,
            **config.to_dict()).model_dump()
        new_segment = segment_from_OPENAICompletion(answer)
        return sequence +new_segment
    @OPENAIretry
    def read(self,sequence,config=None):
        global OPENAI_CLIENT
        if config is None:
            config = self.config
        prompt = sequence.format(type='completion')
        config['max_tokens'] = 0
        answer = OPENAI_CLIENT.completions.create(
            prompt=prompt,
            **config.to_dict(),
            echo=True).model_dump()
        new_segment = segment_from_OPENAICompletion(answer)
        return Sequence(l=[new_segment])
        


#--------------------------------------------
#CompletionGPT
#--------------------------------------------

class AD(CompletionGPT):
    @property
    def engine(self):
        return 'ada'
    @property
    def name(self):
        return 'AD'

class TAD1(CompletionGPT):
    @property
    def engine(self):
        return 'text-ada-001'
    @property
    def name(self):
        return 'TAD1'

class BB(CompletionGPT):
    @property
    def engine(self):
        return 'babbage'
    @property
    def name(self):
        return 'BB'

class TBB1(CompletionGPT):
    @property
    def engine(self):
        return 'text-babbage-001'
    @property
    def name(self):
        return 'TBB1'

class CU(CompletionGPT):
    @property
    def engine(self):
        return 'curie'
    @property
    def name(self):
        return 'CU'

class TCU1(CompletionGPT):
    @property
    def engine(self):
        return 'text-curie-001'
    @property
    def name(self):
        return 'TCU1'

class DV(CompletionGPT):
    @property
    def engine(self):
        return 'davinci'
    @property
    def name(self):
        return 'DV'

class DVB(CompletionGPT):
    @property
    def engine(self):
        return 'davinci-instruct-beta'
    @property
    def name(self):
        return 'DVB'

class TDV1(CompletionGPT):
    @property
    def engine(self):
        return 'text-davinci-001'
    @property
    def name(self):
        return 'TDV1'

class CDV2(CompletionGPT):
    @property
    def engine(self):
        return 'code-davinci-002'
    @property
    def name(self):
        return 'CDV2'

class TDV2(CompletionGPT):
    @property
    def engine(self):
        return 'text-davinci-002'
    @property
    def name(self):
        return 'TDV2'

class TDV3(CompletionGPT):
    @property
    def engine(self):
        return 'text-davinci-003'
    @property
    def name(self):
        return 'TDV3'
    
class GPT35I(CompletionGPT):
    @property
    def engine(self):
        return 'gpt-3.5-turbo-instruct'
    @property
    def name(self):
        return 'GPT35I'
    
class GPT35I_0914(CompletionGPT):
    @property
    def engine(self):
        return 'gpt-3.5-turbo-instruct-0914'
    @property
    def name(self):
        return 'GPT35I_0914'
    
class DV2(CompletionGPT):
    @property
    def engine(self):
        return 'davinci-002'
    @property
    def name(self):
        return 'DV2'
    
class BB2(CompletionGPT):
    @property
    def engine(self):
        return 'babbage-002'
    @property
    def name(self):
        return 'BB2'
    
#--------------------------------------------
#ChatGPT
#--------------------------------------------

class GPT35(ChatGPT):
    @property
    def engine(self):
        return 'gpt-3.5-turbo'
    @property
    def name(self):
        return 'GPT35'
    
class GPT35_0613(ChatGPT):
    @property
    def engine(self):
        return 'gpt-3.5-turbo-0613'
    @property
    def name(self):
        return 'GPT35_0613'

class GPT35_0301(ChatGPT):
    @property
    def engine(self):
        return 'gpt-3.5-turbo-0301'
    @property
    def name(self):
        return 'GPT35_0301'
    
class GPT35_1106(ChatGPT):
    @property
    def engine(self):
        return 'gpt-3.5-turbo-1106'
    @property
    def name(self):
        return 'GPT35_1106'
    
class GPT35_16K(ChatGPT):
    @property
    def engine(self):
        return 'gpt-3.5-turbo-16k'
    @property
    def name(self):
        return 'GPT35_16K'
    
class GPT35_16K_0613(ChatGPT):
    @property
    def engine(self):
        return 'gpt-3.5-turbo-16k-0613'
    @property
    def name(self):
        return 'GPT35_16K_0613'
    
class GPT4(ChatGPT):
    @property
    def engine(self):
        return 'gpt-4'
    @property
    def name(self):
        return 'GPT4'
    
class GPT4_0314(ChatGPT):
    @property
    def engine(self):
        return 'gpt-4-0314'
    @property
    def name(self):
        return 'GPT4_0314'

class GPT4_0613(ChatGPT):
    @property
    def engine(self):
        return 'gpt-4-0613'
    @property
    def name(self):
        return 'GPT4_0613'
    
class GPT4_1106(ChatGPT):
    @property
    def engine(self):
        return 'gpt-4-1106-preview'
    @property
    def name(self):
        return 'GPT4_1106'
    
class GPT4_0125(ChatGPT):
    @property
    def engine(self):
        return 'gpt-4-0125-preview'
    @property
    def name(self):
        return 'GPT4_0125'
    
class GPT4_0125(ChatGPT):
    @property
    def engine(self):
        return 'gpt-4-0125-preview'
    @property
    def name(self):
        return 'GPT4_0125'
    
class GPT4_240409(ChatGPT):
    @property
    def engine(self):
        return 'gpt-4-turbo-2024-04-09'
    @property
    def name(self):
        return 'GPT4_240409'
    
class GPT4V(ChatGPT):
    @property
    def engine(self):
        return 'gpt-4-vision-preview'
    @property
    def name(self):
        return 'GPT4V'
    #GPT4V doesn't handle many features yet
    def complete(self,sequence,config=None):
        """ Complete the given sequence adding a segment generated by the LLM to it"""
        if config is None:
            config = self.config
        messages = sequence_to_OPENAIChat(sequence)
        # The model doesn't handle the following parameters
        del config.d['stop']
        del config.d['logit_bias']
        answer = self.client.chat.completions.create(
            messages=messages,
            **config.to_dict(),
            model=self.engine).model_dump()
        new_segment = segment_from_OPENAIChat(answer)
        return sequence+new_segment

def get_openai_model_classes():
    return OPENAIGPT.subclasses