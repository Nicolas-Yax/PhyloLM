import torch
from flask import Flask, request
import logging
from transformers import GenerationConfig
#import torch.multiprocessing as mp
import multiprocess as mp
#from pathos.helpers import mp
import numpy as np
import os
import time

from lanlab.core.module.models.model import Model
from lanlab.core.module.models.openai_models import segment_from_OPENAICompletion
from lanlab.core.structure.sequence import Sequence
from lanlab.core.module.models.model import ModelConfig
import requests

#Set the path to a folder containing the HF models
HFMODELSPATH = "" #Fill with your HF folder where all models are downloaded

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def default_config():
    """ Default config for the model generation"""
    return {'prompt':None,
            'temperature':0.7,
            'min_tokens':0,
            'max_tokens':8,
            'logprobs':5,
            'stop':[],
            'echo':False,
            'return_logits':False}

def flask_server(port,inp_queue):
    """ Starts a flask server that will handle the requests"""
    app = Flask(__name__)

    @app.route("/completions", methods=['POST'])
    def completions():
        """ Flask route for the completions"""
        global server
        logging.debug('got request')
        config = default_config()
        for k in config:
            try:
                config[k] = request.json[k]
            except KeyError:
                pass
        logging.debug('parsed request')
        parent_conn,child_conn = mp.Pipe()
        inp_queue.put({"config":config,"pipe":child_conn})
        logging.debug('waiting for the model')
        out = parent_conn.recv()#server.ret_queue.get()
        logging.debug('returns')
        return out
    
    app.run(host='0.0.0.0',port=port,threaded=True)
    
class Server:
    """ Server that handles the requests"""
    def __init__(self,model_loader,port):
        self.port = port
        
        self.model_loader = model_loader
        
        self.active = False

        self.batch_size = 32
        self.timeout = 0.5 #In seconds

    def start(self):
        """ Starts the server and creates the process that will handle the requests"""
        self.inp_queue = mp.Queue()
        self.ret_queue = mp.Queue()
        
        
        logging.info('starting flask server')
        self.flask = mp.Process(target=flask_server,args=(self.port,self.inp_queue))
        self.flask.start()
        
        logging.info('starting model hosting process')
        self.process = mp.Process(target=completion_loop,args=(self.model_loader,self.inp_queue,self.ret_queue,self.timeout,self.batch_size))
        self.process.start()
        self.active = True

    def stop(self):
        self.inp_queue.close()
        self.ret_queue.close()
        self.flask.terminate()
        self.process.terminate()
        self.active = False
        
    def __enter__(self):
        if not(self.active):
            self.start()
            time.sleep(5)
        
    def __exit__(self,*args,**kwargs):
        if self.active:
            self.stop()

class HFModel:
    pass
        
class HFModelLister(type):
    """ References all HF models where created """
    # This list will store subclasses of A
    subclasses = []

    def __new__(cls, name, bases, dct):
        # Create the new class
        new_class = super().__new__(cls, name, bases, dct)
        # Check if it's a subclass of A (but not A itself)
        if cls._is_subclass_of_HFModel(bases):
            cls.subclasses.append(new_class)
        return new_class
    
    @classmethod
    def _is_subclass_of_HFModel(cls, bases):
        for base in bases:
            # Check for subclasses of A
            if issubclass(base, HFModel) and hasattr(base,'name') and hasattr(base,'engine'):
                return True
        return False

class HFModelConfig(ModelConfig):
    def __init__(self):
        super().__init__()
        self.add_key('return_logits',False) #HF models can return logits

class HFModel(Model,metaclass=HFModelLister):
    def __init__(self):
        super().__init__()
        
    @property
    def engine(self):
        return None
    
    @property
    def name(self):
        return None
        
    @property
    def timeout(self):
        return None #No timeout by default for self hosted models

    @property
    def name(self):
        raise NotImplementedError
        
    def generation_config(self):
        return {}
        
    def host(self,port=None):
        if port is None:
            port = np.random.randint(48750,58750)
        self.port = port
        server = Server(self.init_model,port=port)
        return server
    
    @property
    def url(self):
        return None

    @property
    def config_class(self):
        return HFModelConfig
    
    def complete(self,sequence,config=None):
        if config is None:
            config = self.config
        prompt = sequence.format(type='completion')
        data = {'prompt':prompt,'logprobs':5,**config.to_dict()}
        answer = requests.post('http://127.0.0.1:'+str(self.port)+'/completions',json=data).json()
        answer['model'] = self.name
        if self['return_logits']:
            answer['logits'] = np.array(answer['choices'][0]['logprobs']['logits'],np.float16)
        segment = segment_from_OPENAICompletion(answer)
        return sequence+segment

    def read(self,sequence,config=None):
        if config is None:
            config = self.config
        prompt = sequence.format(type='completion')
        config['max_tokens'] = 0
        data = {'prompt':prompt,'logprobs':5,'echo':True,**config.to_dict()}
        answer = requests.post('http://127.0.0.1:'+str(self.port)+'/completions',json=data).json()
        answer['model'] = self.name
        segment = segment_from_OPENAICompletion(answer)
        return Sequence(l=[segment])
    
    def init_kv(self,inputs,model):
        """ Initialize the kv dict. Most of the time it is empty but for some models they use custom kv dicts that need to be intialized """
        return None
    
    def init_model(self):
        """ Initialize the model and put the init kv function """ 
        tokenizer,model = self.load_model()
        print('model dtype',model.dtype)
        model.init_kv = self.init_kv #TODO : improve this point as if model already has a init_kv it will conflict
        return tokenizer,model
    

def completion_loop(model_loader,inp_queue,out_queue,timeout,batch_size):
    """ Loop that handles the requests and sends them to the model"""
    def load_model():
        logging.info('loading model')
        tokenizer,model = model_loader()
        logging.info('model loaded')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model#.to(device)
        return model,tokenizer

    last_time = time.perf_counter()
    batches = {}
    
    model,tokenizer = None,None #Lazy loading
    
    def flush(batch,temperature,max_tokens,model,tokenizer):
        """Flushes the batch and sends it to the model and resets the timer"""
        if len(batch)>=1:
            if model is None:
                model,tokenizer = load_model() #Load the model and the tokenizer
            logging.debug('flush batch of size '+str(len(batch)))
            complete(tokenizer,model,batch,temperature,max_tokens)
            logging.debug('completed batch')
        return model,tokenizer

    while True:
        #Get the first item in the queue
        logging.debug('completion loop iter')
        if not(inp_queue.empty()):
            logging.debug('queue not empty')
            top = inp_queue.get(timeout=1)
            logging.debug('got from queue')
            if isinstance(top,dict):
                logging.debug('got completion order')
                #get temperature
                temp = top['config']['temperature']
                max_tokens = top['config']['max_tokens']
                key = (temp,max_tokens)
                if not(key in batches):
                    batches[key] = [top]
                else:
                    batches[key].append(top)
                if len(batches[key])>=batch_size:
                    model,tokenizer = flush(batches[key],temp,max_tokens,model,tokenizer)
                    del batches[key]
                    last_time = time.perf_counter()
        else:
            if time.perf_counter()-last_time>timeout:
                if len(batches)>0:
                    key = list(batches.keys())[0]
                    temp,max_tokens = key
                    model,tokenizer = flush(batches[key],temp,max_tokens,model,tokenizer)
                    del batches[key]
                last_time = time.perf_counter()
            time.sleep(timeout/10)
    
def complete(tokenizer,model,data,temperature,max_tokens):
    """Completes the queries given in data and returns the results with the OPENAI format"""
    configs,pipes = [data_['config'] for data_ in data],[data_['pipe'] for data_ in data]
    prompts = [config['prompt'] for config in configs]
    #Prepare padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    inp_batch = tokenizer(prompts,return_tensors='pt',padding=True)
    
    #temperature = [config['temperature'] for config in configs]
    #min_tokens = [config['min_tokens'] for config in configs]
    #max_tokens = [config['max_tokens'] for config in configs]
    #stop = [config['stop'] if not(config['stop'] is None) else [] for config in configs]
    echo = [config['echo'] for config in configs]
    nb_logprobs = [config['logprobs'] for config in configs]
    return_logits = [config['return_logits'] for config in configs]
    
    #generate the completion with the required parameters
    with torch.no_grad():
        token_ids,logits = generate(model,tokenizer,inp_batch,max_tokens,temperature=temperature)
    results = [dict_to_openai(token_ids[i],logits[i],tokenizer,temperature,return_logits=return_logits[i],nb_logprobs=nb_logprobs[i],inputs_to_remove=inp_batch.input_ids[i] if not echo[i] else None) for i in range(len(configs))]
    for r,p in zip(results,pipes):
        p.send(r)
        p.close()
    
#tokenize the text "Test" with the tokenizer and return the tokens
def tokenize(tokenizer,text):
    """Tokenizes the text with the tokenizer and returns the tokens"""
    return tokenizer(text,return_tensors='pt',padding='longest')

def generate(model,tokenizer,inp_batch,max_tokens,temperature):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    input_token_ids = inp_batch.input_ids.to(device)
    input_attention_mask = inp_batch.attention_mask.to(device)

    #Start the prompt with BOS
    last_token_ids = input_token_ids[:,0]
    token_ids = [input_token_ids[:,0]]
    
    kv = model.init_kv(last_token_ids[:,None],model)
    logits = []
    attention_mask = torch.cat([input_attention_mask,torch.ones((input_attention_mask.shape[0],max_tokens+1),dtype=torch.int64).to(device)],dim=1)
    for i in range(input_token_ids.shape[1]+max_tokens-1):
        #Forward the model
        #generation_config=None
        #if hasattr(model,'generation_config'):
        #    generation_config = model.generation_config
        out = model(last_token_ids[:,None],return_dict=True,past_key_values=kv,use_cache=True,attention_mask=attention_mask[:,:i+1])#,generation_config=generation_config)
        last_logits = out.logits[:,0,:]
        kv = out.past_key_values
        #For phi-1.5b model that requires a different format for storing kv
        if hasattr(kv,'sequence_len_offset'):
            kv.sequence_len_offset += 1
        logits.append(last_logits)

        #Sample new tokens from logits
        if i < input_token_ids.shape[1]-1:
            last_token_ids = input_token_ids[:,i+1]
        else:
            if temperature > 0:
                probs = (last_logits/temperature).softmax(-1).double()
                last_token_ids = torch.multinomial(probs,num_samples=1,replacement=True)[:,0]
            else:
                last_token_ids = last_logits.argmax(-1)
        token_ids.append(last_token_ids)

    return torch.stack(token_ids,dim=1).cpu(),torch.stack(logits,dim=1).cpu()

def dict_to_openai(token_ids,logits,tokenizer,temperature,return_logits=False,nb_logprobs=5,inputs_to_remove=None):
    """ Returns the data in the OPENAI format"""
    #Compute logp associated with these ids
    if temperature > 0:
        logprobs = (logits/temperature).softmax(-1).log()
        generated_tokens_logp = [lprobs[token_id].item() for lprobs,token_id in zip(logprobs,token_ids[1:])]
        #Top logp computations
        top_logp = []
        for lprobs in logprobs:
            best_token_ids = lprobs.argsort()[-nb_logprobs:]
            top_logp.append({tokenizer.convert_tokens_to_string([tokenizer.convert_ids_to_tokens([token_id.item()])[0]]):lprobs[token_id.item()].item() for token_id in best_token_ids})

        #Outputs
        tokens_logprobs = [None] + generated_tokens_logp
        top_logprobs = [None] + top_logp
    else:
        tokens_logprobs = [None]*len(token_ids)
        top_logprobs = [None]*len(token_ids)
    

    #Translate the token ids into text
    logging.debug('Translating tokens into text')
    generated_tokens_raw = [tokenizer.convert_ids_to_tokens([token_id])[0] for token_id in token_ids]
    logging.debug(generated_tokens_raw)
    generated_tokens = [tokenizer.convert_tokens_to_string([t]) for t in generated_tokens_raw]
    logging.debug(generated_tokens)
    generated_sequence = tokenizer.convert_tokens_to_string(generated_tokens_raw)
    logging.debug(generated_sequence)
    logging.debug(inputs_to_remove)
    logging.debug('----------')
    
    #Compute echo -> the bos token as well
    if not(inputs_to_remove is None):
        if tokenizer.bos_token is None:
            tokenizer.bos_token = ''
        generated_tokens_raw = [tokenizer.bos_token]+generated_tokens_raw[inputs_to_remove.shape[0]:]
        logging.debug(generated_tokens_raw)
        generated_tokens = [tokenizer.convert_tokens_to_string([t]) for t in generated_tokens_raw]
        logging.debug(generated_tokens)
        generated_sequence = tokenizer.convert_tokens_to_string(generated_tokens_raw)
        logging.debug(generated_sequence)

        tokens_logprobs = tokens_logprobs[inputs_to_remove.shape[0]:]
        top_logprobs = top_logprobs[inputs_to_remove.shape[0]:]
        
    del generated_tokens[0] #Remove bos
    generated_sequence = generated_sequence[len(tokenizer.bos_token):] #Remove bos
    
    #remove padding tokens
    pad_token = tokenizer.pad_token
    index_pad = [] #Index of the non padded tokens
    for i in range(len(generated_tokens[1:])):
        if generated_tokens[i] == pad_token:
            continue
        index_pad.append(i)
    generated_tokens = [generated_tokens[i] for i in index_pad]
    tokens_logprobs = [tokens_logprobs[i] for i in index_pad]
    top_logprobs = [top_logprobs[i] for i in index_pad]
    logits = logits.type(torch.float16).numpy().astype(np.float16)[index_pad,:]
    logits = logits.tolist()
    
    #Return the dict
    out =  {'choices':[
        {'text':generated_sequence,
         'logprobs':{
             'tokens': generated_tokens,
             'token_logprobs': tokens_logprobs,
             'top_logprobs': top_logprobs,
         },
         'finish_reason':'Not Implemented'
        }
    ]
    }
    if return_logits:
        out['choices'][0]['logprobs']['logits'] = logits
    return out

        
#-------------------------------------------------------------------------------------------------------------- 
#
#                                             LLaMa Models
#
#---------------------------------------------------------------------------------------------------------------
    
    
class LlamaFamily(HFModel):
    def load_model(self):
        from transformers import AutoTokenizer,AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(HFMODELSPATH,self.engine))
        #tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(os.path.join(HFMODELSPATH,self.engine),device_map = 'auto',trust_remote_code=True)#,torch_dtype=torch.float16)
        return tokenizer,model

class Llama7B(LlamaFamily):
    @property
    def engine(self):
        return 'llama-7b'
    @property
    def name(self):
        return 'LLA7'
    @property
    def url(self):
        return 'https://huggingface.co/huggyllama/llama-7b' #Actually the one we use is the original one. This links only toward a huggingface repository containing the model but is not the repository we used
    
class Llama13B(LlamaFamily):
    @property
    def engine(self):
        return 'llama-13b'
    @property
    def name(self):
        return 'LLA13'
    @property
    def url(self):
        return 'https://huggingface.co/huggyllama/llama-13b' #Actually the one we use is the original one. This links only toward a huggingface repository containing the model but is not the repository we used
    
class Alpaca7B(LlamaFamily):
    @property
    def engine(self):
        return 'alpaca-7b'
    @property
    def name(self):
        return 'ALP7'
    @property
    def url(self):
        return 'https://huggingface.co/chavinlo/alpaca-native'
    
class Wizard7B(LlamaFamily):
    @property
    def engine(self):
        return 'wizard-7b'
    @property
    def name(self):
        return 'WIZ7'
    @property
    def url(self):
        return 'https://huggingface.co/WizardLM/WizardLM-7B-V1.0'
    
class Vicuna7B_11(LlamaFamily):
    @property
    def engine(self):
        return 'vicuna-7b-v1.1'
    @property
    def name(self):
        return 'VIC7_11'
    @property
    def url(self):
        return 'https://huggingface.co/lmsys/vicuna-7b-v1.1'
    
class Vicuna7B_13(LlamaFamily):
    @property
    def engine(self):
        return 'vicuna-7b-v1.3'
    @property
    def name(self):
        return 'VIC7_13'
    @property
    def url(self):
        return 'https://huggingface.co/lmsys/vicuna-7b-v1.3'
    
class Vicuna7B_15(LlamaFamily):
    @property
    def engine(self):
        return 'vicuna-7b-v1.5'
    @property
    def name(self):
        return 'VIC7_15'
    @property
    def url(self):
        return 'https://huggingface.co/lmsys/vicuna-7b-v1.5'
    
class Vicuna13B_11(LlamaFamily):
    @property
    def engine(self):
        return 'vicuna-13b-v1.1'
    @property
    def name(self):
        return 'VIC13_11'
    @property
    def url(self):
        return 'https://huggingface.co/lmsys/vicuna-13b-v1.1'
    
class Vicuna13B_13(LlamaFamily):
    @property
    def engine(self):
        return 'vicuna-13b-v1.3'
    @property
    def name(self):
        return 'VIC13_13'
    @property
    def url(self):
        return 'https://huggingface.co/lmsys/vicuna-13b-v1.3'
    
class Vicuna13B_15(LlamaFamily):
    @property
    def engine(self):
        return 'vicuna-13b-v1.5'
    @property
    def name(self):
        return 'VIC13_15'
    @property
    def url(self):
        return 'https://huggingface.co/lmsys/vicuna-13b-v1.5'

class Baize7B(LlamaFamily):
    @property
    def engine(self):
        return 'baize-7b'
    @property
    def name(self):
        return 'BAI7'
    @property
    def url(self):
        return 'https://huggingface.co/project-baize/baize-v2-7b'
    
class Guanaco7B(LlamaFamily):
    @property
    def engine(self):
        return 'guanaco-7b'
    @property
    def name(self):
        return 'GUA7'
    @property
    def url(self):
        return 'https://huggingface.co/JosephusCheung/Guanaco'
    
class TinyLlama(LlamaFamily):
    @property
    def engine(self):
        return 'tiny-llama-fast-tokenizer'
    @property
    def name(self):
        return 'TLLA'
    @property
    def url(self):
        return None
    
class Llama2_7B(LlamaFamily):
    @property
    def engine(self):
        return 'llama-2-7b'
    @property
    def name(self):
        return 'LLA2_7'
    @property
    def url(self):
        return 'https://huggingface.co/meta-llama/Llama-2-7b'
    
class Llama2_13B(LlamaFamily):
    @property
    def engine(self):
        return 'llama-2-13b'
    @property
    def name(self):
        return 'LLA2_13'
    @property
    def url(self):
        return 'https://huggingface.co/meta-llama/Llama-2-13b'
    
class Llama2HF_7B(LlamaFamily):
    @property
    def engine(self):
        return 'llama-2-7b-hf'
    @property
    def name(self):
        return 'LLA2HF_7'
    @property
    def url(self):
        return 'https://huggingface.co/meta-llama/Llama-2-7b-hf'
    
class Llama2HF_13B(LlamaFamily):
    @property
    def engine(self):
        return 'llama-2-13b-hf'
    @property
    def name(self):
        return 'LLA2HF_13'
    @property
    def url(self):
        return 'https://huggingface.co/meta-llama/Llama-2-13b-hf'
    
"""class Llama2GGUF_7B(LlamaFamily):
    def load_model(self):
        from transformers import AutoTokenizer,AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(HFMODELSPATH,self.engine))
        #tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(os.path.join(HFMODELSPATH,self.engine),model_file="llama-2-7b-gguf.Q3_K_S.gguf")
        return tokenizer,model
    @property
    def engine(self):
        return 'llama-2-7b-gguf'
    @property
    def name(self):
        return 'LLA2HF_7'
    
class Llama2GGUF_13B(LlamaFamily):
    @property
    def engine(self):
        return 'llama-2-13b-gguf'
    @property
    def name(self):
        return 'LLA2HF_13'"""
    
class Orca2_7B(LlamaFamily):
    @property
    def engine(self):
        return 'Orca-2-7b'
    @property
    def name(self):
        return 'ORC2_7'
    @property
    def url(self):
        return 'https://huggingface.co/microsoft/Orca-2-7b'
    
class Orca2_13B(LlamaFamily):
    @property
    def engine(self):
        return 'Orca-2-13b'
    @property
    def name(self):
        return 'ORC2_13'
    @property
    def url(self):
        return 'https://huggingface.co/microsoft/Orca-2-13b'
    
    
#-------------------------------------------------------------------------------------------------------------- 
#
#                                             Bloom Models
#
#---------------------------------------------------------------------------------------------------------------
    
class BloomFamily(HFModel):
    def load_model(self):
        from transformers import AutoTokenizer,BloomForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(HFMODELSPATH,self.engine))
        #tokenizer.pad_token = tokenizer.eos_token
        model = BloomForCausalLM.from_pretrained(os.path.join(HFMODELSPATH,self.engine),device_map = 'auto')
        return tokenizer,model

    
class Bloom3B(BloomFamily):
    @property
    def engine(self):
        return 'bloom-3b'
    @property
    def name(self):
        return 'BL3'
    @property
    def url(self):
        return 'https://huggingface.co/bigscience/bloom-3b'
    
class Bloom7B(BloomFamily):
    @property
    def engine(self):
        return 'bloom-7b'
    @property
    def name(self):
        return 'BL7'
    @property
    def url(self):
        return 'https://huggingface.co/bigscience/bloom-7b1'
    
class BloomZ3B(BloomFamily):
    @property
    def engine(self):
        return 'bloomz-3b'
    @property
    def name(self):
        return 'BLZ3'
    @property
    def url(self):
        return 'https://huggingface.co/bigscience/bloomz-3b'
    
class BloomZ7B(BloomFamily):
    @property
    def engine(self):
        return 'bloomz-7b'
    @property
    def name(self):
        return 'BLZ7'
    @property
    def url(self):
        return 'https://huggingface.co/bigscience/bloomz-7b1'
    
class Bloom176B(BloomFamily):
    def load_model(self):
        from transformers import AutoTokenizer,BloomForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(HFMODELSPATH,self.engine))
        #tokenizer.pad_token = tokenizer.eos_token
        model = BloomForCausalLM.from_pretrained(os.path.join(HFMODELSPATH,self.engine),device_map = 'auto',torch_dtype=torch.bfloat16)
        return tokenizer,model
    @property
    def engine(self):
        return 'bloom-176b'
    @property
    def name(self):
        return 'BL176'
    @property
    def url(self):
        return 'https://huggingface.co/bigscience/bloom'
    
class BloomZ176B(BloomFamily):
    def load_model(self):
        from transformers import AutoTokenizer,BloomForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(HFMODELSPATH,self.engine))
        #tokenizer.pad_token = tokenizer.eos_token
        model = BloomForCausalLM.from_pretrained(os.path.join(HFMODELSPATH,self.engine),device_map = 'auto',torch_dtype=torch.bfloat16)
        return tokenizer,model
    @property
    def engine(self):
        return 'bloomz-176b'
    @property
    def name(self):
        return 'BLZ176'
    @property
    def url(self):
        return 'https://huggingface.co/bigscience/bloomz'
    
    
#-------------------------------------------------------------------------------------------------------------- 
#
#                                             Pythia Models
#
#---------------------------------------------------------------------------------------------------------------

class PythiaFamily(HFModel):
    def load_model(self):
        from transformers import AutoTokenizer,GPTNeoXForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(HFMODELSPATH,self.engine))
        tokenizer.pad_token = tokenizer.eos_token
        model = GPTNeoXForCausalLM.from_pretrained(os.path.join(HFMODELSPATH,self.engine),device_map = 'auto')
        return tokenizer,model


class Pythia70M(PythiaFamily):
    @property
    def engine(self):
        return 'pythia-70m'
    @property
    def name(self):
        return 'PY70m'
    @property
    def url(self):
        return 'https://huggingface.co/EleutherAI/pythia-70m'
    
class Pythia160M(PythiaFamily):
    @property
    def engine(self):
        return 'pythia-160m'
    @property
    def name(self):
        return 'PY160m'
    @property
    def url(self):
        return 'https://huggingface.co/EleutherAI/pythia-160m'
    
class Pythia410M(PythiaFamily):
    @property
    def engine(self):
        return 'pythia-410m'
    @property
    def name(self):
        return 'PY410m'
    @property
    def url(self):
        return 'https://huggingface.co/EleutherAI/pythia-410m'
    
class Pythia1B(PythiaFamily):
    @property
    def engine(self):
        return 'pythia-1.4b'
    @property
    def name(self):
        return 'PY1'
    @property
    def url(self):
        return 'https://huggingface.co/EleutherAI/pythia-1.4b'

class Pythia3B(PythiaFamily):
    @property
    def engine(self):
        return 'pythia-2.8b'
    @property
    def name(self):
        return 'PY3'
    @property
    def url(self):
        return 'https://huggingface.co/EleutherAI/pythia-2.8b'
    
class Pythia7B(PythiaFamily):
    @property
    def engine(self):
        return 'pythia-6.9b'
    @property
    def name(self):
        return 'PY7'
    @property
    def url(self):
        return 'https://huggingface.co/EleutherAI/pythia-6.9b'
    
class Pythia12B(PythiaFamily):
    @property
    def engine(self):
        return 'pythia-12b'
    @property
    def name(self):
        return 'PY12'
    @property
    def url(self):
        return 'https://huggingface.co/EleutherAI/pythia-12b'
    
class Dolly3B(PythiaFamily):
    @property
    def engine(self):
        return 'dolly-v2-3b'
    @property
    def name(self):
        return 'DL3'
    @property
    def url(self):
        return 'https://huggingface.co/databricks/dolly-v2-3b'
    
class Dolly7B(PythiaFamily):
    @property
    def engine(self):
        return 'dolly-v2-7b'
    @property
    def name(self):
        return 'DL7'
    @property
    def url(self):
        return 'https://huggingface.co/databricks/dolly-v2-7b'
    
class Dolly12B(PythiaFamily):
    @property
    def engine(self):
        return 'dolly-v2-12b'
    @property
    def name(self):
        return 'DL12'
    @property
    def url(self):
        return 'https://huggingface.co/databricks/dolly-v2-12b'
    
#-------------------------------------------------------------------------------------------------------------- 
#
#                                             PHI Models
#
#---------------------------------------------------------------------------------------------------------------
    
from typing import Any, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
@dataclass
class InferenceParams:
    """Inference parameters passed to model to efficiently calculate
    and store context during inference.
    Reference:
        https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/utils/generation.py.
    Args:
        max_sequence_len: Maximum sequence length.
        max_batch_size: Maximum batch size.
        sequence_len_offset: Sequence length offset.
        batch_size_offset: Batch size offset.
        key_value_memory_dict: Key value memory dictionary.
        fused_ft_kernel: Whether to use fused kernel for fast inference.
        lengths_per_sample: Lengths per sample.
    """

    max_sequence_len: int = field(metadata={"help": "Maximum sequence length."})

    max_batch_size: int = field(metadata={"help": "Maximum batch size."})

    sequence_len_offset: int = field(default=0, metadata={"help": "Sequence length offset."})

    batch_size_offset: int = field(default=0, metadata={"help": "Batch size offset."})

    key_value_memory_dict: Dict[str, Any] = field(
        default_factory=dict, metadata={"help": "Key value memory dictionary."}
    )

    fused_ft_kernel: bool = field(default=False, metadata={"help": "Whether to use fused kernel for fast inference."})

    lengths_per_sample: torch.Tensor = field(default=None, metadata={"help": "Lengths per sample."})
    
class PhiFamily(HFModel):
    def load_model(self):
        from transformers import AutoTokenizer,AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(HFMODELSPATH,self.engine), trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(os.path.join(HFMODELSPATH,self.engine), trust_remote_code=True,device_map = 'auto', torch_dtype="auto")
        return tokenizer,model
    
class Phi1(PhiFamily):
    @property
    def engine(self):
        return 'phi-1'
    @property
    def name(self):
        return 'PHI_1'
    
class Phi15(PhiFamily):
    @property
    def engine(self):
        return 'phi-1_5'
    @property
    def name(self):
        return 'PHI_1.5'
    def init_kv(self,inputs,model):
        return InferenceParams(
                max_batch_size=inputs.shape[0],
                max_sequence_len=model.config.n_positions,
                sequence_len_offset=0,
                batch_size_offset=0,
                fused_ft_kernel=False,
                key_value_memory_dict={},
            )
    
class Phi2(PhiFamily):
    @property
    def engine(self):
        return 'phi-2'
    @property
    def name(self):
        return 'PHI_2'
    
#-------------------------------------------------------------------------------------------------------------- 
#
#                                             StableLM Models
#
#---------------------------------------------------------------------------------------------------------------

class StableLMFamily(HFModel):
    def load_model(self):
        from transformers import AutoTokenizer,AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(HFMODELSPATH,self.engine))
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(os.path.join(HFMODELSPATH,self.engine))
        return tokenizer,model
    
class StableLM3B(StableLMFamily):
    @property
    def engine(self):
        return 'stablelm-base-alpha-3b'
    @property
    def name(self):
        return 'SLM_3'
    
class StableLM7B(StableLMFamily):
    @property
    def engine(self):
        return 'stablelm-base-alpha-7b'
    @property
    def name(self):
        return 'SLM_7'
    
class StableLMT3B(StableLMFamily):
    @property
    def engine(self):
        return 'stablelm-tuned-alpha-3b'
    @property
    def name(self):
        return 'SLMT_3'
    
class StableLMT7B(StableLMFamily):
    @property
    def engine(self):
        return 'stablelm-tuned-alpha-7b'
    @property
    def name(self):
        return 'SLMT_7'

#-------------------------------------------------------------------------------------------------------------- 
#
#                                             Cerebras Models
#
#---------------------------------------------------------------------------------------------------------------
    
class CerebrasFamily(HFModel):
    def load_model(self):
        from transformers import AutoTokenizer,GPT2LMHeadModel
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(HFMODELSPATH,self.engine))
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(os.path.join(HFMODELSPATH,self.engine))
        return tokenizer,model
    
class Cerebras111M(CerebrasFamily):
    @property
    def engine(self):
        return 'cerebras-111m'
    @property
    def name(self):
        return 'CER_111M'
    
class Cerebras256M(CerebrasFamily):
    @property
    def engine(self):
        return 'cerebras-256m'
    @property
    def name(self):
        return 'CER_256M'
    
class Cerebras590M(CerebrasFamily):
    @property
    def engine(self):
        return 'cerebras-590m'
    @property
    def name(self):
        return 'CER_590M'
    
class Cerebras1B(CerebrasFamily):
    @property
    def engine(self):
        return 'cerebras-1.3b'
    @property
    def name(self):
        return 'CER_1'
    
class Cerebras3B(CerebrasFamily):
    @property
    def engine(self):
        return 'cerebras-2.7b'
    @property
    def name(self):
        return 'CER_3'
    
class Cerebras7B(CerebrasFamily):
    @property
    def engine(self):
        return 'cerebras-6.7b'
    @property
    def name(self):
        return 'CER_7'
    
class Cerebras13B(CerebrasFamily):
    @property
    def engine(self):
        return 'cerebras-13b'
    @property
    def name(self):
        return 'CER_13'
    
#-------------------------------------------------------------------------------------------------------------- 
#
#                                             QWEN Models
#
#---------------------------------------------------------------------------------------------------------------
    
class QWENFamily(HFModel):
    def load_model(self):
        from transformers import AutoTokenizer,AutoModelForCausalLM
        #tokenizer = AutoTokenizer.from_pretrained(os.path.join(HFMODELSPATH,self.engine),trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(HFMODELSPATH,self.engine),
                                                  trust_remote_code=True,
                                                  pad_token='<|extra_0|>',
                                                  eos_token='<|endoftext|>')
        #tokenizer.pad_token = tokenizer.eos_token
        #tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        model = AutoModelForCausalLM.from_pretrained(os.path.join(HFMODELSPATH,self.engine),
                                                     trust_remote_code=True,
                                                     device_map = 'auto',
                                                     pad_token_id=tokenizer.pad_token_id)
        model.generation_config = GenerationConfig.from_pretrained(os.path.join(HFMODELSPATH,self.engine), pad_token_id=tokenizer.pad_token_id)
        return tokenizer,model

    
class QWEN2B(QWENFamily):
    @property
    def engine(self):
        return 'Qwen-1_8B'
    @property
    def name(self):
        return 'QWE_2'
    @property
    def url(self):
        return 'https://huggingface.co/Qwen/Qwen-1_8B'
    
class QWEN7B(QWENFamily):
    @property
    def engine(self):
        return 'Qwen-7B'
    @property
    def name(self):
        return 'QWE_7'
    @property
    def url(self):
        return 'https://huggingface.co/Qwen/Qwen-7B'
    
class QWEN14B(QWENFamily):
    @property
    def engine(self):
        return 'Qwen-14B'
    @property
    def name(self):
        return 'QWE_14'
    @property
    def url(self):
        return 'https://huggingface.co/Qwen/Qwen-14B'
    
class QWEN72B(QWENFamily):
    @property
    def engine(self):
        return 'Qwen-72B'
    @property
    def name(self):
        return 'QWE_72'
    @property
    def url(self):
        return 'https://huggingface.co/Qwen/Qwen-72B'
    
    
class QWEN15_500M(QWENFamily):
    @property
    def engine(self):
        return 'Qwen1.5-0.5B'
    @property
    def name(self):
        return 'QWE15_500m'
    @property
    def url(self):
        return 'https://huggingface.co/Qwen/Qwen1.5-0.5B'
    
class QWEN15_2B(QWENFamily):
    @property
    def engine(self):
        return 'Qwen1.5-1.8B'
    @property
    def name(self):
        return 'QWE15_2B'
    @property
    def url(self):
        return 'https://huggingface.co/Qwen/Qwen1.5-1.8B'
    
class QWEN15_4B(QWENFamily):
    @property
    def engine(self):
        return 'Qwen1.5-4B'
    @property
    def name(self):
        return 'QWE15_4'
    @property
    def url(self):
        return 'https://huggingface.co/Qwen/Qwen1.5-4B'
    
class QWEN15_7B(QWENFamily):
    @property
    def engine(self):
        return 'Qwen1.5-7B'
    @property
    def name(self):
        return 'QWE15_7'
    @property
    def url(self):
        return 'https://huggingface.co/Qwen/Qwen1.5-7B'
    
class QWEN15_14B(QWENFamily):
    @property
    def engine(self):
        return 'Qwen1.5-14B'
    @property
    def name(self):
        return 'QWE15_14'
    @property
    def url(self):
        return 'https://huggingface.co/Qwen/Qwen1.5-14B'
    
class QWEN15_32B(QWENFamily):
    @property
    def engine(self):
        return 'Qwen1.5-32B'
    @property
    def name(self):
        return 'QWE15_32'
    @property
    def url(self):
        return 'https://huggingface.co/Qwen/Qwen1.5-32B'
    
class QWEN15_72B(QWENFamily):
    @property
    def engine(self):
        return 'Qwen1.5-72B'
    @property
    def name(self):
        return 'QWE15_72'
    @property
    def url(self):
        return 'https://huggingface.co/Qwen/Qwen1.5-72B'
    
    
class QWEN15Ch_500M(QWENFamily):
    @property
    def engine(self):
        return 'Qwen1.5-0.5B-Chat'
    @property
    def name(self):
        return 'QWE15Ch_500m'
    @property
    def url(self):
        return 'https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat'
    
class QWEN15Ch_2B(QWENFamily):
    @property
    def engine(self):
        return 'Qwen1.5-1.8B-Chat'
    @property
    def name(self):
        return 'QWE15Ch_2B'
    @property
    def url(self):
        return 'https://huggingface.co/Qwen/Qwen1.5-2B-Chat'
    
class QWEN15Ch_4B(QWENFamily):
    @property
    def engine(self):
        return 'Qwen1.5-4B-Chat'
    @property
    def name(self):
        return 'QWE15Ch_4'
    @property
    def url(self):
        return 'https://huggingface.co/Qwen/Qwen1.5-4B-Chat'
    
class QWEN15Ch_7B(QWENFamily):
    @property
    def engine(self):
        return 'Qwen1.5-7B-Chat'
    @property
    def name(self):
        return 'QWE15Ch_7'
    @property
    def url(self):
        return 'https://huggingface.co/Qwen/Qwen1.5-7B-Chat'
    
class QWEN15Ch_14B(QWENFamily):
    @property
    def engine(self):
        return 'Qwen1.5-14B-Chat'
    @property
    def name(self):
        return 'QWE15Ch_14'
    @property
    def url(self):
        return 'https://huggingface.co/Qwen/Qwen1.5-14B-Chat'
    
class QWEN15Ch_32B(QWENFamily):
    @property
    def engine(self):
        return 'Qwen1.5-32B-Chat'
    @property
    def name(self):
        return 'QWE15Ch_32'
    @property
    def url(self):
        return 'https://huggingface.co/Qwen/Qwen1.5-32B-Chat'
    
class QWEN15Ch_72B(QWENFamily):
    @property
    def engine(self):
        return 'Qwen1.5-72B-Chat'
    @property
    def name(self):
        return 'QWE15Ch_72'
    @property
    def url(self):
        return 'https://huggingface.co/Qwen/Qwen1.5-72B-Chat'
    

class QWEN15Ch4b_500M(QWENFamily):
    @property
    def engine(self):
        return 'Qwen1.5-0.5B-Chat-GPTQ-Int4'
    @property
    def name(self):
        return 'QWE15Ch4b_500m'
    
class QWEN15Ch4b_2B(QWENFamily):
    @property
    def engine(self):
        return 'Qwen1.5-1.8B-Chat-GPTQ-Int4'
    @property
    def name(self):
        return 'QWE15Ch4b_2B'
    
class QWEN15Ch4b_4B(QWENFamily):
    @property
    def engine(self):
        return 'Qwen1.5-4B-Chat-GPTQ-Int4'
    @property
    def name(self):
        return 'QWE15Ch4b_4'
    
class QWEN15Ch4b_7B(QWENFamily):
    @property
    def engine(self):
        return 'Qwen1.5-7B-Chat-GPTQ-Int4'
    @property
    def name(self):
        return 'QWE15Ch4b_7'
    
class QWEN15Ch4b_14B(QWENFamily):
    @property
    def engine(self):
        return 'Qwen1.5-14B-Chat-GPTQ-Int4'
    @property
    def name(self):
        return 'QWE15Ch4b_14'
    
class QWEN15Ch4b_72B(QWENFamily):
    @property
    def engine(self):
        return 'Qwen1.5-72B-Chat-GPTQ-Int4'
    @property
    def name(self):
        return 'QWE15Ch_72'
    
    
class QWEN15ChAWQ_500M(QWENFamily):
    @property
    def engine(self):
        return 'Qwen1.5-0.5B-Chat-AWQ'
    @property
    def name(self):
        return 'QWE15ChAWQ_500m'
    
class QWEN15ChAWQ_2B(QWENFamily):
    @property
    def engine(self):
        return 'Qwen1.5-1.8B-Chat-AWQ'
    @property
    def name(self):
        return 'QWE15ChAWQ_2B'
    
class QWEN15ChAWQ_4B(QWENFamily):
    @property
    def engine(self):
        return 'Qwen1.5-4B-Chat-AWQ'
    @property
    def name(self):
        return 'QWE15ChAWQ_4'
    
class QWEN15ChAWQ_7B(QWENFamily):
    @property
    def engine(self):
        return 'Qwen1.5-7B-Chat-AWQ'
    @property
    def name(self):
        return 'QWE15ChAWQ_7'
    
class QWEN15ChAWQ_14B(QWENFamily):
    @property
    def engine(self):
        return 'Qwen1.5-14B-Chat-AWQ'
    @property
    def name(self):
        return 'QWE15ChAWQ_14'
    
class QWEN15ChAWQ_72B(QWENFamily):
    @property
    def engine(self):
        return 'Qwen1.5-72B-Chat-AWQ'
    @property
    def name(self):
        return 'QWE15ChAWQ_72'
    
    
class QWEN15Ch8b_500M(QWENFamily):
    @property
    def engine(self):
        return 'Qwen1.5-0.5B-Chat-GPTQ-Int8'
    @property
    def name(self):
        return 'QWE15Ch8b_500m'
    
class QWEN15Ch8b_2B(QWENFamily):
    @property
    def engine(self):
        return 'Qwen1.5-1.8B-Chat-GPTQ-Int8'
    @property
    def name(self):
        return 'QWE15Ch8b_2B'
    
class QWEN15Ch8b_4B(QWENFamily):
    @property
    def engine(self):
        return 'Qwen1.5-4B-Chat-GPTQ-Int8'
    @property
    def name(self):
        return 'QWE15Ch8b_4'
    
class QWEN15Ch8b_7B(QWENFamily):
    @property
    def engine(self):
        return 'Qwen1.5-7B-Chat-GPTQ-Int8'
    @property
    def name(self):
        return 'QWE15Ch8b_7'
    
class QWEN15Ch8b_14B(QWENFamily):
    @property
    def engine(self):
        return 'Qwen1.5-14B-Chat-GPTQ-Int8'
    @property
    def name(self):
        return 'QWE15Ch8b_14'
    
class QWEN15Ch8b_72B(QWENFamily):
    @property
    def engine(self):
        return 'Qwen1.5-72B-Chat-GPTQ-Int8'
    @property
    def name(self):
        return 'QWE15Ch_72'
    
    
class QWEN15MoE_3B(QWENFamily):
    @property
    def engine(self):
        return 'Qwen1.5-MoE-A2.7B'
    @property
    def name(self):
        return 'QWE15MoE_3'
    
class QWEN15MoECh_3B(QWENFamily):
    @property
    def engine(self):
        return 'Qwen1.5-MoE-A2.7B-Chat'
    @property
    def name(self):
        return 'QWE15MoECh_3'
    
class QWEN15MoECh4b_3B(QWENFamily):
    @property
    def engine(self):
        return 'Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4'
    @property
    def name(self):
        return 'QWE15MoECh4b_3'
    
#-------------------------------------------------------------------------------------------------------------- 
#
#                                             YI Models
#
#---------------------------------------------------------------------------------------------------------------

class YiFamily(HFModel):
    def load_model(self):
        from transformers import AutoTokenizer,AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(HFMODELSPATH,self.engine))
        #tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(os.path.join(HFMODELSPATH,self.engine),device_map = 'auto')
        return tokenizer,model
    
class Yi6B(YiFamily):
    @property
    def engine(self):
        return 'Yi-6B'
    @property
    def name(self):
        return 'YI_6'
    
class Yi34B(YiFamily):
    @property
    def engine(self):
        return 'Yi-34B'
    @property
    def name(self):
        return 'YI_34'

"""
class Yi6B_GGUF(YiFamily):
    @property
    def engine(self):
        return 'Yi-6B-GGUF'
    @property
    def name(self):
        return 'YI_6_GGUF'
    
class Yi34B_GGUF(YiFamily):
    @property
    def engine(self):
        return 'Yi-34B-GGUF'
    @property
    def name(self):
        return 'YI_34_GGUF'
"""
    
#-------------------------------------------------------------------------------------------------------------- 
#
#                                             Mistral Models
#
#---------------------------------------------------------------------------------------------------------------

class MistralFamily(HFModel):
    def load_model(self):
        from transformers import AutoTokenizer,AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(HFMODELSPATH,self.engine))
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(os.path.join(HFMODELSPATH,self.engine),device_map = 'auto')
        return tokenizer,model
    
class Mistral7B(MistralFamily):
    @property
    def engine(self):
        return 'Mistral-7B-v0.1'
    @property
    def name(self):
        return 'MIS_7'
    @property
    def url(self):
        return 'https://huggingface.co/mistralai/Mistral-7B-v0.1'
    
class Mistral7BI(MistralFamily):
    @property
    def engine(self):
        return 'Mistral-7B-Instruct-v0.1'
    @property
    def name(self):
        return 'MISI_7'
    @property
    def url(self):
        return 'https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1'
    
class Zephyr7BA(MistralFamily):
    @property
    def engine(self):
        return 'zephyr-7b-alpha'
    @property
    def name(self):
        return 'ZPHA_7'
    @property
    def url(self):
        return 'https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha'
    
class Zephyr7BB(MistralFamily):
    @property
    def engine(self):
        return 'zephyr-7b-beta'
    @property
    def name(self):
        return 'ZPHB_7'
    @property
    def url(self):
        return 'https://huggingface.co/HuggingFaceH4/zephyr-7b-beta'
    
#-------------------------------------------------------------------------------------------------------------- 
#
#                                             FUYU Models
#
#---------------------------------------------------------------------------------------------------------------

class FUYUFamily(HFModel):
    def load_model(self):
        from transformers import AutoTokenizer,AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(HFMODELSPATH,self.engine))
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(os.path.join(HFMODELSPATH,self.engine),device_map = 'auto')
        return tokenizer,model
    
class FUYU8B(FUYUFamily):
    @property
    def engine(self):
        return 'fuyu-8b'
    @property
    def name(self):
        return 'FUYU8B'
    
#-------------------------------------------------------------------------------------------------------------- 
#
#                                             Falcon Models
#
#---------------------------------------------------------------------------------------------------------------

class FalconFamily(HFModel):
    def load_model(self):
        from transformers import AutoTokenizer,AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(HFMODELSPATH,self.engine))
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(os.path.join(HFMODELSPATH,self.engine),device_map = 'auto')#,torch_dtype=torch.float16)#,trust_remote_code=True)
        return tokenizer,model
    
class FalconRW1B(FalconFamily):
    @property
    def engine(self):
        return 'falcon-rw-1b'
    @property
    def name(self):
        return 'FALRW1B'
    @property
    def url(self):
        return 'https://huggingface.co/tiiuae/falcon-rw-1b'
    
class FalconRW7B(FalconFamily):
    @property
    def engine(self):
        return 'falcon-rw-7b'
    @property
    def name(self):
        return 'FALRW7B'
    @property
    def url(self):
        return 'https://huggingface.co/tiiuae/falcon-rw-7b'
    
class Falcon7B(FalconFamily):
    @property
    def engine(self):
        return 'falcon-7b'
    @property
    def name(self):
        return 'FAL7B'
    @property
    def url(self):
        return 'https://huggingface.co/tiiuae/falcon-7b'
    
class Falcon7BI(FalconFamily):
    @property
    def engine(self):
        return 'falcon-7b-instruct'
    @property
    def name(self):
        return 'FAL7BI'
    @property
    def url(self):
        return 'https://huggingface.co/tiiuae/falcon-7b-instruct'
    
class Falcon40B(FalconFamily):
    @property
    def engine(self):
        return 'falcon-40b'
    @property
    def name(self):
        return 'FAL40B'
    @property
    def url(self):
        return 'https://huggingface.co/tiiuae/falcon-40b'
    
class Falcon40BI(FalconFamily):
    @property
    def engine(self):
        return 'falcon-40b-instruct'
    @property
    def name(self):
        return 'FAL40BI'
    @property
    def url(self):
        return 'https://huggingface.co/tiiuae/falcon-40b-instruct'
    
class Falcon180B(FalconFamily):
    def load_model(self):
        from transformers import AutoTokenizer,AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(HFMODELSPATH,self.engine))
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(os.path.join(HFMODELSPATH,self.engine),device_map = 'auto',torch_dtype=torch.bfloat16)#,trust_remote_code=True)
        return tokenizer,model
    @property
    def engine(self):
        return 'falcon-180b'
    @property
    def name(self):
        return 'FAL180B'
    @property
    def url(self):
        return 'https://huggingface.co/tiiuae/falcon-180B'
    
#-------------------------------------------------------------------------------------------------------------- 
#
#                                             OPT Models
#
#---------------------------------------------------------------------------------------------------------------

class OPTFamily(HFModel):
    def load_model(self):
        from transformers import AutoTokenizer,AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(HFMODELSPATH,self.engine))
        #tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(os.path.join(HFMODELSPATH,self.engine),device_map = 'auto')
        return tokenizer,model
    
class OPT125M(OPTFamily):
    @property
    def engine(self):
        return 'opt-125m'
    @property
    def name(self):
        return 'OPT125M'
    @property
    def url(self):
        return 'https://huggingface.co/facebook/opt-125m'
    
class OPT350M(OPTFamily):
    @property
    def engine(self):
        return 'opt-350m'
    @property
    def name(self):
        return 'OPT350M'
    @property
    def url(self):
        return 'https://huggingface.co/facebook/opt-350m'
    
class OPT1B(OPTFamily):
    @property
    def engine(self):
        return 'opt-1.3b'
    @property
    def name(self):
        return 'OPT1B'
    @property
    def url(self):
        return 'https://huggingface.co/facebook/opt-1.3b'
    
class OPT3B(OPTFamily):
    @property
    def engine(self):
        return 'opt-2.7b'
    @property
    def name(self):
        return 'OPT3B'
    @property
    def url(self):
        return 'https://huggingface.co/facebook/opt-2.7b'
    
class OPT7B(OPTFamily):
    @property
    def engine(self):
        return 'opt-6.7b'
    @property
    def name(self):
        return 'OPT7B'
    @property
    def url(self):
        return 'https://huggingface.co/facebook/opt-6.7b'
    
class OPT13B(OPTFamily):
    @property
    def engine(self):
        return 'opt-13b'
    @property
    def name(self):
        return 'OPT13B'
    @property
    def url(self):
        return 'https://huggingface.co/facebook/opt-13b'
    
class OPT30B(OPTFamily):
    @property
    def engine(self):
        return 'opt-30b'
    @property
    def name(self):
        return 'OPT30B'
    @property
    def url(self):
        return 'https://huggingface.co/facebook/opt-30b'
    
class OPT66B(OPTFamily):
    @property
    def engine(self):
        return 'opt-66b'
    @property
    def name(self):
        return 'OPT66B'
    @property
    def url(self):
        return 'https://huggingface.co/facebook/opt-66b'
    
#-------------------------------------------------------------------------------------------------------------- 
#
#                                             OpenChat Models
#
#---------------------------------------------------------------------------------------------------------------

class GemmaFamily(LlamaFamily):
    pass

class Gemma7B(GemmaFamily):
    @property
    def engine(self):
        return 'gemma-7b'
    @property
    def name(self):
        return 'G7B'
    @property
    def url(self):
        return 'https://huggingface.co/google/gemma-7b'
    
class Gemma2B(GemmaFamily):
    @property
    def engine(self):
        return 'gemma-2b'
    @property
    def name(self):
        return 'G2B'
    @property
    def url(self):
        return 'https://huggingface.co/google/gemma-2b'
    
class Gemma7BI(GemmaFamily):
    @property
    def engine(self):
        return 'gemma-7b-it'
    @property
    def name(self):
        return 'G7BI'
    @property
    def url(self):
        return 'https://huggingface.co/google/gemma-7b-it'
    
class Gemma2BI(GemmaFamily):
    @property
    def engine(self):
        return 'gemma-2b-it'
    @property
    def name(self):
        return 'G2BI'
    @property
    def url(self):
        return 'https://huggingface.co/google/gemma-2b-it'
    
class Gemma1p1_2BI(GemmaFamily):
    @property
    def engine(self):
        return 'gemma-1.1-2b-it'
    @property
    def name(self):
        return 'G1p1_2BI'
    @property
    def url(self):
        return 'https://huggingface.co/google/gemma-1.1-2b-it'
    
class Gemma1p1_7BI(GemmaFamily):
    @property
    def engine(self):
        return 'gemma-1.1-7b-it'
    @property
    def name(self):
        return 'G1p1_7BI'
    @property
    def url(self):
        return 'https://huggingface.co/google/gemma-1.1-7b-it'
    
class CodeGemma2B(GemmaFamily):
    @property
    def engine(self):
        return 'codegemma-2b'
    @property
    def name(self):
        return 'CG2B'
    @property
    def url(self):
        return 'https://huggingface.co/google/codegemma-2b'
    
class CodeGemma7B(GemmaFamily):
    @property
    def engine(self):
        return 'codegemma-7b'
    @property
    def name(self):
        return 'CG7B'
    @property
    def url(self):
        return 'https://huggingface.co/google/codegemma-7b'
    
class CodeGemma7BI(GemmaFamily):
    @property
    def engine(self):
        return 'codegemma-7b-it'
    @property
    def name(self):
        return 'CG7BI'
    @property
    def url(self):
        return 'https://huggingface.co/google/codegemma-7b-it'
    
class ReccurentGemma2B(GemmaFamily):
    @property
    def engine(self):
        return 'recurrentgemma-2b'
    @property
    def name(self):
        return 'RG2B'
    
class ReccurentGemma2BI(GemmaFamily):
    @property
    def engine(self):
        return 'recurrentgemma-2b-it'
    @property
    def name(self):
        return 'RG2BI'
    
#-------------------------------------------------------------------------------------------------------------- 
#
#                                             Cohere Models
#
#---------------------------------------------------------------------------------------------------------------
class CohereFamily(LlamaFamily):
    def load_model(self):
        from transformers import AutoTokenizer,AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(HFMODELSPATH,self.engine))
        #tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(os.path.join(HFMODELSPATH,self.engine),device_map = 'auto',torch_dtype=torch.float16)
        return tokenizer,model

class CohereCommandR1(CohereFamily):
    @property
    def engine(self):
        return 'c4ai-command-r-v01'
    @property
    def name(self):
        return 'CCR1'
    
class CohereCommandRP(CohereFamily):
    @property
    def engine(self):
        return 'c4ai-command-r-plus'
    @property
    def name(self):
        return 'CCRP'
    
class CohereCommandR1_4b(CohereFamily):
    @property
    def engine(self):
        return 'c4ai-command-r-v01-4bit'
    @property
    def name(self):
        return 'CCR1Q'
    
class CohereCommandRP_4b(CohereFamily):
    @property
    def engine(self):
        return 'c4ai-command-r-plus-4bit'
    @property
    def name(self):
        return 'CCRPQ'
    
#-------------------------------------------------------------------------------------------------------------- 
#
#                                             OpenChat Models
#
#---------------------------------------------------------------------------------------------------------------
    
class OpenChatFamily(MistralFamily):
    pass
    
class OpenChat2(OpenChatFamily):
    @property
    def engine(self):
        return 'openchat_v2'
    @property
    def name(self):
        return 'OC2'
    @property
    def url(self):
        return 'https://huggingface.co/openchat/openchat_v2'
    
class OpenChat2W(OpenChatFamily):
    @property
    def engine(self):
        return 'openchat_v2_w'
    @property
    def name(self):
        return 'OC2W'
    @property
    def url(self):
        return 'https://huggingface.co/openchat/openchat_v2_w'
    
class OpenChat31(OpenChatFamily):
    @property
    def engine(self):
        return 'openchat_v3.1'
    @property
    def name(self):
        return 'OC31'
    @property
    def url(self):
        return 'https://huggingface.co/openchat/openchat_v3.1'
    
class OpenChat32(OpenChatFamily):
    @property
    def engine(self):
        return 'openchat_v3.2'
    @property
    def name(self):
        return 'OC32'
    @property
    def url(self):
        return 'https://huggingface.co/openchat/openchat_v3.2'
    
class OpenChat32Super(OpenChatFamily):
    @property
    def engine(self):
        return 'openchat_v3.2_super'
    @property
    def name(self):
        return 'OC32S'
    @property
    def url(self):
        return 'https://huggingface.co/openchat/openchat_v3.2_super'
    
class OpenChat35(OpenChatFamily):
    @property
    def engine(self):
        return 'openchat_3.5'
    @property
    def name(self):
        return 'OC35'
    @property
    def url(self):
        return 'https://huggingface.co/openchat/openchat_3.5'
    
#-------------------------------------------------------------------------------------------------------------- 
#
#                                             Berkeley-Nest Models
#
#---------------------------------------------------------------------------------------------------------------
class BerkeleyNestFamily(OpenChatFamily):
    pass
    
class Starling7BAlpha(BerkeleyNestFamily):
    @property
    def engine(self):
        return 'Starling-LM-7B-alpha'
    @property
    def name(self):
        return 'STL7BA'
    @property
    def url(self):
        return 'https://huggingface.co/berkeley-nest/Starling-LM-7B-alpha'
    
#-------------------------------------------------------------------------------------------------------------- 
#
#                                             NeuralChat Models
#
#---------------------------------------------------------------------------------------------------------------
class NeuralChatFamily(MistralFamily):
    pass
    
class NeuralChat3_7B(NeuralChatFamily):
    @property
    def engine(self):
        return 'neural-chat-7b-v3'
    @property
    def name(self):
        return 'NC3_7B'
    @property
    def url(self):
        return 'https://huggingface.co/Intel/neural-chat-7b-v3'
    
class NeuralChat31_7B(NeuralChatFamily):
    @property
    def engine(self):
        return 'neural-chat-7b-v3-1'
    @property
    def name(self):
        return 'NC31_7B'
    @property
    def url(self):
        return 'https://huggingface.co/Intel/neural-chat-7b-v3-1'
    
#-------------------------------------------------------------------------------------------------------------- 
#
#                                             CausalLM Models
#
#---------------------------------------------------------------------------------------------------------------
class CausalLMFamily(LlamaFamily):
    pass
    
class CausalLM7B(CausalLMFamily):
    @property
    def engine(self):
        return 'causallm-7b'
    @property
    def name(self):
        return 'CLM_7B'
    @property
    def url(self):
        return 'https://huggingface.co/CausalLM/7B'
    
class CausalLM14B(CausalLMFamily):
    @property
    def engine(self):
        return 'causallm-14b'
    @property
    def name(self):
        return 'CLM_14B'
    @property
    def url(self):
        return 'https://huggingface.co/CausalLM/14B'
    
#-------------------------------------------------------------------------------------------------------------- 
#
#                                             TigerBot Models
#
#---------------------------------------------------------------------------------------------------------------
class TigerBotFamily(LlamaFamily):
    pass

class TigerBotBase_7B(TigerBotFamily):
    @property
    def engine(self):
        return 'tigerbot-7b-base'
    @property
    def name(self):
        return 'TBB_7B'
    @property
    def url(self):
        return 'https://huggingface.co/TigerResearch/tigerbot-7b-base'
    
class TigerBotBasev1_7B(TigerBotFamily):
    @property
    def engine(self):
        return 'tigerbot-7b-base-v1'
    @property
    def name(self):
        return 'TBBV1_7B'
    @property
    def url(self):
        return 'https://huggingface.co/TigerResearch/tigerbot-7b-base-v1'
    
class TigerBotBasev2_7B(TigerBotFamily):
    @property
    def engine(self):
        return 'tigerbot-7b-base-v2'
    @property
    def name(self):
        return 'TBBV2_7B'
    @property
    def url(self):
        return 'https://huggingface.co/TigerResearch/tigerbot-7b-base-v2'
    
class TigerBotSFTv1_7B(TigerBotFamily):
    @property
    def engine(self):
        return 'tigerbot-7b-sft-v1'
    @property
    def name(self):
        return 'TBFV1_7B'
    @property
    def url(self):
        return 'https://huggingface.co/TigerResearch/tigerbot-7b-sft-v1'
    
class TigerBotSFTv2_7B(TigerBotFamily):
    @property
    def engine(self):
        return 'tigerbot-7b-sft-v2'
    @property
    def name(self):
        return 'TBFV2_7B'
    @property
    def url(self):
        return 'https://huggingface.co/TigerResearch/tigerbot-7b-sft-v2'
    
class TigerBotChat_7B(TigerBotFamily):
    @property
    def engine(self):
        return 'tigerbot-7b-chat'
    @property
    def name(self):
        return 'TBC_7B'
    @property
    def url(self):
        return 'https://huggingface.co/TigerResearch/tigerbot-7b-chat'
    
"""class TigerBotChat4B_7B(TigerBotFamily):
    @property
    def engine(self):
        return 'tigerbot-7b-chat-4bit'
    @property
    def name(self):
        return 'TBC4B_7B'
    
class TigerBotChat8B_7B(TigerBotFamily):
    @property
    def engine(self):
        return 'tigerbot-7b-chat-8bit'
    @property
    def name(self):
        return 'TBC8B_7B'"""
    
class TigerBotBasev1_13B(TigerBotFamily):
    @property
    def engine(self):
        return 'tigerbot-13b-base-v1'
    @property
    def name(self):
        return 'TBBV1_13B'
    @property
    def url(self):
        return 'https://huggingface.co/TigerResearch/tigerbot-13b-base-v1'
    
class TigerBotBasev2_13B(TigerBotFamily):
    @property
    def engine(self):
        return 'tigerbot-13b-base-v2'
    @property
    def name(self):
        return 'TBBV2_13B'
    @property
    def url(self):
        return 'https://huggingface.co/TigerResearch/tigerbot-13b-base-v2'
    
class TigerBotChatv1_13B(TigerBotFamily):
    @property
    def engine(self):
        return 'tigerbot-13b-chat-v1'
    @property
    def name(self):
        return 'TBCV1_13B'
    @property
    def url(self):
        return 'https://huggingface.co/TigerResearch/tigerbot-13b-chat-v1'

class TigerBotChatv2_13B(TigerBotFamily):
    @property
    def engine(self):
        return 'tigerbot-13b-chat-v2'
    @property
    def name(self):
        return 'TBCV2_13B'
    @property
    def url(self):
        return 'https://huggingface.co/TigerResearch/tigerbot-13b-chat-v2'
    
class TigerBotChatv3_13B(TigerBotFamily):
    @property
    def engine(self):
        return 'tigerbot-13b-chat-v3'
    @property
    def name(self):
        return 'TBCV3_13B'
    @property
    def url(self):
        return 'https://huggingface.co/TigerResearch/tigerbot-13b-chat-v3'
    
class TigerBotChatv4_13B(TigerBotFamily):
    @property
    def engine(self):
        return 'tigerbot-13b-chat-v4'
    @property
    def name(self):
        return 'TBCV4_13B'
    @property
    def url(self):
        return 'https://huggingface.co/TigerResearch/tigerbot-13b-chat-v4'
    
"""class TigerBotChatv4_13B(TigerBotFamily):
    @property
    def engine(self):
        return 'tigerbot-13b-chat-4bit-exl2'
    @property
    def name(self):
        return 'TBC4B_13B'"""
    
#-------------------------------------------------------------------------------------------------------------- 
#
#                                             OpenHermes Models
#
#---------------------------------------------------------------------------------------------------------------
class OpenHermes7B(LlamaFamily):
    @property
    def engine(self):
        return 'OpenHermes-7B'
    @property
    def name(self):
        return 'OH_7B'
    @property
    def url(self):
        return 'https://huggingface.co/teknium/OpenHermes-7B'

class OpenHermes13B(LlamaFamily):
    @property
    def engine(self):
        return 'OpenHermes-13B'
    @property
    def name(self):
        return 'OH_13B'
    @property
    def url(self):
        return 'https://huggingface.co/teknium/OpenHermes-13B'
    
class OpenHermes2_7B(MistralFamily):
    @property
    def engine(self):
        return 'OpenHermes-2-Mistral-7B'
    @property
    def name(self):
        return 'OH2_7B'
    @property
    def url(self):
        return 'https://huggingface.co/teknium/OpenHermes-2-Mistral-7B'
    
class OpenHermes25_7B(MistralFamily):
    @property
    def engine(self):
        return 'OpenHermes-2.5-Mistral-7B'
    @property
    def name(self):
        return 'OH25_7B'
    @property
    def url(self):
        return 'https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B'
    
#-------------------------------------------------------------------------------------------------------------- 
#
#                                             NexusFlow Models
#
#---------------------------------------------------------------------------------------------------------------
class NexusFlowFamily(LlamaFamily):
    pass 

class NexusRaven13B(NexusFlowFamily):
    @property
    def engine(self):
        return 'NexusRaven-13B'
    @property
    def name(self):
        return 'NR_13B'
    
class NexusRavenv2_13B(NexusFlowFamily):
    @property
    def engine(self):
        return 'NexusRaven-V2-13B'
    @property
    def name(self):
        return 'NR2_13B'
    
#---------------------------------------------------------------------------------------------------------------

class Baize13B(LlamaFamily):
    @property
    def engine(self):
        return 'baize-13b'
    @property
    def name(self):
        return 'BAI13B'
    
class BaizeLora13B(LlamaFamily):
    @property
    def engine(self):
        return 'baize-lora-13B'
    @property
    def name(self):
        return 'BAL13B'
    
class BaizeLora7B(LlamaFamily):
    @property
    def engine(self):
        return 'baize-lora-7B'
    @property
    def name(self):
        return 'BAL7B'
    
class Btlm3B8kBase(LlamaFamily):
    @property
    def engine(self):
        return 'btlm-3b-8k-base'
    @property
    def name(self):
        return 'BTLMB3B'
    
class Btlm3B8kChat(LlamaFamily):
    @property
    def engine(self):
        return 'btlm-3b-8k-chat'
    @property
    def name(self):
        return 'BTLMC3B'
    
class ChimeraInstChat13B(LlamaFamily):
    @property
    def engine(self):
        return 'chimera-inst-chat-13b'
    @property
    def name(self):
        return 'CHI13B'
    @property
    def url(self):
        return 'https://huggingface.co/Yhyu13/chimera-inst-chat-13b-hf'
    
class ChimeraInstChat7B(LlamaFamily):
    @property
    def engine(self):
        return 'chimera-inst-chat-7b'
    @property
    def name(self):
        return 'CHI7B'
    @property
    def url(self):
        return 'https://huggingface.co/Yhyu13/chimera-inst-chat-7b-hf'
    
class Chupacabra7B(LlamaFamily):
    @property
    def engine(self):
        return 'Chupacabra-7B'
    @property
    def name(self):
        return 'CHU7B'
    
class DistilabeledMarcoro147Bslerp(LlamaFamily):
    @property
    def engine(self):
        return 'distilabeled-Marcoro14-7B-slerp'
    @property
    def name(self):
        return 'DISMAR7BS'
    
class Docsgpt7BMistral(LlamaFamily):
    @property
    def engine(self):
        return 'docsgpt-7b-mistral'
    @property
    def name(self):
        return 'DOCMIS7B'
    @property
    def url(self):
        return 'https://huggingface.co/Arc53/docsgpt-7b-mistral'
    
class Dolphin221Mistral7B(LlamaFamily):
    @property
    def engine(self):
        return 'dolphin-2.2.1-mistral-7b'
    @property
    def name(self):
        return 'DOLMIS7B'
    
class DragonMistral7BV0(LlamaFamily):
    @property
    def engine(self):
        return 'dragon-mistral-7b-v0'
    @property
    def name(self):
        return 'DRAMIS7B'
    
class FrankenBeagle11B(LlamaFamily):
    @property
    def engine(self):
        return 'franken-Beagle-11B'
    @property
    def name(self):
        return 'FRAB11B'
    
class GoBruins(LlamaFamily):
    @property
    def engine(self):
        return 'go-bruins'
    @property
    def name(self):
        return 'GOB'
    
class GoBruinsV2(LlamaFamily):
    @property
    def engine(self):
        return 'go-bruins-v2'
    @property
    def name(self):
        return 'GOBV2'
    
class GreenNodeMini7BMultilingualV1olet(LlamaFamily):
    @property
    def engine(self):
        return 'GreenNode-mini-7B-multilingual-v1olet'
    @property
    def name(self):
        return 'GRNMUL7B'
    
class LeoScorpius7B(LlamaFamily):
    @property
    def engine(self):
        return 'LeoScorpius-7B'
    @property
    def name(self):
        return 'LEOS7B'
    
class LeoScorpius7BChatDPO(LlamaFamily):
    @property
    def engine(self):
        return 'LeoScorpius-7B-Chat-DPO'
    @property
    def name(self):
        return 'LEOS7BCD'
    
class MAmmoTH7BMistral(LlamaFamily):
    @property
    def engine(self):
        return 'MAmmoTH-7B-Mistral'
    @property
    def name(self):
        return 'MAMIS7B'
    
class Marcoro147Bslerp(LlamaFamily):
    @property
    def engine(self):
        return 'Marcoro14-7B-slerp'
    @property
    def name(self):
        return 'MARSLE7B'
    
class MetaMathCybertronStarling(LlamaFamily):
    @property
    def engine(self):
        return 'MetaMath-Cybertron-Starling'
    @property
    def name(self):
        return 'METSTA'
    
'''class Mistral7BInstructV01(LlamaFamily):
    @property
    def engine(self):
        return 'Mistral-7B-Instruct-v0.1'
    @property
    def name(self):
        return 'MISIN1'
    @property
    def url(self):
        return 'https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1' '''
    
class Mistral7BInstructV02(LlamaFamily):
    @property
    def engine(self):
        return 'Mistral-7B-Instruct-v0.2'
    @property
    def name(self):
        return 'MISIN2'
    @property
    def url(self):
        return 'https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2'
    
class Mistral7BMerge14V01(LlamaFamily):
    @property
    def engine(self):
        return 'Mistral-7B-Merge-14-v0.1'
    @property
    def name(self):
        return 'MISM14'
    
class Mistral7BOpenOrca(LlamaFamily):
    @property
    def engine(self):
        return 'Mistral-7B-OpenOrca'
    @property
    def name(self):
        return 'MISORC'
    
class MistralHermesCodePro7BV1(LlamaFamily):
    @property
    def engine(self):
        return 'MistralHermes-CodePro-7B-v1'
    @property
    def name(self):
        return 'MISCOP7BV1'
    
class Mixtral8x7BInstructV01(LlamaFamily):
    @property
    def engine(self):
        return 'Mixtral-8x7B-Instruct-v0.1'
    @property
    def name(self):
        return 'MIX8x7Iv1'
    @property
    def url(self):
        return 'https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1'
    
class Mixtral8x7BV01(LlamaFamily):
    @property
    def engine(self):
        return 'Mixtral-8x7B-v0.1'
    @property
    def name(self):
        return 'MIX8x7v1'
    @property
    def url(self):
        return 'https://huggingface.co/mistralai/Mixtral-8x7B-v0.1'
    
class Mixtral8x22BV01(LlamaFamily):
    @property
    def engine(self):
        return 'Mixtral-8x22B-v0.1'
    @property
    def name(self):
        return 'MIX8x22v1'
    
class Mpt7B(LlamaFamily):
    @property
    def engine(self):
        return 'mpt-7b'
    @property
    def name(self):
        return 'MPT7B'
    
class Mt0Xl(LlamaFamily):
    @property
    def engine(self):
        return 'mt0-xl'
    @property
    def name(self):
        return 'MT0XL'
    
class NeuralBeagle11B(LlamaFamily):
    @property
    def engine(self):
        return 'NeuralBeagle-11B'
    @property
    def name(self):
        return 'NEUB11B'
    
class NeuralBeagle147B(LlamaFamily):
    @property
    def engine(self):
        return 'NeuralBeagle14-7B'
    @property
    def name(self):
        return 'NEUB14B'
    
class NeuralHermes25Mistral7B(LlamaFamily):
    @property
    def engine(self):
        return 'NeuralHermes-2.5-Mistral-7B'
    @property
    def name(self):
        return 'NEUHER7B'
    @property
    def url(self):
        return 'https://huggingface.co/mlabonne/NeuralHermes-2.5-Mistral-7B'

class NexusRaven13B(LlamaFamily):
    @property
    def engine(self):
        return 'NexusRaven-13B'
    @property
    def name(self):
        return 'NEXRAV13B'
    
class NexusRavenV213B(LlamaFamily):
    @property
    def engine(self):
        return 'NexusRaven-V2-13B'
    @property
    def name(self):
        return 'NEXRAVV213B'
    
class OasstSft4Pythia12bEpoch35(LlamaFamily):
    @property
    def engine(self):
        return 'oasst-sft-4-pythia-12b-epoch-3.5'
    @property
    def name(self):
        return 'OASSFT4P'
    @property
    def url(self):
        return 'https://huggingface.co/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5'
    
class OpenHermes25NeuralChatV33Slerp(LlamaFamily):
    @property
    def engine(self):
        return 'OpenHermes-2.5-neural-chat-v3-3-Slerp'
    @property
    def name(self):
        return 'OPEHERNCV33S'
    
class Orca213B(LlamaFamily):
    @property
    def engine(self):
        return 'Orca-2-13b'
    @property
    def name(self):
        return 'ORC213B'
    
class Orca27B(LlamaFamily):
    @property
    def engine(self):
        return 'Orca-2-7b'
    @property
    def name(self):
        return 'ORC27B'
    
class PhoenixInstChat7B(LlamaFamily):
    @property
    def engine(self):
        return 'phoenix-inst-chat-7b'
    @property
    def name(self):
        return 'PHOINSC7B'
    
class RedPajamaINCITE7BBase(LlamaFamily):
    @property
    def engine(self):
        return 'RedPajama-INCITE-7B-Base'
    @property
    def name(self):
        return 'REDINC7BB'
    
class SakuraSOLARInstructDPOV2(LlamaFamily):
    @property
    def engine(self):
        return 'Sakura-SOLAR-Instruct-DPO-v2'
    @property
    def name(self):
        return 'SAKINSOLV2'
    
class SakuraSOLRCAInstructDPO(LlamaFamily):
    @property
    def engine(self):
        return 'Sakura-SOLRCA-Instruct-DPO'
    @property
    def name(self):
        return 'SAKINRCADPO'
    
class SakuraSOLRCAMathInstructDPOV1(LlamaFamily):
    @property
    def engine(self):
        return 'Sakura-SOLRCA-Math-Instruct-DPO-v1'
    @property
    def name(self):
        return 'SAKRCAMINDPOV1'
    
class SakuraSOLRCAMathInstructDPOV2(LlamaFamily):
    @property
    def engine(self):
        return 'Sakura-SOLRCA-Math-Instruct-DPO-v2'
    @property
    def name(self):
        return 'SAKRCAMINDPOV2'
    
class SauerkrautLMUNASOLARInstruct(LlamaFamily):
    @property
    def engine(self):
        return 'SauerkrautLM-UNA-SOLAR-Instruct'
    @property
    def name(self):
        return 'SAUUNASOLI'
    
class SciPhiMistral7B32k(LlamaFamily):
    @property
    def engine(self):
        return 'SciPhi-Mistral-7B-32k'
    @property
    def name(self):
        return 'SCIMIS7B32K'
    
class StableCode3B(LlamaFamily):
    @property
    def engine(self):
        return 'stable-code-3b'
    @property
    def name(self):
        return 'STACO3B'
    
class StableCodeCompletionAlpha3B(LlamaFamily):
    @property
    def engine(self):
        return 'stablecode-completion-alpha-3b'
    @property
    def name(self):
        return 'STACOA3B'
    
class StableCodeCompletionAlpha3B4k(LlamaFamily):
    @property
    def engine(self):
        return 'stablecode-completion-alpha-3b-4k'
    @property
    def name(self):
        return 'STACOA3B4K'
    
class Stablelm216b(LlamaFamily):
    @property
    def engine(self):
        return 'stablelm-2-1_6b'
    @property
    def name(self):
        return 'STALM216B'
    
class Stablelm2Zephyr16b(LlamaFamily):
    @property
    def engine(self):
        return 'stablelm-2-zephyr-1_6b'
    @property
    def name(self):
        return 'STALMZEP16B'
    
class Stablelm7BSftV7Epoch3(LlamaFamily):
    @property
    def engine(self):
        return 'stablelm-7b-sft-v7-epoch-3'
    @property
    def name(self):
        return 'STALMSFT7BE3'
    
class StablelmBaseAlpha3B(LlamaFamily):
    @property
    def engine(self):
        return 'stablelm-base-alpha-3b'
    @property
    def name(self):
        return 'STALMBAA3B'
    
class StablelmBaseAlpha7B(LlamaFamily):
    @property
    def engine(self):
        return 'stablelm-base-alpha-7b'
    @property
    def name(self):
        return 'STALMBAA7B'
    
class StablelmTunedAlpha3B(LlamaFamily):
    @property
    def engine(self):
        return 'stablelm-tuned-alpha-3b'
    @property
    def name(self):
        return 'STALMTUA3B'
    
class StablelmTunedAlpha7B(LlamaFamily):
    @property
    def engine(self):
        return 'stablelm-tuned-alpha-7b'
    @property
    def name(self):
        return 'STALMTUA7B'
    
class StablelmZephyr3B(LlamaFamily):
    @property
    def engine(self):
        return 'stablelm-zephyr-3b'
    @property
    def name(self):
        return 'STALMZEP3B'
    
class TrinityV1(LlamaFamily):
    @property
    def engine(self):
        return 'trinity-v1'
    @property
    def name(self):
        return 'TRIV1'
    
class UNATheBeagle7bV1(LlamaFamily):
    @property
    def engine(self):
        return 'UNA-TheBeagle-7b-v1'
    @property
    def name(self):
        return 'UNAB7BV1'
    
class V1oletMarcoroniGoBruinsMerge7B(LlamaFamily):
    @property
    def engine(self):
        return 'v1olet_marcoroni-go-bruins-merge-7B'
    @property
    def name(self):
        return 'V1MAROGOB7BM'
    
class HarmoniousAnthea(LlamaFamily):
    @property
    def engine(self):
        return 'H4rmoniousAnthea'
    @property
    def name(self):
        return 'H4A'
    @property
    def ulr(self):
        return 'https://huggingface.co/neovalle/H4rmoniousAnthea'
    
class Neuronovo9B(LlamaFamily):
    @property
    def engine(self):
        return 'neuronovo-9B-v0.1'
    @property
    def name(self):
        return 'NNV901'
    
class TenyxChat7B(LlamaFamily):
    @property
    def engine(self):
        return 'TenyxChat-7B-v1'
    @property
    def name(self):
        return 'TC71'
    @property
    def url(self):
        return 'https://huggingface.co/tenyx/TenyxChat-7B-v1'
    
class MedChat35(LlamaFamily):
    @property
    def engine(self):
        return 'MedChat3.5'
    @property
    def name(self):
        return 'MC35'
    @property
    def url(self):
        return 'https://huggingface.co/Imran1/MedChat3.5'
    
class Llava16Vicuna7B(LlamaFamily):
    @property
    def engine(self):
        return 'llava-v1.6-vicuna-7b'
    @property
    def name(self):
        return 'LLV16VIC7'
    
class GraphGPT7B(LlamaFamily):
    @property
    def engine(self):
        return 'GraphGPT-7B-mix-all'
    @property
    def name(self):
        return 'GRGPT7B'
   
    
def get_hf_model_classes():
    return HFModel.subclasses