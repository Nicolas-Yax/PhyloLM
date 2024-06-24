from lanlab.core.tools.dict_tools import SafeDict
from lanlab.core.tools.configurable import Configurable
from lanlab.core.module.module import Module
from lanlab.core.structure.batch import Batch
from lanlab.core.structure.sequence import Sequence
from lanlab.core.structure.segment import Segment

import logging
import pickle
import os
import time
import numpy as np
from queue import Empty
#from pathos.helpers import mp
import multiprocess as mp
#import pathos.multiprocessing as mp
Process = mp.Process
Queue = mp.Queue
Manager = mp.Manager

BATCH_SIZE = 1 #number of processes

def set_number_workers(n):
    global BATCH_SIZE
    BATCH_SIZE = n

class TimeoutError(Exception):
    """ Raised when a process doesn't return in time"""
    pass

class CallProcess(Process):
    """ Runs a function in an asyncronous process and returns the result in a queue """
    def __init__(self,model,seq,mode,queue):
        super().__init__()
        self.daemon = True
        self.model = model
        self.seq = seq
        self.mode = mode
        self.queue = queue

    def run(self):
        logging.debug('call process running')
        if self.mode == 'complete':
            r = self.model.complete(self.seq)
        if self.mode == 'read':
            r = self.model.read(self.seq)
        logging.debug('call process putting in queue')
        self.queue.put(r)

NB_TIMEOUT_MAX = 100 #maximum number of timeouts before stopping the program
NB_TIMEOUT = 0 #number of timeouts since the beginning of the program

class BasicWorker(Process):
    """ Worker that collects computation from the input_queue and creates a CallProcess to run the function. If the function doesn't return in time, the process is terminated and the computation is rerun."""
    def __init__(self,i,inp_queue,out_queue,timeout):
        super().__init__()
        self.inp_queue = inp_queue
        self.out_queue = out_queue
        self.i = i
        self.stage = 0

        self.alive = True
        self.timeout = timeout

    def run(self):
        global NB_TIMEOUT,NB_TIMEOUT_MAX
        while self.alive:
            try:
                logging.debug('start loop basic worker a')
                #Get a computation to run from the queue
                #time.sleep(5)
                #continue
                model,seq,mode,index = self.inp_queue.get(timeout=0.01)
                logging.debug('start loop basic worker b')
                while True:
                    logging.debug('basic worker process inner loop')
                    #Spawns a Call Process to run the computation
                    call_queue = Queue()
                    logging.debug('basic worker process making callprocess')
                    call_process = CallProcess(model,seq,mode,call_queue)
                    logging.debug('basic worker process starting callprocess')
                    call_process.start()
                    try:
                        #Get the result from the Call Process
                        logging.debug('basic worker process getting result from queue')
                        new_seq = call_queue.get(timeout=self.timeout)
                        logging.debug('basic worker process closing queue')
                        call_queue.close()
                        logging.debug('basic worker process joining queue')
                        call_process.join()
                        
                        #If the Call Process returns in time, break the loop
                        break
                    except Empty:
                        #If the Call Process doesn't return in time, terminate it and rerun the computation
                        print('TIMEOUT')
                        call_queue.close()
                        call_process.terminate()
                        call_process.join()
                        NB_TIMEOUT += 1
                        if NB_TIMEOUT >= NB_TIMEOUT_MAX:
                            assert False #Too many timeouts
                #Put the result in the output queue
                logging.debug('basic worker process putting in out queue')
                self.out_queue.put((new_seq,index))
            except Empty:
                logging.debug('start loop basic worker - empty 1')
                time.sleep(1)
        
    def function(self,model,seq):
        raise NotImplementedError

    def terminate(self):
        self.alive = False

class BasicCompleteWorker(BasicWorker):
    """ Worker that runs the complete function of the model"""
    def function(self,model,seq):
        return model.complete(seq)

class BasicReadWorker(BasicCompleteWorker):
    """ Worker that runs the read function of the model"""
    def function(self,model,seq):
        return model.read(seq)

class ModelConfig(SafeDict):
    def __init__(self):
        super().__init__()
        self.add_key('temperature',0.7)
        self.add_key('max_tokens',16)
        self.add_key('top_p',1)

class Model(Module,Configurable):
    """ Language Model interface class """
    def __init__(self):
        Module.__init__(self)
        Configurable.__init__(self)
        self._mode = 'complete'

    def config_class(self):
        return ModelConfig()
    
    def complete(struct):
        raise NotImplementedError
    
    def read(struct):
        raise NotImplementedError
    
    def mode(self,mode):
        assert mode in ['complete','read']
        self._mode = mode
        return self
    
    def configure(self,**kwargs):
        for k,v in kwargs.items():
            self.config[k] = v
        return self
    
    def run(self,struct):
        #Loads the model if needed and if hasn't been done already
        if isinstance(struct,Batch):
            return self.run_batch(struct)
        elif isinstance(struct,Sequence):
            return self.run_single(struct)
        elif isinstance(struct,Segment):
            return self.run_single(Sequence([struct]))
        elif isinstance(struct,str):
            return self.run_single(Sequence([Segment({'text':struct,'origin':'user'})]))
        else:
            raise ValueError("Input is not a Structure: " + str(type(struct)))
        
    def run_single(self,struct):
        if self._mode == 'read':
            return self.read(struct)
        elif self._mode == 'complete':
            return self.complete(struct)
        else:
            raise ValueError("Unknown mode: " + self.mode)
        
    def run_batch(self,batch):
        #Get worker class
        if self._mode == 'read':
            worker_class = BasicReadWorker
        elif self._mode == 'complete':
            worker_class= BasicCompleteWorker
        else:
            raise ValueError("Unknown mode: " + self.mode)
        #Create workers
        logging.debug('creating workers')
        inp_queue = Queue()
        out_queue = Queue()
        logging.debug('creating workers b'+str(BATCH_SIZE))
        workers = [BasicWorker(i,inp_queue,out_queue,self.timeout) for i in range(BATCH_SIZE)]
        #Fill input queue
        logging.debug('fill input queue')
        for index in np.ndindex(batch.array.shape):
            inp_queue.put((self,batch.array[index],self._mode,index))
        #Start workers
        logging.debug('starting workers')
        for worker in workers:
            worker.start()
        #Get results
        logging.debug('get results')
        time.sleep(3)
        logging.debug('get results b')
        results = Batch(batch.shape)
        #got_data = np.zeros(batch.array.shape,dtype=bool)
        got_data = {}
        for index in np.ndindex(batch.array.shape):
            got_data[index] = time.perf_counter()
        logging.debug('looping get results')
        while len(got_data) > 0 :#not got_data.all():
            #print(len(got_data))
            try:
                logging.debug('loop iter get results')
                seq,index = out_queue.get(timeout=1)
                logging.debug('loop iter get results 2')
                results[index] = seq
                del got_data[index]
            except Empty:
                pass
        #Close queues
        logging.debug('close queues')
        inp_queue.close()
        out_queue.close()
        #Terminate workers
        logging.debug('terminqte workers')
        for worker in workers:
            worker.alive = False
            worker.kill()
        #Join workers
        for worker in workers:
            worker.join()
        #Close workers
        for worker in workers:
            worker.close()

        return results
    @property
    def timeout(self):
        return 60 #rerun the computation after 60 sec if the server doesn't answer

    @property
    def surname(self):
        """ Name of the model (the short version if possible)"""
        return NotImplementedError
    
    def save(self,study):
        with open(os.path.join(study.path,'model.p'),'wb') as f:
            pickle.dump(self,f)
