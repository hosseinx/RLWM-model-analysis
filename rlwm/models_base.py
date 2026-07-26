import random
import numpy as np
from abc import ABC, abstractmethod
 

# Softmax helper function
def action_softmax(action_func, beta):
    scale = np.sum([np.exp(beta*f) for a, f in action_func.items()])
    sft_func = {a: np.exp(beta*f)/scale for a, f in action_func.items()} 
    return sft_func
        
# Helper function to extract stimulus prefix
def get_stimulus_group(stimulus):
    return stimulus[:-1]

# Returns a model expanding parameters into arguments
def get_model(model_func, params):
    #print(model_func.__name__, type(params), params)
    model = None
    if isinstance(params, dict):
        model = model_func(**params)
    if isinstance(params, list):
        model = model_func(*params)
    return model

# Base model
class BaseModel(ABC):
    
    @abstractmethod
    def init_model(self, stimuli, actions):
        pass

    @abstractmethod
    def learn_sample(self, stimulus, action, reward, block_size):
        pass
    
    @abstractmethod
    def get_policy(self, stimulus, block_size=None, test=False):
        pass

    def get_action(self, stimulus, block_size=None, test=False, **kwargs):
        pi = self.get_policy(stimulus, block_size, test, **kwargs)
        actions, probs = list(pi.keys()), list(pi.values())
        action = np.random.choice(actions, p=probs)
        return action

