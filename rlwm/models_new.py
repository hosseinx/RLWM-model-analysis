import random
import numpy as np
from abc import ABC, abstractmethod
from .models_base import BaseModel, action_softmax, get_stimulus_group 


# Collins RLWMi + deduction version 1
class RLWMnew1(BaseModel):
    '''RLWM model with deduction of non random answers'''  
    def __init__(self, learning_rate, beta, coupled=False):
        self.learning_rate = learning_rate
        self.beta = beta                        # softmax temperature
        self.eps = 0.0                          # noise ratio
        self.phi = 0.0                          # forgetting ratio / decay
        self.pers = 0.0                         # perseveration param
        self.init = 0.0                         # init bias param
        self.eta3_wm = 0.0                      # wm weight in policy calculation
        self.eta6_wm = 0.0                      # wm weight in policy calculation
        self.coupled = coupled                  # True for RL + WM interacting model
        self.__stmap = {}                       # Map of stimuli and respective actions
        self.__known_stimuli = set()            # stimuli already processed for init bias
        self.__known_answers = {}
        self.__Q = {}     
        self.__W = {}
        self.__Q_init = 0.0     
        self.__W_init = 0.0

    def init_model(self, stimuli, actions):
        actions = set(actions)
        stimuli = set(stimuli)
        self.__stmap = {st: actions for st in stimuli}
        self.__Q_init = 1./len(actions) # alternative: 0 
        self.__W_init = 1./len(actions) # alternative: 0 
        for st in stimuli:
            self.__Q[st] = {ac: self.__Q_init for ac in actions}
            self.__W[st] = {ac: self.__W_init for ac in actions}

    def learn_sample(self, stimulus, action, reward, block_size):
        #print(sample, block_size)
        st, ac, rt = stimulus, action, reward
        # Block size dependent parameters
        eta_wm = self.eta3_wm if block_size == 3 else self.eta6_wm
        # Forgetting - fix to case with different Q/W
        if self.phi > 0:
            for s, actions in self.__stmap.items():
                for a in actions:
                    self.__Q[s][a] = (1.-self.phi)*self.__Q[s][a] + self.phi*self.__Q_init
                    self.__W[s][a] = (1.-self.phi)*self.__W[s][a] + self.phi*self.__W_init
        # Initial bias update  
        if st not in self.__known_stimuli:
            self.__Q[st][ac] = self.__Q_init + self.init*(1.0 - self.__Q_init)
            self.__known_stimuli.add(st)
        # Rewarding deduction/induction 
        if st not in self.__known_answers and rt > 0:
            st_group = get_stimulus_group(st)
            st_coef = 1. - (3./block_size)
            for s, actions in self.__stmap.items():
                if s != st and get_stimulus_group(s) == st_group:
                    # Change here to affect only one learning mechanism
                    self.__Q[s][ac] = self.__Q[s][ac]*st_coef
                    self.__W[s][ac] = self.__W[s][ac]*st_coef
            self.__known_answers[st] = ac        
        # Delta calculation
        if self.coupled:
            delta = rt - (eta_wm*self.__W[st][ac] + (1.-eta_wm)*self.__Q[st][ac])
        else:
            delta = rt - self.__Q[st][ac]
        # Perseveration
        lr = self.learning_rate
        if delta < 0.:
            lr = lr*(1. - self.pers)
        # Function updates
        self.__Q[st][ac] = self.__Q[st][ac] + lr*delta  
        self.__W[st][ac] = rt  

    def get_policy(self, stimulus, block_size=None, test=False):
        Q_st = self.__Q[stimulus]
        W_st = self.__W[stimulus]
        pi_rl = action_softmax(Q_st, self.beta)
        pi_wm = action_softmax(W_st, self.beta)
        # Undirected noise
        n_a = len(pi_rl.keys())
        pi_rl = {ac: ((1. - self.eps)*p + self.eps/n_a) for ac, p in pi_rl.items()}
        pi_wm = {ac: ((1. - self.eps)*p + self.eps/n_a) for ac, p in pi_wm.items()}
        # Final policy - mixed WM and RL
        if test:
            return pi_rl
        pi = {}
        eta_wm = self.eta3_wm if block_size == 3 else self.eta6_wm
        for ac in pi_rl.keys():
            pi[ac] = eta_wm*pi_wm[ac] + (1. - eta_wm)*pi_rl[ac]
        return pi


# Collins RLWMi + deduction version 2
class RLWMnew2(BaseModel):
    '''RLWM model with deduction of non random answers'''  
    def __init__(self, learning_rate, beta, coupled=False):
        self.learning_rate = learning_rate
        self.beta = beta                        # softmax temperature
        self.eps = 0.0                          # noise ratio
        self.phi = 0.0                          # forgetting ratio / decay
        self.pers = 0.0                         # perseveration param
        self.init = 0.0                         # init bias param
        self.eta3_wm = 0.0                      # wm weight in policy calculation
        self.eta6_wm = 0.0                      # wm weight in policy calculation
        self.gamma_rl = 1.0
        self.gamma_wm = 1.0
        self.coupled = coupled                  # True for RL + WM interacting model
        self.__stmap = {}                       # Map of stimuli and respective actions
        self.__known_stimuli = set()            # stimuli already processed for init bias
        self.__known_answers = {}
        self.__Q = {}     
        self.__W = {}
        self.__Q_init = 0.0     
        self.__W_init = 0.0

    def init_model(self, stimuli, actions):
        actions = set(actions)
        stimuli = set(stimuli)
        self.__stmap = {st: actions for st in stimuli}
        self.__Q_init = 1./len(actions) # alternative: 0 
        self.__W_init = 1./len(actions) # alternative: 0 
        for st in stimuli:
            self.__Q[st] = {ac: self.__Q_init for ac in actions}
            self.__W[st] = {ac: self.__W_init for ac in actions}

    def learn_sample(self, stimulus, action, reward, block_size):
        #print(sample, block_size)
        st, ac, rt = stimulus, action, reward
        # Block size dependent parameters
        eta_wm = self.eta3_wm if block_size == 3 else self.eta6_wm
        # Forgetting - fix to case with different Q/W
        if self.phi > 0:
            for s, actions in self.__stmap.items():
                for a in actions:
                    self.__Q[s][a] = (1.-self.phi)*self.__Q[s][a] + self.phi*self.__Q_init
                    self.__W[s][a] = (1.-self.phi)*self.__W[s][a] + self.phi*self.__W_init
        # Initial bias update  
        if st not in self.__known_stimuli:
            self.__Q[st][ac] = self.__Q_init + self.init*(1.0 - self.__Q_init)
            self.__known_stimuli.add(st)
        # Rewarding deduction/induction 
        if st not in self.__known_answers and rt > 0:
            st_group = get_stimulus_group(st)
            for s, actions in self.__stmap.items():
                if s != st and get_stimulus_group(s) == st_group:
                    if s not in self.__known_answers or self.__known_answers[s] != ac:
                        # Change here to affect only one learning mechanism
                        self.__Q[s][ac] = self.__Q[s][ac]*self.gamma_rl
                        self.__W[s][ac] = self.__W[s][ac]*self.gamma_wm
            self.__known_answers[st] = ac        
        # Delta calculation
        if self.coupled:
            delta = rt - (eta_wm*self.__W[st][ac] + (1.-eta_wm)*self.__Q[st][ac])
        else:
            delta = rt - self.__Q[st][ac]
        # Perseveration
        lr = self.learning_rate
        if delta < 0.:
            lr = lr*(1. - self.pers)
        # Function updates
        self.__Q[st][ac] = self.__Q[st][ac] + lr*delta  
        self.__W[st][ac] = rt  

    def get_policy(self, stimulus, block_size=None, test=False):
        Q_st = self.__Q[stimulus]
        W_st = self.__W[stimulus]
        pi_rl = action_softmax(Q_st, self.beta)
        pi_wm = action_softmax(W_st, self.beta)
        # Undirected noise
        n_a = len(pi_rl.keys())
        pi_rl = {ac: ((1. - self.eps)*p + self.eps/n_a) for ac, p in pi_rl.items()}
        pi_wm = {ac: ((1. - self.eps)*p + self.eps/n_a) for ac, p in pi_wm.items()}
        # Final policy - mixed WM and RL
        if test:
            return pi_rl
        pi = {}
        eta_wm = self.eta3_wm if block_size == 3 else self.eta6_wm
        for ac in pi_rl.keys():
            pi[ac] = eta_wm*pi_wm[ac] + (1. - eta_wm)*pi_rl[ac]
        return pi


# Collins RLWMi + deduction version 3
class RLWMnew3(BaseModel):
    '''RLWM model with deduction of non random answers'''
    def __init__(self, learning_rate, beta, coupled=False):
        self.learning_rate = learning_rate
        self.beta = beta                        # softmax temperature
        self.eps = 0.0                          # noise ratio
        self.phi = 0.0                          # forgetting ratio / decay
        self.pers = 0.0                         # perseveration param
        self.init = 0.0                         # init bias param
        self.eta3_wm = 0.0                      # wm weight in policy calculation
        self.eta6_wm = 0.0                      # wm weight in policy calculation
        self.gamma_rl = 1.0
        self.gamma_wm = 1.0
        self.coupled = coupled                  # True for RL + WM interacting model
        self.__stmap = {}                       # Map of stimuli and respective actions
        self.__known_stimuli = set()            # stimuli already processed for init bias
        self.__known_answers = {}
        self.__Q = {}
        self.__W = {}
        self.__Q_init = 0.0
        self.__W_init = 0.0

    def init_model(self, stimuli, actions):
        actions = set(actions)
        stimuli = set(stimuli)
        self.__stmap = {st: actions for st in stimuli}
        self.__Q_init = 1./len(actions) # alternative: 0
        self.__W_init = 1./len(actions) # alternative: 0
        for st in stimuli:
            self.__Q[st] = {ac: self.__Q_init for ac in actions}
            self.__W[st] = {ac: self.__W_init for ac in actions}

    def learn_sample(self, stimulus, action, reward, block_size):
        #print(sample, block_size)
        st, ac, rt = stimulus, action, reward
        # Block size dependent parameters
        eta_wm = self.eta3_wm if block_size == 3 else self.eta6_wm
        # Forgetting - fix to case with different Q/W
        if self.phi > 0:
            for s, actions in self.__stmap.items():
                for a in actions:
                    self.__Q[s][a] = (1.-self.phi)*self.__Q[s][a] + self.phi*self.__Q_init
                    self.__W[s][a] = (1.-self.phi)*self.__W[s][a] + self.phi*self.__W_init
        # Initial bias update
        if st not in self.__known_stimuli:
            self.__Q[st][ac] = self.__Q_init + self.init*(1.0 - self.__Q_init)
            self.__known_stimuli.add(st)
        # Rewarding deduction/induction
        if st not in self.__known_answers and rt > 0:
            st_group = get_stimulus_group(st)
            for s, actions in self.__stmap.items():
                if s != st and get_stimulus_group(s) == st_group:
                    if s not in self.__known_answers or self.__known_answers[s] != ac:
                        # Change here to affect only one learning mechanism
                        self.__Q[s][ac] = self.__Q[s][ac]*(1. - self.gamma_rl/block_size)
                        self.__W[s][ac] = self.__W[s][ac]*(1. - self.gamma_wm/block_size)
            self.__known_answers[st] = ac
        # Delta calculation
        if self.coupled:
            delta = rt - (eta_wm*self.__W[st][ac] + (1.-eta_wm)*self.__Q[st][ac])
        else:
            delta = rt - self.__Q[st][ac]
        # Perseveration
        lr = self.learning_rate
        if delta < 0.:
            lr = lr*(1. - self.pers)
        # Function updates
        self.__Q[st][ac] = self.__Q[st][ac] + lr*delta
        self.__W[st][ac] = rt

    def get_policy(self, stimulus, block_size=None, test=False):
        Q_st = self.__Q[stimulus]
        W_st = self.__W[stimulus]
        pi_rl = action_softmax(Q_st, self.beta)
        pi_wm = action_softmax(W_st, self.beta)
        # Undirected noise
        n_a = len(pi_rl.keys())
        pi_rl = {ac: ((1. - self.eps)*p + self.eps/n_a) for ac, p in pi_rl.items()}
        pi_wm = {ac: ((1. - self.eps)*p + self.eps/n_a) for ac, p in pi_wm.items()}
        # Final policy - mixed WM and RL
        if test:
            return pi_rl
        pi = {}
        eta_wm = self.eta3_wm if block_size == 3 else self.eta6_wm
        for ac in pi_rl.keys():
            pi[ac] = eta_wm*pi_wm[ac] + (1. - eta_wm)*pi_rl[ac]
        return pi


# Collins RLWMi + deduction version 4
class RLWMnew4(BaseModel):
    '''RLWM model with deduction of non random answers'''  
    def __init__(self, learning_rate, beta, coupled=False):
        self.learning_rate = learning_rate
        self.beta = beta                        # softmax temperature
        self.eps = 0.0                          # noise ratio
        self.phi = 0.0                          # forgetting ratio / decay
        self.pers = 0.0                         # perseveration param
        #self.init = 0.0                         # init bias param
        self.eta3_wm = 0.0                      # wm weight in policy calculation
        self.eta6_wm = 0.0                      # wm weight in policy calculation
        self.gamma_pos = 0.0
        self.gamma_neg = 0.0
        self.coupled = coupled                  # True for RL + WM interacting model
        self.__stmap = {}                       # Map of stimuli and respective actions
        self.__known_stimuli = set()            # stimuli already processed for init bias
        self.__Q = {}     
        self.__W = {}
        self.__Q_init = 0.0     
        self.__W_init = 0.0
        self.__Q_init_func = {}
        self.__W_init_func = {}

    def init_model(self, stimuli, actions):
        actions = set(actions)
        stimuli = set(stimuli)
        self.__stmap = {st: actions for st in stimuli}
        self.__Q_init = 1./len(actions) # alternative: 0 
        self.__W_init = 1./len(actions) # alternative: 0 
        for st in stimuli:
            self.__Q[st] = {ac: self.__Q_init for ac in actions}
            self.__W[st] = {ac: self.__W_init for ac in actions}
            self.__Q_init_func[st] = {ac: self.__Q_init for ac in actions}
            self.__W_init_func[st] = {ac: self.__W_init for ac in actions}


    def learn_sample(self, stimulus, action, reward, block_size):
        #print(sample, block_size)
        st, ac, rt = stimulus, action, reward
        # Block size dependent parameters
        eta_wm = self.eta3_wm if block_size == 3 else self.eta6_wm
        # Forgetting - fix to case with different Q/W
        if self.phi > 0:
            for s, actions in self.__stmap.items():
                for a in actions:
                    self.__Q[s][a] = (1.-self.phi)*self.__Q[s][a] + self.phi*self.__Q_init
                    self.__W[s][a] = (1.-self.phi)*self.__W[s][a] + self.phi*self.__W_init
        # Initial bias update  
        #if st not in self.__known_stimuli:
        #    self.__Q[st][ac] = self.__Q_init + self.init*(1.0 - self.__Q_init)
        #    self.__known_stimuli.add(st)
        # Rewarding deduction/induction 
        st_group = get_stimulus_group(st)
        for s, actions in self.__stmap.items():
            if s != st and get_stimulus_group(s) == st_group:
                if rt > 0.:
                    self.__Q_init_func[s][ac] = self.__Q_init_func[s][ac]*(1. - self.gamma_pos*3./block_size)
                    self.__W_init_func[s][ac] = self.__W_init_func[s][ac]*(1. - self.gamma_pos*3./block_size)
                else:
                    for a in [a for a in actions if a != ac]:
                        self.__Q_init_func[s][a] = self.__Q_init_func[s][a]*(1. + self.gamma_neg*3./block_size)
                        self.__W_init_func[s][a] = self.__W_init_func[s][a]*(1. + self.gamma_neg*3./block_size)
                if s not in self.__known_stimuli:
                    self.__Q[s] = self.__Q_init_func[s].copy()
                    self.__W[s] = self.__W_init_func[s].copy()
        self.__known_stimuli.add(st)
        # Delta calculation
        if self.coupled:
            delta = rt - (eta_wm*self.__W[st][ac] + (1.-eta_wm)*self.__Q[st][ac])
        else:
            delta = rt - self.__Q[st][ac]
        # Perseveration
        lr = self.learning_rate
        if delta < 0.:
            lr = lr*(1. - self.pers)
        # Function updates
        self.__Q[st][ac] = self.__Q[st][ac] + lr*delta  
        self.__W[st][ac] = rt  

    def get_policy(self, stimulus, block_size=None, test=False):
        Q_st = self.__Q[stimulus]
        W_st = self.__W[stimulus]
        pi_rl = action_softmax(Q_st, self.beta)
        pi_wm = action_softmax(W_st, self.beta)
        # Undirected noise
        n_a = len(pi_rl.keys())
        pi_rl = {ac: ((1. - self.eps)*p + self.eps/n_a) for ac, p in pi_rl.items()}
        pi_wm = {ac: ((1. - self.eps)*p + self.eps/n_a) for ac, p in pi_wm.items()}
        # Final policy - mixed WM and RL
        if test:
            return pi_rl
        pi = {}
        eta_wm = self.eta3_wm if block_size == 3 else self.eta6_wm
        for ac in pi_rl.keys():
            pi[ac] = eta_wm*pi_wm[ac] + (1. - eta_wm)*pi_rl[ac]
        return pi



# Model: interacting RL+WM version 1
def model_new1(learning_rate, beta, decay, pers, eps, init, eta3_wm, eta6_wm):
    model = RLWMnew1(learning_rate, beta, coupled=True)
    model.phi = decay
    model.pers = pers
    model.eps = eps
    model.init = init
    model.eta3_wm = eta3_wm
    model.eta6_wm = eta6_wm
    return model

# Model: interacting RL+WM version 2
def model_new2(learning_rate, beta, decay, pers, eps, init, eta3_wm, eta6_wm, gamma_rl, gamma_wm):
    model = RLWMnew2(learning_rate, beta, coupled=True)
    model.phi = decay
    model.pers = pers
    model.eps = eps
    model.init = init
    model.eta3_wm = eta3_wm
    model.eta6_wm = eta6_wm
    model.gamma_rl = gamma_rl
    model.gamma_wm = gamma_wm
    return model

## Model: interacting RL+WM version 3
def model_new3(learning_rate, beta, decay, pers, eps, init, eta3_wm, eta6_wm, gamma_rl, gamma_wm):
    model = RLWMnew3(learning_rate, beta, coupled=True)
    model.phi = decay
    model.pers = pers
    model.eps = eps
    model.init = init
    model.eta3_wm = eta3_wm
    model.eta6_wm = eta6_wm
    model.gamma_rl = gamma_rl
    model.gamma_wm = gamma_wm
    return model

## Model: interacting RL+WM version 4
def model_new4(learning_rate, beta, decay, pers, eps, eta3_wm, eta6_wm, gamma_pos, gamma_neg):
    model = RLWMnew3(learning_rate, beta, coupled=True)
    model.phi = decay
    model.pers = pers
    model.eps = eps
    #model.init = init
    model.eta3_wm = eta3_wm
    model.eta6_wm = eta6_wm
    model.gamma_pos = gamma_pos
    model.gamma_neg = gamma_neg
    return model

