import random
import numpy as np
from abc import ABC, abstractmethod
from .models_base import BaseModel, action_softmax, get_stimulus_group


# Classic RL model class
class CollinsRLClassic(BaseModel):
    '''RL model with additional mechanisms'''  
    def __init__(self, learning_rate, beta):
        self.learning_rate = learning_rate
        self.beta = beta                    # softmax temperature
        self.__stmap = {}                   # Map of stimuli and respective actions
        self.__known_stimuli = set()        # stimuli already processed for init bias
        self.__Q = {}     
        self.__Q_init = 0.0     

    def init_model(self, stimuli, actions):
        actions = set(actions)
        stimuli = set(stimuli)
        self.__stmap = {st: actions for st in stimuli}
        self.__Q_init = 1./len(actions) # alternative: 0 
        for st in stimuli:
            self.__Q[st] = {ac: self.__Q_init for ac in actions}

    def learn_sample(self, stimulus, action, reward, block_size, **kwargs):
        st, ac, rt, bs = stimulus, action, reward, block_size
        self.__known_stimuli.add(st)
        # Delta calculation
        delta = rt - self.__Q[st][ac]
        # Function updates
        self.__Q[st][ac] = self.__Q[st][ac] + self.learning_rate*delta  

    def get_policy(self, stimulus, block_size=None, test=False, **kwargs):
        Q_st = self.__Q[stimulus] 
        pi_rl = action_softmax(Q_st, self.beta)
        return pi_rl


# Generic model class
class CollinsRLBest(BaseModel):
    '''RL model with additional mechanisms'''  
    def __init__(self, learning_rate, beta):
        self.lr3_train = learning_rate
        self.lr6_train = learning_rate
        self.lr3_test = learning_rate
        self.lr6_test = learning_rate
        self.beta = beta                            # softmax temperature
        self.eps = 0.0                              # noise ratio
        self.phi = 0.0                              # forgetting ratio / decay
        self.pers = 0.0                             # perseveration param
        self.init = 0.0                             # init bias param
        self.__stmap = {}                           # Map of stimuli and respective actions
        self.__known_stimuli = set()                # stimuli already processed for init bias
        self.__Q_train = {}      
        self.__Q_test = {}       
        self.__Q_init = 0.0     

    def init_model(self, stimuli, actions):
        actions = set(actions)
        stimuli = set(stimuli)
        self.__stmap = {st: actions for st in stimuli}
        self.__Q_init = 1./len(actions) # alternative: 0 
        for st in stimuli:
            self.__Q_train[st] = {ac: self.__Q_init for ac in actions}
            self.__Q_test[st] = {ac: self.__Q_init for ac in actions}

    def learn_sample(self, stimulus, action, reward, block_size):
        #print(sample, block_size)
        st, ac, rt = stimulus, action, reward
        # Block size dependent parameters
        lr_train = self.lr3_train if block_size == 3 else self.lr6_train
        lr_test = self.lr3_test if block_size == 3 else self.lr6_test
        # Forgetting - fix to case with different Q/W
        for s, actions in self.__stmap.items():
            for a in actions:
                self.__Q_train[s][a] = (1.-self.phi)*self.__Q_train[s][a] + self.phi*self.__Q_init
                self.__Q_test[s][a] = (1.-self.phi)*self.__Q_test[s][a] + self.phi*self.__Q_init
        # Initial bias update  
        if st not in self.__known_stimuli:
            self.__Q_train[st][ac] = self.__Q_init + self.init*(1.0 - self.__Q_init)
            #self.__Q_test[st][ac] = self.__Q_init + self.init*(1.0 - self.__Q_init)
            self.__known_stimuli.add(st)
        # Delta calculation
        delta_train = rt - self.__Q_train[st][ac]
        delta_test = rt - self.__Q_test[st][ac]
        # Perseveration
        if delta_train < 0:
            lr_train = lr_train*(1. - self.pers)
        if delta_test < 0:  
            lr_test = lr_test*(1. - self.pers)
        # Function updates
        self.__Q_train[st][ac] = self.__Q_train[st][ac] + lr_train*delta_train  
        self.__Q_test[st][ac] = self.__Q_test[st][ac] + lr_test*delta_test  

    def get_policy(self, stimulus, block_size=None, test=False):
        Q_st = self.__Q_test[stimulus] if test else self.__Q_train[stimulus]
        pi_rl = action_softmax(Q_st, self.beta)
        # Undirected noise
        n_a = len(pi_rl)
        pi = {ac: ((1. - self.eps)*p + self.eps/n_a) for ac, p in pi_rl.items()}
        return pi


# Collins RLWMi model class
class CollinsRLWM(BaseModel):
    '''RL model with additional mechanisms'''
    def __init__(self, learning_rate, beta, coupled=False):
        self.learning_rate = learning_rate
        self.beta = beta                      # softmax temperature
        self.eps = 0.0                        # noise ratio / lapse rate
        self.phi = 0.0                        # forgetting ratio / decay
        self.pers = 0.0                       # perseveration param
        self.init = 0.0                       # init bias param
        self.eta3_wm = 0.0                    # wm weight in policy calculation (set size 3)
        self.eta6_wm = 0.0                    # wm weight in policy calculation (set size 6)
        self.coupled = coupled                # True for RL + WM interacting model (RLWMi)

        # --- Internal State Variables ---
        self._stmap = {}                      # Map of stimuli and respective actions (protected)
        self._known_stimuli = set()           # stimuli already processed for init bias (protected)
        self._Q = {}                          # Q-values (protected, WAS __Q)
        self._W = {}                          # WM-values (protected, WAS __W)
        self._Q_init = 0.0                    # Initial Q value (protected)
        self._W_init = 0.0                    # Initial W value (protected, although often 0)
        self.stim_map = None                  # To store stimulus -> correct_response map

    def set_stim_map(self, stim_map):
        """Sets the stimulus-to-correct-response map needed for bias/other models."""
        self.stim_map = stim_map

    def init_model(self, stimuli, actions):
        """Initializes Q and W tables."""
        actions = set(actions)
        stimuli = set(stimuli)
        self._stmap = {st: actions for st in stimuli}
        self._Q_init = 1.0 / len(actions) # Uniform initial Q-value
        self._W_init = 0.0                # WM typically starts at 0
        self._Q = {}
        self._W = {}
        self._known_stimuli = set() # Reset known stimuli for init bias
        for st in stimuli:
            self._Q[st] = {ac: self._Q_init for ac in actions} # Use protected _Q
            self._W[st] = {ac: self._W_init for ac in actions} # Use protected _W

    # --- learn_sample METHOD ---
    # Modified to accept congruency and use protected _Q, _W
    def learn_sample(self, stimulus, action, reward, block_size, congruency=None): # Added congruency=None
        st, ac, rt = stimulus, action, reward
        bs = block_size # Alias for consistency if used elsewhere

        # Determine WM weight based on block size
        # eta_bs is used locally in this method if coupled=True
        eta_bs = 0.0 # Default value
        if bs == 3:
             eta_bs = self.eta3_wm
        elif bs == 6:
             eta_bs = self.eta6_wm

        # 1. Decay/Forgetting (applied BEFORE learning from current trial)
        if self.phi > 0:
            for s, actions_set in self._stmap.items(): # Iterate using protected _stmap if needed, but __stmap might be okay if only used internally
                for a in actions_set:
                    # Decay towards initial values
                    self._Q[s][a] = (1.0 - self.phi) * self._Q[s][a] + self.phi * self._Q_init # Use protected _Q
                    self._W[s][a] = (1.0 - self.phi) * self._W[s][a] + self.phi * self._W_init # Use protected _W

        # 2. Initial Bias Update (only for the first encounter with a stimulus)
        if st not in self._known_stimuli:
            # Boost the value of the first chosen action
            self._Q[st][ac] = self._Q_init + self.init * (1.0 - self._Q_init) # Use protected _Q
            # Mark stimulus as seen for init bias purposes
            self._known_stimuli.add(st)
            # Note: The original code only updated Q for init bias. If W should also be biased, add here.

        # 3. Calculate Prediction Error (delta) and Update Q-value
        # Learning Rate (potentially adjusted by perseveration)
        lr = self.learning_rate
        
        # Calculate delta based on whether models are coupled (RLWMi) or not
        if self.coupled:
            # RLWMi: WM expectation contributes to RL's RPE
            expected_value = (1.0 - eta_bs) * self._Q[st][ac] + eta_bs * self._W[st][ac] # Use protected _Q, _W
            delta = rt - expected_value
        else:
            # Independent RL: RPE is based only on Q-value
            delta = rt - self._Q[st][ac] # Use protected _Q

        # Adjust learning rate for negative feedback if perseveration parameter is active
        if delta < 0.0:
            lr = lr * (1.0 - self.pers)

        # Update Q-value using the (potentially adjusted) learning rate and calculated delta
        self._Q[st][ac] = self._Q[st][ac] + lr * delta # Use protected _Q

        # 4. Update W-value (WM learns faster, often just storing the outcome)
        self._W[st][ac] = rt # Use protected _W


    # --- get_policy METHOD ---
    # Modified to accept congruency and use protected _Q, _W
    def get_policy(self, stimulus, block_size=None, test=False, congruency=None): # Added congruency=None
        st = stimulus
        bs = block_size # Alias

        # Get current Q and W values for the stimulus
        Q_st = self._Q[st] # Use protected _Q
        W_st = self._W[st] # Use protected _W

        # Calculate softmax policies for RL and WM separately
        pi_rl = action_softmax(Q_st, self.beta)
        pi_wm = action_softmax(W_st, self.beta)

        # In the test phase, policy is determined solely by RL
        if test:
            # Apply epsilon noise/lapse rate even during testing if needed
            final_pi = pi_rl
            if self.eps > 0:
                 n_a = len(final_pi)
                 uniform = 1.0 / n_a
                 final_pi = {ac: (1.0 - self.eps) * p + self.eps * uniform for ac, p in final_pi.items()}
            return final_pi

        # During learning phase, combine RL and WM policies based on eta
        eta_wm = 0.0 # Default if block_size is None
        if bs == 3:
            eta_wm = self.eta3_wm
        elif bs == 6:
            eta_wm = self.eta6_wm

        # Combine policies: pi = eta * pi_wm + (1-eta) * pi_rl
        pi = {}
        for ac in pi_rl.keys():
            pi[ac] = eta_wm * pi_wm[ac] + (1.0 - eta_wm) * pi_rl[ac]

        # Apply epsilon noise/lapse rate to the final combined policy
        if self.eps > 0:
            n_a = len(pi)
            uniform = 1.0 / n_a
            pi = {ac: (1.0 - self.eps) * p + self.eps * uniform for ac, p in pi.items()}

        return pi

# RLWM model class based on most recent source 
class CollinsRLWMalt1(BaseModel):
    '''RL model with additional mechanisms'''  
    def __init__(self, learning_rate, beta, K, coupled=False):
        self.alpha_rl = learning_rate
        self.alpha_wm = learning_rate
        self.beta = beta                        # softmax temperature
        self.eps = 0.0                          # noise ratio
        self.phi = 0.0                          # forgetting ratio / decay
        self.pers = 0.0                         # perseveration param
        self.eta_wm = 0.0                       # wm weight in policy calculation
        self.K = K                              # WM capacity
        self.coupled = coupled                  # True for RL + WM interacting model
        self.__stmap = {}                       # Map of stimuli and respective actions
        self.__known_stimuli = set()            # stimuli already processed for init bias
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
        eta_wm = self.eta_wm*min([1., self.K/block_size])
        # Forgetting 
        for s, actions in self.__stmap.items():
            for a in actions:
                #self.__Q[s][a] = (1.-self.phi)*self.__Q[s][a] + self.phi*self.__Q_init
                self.__W[s][a] = (1.-self.phi)*self.__W[s][a] + self.phi*self.__W_init
        # Delta calculation
        if self.coupled:
            delta_rl = rt - (eta_wm*self.__W[st][ac] + (1.-eta_wm)*self.__Q[st][ac])
        else:
            delta_rl = rt - self.__Q[st][ac]
        delta_wm = rt - self.__W[st][ac]
        # Perseveration
        alpha_rl = self.alpha_rl
        alpha_wm = self.alpha_wm
        if rt < 1.:
            alpha_rl = alpha_rl*(1. - self.pers)
            alpha_wm = alpha_wm*(1. - self.pers)
        # Function updates
        #print(delta_rl, alpha_rl, round(alpha_rl*delta_rl, 3))
        self.__Q[st][ac] = self.__Q[st][ac] + alpha_rl*delta_rl  
        self.__W[st][ac] = self.__W[st][ac] + alpha_wm*delta_wm 

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
        eta_wm = self.eta_wm*min([1., self.K/block_size])
        for ac in pi_rl.keys():
            pi[ac] = eta_wm*pi_wm[ac] + (1. - eta_wm)*pi_rl[ac]
        return pi


# RLWM model merge of rlwmi and alt1 
class CollinsRLWMalt2(BaseModel):
    '''RL model with additional mechanisms'''  
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
                    #self.__Q[s][a] = (1.-self.phi)*self.__Q[s][a] + self.phi*self.__Q_init
                    self.__W[s][a] = (1.-self.phi)*self.__W[s][a] + self.phi*self.__W_init
        # Initial bias update  
        if st not in self.__known_stimuli:
            self.__Q[st][ac] = self.__Q_init + self.init*(1.0 - self.__Q_init)
            self.__known_stimuli.add(st)
        # Delta calculation
        if self.coupled:
            delta = rt - (eta_wm*self.__W[st][ac] + (1.-eta_wm)*self.__Q[st][ac])
        else:
            delta = rt - self.__Q[st][ac]
        # Perseveration
        lr = self.learning_rate
        if rt < 1.0:
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


# WM only model 
class CollinsWM(BaseModel):
    '''WM only model with additional mechanisms'''  
    def __init__(self, alpha_wm, beta, K):
        self.alpha_wm = alpha_wm
        self.beta = beta                        # softmax temperature
        self.eps = 0.0                          # noise ratio
        self.phi = 0.0                          # forgetting ratio / decay
        self.pers = 0.0                         # perseveration param
        self.eta_wm = 0.0                       # wm weight in policy calculation
        self.K = K                              # WM capacity
        self.__stmap = {}                       # Map of stimuli and respective actions
        self.__W = {}
        self.__W_init = 0.0

    def init_model(self, stimuli, actions):
        actions = set(actions)
        stimuli = set(stimuli)
        self.__stmap = {st: actions for st in stimuli}
        self.__W_init = 1./len(actions) # alternative: 0 
        for st in stimuli:
            self.__W[st] = {ac: self.__W_init for ac in actions}

    def learn_sample(self, stimulus, action, reward, block_size, **kwargs):
        #print(sample, block_size)
        st, ac, rt = stimulus, action, reward
        # Block size dependent parameters
        eta_wm = self.eta_wm*min([1., self.K/block_size])
        # Forgetting 
        for s, actions in self.__stmap.items():
            for a in actions:
                self.__W[s][a] = (1.-self.phi)*self.__W[s][a] + self.phi*self.__W_init
        delta_wm = rt - self.__W[st][ac]
        # Perseveration
        alpha_wm = self.alpha_wm
        if rt < 1.:
            alpha_wm = alpha_wm*(1. - self.pers)
        # Function updates
        self.__W[st][ac] = self.__W[st][ac] + alpha_wm*delta_wm 

    def get_policy(self, stimulus, block_size=None, test=False,  **kwargs):
        W_st = self.__W[stimulus]
        pi_wm = action_softmax(W_st, self.beta)
        # Undirected noise
        n_a = len(pi_wm.keys())
        pi_other = {ac: 1/n_a for ac in pi_wm.keys()}
        pi_wm = {ac: ((1. - self.eps)*p + self.eps/n_a) for ac, p in pi_wm.items()}
        # Final policy - mixed WM and RL
        if test:
            return pi_other
        pi = {}
        eta_wm = self.eta_wm*min([1., self.K/block_size])
        for ac in pi_wm.keys():
            pi[ac] = eta_wm*pi_wm[ac] + (1. - eta_wm)*pi_other[ac]
        return pi

    
# === Alternative Model 2: RLWMi + seperate Learning rates for conflict conditions ===
# =============================================================
class CollinsRLWM_C(CollinsRLWM):
    """
    Inherits from CollinsRLWM and introduces conflict-based learning rate modulation.
    This model uses two separate learning rates for congruent and incongruent trials
    instead of a single unified learning rate.
    """
    def __init__(self, alpha_congruent, alpha_incongruent, beta, coupled=True):
        # Call the parent class constructor, temporarily setting the learning rate 
        # to one of the alphas (it will be overridden later in learn_sample)
        super().__init__(alpha_congruent, beta, coupled)
        
        # Store the two separate learning rates
        self.alpha_congruent = alpha_congruent
        self.alpha_incongruent = alpha_incongruent
        # Other parameters (phi, pers, eps, init, eta) are initialized in parent methods

    # Override learn_sample
    def learn_sample(self, stimulus, action, reward, block_size, congruency=None):
        st, ac, rt = stimulus, action, reward
        bs = block_size

        # Determine working memory contribution (eta) based on block size
        eta_bs = 0.0
        if bs == 3:
            eta_bs = self.eta3_wm
        elif bs == 6:
            eta_bs = self.eta6_wm

        # 1. Decay/Forgetting (inherited from parent but reimplemented here)
        if self.phi > 0:
            for s, actions_set in self._stmap.items():
                for a in actions_set:
                    self._Q[s][a] = (1.0 - self.phi) * self._Q[s][a] + self.phi * self._Q_init
                    self._W[s][a] = (1.0 - self.phi) * self._W[s][a] + self.phi * self._W_init

        # 2. Initial bias update (inherited from parent)
        if st not in self._known_stimuli:
            self._Q[st][ac] = self._Q_init + self.init * (1.0 - self._Q_init)
            self._known_stimuli.add(st)

        # 3. Compute prediction error (delta)
        if self.coupled:
            expected_value = (1.0 - eta_bs) * self._Q[st][ac] + eta_bs * self._W[st][ac]
            delta = rt - expected_value
        else:
            delta = rt - self._Q[st][ac]

        # === Key modification in this model ===
        # 4. Select learning rate based on congruency
        if congruency == 'incongruent':
            lr = self.alpha_incongruent
        else:  # includes 'congruent' and None
            lr = self.alpha_congruent
        # =======================================

        # 5. Apply perseveration (inherited from parent)
        if delta < 0.0:
            lr = lr * (1.0 - self.pers)

        # 6. Update Q-value
        self._Q[st][ac] = self._Q[st][ac] + lr * delta

        # 7. Update W-value (in this model, W simply stores the outcome)
        self._W[st][ac] = rt

    # No need to override get_policy — inherited version is used



# --- Factory function for the new model (rlwm-c) ---
def model_rlwmi_c(alpha_congruent, alpha_incongruent, beta, decay, pers, eps, init, eta3_wm, eta6_wm):
    """
    Factory function for creating an instance of the CollinsRLWM_LRMod model.
    """
    # Instantiate the new CollinsRLWM_LRMod class
    model = CollinsRLWM_C(alpha_congruent, alpha_incongruent, beta, coupled=True)
    
    # Set additional parameters (inherited from the parent or not included in __init__)
    model.phi = decay
    model.pers = pers
    model.eps = eps
    model.init = init
    model.eta3_wm = eta3_wm
    model.eta6_wm = eta6_wm
    
    return model


# Model: Classic RL
def model_classic(learning_rate, beta):
    model = CollinsRLClassic(learning_rate, beta)
    return model


# Model: Best RL with improvements
def model_best(lr3_train, lr6_train, lr3_test, lr6_test, beta, decay, pers, eps, init):
    model = CollinsRLBest(max(lr3_train, lr6_train), beta)
    model.lr3_train = lr3_train
    model.lr6_train = lr6_train
    model.lr3_test = lr3_test
    model.lr6_test = lr6_test
    model.beta = beta
    model.phi = decay
    model.pers = pers
    model.eps = eps
    model.init = init
    return model
    
    
# Model: non-interacting RL+WM  
def model_rlwm(learning_rate, beta, decay, pers, eps, init, eta3_wm, eta6_wm):
    model = CollinsRLWM(learning_rate, beta, coupled=False)
    model.phi = decay
    model.pers = pers
    model.eps = eps
    model.init = init
    model.eta3_wm = eta3_wm
    model.eta6_wm = eta6_wm
    return model
    
    
# Model: interacting RL+WM
def model_rlwmi(learning_rate, beta, decay, pers, eps, init, eta3_wm, eta6_wm):
    model = CollinsRLWM(learning_rate, beta, coupled=True)
    model.phi = decay
    model.pers = pers
    model.eps = eps
    model.init = init
    model.eta3_wm = eta3_wm
    model.eta6_wm = eta6_wm
    return model


# Model: RLWM alternative version
def model_rlwma(alpha_rl, alpha_wm, beta, decay, pers, eps, eta_wm, K):
    model = CollinsRLWMalt1(alpha_rl, beta, K, coupled=False)
    model.alpha_rl = alpha_rl
    model.alpha_wm = alpha_wm
    model.beta = beta
    model.phi = decay
    model.pers = pers
    model.eps = eps
    model.eta_wm = eta_wm
    model.K = K
    return model
     

# Model: merge of RLWM and alt1
def model_rlwmb(learning_rate, beta, decay, pers, eps, init, eta3_wm, eta6_wm):
    model = CollinsRLWMalt2(learning_rate, beta, coupled=True)
    model.phi = decay
    model.pers = pers
    model.eps = eps
    model.init = init
    model.eta3_wm = eta3_wm
    model.eta6_wm = eta6_wm
    return model

# Model: WM only model
def model_wm(alpha_wm, beta, decay, pers, eps, eta_wm, K):
    model = CollinsWM(alpha_wm, beta, K)
    model.alpha_wm = alpha_wm
    model.beta = beta
    model.phi = decay
    model.pers = pers
    model.eps = eps
    model.eta_wm = eta_wm
    model.K = K
    return model
