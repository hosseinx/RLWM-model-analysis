import numpy as np
from collections import defaultdict
import copy
from . import models_base, transformation

#def simulate_model_random(model, response_map, n_trials):
#    pass


def simulate_model_session(model, session):

    out_train = []
    for trial in session.train_set:
        st, ac, rt, bs, congruency = trial  # <-- Fix 1: 5 variables
        
        # Passing congruency to get_action (which goes to get_policy)
        ac_i = model.get_action(st, bs, test=False, congruency=congruency) # <-- Fix 2
        rt_i = session.get_reward(st, ac_i)
        #... (commented lines)
        
        # Adding congruency to the output
        out_train.append((st, ac_i, rt_i, bs, congruency)) # <-- Fix 3
        
        # Passing congruency to learn_sample
        model.learn_sample(st, ac_i, rt_i, bs, congruency=congruency) # <-- Fix 4

    out_test = []
    for trial in session.test_set:
        st, ac, rt, bs, congruency = trial  # <-- Fix 5: 5 variables
        
        # Passing congruency (even during testing)
        ac_i = model.get_action(st, bs, test=True, congruency=congruency) # <-- Fix 6
        rt_i = session.get_reward(st, ac_i)
        
        # Adding congruency to the output
        out_test.append((st, ac_i, rt_i, bs, congruency)) # <-- Fix 7
    
    model_session = copy.deepcopy(session)
    model_session.train_set = out_train
    model_session.test_set = out_test
    return model_session


def simulate_session(model_func, params, session):
 
    #print(f'Params: {params}')
    model = models_base.get_model(model_func, params)
    model.init_model(session.possible_stimuli, session.possible_actions)
    model_session = simulate_model_session(model, session)
    return model_session
        

def average_rts_dict(rts_dict_list):

    rt_avg = defaultdict(list)
    for rt_dict in rts_dict_list:
        for k, rt_series in rt_dict.items():
            rt_avg[k].append(rt_series)
    rt_avg = {k: np.mean(rts_list, axis=0) for k, rts_list in rt_avg.items()}
    return rt_avg
    
    
def simulate_model_curves(model_func, params, session, epochs):

    model_session_list = []
    for i in range(epochs):
        model_session = simulate_session(model_func, params, session)
        model_session_list.append(model_session)

    st_rt_avg_train, st_rt_avg_test = transformation.average_stimuli_probs(model_session_list)
    bs_rt_avg_train, bs_rt_avg_test = transformation.average_block_probs(model_session_list)

    return st_rt_avg_train, st_rt_avg_test, bs_rt_avg_train, bs_rt_avg_test


def simulate_all_sessions(model_func, param_dict, session_list, epochs):

    st_rt_avg_train = []
    st_rt_avg_test = []
    bs_rt_avg_train = []
    bs_rt_avg_test = []

    for session in session_list:

        print(f"Simulating model for session {session.caseid}")
        params = param_dict[session.caseid]
        curves = simulate_model_curves(model_func, params, session, epochs)
        
        st_rt_avg_train.append(curves[0])
        st_rt_avg_test.append(curves[1])
        bs_rt_avg_train.append(curves[2])
        bs_rt_avg_test.append(curves[3])
        
    st_rt_avg_train = average_rts_dict(st_rt_avg_train)
    st_rt_avg_test = average_rts_dict(st_rt_avg_test)
    bs_rt_avg_train = average_rts_dict(bs_rt_avg_train)
    bs_rt_avg_test = average_rts_dict(bs_rt_avg_test)

    return st_rt_avg_train, st_rt_avg_test, bs_rt_avg_train, bs_rt_avg_test