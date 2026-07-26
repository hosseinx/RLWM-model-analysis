import os
import hyperopt
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from . import optimization
from .models_base import get_model
import pickle

OPT_HYPEROPT_EVALSTEP = 100

# Generator of each cost function
def score_dict_gen(model_func, session):
    def opt_func(params):
        model = get_model(model_func, params)
        model.init_model(session.possible_stimuli, session.possible_actions)
        loss = optimization.cost_half_llh(model, session)
        #print(f'Params: {params}, Cost: {cost_total}')
        return {'loss': loss, 'status': STATUS_OK}
    return opt_func


# Load params from model file
def load_params(model_file):
    params = {}
    loss = None
    try:
        trials = pickle.load(open(model_file, 'rb'))
        loss = trials.best_trial['result']['loss']
        #num_trials = len(trials) 
        #params = trials.best_trial['misc']['vals']
        params = trials.argmin
    except:
        pass
    return params, loss


# Hyperparameter-optimization: only run if needed
def search_solution(model_func, opt_bounds, session, n_reps, model_file=None):

    global OPT_HYPEROPT_EVALSTEP

    # Try to load past trials file
    trials = Trials()
    if model_file:
        try:        
            trials_p = pickle.load(open(model_file, 'rb'))
            trials = trials_p
        except:
            pass
    eval_step = OPT_HYPEROPT_EVALSTEP
    eval_start = len(trials.trials)
    eval_end = max(n_reps, eval_start)
    # Define score func and search space
    score_func = score_dict_gen(model_func, session)
    search_space = {}
    for pname, x in opt_bounds.items():
        search_space[pname] = hp.uniform(pname, x[0], x[1]) 
    # Main optimization loop
    print(f"Optimizing case {session.caseid}")
    tcount = eval_start
    while tcount <= eval_end:
        #print(f'Running cycle to {tcount}')
        best_param = fmin(score_func, 
                          search_space, 
                          algo=tpe.suggest, 
                          max_evals=min(tcount+eval_step, eval_end),
                          show_progressbar=False, 
                          trials = trials
                          )
        if model_file:
            pickle.dump(trials, open(model_file, 'wb'))
        best_func = score_func(best_param)
        tcount += eval_step
    return best_param, best_func['loss']

