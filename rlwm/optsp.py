import os
import sys
import random
import numpy as np
import pickle
import inspect
import scipy.optimize as optimize
from . import session, optimization
from .models_base import get_model

OPT_SCIPY_EVALMAX = 1000    # Max evaluations of a function in a repetition
OPT_SCIPY_REPMAX = 100      # Max extra/failed repetitions of optimization
OPT_TOL = 1e-10

# Generator of each cost function
def generate_optfunc(model_func, session):
    def opt_func(params):
        params = params.tolist()
        model = get_model(model_func, params)
        model.init_model(session.possible_stimuli, session.possible_actions)
        cost_total = optimization.cost_half_llh(model, session)
        #print(f'Params: {params}, Cost: {cost_total}')
        return cost_total
    return opt_func

# Load params from model file
def load_params(model_file):
    params = {}
    loss = None
    try:
        trials = pickle.load(open(model_file, 'rb'))
        loss = trials['min_val']
        #num_trials = trials['n_valid']
        params = trials['params']
    except:
        pass
    return params, loss


# Search solution given constraints, repeating n_reps times
def search_solution(model_func, opt_bounds, session, n_reps, model_file=None):

    global OPT_SCIPY_EVALMAX
    global OPT_SCIPY_REPMAX
    global OPT_TOL
 
    # Try to load file with past runs
    trials = {}
    trials['params'] = {}
    trials['min_val'] = sys.float_info.max
    trials['n_valid'] = 0
    if model_file:
        try:
            trials_p = pickle.load(open(model_file, 'rb'))
            trials = trials_p
        except:
            pass
    # Define function and bounds for minimize()
    opt_func = generate_optfunc(model_func, session)
    param_names = inspect.getfullargspec(model_func).args
    bounds_list = [opt_bounds[pname] for pname in param_names]
    # Optmization loop
    print(f"Optimizing case {session.caseid}")
    rep_count = 0
    while trials['n_valid'] < n_reps and rep_count < n_reps + OPT_SCIPY_REPMAX:
        x_init = [random.uniform(x0, x1) for x0, x1 in bounds_list]
        opt_res = optimize.minimize(opt_func, x_init,
                                    #args=(model_func, session),
                                    method='Powell', #'L-BFGS-B', 'TNC', 'SLSQP', *'Powell', 'trust-constr'
                                    bounds=bounds_list, 
                                    #tol=OPT_TOL,
                                    #options={'disp': True},
                                    #options={'maxiter': OPT_SCIPY_EVALMAX}
                                    )
        #print(trials['n_valid'], opt_res.success)
        if opt_res.success:
            if opt_res.fun < trials['min_val']:
                trials['min_val'] = opt_res.fun
                trials['params'] = {pname: v for pname, v in zip(param_names, opt_res.x.tolist())}
            trials['n_valid'] += 1
            if model_file:
                pickle.dump(trials, open(model_file, "wb"))
        rep_count += 1
    return trials['params'], trials['min_val']      

