import os
from multiprocessing import freeze_support
import rlwm.session as session
import rlwm.models_collins as models_collins
import rlwm.models_new as models_new
import rlwm.optsp as optsp
import rlwm.optho as optho
import rlwm.optimization as optimization
import params
from rlwm.models_collins import model_wm 

RUN_BATCH='my_experiments' #'batch_03'
RUN_SUFFIX='' #'c'
RUN_CNR='beta'

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, 'data', RUN_BATCH)
OUTPUT_PATH = os.path.join(BASE_PATH, 'output', RUN_BATCH, RUN_CNR)
MODEL_PATH = os.path.join(BASE_PATH, 'models', RUN_BATCH, RUN_CNR)
#MODEL_PATH = None
OPT_REPS = 20
OPT_EVALMAX = 1000


#CASEIDS = params.CASEIDS_BATCH03

CASEIDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
def main():

    # Load all datasets
    session_list = []
    for id in CASEIDS:
      ds = session.load_session(id, DATA_PATH, RUN_SUFFIX)
      session_list.append(ds)
    print(f'{len(session_list)} cases loaded')


    bounds_classic = {'learning_rate': (0., 1.),
                      'beta':          (0., 50)
                     }

    bounds_best = {'lr3_train':     (0., 1.),
                   'lr6_train':     (0., 1.),
                   'lr3_test':      (0., 1.),
                   'lr6_test':      (0., 1.),
                   'beta':          (0., 50.),
                   'decay':         (0., 1.),
                   'pers':          (0., 1.),
                   'eps':           (0., 1.),
                   'init':          (0., 1.),
                  }

    bounds_rlwm = {'learning_rate': (0., 1.),
                   'beta':          (0., 50.),
                   'decay':         (0., 1.),
                   'pers':          (0., 1.),
                   'eps':           (0., 1.),
                   'init':          (0., 1.),
                   'eta3_wm':       (0., 1.),
                   'eta6_wm':       (0., 1.)
                  }

    bounds_rlwmi = {'learning_rate': (0., 1.),
                    'beta':          (0, 50.),
                    'decay':         (0., 1.),
                    'pers':          (0., 1.),
                    'eps':           (0., 1.),
                    'init':          (0., 1.),
                    'eta3_wm':       (0., 1.),
                    'eta6_wm':       (0., 1.)
                   }

    bounds_rlwma = {'alpha_rl':     (0., 1.),
                    'alpha_wm':     (0. , 1.),
                    'beta':         (0., 50.),
                    'decay':        (0., 1.),
                    'pers':         (0., 1.),
                    'eps':          (0., 1.),
                    'eta_wm':       (0., 1.),
                    'K':            (0., 6.)
                   }

    bounds_rlwmb = {'learning_rate': (0., 1.),
                    'beta':          (0., 50.),
                    'decay':         (0., 1.),
                    'pers':          (0., 1.),
                    'eps':           (0., 1.),
                    'init':          (0., 0.),
                    'eta3_wm':       (0., 1.),
                    'eta6_wm':       (0., 1.)
                   }


    bounds_wm = {'alpha_wm':      (1., 1.),
                 'beta':          (0., 50.),
                 'decay':         (0., 1.),
                 'pers':          (0., 1.),
                 'eps':           (0., 1.),
                 'eta_wm':        (0., 1.),
                 'K':             (6., 6.)
                }

    bounds_new2 = {'learning_rate': (0., 1.),
                    'beta':          (0, 50.),
                    'decay':         (0., 1.),
                    'pers':          (0., 1.),
                    'eps':           (0., 1.),
                    'init':          (0., 1.),
                    'eta3_wm':       (0., 1.),
                    'eta6_wm':       (0., 1.),
                    'gamma_rl':      (0., 1.),
                    'gamma_wm':      (0., 1.)
                   }

    bounds_new3 = {'learning_rate': (0., 1.),
                    'beta':          (0, 50.),
                    'decay':         (0., 1.),
                    'pers':          (0., 1.),
                    'eps':           (0., 1.),
                    'init':          (0., 1.),
                    'eta3_wm':       (0., 1.),
                    'eta6_wm':       (0., 1.),
                    'gamma_rl':      (0., 6.),
                    'gamma_wm':      (0., 6.)
                   }

    bounds_new4 = {'learning_rate': (0., 1.),
                    'beta':          (0, 50.),
                    'decay':         (0., 1.),
                    'pers':          (0., 1.),
                    'eps':           (0., 1.),
                    #'init':          (0., 1.),
                    'eta3_wm':       (0., 1.),
                    'eta6_wm':       (0., 1.),
                    'gamma_pos':      (0., 1.),
                    'gamma_neg':      (0., 1.)
                   }
    bounds_rlwmi_c = {
                     'alpha_congruent':   (0., 1.),  # <--- New parameter 1 (congruent learning rate)
                     'alpha_incongruent': (0., 1.),  # <--- New parameter 2 (incongruent learning rate)
                     'beta':              (0., 50.), 
                     'decay':             (0., 1.),
                     'pers':              (0., 1.),
                     'eps':               (0., 1.),
                     'init':              (0., 1.),
                     'eta3_wm':           (0., 1.),
                     'eta6_wm':           (0., 1.)
                    }

    opt_bounds = bounds_classic# <--- Use new bounds
    opt_modelfunc = models_collins.model_classic # <--- Use new model function
    opt_model_name = 'model_classic'  # <--- New name for output file
    
    opt_solver = 'hyperopt'
    opt_filename = 'param_' + opt_solver + '_' + opt_model_name

    opt_session_list = session_list
    opt_reps = OPT_REPS
    opt_evalmax = OPT_EVALMAX

    print(f'Optimizing cases {[s.caseid for s in opt_session_list]}')


    params, loss = optimization.search_solution_mp(model_func=opt_modelfunc,
                                                 opt_bounds=opt_bounds,
                                                 session_list=opt_session_list,
                                                 n_reps=opt_reps,
                                                 solver=opt_solver,
                                                 models_path=MODEL_PATH,
                                                 model_name=opt_model_name,
                                                 n_jobs=8
                                          )

    optimization.save_param_dict(os.path.join(OUTPUT_PATH, opt_filename), params, loss)


if __name__ == '__main__':
    freeze_support()
    main()