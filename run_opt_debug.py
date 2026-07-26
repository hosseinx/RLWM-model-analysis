import os
from multiprocessing import freeze_support
import rlwm.session as session
import rlwm.models as models
import rlwm.optsp as optsp
import rlwm.optho as optho
import rlwm.optimization as optimization

BASE_PATH = '/srv/black/data/rlwm'
DATA_PATH = os.path.join(BASE_PATH, 'dados')
OUTPUT_PATH = os.path.join(BASE_PATH, 'output')
#MODEL_PATH = os.path.join(BASE_PATH, 'models/geral')
MODEL_PATH = None
OPT_REPS = 20
OPT_EVALMAX = 1000

CASEIDS = [1, 2, 5, 6, 7, 8, 9, 12, 13, 14, 17, 25, 26, 27, 29, 37, 49, 54, 57, 59, 62, 64, 66, 72, 76, 77, 79, 84, 91, 92, 94, 97, 102, 105, 110, 118, 127, 132, 153, 155, 159, 161, 164, 166, 172, 174, 175, 178, 179, 180, 181, 182, 184, 185, 187, 189, 195, 197, 200, 201, 202, 203, 204, 206, 207, 208, 209, 210, 211, 212, 214, 215, 216, 217, 218, 220, 221, 223, 224, 226, 227]

#CASEIDS = CASEIDS[:5]
CASEIDS = [2]



def main():

    # Load all datasets
    session_list = []
    for id in CASEIDS:
      ds = session.load_session(id, DATA_PATH)
      session_list.append(ds)
    print(f'{len(session_list)} cases loaded')


    bounds_classic = {'learning_rate': (0., 1.),
                      'beta':          (50., 50)
                     }

    bounds_best = {'lr3_train':     (0., 1.),
                   'lr6_train':     (0., 1.),
                   'lr3_test':      (0., 1.),
                   'lr6_test':      (0., 1.),
                   'beta':          (50., 50.),
                   'decay':         (0., 1.),
                   'pers':          (0., 1.),
                   'eps':           (0., 1.),
                   'init':          (0., 1.),
                  }

    bounds_rlwm = {'learning_rate': (0., 1.),
                   'beta':          (50., 50.),
                   'decay':         (0., 1.),
                   'pers':          (0., 1.),
                   'eps':           (0., 1.),
                   'init':          (0., 1.),
                   'eta3_wm':       (0., 1.),
                   'eta6_wm':       (0., 1.)
                  }

    bounds_rlwmi = {'learning_rate': (0., 1.),
                    'beta':          (50, 50.),
                    'decay':         (0., 1.),
                    'pers':          (0., 1.),
                    'eps':           (0., 1.),
                    'init':          (0., 1.),
                    'eta3_wm':       (0., 1.),
                    'eta6_wm':       (0., 1.)
                   }

    bounds_rlwma = {'alpha_rl':     (0., 1.),
                    'alpha_wm':     (1., 1.),
                    'beta':         (0., 500.),
                    'decay':        (0., 1.),
                    'pers':         (0., 0.),
                    'eps':          (0., 1.),
                    'eta_wm':       (0., 1.),
                    'K':            (6., 6.)
                   }

    bounds_rlwmb = {'learning_rate': (0., 1.),
                    'beta':          (50., 50.),
                    'decay':         (0., 1.),
                    'pers':          (0., 1.),
                    'eps':           (0., 1.),
                    'init':          (0., 0.),
                    'eta3_wm':       (0., 1.),
                    'eta6_wm':       (0., 1.)
                   }


    bounds_wm = {'alpha_wm':      (1., 1.),
                 'beta':          (50., 50.),
                 'decay':         (0., 1.),
                 'pers':          (0., 1.),
                 'eps':           (0., 1.),
                 'eta_wm':        (0., 1.),
                 'K':             (6., 6.)
                }


    opt_bounds = bounds_rlwmi
    opt_modelfunc = models.model_rlwmi
    opt_model_name = 'model_rlwmi-debug'
    
    opt_solver = 'scipy'
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
