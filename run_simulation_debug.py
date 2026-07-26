import os
from multiprocessing import freeze_support
import rlwm.session as session
import rlwm.models_collins as models
import rlwm.optimization as optimization
import rlwm.simulation as simulation
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import numpy as np

# مقادیر خودتان را اینجا وارد کنید (باید با run_opt.py یکی باشد)
RUN_BATCH = 'my_experiments'
RUN_CNR = 'beta'
RUN_SUFFIX = ''

# مسیرهای ساده و جدید را تعریف کنید
BASE_PATH = 'C:/Users/hosse/Documents/rlwm-main'
DATA_PATH = os.path.join(BASE_PATH, 'data', RUN_BATCH)
MODEL_PATH = os.path.join(BASE_PATH, 'models', RUN_BATCH, RUN_CNR)
OUTPUT_PATH = os.path.join(BASE_PATH, 'output', RUN_BATCH, RUN_CNR)

# مطمئن شوید که پوشه خروجی وجود دارد
os.makedirs(OUTPUT_PATH, exist_ok=True)

CASEIDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]

def main():
    # Load all datasets
    session_list = []
    for id in CASEIDS:
        ds = session.load_session(id, DATA_PATH, RUN_SUFFIX)
        session_list.append(ds)
    print(f'{len(session_list)} cases loaded')

    opt_modelfunc = models.model_rlwmi_lrmod
    opt_model_name = 'model_rlwmi_lrmod'
    opt_solver = 'scipy'
    sim_epochs = 1
    p_sp = {}
    f_sp = {}

    print(f'Loading params for cases {[s.caseid for s in session_list]}')
    for s in session_list:
        p, f = optimization.get_model_params(opt_modelfunc,
                                             s,
                                             solver=opt_solver,
                                             model_name=opt_model_name,
                                             model_path=MODEL_PATH)
        p_sp[s.caseid] = p
        f_sp[s.caseid] = f

    param_dict = p_sp
    session_epoch = defaultdict(list)
    
    for s in tqdm(session_list):
        caseid = s.caseid
        base_seed = 12345
        for i in range(sim_epochs):
            np.random.seed(base_seed+i)
            session_epoch[caseid].append(simulation.simulate_session(opt_modelfunc, param_dict[caseid], s))

    # بخش ذخیره خروجی با نام متغیر اصلاح شده
    all_sim_trials = []
    for caseid, sim_sessions in session_epoch.items():
        # ==> اصلاحیه: نام متغیر از 'session' به 'sim_data' تغییر کرد <==
        for i, sim_data in enumerate(sim_sessions):
            for trial_num, trial in enumerate(sim_data.train_set):
                all_sim_trials.append({
                    'caseid': caseid,
                    'epoch': i + 1,
                    'trial': trial_num + 1,
                    'stimulus': trial[0],
                    'action': trial[1],
                    'reward': trial[2],
                    'block_size': trial[3],
                    'congruency': trial[4]
                })

    df_sim = pd.DataFrame(all_sim_trials)
    output_filename = os.path.join(OUTPUT_PATH, 'simulated_data_rlwmi_lrmod.csv')
    df_sim.to_csv(output_filename, index=False)
    
    print(f"Simulated data saved to {output_filename}")

if __name__ == '__main__':
    freeze_support()
    main()