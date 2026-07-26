import os
from collections import Counter
import numpy as np
import pandas as pd

trainfile_root = '-RLWM-Parte 1_'
testfile_root = '-RLWM-Parte 2_'


class DataSession():
    '''Base class containing experiment data'''
    def __init__(self, caseid):
        self.caseid = caseid
        self.possible_stimuli = []
        self.possible_actions = []
        self.response_map = {}
        self.train_set = []
        self.test_set = []
        self.st_maxlen_train = 0
        self.st_maxlen_test = 0

    def get_reward(self, stimulus, action):
        ac_correct = self.response_map.loc[stimulus]['correct']
        return 1. if action == ac_correct else 0.

    def get_blocksize(self, stimulus):
        return self.response_map.loc[stimulus]['Block']

    @classmethod
    def from_sequence(cls, caseid, train_set, test_set, response_dict):
        session = cls(caseid)
        session.possible_stimuli = list(set([trial[0] for trial in train_set]))
        session.possible_actions = list(set([trial[1] for trial in train_set]))
        session.train_set = train_set
        session.test_set = test_set

        # Create map stimulus to response and block size
        blocksize_dict = {trial[0]: trial[3] for trial in train_set}

        if not isinstance(response_dict, dict):
            response_dict = {trial[0]: trial[1] for trial in train_set if trial[2] > 0}

        block_col = [blocksize_dict[st] for st in session.possible_stimuli]
        respo_col = [response_dict[st] for st in session.possible_stimuli]

        session.response_map = pd.DataFrame(
            zip(session.possible_stimuli, respo_col, block_col),
            columns=['Stimulus_Pair', 'correct', 'Block']
        )

        # 🔥 Main fix
        session.response_map = session.response_map.set_index('Stimulus_Pair')

        return session

    # Find this method in the rlwm/session.py file and replace it with the code below
    @classmethod
    def from_df(cls, caseid, df_train, df_test):
        session = cls(caseid)

        # ========== Line added to resolve the KeyError issue ==========
        # Drop any row with a missing (NaN) value in the stimulus column
        df_train.dropna(subset=['Stimulus_Pair'], inplace=True)
        df_test.dropna(subset=['Stimulus_Pair'], inplace=True)
        # ==============================================================

        session.possible_stimuli = df_train['Stimulus_Pair'].unique().tolist()
        session.possible_actions = df_train['response'].unique().tolist()
        
        map_df = df_train[['Stimulus_Pair', 'correct', 'Block']].drop_duplicates(subset=['Stimulus_Pair'])
        session.response_map = map_df.set_index('Stimulus_Pair')

        # session.train_set = list(zip(df_train['Stimulus_Pair'], df_train['response'], df_train['correct'], df_train['Block']))
        # session.test_set = list(zip(df_test['Stimulus_Pair'], df_test['response'], df_test['correct'], df_test['Block']))
        if 'congruency' in df_train.columns:
            session.train_set = list(zip(df_train['Stimulus_Pair'], df_train['response'], df_train['correct'], df_train['Block'], df_train['congruency']))
        else:
            print("Warning: 'congruency' column not found in training data. Bias model might not work correctly.")
            # Fallback if 'congruency' is missing
            session.train_set = list(zip(df_train['Stimulus_Pair'], df_train['response'], df_train['correct'], df_train['Block'], [None]*len(df_train))) # Add None as placeholder

        if 'congruency' in df_test.columns:
            session.test_set = list(zip(df_test['Stimulus_Pair'], df_test['response'], df_test['correct'], df_test['Block'], df_test['congruency']))
        else:
            # Assuming test set might also need congruency...
            session.test_set = list(zip(df_test['Stimulus_Pair'], df_test['response'], df_test['correct'], df_test['Block'], [None]*len(df_test))) # Add None as placeholder
            
        return session

class tsDataSession(DataSession):
    '''Extended class adding response time to data series'''
    def __init__(self, caseid):
        super().__init__(caseid)
        self.train_ts = []
        self.test_ts = []

    @classmethod
    def from_df(cls, caseid, df_train, df_test):
        session = super(tsDataSession, cls).from_df(caseid, df_train, df_test)
        session.train_ts = df_train['response_time'].tolist()
        session.test_ts = df_test['response_time'].tolist()
        return session


def load_dataset(caseid, data_path, suffix=''):
        global trainfile_root
        global testfile_root
        train_filename = str(caseid) + trainfile_root + suffix + '.csv'
        test_filename = str(caseid) + testfile_root + suffix + '.csv'
        df_train = pd.read_csv(os.path.join(data_path, train_filename), sep=',')
        df_test = pd.read_csv(os.path.join(data_path, test_filename), sep=',')
        return df_train, df_test


def load_session(caseid, data_path, suffix=''):
    df_train, df_test = load_dataset(caseid, data_path, suffix)
    ds = tsDataSession.from_df(caseid, df_train, df_test)
    return ds