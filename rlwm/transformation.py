import numpy as np
from collections import defaultdict


# Aggragate reward seguence into {stimulus: series} dict
def aggregate_stimuli_trial(trial_sequence):

    st_rt_list = defaultdict(list)

    for trial in trial_sequence:
        st, ac, rt, bs = trial
        st_rt_list[st].append(rt)

    st_rt_dict = {st: np.array(rt_series) for st, rt_series in st_rt_list.items()}
    return st_rt_dict

# Same as aggragate_stimuli_trial, but for both datasets of session
def aggregate_stimuli(session):

    st_rt_train = defaultdict(list)
    st_rt_test = defaultdict(list)

    for trial in session.train_set:
        st, ac, rt, bs = trial
        st_rt_train[st].append(rt)

    for trial in session.test_set:
        st, ac, rt, bs = trial
        st_rt_test[st].append(rt)

    st_rt_train = {st: np.array(rt_series) for st, rt_series in st_rt_train.items()}
    st_rt_test = {st: np.array(rt_series) for st, rt_series in st_rt_test.items()}

    return st_rt_train, st_rt_test
    
    
# Calcula a sequencia com o reward médio para cada estímulo 
def average_stimuli_probs(session_list):

    st_rt_avg_train = defaultdict(list)
    st_rt_avg_test = defaultdict(list)

    for session in session_list:

        st_rt_train, st_rt_test = aggregate_stimuli(session)

        for st, rt_series in st_rt_train.items():
            st_rt_avg_train[st].append(rt_series)

        for st, rt_series in st_rt_test.items():
            st_rt_avg_test[st].append(rt_series)

    st_rt_avg_train = {st: np.mean(rts_list, axis=0) for st, rts_list in st_rt_avg_train.items()}
    st_rt_avg_test = {st: np.mean(rts_list, axis=0) for st, rts_list in st_rt_avg_test.items()}

    return st_rt_avg_train, st_rt_avg_test


# Calcula a sequencia com o reward médio para cada tamanho de bloco 
def average_block_probs(session_list):

    bs_rt_avg_train = defaultdict(list)
    bs_rt_avg_test = defaultdict(list)

    for session in session_list:

        bs_rt_train = defaultdict(list)
        bs_rt_test = defaultdict(list)

        #print("Aggragating session " + str(session.caseid))
        st_rt_train, st_rt_test = aggregate_stimuli(session)

        for st, rt_series in st_rt_train.items():
            bs = session.get_blocksize(st)
            bs_rt_train[bs].append(rt_series)

        for st, rt_series in st_rt_test.items():
            bs = session.get_blocksize(st)
            bs_rt_test[bs].append(rt_series)
 
        for bs, rts_list in bs_rt_train.items():
            bs_rt_avg_train[bs].append(np.mean(rts_list, axis=0))

        for bs, rts_list in bs_rt_test.items():
            bs_rt_avg_test[bs].append(np.mean(rts_list, axis=0))

    bs_rt_avg_train = {bs: np.mean(rts_list, axis=0) for bs, rts_list in bs_rt_avg_train.items()}
    bs_rt_avg_test = {bs: np.mean(rts_list, axis=0) for bs, rts_list in bs_rt_avg_test.items()}

    return bs_rt_avg_train, bs_rt_avg_test


# Calcula a sequencia com o reward médio para cada tamanho de bloco e caseid 
def average_caseid_block_probs(session_list):

    id_bs_rt_train = {}
    id_bs_rt_test = {}

    for session in session_list:

        bs_rt_train = defaultdict(list)
        bs_rt_test = defaultdict(list)

        #print("Aggragating session " + str(session.caseid))
        st_rt_train, st_rt_test = aggregate_stimuli(session)

        for st, rt_series in st_rt_train.items():
            bs = session.get_blocksize(st)
            bs_rt_train[bs].append(rt_series)

        for st, rt_series in st_rt_test.items():
            bs = session.get_blocksize(st)
            bs_rt_test[bs].append(rt_series)
 
        id_bs_rt_train[session.caseid] = {}
        id_bs_rt_test[session.caseid] = {}

        for bs, rts_list in bs_rt_train.items():
            id_bs_rt_train[session.caseid][bs] = np.mean(rts_list, axis=0)

        for bs, rts_list in bs_rt_test.items():
            id_bs_rt_test[session.caseid][bs] = np.mean(rts_list, axis=0)

    return id_bs_rt_train, id_bs_rt_test
    

# Calcula a taxa de reward assintotica para cada tamanho de bloco 
def range_block_probs(session_list, start_pos=None, end_pos=None):

    bs_as_pairs_train = []
    bs_as_pairs_test = []
    
    for session in session_list:

        bs_as_train = defaultdict(list)
        bs_as_test = defaultdict(list)

        st_rt_train, st_rt_test = aggregate_stimuli(session)

        for st, series in st_rt_train.items():
            bs = session.get_blocksize(st)
            as_prob = np.mean(series[start_pos:end_pos])
            bs_as_train[bs].append(as_prob)
        
        for st, series in st_rt_test.items():
            bs = session.get_blocksize(st)
            as_prob = np.mean(series)
            bs_as_test[bs].append(as_prob)

        bs_as_pairs_train.append({bs: np.mean(samples) for bs, samples in bs_as_train.items()})
        bs_as_pairs_test.append({bs: np.mean(samples) for bs, samples in bs_as_test.items()})

    return bs_as_pairs_train, bs_as_pairs_test


# Count total perseverance errors
def count_perseverance(session_list):

    pers_count = {}
    for session in session_list:
        pers_count[session.caseid] = 0
        past_errs = defaultdict(set)
        for trial in session.train_set:
            st, ac, rt, bs = trial
            if rt == 0:
                if ac in past_errs[st]:
                    pers_count[session.caseid] +=1
                else:
                    past_errs[st].add(ac)              
    return pers_count


# Count perseverance errors by stimulus 
def count_perseverance_bs(session_list):

    pers_count = {}
    for session in session_list:
        pers_count[session.caseid] = defaultdict(lambda: 0)
        past_errs = defaultdict(set)
        for trial in session.train_set:
            st, ac, rt, bs = trial
            if rt == 0:
                if ac in past_errs[st]:
                    pers_count[session.caseid][bs] +=1
                else:
                    past_errs[st].add(ac)              
    return pers_count


# Mark if current trial is a perseverance error
def count_trial_perseverance(session):

    pers_train = []
    pers_test = []
    past_errs = defaultdict(set)

    for trial in session.train_set:
        st, ac, rt, bs = trial
        pers_error = 0
        if rt == 0:
            if ac in past_errs[st]:
                pers_error = 1
            else:
                past_errs[st].add(ac)
        pers_train.append(pers_error)        
    for trial in session.test_set:
        st, ac, rt, bs = trial
        pers_error = 0
        if rt == 0:
            if ac in past_errs[st]:
                pers_error = 1
        pers_test.append(pers_error)

    return pers_train, pers_test


# Count last time since a correct response
def count_trial_delay(session):

    train_delay = []
    test_delay = []
    last_resp = {}
    i = 0

    for trial in session.train_set:
        st, ac, rt, bs = trial
        delay = 0 # default value , alternative: -1
        if st in last_resp:
            delay = i - last_resp[st]
        train_delay.append(delay)
        if rt > 0:
            last_resp[st] = i
        i += 1
    for trial in session.test_set:
        st, ac, rt, bs = trial
        delay = -1
        if st in last_resp:
            delay = i - last_resp[st]
        test_delay.append(delay)
        #if rt > 0:
        #    last_resp[st] = i
        i += 1
    return train_delay, test_delay


# Count number of correct responses before current trial
def count_trial_rpred(session):

    train_rpred = []
    test_rpred = []
    count_resp = defaultdict(lambda: 0)

    for trial in session.train_set:
        st, ac, rt, bs = trial
        train_rpred.append(count_resp[st])
        if rt > 0:
            count_resp[st] += 1
    for trial in session.test_set:
        st, ac ,rt, bs = trial
        test_rpred.append(count_resp[st])
        #if rt > 0:
        #    count_resp[st] += 1
    return train_rpred, test_rpred

# Estimate the block number sequence of each trial
def count_trial_block(session):

    train_bnumber = []
    test_bnumber = []

    current_prefix = ""
    current_block = 0
    for trial in session.train_set:
        st, ac, rt, bs = trial
        prefix = st[:-1]
        if prefix != current_prefix:
            current_block += 1
            current_prefix = prefix
        train_bnumber.append(current_block)

    current_prefix = "" 
    current_block = 0
    for trial in session.test_set:
        st, ac, rt, bs = trial
        prefix = st[:-1]
        if prefix != current_prefix:
            current_block += 1
            current_prefix = prefix
        test_bnumber.append(current_block)
    return train_bnumber, test_bnumber

