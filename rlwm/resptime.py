import numpy as np
from collections import defaultdict

# Helper function to remove outliers - replace them with NaN
def _clear_list(st_list, st_limits=None):
    st_list_ret = st_list
    if st_limits is not None:
        st_list_ret = [np.nan if x < st_limits[0] or x > st_limits[1] else x for x in st_list]
    return st_list_ret


# Aggregate reward and response time into {stimulus: series} dicts
def aggregate_stimuli_ts(trial_sequence, ts_sequence):

    st_rt_list = defaultdict(list)
    st_ts_list = defaultdict(list)

    for trial, ts in zip(trial_sequence, ts_sequence):
        st, ac, rt, bs = trial
        st_rt_list[st].append(rt)
        st_ts_list[st].append(ts)

    st_rt_dict = {st: np.array(rt_series) for st, rt_series in st_rt_list.items()}
    st_ts_dict = {st: np.array(ts_series) for st, ts_series in st_ts_list.items()}
    return st_rt_dict, st_ts_dict


# Calculates sequence with average response time for each stimulus
def average_stimuli_ts(session_list, st_limits=None):

    st_ts_avg_train = defaultdict(list)
    st_ts_avg_test = defaultdict(list)

    for session in session_list:

        _, st_ts_train = aggregate_stimuli_ts(session.train_set, session.train_ts)
        _, st_ts_test = aggregate_stimuli_ts(session.test_set, session.test_ts)

        for st, ts_series in st_st_train.items():
            st_ts_avg_train[st].append(_clear_list(ts_series, st_limits))

        for st, ts_series in st_ts_test.items():
            st_ts_avg_test[st].append(_clear_list(ts_series, st_limits))


    st_ts_avg_train = {st: np.nanmean(ts_list, axis=0) for st, ts_list in st_ts_avg_train.items()}
    st_ts_avg_test = {st: np.nanmean(ts_list, axis=0) for st, ts_list in st_ts_avg_test.items()}

    return st_ts_avg_train, st_ts_avg_test


# Calculates sequence with average resonse time for each block size
def average_block_ts(session_list, st_limits=None):

    bs_ts_avg_train = defaultdict(list)
    bs_ts_avg_test = defaultdict(list)

    for session in session_list:

        bs_ts_train = defaultdict(list)
        bs_ts_test = defaultdict(list)

        _, st_ts_train = aggregate_stimuli_ts(session.train_set, session.train_ts)
        _, st_ts_test = aggregate_stimuli_ts(session.test_set, session.test_ts)

        for st, ts_series in st_ts_train.items():
            bs = session.get_blocksize(st)
            bs_ts_train[bs].append(_clear_list(ts_series, st_limits))

        for st, ts_series in st_ts_test.items():
            bs = session.get_blocksize(st)
            bs_ts_test[bs].append(_clear_list(ts_series, st_limits))

        for bs, ts_list in bs_ts_train.items():
            bs_ts_avg_train[bs].append(np.nanmean(ts_list, axis=0))

        for bs, ts_list in bs_ts_test.items():
            bs_ts_avg_test[bs].append(np.nanmean(ts_list, axis=0))

    bs_ts_avg_train = {bs: np.nanmean(ts_list, axis=0) for bs, ts_list in bs_ts_avg_train.items()}
    bs_ts_avg_test = {bs: np.nanmean(ts_list, axis=0) for bs, ts_list in bs_ts_avg_test.items()}

    return bs_ts_avg_train, bs_ts_avg_test

