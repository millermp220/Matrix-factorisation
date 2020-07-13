from concurrent.futures._base import Error

import cornac
import os
import numpy as np
from cornac.data import reader
from cornac.eval_methods import RatioSplit
from cornac import experiment
from cornac.data import Reader
from resource import *
from string import ascii_letters, punctuation, whitespace
import time
import csv
import re
import fcntl
from cornac.models.mf.recom_mf import MF


class Model:
    before_count = 100
    after_count = 100

    # @classmethod
    # def before_memory_copy(cls):
    #     print(getrusage(RUSAGE_SELF))
    #     with open("before_system_memory_copy%s.txt" % cls.before_count, "w") as write:
    #         for line in memory_read():
    #             write.write(line)
    #     cls.before_count = cls.before_count - 10
    #
    # @classmethod
    # def after_memory_copy(cls):
    #     print(getrusage(RUSAGE_SELF))
    #     with open("after_system_memory_copy%s.txt" % cls.after_count, "w") as write:
    #         for line in memory_read():
    #             write.write(line)
    #     cls.after_count = cls.after_count - 10


#
#
# def memory_read():  # method to provide system memory info before and after model
#     # use context manager to prevent memory leak
#     pid = os.getpid()
#     with open("/proc/%s/status" % pid, "r") as proc:
#         proc_contents = proc.read()
#         # for line in proc:
#         #     print(line, end=" ")
#         return proc_contents

# obtain list of virtual memory stats


# return vm stats from proc file
def get_memory():
    pid = os.getpid()
    with open("/proc/%s/status" % pid, "r") as proc:
        lines = [line.strip() for line in proc]
        return lines[10:19]


# use regex to strip amounts from get_memory()
def mem_capture():
    lines = get_memory()
    # pattern_text = r'(?P<mem_type>\w+):\s+(?P<amount>\d+\s+(?P<units>\w+)'
    # pattern = re.compile(r'(?P<mem_type>\w+):\s+(?P<amount>\d+\s+(?P<units>\w+)')
    values = []
    for line in lines:
        match = re.search(r'([0-9]+)', line)
        if match:
            values.append(int(match.group()))
    return values


# @param parsed contents of virtual memory statistics
# calculate difference in memory consumption (after - before)
# return vm stats for printing
def memory_calculation(mem_contents):
    after_shot = mem_capture()
    vmpeak = after_shot[0] - mem_contents[0]
    vmsize = after_shot[1] - mem_contents[1]
    vmlck = after_shot[2] - mem_contents[2]
    vmhwm = after_shot[3] - mem_contents[3]
    vmrss = after_shot[4] - mem_contents[4]
    vmdata = after_shot[5] - mem_contents[5]
    vmstk = after_shot[6] - mem_contents[6]
    vmexe = after_shot[7] - mem_contents[7]
    vmlib = after_shot[8] - mem_contents[8]
    return [vmpeak, vmsize, vmlck, vmhwm, vmrss,
            vmdata, vmstk, vmexe, vmlib]


# append memory stats to file
def mem_write(mem_calc, name, model_param):
    # infinite loop to ensure each thread call results in data written to file
    while True:
        try:
            mem_calc.insert(0, name)
            mem_calc.insert(0, model_param)
            with open("Memory_Recordings.csv", "a") as rec:
                # lock file
                fcntl.flock(rec, fcntl.LOCK_EX | fcntl.LOCK_NB)
                writer = csv.writer(rec)
                writer.writerow([r for r in mem_calc])
                # unlock file
                fcntl.flock(rec, fcntl.LOCK_UN)
                break
        except Error:
            # if file locked wait 0.05 seconds and try again
            time.sleep(0.05)


# obtain system and user time
# return list of attributes for analysis
def time_capture():
    list_time = getrusage(RUSAGE_SELF)
    ru_utime = list_time[0]
    ru_stime = list_time[1]
    return [ru_utime, ru_stime]


# @params system/user time before training/testing
# return after-before time for accurate measurements of training/testing time
def time_calculation(before_time):
    time_taken = time_capture()
    time_taken[0] = time_taken[0] - before_time[0]
    time_taken[1] = time_taken[1] - before_time[1]
    return time_taken


# @param result of time calculation
# append times taken to file
def time_write(time_calc, name, model_param):
    time_calc.insert(0, name)
    time_calc.insert(0, model_param)
    # infinite loop to ensure each thread call results in data written to file
    while True:
        try:
            with open("Time_Recordings.csv", "a") as rec:
                # lock file
                fcntl.flock(rec, fcntl.LOCK_EX | fcntl.LOCK_NB)
                writer = csv.writer(rec)
                writer.writerow([r for r in time_calc])
                # unlock file
                fcntl.flock(rec, fcntl.LOCK_UN)
                break
        except Error:
            time.sleep(0.05)  # wait before retrying


# @param each samples ratio split (0.8/0.2) train/test split for analysis
def run_experiment(ratio_split, name, model_param):
    # naive time recording
    start_time = time.time()
    mf = cornac.models.MF(
        k=10,
        max_iter=25,
        learning_rate=0.01,
        lambda_reg=0.02,
        use_bias=True,
        early_stop=True,
        verbose=True,
    )
    # metrics (Accuracy/rating)
    mae = cornac.metrics.MAE()  # mean absolute error
    mse = cornac.metrics.MSE()  # mean squared error
    rmse = cornac.metrics.RMSE()  # root mean squared error
    # metrics (Accuracy/Ranking)
    auc = cornac.metrics.AUC()  # area under curve
    f1 = cornac.metrics.FMeasure(k=-1)  # f measure (utility of item per user)
    rec_10 = cornac.metrics.Recall(k=10)  # recall
    pre_10 = cornac.metrics.Precision(k=10)  # precision
    ndcg = cornac.metrics.NDCG()  # normalised dcg = (discount cumulative gain/ideal discount cumulative gain)
    ncrr = cornac.metrics.NCRR()  # normalised cumulative reciprocal rank
    mAp = cornac.metrics.MAP()  # mean average precision
    mrr = cornac.metrics.MRR()  # mean reciprocal rank
    the_metrics = {"mae": mae, "mse": mse, "rmse": rmse, "auc": auc,
                   "fmeasure": f1, "rec10": rec_10, "pre10": pre_10,
                   "ndcg": ndcg, "ncrr": ncrr, "map": mAp, "mmr": mrr}

    # create experiment
    exp = cornac.Experiment(
        eval_method=ratio_split,
        models=[mf],
        metrics=[mae, mse, rmse, auc, f1, rec_10, pre_10, ndcg, ncrr, mAp, mrr],
        save_dir="trained.csv"
    )
    # run experiment
    exp.run()
    # get results from experiment
    # list_to_string(exp)

    # timestamp (rough)
    print("--- %s seconds ---" % (time.time() - start_time))
    # convert results into list of strings
    lister = []
    lister = str(exp.result)
    # extract only the values of the metrics from the list
    metrics = lister[277:len(lister)]
    return metrics


# use regex to extract float values of accuracy/rank metrics
# return list of float values
def metric_capture(metrics):
    results = " "
    values = []
    for line in metrics:
        results += line
    metrics = results.split(" | ")
    for line in metrics:
        match = re.search(r'(\d*\.\d+|\d+\d+\d)+', str(line))
        if match:
            values.append(float(match.group()))
    return values


# @param list of float value metrics
# append to csv file
def metric_write(metrics, name, model_param):
    metrics.insert(0, name)
    metrics.insert(0, model_param)
    # infinite loop to ensure each thread call results in data written to file
    while True:
        try:
            with open("Metric_Recordings.csv", "a") as rec:
                # lock file
                fcntl.flock(rec, fcntl.LOCK_EX | fcntl.LOCK_NB)
                writer = csv.writer(rec)
                writer.writerow([r for r in metrics])
                # unlock file
                fcntl.flock(rec, fcntl.LOCK_UN)
                break
        except Error:
            time.sleep(0.05)  # wait before retrying
