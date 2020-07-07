import cornac
import os
import numpy as np
from cornac.data import reader
from cornac.eval_methods import RatioSplit
from cornac.data import Reader
from resource import *
import time
from cornac.models.mf.recom_mf import MF


class Model:
    before_count = 100
    after_count = 100

    @classmethod
    def before_memory_copy(cls):
        print(getrusage(RUSAGE_SELF))
        with open("before_system_memory_copy%s.txt" % cls.before_count, "w") as write:
            for line in memory_read():
                write.write(line)
        cls.before_count = cls.before_count - 10

    @classmethod
    def after_memory_copy(cls):
        print(getrusage(RUSAGE_SELF))
        with open("after_system_memory_copy%s.txt" % cls.after_count, "w") as write:
            for line in memory_read():
                write.write(line)
        cls.after_count = cls.after_count - 10


def memory_read():  # method to provide system memory info before and after model
    # use context manager to prevent memory leak
    pid = os.getpid()
    with open("/proc/%s/status" % pid, "r") as proc:
        proc_contents = proc.read()
        # for line in proc:
        #     print(line, end=" ")
        return proc_contents


def run_experiment(ratio_split):
    # naive time recording
    start_time = time.time()
    # pre-training system memory snapshot
    Model.before_memory_copy()
    mf = cornac.models.MF(
        k=10,
        max_iter=25,
        learning_rate=0.01,
        lambda_reg=0.02,
        use_bias=True,
        early_stop=True,
        verbose=True,
    )
    # system/user time recordings
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

    # runit?
    cornac.Experiment(
        eval_method=ratio_split,
        models=[mf],
        metrics=[mae, mse, rmse, auc, f1, rec_10, pre_10, ndcg, ncrr, mAp, mrr]
    ).run()
    # post train/test memoy snapshot
    Model.after_memory_copy()
    print("--- %s seconds ---" % (time.time() - start_time))
