import cornac
from cornac.eval_methods import RatioSplit
import Model
import time
from cornac.data import Reader
from sklearn.utils import resample

from sklearn.metrics import accuracy_score


class BootStraps:
    ratio_splitter: RatioSplit = None
    size = None
    sample = None

    def __init__(self, size, sample, ratio_splitter):
        self.size = size
        self.sample = sample
        self.ratio_splitter = ratio_splitter


def main():
    # naive time recording
    start_time = time.time()
    # pre-training system memory snapshot
    Model.before_memory_copy()
    # import movielens dataset
    dset = cornac.datasets.movielens.load_feedback(variant="100k", reader=Reader())

    # use bootstrap sampling to obtain 90/80../10% random samples,90% of 100, 80% of 90000 etc
    # use ratio splitter 0.8 train/test split
    full = BootStraps(100000, dset, get_ratio_split(dset))
    ninety = BootStraps(90000, sample_size(dset, 90000), get_ratio_split(sample_size(dset, 90000)))
    eighty = BootStraps(80000, sample_size(ninety.sample, 80000), get_ratio_split(sample_size(ninety.sample, 80000)))
    seventy = BootStraps(70000, sample_size(eighty.sample, 70000), get_ratio_split(sample_size(eighty.sample, 70000)))
    sixty = BootStraps(60000, sample_size(seventy.sample, 60000), get_ratio_split(sample_size(seventy.sample, 60000)))
    fifty = BootStraps(50000, sample_size(sixty.sample, 50000), get_ratio_split(sample_size(sixty.sample, 50000)))
    forty = BootStraps(40000, sample_size(fifty.sample, 40000), get_ratio_split(sample_size(fifty.sample, 40000)))
    thirty = BootStraps(30000, sample_size(forty.sample, 30000), get_ratio_split(sample_size(forty.sample, 30000)))
    twenty = BootStraps(20000, sample_size(thirty.sample, 20000), get_ratio_split(sample_size(thirty.sample, 20000)))
    ten = BootStraps(10000, sample_size(twenty.sample, 10000), get_ratio_split(sample_size(twenty.sample, 10000)))
    # list of all samples
    samples = [full, ninety, eighty, seventy, sixty,
               fifty, forty, thirty, twenty,
               ten]
    # run experiment on each for results
    for x in samples:
        train_test(x)
    # post train/test memoy snapshot
    Model.after_memory_copy()
    print("--- %s seconds ---" % (time.time() - start_time))


# @param randomly sampled subset of movielens data, perform 0.8/0.2 train/test split
# return shuffled and split data-sets
def get_ratio_split(re_sample):
    ratio_splitter = cornac.eval_methods.ratio_split.RatioSplit(data=re_sample, test_size=0.2,
                                                                exclude_unknowns=False,
                                                                verbose=True)
    return ratio_splitter


# @param
# data: whole data-set to be analysed
#    x: integer value corresponding to required percentage of data-set (e.g. 10000/100000 = 10%)
# return randomly sampled subset of data
def sample_size(data, x):
    boot = resample(data, replace=False, n_samples=x, random_state=1)
    return boot


# @param bootstrap object wrapping int size re-sample of data set and ratio split of that data
# method trains model on data sets then runs experiment on each
def train_test(bootstrap):
    training_data = bootstrap.ratio_splitter.train_set
    test_data = bootstrap.ratio_splitter.test_set
    # model.fit(training_data)
    Model.run_experiment(bootstrap.ratio_splitter)


# call to main method of project
main()
