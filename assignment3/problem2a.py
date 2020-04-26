import numpy as np
import math
from pyspark import SparkContext
import time


def start_process(workers, file_name):
    sc = SparkContext(master='local[' + str(workers) + ']').getOrCreate()
    distributed_file = sc.textFile(file_name)
    values = distributed_file.map(lambda l: float(l.split('\t')[2]))
    return values, distributed_file.count(), sc


def calculate_mean(values, length):
    return values.sum() / length


def calculate_std(values, mean, length):
    std_sum = values.map(lambda x: (x - mean)**2).sum()
    return np.sqrt(std_sum / length)


def make_histogram(values, bin_num):
    max_value = values.max()
    min_value = values.min()
    value_range = max_value - min_value
    step = float(value_range) / float(bin_num)
    bins = values.map(lambda x: (bin_num - 1) if x == max_value else
                      math.floor((x - min_value) / step)) \
                 .map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y) \
                 .sortByKey().collect()
    return bins, min_value, max_value, step


def calculate_median(values, length):
    values = values.sortBy(lambda x: x).collect()
    if length % 2 == 0:
        median = (values[int((length / 2) - 1)] + values[int(length / 2)]) / 2
    else:
        median = values[int(length / 2)]
    return median


def run_workers(workers):
    start = time.time()
    val, length, sc = start_process(workers, '/data/2020-DAT346-DIT873-TLSD/DATASETS/assignment3.dat')
    print("{0:19s} {1:-8d}".format("Count", length))
    mean = calculate_mean(val, length)
    print("{0:20s} {1:-3.5f}".format("Mean", mean))
    std = calculate_std(val, mean, length)
    print("{0:20s} {1:-3.5f}".format("Standard deviation", std))
    median = calculate_median(val, length)
    print("{0:20s} {1:-3.5f}".format("Median", median))
    bins, mini, maxi, step = make_histogram(val, 10)
    print("{0:20s} {1:-3.5f}".format("Min value", mini))
    print("{0:20s} {1:-3.5f}".format("Max value", maxi))
    print("---------------------------------")
    print("{0:11s} {1:10s}".format("Bin range", "Occurrences"))
    for tup in bins:
        print("{0:.2f} - {1:.2f} {2:-10d}".format(mini + tup[0] * step, (mini + tup[0] * step)
                                                  + step - 0.0001, tup[1]))
    sc.stop()
    end = time.time()
    duration = end - start
    print(f"Duration for {workers} workers: {duration} seconds.")
    return duration


if __name__ == "__main__":
    run_workers(4)
