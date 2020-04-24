from pyspark import SparkContext
import time
import matplotlib.pyplot as plt


def start_process(workers, file_name):
    sc = SparkContext(master='local[' + str(workers) + ']').getOrCreate()
    distributed_file = sc.textFile(file_name)
    values = distributed_file.map(lambda l: float(l.split('\t')[2]))
    return values, distributed_file.count(), sc


def calculate_median(values, length):
    values = values.sortBy(lambda x: x).collect()
    if length % 2 == 0:
        median = (values[int((length / 2) - 1)] + values[int(length / 2)]) / 2
    else:
        median = values[int(length / 2)]
    return median


def run_workers(workers):
    start = time.time()
    val, len, sc = start_process(workers, '/data/2020-DAT346-DIT873-TLSD/DATASETS/assignment3.dat')
    median = calculate_median(val, len)
    print("{0:15s} {1:-8.5f}".format("Median", median))
    sc.stop()
    end = time.time()
    duration = end - start
    print(f"Duration for {workers} workers: {duration} seconds.")
    return duration


def main_multiprocessing():
    cores = [1, 2, 4, 8, 16, 32]
    durations =[run_workers(core) for core in cores]
    theoretical_durations = [(durations[0] / core) for core in cores]
    speedups = [(durations[0] / duration) for duration in durations]

    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(12, 6)
    fig.suptitle(f"Statistics computation performance with PySpark")

    axes[0].grid()
    axes[0].set_xscale("log", basex=2)
    axes[0].set_xlabel("Cores")
    axes[0].set_ylabel("Speedup")
    axes[0].plot(cores, cores, "ro-", label="Theoretical speedup")
    axes[0].plot(cores, speedups, "bo-", label="Actual speedup")
    axes[0].legend()

    axes[1].grid()
    axes[1].set_xscale("log", basex=2)
    axes[1].set_xlabel("Cores")
    axes[1].set_ylabel("Time taken (s)")
    axes[1].plot(cores, theoretical_durations, "ro-", label="Theoretical time")
    axes[1].plot(cores, durations, "bo-", label="Actual time")
    axes[1].legend()

    plt.savefig("problem2b.png")
    # plt.show()


if __name__ == "__main__":
    main_multiprocessing()