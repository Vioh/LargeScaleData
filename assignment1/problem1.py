import sys
import os
import time
import importlib
import matplotlib.pyplot as plt
from argparse import Namespace

compute_pi = importlib.import_module("mp-pi-montecarlo-pool").compute_pi


def disablePrint():
    sys.stdout = open(os.devnull, 'w')


def enablePrint():
    sys.stdout = sys.__stdout__


def measure_duration(steps, cores):
    """Return number of seconds it takes to compute pi"""
    print("Measuring duration for %d cores..." % cores)
    args = Namespace(steps=steps, workers=cores)
    disablePrint()
    start_time = time.time()
    compute_pi(args)
    end_time = time.time()
    enablePrint()
    return end_time - start_time


def main():
    steps = int(1e7)
    cores = [1, 2, 4, 8, 16, 32]
    durations = [measure_duration(steps, core) for core in cores]
    theoretical_durations = [(durations[0] / core) for core in cores]
    speedups = [(durations[0] / duration) for duration in durations]
    
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(12, 6)
    fig.suptitle("Number of steps = " + str(steps))

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

    plt.savefig("problem1.png")
    plt.show()


if __name__ == "__main__":
    main()
