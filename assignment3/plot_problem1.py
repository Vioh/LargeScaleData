import os
import sys
import time
import argparse
import matplotlib.pyplot as plt


def compute(num_cores):
    script_path = os.path.dirname(os.path.abspath(__file__)) + "/problem1.py"
    command = ["python3", script_path, "--runner", "local", "--num-cores", str(num_cores)] + sys.argv[1:]

    start_time = time.time()
    os.system(" ".join(command))
    duration = time.time() - start_time

    print("Time taken for {} cores: {} seconds".format(num_cores, duration))
    print("====================================================")
    return duration


def main():
    cores = [1, 2, 4, 8, 16, 32]
    durations = [compute(core) for core in cores]
    theoretical_durations = [(durations[0] / core) for core in cores]
    speedups = [(durations[0] / duration) for duration in durations]

    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(12, 6)
    fig.suptitle("Statistics computation performance with MRJob")

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
    parser = argparse.ArgumentParser(description="Plot speedup graph for problem 1")
    parser.add_argument("--group", "-g",
                        default=None,
                        type = int,
                        help="Group to compute statistics on")
    parser.add_argument("--bins", "-b",
                        default=10,
                        type = int,
                        help="Number of bins to plot histogram")
    parser.add_argument("filepath",
                        help="File to be processed")
    parser.parse_args()
    main()
