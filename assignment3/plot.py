import os
import sys
import time
import argparse
import matplotlib.pyplot as plt
from datetime import datetime


def compute(script, num_cores):
    script_path = os.path.dirname(os.path.abspath(__file__)) + "/" + script
    script_args = ["--local-tmp-dir", "output", "--runner", "local", "--num-cores", str(num_cores)] + sys.argv[1:]
    command = ["python3", script_path] + script_args

    start_time = time.time()
    os.system(" ".join(command))
    duration = time.time() - start_time

    print("Time taken for {} cores: {} seconds".format(num_cores, duration))
    print("====================================================")
    return duration


def main(args):
    cores = [1, 2, 4, 8, 16, 32]
    durations = [compute(args.script, core) for core in cores]
    theoretical_durations = [(durations[0] / core) for core in cores]
    speedups = [(durations[0] / duration) for duration in durations]

    f = 0.8
    amdahl_speedups = [1/((1-f)+(f/core)) for core in cores]
    amdahl_durations = [(durations[0]/speedup) for speedup in amdahl_speedups]

    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(12, 6)
    fig.suptitle("Statistics computation performance with MRJob")

    axes[0].grid()
    axes[0].set_xscale("log", basex=2)
    axes[0].set_xlabel("Cores")
    axes[0].set_ylabel("Speedup")
    axes[0].plot(cores, cores, "ro-", label="Theoretical speedup (100% in parallel)")
    axes[0].plot(cores, amdahl_speedups, "mo-", label="Theoretical speedup (80% in parallel)")
    axes[0].plot(cores, speedups, "bo-", label="Actual speedup")
    axes[0].legend()

    axes[1].grid()
    axes[1].set_xscale("log", basex=2)
    axes[1].set_xlabel("Cores")
    axes[1].set_ylabel("Time taken (s)")
    axes[1].plot(cores, theoretical_durations, "ro-", label="Theoretical time (100% in parallel)")
    axes[1].plot(cores, amdahl_durations, "mo-", label="Theoretical time (80% in parallel")
    axes[1].plot(cores, durations, "bo-", label="Actual time")
    axes[1].legend()

    if args.output is None:
        now = datetime.today().strftime("%y%m%d_%H%M%S")
        plt.savefig(os.path.dirname(os.path.abspath(__file__)) + "/output/" + now + ".png")
    else:
        plt.savefig(os.path.dirname(os.path.abspath(__file__)) + "/" + args.output)
    plt.show()


if __name__ == "__main__":

    examples = """Examples of usage:
        python plot.py -s problem1a.py /data/2020-DAT346-DIT873-TLSD/DATASETS/assignment3.dat
        python plot.py -s problem1e.py /data/2020-DAT346-DIT873-TLSD/DATASETS/assignment3.dat
        python plot.py -s problem1.py /data/2020-DAT346-DIT873-TLSD/DATASETS/assignment3.dat
        python plot.py -o out.png -s problem1.py /data/2020-DAT346-DIT873-TLSD/DATASETS/assignment3.dat
    """
    parser = argparse.ArgumentParser(description="Plot speedup graph for problem 1",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=examples)
    parser.add_argument("--output", "-o",
                        default=None,
                        type=str,
                        help="Name of the file to output the plot")

    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument("--script", "-s",
                                required=True,
                                type=str,
                                help="Name of the script to use for plotting")

    known_args, other_args = parser.parse_known_args()
    sys.argv = sys.argv[:1] + other_args
    main(known_args)
