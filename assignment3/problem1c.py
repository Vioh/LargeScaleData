import os
import uuid
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def run(args, n):
    tempfile = "/tmp/" + str(uuid.uuid4())
    os.system("head -n " + str(n) + " " + args.datafile + " > " + tempfile)

    script_path = os.path.dirname(os.path.abspath(__file__)) + "/problem1a.py"
    script_args = ["--local-tmp-dir", "output", "--runner", "local", "--num-cores", str(args.workers), tempfile]
    command = ["python3", script_path] + script_args

    start_time = time.time()
    os.system(" ".join(command))
    duration = time.time() - start_time

    os.system("rm " + tempfile)
    print("Time taken for {} datapoints: {} seconds".format(n, duration))
    print("====================================================")
    return duration


def main(args):
    ns = [f*args.n for f in [1, 2, 4, 8, 16, 32]]
    durations = [run(args, n) for n in ns]

    coef = np.polyfit(ns, durations, 1)
    fit = np.poly1d(coef)

    fig, ax = plt.subplots()
    fig.set_size_inches(6, 6)

    ax.grid()
    ax.set_xlabel("n")
    ax.set_ylabel("Time (s)")
    ax.plot(ns, durations, "bo")
    ax.plot(ns, fit(ns), "k--")

    label = "" if args.label is None else args.label + "_"
    now = datetime.today().strftime("%y%m%d_%H%M%S")
    outfile = os.path.dirname(os.path.abspath(__file__)) + "/output/" + label + now + ".png"
    plt.savefig(outfile)
    plt.show()


if __name__ == "__main__":

    examples = """Examples of usage:
        python problem1c.py -l best /data/2020-DAT346-DIT873-TLSD/DATASETS/assignment3.dat
    """
    parser = argparse.ArgumentParser(description="Plot empirical computational complexity for problem1a.py",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=examples)

    parser.add_argument("--label", "-l",
                        default=None,
                        type=str,
                        help="Label for the run")
    parser.add_argument("--workers", "-w",
                        default=8,
                        type=int,
                        help="Number of workers")
    parser.add_argument("--n", "-n",
                        default=10000,
                        type=int,
                        help="Smallest length of data")
    parser.add_argument("datafile",
                        type=str,
                        help="Name of the data file")
    args = parser.parse_args()
    main(args)
