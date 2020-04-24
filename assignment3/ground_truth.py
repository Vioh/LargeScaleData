import argparse
import numpy as np

def main(args):
    values = []
    lines = open(args.filepath, "r")

    for line in lines:
        _, group, value = line.strip().split("\t")

        if args.group is None or int(group) == args.group:
            values.append(float(value))

    values = np.array(values)
    print("Count:", np.size(values))
    print("Mean:", np.mean(values))
    print("Std:", np.std(values))
    print("Min:", np.min(values))
    print("Max:", np.max(values))
    print("Median:", np.median(values))
    print("====================")

    hist = np.histogram(values, bins=args.bins)
    for idx, bin_count in enumerate(hist[0]):
        print("Bin{}: {}".format(idx+1, bin_count))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute ground-truth statistics for assignment 3")
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
    args = parser.parse_args()
    main(args)
