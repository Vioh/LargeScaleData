import sys
import math
import heapq
from mrjob.job import MRJob

class ComputeStatistics(MRJob):

    def configure_args(self):
        super().configure_args()
        self.add_passthru_arg("--group", "-g", type=int, default=None,
                              help="Group to compute statistics on")
        self.add_passthru_arg("--bins", "-b", type=int, default=10,
                              help="Number of bins to plot histogram")


    def mapper(self, _, line):
        _, group, value = line.split("\t")
        filter_group = self.options.group

        if filter_group is None or int(group) == filter_group:
            yield (None, float(value))


    def combiner(self, _, values):
        values = list(values)
        sum_of_values, sum_of_squares, n_values = 0, 0, 0
        min_value, max_value = math.inf, -math.inf

        for v in values:
            sum_of_values += v
            sum_of_squares += v**2
            n_values += 1

            if v < min_value:
                min_value = v
            if v > max_value:
                max_value = v

        step = (max_value - min_value) / self.options.bins
        bins = [0] * self.options.bins

        for v in values:
            bin_idx = -1 if v == max_value else int((v - min_value) / step)
            bins[bin_idx] += 1

        yield (None, (sum_of_values, sum_of_squares, n_values, min_value, max_value, bins))


    def reducer(self, _, tuples):
        sum_of_values, sum_of_squares, n_values = 0, 0, 0
        min_value, max_value = math.inf, -math.inf
        bins = [0] * self.options.bins

        for t in tuples:
            sum_of_values += t[0]
            sum_of_squares += t[1]
            n_values += t[2]
            bins = [x + y for x, y in zip(bins, t[-1])]

            if t[3] < min_value:
                min_value = t[3]
            if t[4] > max_value:
                max_value = t[4]

        mean_value = sum_of_values / n_values
        stdev = math.sqrt(sum_of_squares / n_values - mean_value**2)

        yield("Count", n_values)
        yield("Mean", mean_value)
        yield("Stdev", stdev)
        yield("Min", min_value)
        yield("Max", max_value)

        for idx, bin_count in enumerate(bins):
            yield("Bin{}".format(idx+1), bin_count)


if __name__ == "__main__":
    ComputeStatistics.run()
