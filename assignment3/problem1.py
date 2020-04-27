import sys
import math
import heapq
from mrjob.job import MRJob

# Keys:
#   stats1 will be used as key for [mean, stdev]
#   stats2 will be used as key for [min, max, median, bins]

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
        value = float(value)

        if filter_group is None or int(group) == filter_group:
            yield ("stats1", value)
            yield ("stats2", value)


    def combiner(self, key, values):
        if key == "stats1":
            yield ("stats1", self.combiner_stats1(values))
        elif key == "stats2":
            yield ("stats2", self.combiner_stats2(values))


    def reducer(self, key, data):
        if key == "stats1":
            mean_value, stdev, n_values = self.reducer_stats1(data)

            yield ("Count", n_values)
            yield ("Mean", mean_value)
            yield ("Stdev", stdev)

        elif key == "stats2":
            min_value, max_value, median, bins = self.reducer_stats2(data)

            yield("Min", min_value)
            yield("Max", max_value)
            yield("Median", median)

            for idx, bin_count in enumerate(bins):
                yield("Bin{}".format(idx+1), bin_count)


    def combiner_stats1(self, values):
        sum_of_values, sum_of_squares, n_values = 0, 0, 0

        for v in values:
            sum_of_values += v
            sum_of_squares += v**2
            n_values += 1

        return sum_of_values, sum_of_squares, n_values


    def combiner_stats2(self, values):
        return sorted(values)


    def reducer_stats1(self, tuples):
        sum_of_values, sum_of_squares, n_values = 0, 0, 0

        for t in tuples:
            sum_of_values += t[0]
            sum_of_squares += t[1]
            n_values += t[2]

        mean_value = sum_of_values / n_values
        stdev = math.sqrt(sum_of_squares / n_values - mean_value**2)
        return mean_value, stdev, n_values


    def reducer_stats2(self, sorted_lists):
        values = list(heapq.merge(*sorted_lists))
        n_values = len(values)
        min_value = values[0]
        max_value = values[-1]

        if n_values % 2 == 0:
            median = (values[int((n_values/2) - 1)] + values[int(n_values/2)]) / 2
        else:
            median = values[int(n_values/2)]

        step = (max_value - min_value) / self.options.bins
        bins = [0] * self.options.bins
        for v in values:
            bin_idx = -1 if v == max_value else int((v - min_value) / step)
            bins[bin_idx] += 1

        return min_value, max_value, median, bins


if __name__ == "__main__":
    ComputeStatistics.run()
