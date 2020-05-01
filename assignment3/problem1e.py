import sys
import math
import heapq
from mrjob.job import MRJob

class ComputeMedian(MRJob):

    def configure_args(self):
        super().configure_args()
        self.add_passthru_arg("--group", "-g", type=int, default=None,
                              help="Group to compute statistics on")


    def mapper(self, _, line):
        _, group, value = line.split("\t")
        filter_group = self.options.group

        if filter_group is None or int(group) == filter_group:
            yield (None, float(value))


    def combiner(self, _, values):
        yield (None, sorted(values))


    def reducer(self, _, sorted_lists):
        values = list(heapq.merge(*sorted_lists))
        n_values = len(values)

        if n_values % 2 == 0:
            median = (values[int((n_values/2) - 1)] + values[int(n_values/2)]) / 2
        else:
            median = values[int(n_values/2)]

        yield("Count", n_values)
        yield("Median", median)


if __name__ == "__main__":
    ComputeMedian.run()
