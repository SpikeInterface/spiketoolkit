import pytest
import numpy as np
from numpy.testing import assert_array_equal

import spikeextractors as se

from spiketoolkit.comparison import SortingComparison


def make_sorting(times1, labels1, times2, labels2):
    sorting1 = se.NumpySortingExtractor()
    sorting2 = se.NumpySortingExtractor()
    sorting1.set_times_labels(np.array(times1), np.array(labels1))
    sorting2.set_times_labels(np.array(times2), np.array(labels2))
    return sorting1, sorting2


def test_SortingComparison():
    # simple match
    sorting1, sorting2 = make_sorting([100, 200, 300, 400], [0, 0, 1, 0],
                                      [101, 201, 301, ], [0, 0, 5])
    sc = SortingComparison(sorting1, sorting2, count=True)

    sc._do_confusion()
    print(sc._confusion_matrix)

    methods = ['by_spiketrain', 'pooled_with_sum', 'pooled_with_average', ]
    for method in methods:
        perf = sc.get_performance(method=method)
        # ~ print(perf)

    for method in methods:
        sc.print_performance(method=method)


if __name__ == '__main__':
    test_SortingComparison()