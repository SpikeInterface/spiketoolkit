import pytest
import numpy as np
from numpy.testing import assert_array_equal

import spikeextractors as se

from spiketoolkit.comparison import compare_multiple_sorters


def make_sorting(times1, labels1, times2, labels2, times3, labels3):
    sorting1 = se.NumpySortingExtractor()
    sorting2 = se.NumpySortingExtractor()
    sorting3 = se.NumpySortingExtractor()
    sorting1.set_times_labels(np.array(times1), np.array(labels1))
    sorting2.set_times_labels(np.array(times2), np.array(labels2))
    sorting3.set_times_labels(np.array(times3), np.array(labels3))
    return sorting1, sorting2, sorting3


def test_compare_multiple_sorters():
    # simple match
    sorting1, sorting2, sorting3 = make_sorting([100, 200, 300, 400, 500, 600, 700, 800, 900],
                                                [0, 1, 2, 0, 1, 2, 0, 1, 2],
                                                [101, 201, 301, 400, 501, 598, 702, 801, 899, 1000, 1100, 2000, 3000],
                                                [0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 3, 4, 4],
                                                [101, 201, 301, 400, 500, 600, 700, 800, 900, 1000, 1100, 2000, 3000,
                                                 3100, 3200, 3300],
                                                [0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 3, 4, 4, 5, 5, 5],)
    msc = compare_multiple_sorters([sorting1, sorting2, sorting3])

    agr = msc._do_agreement_matrix()
    print(agr)

    assert len(msc.get_agreement_sorting(minimum_matching=3).get_unit_ids()) == 3
    assert len(msc.get_agreement_sorting(minimum_matching=2).get_unit_ids()) == 5
    assert len(msc.get_agreement_sorting().get_unit_ids()) == 6


if __name__ == '__main__':
    test_compare_multiple_sorters()