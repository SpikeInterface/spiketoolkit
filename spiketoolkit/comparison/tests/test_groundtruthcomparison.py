import pytest
import numpy as np
from numpy.testing import assert_array_equal

import spikeextractors as se

from spiketoolkit.comparison import compare_sorter_to_ground_truth



def make_sorting(times1, labels1, times2, labels2):
    gt_sorting = se.NumpySortingExtractor()
    other_sorting = se.NumpySortingExtractor()
    gt_sorting.set_times_labels(np.array(times1), np.array(labels1))
    other_sorting.set_times_labels(np.array(times2), np.array(labels2))
    return gt_sorting, other_sorting
    


def test_compare_sorter_to_ground_truth():
    # simple match
    gt_sorting, other_sorting = make_sorting([100, 200, 300, 400], [0, 0, 1, 0], 
                                                            [101, 201, 301, 600], [0, 0, 5, 6])
    sc = compare_sorter_to_ground_truth(gt_sorting, other_sorting, exhaustive_gt=True)
    
    sc._do_confusion()
    #~ print(sc._confusion_matrix)
    
    raw_count = sc.get_raw_count()
    #~ print(raw_count)
    
    methods = ['raw_count', 'by_spiketrain', 'pooled_with_sum', 'pooled_with_average',]
    for method in methods:
        perf = sc.get_performance(method=method)
        #~ print(perf)
    
    for method in methods:
        sc.print_performance(method=method)
    
    
    # units #6 in other_osrting is fake
    fake_ids = sc.get_fake_units_list()
    assert_array_equal(fake_ids, [6])
    
    nb_fake = sc.get_number_fake_units()
    assert nb_fake == 1
    #~ print(nb_fake)
    
    
if __name__ == '__main__':
    test_compare_sorter_to_ground_truth()