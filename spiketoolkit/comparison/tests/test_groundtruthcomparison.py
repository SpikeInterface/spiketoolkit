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
    
    sc._do_confusion_matrix()
    #~ print(sc._confusion_matrix)
    
    
    methods = ['raw_count', 'by_spiketrain', 'pooled_with_sum', 'pooled_with_average',]
    for method in methods:
        perf = sc.get_performance(method=method)
        #~ print(perf)
    
    for method in methods:
        sc.print_performance(method=method)
    

    # test well detected units depending on thresholds
    good_units = sc.get_well_detected_units(tp_thresh=0.95)
    assert_array_equal(good_units, [1])
    good_units = sc.get_well_detected_units(tp_thresh=None, accuracy_thresh=.6)
    assert_array_equal(good_units, [0, 1])
    good_units = sc.get_well_detected_units(tp_thresh=None, fp_thresh=0.05)
    assert_array_equal(good_units, [0, 1])
    good_units = sc.get_well_detected_units(tp_thresh=None, cl_thresh=0.05)
    assert_array_equal(good_units, [0, 1])
    
    # combine thresh
    good_units = sc.get_well_detected_units(tp_thresh=0.95, accuracy_thresh=.6)
    assert_array_equal(good_units, [1])
    
    # count
    nb_ok = sc.count_well_detected_units(tp_thresh=0.95)
    assert nb_ok == 1
    
    
    # units #6 in other_osrting is fake
    fake_ids = sc.get_fake_units_in_other()
    assert_array_equal(fake_ids, [6])
    nb_fake = sc.count_fake_units_in_other()
    assert nb_fake == 1
    #~ print(nb_fake)
    
    


def test_get_performance():
    
    
    ######
    # simple match
    gt_sorting, other_sorting = make_sorting([100, 200, 300, 400], [0, 0, 1, 0], 
                                                            [101, 201, 301, ], [0, 0, 5])
    sc = compare_sorter_to_ground_truth(gt_sorting, other_sorting, exhaustive_gt=True, delta_frames=10)
    
    
    perf = sc.get_performance('raw_count')
    assert perf.loc[0, 'tp'] == 2
    assert perf.loc[1, 'tp'] == 1
    assert perf.loc[0, 'fn'] == 1
    assert perf.loc[1, 'fn'] == 0
    assert perf.loc[0, 'fp'] == 0
    assert perf.loc[1, 'fp'] == 0
    
    perf = sc.get_performance('pooled_with_sum')
    assert perf['tp_rate'] == 0.75
    assert perf['fn_rate'] == 0.25
    
    perf = sc.get_performance('by_spiketrain')
    assert perf.loc[0, 'tp_rate'] == 2 / 3.
    assert perf.loc[0, 'cl_rate'] == 0
    assert perf.loc[0, 'fn_rate'] == 1 / 3.
    assert perf.loc[0, 'fp_rate'] == 0 
    
    ######
    # match when 2 units fire at same time
    gt_sorting, other_sorting = make_sorting([100, 100, 200, 200, 300], [0, 1, 0, 1, 0], 
                                                            [100, 100, 200, 200, 300], [0, 1, 0, 1, 0],)
    sc = compare_sorter_to_ground_truth(gt_sorting, other_sorting, exhaustive_gt=True)
    
    perf = sc.get_performance('raw_count')
    assert perf.loc[0, 'tp'] == 3
    assert perf.loc[0, 'cl'] == 0
    assert perf.loc[0, 'fn'] == 0
    assert perf.loc[0, 'fp'] == 0
    assert perf.loc[0, 'nb_gt'] == 3
    assert perf.loc[0, 'nb_other'] == 3
    
    perf = sc.get_performance('pooled_with_sum')
    assert perf['tp_rate'] == 1.
    assert perf['fn_rate'] == 0.

    
    
if __name__ == '__main__':
    test_compare_sorter_to_ground_truth()
    #~ test_get_performance()