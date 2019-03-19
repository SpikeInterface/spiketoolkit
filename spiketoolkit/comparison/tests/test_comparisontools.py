import pytest
import numpy as np

import spikeextractors as se

from spiketoolkit.comparison import get_matching, get_counting, get_confusion




def make_sorting(times1, labels1, times2, labels2):
    sorting1 = se.NumpySortingExtractor()
    sorting2 = se.NumpySortingExtractor()
    sorting1.setTimesLabels(np.array(times1), np.array(labels1))
    sorting2.setTimesLabels(np.array(times2), np.array(labels2))
    return sorting1, sorting2
    
    


def test_get_matching():
    delta_tp=10
    min_accuracy=0.5

    sorting1 = se.NumpySortingExtractor()
    sorting2 = se.NumpySortingExtractor()
    
    sorting1, sorting2 = make_sorting([100, 200, 300, 400], [0, 0, 1, 0], 
                                                            [101, 201, 301, ], [0, 0, 5])
    
    event_counts_1,  event_counts_2, matching_event_counts_12, best_match_units_12, matching_event_counts_21, \
                best_match_units_21, unit_map12,  unit_map21 = get_matching(sorting1, sorting2, delta_tp, min_accuracy)
    
    
    
def test_get_counting():
    pass

def test_get_confusion():
    pass




if __name__ == '__main__':
    test_get_matching()
    #~ test_get_counting()
