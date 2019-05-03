import unittest
import pytest

from spiketoolkit.sorters import TridesclousSorter, run_tridesclous
from spiketoolkit.sorters.tests.common_tests import SorterCommonTestSuite

import spikeextractors as se




# This run several tests
@pytest.mark.skipif(not TridesclousSorter.installed, reason='tridesclous not installed')
class TridesclousCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = TridesclousSorter

@pytest.mark.skipif(not TridesclousSorter.installed, reason='tridesclous not installed')
def test_run_tridesclous():
    recording, sorting_gt = se.example_datasets.toy_example(num_channels=4, duration=30)
    
    params = TridesclousSorter.default_params()
    sorting = run_tridesclous(recording,  **params)
    
    print(sorting)
    print(sorting.get_unit_ids())
    for unit_id in sorting.get_unit_ids():
        print('unit #', unit_id, 'nb', len(sorting.get_unit_spike_train(unit_id)))



if __name__ == '__main__':
    #~ test_run_tridesclous()
    #~ TridesclousCommonTestSuite().test_on_toy()
    #~ TridesclousCommonTestSuite().test_several_groups()
    TridesclousCommonTestSuite().test_with_BinDatRecordingExtractor()
    #~ unittest.main()
    
    
    
