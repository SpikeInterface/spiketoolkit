import pytest

from spiketoolkit.sorters import TridesclousSorter, run_tridesclous
from spiketoolkit.sorters.tests.common_tests import SorterCommonTestSuite

import spikeextractors as se


# This run several tests
@pytest.mark.skipif(not TridesclousSorter.installed)
class TridesclousCommonTestSuite(SorterCommonTestSuite):
    SorterCLass = TridesclousSorter

@pytest.mark.skipif(not TridesclousSorter.installed)
def test_run_tridesclous():
    recording, sorting_gt = se.example_datasets.toy_example1(num_channels=4, duration=30)
    
    params = TridesclousSorter.default_params()
    sorting = run_tridesclous(recording,  **params)
    
    print(sorting)
    print(sorting.getUnitIds())
    for unit_id in sorting.getUnitIds():
        print('unit #', unit_id, 'nb', len(sorting.getUnitSpikeTrain(unit_id)))



if __name__ == '__main__':
    test_run_tridesclous()
    #~ TridesclousCommonTestSuite().test_on_toy()
    #~ TridesclousCommonTestSuite().test_several_groups()
    
    
    
