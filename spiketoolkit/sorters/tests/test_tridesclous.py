import pytest
import spikeextractors as se
from spiketoolkit.sorters import TridesclousSorter, run_tridesclous

from spiketoolkit.sorters.tests.common_tests import SorterCommonTestSuite


# This run several tests
@pytest.mark.skipif(not TridesclousSorter.installed)
class TridesclousCommonTestSuite(SorterCommonTestSuite):
    SorterCLass = TridesclousSorter


    
if __name__ == '__main__':
    #~ TridesclousCommonTestSuite().test_on_toy()
    TridesclousCommonTestSuite().test_several_groups()
    
    
    
