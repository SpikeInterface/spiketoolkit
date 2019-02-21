import pytest
import spikeextractors as se
from spiketoolkit.sorters import SpykingcircusSorter

from spiketoolkit.sorters.tests.common_tests import SorterCommonTestSuite

# This run several tests
@pytest.mark.skipif(not SpykingcircusSorter.installed)
class SpykingcircusCommonTestSuite(SorterCommonTestSuite):
    SorterCLass = SpykingcircusSorter



if __name__ == '__main__':
    SpykingcircusCommonTestSuite().test_on_toy()
    #~ SpykingcircusCommonTestSuite().test_several_groups()
    

    
