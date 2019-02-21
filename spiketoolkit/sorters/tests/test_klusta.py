import pytest
import spikeextractors as se
from spiketoolkit.sorters import KlustaSorter

from spiketoolkit.sorters.tests.common_tests import SorterCommonTestSuite

# This run several tests
@pytest.mark.skipif(not KlustaSorter.installed, reason='klusta not installed')
class KlustaCommonTestSuite(SorterCommonTestSuite):
    SorterCLass = KlustaSorter



if __name__ == '__main__':
    KlustaCommonTestSuite().test_on_toy()
    KlustaCommonTestSuite().test_several_groups()
    
