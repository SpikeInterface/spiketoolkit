import unittest
import pytest
import spikeextractors as se
from spiketoolkit.sorters import Mountainsort4Sorter

from spiketoolkit.sorters.tests.common_tests import SorterCommonTestSuite

# This run several tests
@pytest.mark.skipif(not Mountainsort4Sorter.installed, reason='moutainsort4 not installed')
class Mountainsort4CommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
   SorterClass = Mountainsort4Sorter



if __name__ == '__main__':
   #~ Mountainsort4CommonTestSuite().test_on_toy()
    Mountainsort4CommonTestSuite().test_several_groups()
    
