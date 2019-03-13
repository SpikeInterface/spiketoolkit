import unittest
import pytest
import spikeextractors as se
from spiketoolkit.sorters import YassSorter

from spiketoolkit.sorters.tests.common_tests import SorterCommonTestSuite

# This run several tests
@pytest.mark.skipif(not YassSorter.installed, reason='yass not installed')
class YassCommonTestSuite(SorterCommonTestSuite, unittest.TestCase):
    SorterClass = YassSorter



if __name__ == '__main__':
    YassCommonTestSuite().test_on_toy()
    YassCommonTestSuite().test_several_groups()
    
