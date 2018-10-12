import os, sys
import unittest

def append_to_path(dir0): # A convenience function
    if dir0 not in sys.path:
        sys.path.append(dir0)
append_to_path(os.getcwd()+'/..')

import spikewidgets as sw
import spiketoolkit as st
 
class Test001(unittest.TestCase):
    def setUp(self):
      pass
        
    def tearDown(self):
        pass
     
    def test_toy_example1(self):
      recording,sorting_true=sw.example_datasets.toy_example1(duration=5)
      sorting=st.sorters.mountainsort4(recording,detect_sign=-1,adjacency_radius=-1)

if __name__ == '__main__':
    unittest.main()
