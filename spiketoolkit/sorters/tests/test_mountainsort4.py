import pytest
import spikeextractors as se
from spiketoolkit.sorters import Mountainsort4Sorter, run_mountainsort4

@pytest.mark.skipif(not Mountainsort4Sorter.installed)
def test_mountainsort4():
    recording, sorting_gt = se.example_datasets.toy_example1(num_channels=4, duration=30)
    print(recording)
    print(sorting_gt)
    
    params = Mountainsort4Sorter.default_params()
    sorting_tdc = run_mountainsort4(recording,  output_folder='test_tdc', debug=False, **params)
    
    print(sorting_tdc)
    print(sorting_tdc.getUnitIds())
    for unit_id in sorting_tdc.getUnitIds():
        print('unit #', unit_id, 'nb', len(sorting_tdc.getUnitSpikeTrain(unit_id)))
    
    
    
    
if __name__ == '__main__':
    test_mountainsort4()
    
