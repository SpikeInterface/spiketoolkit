import pytest
import spikeextractors as se
from spiketoolkit.sorters import TridesclousSorter, run_tridesclous

@pytest.mark.skipif(not TridesclousSorter.installed)
def test_tdc():
    recording, sorting_gt = se.example_datasets.toy_example1(num_channels=4, duration=30)
    print(recording)
    print(sorting_gt)
    
    params = TridesclousSorter.default_params()
    sorting_tdc = run_tridesclous(recording,  output_folder='test_tdc', debug=False, **params)
    
    print(sorting_tdc)
    print(sorting_tdc.getUnitIds())
    for unit_id in sorting_tdc.getUnitIds():
        print('unit #', unit_id, 'nb', len(sorting_tdc.getUnitSpikeTrain(unit_id)))
    
    
    
    
if __name__ == '__main__':
    test_tdc()
    
