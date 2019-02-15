import pytest
import spikeextractors as se
from spiketoolkit.sorters import SpykingcircusSorter, run_spykingcircus

@pytest.mark.skipif(not SpykingcircusSorter.installed)
def test_tdc():
    recording, sorting_gt = se.example_datasets.toy_example1(num_channels=4, duration=30)
    print(recording)
    print(sorting_gt)
    
    params = SpykingcircusSorter.default_params()
    sorting_sc = run_spykingcircus(recording,  output_folder='test_spykingcircus', debug=False, **params)
    
    print(sorting_sc)
    print(sorting_sc.getUnitIds())
    for unit_id in sorting_sc.getUnitIds():
        print('unit #', unit_id, 'nb', len(sorting_sc.getUnitSpikeTrain(unit_id)))
    
    
    
    
if __name__ == '__main__':
    test_tdc()
    
