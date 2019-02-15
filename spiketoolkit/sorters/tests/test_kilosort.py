import pytest
import spikeextractors as se
from spiketoolkit.sorters import KilosortSorter, run_kilosort

@pytest.mark.skipif(not KilosortSorter.installed)
def test_tdc():
    recording, sorting_gt = se.example_datasets.toy_example1(num_channels=32, duration=30)
    print(recording)
    print(sorting_gt)

    params = KilosortSorter.default_params()
    sorting_tdc = run_kilosort(recording,  output_folder='test_kilosort', debug=False, **params)
    
    print(sorting_tdc)
    print(sorting_tdc.getUnitIds())
    for unit_id in sorting_tdc.getUnitIds():
        print('unit #', unit_id, 'nb', len(sorting_tdc.getUnitSpikeTrain(unit_id)))
    
    
    
    
if __name__ == '__main__':
    test_tdc()
    
