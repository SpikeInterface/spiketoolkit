import pytest
import spikeextractors as se
from spiketoolkit.sorters import IronclustSorter, run_ironclust

@pytest.mark.skipif(not IronclustSorter.installed)
def test_tdc():
    recording, sorting_gt = se.example_datasets.toy_example1(num_channels=4, duration=30)
    print(recording)
    print(sorting_gt)
    
    params = IronclustSorter.default_params()
    sorting_ic = run_ironclust(recording,  output_folder='test_ironclust', debug=False, **params)
    
    print(sorting_ic)
    print(sorting_ic.getUnitIds())
    for unit_id in sorting_ic.getUnitIds():
        print('unit #', unit_id, 'nb', len(sorting_ic.getUnitSpikeTrain(unit_id)))
    
    
    
    
if __name__ == '__main__':
    test_tdc()
    
