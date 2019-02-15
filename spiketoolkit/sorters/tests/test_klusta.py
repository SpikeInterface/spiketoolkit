import pytest
import spikeextractors as se
from spiketoolkit.sorters import KlustaSorter, run_klusta


@pytest.mark.skipif(not KlustaSorter.installed)
def test_klusta():
    recording, sorting_gt = se.example_datasets.toy_example1(num_channels=4, duration=30)
    print(recording)
    print(sorting_gt)
    
    params = KlustaSorter.default_params()
    sorting_K = run_klusta(recording, output_folder='test_klusta', **params)
    
    print(sorting_K)
    print(sorting_K.getUnitIds())
    for unit_id in sorting_K.getUnitIds():
        print('unit #', unit_id, 'nb', len(sorting_K.getUnitSpikeTrain(unit_id)))
    
    
    
    
if __name__ == '__main__':
    test_klusta()
