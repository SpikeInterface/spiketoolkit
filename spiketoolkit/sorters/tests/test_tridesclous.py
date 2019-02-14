import pytest
from spiketoolkit.sorters import TridesclousSorter, run_tridesclous

@pytest.skipif(not TridesclousSorter.installed)
def test_tdc():
    recording, sorting_gt = se.example_datasets.toy_example1(num_channels=4, duration=30)
    print(recording)
    print(sorting)
    
    params = st.sorters.tridesclous_default_params
    sorting_tdc = st.sorters.tridesclous(recording=recording, 
            output_folder='test_tdc_workdir', debug=False, **params)
    
    print(sorting_tdc)
    print(sorting_tdc.getUnitIds())
    for unit_id in sorting_tdc.getUnitIds():
        print('unit #', unit_id, 'nb', len(sorting_tdc.getUnitSpikeTrain(unit_id)))
    
    
    
    
if __name__ == '__main__':
    test_tdc()
    
