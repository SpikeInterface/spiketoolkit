import spikeextractors as se
import spiketoolkit as st
import spikewidgets as sw

def test_tdc():
    recording, sorting = se.example_datasets.toy_example1(num_channels=4, duration=30)
    print(recording)
    print(sorting)
    
    params = st.sorters.tridesclous_default_params
    sorting_tdc = st.sorters.tridesclous(recording=recording, 
            output_folder='test_tdc_workdir', debug=True, **params)
    
    
    
    
    
if __name__ == '__main__':
    test_tdc()
    