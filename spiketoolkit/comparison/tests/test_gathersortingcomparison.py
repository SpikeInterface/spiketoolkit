
import os
import shutil
import time

import pytest
from spiketoolkit.sorters import run_sorters, collect_results
from spiketoolkit.comparison import gather_sorting_comparison

import spikeextractors as se




path = 'test_TDC_vs_HS2/'
working_folder = path + '/working_folder'


def setup_module():
    _run_sorters()
    

def _run_sorters():
    path = 'test_TDC_vs_HS2/'

    if os.path.exists(path):
        shutil.rmtree(path)

    
    # cerate several recording/sorting
    rec0, gt_sorting0 = se.example_datasets.toy_example(num_channels=4, duration=30)
    rec1, gt_sorting1 = se.example_datasets.toy_example(num_channels=32, duration=30)
    
    #~ se.NumpySortingExtractor.writeSorting(gt_sorting0, path+'gt_sorting0')
    #~ se.NumpySortingExtractor.writeSorting(gt_sorting1, path+'gt_sorting1')
    
    #~ print(gt_sorting0)
    
    #~ exit()
    
    recording_dict = {'toy_tetrode' : rec0, 'toy_probe32': rec1}
    
    sorter_list = ['tridesclous', 'herdingspikes']
    
    
    # simple loop
    t0 = time.perf_counter()
    run_sorters(sorter_list, recording_dict, working_folder, engine=None)
    t1 = time.perf_counter()
    print('total run time', t1-t0)
    


def test_gather_sorting_comparison():
    """
    This test illustrate how to lauche several sorter on several datasets
    and then collect all result in one function.
    """
    
    
    
    #~ ground_truths = {'toy_tetrode' : gt_sorting0, 'toy_probe32': gt_sorting1}
    
    ground_truths = {}
    
    comp_dataframes = gather_sorting_comparison(working_folder, ground_truths,use_multi_index=True)
    for k, df in comp_dataframes.items():
        print('*'*10)
        print(k)
        print(df)
        


    

if __name__ == '__main__':
    #~ setup_module()
    test_gather_sorting_comparison() 


