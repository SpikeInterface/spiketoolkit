
import os
import shutil
import time
import pickle

import pytest
#~ from spiketoolkit.sorters import run_sorters, collect_results
from spiketoolkit.comparison import (setup_comparison_study, run_study_sorters,
            aggregate_sorting_comparison, aggregate_performances_table)

import spikeextractors as se




study_folder = 'test_TDC_vs_HS2/'



def setup_module():
    if os.path.exists(study_folder):
        shutil.rmtree(study_folder)
    
    _setup_comparison_study()
    
    _run_study_sorters()


def _setup_comparison_study():
    rec0, gt_sorting0 = se.example_datasets.toy_example(num_channels=4, duration=30)
    rec1, gt_sorting1 = se.example_datasets.toy_example(num_channels=32, duration=30)
    
    gt_dict = {
        'toy_tetrode' : (rec0, gt_sorting0),
        'toy_probe32' : (rec1, gt_sorting1),
    }
    
    setup_comparison_study(study_folder, gt_dict)


def _run_study_sorters():
    sorter_list = ['tridesclous', 'herdingspikes']
    run_study_sorters(study_folder, sorter_list)

    



def test_aggregate_sorting_comparison():
    comparisons = aggregate_sorting_comparison(study_folder, exhaustive_gt=True)
    for (rec_name, sorter_name), comp in comparisons.items():
        print(comp.print_summary())
    
def test_aggregate_performances_table():
    
    dataframes = aggregate_performances_table(study_folder, exhaustive_gt=True)

    
    for k, df in dataframes.items():
        print('*'*10)
        print(k)
        print(df)
        
        
        


    

if __name__ == '__main__':
    #~ setup_module()
    
    #~ _run_study_sorters()
    
    test_aggregate_sorting_comparison() 
    test_aggregate_performances_table()


