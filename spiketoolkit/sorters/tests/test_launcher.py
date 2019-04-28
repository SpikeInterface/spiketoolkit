import os
import shutil
import time

import pytest
from spiketoolkit.sorters import run_sorters, collect_results

import spikeextractors as se



def test_run_sorters_with_list():
    rec0, _ = se.example_datasets.toy_example(num_channels=4, duration=30)
    rec1, _ = se.example_datasets.toy_example(num_channels=8, duration=30)
    
    recording_list = [rec0, rec1]
    sorter_list = ['tridesclous']
    working_folder = 'test_run_sorters_list'
    if os.path.exists(working_folder):
        shutil.rmtree(working_folder)
    run_sorters(sorter_list, recording_list, working_folder, debug=False)
    
    

def test_run_sorters_with_dict():
    rec0, _ = se.example_datasets.toy_example(num_channels=4, duration=30)
    rec1, _ = se.example_datasets.toy_example(num_channels=8, duration=30)
    
    recording_dict = {'toy_tetrode' : rec0, 'toy_octotrode': rec1}
    
    # sorter_list = ['mountainsort4', 'klusta', 'tridesclous']
    sorter_list = ['tridesclous',  'klusta',]
    #~ sorter_list = ['tridesclous', ]
    #~ sorter_list = ['tridesclous', 'herdingspikes']
    
    working_folder = 'test_run_sorters_dict'
    if os.path.exists(working_folder):
        shutil.rmtree(working_folder)
    
    
    # simple loop
    t0 = time.perf_counter()
    results = run_sorters(sorter_list, recording_dict, working_folder, engine=None)
    t1 = time.perf_counter()
    print(t1-t0)
    print(results)


def test_run_sorters_multiprocessing():
    
    recording_dict = {}
    for i in range(8):
        rec, _ = se.example_datasets.toy_example(num_channels=8, duration=30)
        recording_dict['rec_'+str(i)] = rec
    
    # sorter_list = ['mountainsort4', 'klusta', 'tridesclous']
    sorter_list = ['tridesclous',  'klusta',]
    #~ sorter_list = ['tridesclous', 'herdingspikes']
    
    working_folder = 'test_run_sorters_mp'
    if os.path.exists(working_folder):
        shutil.rmtree(working_folder)

    # multiprocessing
    t0 = time.perf_counter()
    run_sorters(sorter_list, recording_dict, working_folder, engine='multiprocessing', engine_kargs={'processes':4})
    t1 = time.perf_counter()
    print(t1-t0)


def test_collect_results():
    working_folder = 'test_run_sorters_dict'
    results = collect_results(working_folder)
    print(results)
    
    
if __name__ == '__main__':
    #~ test_run_sorters_with_list()
    
    test_run_sorters_with_dict()
    
    #~ test_run_sorters_multiprocessing()
    
    #~ test_collect_results()
    
