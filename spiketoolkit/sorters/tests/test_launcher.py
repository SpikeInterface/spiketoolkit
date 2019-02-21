import os
import shutil

import pytest
from spiketoolkit.sorters import run_sorters

import spikeextractors as se




def test_run_sorters():
    rec0, _ = se.example_datasets.toy_example1(num_channels=4, duration=30)
    rec1, _ = se.example_datasets.toy_example1(num_channels=8, duration=30)
    
    recording_dict = {'toy_tetrode' : rec0, 'toy_octotrode': rec1}
    
    sorter_list = ['klusta', 'mountainsort4', 'spykingcircus', 'tridesclous']
    
    working_folder = 'test_run_sorters'
    if os.path.exists(working_folder):
        shutil.rmtree(working_folder)
    
    run_sorters(recording_dict, sorter_list, working_folder)
    
    
    
if __name__ == '__main__':
    test_run_sorters()