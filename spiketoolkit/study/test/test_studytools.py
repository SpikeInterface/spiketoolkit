
import os
import shutil
import time
import pickle

import pytest
#~ from spiketoolkit.sorters import run_sorters, collect_results
from spiketoolkit.study import (setup_comparison_study, run_study_sorters)
from spiketoolkit.study.studytools import iter_computed_names, iter_computed_sorting

import spikeextractors as se




study_folder = 'test_TDC_vs_HS2/'



def setup_module():
    if os.path.exists(study_folder):
        shutil.rmtree(study_folder)

    _setup_comparison_study()

    _run_study_sorters()


def _setup_comparison_study():
    rec0, gt_sorting0 = se.example_datasets.toy_example(num_channels=4, duration=30, seed=0)
    rec1, gt_sorting1 = se.example_datasets.toy_example(num_channels=32, duration=30, seed=0)

    gt_dict = {
        'toy_tetrode' : (rec0, gt_sorting0),
        'toy_probe32' : (rec1, gt_sorting1),
    }

    setup_comparison_study(study_folder, gt_dict)


def _run_study_sorters():
    sorter_list = ['tridesclous', 'herdingspikes']
    run_study_sorters(study_folder, sorter_list)


def test_loops():
    names = list(iter_computed_names(study_folder))
    #~ print(names)
    for rec_name, sorter_name, sorting in iter_computed_sorting(study_folder):
        print(rec_name, sorter_name)
        print(sorting)
    
    
if __name__ == '__main__':
    #~ setup_module()
    
    test_loops()

