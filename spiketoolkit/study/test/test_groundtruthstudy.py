import os
import shutil
import time
import pickle

import pytest

import spikeextractors as se

from spiketoolkit.study import GroundTruthStudy






study_folder = 'test_groundtruthstudy/'



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
    
    study = GroundTruthStudy.setup(study_folder, gt_dict)



def _run_study_sorters():
    study = GroundTruthStudy(study_folder)
    sorter_list = ['tridesclous', 'herdingspikes']
    study.run_sorters(sorter_list)


def test_extract_sortings():
    study = GroundTruthStudy(study_folder)
    print(study)
    
    for rec_name in study.rec_names:
        gt_sorting = study.get_gt(rec_name)
        #~ print(rec_name, gt_sorting)
    
    study.copy_sortings()
    study.run_comparisons()
    
    
    perf = study.aggregate_performance_by_units()
    #~ print(perf)
    count_units = study.aggregate_count_units()
    #~ print(count_units)
    
    dataframes = study.aggregate_dataframes()
    #~ for name, df in dataframes.items():
        #~ print(name)
        #~ print(df)
    
    
    


if __name__ == '__main__':
    #~ setup_module()
    test_extract_sortings()

