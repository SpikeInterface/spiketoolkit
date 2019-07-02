from pathlib import Path
import os

import numpy as np
import pandas as pd

import spikeextractors as se

from spiketoolkit.comparison.groundtruthcomparison import compare_sorter_to_ground_truth, _perf_keys

from .studytools import (setup_comparison_study, run_study_sorters, get_rec_names, 
                                        get_recordings, copy_sortings_to_npz, iter_computed_names,
                                        iter_computed_sorting, collect_run_times)





class GroundTruthStudy:
    def __init__(self, study_folder=None):
        self.study_folder = Path(study_folder)
        
        self.scan_folder()
        self.computed_names = []
        
        self.comparisons = None
        self.exhaustive_gt = None
    
    def __repr__(self):
        t = 'Groud truth study\n'
        t += '  ' + str(self.study_folder)+ '\n'
        t += '  recordings: {} {}\n'.format(len(self.rec_names), self.rec_names)
        if len(self.sorter_names):
            t += '  sorters: {} {}\n'.format(len(self.sorter_names), self.sorter_names)
        
        return t
    
    def scan_folder(self):
        self.rec_names = get_rec_names(self.study_folder)
        
        # scan computed names
        self.computed_names = list(iter_computed_names(self.study_folder))  # list of pair (rec_name,sorter_name)
        self.sorter_names = np.unique([e for _, e in iter_computed_names(self.study_folder)]).tolist()

    @classmethod
    def setup(cls, study_folder, gt_dict):
        setup_comparison_study(study_folder, gt_dict)
        return cls(study_folder)
    
    def run_sorters(self, sorter_list, sorter_params={}, mode='keep',
                                        engine='loop', engine_kargs={}):
        run_study_sorters(self.study_folder, sorter_list, sorter_params=sorter_params,
                                    engine=engine, engine_kargs=engine_kargs)

    def get_ground_truth(self, rec_name):
        sorting = se.NpzSortingExtractor(self.study_folder / 'ground_truth' / (rec_name+'.npz'))
        return sorting
    
    def copy_sortings(self):
        copy_sortings_to_npz(self.study_folder)
        self.scan_folder()
        
    
    def run_comparisons(self, exhaustive_gt=False):
        self.comparisons = {}
        for rec_name,sorter_name, sorting in iter_computed_sorting(self.study_folder):
            gt_sorting = self.get_ground_truth(rec_name)
            sc = compare_sorter_to_ground_truth(gt_sorting, sorting, exhaustive_gt=exhaustive_gt)
            self.comparisons[(rec_name, sorter_name)] = sc
        self.exhaustive_gt = exhaustive_gt
    
    def aggregate_run_times(self):
        return collect_run_times(self.study_folder)
        
    
    def aggregate_performance_by_units(self):
        assert self.comparisons is not None, 'run_comparisons first'
        
        perf_by_units = []
        for rec_name,sorter_name, sorting in iter_computed_sorting(self.study_folder):
            comp = self.comparisons[(rec_name, sorter_name)]
            
            perf = comp.get_performance(method='by_unit', output='pandas')
            perf['rec_name'] = rec_name
            perf['sorter_name'] = sorter_name
            perf = perf.reset_index()
            perf_by_units.append(perf)
            
        perf_by_units = pd.concat(perf_by_units)
        perf_by_units = perf_by_units.set_index(['rec_name', 'sorter_name', 'gt_unit_id'])
        
        return perf_by_units
    
    def aggregate_count_units(self, **karg_thresh):
        assert self.comparisons is not None, 'run_comparisons first'
        
        index = pd.MultiIndex.from_tuples(self.computed_names, names=['rec_name', 'sorter_name'])
        
        count_units = pd.DataFrame(index=index, columns=['num_gt', 'num_sorter', 'num_well_detected', 'num_redundant'])
        
        if self.exhaustive_gt:
            count_units['num_false_positive'] = None
            count_units['num_bad'] = None

        for rec_name,sorter_name, sorting in iter_computed_sorting(self.study_folder):
            gt_sorting = self.get_ground_truth(rec_name)
            comp = self.comparisons[(rec_name, sorter_name)]

            count_units.loc[(rec_name, sorter_name), 'num_gt'] = len(gt_sorting.get_unit_ids())
            count_units.loc[(rec_name, sorter_name), 'num_sorter'] = len(sorting.get_unit_ids())
            count_units.loc[(rec_name, sorter_name), 'num_well_detected'] = comp.count_well_detected_units(**karg_thresh)
            count_units.loc[(rec_name, sorter_name), 'num_redundant'] = comp.count_redundant_units()
            if self.exhaustive_gt:
                count_units.loc[(rec_name, sorter_name), 'num_false_positive'] = comp.count_false_positive_units()
                count_units.loc[(rec_name, sorter_name), 'num_bad'] = comp.count_bad_units()
        
        return count_units
        
    def aggregate_dataframes(self, copy_into_folder=True, **karg_thresh):
        dataframes = {}
        dataframes['run_times'] = self.aggregate_run_times().reset_index()
        perfs = self.aggregate_performance_by_units()
        dataframes['perf_by_units'] = perfs.reset_index()
        dataframes['perf_pooled_with_average'] = perfs.groupby(['rec_name', 'sorter_name']).mean().reset_index()
        dataframes['count_units'] = self.aggregate_count_units(**karg_thresh).reset_index()
        
        if copy_into_folder:
            tables_folder = self.study_folder / 'tables'
            if not os.path.exists(tables_folder):
                os.makedirs(str(tables_folder))
            
            for name, df in dataframes.items():
                df.to_csv(str(tables_folder / (name + '.csv')), sep='\t', index=False)
        
        return dataframes
    
    
    
