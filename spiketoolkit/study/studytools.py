"""
High level tools to run many groundtruth comparison with
many sorter on many recordings and then collect and aggregate results
in an easy way.

The all mechanism is based on an intrinsinct organisation
into a "study_folder" with several subfolder:
  * raw_files : contain a copy in binary format of recordings
  * sorter_folders : contains output of sorters
  * ground_truth : contains a copy of sorting ground  in npz format
  * sortings: contains light copy of all sorting in npz format
  * tables: some table in cvs format
"""

from pathlib import Path
import shutil
import json
import os

import pandas as pd
import spikeextractors as se

from spikesorters.sorterlist import sorter_dict

from spiketoolkit.sorters import run_sorters, iter_output_folders, iter_sorting_output
from spiketoolkit.comparison.groundtruthcomparison import compare_sorter_to_ground_truth, _perf_keys


def setup_comparison_study(study_folder, gt_dict):
    """
    Based on a dict of (recordnig, sorting) create the study folder.
    

    Parameters
    ----------
    study_folder: str
        The study folder.
    
    gt_dict : a dict of tuple (recording, sorting_gt)
        Dict of tuple that contain recording and sorting ground truth
    """

    study_folder = Path(study_folder)
    assert not os.path.exists(study_folder), 'study_folder already exists'

    os.makedirs(str(study_folder))
    os.makedirs(str(study_folder / 'raw_files'))
    os.makedirs(str(study_folder / 'ground_truth'))
    os.makedirs(str(study_folder / 'sortings'))
    os.makedirs(str(study_folder / 'sortings/run_log' ))
    
    

    for rec_name, (recording, sorting_gt) in gt_dict.items():
        # write recording as binary format + json + prb
        raw_filename = study_folder / 'raw_files' / (rec_name + '.dat')
        prb_filename = study_folder / 'raw_files' / (rec_name + '.prb')
        json_filename = study_folder / 'raw_files' / (rec_name + '.json')
        num_chan = recording.get_num_channels()
        chunksize = 2 ** 24 // num_chan
        sr = recording.get_sampling_frequency()

        se.write_binary_dat_format(recording, raw_filename, time_axis=0, dtype='float32', chunksize=chunksize)
        se.save_probe_file(recording, prb_filename, format='spyking_circus')
        with open(json_filename, 'w', encoding='utf8') as f:
            info = dict(sample_rate=sr, num_chan=num_chan, dtype='float32', frames_first=True)
            json.dump(info, f, indent=4)

        # write recording sorting_gt as with npz format
        se.NpzSortingExtractor.write_sorting(sorting_gt, study_folder / 'ground_truth' / (rec_name + '.npz'))

    # make an index of recording names
    with open(study_folder / 'names.txt', mode='w', encoding='utf8') as f:
        for rec_name in gt_dict:
            f.write(rec_name + '\n')


def get_rec_names(study_folder):
    """
    Get list of keys of recordings.
    Read from the 'names.txt' file in stufy folder.
    
    Parameters
    ----------
    study_folder: str
        The study folder.
    
    Returns
    ----------
    
    rec_names: list
        LIst of names.
    """
    with open(study_folder / 'names.txt', mode='r', encoding='utf8') as f:
        rec_names = f.read()[:-1].split('\n')
    return rec_names


def get_recordings(study_folder):
    """
    Get ground recording as a dict.
    
    They are read from the 'raw_files' folder with binary format.
    
    Parameters
    ----------
    study_folder: str
        The study folder.
    
    Returns
    ----------
    
    recording_dict: dict
        Dict of rexording.
        
    """
    study_folder = Path(study_folder)

    rec_names = get_rec_names(study_folder)
    recording_dict = {}
    for rec_name in rec_names:
        raw_filename = study_folder / 'raw_files' / (rec_name + '.dat')
        prb_filename = study_folder / 'raw_files' / (rec_name + '.prb')
        json_filename = study_folder / 'raw_files' / (rec_name + '.json')
        with open(json_filename, 'r', encoding='utf8') as f:
            info = json.load(f)

        rec = se.BinDatRecordingExtractor(raw_filename, info['sample_rate'], info['num_chan'],
                                          info['dtype'], frames_first=info['frames_first'])
        se.load_probe_file(rec, prb_filename)

        recording_dict[rec_name] = rec

    return recording_dict


def get_ground_truths(study_folder):
    """
    Get ground truth sorting extractor as a dict.
    
    They are read from the 'ground_truth' folder with npz format.
    
    Parameters
    ----------
    study_folder: str
        The study folder.
    
    Returns
    ----------
    
    ground_truths: dict
        Dict of sorintg_gt.
    
    """
    study_folder = Path(study_folder)
    rec_names = get_rec_names(study_folder)
    ground_truths = {}
    for rec_name in rec_names:
        sorting = se.NpzSortingExtractor(study_folder / 'ground_truth' / (rec_name + '.npz'))
        ground_truths[rec_name] = sorting
    return ground_truths


def run_study_sorters(study_folder, sorter_list, sorter_params={}, mode='keep',
                      engine='loop', engine_kargs={}):
    """
    Run all sorter on all recordings.
    
    
    Wrapper on top of st.sorter.run_sorters(...)


    Parameters
    ----------
    study_folder: str
        The study folder.
    
    sorter_params: dict of dict with sorter_name as key
        This allow to overwritte default params for sorter.
    
    mode: 'raise_if_exists' or 'overwrite' or 'keep'
        The mode when the subfolder of recording/sorter already exists.
            * 'raise' : raise error if subfolder exists
            * 'overwrite' : force recompute
            * 'keep' : do not compute again if f=subfolder exists and log is OK

    engine: str
        'loop' or 'multiprocessing'
    
    engine_kargs: dict
        This contains kargs specific to the launcher engine:
            * 'loop' : no kargs
            * 'multiprocessing' : {'processes' : } number of processes
    
    
    """
    study_folder = Path(study_folder)
    sorter_folders = study_folder / 'sorter_folders'

    recording_dict = get_recordings(study_folder)

    run_sorters(sorter_list, recording_dict, sorter_folders, sorter_params=sorter_params,
                grouping_property=None, mode=mode, engine=engine, engine_kargs=engine_kargs,
                with_output=False)

    # results are copied so the heavy sorter_folders can be removed
    copy_sortings_to_npz(study_folder)

def copy_sortings_to_npz(study_folder):
    """
    Collect sorting and copy then in npz format into a separate folder.
    Also copy 
    
    """
    study_folder = Path(study_folder)
    sorter_folders = study_folder / 'sorter_folders'
    sorting_folders = study_folder / 'sortings'

    if not os.path.exists(sorting_folders):
        os.makedirs(str(sorting_folders))

    for rec_name,sorter_name, output_folder in iter_output_folders(sorter_folders):
        SorterClass = sorter_dict[sorter_name]
        sorting = SorterClass.get_result_from_folder(output_folder)
        fname = rec_name+'[#]'+sorter_name
        se.NpzSortingExtractor.write_sorting(sorting, sorting_folders / (fname +'.npz'))
        if os.path.exists(output_folder / 'run_log.txt'):
            shutil.copyfile(output_folder / 'run_log.txt', sorting_folders / 'run_log' / (fname +'.txt'))


def iter_computed_names(study_folder):
    sorting_folder = Path(study_folder) / 'sortings'
    for filename in os.listdir(sorting_folder):
        if filename.endswith('.npz') and '[#]' in filename:
            rec_name, sorter_name = filename.replace('.npz', '').split('[#]')
            yield rec_name, sorter_name

def iter_computed_sorting(study_folder):
    """
    Iter over sorting files.
    """
    sorting_folder = Path(study_folder) / 'sortings'
    for filename in os.listdir(sorting_folder):
        if filename.endswith('.npz') and '[#]' in filename:
            rec_name, sorter_name = filename.replace('.npz', '').split('[#]')
            sorting = se.NpzSortingExtractor(sorting_folder / filename)
            yield rec_name, sorter_name, sorting


def collect_run_times(study_folder):
    """
    Collect run times in a working folder and sotre it in CVS files.

    The output is list of (rec_name, sorter_name, run_time)
    """
    study_folder = Path(study_folder)
    sorting_folders = study_folder / 'sortings'
    log_folder = sorting_folders / 'run_log'
    tables_folder = study_folder / 'tables'

    if not os.path.exists(tables_folder):
        os.makedirs(str(tables_folder))

    run_times = []
    for filename in os.listdir(log_folder):
        if filename.endswith('.txt') and '[#]' in filename:
            rec_name, sorter_name = filename.replace('.txt', '').split('[#]')
            with open(log_folder / filename, mode='r') as logfile:
                run_time = float(logfile.readline().replace('run_time:', ''))
            run_times.append((rec_name, sorter_name, run_time))
        
    run_times = pd.DataFrame(run_times, columns=['rec_name', 'sorter_name', 'run_time'])
    run_times = run_times.set_index(['rec_name', 'sorter_name'])
    
    return run_times
    


def aggregate_sorting_comparison(study_folder, exhaustive_gt=False):
    """
    Loop over output folder in a tree to collect sorting output and run 
    ground_truth_comparison on them.
    
    Parameters
    ----------
    study_folder: str
        The study folder.
    exhaustive_gt: bool (default True)
        Tell if the ground true is "exhaustive" or not. In other world if the
        GT have all possible units. It allows more performance measurment.
        For instance, MEArec simulated dataset have exhaustive_gt=True

    Returns
    ----------
    comparisons: a dict of SortingComparison

    """

    study_folder = Path(study_folder)
    sorter_folders = study_folder / 'sorter_folders'

    ground_truths = get_ground_truths(study_folder)
    results = collect_study_sorting(study_folder)

    comparisons = {}
    for (rec_name, sorter_name), sorting in results.items():
        gt_sorting = ground_truths[rec_name]
        sc = compare_sorter_to_ground_truth(gt_sorting, sorting, exhaustive_gt=exhaustive_gt)
        comparisons[(rec_name, sorter_name)] = sc

    return comparisons


def aggregate_performances_table(study_folder, exhaustive_gt=False, **karg_thresh):
    """
    Aggregate some results into dataframe to have a "study" overview on all recordingXsorter.
    
    Tables are:
      * run_times: run times per recordingXsorter
      * perf_pooled_with_sum: GroundTruthComparison.see get_performance
      * perf_pooled_with_average: GroundTruthComparison.see get_performance
      * count_units: given some threhold count how many units : 'well_detected', 'redundant', 'false_postive_units, 'bad'
    
    Parameters
    ----------
    study_folder: str
        The study folder.

    karg_thresh: dict
        Threholds paramerts used for the "count_units" table.
    
    Returns
    ----------

    dataframes: a dict of DataFrame
        Return several usefull DataFrame to compare all results.
        Note that count_units depend on karg_thresh.
    """
    study_folder = Path(study_folder)
    sorter_folders = study_folder / 'sorter_folders'
    tables_folder = study_folder / 'tables'

    comparisons = aggregate_sorting_comparison(study_folder, exhaustive_gt=exhaustive_gt)
    ground_truths = get_ground_truths(study_folder)
    results = collect_study_sorting(study_folder)

    study_folder = Path(study_folder)

    dataframes = {}

    # get run times:
    run_times = pd.read_csv(str(tables_folder / 'run_times.csv'), sep='\t')
    run_times.columns = ['rec_name', 'sorter_name', 'run_time']
    run_times = run_times.set_index(['rec_name', 'sorter_name', ])
    dataframes['run_times'] = run_times

    perf_pooled_with_sum = pd.DataFrame(index=run_times.index, columns=_perf_keys)
    dataframes['perf_pooled_with_sum'] = perf_pooled_with_sum

    perf_pooled_with_average = pd.DataFrame(index=run_times.index, columns=_perf_keys)
    dataframes['perf_pooled_with_average'] = perf_pooled_with_average

    count_units = pd.DataFrame(index=run_times.index,
                               columns=['num_gt', 'num_sorter', 'num_well_detected', 'num_redundant'])
    dataframes['count_units'] = count_units
    if exhaustive_gt:
        count_units['num_false_positive'] = None
        count_units['num_bad'] = None

    perf_by_spiketrain = []

    for (rec_name, sorter_name), comp in comparisons.items():
        gt_sorting = ground_truths[rec_name]
        sorting = results[(rec_name, sorter_name)]

        perf = comp.get_performance(method='pooled_with_sum', output='pandas')
        perf_pooled_with_sum.loc[(rec_name, sorter_name), :] = perf

        perf = comp.get_performance(method='pooled_with_average', output='pandas')
        perf_pooled_with_average.loc[(rec_name, sorter_name), :] = perf

        perf = comp.get_performance(method='by_spiketrain', output='pandas')
        perf['rec_name'] = rec_name
        perf['sorter_name'] = sorter_name
        perf = perf.reset_index()

        perf_by_spiketrain.append(perf)

        count_units.loc[(rec_name, sorter_name), 'num_gt'] = len(gt_sorting.get_unit_ids())
        count_units.loc[(rec_name, sorter_name), 'num_sorter'] = len(sorting.get_unit_ids())
        count_units.loc[(rec_name, sorter_name), 'num_well_detected'] = comp.count_well_detected_units(**karg_thresh)
        count_units.loc[(rec_name, sorter_name), 'num_redundant'] = comp.count_redundant_units()
        if exhaustive_gt:
            count_units.loc[(rec_name, sorter_name), 'num_false_positive'] = comp.count_false_positive_units()
            count_units.loc[(rec_name, sorter_name), 'num_bad'] = comp.count_bad_units()

    perf_by_spiketrain = pd.concat(perf_by_spiketrain)
    perf_by_spiketrain = perf_by_spiketrain.set_index(['rec_name', 'sorter_name', 'gt_unit_id'])
    dataframes['perf_by_spiketrain'] = perf_by_spiketrain

    return dataframes
