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
import os
import json

import pandas as pd

import spikeextractors as se

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
    
    
    for rec_name, (recording, sorting_gt) in gt_dict.items():
        
        # write recording as binary format + json + prb
        raw_filename = study_folder / 'raw_files' / (rec_name+'.dat')
        prb_filename = study_folder / 'raw_files' / (rec_name+'.prb')
        json_filename = study_folder / 'raw_files' / (rec_name+'.json')
        num_chan = recording.get_num_channels()
        chunksize = 2**24// num_chan
        sr = recording.get_sampling_frequency()
        
        se.write_binary_dat_format(recording, raw_filename, time_axis=0, dtype='float32', chunksize=chunksize)
        se.save_probe_file(recording, prb_filename, format='spyking_circus')
        with open(json_filename, 'w', encoding='utf8') as f:
            info = dict(sample_rate=sr, num_chan=num_chan, dtype='float32', frames_first=True)
            json.dump(info, f, indent=4)
        
        # write recording sorting_gt as with npz format
        se.NpzSortingExtractor.write_sorting(sorting_gt, study_folder / 'ground_truth' / (rec_name+'.npz'))
    
    # make an index of recording names
    with open(study_folder / 'names.txt', mode='w', encoding='utf8') as f:
        for rec_name in  gt_dict:
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
        raw_filename = study_folder / 'raw_files' / (rec_name+'.dat')
        prb_filename = study_folder / 'raw_files' / (rec_name+'.prb')
        json_filename = study_folder / 'raw_files' / (rec_name+'.json')
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
        sorting = se.NpzSortingExtractor(study_folder / 'ground_truth' / (rec_name+'.npz'))
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
    
    run_sorters(sorter_list, recording_dict,  sorter_folders, sorter_params=sorter_params,
                    grouping_property=None, mode=mode, engine=engine, engine_kargs=engine_kargs,
                    with_output=False)
    
    # results are copied so the heavy sorter_folders can be removed
    copy_sortings_to_npz(study_folder)
    collect_run_times(study_folder)


def copy_sortings_to_npz(study_folder):
    """
    Collect sorting and copy then in the same format to get a lightweigth version.
    """
    study_folder = Path(study_folder)
    sorter_folders = study_folder / 'sorter_folders'
    sorting_folders = study_folder / 'sortings'
    
    if not os.path.exists(sorting_folders):
        os.makedirs(str(sorting_folders))
    
    for rec_name,sorter_name, sorting in    iter_sorting_output(sorter_folders):
        se.NpzSortingExtractor.write_sorting(sorting, sorting_folders / (rec_name+'[#]'+sorter_name+'.npz'))


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
    sorter_folders = study_folder / 'sorter_folders'
    tables_folder = study_folder / 'tables'

    if not os.path.exists(tables_folder):
        os.makedirs(str(tables_folder))
    
    run_times = []
    for rec_name, sorter_name, output_folder in iter_output_folders(sorter_folders):
        if os.path.exists(output_folder / 'run_log.txt'):
            with open(output_folder / 'run_log.txt', mode='r') as logfile:
                run_time = float(logfile.readline().replace('run_time:', ''))
            run_times.append((rec_name, sorter_name, run_time))

    run_times = pd.DataFrame(run_times, columns=['rec_name', 'sorter_name', 'run_time'])
    run_times.to_csv(str(tables_folder / 'run_times.csv'), sep='\t', index=False)
    

    
    
    