"""
Utils functions to launch several sorter on several recording in parralelle or not.
"""
import os
from pathlib import Path

from .sorterlist import sorter_dict, run_sorter

import multiprocessing


def _run_one(arg_list):
    # the multiprocessing python module force to have one unique tuple argument
    rec_name, recording, sorter_name, output_folder, debug = arg_list
    
    os.makedirs(output_folder)
    params = sorter_dict[sorter_name].default_params()
    run_sorter(sorter_name, recording, output_folder=output_folder, debug=debug, **params)
    
    

def run_sorters(sorter_list, recording_dict_or_list,  working_folder, engine=None, debug=False):
    """
    This run several sorter on several recording.
    Simple implementation will nested loops.
    
    Need to be done with multiprocessing.
    
    sorter_list: list of str (sorter names)
    recording_dict_or_list: a dict (or a list) of recording
    working_folder : str
    
    """
    
    assert not os.path.exists(working_folder), 'working_folder already exists, please remove it'
    
    if isinstance(recording_dict_or_list, list):
        # in case of list
        recording_dict = { 'recording_{}'.format(i): rec for i, rec in enumerate(recording_dict_or_list) }
    elif isinstance(recording_dict_or_list, dict):
        recording_dict = recording_dict_or_list
    else:
        raise(ValueError('bad recording dict'))
    
    
    working_folder = Path(working_folder)
    
    task_list = []
    for rec_name, recording in recording_dict.items():
        for sorter_name in sorter_list:
            output_folder = working_folder / rec_name / sorter_name
            task_list.append((rec_name, recording, sorter_name, output_folder, debug))
            
            

    
    if engine is None:
        # simple loop
        for arg_list in task_list:
            _run_one(arg_list)
    
    elif engine == 'multiprocessing':
        pool = multiprocessing.Pool()
        pool.map(_run_one, task_list)
        
    
    