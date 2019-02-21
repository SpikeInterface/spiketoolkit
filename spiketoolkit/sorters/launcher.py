"""
Utils functions to launch several sorter on several recording in parralelle or not.
"""
import os
from pathlib import Path

from .sorterlist import sorter_dict, run_sorter



def run_sorters(recording_dict, sorter_list, working_folder, debug=False):
    """
    This run several sorter on several recording.
    Simple implementation will nested loops.
    
    Need to be done with multiprocessing.
    """
    
    assert not os.path.exists(working_folder), 'working_folder already exists, please remove it'
    
    working_folder = Path(working_folder)
    
    for rec_name, recording in recording_dict.items():
        for sorter_name in sorter_list:
            output_folder = working_folder / rec_name / sorter_name
            os.makedirs(output_folder)
            params = sorter_dict[sorter_name].default_params()
            run_sorter(sorter_name, recording, output_folder=output_folder, debug=debug, **params)
        
    