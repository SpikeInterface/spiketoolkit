"""
Utils functions to launch several sorter on several recording in parralelle or not.
"""
import os
from pathlib import Path
import multiprocessing

import spikeextractors as se

from .sorterlist import sorter_dict, run_sorter




def _run_one(arg_list):
    # the multiprocessing python module force to have one unique tuple argument
    rec_name, recording, sorter_name, output_folder,grouping_property, debug, write_log = arg_list

    #~ try:
    if True:
        SorterClass = sorter_dict[sorter_name]
        sorter = SorterClass(recording=recording, output_folder=output_folder, grouping_property=grouping_property,
                             parallel=True, debug=debug, delete_output_folder=False)
        params = SorterClass.default_params()
        sorter.set_params(**params)

        run_time = sorter.run()
    #~ except:
        #~ run_time = None

    if write_log and run_time is not None:
        with open(output_folder / 'run_log.txt', mode='w') as f:
            f.write('run_time: {}\n'.format(run_time))


def run_sorters(sorter_list, recording_dict_or_list,  working_folder, grouping_property=None,
                            shared_binary_copy=False, engine=None, engine_kargs={}, debug=False, write_log=True):
    """
    This run several sorter on several recording.
    Simple implementation will nested loops.

    Need to be done with multiprocessing.

    sorter_list: list of str (sorter names)
    recording_dict_or_list: a dict (or a list) of recording
    working_folder : str

    engine = None ( = 'loop') or 'multiprocessing'
    processes = only if 'multiprocessing' if None then processes=os.cpu_count()
    debug=True/False to control sorter verbosity


    Note: engine='multiprocessing' use the python multiprocessing module.
    This do not allow to have subprocess in subprocess.
    So sorter that already use internally multiprocessing, this will fail.

    Parameters
    ----------
    
    sorter_list: list of str
        List of sorter name.
    
    recording_dict_or_list: dict or list
        A dict of recording. The key will be the name of the recording.
        In a list is given then the name will be recording_0, recording_1, ...
    
    working_folder: str
        The working directory.
        This must not exists before calling this function.
    
    grouping_property: str
        The property of grouping given to sorters.
    
    shared_binary_copy: False default
        Before running each sorter, all recording are copied inside 
        the working_folder with the raw binary format (BinDatRecordingExtractor)
        and new recording are instantiated as BinDatRecordingExtractor.
        This avoids multiple copy inside each sorter of the same file but
        imply a global of all files.

    engine: str
        'loop' or 'multiprocessing'
    
    engine_kargs: dict
        This contains kargs specific to the launcher engine:
            * 'loop' : no kargs
            * 'multiprocessing' : {'processes' : } number of processes
    
    debug: bool
        default True
    
    write_log: bool
        default True
    
    Output
    ----------
    
    results : dict
        The output is nested dict[rec_name][sorter_name] of SortingExtrator.



    """

    assert not os.path.exists(working_folder), 'working_folder already exists, please remove it'
    working_folder = Path(working_folder)
    
    for sorter_name in sorter_list:
        assert sorter_name in sorter_dict, '{} is not in sorter list'.format(sorter_name)

    if isinstance(recording_dict_or_list, list):
        # in case of list
        recording_dict = { 'recording_{}'.format(i): rec for i, rec in enumerate(recording_dict_or_list) }
    elif isinstance(recording_dict_or_list, dict):
        recording_dict = recording_dict_or_list
    else:
        raise(ValueError('bad recording dict'))

    if shared_binary_copy:
        os.makedirs(working_folder / 'raw_files')
        old_rec_dict = dict(recording_dict)
        recording_dict = {}
        for rec_name, recording in old_rec_dict.items():
            if grouping_property is not None:
                recording_list = se.get_sub_extractors_by_property(recording, grouping_property)
                n_group = len(recording_list)
                assert n_group == 1, 'shared_binary_copy work only when one group'
                recording = recording_list[0]
                grouping_property = None
            
            raw_filename = working_folder / 'raw_files' / (rec_name+'.raw')
            prb_filename = working_folder / 'raw_files' / (rec_name+'.prb')
            n_chan = recording.get_num_channels()
            chunksize = 2**24// n_chan
            sr = recording.get_sampling_frequency()
            
            # save binary
            se.write_binary_dat_format(recording, raw_filename, time_axis=0, dtype='float32', chunksize=chunksize)
            # save location (with PRB format)
            se.save_probe_file(recording, prb_filename, format='spyking_circus')
            
            # make new  recording
            new_rec = se.BinDatRecordingExtractor(raw_filename, sr, n_chan, 'float32', frames_first=True)
            se.load_probe_file(new_rec, prb_filename)
            recording_dict[rec_name] = new_rec

    task_list = []
    for rec_name, recording in recording_dict.items():
        for sorter_name in sorter_list:
            output_folder = working_folder / 'output_folders' / rec_name / sorter_name
            task_list.append((rec_name, recording, sorter_name, output_folder, grouping_property, debug, write_log))

    if engine is None or engine == 'loop':
        # simple loop in main process
        for arg_list in task_list:
            # print(arg_list)
            _run_one(arg_list)

    elif engine == 'multiprocessing':
        # use mp.Pool
        processes = engine_kargs.get('processes', None)
        pool = multiprocessing.Pool(processes)
        pool.map(_run_one, task_list)


    if write_log:
        # collect run time and write to cvs
        with open(working_folder / 'run_time.csv', mode='w') as f:
            for task in task_list:
                rec_name = task[0]
                sorter_name = task[2]
                output_folder = task[3]
                if os.path.exists(output_folder / 'run_log.txt'):
                    with open(output_folder / 'run_log.txt', mode='r') as logfile:
                        run_time = float(logfile.readline().replace('run_time:', ''))

                    txt = '{}\t{}\t{}\n'.format(rec_name, sorter_name,run_time)
                    f.write(txt)

    results = collect_results(working_folder)
    return results


def collect_results(working_folder):
    """
    Collect results in a working_folder.

    The output is nested dict[rec_name][sorter_name] of SortingExtrator.

    """
    results = {} 
    working_folder = Path(working_folder)
    output_folders = working_folder/'output_folders'

    for rec_name in os.listdir(output_folders):
        if not os.path.isdir(output_folders / rec_name):
            continue
        # print(rec_name)
        results[rec_name] = {}
        for sorter_name in os.listdir(output_folders / rec_name):
            # print('  ', sorter_name)
            output_folder = output_folders / rec_name / sorter_name
            #~ print(output_folder)
            if not os.path.isdir(output_folder):
                continue
            SorterClass = sorter_dict[sorter_name]
            results[rec_name][sorter_name] = SorterClass.get_result_from_folder(output_folder)

    return results
