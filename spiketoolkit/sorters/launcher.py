"""
Utils functions to launch several sorter on several recording in parralelle or not.
"""
import os
from pathlib import Path

from .sorterlist import sorter_dict, run_sorter

import multiprocessing


def _run_one(arg_list):
    # the multiprocessing python module force to have one unique tuple argument
    rec_name, recording, sorter_name, output_folder,grouping_property, debug, write_log = arg_list

    try:
        SorterClass = sorter_dict[sorter_name]
        sorter = SorterClass(recording=recording, output_folder=output_folder, grouping_property=grouping_property,
                            parallel=True, debug=debug, delete_output_folder=False)
        params = SorterClass.default_params()
        sorter.set_params(**params)

        run_time = sorter.run()
    except:
        run_time = None

    if write_log and run_time is not None:
        with open(output_folder / 'run_log.txt', mode='w') as f:
            f.write('run_time: {}\n'.format(run_time))


def run_sorters(sorter_list, recording_dict_or_list,  working_folder, grouping_property=None,
                            engine=None, processes=None, debug=False, write_log=True):
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

    """

    assert not os.path.exists(working_folder), 'working_folder already exists, please remove it'

    for sorter_name in sorter_list:
        assert sorter_name in sorter_dict, '{} is not in sorter list'.format(sorter_name)

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
            task_list.append((rec_name, recording, sorter_name, output_folder, grouping_property, debug, write_log))

    if engine is None or engine == 'loop':
        # simple loop in main process
        for arg_list in task_list:
            _run_one(arg_list)

    elif engine == 'multiprocessing':
        # use mp.Pool
        pool = multiprocessing.Pool(processes)
        pool.map(_run_one, task_list)


    if write_log:
        # collect run time and write to cvs
        with open(working_folder / 'run_time.csv', mode='w') as f:
            for task in task_list:
                rec_name = task[0]
                sorter_name = task[2]
                output_folder = task[3]
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

    for rec_name in os.listdir(working_folder):
        if not os.path.isdir(working_folder / rec_name):
            continue
        # print(rec_name)
        results[rec_name] = {}
        for sorter_name in os.listdir(working_folder / rec_name):
            # print('  ', sorter_name)
            output_folder = working_folder / rec_name / sorter_name
            if not os.path.isdir(output_folder):
                continue
            SorterClass = sorter_dict[sorter_name]
            results[rec_name][sorter_name] = SorterClass.get_result_from_folder(output_folder)

    return results
