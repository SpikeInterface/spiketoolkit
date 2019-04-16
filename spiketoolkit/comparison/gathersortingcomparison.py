from pathlib import Path

import pandas as pd


from spiketoolkit.sorters import run_sorters, collect_results
from .sortingcomparison import SortingComparison, compute_performance, _perf_keys

def gather_sorting_comparison(working_folder, ground_truths, use_multi_index=True):
    """
    Loop over output folder in a tree to collect sorting from
    several sorter on several dataset and returns sythetic DataFrame with 
    several metrics (performance, run_time, ...)
    
    Use SortingComparison internally.
    
    
    Parameters
    ----------
    working_folder: str
        The folrder where sorter.run_sorters have done the job.
    ground_truths: dict
        A dict where each key is the recording label and each value 
        the SortingExtractor containing the ground truth.
    use_multi_index: bool (True by default)
        Use (or not) multi index for output dataframe.
        Multiindex is composed from (rec_name, sorter_name).

    Returns
    ----------
    
    out_dataframes: a dict of DataFrame
        Return several usefull DataFrame to compare all results:
          * run_times
          * performances
    """
    
    working_folder = Path(working_folder)
    
    out_dataframes = {}
    
    # get run times:
    run_times = pd.read_csv(working_folder /  'run_time.csv', sep='\t', header=None)
    run_times.columns = ['rec_name', 'sorter_name', 'run_time']
    run_times = run_times.set_index(['rec_name', 'sorter_name',])
    out_dataframes['run_times'] = run_times
    
    
    #~ columns =  ['tp_rate', 'fn_rate']
    performances = pd.DataFrame(index=run_times.index, columns=_perf_keys)
    out_dataframes['performances'] = performances
    
    results = collect_results(working_folder)
    for rec_name, result_one_dataset in results.items():
        #~ print()
        #~ print(rec_name)
        for sorter_name, sorting in result_one_dataset.items():
            #~ print(sorter_name)
            #~ print(sorting)
            
            gt_sorting = ground_truths[rec_name]

            comp = SortingComparison(gt_sorting, sorting, count=True)
            
            perf = compute_performance(comp, verbose=False, output='pandas')
            
            performances.loc[(rec_name, sorter_name), :] = perf

    
    if not use_multi_index:
        for k, df in out_dataframes.items():
            out_dataframes[k] = df.reset_index()
    
    return out_dataframes


