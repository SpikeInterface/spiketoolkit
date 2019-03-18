from pathlib import Path

import pandas as pd


from spiketoolkit.sorters import run_sorters, collect_results
from .sortingcomparison import SortingComparison

def gather_sorting_comparison(working_folder, ground_truths, use_multi_index=True):
    """
    Loop over output folder in a tree to collect sorting from
    several sorter on several dataset.
    
    Compare then to ground_truth given as dict.
    
    return dict containing several pandas DataFrame from several metrics.
    
    Dataframes for results have a MultiIndex apporach (rec_name, sorter_name).
    this can be convienent in some situation and anoying in others.
    
    
    
    
    """
    
    working_folder = Path(working_folder)
    
    out_dataframes = {}
    
    # get run times:
    run_times = pd.read_csv(working_folder /  'run_time.csv', sep='\t', header=None)
    run_times.columns = ['rec_name', 'sorter_name', 'run_time']
    run_times = run_times.set_index(['rec_name', 'sorter_name',])
    out_dataframes['run_times'] = run_times
    
    
    columns =  ['fn_rate', 'tp_rate']
    performances = pd.DataFrame(index=run_times.index, columns=columns)
    out_dataframes['performances'] = performances
    
    results = collect_results(working_folder)
    for rec_name, result_one_dataset in results.items():
        #~ print()
        #~ print(rec_name)
        for sorter_name, sorting in result_one_dataset.items():
            #~ print(sorter_name)
            #~ print(sorting)
            
            gt_sorting = ground_truths[rec_name]

            comp = st.comparison.SortingComparison(gt_sorting, sorting, count=True)

            counts = comp.counts
            performance.loc[(rec_name, sorter_name), 'tp_rate'] = float(counts['TP']) / counts['TOT_ST1'] * 100
            performance.loc[(rec_name, sorter_name), 'fn_rate'] = float(counts['FN']) / counts['TOT_ST1'] * 100

    
    if not use_multi_index:
        for k, df in out_dataframes.items():
            out_dataframes[k] = df.reset_index()
    
    return out_dataframes


