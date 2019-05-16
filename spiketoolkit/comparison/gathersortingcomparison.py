from pathlib import Path

import pandas as pd


from spiketoolkit.sorters import run_sorters, collect_results
from .sortingcomparison import compare_two_sorters, _perf_keys

def gather_sorting_comparison(working_folder, ground_truths, use_multi_index=True, threshold=95):
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
    comparisons: a dict of SortingComparison

    out_dataframes: a dict of DataFrame
        Return several usefull DataFrame to compare all results:
          * run_times
          * performances
    """

    working_folder = Path(working_folder)

    comparisons = {}
    out_dataframes = {}


    # get run times:
    run_times = pd.read_csv(working_folder /  'run_time.csv', sep='\t', header=None)
    run_times.columns = ['rec_name', 'sorter_name', 'run_time']
    run_times = run_times.set_index(['rec_name', 'sorter_name',])
    out_dataframes['run_times'] = run_times

    perf_pooled_with_sum = pd.DataFrame(index=run_times.index, columns=_perf_keys)
    out_dataframes['perf_pooled_with_sum'] = perf_pooled_with_sum

    perf_pooled_with_average = pd.DataFrame(index=run_times.index, columns=_perf_keys)
    out_dataframes['perf_pooled_with_average'] = perf_pooled_with_average
    
    above_keys = ['tp_rate', 'accuracy',	'sensitivity']
    above_columns = [ 'nb_above(with_{})'.format(k) for k in above_keys]
    nb_units_above_threshold = pd.DataFrame(index=run_times.index, columns=above_columns)
    out_dataframes['nb_units_above_threshold'] = nb_units_above_threshold
    

    results = collect_results(working_folder)
    for rec_name, result_one_dataset in results.items():
        for sorter_name, sorting in result_one_dataset.items():
            gt_sorting = ground_truths[rec_name]

            sorting_comp = compare_two_sorters(gt_sorting, sorting, count=True)

            comparisons[(rec_name, sorter_name)] = sorting_comp

            perf = sorting_comp.get_performance(method='pooled_with_sum', output='pandas')
            perf_pooled_with_sum.loc[(rec_name, sorter_name), :] = perf

            perf = sorting_comp.get_performance(method='pooled_with_average', output='pandas')
            perf_pooled_with_average.loc[(rec_name, sorter_name), :] = perf
            
            for k, col in zip(above_keys, above_columns):
                nb_units_above_threshold.loc[(rec_name, sorter_name), col] = sorting_comp.get_number_units_above_threshold(columns=k, threshold=threshold)


    if not use_multi_index:
        for k, df in out_dataframes.items():
            out_dataframes[k] = df.reset_index()

    return comparisons, out_dataframes
