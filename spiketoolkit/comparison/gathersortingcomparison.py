from pathlib import Path

import pandas as pd


from spiketoolkit.sorters import run_sorters, collect_results
from .groundtruthcomparison import compare_sorter_to_ground_truth, _perf_keys

def gather_sorting_comparison(working_folder, ground_truths, use_multi_index=True, exhaustive_gt=False, **karg_thresh):
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
    exhaustive_gt: bool (default True)
        Tell if the ground true is "exhaustive" or not. In other world if the
        GT have all possible units. It allows more performance measurment.
        For instance, MEArec simulated dataset have exhaustive_gt=True
    **karg_thresh: 
        Extra thresh kkargs are passed to 
        GroundTruthComparison.get_well_detected_units for
        See doc there.

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
    
    count_units = pd.DataFrame(index=run_times.index, columns=['num_gt', 'num_sorter', 'num_well_detected', 'num_redundant'])
    out_dataframes['count_units'] = count_units
    if exhaustive_gt:
        count_units['num_false_positive'] = None
        count_units['num_bad'] = None

    results = collect_results(working_folder)
    for rec_name, result_one_dataset in results.items():
        for sorter_name, sorting in result_one_dataset.items():
            gt_sorting = ground_truths[rec_name]

            sc = compare_sorter_to_ground_truth(gt_sorting, sorting)

            comparisons[(rec_name, sorter_name)] = sc

            perf = sc.get_performance(method='pooled_with_sum', output='pandas')
            perf_pooled_with_sum.loc[(rec_name, sorter_name), :] = perf

            perf = sc.get_performance(method='pooled_with_average', output='pandas')
            perf_pooled_with_average.loc[(rec_name, sorter_name), :] = perf
            
            count_units.loc[(rec_name, sorter_name), 'num_gt'] = len(gt_sorting.get_unit_ids())
            count_units.loc[(rec_name, sorter_name), 'num_sorter'] = len(sorting.get_unit_ids())
            count_units.loc[(rec_name, sorter_name), 'num_well_detected'] = sc.count_well_detected_units(**karg_thresh)
            count_units.loc[(rec_name, sorter_name), 'num_redundant'] = sc.count_redundant_units()
            if exhaustive_gt:
                count_units.loc[(rec_name, sorter_name), 'num_false_positive'] = sc.count_false_positive_units()
                count_units.loc[(rec_name, sorter_name), 'num_bad'] = sc.count_bad_units()

    if not use_multi_index:
        for k, df in out_dataframes.items():
            out_dataframes[k] = df.reset_index()

    return comparisons, out_dataframes
