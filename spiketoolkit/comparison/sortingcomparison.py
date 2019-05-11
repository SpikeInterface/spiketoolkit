import numpy as np
import spikeextractors as se
from scipy.optimize import linear_sum_assignment

import pandas as pd

from .comparisontools import (count_matching_events, compute_agreement_score,
                                                do_matching, do_score_labels, do_counting, do_confusion_matrix)


class SortingComparison():
    def __init__(self, sorting1, sorting2, sorting1_name=None, sorting2_name=None, delta_frames=10, min_accuracy=0.5,
                 count=False, n_jobs=1, verbose=False):
        self._sorting1 = sorting1
        self._sorting2 = sorting2
        self.sorting1_name = sorting1_name
        self.sorting2_name = sorting2_name
        self._delta_frames = delta_frames
        self._min_accuracy = min_accuracy
        self._n_jobs = n_jobs
        if verbose:
            print("Matching...")
        self._do_matching()

        self._mixed_counts = None
        if count:
            if verbose:
                print("Counting...")
            self._do_counting(verbose=verbose)

    def get_sorting1(self):
        # Samuel EDIT : why not a direct attribute acees  with self.sorting1 ?
        return self._sorting1

    def get_sorting2(self):
        # Samuel EDIT : why not a direct attribute acees  with self.sorting2 ?
        return self._sorting2

    def get_labels1(self, unit_id):
        if unit_id in self._sorting1.get_unit_ids():
            return self._labels_st1[unit_id]
        else:
            raise Exception("Unit_id is not a valid unit")

    def get_labels2(self, unit_id):
        if unit_id in self._sorting1.get_unit_ids():
            return self._labels_st1[unit_id]
        else:
            raise Exception("Unit_id is not a valid unit")

    def get_mapped_sorting1(self):
        """
        Returns a MappedSortingExtractor for sorting 1.

        The returned MappedSortingExtractor.get_unit_ids returns the unit_ids of sorting 1.

        The returned MappedSortingExtractor.get_mapped_unit_ids returns the mapped unit_ids
        of sorting 2 to the units of sorting 1 (if units are not mapped they are labeled as -1).

        The returned MappedSortingExtractor.get_unit_spikeTrains returns the the spike trains
        of sorting 2 mapped to the unit_ids of sorting 1.
        """
        return MappedSortingExtractor(self._sorting2, self._unit_map12)

    def get_mapped_sorting2(self):
        """
        Returns a MappedSortingExtractor for sorting 2.

        The returned MappedSortingExtractor.get_unit_ids returns the unit_ids of sorting 2.

        The returned MappedSortingExtractor.get_mapped_unit_ids returns the mapped unit_ids
        of sorting 1 to the units of sorting 2 (if units are not mapped they are labeled as -1).

        The returned MappedSortingExtractor.get_unit_spikeTrains returns the the spike trains
        of sorting 1 mapped to the unit_ids of sorting 2.
        """
        return MappedSortingExtractor(self._sorting1, self._unit_map21)

    def get_matching_event_count(self, unit1, unit2):
        if (unit1 is not None) and (unit2 is not None):
            if unit1 != -1:
                a = self._matching_event_counts_12[unit1]
                if unit2 in a:
                    return a[unit2]
                else:
                    return 0
            else:
                return 0
        else:
            raise Exception('get_matching_event_count: unit1 and unit2 must not be None.')

    def _compute_safe_frac(self, numer, denom):
        if denom == 0:
            return 0
        return float(numer) / denom

    def get_best_unit_match1(self, unit1):
        if unit1 in self._best_match_units_12:
            return self._best_match_units_12[unit1]
        else:
            return None

    def get_best_unit_match2(self, unit2):
        if unit2 in self._best_match_units_21:
            return self._best_match_units_21[unit2]
        else:
            return None

    def get_matching_unit_list1(self, unit1):
        a = self._matching_event_counts_12[unit1]
        return list(a.keys())

    def get_matching_unit_list2(self, unit2):
        a = self._matching_event_counts_21[unit2]
        return list(a.keys())

    def get_agreement_fraction(self, unit1=None, unit2=None):
        # Samuel NOTE: I guess that this function is no more necessary
        # please confirm this
        if (unit1 is not None) and (unit2 is None):
            if unit1 != -1:
                unit2 = self.get_best_unit_match1(unit1)
                if unit2 is None or unit2 == -1:
                    return 0
            else:
                return 0
        if (unit1 is None) and (unit2 is not None):
            if unit1 != -1 and unit2 != -1:
                unit1 = self.get_best_unit_match2(unit2)
                if unit1 is None or unit1 == -1:
                    return 0
            else:
                return 0
        if (unit1 is None) and (unit2 is None):
            raise Exception('get_agreement_fraction: at least one of unit1 and unit2 must not be None.')

        if unit1 != -1 and unit2 != -1:
            a = self._matching_event_counts_12[unit1]
            if unit2 not in a:
                return 0
        else:
            return 0
        return compute_agreement_score(a[unit2], self._event_counts_1[unit1], self._event_counts_2[unit2])

    def get_false_positive_fraction(self, unit1, unit2=None):
        if unit1 is None:
            raise Exception('get_false_positive_fraction: unit1 must not be None')
        if unit2 is None:
            unit2 = self.get_best_unit_match1(unit1)
            if unit2 is None or unit2 == -1:
                return 0

        if unit1 != -1 and unit2 != -1:
            a = self._matching_event_counts_12[unit1]
            if unit2 not in a:
                return 0
        else:
            return 0
        return 1 - self._compute_safe_frac(a[unit2], self._event_counts_2[unit2])

    def get_false_negative_fraction(self, unit1, unit2=None):
        if unit1 is None:
            raise Exception('get_false_positive_fraction: unit1 must not be None')
        if unit2 is None:
            unit2 = self.get_best_unit_match1(unit1)
            if unit2 is None:
                return 0

        if unit1 != -1 and unit2 != -1:
            a = self._matching_event_counts_12[unit1]
            if unit2 not in a:
                return 0
        else:
            return 0
        return 1 - self._compute_safe_frac(a[unit2], self._event_counts_1[unit1])

    def compute_counts(self):
        if self._mixed_counts is None:
            self._do_counting(verbose=False)

    def plot_confusion_matrix(self, xlabel=None, ylabel=None):
        # Samuel EDIT
        # This must be moved in spikewidget
        import matplotlib.pylab as plt

        if self._mixed_counts is None:
            self._do_counting(verbose=False)

        sorting1 = self._sorting1
        sorting2 = self._sorting2
        unit1_ids = sorting1.get_unit_ids()
        unit2_ids = sorting2.get_unit_ids()
        N1 = len(unit1_ids)
        N2 = len(unit2_ids)
        st1_idxs, st2_idxs = self._do_confusion()
        fig, ax = plt.subplots()
        # Using matshow here just because it sets the ticks up nicely. imshow is faster.
        ax.matshow(self._confusion_matrix, cmap='Greens')

        for (i, j), z in np.ndenumerate(self._confusion_matrix):
            if z != 0:
                if z > np.max(self._confusion_matrix) / 2.:
                    ax.text(j, i, '{:d}'.format(z), ha='center', va='center', color='white')
                else:
                    ax.text(j, i, '{:d}'.format(z), ha='center', va='center', color='black')

        ax.axhline(int(N1 - 1) + 0.5, color='black')
        ax.axvline(int(N2 - 1) + 0.5, color='black')

        # Major ticks
        ax.set_xticks(np.arange(0, N2 + 1))
        ax.set_yticks(np.arange(0, N1 + 1))
        ax.xaxis.tick_bottom()
        # Labels for major ticks
        ax.set_xticklabels(np.append(st2_idxs, 'FN'), fontsize=12)
        ax.set_yticklabels(np.append(st1_idxs, 'FP'), fontsize=12)

        if xlabel == None:
            if self.sorting2_name is None:
                ax.set_xlabel('Sorting 2', fontsize=15)
            else:
                ax.set_xlabel(self.sorting2_name, fontsize=15)
        else:
            ax.set_xlabel(xlabel, fontsize=20)
        if ylabel == None:
            if self.sorting1_name is None:
                ax.set_ylabel('Sorting 1', fontsize=15)
            else:
                ax.set_ylabel(self.sorting1_name, fontsize=15)
        else:
            ax.set_ylabel(ylabel, fontsize=20)

        return ax

    def _do_matching(self):
        self._event_counts_1,  self._event_counts_2, self._matching_event_counts_12,\
            self._best_match_units_12, self._matching_event_counts_21,\
            self._best_match_units_21,self._unit_map12,\
            self._unit_map21 = do_matching(self._sorting1, self._sorting2, self._delta_frames, self._min_accuracy, self._n_jobs)

    def _do_counting(self, verbose=False):
        self._labels_st1, self._labels_st2 = do_score_labels(self._sorting1, self._sorting2,
                                                             self._delta_frames, self._unit_map12)
        self._mixed_counts = do_counting(self._sorting1, self._sorting2, self._unit_map12,
                                         self._labels_st1, self._labels_st2)

    def _do_confusion(self):
        self._confusion_matrix,  st1_idxs, st2_idxs = do_confusion_matrix(self._sorting1, self._sorting2,
                                                self._unit_map12, self._labels_st1, self._labels_st2)

        return st1_idxs, st2_idxs

    def get_performance(self, method='by_spiketrain', output='pandas'):
        """
        Compute performance rate with several method:
          * 'by_spiketrain'
          * 'pooled_with_sum'
          * 'pooled_with_average'

        Parameters
        ----------
        method: str
            'by_spiketrain', 'pooled_with_sum' or 'pooled_with_average'
        output: str
            'pandas' or 'dict'

        Returns
        -------
        perf: dict or pandas dataframe
            Dictionary or dataframe (based on 'output') with performance entries
        """
        if method != 'by_spiketrain' and method != 'pooled_with_sum' and method != 'pooled_with_average':
            raise Exception("'method' can be 'by_spiketrain', 'pooled_with_average', or 'pooled_with_sum'")

        if self._mixed_counts is None:
            self._do_counting()
        if method == 'by_spiketrain':
            assert output=='pandas', "Output must be pandas for by_spiketrain"

            unit1_ids = self._sorting1.get_unit_ids()
            perf = pd.DataFrame(index=unit1_ids, columns=_perf_keys)

            for u1 in unit1_ids:
                counts = self._mixed_counts['by_spiketrains'][u1]

                perf.loc[u1, 'tp_rate'] = counts['TP'] / counts['NB_SPIKE_1'] * 100
                perf.loc[u1, 'cl_rate'] = counts['CL'] / counts['NB_SPIKE_1'] * 100
                perf.loc[u1, 'fn_rate'] = counts['FN'] / counts['NB_SPIKE_1'] * 100
                perf.loc[u1, 'fp_rate_st1'] = counts['FP'] / counts['NB_SPIKE_1'] * 100
                if counts['NB_SPIKE_2'] > 0:
                    perf.loc[u1, 'fp_rate_st2'] = counts['FP'] / counts['NB_SPIKE_2'] * 100
                else:
                    perf.loc[u1, 'fp_rate_st2'] = np.nan

            perf['accuracy'] = perf['tp_rate'] / (perf['tp_rate'] + perf['fn_rate']+perf['fp_rate_st1']) * 100
            perf['sensitivity'] = perf['tp_rate'] / (perf['tp_rate'] + perf['fn_rate']) * 100
            perf['miss_rate'] = perf['fn_rate'] / (perf['tp_rate'] + perf['fn_rate']) * 100
            perf['precision'] = perf['tp_rate'] / (perf['tp_rate'] + perf['fp_rate_st1']) * 100
            perf['false_discovery_rate'] = perf['fp_rate_st1'] / (perf['tp_rate'] + perf['fp_rate_st1']) * 100

        elif method == 'pooled_with_sum':
            counts = self._mixed_counts['pooled_with_sum']

            tp_rate = float(counts['TP']) / counts['TOT_ST1'] * 100
            cl_rate = float(counts['CL']) / counts['TOT_ST1'] * 100
            fn_rate = float(counts['FN']) / counts['TOT_ST1'] * 100
            fp_rate_st1 = float(counts['FP']) / counts['TOT_ST1'] * 100
            if counts['TOT_ST2'] > 0:
                fp_rate_st2 = float(counts['FP']) / counts['TOT_ST2'] * 100
                accuracy = tp_rate / (tp_rate + fn_rate + fp_rate_st1) * 100
                sensitivity = tp_rate / (tp_rate + fn_rate) * 100
                miss_rate = fn_rate / (tp_rate + fn_rate) * 100
                precision = tp_rate / (tp_rate + fp_rate_st1) * 100
                false_discovery_rate = fp_rate_st1 / (tp_rate + fp_rate_st1) * 100
            else:
                fp_rate_st2 = np.nan
                accuracy = 0.
                sensitivity = 0.
                miss_rate = np.nan
                precision = 0.
                false_discovery_rate = np.nan


            perf = {'tp_rate': tp_rate, 'fn_rate': fn_rate, 'cl_rate': cl_rate, 'fp_rate_st1': fp_rate_st1,
                    'fp_rate_st2': fp_rate_st2, 'accuracy': accuracy, 'sensitivity': sensitivity,
                    'precision': precision, 'miss_rate': miss_rate, 'false_discovery_rate': false_discovery_rate}

            if output == 'pandas':
                perf = pd.Series(perf)

        elif method == 'pooled_with_average':
            perf = self.get_performance(method='by_spiketrain').mean(axis=0)
            if output == 'dict':
                perf = perf.to_dict()

        return perf


    def print_performance(self, method='by_spiketrain'):
        if method == 'by_spiketrain':
            perf = self.get_performance(method=method, output='pandas')
            #~ print(perf)
            d = {k: perf[k].tolist() for k in perf.columns}
            txt = _template_txt_performance.format(method=method, **d)
            print(txt)

        elif method == 'pooled_with_sum':
            perf = self.get_performance(method=method, output='dict')
            txt = _template_txt_performance.format(method=method, **perf)
            print(txt)

        elif method == 'pooled_with_average':
            perf = self.get_performance(method=method, output='dict')
            txt = _template_txt_performance.format(method=method, **perf)
            print(txt)
    
    def get_number_units_above_threshold(self, columns='accuracy', threshold=95, ):
        perf = self.get_performance(method='by_spiketrain', output='pandas')
        nb = (perf[columns] > threshold).sum()
        return nb
        
        


class MappedSortingExtractor(se.SortingExtractor):
    def __init__(self, sorting, unit_map):
        se.SortingExtractor.__init__(self)
        self._sorting = sorting
        self._unit_map = unit_map
        self._unit_ids = list(self._unit_map.keys())

    def get_unit_ids(self, unit_ids=None):
        if unit_ids is None:
            return self._unit_ids
        else:
            return self._unit_ids[unit_ids]

    def get_mapped_unit_ids(self, unit_ids=None):
        if unit_ids is None:
            return list(self._unit_map.values())
        elif isinstance(unit_ids, (int, np.integer)):
            return self._unit_map[unit_ids]
        else:
            return list([self._unit_map[u] for u in self._unit_ids if u in unit_ids])

    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        unit2 = self._unit_map[unit_id]
        if unit2 != -1:
            return self._sorting.get_unit_spike_train(unit2, start_frame=start_frame, end_frame=end_frame)
        else:
            print(unit_id, " is not matched!")
            return None


def compare_two_sorters(sorting1, sorting2, sorting1_name=None, sorting2_name=None, delta_frames=10, min_accuracy=0.5,
                        count=False, n_jobs=1, verbose=False):
    '''
    Compares two spike sorter outputs.

    - Spike trains are matched based on their agreement scores
    - Individual spikes are labelled as true positives (TP), false negatives (FN), false positives 1 (FP from spike
    train 1), false positives 2 (FP from spike train 2), misclassifications (CL)

    It also allows to compute_performance and confusion matrix.

    Parameters
    ----------
    sorting1: SortingExtractor
        The first sorting for the comparison
    sorting2: SortingExtractor
        The second sorting for the comparison
    sorting1_name: str
        The name of sorter 1
    sorting2_name: : str
        The name of sorter 2
    delta_frames: int
        Number of frames to consider coincident spikes (default 10)
    min_accuracy: float
        Minimum agreement score to match units (default 0.5)
    count: bool
        If True, counts are performed at initialization
     n_jobs: int
        Number of cores to use in parallel. Uses all availible if -1
    verbose: bool
        If True, output is verbose
    Returns
    -------
    sorting_comparison: SortingComparison
        The SortingComparison object

    '''
    return SortingComparison(sorting1, sorting2, sorting1_name, sorting2_name, delta_frames, min_accuracy,
                             count, n_jobs, verbose)

# usefull for gathercomparison
_perf_keys = ['tp_rate', 'fn_rate', 'cl_rate','fp_rate_st1', 'fp_rate_st2', 'accuracy', 'sensitivity', 'precision',
              'miss_rate', 'false_discovery_rate']



_template_txt_performance = """PERFORMANCE
Method : {method}
TP : {tp_rate} %
CL : {cl_rate} %
FN : {fn_rate} %
FP (%ST1): {fp_rate_st1} %
FP (%ST2): {fp_rate_st2} %

ACCURACY: {accuracy}
SENSITIVITY: {sensitivity}
MISS RATE: {miss_rate}
PRECISION: {precision}
FALSE DISCOVERY RATE: {false_discovery_rate}
"""
