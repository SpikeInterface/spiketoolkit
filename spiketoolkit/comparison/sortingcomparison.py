import numpy as np
import spikeextractors as se
from scipy.optimize import linear_sum_assignment

import pandas as pd

from .comparisontools import (count_matching_events, compute_agreement_score,
                                                do_matching, do_counting, do_confusion_matrix)



class SortingComparison():
    def __init__(self, sorting1, sorting2, sorting1_name=None, sorting2_name=None, delta_tp=10, min_accuracy=0.5,
                 count=False, verbose=False):
        """
        TODO : doc here

        the variable delta_tp should be strongly documented as it affect the match

        the variable min_accuracy should be strongly documented as it affect the match
        """
        self._sorting1 = sorting1
        self._sorting2 = sorting2
        self.sorting1_name = sorting1_name
        self.sorting2_name = sorting2_name
        self._delta_tp = delta_tp
        self._min_accuracy = min_accuracy
        if verbose:
            print("Matching...")
        self._do_matching()

        self._counts = None
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

        The returned MappedSortingExtractor.getUnitspikeTrains returns the the spike trains
        of sorting 2 mapped to the unit_ids of sorting 1.
        """
        return MappedSortingExtractor(self._sorting2, self._unit_map12)

    def get_mapped_sorting2(self):
        # Samuel EDIT : the use case of this must documented
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
        if self._counts is None:
            self._do_counting(verbose=False)

    def plot_confusion_matrix(self, xlabel=None, ylabel=None):
        # Samuel EDIT
        # This must be moved in spikewidget
        import matplotlib.pylab as plt

        if self._counts is None:
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
            self._unit_map21 = do_matching(self._sorting1, self._sorting2, self._delta_tp, self._min_accuracy)

    def _do_counting(self, verbose=False):
        self._counts, self._labels_st1, self._labels_st2 = do_counting(self._sorting1, self._sorting2,
                                                    self._delta_tp, self._unit_map12)

    def _do_confusion(self):
        self._confusion_matrix,  st1_idxs, st2_idxs = do_confusion_matrix(self._sorting1, self._sorting2,
                                                self._unit_map12, self._labels_st1, self._labels_st2)

        return st1_idxs, st2_idxs


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


def compute_performance(SC, verbose=True, output='dict'):
    """
    Return some performance value for comparison.

    Parameters
    -------
    SC: SortingComparison instance
        The SortingComparison

    verbose: bool
        Display on console or not

    output: dict or pandas


    Returns
    ----------

    performance: dict or pandas.Serie depending output param

    """
    counts = SC._counts

    tp_rate = float(counts['TP']) / counts['TOT_ST1'] * 100
    cl_rate = float(counts['CL']) / counts['TOT_ST1'] * 100
    fn_rate = float(counts['FN']) / counts['TOT_ST1'] * 100
    fp_st1 = float(counts['FP']) / counts['TOT_ST1'] * 100
    fp_st2 = float(counts['FP']) / counts['TOT_ST2'] * 100

    accuracy = tp_rate / (tp_rate + fn_rate + fp_st1) * 100
    sensitivity = tp_rate / (tp_rate + fn_rate) * 100
    miss_rate = fn_rate / (tp_rate + fn_rate) * 100
    precision = tp_rate / (tp_rate + fp_st1) * 100
    false_discovery_rate = fp_st1 / (tp_rate + fp_st1) * 100

    performance = {'tp': tp_rate, 'cl': cl_rate, 'fn': fn_rate, 'fp_st1': fp_st1, 'fp_st2': fp_st2,
                   'accuracy': accuracy, 'sensitivity': sensitivity, 'precision': precision, 'miss_rate': miss_rate,
                   'false_disc_rate': false_discovery_rate}

    if verbose:
        txt = _txt_performance.format(**performance)
        print(txt)

    if output == 'dict':
        return performance
    elif output == 'pandas':
        return pd.Series(performance)

# usefull for gathercomparison
_perf_keys = ['tp', 'cl','fp_st1', 'fp_st2', 'accuracy', 'sensitivity', 'precision', 'miss_rate', 'false_disc_rate']


_txt_performance = """PERFORMANCE
TP : {tp} %
CL : {cl} %
FN : {fn} %
FP (%ST1): {fp_st1} %
FP (%ST2): {fp_st2} %

ACCURACY: {accuracy}
SENSITIVITY: {sensitivity}
MISS RATE: {miss_rate}
PRECISION: {precision}
FALSE DISCOVERY RATE: {false_disc_rate}
"""
