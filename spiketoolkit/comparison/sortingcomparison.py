

import spikeextractors as se
from .basecomparison import BaseComparison
from .comparisontools import compute_agreement_score

import numpy as np


class SortingComparison(BaseComparison):
    """
    Class for comparison of two sorters when no assumption is done.
    
    
    
    """


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
        # needed by MultiSortingComparison
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

    #~ def get_false_positive_fraction(self, unit1, unit2=None):
        #~ if unit1 is None:
            #~ raise Exception('get_false_positive_fraction: unit1 must not be None')
        #~ if unit2 is None:
            #~ unit2 = self.get_best_unit_match1(unit1)
            #~ if unit2 is None or unit2 == -1:
                #~ return 0

        #~ if unit1 != -1 and unit2 != -1:
            #~ a = self._matching_event_counts_12[unit1]
            #~ if unit2 not in a:
                #~ return 0
        #~ else:
            #~ return 0
        #~ return 1 - self._compute_safe_frac(a[unit2], self._event_counts_2[unit2])

    #~ def get_false_negative_fraction(self, unit1, unit2=None):
        #~ if unit1 is None:
            #~ raise Exception('get_false_positive_fraction: unit1 must not be None')
        #~ if unit2 is None:
            #~ unit2 = self.get_best_unit_match1(unit1)
            #~ if unit2 is None:
                #~ return 0

        #~ if unit1 != -1 and unit2 != -1:
            #~ a = self._matching_event_counts_12[unit1]
            #~ if unit2 not in a:
                #~ return 0
        #~ else:
            #~ return 0
        #~ return 1 - self._compute_safe_frac(a[unit2], self._event_counts_1[unit1])


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
                        n_jobs=1, verbose=False):
    '''
    Compares two spike sorter outputs.

    - Spike trains are matched based on their agreement scores
    - Individual spikes are labelled as true positives (TP), false negatives (FN), false positives 1 (FP from spike
    train 1), false positives 2 (FP from spike train 2), misclassifications (CL)

    It also allows to get confusion matrix and agreement fraction, false positive fraction and
    false negative fraction.

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
                             n_jobs, verbose)

