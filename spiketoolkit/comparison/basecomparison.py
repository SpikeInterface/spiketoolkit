import numpy as np
from .comparisontools import (do_matching, do_score_labels, do_confusion_matrix)


class BaseComparison:
    """
    Base class shared by SortingComparison and GroundTruthComparison
    """
    def __init__(self, sorting1, sorting2, sorting1_name=None, sorting2_name=None, delta_time=0.3, min_accuracy=0.5,
                 n_jobs=1, verbose=False, sampling_frequency=None):
        self.sorting1 = sorting1
        self.sorting2 = sorting2
        if sorting1_name is None:
            sorting1_name = 'sorting 1'
        if sorting2_name is None:
            sorting2_name = 'sorting 2'
        self.sorting1_name = sorting1_name
        self.sorting2_name = sorting2_name
        if self.sorting1.get_sampling_frequency() is not None:
            assert self.sorting1.get_sampling_frequency() == self.sorting2.get_sampling_frequency(), \
                "The two sorting extractors must have the same sampling frequency"
            sampling_frequency = self.sorting1.get_sampling_frequency()
        elif sampling_frequency is None:
            raise Exception("SortingExtractors do not have sampling frequency information. Provide it with the "
                            "'sampling_frequency' argument.")
        self._delta_frames = int(delta_time * sampling_frequency)
        self._min_accuracy = min_accuracy
        self._n_jobs = n_jobs
        self.verbose = verbose

        self._do_matching()
        self._do_score_labels()

        # confusion matrix is compute on demand
        self._confusion_matrix = None

    def get_labels1(self, unit_id):
        if unit_id in self.sorting1.get_unit_ids():
            return self._labels_st1[unit_id]
        else:
            raise Exception("Unit_id is not a valid unit")

    def get_labels2(self, unit_id):
        if unit_id in self.sorting1.get_unit_ids():
            return self._labels_st1[unit_id]
        else:
            raise Exception("Unit_id is not a valid unit")

    def _do_matching(self):
        if self.verbose:
            print("Matching...")

        self._event_counts_1, self._event_counts_2, self._matching_event_counts_12, \
        self._best_match_units_12, self._matching_event_counts_21, \
        self._best_match_units_21, self._unit_map12, \
        self._unit_map21 = do_matching(self.sorting1, self.sorting2, self._delta_frames, self._min_accuracy,
                                       self._n_jobs)

    def _do_score_labels(self):
        if self.verbose:
            print("do_score_labels...")

        self._labels_st1, self._labels_st2 = do_score_labels(self.sorting1, self.sorting2,
                                                             self._delta_frames, self._unit_map12)

    def _do_confusion_matrix(self):
        if self.verbose:
            print("do_confusion_matrix...")
        self._confusion_matrix, self._st1_idxs, self._st2_idxs = do_confusion_matrix(self.sorting1, self.sorting2,
                                                                                     self._unit_map12, self._labels_st1,
                                                                                     self._labels_st2)

    def get_confusion_matrix(self):
        """
        Computes the confusion matrix.

        Returns
        ------
        confusion_matrix: np.array
            The confusion matrix
        st1_idxs: np.array
            Array with order of units1 in confusion matrix
        st2_idxs: np.array
            Array with order of units2 in confusion matrix
        """
        if self._confusion_matrix is None:
            self._do_confusion_matrix()
        return self._confusion_matrix, self._st1_idxs, self._st2_idxs
