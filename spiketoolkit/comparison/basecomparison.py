import numpy as np
from .comparisontools import (do_matching, do_score_labels, do_confusion_matrix)


class BaseComparison:
    """
    Base class shared by SortingComparison and GroundTruthComparison
    """
    def __init__(self, sorting1, sorting2, sorting1_name=None, sorting2_name=None, delta_time=0.3, min_accuracy=0.5,
                 n_jobs=1, verbose=False, sampling_frequency=None, compute_labels=True, compute_misclassification=False):
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
        # elif sampling_frequency is None:
        #     raise Exception("SortingExtractors do not have sampling frequency information. Provide it with the "
        #                     "'sampling_frequency' argument.")
        if sampling_frequency is not None:
            self._delta_frames = int(delta_time / 1000 * sampling_frequency)
        else:
            print("Warning: sampling frequency information not found. Setting delta_frames to 10.")
            self._delta_frames = 10
        self._min_accuracy = min_accuracy
        self._n_jobs = n_jobs
        self._compute_labels = compute_labels
        self._compute_misclassification = compute_misclassification
        self._verbose = verbose

        self._do_matching()
        
        self._labels_st1 = None
        self._labels_st2 = None
        if self._compute_labels:
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
        if self._verbose:
            print("Matching...")

        self._event_counts_1, self._event_counts_2, self._matching_event_counts_12, \
        self._best_match_units_12, self._matching_event_counts_21, \
        self._best_match_units_21, self._unit_map12, \
        self._unit_map21 = do_matching(self.sorting1, self.sorting2, self._delta_frames, self._min_accuracy,
                                       self._n_jobs)

    def _do_score_labels(self):
        if self._verbose:
            print("Adding labels...")
        self._labels_st1, self._labels_st2 = do_score_labels(self.sorting1, self.sorting2,
                                                             self._delta_frames, self._unit_map12,
                                                             self._compute_misclassification)

    def _do_confusion_matrix(self):
        if self._verbose:
            print("Computing confusion matrix...")
        if self._labels_st1 is None:
            self._do_score_labels()
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

    def plot_confusion_matrix(self, xlabel=None, ylabel=None, ax=None, count_text=True):
        # Samuel EDIT
        # This must be moved in spikewidget
        # Please wait for me Alessio.
        import matplotlib.pylab as plt

        if self._confusion_matrix is None:
            self._do_confusion_matrix()

        sorting1 = self.sorting1
        sorting2 = self.sorting2
        unit1_ids = sorting1.get_unit_ids()
        unit2_ids = sorting2.get_unit_ids()
        N1 = len(unit1_ids)
        N2 = len(unit2_ids)

        if ax is None:
            fig, ax = plt.subplots()

        # Using matshow here just because it sets the ticks up nicely. imshow is faster.
        ax.matshow(self._confusion_matrix, cmap='Greens')

        if count_text:
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
        ax.set_xticklabels(np.append(self._st2_idxs, 'FN'), fontsize=12)
        ax.set_yticklabels(np.append(self._st1_idxs, 'FP'), fontsize=12)

        if xlabel is None:
            ax.set_xlabel(self.sorting2_name, fontsize=15)
        else:
            ax.set_xlabel(xlabel, fontsize=20)

        if ylabel is None:
            ax.set_ylabel(self.sorting1_name, fontsize=15)
        else:
            ax.set_ylabel(ylabel, fontsize=20)

        return ax
