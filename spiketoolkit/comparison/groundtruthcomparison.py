import numpy as np
import pandas as pd

from .basecomparison import BaseComparison
from .comparisontools import compute_agreement_score


# Note for dev,  because of  BaseComparison internally:
#     sorting1 = gt_sorting
#     sorting2 = tested_sorting

class GroundTruthComparison(BaseComparison):
    """
    Class to compare a sorter to ground truth (GT)
    
    This class can:
      * compute a "macth between gt_sorting and tested_sorting
      * compte th score label (TP, FN, CL, FP) for each spike
      * count by unit of GT the total of each (TP, FN, CL, FP) into a Dataframe 
        GroundTruthComparison.count
      * compute the confusion matrix .get_confusion_matrix()
      * compute some performance metric with several strategy based on 
        the count score by unit
      * count how much well detected units with some threshold selection
      * count false positve detected units
      * count units detected twice (or more)
      * summary all this
    """

    def __init__(self, gt_sorting, tested_sorting, gt_name=None, tested_name=None,
                 delta_time=0.3, min_accuracy=0.5, exhaustive_gt=False,
                 n_jobs=-1, compute_labels=True, compute_misclassification=True, verbose=False):
        if gt_name is None:
            gt_name = 'ground truth'
        if tested_name is None:
            tested_name = 'tested'
        BaseComparison.__init__(self, gt_sorting, tested_sorting, sorting1_name=gt_name, sorting2_name=tested_name,
                                delta_time=delta_time, min_accuracy=min_accuracy, n_jobs=n_jobs, compute_labels=compute_labels,
                                compute_misclassification=compute_misclassification, verbose=verbose)
        self.exhaustive_gt = exhaustive_gt
        self._do_count()

    def _do_count(self):
        """
        Do raw count into a dataframe.
        """
        unit1_ids = self.sorting1.get_unit_ids()
        columns = ['tp', 'fn', 'cl', 'fp', 'num_gt', 'num_tested', 'tested_id']
        self.count = pd.DataFrame(index=unit1_ids, columns=columns)
        self.count.index.name = 'gt_unit_id'
        for u1 in unit1_ids:
            u2 = self._unit_map12[u1]

            self.count.loc[u1, 'tp'] = np.sum(self._labels_st1[u1] == 'TP')
            self.count.loc[u1, 'cl'] = sum(e.startswith('CL') for e in self._labels_st1[u1])
            self.count.loc[u1, 'fn'] = np.sum(self._labels_st1[u1] == 'FN')
            self.count.loc[u1, 'num_gt'] = self._labels_st1[u1].size
            self.count.loc[u1, 'tested_id'] = u2

            if u2 == -1:
                self.count.loc[u1, 'fp'] = 0
                self.count.loc[u1, 'num_tested'] = 0
            else:
                self.count.loc[u1, 'fp'] = np.sum(self._labels_st2[u2] == 'FP')
                self.count.loc[u1, 'num_tested'] = self._labels_st2[u2].size

    def get_performance(self, method='by_unit', output='pandas'):
        """
        Get performance rate with several method:
          * 'raw_count' : just render the raw count table
          * 'by_unit' : render perf as rate unit by unit of the GT
          * 'pooled_with_sum' : pool all spike with a sum and compute rate
          * 'pooled_with_average' : compute rate unit by unit and average

        Parameters
        ----------
        method: str
            'by_unit', 'pooled_with_sum' or 'pooled_with_average'
        output: str
            'pandas' or 'dict'

        Returns
        -------
        perf: pandas dataframe/series (or dict)
            dataframe/series (based on 'output') with performance entries
        """
        possibles = ('raw_count', 'by_unit', 'pooled_with_sum', 'pooled_with_average')
        if method not in possibles:
            raise Exception("'method' can be " + ' or '.join(possibles))

        if method == 'raw_count':
            perf = self.count
            
        elif method == 'by_unit':
            unit1_ids = self.sorting1.get_unit_ids()
            perf = pd.DataFrame(index=unit1_ids, columns=_perf_keys)
            perf.index.name = 'gt_unit_id'
            c = self.count
            tp, cl, fn, fp, num_gt = c['tp'], c['cl'], c['fn'], c['fp'], c['num_gt']
            perf = _compute_perf(tp, cl, fn, fp, num_gt, perf)

        elif method == 'pooled_with_sum':
            # here all spike from units are polled with sum.
            perf = pd.Series(index=_perf_keys)
            c = self.count
            tp, cl, fn, fp, num_gt = c['tp'].sum(), c['cl'].sum(), c['fn'].sum(), c['fp'].sum(), c['num_gt'].sum()
            perf = _compute_perf(tp, cl, fn, fp, num_gt, perf)

        elif method == 'pooled_with_average':
            perf = self.get_performance(method='by_unit').mean(axis=0)

        if output == 'dict' and isinstance(perf, pd.Series):
            perf = perf.to_dict()

        return perf

    def print_performance(self, method='by_unit'):
        """
        Print performance with the selected method
        """

        if self._compute_misclassification:
            template_txt_performance = _template_txt_performance_with_cl
        else:
            template_txt_performance = _template_txt_performance
        
        if method == 'by_unit':
            perf = self.get_performance(method=method, output='pandas')
            perf = perf * 100
            # ~ print(perf)
            d = {k: perf[k].tolist() for k in perf.columns}
            txt = template_txt_performance.format(method=method, **d)
            print(txt)

        elif method == 'pooled_with_sum':
            perf = self.get_performance(method=method, output='pandas')
            perf = perf * 100
            txt = template_txt_performance.format(method=method, **perf.to_dict())
            print(txt)

        elif method == 'pooled_with_average':
            perf = self.get_performance(method=method, output='pandas')
            perf = perf * 100
            txt = template_txt_performance.format(method=method, **perf.to_dict())
            print(txt)

    def print_summary(self, min_redundant_agreement=0.3, **kargs_well_detected):
        """
        Print a global performance summary that depend on the context:
          * exhaustive= True/False
          * how many gt units (one or several)
        
        This summary mix several performance metrics.
        """
        txt = _template_summary_part1

        d = dict(
            num_gt=len(self._labels_st1),
            num_tested=len(self._labels_st2),
            num_well_detected=self.count_well_detected_units(**kargs_well_detected),
            num_redundant=self.count_redundant_units(min_redundant_agreement=min_redundant_agreement),
        )

        if self.exhaustive_gt:
            txt = txt + _template_summary_part2
            d['num_false_positive_units'] = self.count_false_positive_units()
            d['num_bad'] = self.count_bad_units()

        txt = txt.format(**d)

        print(txt)

    def get_well_detected_units(self, **thresholds):
        """
        Get the units in GT that are well detected with a comninaison a treshold level
        on some columns (accuracy, recall, precision, miss_rate, ...):
        
        
        
        By default threshold is {'accuray'=0.95} meaning that all units with
        accuracy above 0.95 are selected.
        
        For some thresholds columns units are below the threshold for instance
        'miss_rate', 'false_discovery_rate', 'misclassification_rate'
        
        If several thresh are given the the intersect of selection is kept.
        
        For instance threholds = {'accuracy':0.9, 'miss_rate':0.1 }
        give units with accuracy>0.9 AND miss<0.1
        Parameters
        ----------
        **thresholds : dict
            A dict that contains some threshold of columns of perf Dataframe.
            If sevral threhold they are combined.
        """
        if len(thresholds) == 0:
            thresholds = {'accuracy' : 0.95 }
        
        _above = ['accuracy', 'recall', 'precision',]
        _below = ['false_discovery_rate',  'miss_rate', 'misclassification_rate']
        
        perf = self.get_performance(method='by_unit')
        keep = perf['accuracy'] >= 0 # tale all
        
        for col, thresh in thresholds.items():
            if col in _above:
                keep = keep & (perf[col] >= thresh)
            elif col in _below:
                keep = keep & (perf[col] <= thresh)
            else:
                raise ValueError('Threshold column do not exits', col)

        return perf[keep].index.tolist()

    def count_well_detected_units(self, **kargs):
        """
        Count how many well detected units.
        Kargs are the same as get_well_detected_units.
        """
        return len(self.get_well_detected_units(**kargs))

    def get_false_positive_units(self):
        """
        Return units list of "false positive units" from tested_sorting.
        
        "false positive units" ara defined as units in tested that
        are not matched at all in GT units.
        
        Need exhaustive_gt=True
        """
        assert self.exhaustive_gt, 'false_positive_units list is valid only if exhaustive_gt=True'
        fake_ids = []
        unit2_ids = self.sorting2.get_unit_ids()
        for u2 in unit2_ids:
            if self._best_match_units_21[u2] == -1:
                fake_ids.append(u2)
        return fake_ids

    def count_false_positive_units(self):
        """
        See get_false_positive_units.
        """
        return len(self.get_false_positive_units())

    def get_redundant_units(self, min_redundant_agreement=0.3):
        """
        Return "redundant units"
        
        
        "redundant units" are defined as units in tested
        that match a GT units with a big agreement score
        but it is not the best match.
        In other world units in GT that detected twice or more.
        
        Parameters
        ----------
        min_redundant_agreement: float (default 0.3)
            The minimum agreement between gt and tested units
            that are best match to be counted as "redundant" units.
        
        """
        best_match = list(self._unit_map12.values())
        redundant_ids = []
        unit2_ids = self.sorting2.get_unit_ids()
        for u2 in unit2_ids:
            if u2 not in best_match and self._best_match_units_21[u2] != -1:
                u1 = self._best_match_units_21[u2]
                if self._unit_map12[u1] == -1:
                    continue
                if u2 == self._unit_map12[u1]:
                    continue

                num_matches = self._matching_event_counts_12[u1].get(u2, 0)
                num1 = self._event_counts_1[u1]
                num2 = self._event_counts_2[u2]
                agree_score = compute_agreement_score(num_matches, num1, num2)

                if agree_score > min_redundant_agreement:
                    redundant_ids.append(u2)

        return redundant_ids

    def count_redundant_units(self, min_redundant_agreement=0.3):
        """
        See get_redundant_units.
        """
        return len(self.get_redundant_units(min_redundant_agreement=min_redundant_agreement))

    def get_bad_units(self):
        """
        Return units list of "bad units".
        
        "bad units" are defined as units in tested that are not
        in the best match list of GT units.
        
        So it is the union of "false positive units" + "redundant units".
        
        Need exhaustive_gt=True
        """
        assert self.exhaustive_gt, 'bad_units list is valid only if exhaustive_gt=True'
        best_match = list(self._unit_map12.values())
        bad_ids = []
        unit2_ids = self.sorting2.get_unit_ids()
        for u2 in unit2_ids:
            if u2 not in best_match:
                bad_ids.append(u2)
        return bad_ids

    def count_bad_units(self):
        """
        See get_bad_units
        """
        return len(self.get_bad_units())


def _compute_perf(tp, cl, fn, fp, num_gt, perf):
    """
    This compte perf formula.
    this trick here is that it works both on pd.Series and pd.Dataframe
    line by line.
    This it is internally used by perf by psiketrain and poll_with_sum.
    
    https://en.wikipedia.org/wiki/Sensitivity_and_specificity
    
    Note :
      * we don't have TN because it do not make sens here.
      * 'accuracy' = 'tp_rate' because TN=0
      * 'recall' = 'sensitivity'
    """

    perf['accuracy'] = tp / (tp + fn + fp)
    perf['recall'] = tp / (tp + fn)
    perf['precision'] = tp / (tp + fp)
    perf['false_discovery_rate'] = fp / (tp + fp)
    perf['miss_rate'] = fn / num_gt
    perf['misclassification_rate'] = cl / num_gt

    return perf


# usefull also for gathercomparison
_perf_keys = ['accuracy', 'recall', 'precision', 'false_discovery_rate', 'miss_rate', 'misclassification_rate']

_template_txt_performance = """PERFORMANCE
Method : {method}

ACCURACY: {accuracy}
RECALL: {recall}
PRECISION: {precision}
FALSE DISCOVERY RATE: {false_discovery_rate}
MISS RATE: {miss_rate}
"""

_template_txt_performance_with_cl = _template_txt_performance + 'MISS CLASSIFICATION RATE: {misclassification_rate}\n'


_template_summary_part1 = """SUMMARY
GT num_units: {num_gt}
TESTED num_units: {num_tested}
num_well_detected: {num_well_detected} 
num_redundant: {num_redundant}
"""

_template_summary_part2 = """num_false_positive_units {num_false_positive_units}
num_bad: {num_bad}
"""


def compare_sorter_to_ground_truth(gt_sorting, tested_sorting, gt_name=None, tested_name=None,
                                   delta_time=0.3, min_accuracy=0.5, exhaustive_gt=True, n_jobs=-1,
                                   compute_labels=True, compute_misclassification=False, verbose=False):
    '''
    Compares a sorter to a ground truth.

    - Spike trains are matched based on their agreement scores
    - Individual spikes are labelled as true positives (TP), false negatives (FN),
    false positives 1 (FP), misclassifications (CL)

    It also allows to compute_performance and confusion matrix.

    Parameters
    ----------
    gt_sorting: SortingExtractor
        The first sorting for the comparison
    tested_sorting: SortingExtractor
        The second sorting for the comparison
    gt_name: str
        The name of sorter 1
    tested_name: : str
        The name of sorter 2
    delta_time: float
        Number of ms to consider coincident spikes (default 0.3 ms)
    min_accuracy: float
        Minimum agreement score to match units (default 0.5)
    exhaustive_gt: bool (default True)
        Tell if the ground true is "exhaustive" or not. In other world if the
        GT have all possible units. It allows more performance measurement.
        For instance, MEArec simulated dataset have exhaustive_gt=True
    n_jobs: int
        Number of cores to use in parallel. Uses all available if -1
    compute_labels: bool
        If True, labels are computed at instantiation (default True)
    compute_misclassification: bool
        If True, misclassification errors are computed (default False)
    verbose: bool
        If True, output is verbose
    Returns
    -------
    sorting_comparison: SortingComparison
        The SortingComparison object

    '''
    return GroundTruthComparison(gt_sorting=gt_sorting, tested_sorting=tested_sorting, gt_name=gt_name,
                                 tested_name=tested_name, delta_time=delta_time, min_accuracy=min_accuracy,
                                 exhaustive_gt=exhaustive_gt, n_jobs=n_jobs, compute_labels=compute_labels,
                                 compute_misclassification=compute_misclassification, verbose=verbose)
