from .basecomparison import BaseComparison

import numpy as np
import pandas as pd


# Note for dev,  because of  BaseComparison internally:
#     sorting1 = gt_sorting
#     sorting2 = other_sorting

class GroundTruthComparison(BaseComparison):
    """
    Class to compare a sorter to ground truth (GT)
    
    This class can:
      * compute a "macth between gt_sorting and other_sorting
      * compte th score label (TP, FN, CL, FP) for each spike
      * count by spiketrain of GT the total of each (TP, FN, CL, FP) into a Dataframe 
        GroundTruthComparison.count
      * compute the confusion matrix .get_confusion_matrix()
      * compute some performance metric with several strategy based on 
        the count score by spiketrain
      * count how much well detected units with some threshold selection
      * count false positve detected units
      * count units detected twice (or more)
      * summary all this
    """
    def __init__(self, gt_sorting, other_sorting, gt_name=None, other_name=None,
                delta_frames=10, min_accuracy=0.5, exhaustive_gt=False,
                n_jobs=1, verbose=False):
        BaseComparison.__init__(self, gt_sorting, other_sorting, sorting1_name=gt_name, sorting2_name=other_name,
                                    delta_frames=delta_frames, min_accuracy=min_accuracy, n_jobs=n_jobs, verbose=verbose)
        self.exhaustive_gt = exhaustive_gt

        self._do_count()

    
    def _do_count(self):
        """
        Do raw count into a dataframe.
        """
        unit1_ids = self._sorting1.get_unit_ids()
        columns = ['tp', 'fn', 'cl','fp', 'num_gt', 'num_other', 'other_id']
        self.count = pd.DataFrame(index=unit1_ids, columns=columns)
        for u1 in unit1_ids:
            u2 = self._unit_map12[u1]
            
            self.count.loc[u1, 'tp'] = np.sum(self._labels_st1[u1] == 'TP')
            self.count.loc[u1, 'cl'] = sum(e.startswith('CL') for e in self._labels_st1[u1])
            self.count.loc[u1, 'fn'] = np.sum(self._labels_st1[u1] == 'FN')
            self.count.loc[u1, 'num_gt'] = self._labels_st1[u1].size
            self.count.loc[u1, 'other_id'] = u2

            if u2==-1:
                self.count.loc[u1, 'fp'] = 0
                self.count.loc[u1, 'num_other'] = 0
            else:
                self.count.loc[u1, 'fp'] = np.sum(self._labels_st2[u2] == 'FP')
                self.count.loc[u1, 'num_other'] = self._labels_st2[u2].size
            

    def get_performance(self, method='by_spiketrain', output='pandas'):
        """
        Get performance rate with several method:
          * 'raw_count' : just render the raw count table
          * 'by_spiketrain' : render perf as rate spiketrain by spiketrain of the GT
          * 'pooled_with_sum' : pool all spike with a sum and compute rate
          * 'pooled_with_average' : compute rate spiketrain by spiketrain and average

        Parameters
        ----------
        method: str
            'by_spiketrain', 'pooled_with_sum' or 'pooled_with_average'
        output: str
            'pandas' or 'dict'

        Returns
        -------
        perf: pandas dataframe/series (or dict)
            dataframe/series (based on 'output') with performance entries
        """
        possibles = ('raw_count', 'by_spiketrain', 'pooled_with_sum', 'pooled_with_average')
        if method not in possibles:
            raise Exception("'method' can be " + ' or '.join(possibles))


        if method =='raw_count':
            perf = self.count
            
        elif method == 'by_spiketrain':
            unit1_ids = self._sorting1.get_unit_ids()
            perf = pd.DataFrame(index=unit1_ids, columns=_perf_keys)
            
            counts = self.count
            perf['tp_rate'] = counts['tp'] / counts['num_gt']
            perf['cl_rate'] = counts['cl'] / counts['num_gt']
            perf['fn_rate'] = counts['fn'] / counts['num_gt']
            perf['fp_rate'] = counts['fp'] / counts['num_gt']
            
            perf['accuracy'] = perf['tp_rate'] / (perf['tp_rate'] + perf['fn_rate']+perf['fp_rate'])
            perf['sensitivity'] = perf['tp_rate'] / (perf['tp_rate'] + perf['fn_rate'])
            perf['miss_rate'] = perf['fn_rate'] / (perf['tp_rate'] + perf['fn_rate'])
            perf['precision'] = perf['tp_rate'] / (perf['tp_rate'] + perf['fp_rate'])
            perf['false_discovery_rate'] = perf['fp_rate'] / (perf['tp_rate'] + perf['fp_rate'])

        elif method == 'pooled_with_sum':
            perf = pd.Series(index=_perf_keys)
            
            sum_num_gt = int(self.count['num_gt'].sum())
            
            perf['tp_rate'] = self.count['tp'].sum() / sum_num_gt
            perf['cl_rate'] = self.count['cl'].sum() / sum_num_gt
            perf['fn_rate'] = self.count['fn'].sum() / sum_num_gt
            perf['fp_rate'] = self.count['fp'].sum() / sum_num_gt
            
            if (perf['tp_rate'] + perf['fn_rate']) > 0:
                perf['accuracy'] = perf['tp_rate'] / (perf['tp_rate'] + perf['fn_rate'] + perf['fp_rate'])
                perf['sensitivity'] = perf['tp_rate'] / (perf['tp_rate'] + perf['fn_rate'])
                perf['miss_rate'] = perf['fn_rate'] / (perf['tp_rate'] + perf['fn_rate'])
                perf['precision'] = perf['tp_rate'] / (perf['tp_rate'] + perf['fp_rate'])
                perf['false_discovery_rate'] = perf['fp_rate'] / (perf['tp_rate'] + perf['fp_rate'])
            else:
                perf['accuracy'] = 0.
                perf['sensitivity'] = 0.
                perf['miss_rate'] = np.nan
                perf['precision'] = 0.
                perf['false_discovery_rate'] = np.nan

        elif method == 'pooled_with_average':
            perf = self.get_performance(method='by_spiketrain').mean(axis=0)
        
        if output == 'dict' and isinstance(perf, pd.Series):
            perf = perf.to_dict()

        return perf

    def print_performance(self, method='by_spiketrain'):
        """
        Print performance with the selected method
        """
        if method == 'by_spiketrain':
            perf = self.get_performance(method=method, output='pandas')
            perf = perf * 100
            #~ print(perf)
            d = {k: perf[k].tolist() for k in perf.columns}
            txt = _template_txt_performance.format(method=method, **d)
            print(txt)

        elif method == 'pooled_with_sum':
            perf = self.get_performance(method=method, output='pandas')
            perf = perf * 100
            txt = _template_txt_performance.format(method=method, **perf.to_dict())
            print(txt)

        elif method == 'pooled_with_average':
            perf = self.get_performance(method=method, output='pandas')
            perf = perf * 100
            txt = _template_txt_performance.format(method=method, **perf.to_dict())
            print(txt)
    
    def print_summary(self, **kargs_well_detected):
        """
        Print a global performance summary that depend on the context:
          * exhaustive= True/False
          * how many gt units (one or several)
        
        This summary mix several performance metrics.
        """
        txt = _template_summary_part1

        d = dict(
            num_gt=len(self._labels_st1),
            num_other=len(self._labels_st2),
            num_well_detected = self.count_well_detected_units(**kargs_well_detected),
            num_bad=self.count_bad_units(),
            num_redundant=self.count_redundant_units(),
        )
        
        if self.exhaustive_gt:
            txt = txt + _template_summary_part2
            d['num_false_positive_units'] = self.count_false_positive_units()
        
        txt = txt.format(**d)

        print(txt)
    
    
    def get_well_detected_units(self, tp_thresh=None, accuracy_thresh=None,
                cl_thresh=None, fp_thresh=None):
        """
        Get the units in GT that are well detected with a comninaison a treshold level
        on some columns (tp_rate, accuracy_rate, cl_rate, fp_rate):
        
        If several thresh are given the the intersect of selection is kept.
        
        if all parameters are None then default value is tp_rate=0.95
        
        Parameters
        ----------
        tp_thresh:
            Threshold tp_rate score above a units is selected
        
        accuracy_thresh:
            Threshold accuracy score above a units is selected
        
        cl_thresh:
            Threshold cl_rate score under a units is selected
        
        fp_thresh:
            Threshold fp_rate score under a units is selected
        
        """
        if all(e is None for e in (tp_thresh, accuracy_thresh, cl_thresh, fp_thresh)):
            # default value
            tp_thresh=0.95
        
        perf = self.get_performance(method='by_spiketrain')
        keep = perf['tp_rate']>=0
        
        if tp_thresh is not None:
            keep = keep & (perf['tp_rate'] >= tp_thresh)
        
        if accuracy_thresh is not None:
            keep = keep & (perf['accuracy'] >= accuracy_thresh)
        
        if cl_thresh is not None:
            keep = keep & (perf['cl_rate'] <= cl_thresh)
        
        if fp_thresh is not None:
            keep = keep & (perf['fp_rate'] <= fp_thresh)
        
        return perf[keep].index.tolist()
    
    def count_well_detected_units(self, **kargs):
        """
        Count how many well detected units.
        Kargs are the same as get_well_detected_units.
        """
        return len(self.get_well_detected_units(**kargs))
    
    def get_false_positive_units(self):
        """
        Return units listof "false positive units" from other_sorting.
        
        "false positive units" ara defined as units in other that
        are not matched at all in GT units.
        
        Need exhaustive_gt=True
        """
        assert self.exhaustive_gt, 'false_positive_units list is valid only if exhaustive_gt=True'
        fake_ids = []
        unit2_ids = self._sorting2.get_unit_ids()
        for u2 in unit2_ids:
            if self._best_match_units_21[u2] == -1:
                fake_ids.append(u2)
        return fake_ids
    
    def count_false_positive_units(self):
        """
        See get_false_positive_units.
        """
        return len(self.get_false_positive_units())
    
    def get_redundant_units(self):
        """
        Return "redundant units"
        
        
        "redundant units" are defined as units in other
        that match a GT units but it is not the best match.
        In other world units in GT that detected twice or more.
        
        """
        best_match = list(self._unit_map12.values())
        redundant_ids = []
        unit2_ids = self._sorting2.get_unit_ids()
        for u2 in unit2_ids:
            if u2 not in best_match and self._best_match_units_21[u2] != -1:
                redundant_ids.append(u2)
        return redundant_ids
    
    def count_redundant_units(self):
        """
        See get_redundant_units.
        """
        return len(self.get_redundant_units())
    
    def get_bad_units(self):
        """
        Return units list of "bad units".
        
        "bad units" are defined as units in other that are not
        in the best match list of GT units.
        
        So it is the union of "false positive units" + "redundant units".
        
        Need exhaustive_gt=True
        """
        assert self.exhaustive_gt, 'bad_units list is valid only if exhaustive_gt=True'
        best_match = list(self._unit_map12.values())
        bad_ids = []
        unit2_ids = self._sorting2.get_unit_ids()
        for u2 in unit2_ids:
            if u2 not in best_match:
                bad_ids.append(u2)
        return bad_ids
    
    def count_bad_units(self):
        """
        See get_bad_units
        """
        return len(self.get_bad_units())



    
# usefull also for gathercomparison
_perf_keys = ['tp_rate', 'fn_rate', 'cl_rate','fp_rate',  'accuracy', 'sensitivity', 'precision',
              'miss_rate', 'false_discovery_rate']



_template_txt_performance = """PERFORMANCE
Method : {method}
TP : {tp_rate} %
CL : {cl_rate} %
FN : {fn_rate} %
FP: {fp_rate} %

ACCURACY: {accuracy}
SENSITIVITY: {sensitivity}
MISS RATE: {miss_rate}
PRECISION: {precision}
FALSE DISCOVERY RATE: {false_discovery_rate}
"""

_template_summary_part1 = """SUMMARY
GT num_units: {num_gt}
OTHER num_units: {num_other}
num_well_detected: {num_well_detected} 
num_bad: {num_bad}
"""

_template_summary_part2 = """num_false_positive_units {num_false_positive_units}
"""


    
    


def compare_sorter_to_ground_truth(gt_sorting, other_sorting, gt_name=None, other_name=None, 
                delta_frames=10, min_accuracy=0.5, exhaustive_gt=True, n_jobs=1, verbose=False):
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
    other_sorting: SortingExtractor
        The second sorting for the comparison
    gt_name: str
        The name of sorter 1
    other_name: : str
        The name of sorter 2
    delta_frames: int
        Number of frames to consider coincident spikes (default 10)
    min_accuracy: float
        Minimum agreement score to match units (default 0.5)
    exhaustive_gt: bool (default True)
        Tell if the ground true is "exhaustive" or not. In other world if the
        GT have all possible units. It allows more performance measurement.
        For instance, MEArec simulated dataset have exhaustive_gt=True
     n_jobs: int
        Number of cores to use in parallel. Uses all availible if -1
    verbose: bool
        If True, output is verbose
    Returns
    -------
    sorting_comparison: SortingComparison
        The SortingComparison object

    '''
    return GroundTruthComparison(gt_sorting, other_sorting, gt_name, other_name,
                            delta_frames, min_accuracy, exhaustive_gt, n_jobs, verbose)
