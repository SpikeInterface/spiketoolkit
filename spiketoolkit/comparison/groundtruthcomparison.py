from .basecomparison import BaseComparison

import numpy as np
import pandas as pd


class GroundTruthComparison(BaseComparison):
    """
    Class to compare a sorter to ground truth.
    """
    def __init__(self, gt_sorting, other_sorting, gt_name=None, other_name=None,
                delta_frames=10, min_accuracy=0.5, exhaustive_gt=False,
                count=False, n_jobs=1, verbose=False):
        BaseComparison.__init__(self, gt_sorting, other_sorting, sorting1_name=gt_name, sorting2_name=other_name,
                                    delta_frames=delta_frames, min_accuracy=min_accuracy, count=count, n_jobs=n_jobs, verbose=verbose)
        self.exhaustive_gt = exhaustive_gt

    
    def get_raw_count(self):
        """
        Get score count with a DataFrame with 4 columns
        ['tp', 'fn', 'cl','fp', 'nb_gt', 'nb_other']
        and one line per ground truth units
        
        """

        unit1_ids = self._sorting1.get_unit_ids()
        columns = ['tp', 'fn', 'cl','fp', 'nb_gt', 'nb_other']
        df = pd.DataFrame(index=unit1_ids, columns=columns)
        for u1 in unit1_ids:
            counts = self._mixed_counts['by_spiketrains'][u1]
            df.loc[u1, 'tp'] = counts['TP']
            df.loc[u1, 'fn'] = counts['FP']
            df.loc[u1, 'cl'] = counts['CL']
            df.loc[u1, 'fp'] = counts['TP']
            df.loc[u1, 'nb_gt'] = counts['NB_SPIKE_1']
            df.loc[u1, 'nb_other'] = counts['NB_SPIKE_2']
            
            
        return df


    def get_performance(self, method='by_spiketrain', output='pandas'):
        """
        Compute performance rate with several method:
          * 'raw_count'
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
        possibles = ('raw_count', 'by_spiketrain', 'pooled_with_sum', 'pooled_with_average')
        if method not in possibles:
            raise Exception("'method' can be " + ' or '.join(possibles))

        if self._mixed_counts is None:
            self._do_counting()
        
        if method =='raw_count':
            assert output=='pandas', "Output must be pandas for raw_count"
            
            perf = self.get_raw_count()
            
        if method == 'by_spiketrain':
            assert output=='pandas', "Output must be pandas for by_spiketrain"

            unit1_ids = self._sorting1.get_unit_ids()
            perf = pd.DataFrame(index=unit1_ids, columns=_perf_keys)

            for u1 in unit1_ids:
                counts = self._mixed_counts['by_spiketrains'][u1]

                perf.loc[u1, 'tp_rate'] = counts['TP'] / counts['NB_SPIKE_1']
                perf.loc[u1, 'cl_rate'] = counts['CL'] / counts['NB_SPIKE_1']
                perf.loc[u1, 'fn_rate'] = counts['FN'] / counts['NB_SPIKE_1']
                perf.loc[u1, 'fp_rate_st1'] = counts['FP'] / counts['NB_SPIKE_1']
                if counts['NB_SPIKE_2'] > 0:
                    perf.loc[u1, 'fp_rate_st2'] = counts['FP'] / counts['NB_SPIKE_2']
                else:
                    perf.loc[u1, 'fp_rate_st2'] = np.nan

            perf['accuracy'] = perf['tp_rate'] / (perf['tp_rate'] + perf['fn_rate']+perf['fp_rate_st1'])
            perf['sensitivity'] = perf['tp_rate'] / (perf['tp_rate'] + perf['fn_rate'])
            perf['miss_rate'] = perf['fn_rate'] / (perf['tp_rate'] + perf['fn_rate'])
            perf['precision'] = perf['tp_rate'] / (perf['tp_rate'] + perf['fp_rate_st1'])
            perf['false_discovery_rate'] = perf['fp_rate_st1'] / (perf['tp_rate'] + perf['fp_rate_st1'])

        elif method == 'pooled_with_sum':
            counts = self._mixed_counts['pooled_with_sum']

            tp_rate = float(counts['TP']) / counts['TOT_ST1']
            cl_rate = float(counts['CL']) / counts['TOT_ST1']
            fn_rate = float(counts['FN']) / counts['TOT_ST1']
            fp_rate_st1 = float(counts['FP']) / counts['TOT_ST1']
            if counts['TOT_ST2'] > 0:
                fp_rate_st2 = float(counts['FP']) / counts['TOT_ST2']
                accuracy = tp_rate / (tp_rate + fn_rate + fp_rate_st1)
                sensitivity = tp_rate / (tp_rate + fn_rate)
                miss_rate = fn_rate / (tp_rate + fn_rate)
                precision = tp_rate / (tp_rate + fp_rate_st1)
                false_discovery_rate = fp_rate_st1 / (tp_rate + fp_rate_st1)
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

    
    def get_number_units_above_threshold(self, columns='accuracy', threshold=95, ):
        perf = self.get_performance(method='by_spiketrain', output='pandas')
        nb = (perf[columns] > threshold).sum()
        return nb
    
    
    
# usefull also for gathercomparison
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

    
    


def compare_sorter_to_ground_truth(gt_sorting, other_sorting, gt_name=None, other_name=None, 
                delta_frames=10, min_accuracy=0.5, exhaustive_gt=False, count=False, n_jobs=1, verbose=False):
    '''
    Compares a sorter to a ground truth.

    - Spike trains are matched based on their agreement scores
    - Individual spikes are labelled as true positives (TP), false negatives (FN), false positives 1 (FP from spike
    train 1), false positives 2 (FP from spike train 2), misclassifications (CL)

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
    exhaustive_gt: bool (default False)
        Tell if the ground true is "exhaustive" or not. In other world if the
        GT have all possible units. It allows more performance measurment.
        For instance, MEArec simulated dataset have exhaustive_gt=True
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
    return GroundTruthComparison(gt_sorting, other_sorting, gt_name, other_name,
                            delta_frames, min_accuracy, exhaustive_gt, count, n_jobs, verbose)
