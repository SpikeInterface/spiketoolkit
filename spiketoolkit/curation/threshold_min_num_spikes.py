from .ThresholdCurator import ThresholdCurator
import spiketoolkit as st
'''
Basic example of a curation module. They can inherit from the
CurationSortingExtractor to allow for excluding, merging, and splitting of units.
'''

class ThresholdMinNumSpikes(ThresholdCurator):

    curator_name = 'ThresholdMinNumSpikes'
    installed = True  # check at class level if installed or not
    _gui_params = [
        {'name': 'sampling_frequency', 'type': 'float', 'title': "The sampling frequency of recording"},
        {'name': 'threshold', 'type': 'float', 'value':5.0, 'default':5.0, 'title': "The threshold for the given metric."},
        {'name': 'threshold_sign', 'type': 'str', 'value':'greater', 'default':'greater', 'title': "If 'less', will threshold any metric less than the given threshold. If 'greater', will threshold any metric greater than the given threshold."},
    ]
    installation_mesg = "" # err

    def __init__(self, sorting, threshold=50, threshold_sign='less', sampling_frequency=None, metric_calculator=None):
        if sampling_frequency is None and sorting.get_sampling_frequency() is None:
            raise ValueError("Please pass in a sampling frequency (your SortingExtractor does not have one specified)")
        elif sampling_frequency is None:
            self._sampling_frequency = sorting.get_sampling_frequency()
        else:
            self._sampling_frequency = sampling_frequency
        if metric_calculator is None:
            self._metric_calculator = st.validation.MetricCalculator(sorting, sampling_frequency=self._sampling_frequency, \
                                                                     unit_ids=None, epoch_tuples=None, epoch_names=None)
            self._metric_calculator.compute_num_spikes()
        else:
            self._metric_calculator = metric_calculator
            if 'num_spikes' not in self._metric_calculator.get_metrics_dict().keys():
                self._metric_calculator.compute_num_spikes()
        num_spikes_epochs = self._metric_calculator.get_metrics_dict()['num_spikes'][0] 

        ThresholdCurator.__init__(self, sorting=sorting, metrics_epoch=num_spikes_epochs)
        self.threshold_sorting(threshold=threshold, threshold_sign=threshold_sign)


def threshold_min_num_spikes(sorting, threshold=50, threshold_sign='less', sampling_frequency=None, metric_calculator=None):
    '''
    Excludes units with number of spikes less than the given threshold

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting extractor to be thresholded.
    threshold:
        The threshold for the given metric.
    threshold_sign: str
        If 'less', will threshold any metric less than the given threshold.
        If 'greater', will threshold any metric greater than the given threshold.
    sampling_frequency: float
        The sampling frequency of recording
    metric_calculator: MetricCalculator
        A metric calculator can be passed in with cached metrics.
    Returns
    -------
    thresholded_sorting: ThresholdMinNumSpikes
        The thresholded sorting extractor

    '''
    return ThresholdMinNumSpikes(
        sorting=sorting, 
        threshold=threshold, 
        threshold_sign=threshold_sign, 
        sampling_frequency=sampling_frequency,
        metric_calculator=metric_calculator
    )
