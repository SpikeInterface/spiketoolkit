from .thresholdcurator import ThresholdCurator
import spiketoolkit as st


#No sampling frequency needed, special curator that does NOT use the metric calculator.

class ThresholdNumSpikes(ThresholdCurator):
    curator_name = 'ThresholdNumSpikes'
    installed = True  # check at class level if installed or not
    curator_gui_params = [
        {'name': 'threshold', 'type': 'float', 'title':
            "The threshold for the given metric."},
        {'name': 'threshold_sign', 'type': 'str',
         'title': "If 'less', will threshold any metric less than the given threshold. "
                  "If 'less_or_equal', will threshold any metric less than or equal to the given threshold. "
                  "If 'greater', will threshold any metric greater than the given threshold. "
                  "If 'greater_or_equal', will threshold any metric greater than or equal to the given threshold."},
    ]
    installation_mesg = ""  # err

    def __init__(self, sorting, threshold, threshold_sign, metric_calculator=None):
        metric_name = 'num_spikes'
        if metric_calculator is not None:
            self._metric_calculator = metric_calculator
            if metric_name not in self._metric_calculator.get_metrics_dict().keys():
                self._metric_calculator.compute_num_spikes()
            num_spikes_epoch = self._metric_calculator.get_metrics_dict()[metric_name][0]
        else:
            num_spikes_epoch = [len(sorting.get_unit_spike_train(unit_id)) for unit_id in sorting.get_unit_ids()]
        ThresholdCurator.__init__(self, sorting=sorting, metrics_epoch=num_spikes_epoch)
        self.threshold_sorting(threshold=threshold, threshold_sign=threshold_sign)


def threshold_num_spikes(sorting, threshold=100, threshold_sign='less', metric_calculator=None):
    '''
    Excludes units based on number of spikes.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting extractor to be thresholded.
    threshold:
        The threshold for the given metric.
    threshold_sign: str
        If 'less', will remove any units with metric scores less than the given threshold.
        If 'less_or_equal', will remove any units with metric scores less than or equal to the given threshold.
        If 'greater', will remove any units with metric scores greater than the given threshold.
        If 'greater_or_equal', will remove any units with metric scores greater than or equal to the given threshold.
    metric_calculator: MetricCalculator
        A metric calculator can be passed in with cached metrics.
    Returns
    -------
    thresholded_sorting: ThresholdNumSpikes
        The thresholded sorting extractor

    '''
    return ThresholdNumSpikes(
        sorting=sorting,
        threshold=threshold,
        threshold_sign=threshold_sign,
        metric_calculator=metric_calculator
    )
