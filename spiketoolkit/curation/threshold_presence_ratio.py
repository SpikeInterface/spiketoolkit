from .thresholdcurator import ThresholdCurator
import spiketoolkit as st


class ThresholdPresenceRatio(ThresholdCurator):
    curator_name = 'ThresholdPresenceRatio'
    installed = True  # check at class level if installed or not
    curator_gui_params = [
        {'name': 'threshold', 'type': 'float', 'value': .50, 'default': .50, 'title':
            "The threshold for the given metric."},
        {'name': 'threshold_sign', 'type': 'str', 'value': 'less', 'default': 'less',
         'title': "If 'less', will threshold any metric less than the given threshold. "
                  "If 'greater', will threshold any metric greater than the given threshold."},
    ]
    installation_mesg = ""  # err

    def __init__(self, sorting, threshold=.50, threshold_sign='less', sampling_frequency=None, metric_calculator=None):
        metric_name = 'presence_ratio'
        if sampling_frequency is None and sorting.get_sampling_frequency() is None:
            raise ValueError("Please pass in a sampling frequency (your SortingExtractor does not have one specified)")
        elif sampling_frequency is None:
            self._sampling_frequency = sorting.get_sampling_frequency()
        else:
            self._sampling_frequency = sampling_frequency
        if metric_calculator is None:
            self._metric_calculator = st.validation.MetricCalculator(sorting,
                                                                     sampling_frequency=self._sampling_frequency,
                                                                     unit_ids=None, epoch_tuples=None, epoch_names=None)
            self._metric_calculator.compute_presence_ratios()
        else:
            self._metric_calculator = metric_calculator
            if metric_name not in self._metric_calculator.get_metrics_dict().keys():
                self._metric_calculator.compute_presence_ratios()
        presence_ratios_epoch = self._metric_calculator.get_metrics_dict()[metric_name][0]

        ThresholdCurator.__init__(self, sorting=sorting, metrics_epoch=presence_ratios_epoch)
        self.threshold_sorting(threshold=threshold, threshold_sign=threshold_sign)


def threshold_presence_ratio(sorting, threshold=.50, threshold_sign='less', sampling_frequency=None,
                             metric_calculator=None):
    '''
    Excludes units based on presence ratios.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting extractor to be thresholded.
    threshold:
        The threshold for the given metric.
    threshold_sign: str
        If 'less', will threshold any metric less than the given threshold.
        If 'less_or_equal', will threshold any metric less than or equal to the given threshold.
        If 'greater', will threshold any metric greater than the given threshold.
        If 'greater_or_equal', will threshold any metric greater than or equal to the given threshold.
    sampling_frequency: float
        The sampling frequency of recording
    metric_calculator: MetricCalculator
        A metric calculator can be passed in with cached metrics.
    Returns
    -------
    thresholded_sorting: ThresholdPresenceRatio
        The thresholded sorting extractor

    '''
    return ThresholdPresenceRatio(
        sorting=sorting,
        threshold=threshold,
        threshold_sign=threshold_sign,
        sampling_frequency=sampling_frequency,
        metric_calculator=metric_calculator
    )
