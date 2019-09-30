from .thresholdcurator import ThresholdCurator
import spiketoolkit as st


class ThresholdISIViolations(ThresholdCurator):
    curator_name = 'ThresholdISIViolations'
    installed = True  # check at class level if installed or not
    curator_gui_params = [
        {'name': 'threshold', 'type': 'float', 'value': 5.0, 'default': 5.0,
         'title': "The threshold for the given metric."},
        {'name': 'threshold_sign', 'type': 'str', 'value': 'greater', 'default': 'greater',
         'title': "If 'less', will threshold any metric less than the given threshold. "
                  "If 'greater', will threshold any metric greater than the given threshold."},
        {'name': 'isi_threshold', 'type': 'float', 'value': 0.0015, 'default': 0.0015,
         'title': "ISI threshold for calculating violations"},
        {'name': 'min_isi', 'type': 'float', 'value': 0.000166, 'default': 0.000166,
         'title': "Min ISI for calculating violations"},
    ]
    installation_mesg = ""  # err

    def __init__(self, sorting, threshold=5.0, threshold_sign='greater', isi_threshold=0.0015, min_isi=0.000166, \
                 sampling_frequency=None, metric_calculator=None):
        metric_name = 'isi_viol'
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
            self._metric_calculator.compute_isi_violations(isi_threshold=isi_threshold, min_isi=min_isi)
        else:
            self._metric_calculator = metric_calculator
            if metric_name not in self._metric_calculator.get_metrics_dict().keys():
                self._metric_calculator.compute_isi_violations(isi_threshold=isi_threshold, min_isi=min_isi)
        isi_violations_epoch = self._metric_calculator.get_metrics_dict()[metric_name][0]

        ThresholdCurator.__init__(self, sorting=sorting, metrics_epoch=isi_violations_epoch)
        self.threshold_sorting(threshold=threshold, threshold_sign=threshold_sign)


def threshold_isi_violations(sorting, threshold=5.0, threshold_sign='greater', isi_threshold=0.0015, min_isi=0.000166,
                             sampling_frequency=None, metric_calculator=None):
    '''
    Excludes units based on isi violations.

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
    isi_threshold: float
            The isi threshold for calculating isi violations.
    min_isi: float
            The minimum expected isi value.
    sampling_frequency: float
        The sampling frequency of recording
    metric_calculator: MetricCalculator
        A metric calculator can be passed in with cached metrics.
    Returns
    -------
    thresholded_sorting: ThresholdISIViolations
        The thresholded sorting extractor

    '''
    return ThresholdISIViolations(
        sorting=sorting,
        threshold=threshold,
        threshold_sign=threshold_sign,
        isi_threshold=isi_threshold,
        min_isi=min_isi,
        sampling_frequency=sampling_frequency,
        metric_calculator=metric_calculator
    )
