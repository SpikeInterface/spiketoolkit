from .CurationSortingExtractor import CurationSortingExtractor
import spiketoolkit as st

class ThresholdMaxISIViolations(CurationSortingExtractor):

    curator_name = 'ThresholdMaxISIViolations'
    installed = True  # check at class level if installed or not
    _gui_params = [
        {'name': 'sampling_frequency', 'type': 'float', 'title': "The sampling frequency of recording"},
        {'name': 'max_isi_threshold', 'type': 'float', 'value':5.0, 'default':5.0, 'title': "Maximum ISI violation ratio of a unit for it to valid"},
        {'name': 'isi_threshold', 'type': 'float', 'value':0.0015, 'default':0.0015, 'title': "ISI threshold for calculating violations"},
        {'name': 'min_isi', 'type': 'float', 'value':0.000166, 'default':0.000166, 'title': "Min ISI for calculating violations"},
    ]
    installation_mesg = "" # err

    def __init__(self, sorting, max_isi_threshold=5.0, isi_threshold=0.0015, min_isi=0.000166, \
                 sampling_frequency=None, metric_calculator=None):
        CurationSortingExtractor.__init__(self, parent_sorting=sorting)
        if sampling_frequency is None and sorting.get_sampling_frequency() is None:
            raise ValueError("Please pass in a sampling frequency (your SortingExtractor does not have one specified)")
        elif sampling_frequency is None:
            self._sampling_frequency = sorting.get_sampling_frequency()
        else:
            self._sampling_frequency = sampling_frequency
        self._max_isi_threshold = max_isi_threshold

        if metric_calculator is None:
            self._metric_calculator = st.validation.MetricCalculator(sorting, sampling_frequency=self._sampling_frequency, \
                                                                     unit_ids=None, epoch_tuples=None, epoch_names=None)
            self._metric_calculator.compute_isi_violations(isi_threshold=isi_threshold, min_isi=min_isi)
        else:
            self._metric_calculator = metric_calculator
            if 'isi_viol' not in self._metric_calculator.get_metrics_dict().keys():
                self._metric_calculator.compute_isi_violations(isi_threshold=isi_threshold, min_isi=min_isi)
        isi_violations_epochs = self._metric_calculator.get_metrics_dict()['isi_viol'][0]  
        units_to_be_excluded = []
        for i, unit_id in enumerate(sorting.get_unit_ids()):
            if isi_violations_epochs[i] > max_isi_threshold:
                units_to_be_excluded.append(unit_id)
        self.exclude_units(units_to_be_excluded)


def threshold_max_isi_violations(sorting, max_isi_threshold=5.0, isi_threshold=0.0015, min_isi=0.000166, \
                                 sampling_frequency=None, metric_calculator=None):
    '''
    Excludes units with ISI ratios greater than or equal to the max_ISI_threshold.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting extractor to be thresholded.
    max_isi_threshold: float
        Maximum ratio between the frequency of spikes present within 0- to 2-ms
        (refractory period) interspike interval (ISI) and those at 0- to 20-ms interval.
    isi_threshold: float
            The isi threshold for calculating isi violations.
    min_isi: float
            The minimum expected isi value.
    sampling_frequency: float
        The sampling frequency of recording
    metric_calculator: MetricCalculator
        A metric calculator can be passed in with cached 
    Returns
    -------
    thresholded_sorting: ThresholdMaxISIViolations
        The thresholded sorting extractor

    '''
    return ThresholdMaxISIViolations(
        sorting=sorting, 
        max_isi_threshold=max_isi_threshold, 
        isi_threshold=isi_threshold, 
        min_isi=min_isi, \
        sampling_frequency=sampling_frequency, 
        metric_calculator=metric_calculator
    )
