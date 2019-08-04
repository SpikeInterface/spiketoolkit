from .CurationSortingExtractor import CurationSortingExtractor
import spiketoolkit as st
'''
Basic example of a curation module. They can inherit from the
CurationSortingExtractor to allow for excluding, merging, and splitting of units.
'''

class ThresholdMinNumSpikes(CurationSortingExtractor):

    curator_name = 'ThresholdMinNumSpikes'
    installed = True  # check at class level if installed or not
    _gui_params = [
        {'name': 'sampling_frequency', 'type': 'float', 'title': "The sampling frequency of recording"},
        {'name': 'min_spike_threshold', 'type': 'int', 'value':50, 'default':50, 'title': "Minimum number of spikes for which a unit is removed."},
    ]
    installation_mesg = "" # err

    def __init__(self, sorting, min_spike_threshold=50, sampling_frequency=None, metric_calculator=None):
        CurationSortingExtractor.__init__(self, parent_sorting=sorting)
        if sampling_frequency is None and sorting.get_sampling_frequency() is None:
            raise ValueError("Please pass in a sampling frequency (your SortingExtractor does not have one specified)")
        elif sampling_frequency is None:
            self._sampling_frequency = sorting.get_sampling_frequency()
        else:
            self._sampling_frequency = sampling_frequency
        self._min_spike_threshold = min_spike_threshold
        if metric_calculator is None:
            self._metric_calculator = st.validation.MetricCalculator(sorting, sampling_frequency=self._sampling_frequency, \
                                                                     unit_ids=None, epoch_tuples=None, epoch_names=None)
            self._metric_calculator.compute_num_spikes()
        else:
            self._metric_calculator = metric_calculator
            if 'num_spikes' not in self._metric_calculator.get_metrics_dict().keys():
                self._metric_calculator.compute_num_spikes()
        num_spikes_epochs = self._metric_calculator.get_metrics_dict()['num_spikes'][0] 
        units_to_be_excluded = []
        for i, unit_id in enumerate(sorting.get_unit_ids()):
            if num_spikes_epochs[i] < min_spike_threshold:
                units_to_be_excluded.append(unit_id)
        self.exclude_units(units_to_be_excluded)


def threshold_min_num_spikes(sorting, min_spike_threshold=50, sampling_frequency=None, metric_calculator=None):
    '''
    Excludes units with number of spikes less than the given threshold

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting extractor to be thresholded.
    min_spike_threshold: int
        Minimum spikes for which a unit is removed from the sorting result.
    sampling_frequency: float
        The sampling frequency of recording
    metric_calculator: MetricCalculator
        A metric calculator can be passed in with cached 
    Returns
    -------
    thresholded_sorting: ThresholdMinNumSpikes
        The thresholded sorting extractor

    '''
    return ThresholdMinNumSpikes(
        sorting=sorting, 
        min_spike_threshold=min_spike_threshold, 
        sampling_frequency=sampling_frequency,
        metric_calculator=metric_calculator
    )
