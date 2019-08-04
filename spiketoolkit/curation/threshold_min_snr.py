from .CurationSortingExtractor import CurationSortingExtractor
import spiketoolkit as st
'''
Basic example of a curation module. They can inherit from the
CurationSortingExtractor to allow for excluding, merging, and splitting of units.
'''

class ThresholdMinSNR(CurationSortingExtractor):

    curator_name = 'ThresholdMinSNR'
    installed = False  # check at class level if installed or not
    _gui_params = [
        {'name': 'min_snr_threshold', 'type': 'float', 'value':5.0, 'default':5.0, 'title': "Minimum snr for which a unit is removed."},
        {'name': 'snr_mode', 'type': 'str', 'value':'mad', 'default':'mad', 'title': "Mode to compute noise SNR ('mad' | 'std' - default 'mad')"},
        {'name': 'snr_noise_duration', 'type': 'float', 'value':10.0, 'default':10.0, 'title': "Number of seconds to compute noise level from (default 10.0)."},
        {'name': 'max_snr_waveforms', 'type': 'float', 'value':1000, 'default':1000, 'title': "Maximum number of waveforms to compute templates from (default 1000)."},
   ]
    installation_mesg = "" # err

    def __init__(self, sorting, recording, min_snr_threshold=5.0, snr_mode='mad', snr_noise_duration=10.0, \
                 max_snr_waveforms=1000, metric_calculator=None):
        CurationSortingExtractor.__init__(self, parent_sorting=sorting)
        self._min_snr_threshold = min_snr_threshold
        if metric_calculator is None:
            self._metric_calculator = st.validation.MetricCalculator(sorting, sampling_frequency=recording.get_sampling_frequency(), \
                                                                     unit_ids=None, epoch_tuples=None, epoch_names=None)
            self._metric_calculator.store_recording(recording)
            self._metric_calculator.compute_snrs(snr_mode, snr_noise_duration, max_snr_waveforms)
        else:
            self._metric_calculator = metric_calculator
            if 'snr' not in self._metric_calculator.get_metrics_dict().keys():
                self._metric_calculator.store_recording(recording)
                self._metric_calculator.compute_snrs(snr_mode, snr_noise_duration, max_snr_waveforms)
        snrs_epochs = self._metric_calculator.get_metrics_dict()['snr'][0] 
        units_to_be_excluded = []
        for i, unit_id in enumerate(sorting.get_unit_ids()):
            if snrs_epochs[i] < min_snr_threshold:
                units_to_be_excluded.append(unit_id)
        self.exclude_units(units_to_be_excluded)


def threshold_min_snr(sorting, recording, min_snr_threshold=5.0, snr_mode='mad', snr_noise_duration=10.0, \
                      max_snr_waveforms=1000, metric_calculator=None):
    '''
    Excludes units with number of spikes less than the given threshold

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting extractor to be thresholded.
    recording: RecordingExtractor
        The recording extractor to compute SNR with.
    min_snr_threshold: float
        The min snr threshold for which a unit is removed from the sorting.
    mode: str
        Mode to compute noise SNR ('mad' | 'std' - default 'mad')
    noise_duration: float
        Number of seconds to compute noise level from (default 10.0)
    max_snr_waveforms: int
        Maximum number of waveforms to compute templates from (default 1000)
    metric_calculator: MetricCalculator
        A metric calculator can be passed in with cached 
    Returns
    -------
    thresholded_sorting: ThresholdMinSNR
        The thresholded sorting extractor

    '''
    return ThresholdMinSNR(
        sorting=sorting, 
        recording=recording,
        min_snr_threshold=min_snr_threshold,
        snr_mode=snr_mode, 
        snr_noise_duration=snr_noise_duration,
        max_snr_waveforms=max_snr_waveforms,
        metric_calculator=metric_calculator
    )
