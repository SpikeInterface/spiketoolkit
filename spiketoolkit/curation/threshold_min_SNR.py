from .CurationSortingExtractor import CurationSortingExtractor
from spiketoolkit.validation.qualitymetrics import compute_unit_SNR

'''
Basic example of a curation module. They can inherit from the
CurationSortingExtractor to allow for excluding, merging, and splitting of units.
'''

class ThresholdMinSNR(CurationSortingExtractor):

    curator_name = 'ThresholdMinSNR'
    installed = False  # check at class level if installed or not
    _gui_params = [
        {'name': 'min_SNR_threshold', 'type': 'float', 'value':4.0, 'default':4.0, 'title': "Minimum SNR of a unit for it to valid"},
        {'name': 'seconds', 'type': 'int', 'value':10, 'default':10, 'title': "Number of seconds to compute SNR"},
        {'name': 'max_num_waveforms', 'type': 'int', 'value':1000, 'default':1000, 'title': "Max number of waveforms to compute SNR"},
        {'name': 'apply_filter', 'type': 'bool', 'value':False, 'default':False, 'title': "If true, applies bandpass filter to the data"},
        {'name': 'freq_min', 'type': 'float', 'value':300.0, 'default':300.0, 'title': "Minimum bandpass filter frequency"},
        {'name': 'freq_max', 'type': 'float', 'value':6000.0, 'default':6000.0, 'title': "Maximum bandpass filter frequency"},
    ]
    installation_mesg = "" # err

    def __init__(self, recording, sorting, min_SNR_threshold, mode, seconds, 
                 max_num_waveforms, apply_filter, freq_min, freq_max):
        CurationSortingExtractor.__init__(self, parent_sorting=sorting)
        self._recording = recording
        self._sorting = sorting
        self._min_SNR_threshold = min_SNR_threshold

        units_to_be_excluded = []
        snr_list = compute_unit_SNR(self._recording, self._sorting, mode=mode, 
                                    seconds=seconds, 
                                    max_num_waveforms=max_num_waveforms, 
                                    apply_filter=apply_filter, 
                                    freq_min=freq_min, freq_max=freq_max)
        for i, unit_id in enumerate(self._sorting.get_unit_ids()):
            if snr_list[i] < self._min_SNR_threshold:
                units_to_be_excluded.append(unit_id)
        self.exclude_units(units_to_be_excluded)


def threshold_min_SNR(recording, sorting, min_SNR_threshold=4.0, mode='mad',
                      seconds=10, max_num_waveforms=1000, apply_filter=False, 
                      freq_min=300, freq_max=6000):
    '''
    Excludes units with SNR less than the min_SNR_threshold.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor
    sorting: SortingExtractor
        The sorting extractor
    min_SNR_threshold: Float
        Minimum SNR of a unit for it to valid
    mode: str
        Mode to compute noise SNR ('mad' | 'std' - default 'mad')
    seconds: float
        Number of seconds to compute noise level from (default 10)
    max_num_waveforms: int
        Maximum number of waveforms to cpmpute templates from (default 1000)
    apply_filter: bool
        If True, recording is filtered before computing noise level
    freq_min: float
        High-pass frequency for optional filter (default 300 Hz)
    freq_max: float
        Low-pass frequency for optional filter (default 6000 Hz)

    Returns
    -------
    thresholded_sorting: ThresholdMinSNR
        The thresholded sorting extractor

    '''
    return ThresholdMinSNR(
        recording=recording,
        sorting=sorting,
        min_SNR_threshold=min_SNR_threshold,
        mode=mode,
        seconds=seconds,
        max_num_waveforms=max_num_waveforms,
        apply_filter=apply_filter,
        freq_min=freq_min,
        freq_max=freq_max 
    )
