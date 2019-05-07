from spikeextractors import CurationSortingExtractor
from spiketoolkit.validation.qualitymetrics import compute_unit_SNR

'''
Basic example of a postprocessing module. They can inherit from the
CurationSortingExtractor to allow for excluding, merging, and splitting of units.
'''

class ThresholdMinSNR(CurationSortingExtractor):

    preprocessor_name = 'ThresholdMinSNR'
    installed = True  # check at class level if installed or not
    _gui_params = [
        {'name': 'recording', 'type': 'RecordingExtractor', 'title': "Recording extractor"},
        {'name': 'sorting', 'type': 'SortingExtractor', 'title': "Sorting extractor"},
        {'name': 'min_SNR_threshold', 'type': 'float', 'value':4.0, 'default':4.0, 'title': "Minimum SNR of a unit for it to valid"},
    ]
    installation_mesg = "" # err

    def __init__(self, recording, sorting, min_SNR_threshold=4.0):
        CurationSortingExtractor.__init__(self, parent_sorting=sorting)
        self._recording = recording
        self._sorting = sorting
        self._min_SNR_threshold = min_SNR_threshold

        units_to_be_excluded = []
        snr_list = compute_unit_SNR(self._recording, self._sorting)
        for i, unit_id in enumerate(self._sorting.get_unit_ids()):
            if snr_list[i] < self._min_SNR_threshold:
                units_to_be_excluded.append(unit_id)
        self.exclude_units(units_to_be_excluded)


def threshold_min_SNR(recording, sorting, min_SNR_threshold=4.0):
    '''
    Excludes units with SNR less than the min_SNR_threshold.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor associated with the sorting extractor to be thresholded.
    sorting: SortingExtractor
        The sorting extractor to be thresholded.
    min_SNR_threshold: Float
        Minimum SNR of a unit for it to valid

    Returns
    -------
    thresholded_sorting: ThresholdMinSNR
        The thresholded sorting extractor

    '''
    return ThresholdMinSNR(
        recording=recording,
        sorting=sorting,
        min_SNR_threshold=min_SNR_threshold,
    )
