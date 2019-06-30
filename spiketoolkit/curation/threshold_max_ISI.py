from .CurationSortingExtractor import CurationSortingExtractor
from spiketoolkit.validation.biophysicalmetrics import compute_ISI_violation_ratio

'''
Basic example of a curation module. They can inherit from the
CurationSortingExtractor to allow for excluding, merging, and splitting of units.
'''

class ThresholdMaxISI(CurationSortingExtractor):

    curator_name = 'ThresholdMaxISI'
    installed = True  # check at class level if installed or not
    _gui_params = [
        {'name': 'sampling_frequency', 'type': 'float', 'title': "The sampling frequency of recording"},
        {'name': 'max_ISI_threshold', 'type': 'float', 'value':0.4, 'default':0.4, 'title': "Maximum ISI violation ratio of a unit for it to valid"},
    ]
    installation_mesg = "" # err

    def __init__(self, sorting, sampling_frequency, max_ISI_threshold=0.4):
        CurationSortingExtractor.__init__(self, parent_sorting=sorting)
        self._sorting = sorting
        self._sampling_frequency = sampling_frequency
        self._max_ISI_threshold = max_ISI_threshold

        units_to_be_excluded = []
        isi_ratio_list = compute_ISI_violation_ratio(self._sorting, self._sampling_frequency)
        for i, unit_id in enumerate(self._sorting.get_unit_ids()):
            if isi_ratio_list[i] >= self._max_ISI_threshold:
                units_to_be_excluded.append(unit_id)
        self.exclude_units(units_to_be_excluded)


def threshold_max_ISI(sorting, sampling_frequency, max_ISI_threshold=0.4):
    '''
    Excludes units with ISI ratios greater than or equal to the max_ISI_threshold.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting extractor to be thresholded.
    sampling_frequency: float
        The sampling frequency of recording
    max_ISI_threshold: float
        Maximum ratio between the frequency of spikes present within 0- to 2-ms
        (refractory period) interspike interval (ISI) and those at 0- to 20-ms
        interval.
    Returns
    -------
    thresholded_sorting: ThresholdMaxISI
        The thresholded sorting extractor

    '''
    return ThresholdMaxISI(
        sorting=sorting,
        sampling_frequency=sampling_frequency,
        max_ISI_threshold=max_ISI_threshold,
    )
