from .CurationSortingExtractor import CurationSortingExtractor

'''
Basic example of a curation module. They can inherit from the
CurationSortingExtractor to allow for excluding, merging, and splitting of units.
'''

class ThresholdMinNumSpike(CurationSortingExtractor):

    curator_name = 'ThresholdMinNumSpike'
    installed = True  # check at class level if installed or not
    _gui_params = [
        {'name': 'min_num_spike_threshold', 'type': 'int', 'value':50, 'default':50, 'title': "Minimum number of spikes in a unit for it to valid"},
    ]
    installation_mesg = "" # err

    def __init__(self, sorting, min_num_spike_threshold=50):
        CurationSortingExtractor.__init__(self, parent_sorting=sorting)
        self._sorting = sorting
        self._min_num_spike_threshold = min_num_spike_threshold

        units_to_be_excluded = []
        for unit_id in self.get_unit_ids():
            spike_train_size = len(self.get_unit_spike_train(unit_id))
            if(spike_train_size < self._min_num_spike_threshold):
                units_to_be_excluded.append(unit_id)
        self.exclude_units(units_to_be_excluded)

def threshold_min_num_spikes(sorting, min_num_spike_threshold=50):
    '''
    Excludes units with number of spikes less than the min_num_spike_threshold.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting extractor to be thresholded.
    min_num_spike_threshold: int
        Minimum number of spikes in a unit for it to valid

    Returns
    -------
    thresholded_sorting: ThresholdMinNumSpike
        The thresholded sorting extractor

    '''
    return ThresholdMinNumSpike(
        sorting=sorting,
        min_num_spike_threshold=min_num_spike_threshold,
    )
