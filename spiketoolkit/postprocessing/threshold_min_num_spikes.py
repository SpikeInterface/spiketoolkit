from spikeextractors import CurationSortingExtractor

'''
Basic example of a postprocessing module. They can inherit from the
CurationSortingExtractor to allow for excluding, merging, and splitting of units.
'''

class ThresholdMinNumSpike(CurationSortingExtractor):
    '''A SortingExtractor that automatically excludes units with num spikes
    less than the minimum num spikethreshold number specified by the user.


    '''
    def __init__(self, sorting, min_num_spike_threshold=50):
        CurationSortingExtractor.__init__(self, parent_sorting=sorting)
        self._sorting = sorting
        self._min_num_spike_threshold = min_num_spike_threshold

        units_to_be_excluded = []
        for unit_id in self.getUnitIds():
            spike_train_size = len(self.getUnitSpikeTrain(unit_id))
            if(spike_train_size < self._min_num_spike_threshold):
                units_to_be_excluded.append(unit_id)
        self.excludeUnits(units_to_be_excluded)

def threshold_min_num_spikes(sorting, min_num_spike_threshold=50):
    return ThresholdMinNumSpike(
        sorting=recording,
        min_num_spike_threshold=min_num_spike_threshold,
    )
