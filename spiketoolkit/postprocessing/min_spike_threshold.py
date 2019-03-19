from spikeextractors import CuratedSortingExtractor

class MinSpikeThreshold(CuratedSortingExtractor):
    def __init__(self, sorting, min_spike_threshold=50):
        CuratedSortingExtractor.__init__(self, parent_sorting=sorting)
        self._sorting = sorting
        self._min_spike_threshold = min_spike_threshold

        units_to_be_excluded = []
        for unit_id in self.getUnitIds():
            spike_train_size = len(self.getUnitSpikeTrain(unit_id))
            if(spike_train_size < self._min_spike_threshold):
                units_to_be_excluded.append(unit_id)
        self.excludeUnits(units_to_be_excluded)

def min_spike_threshold(sorting, min_spike_threshold=50):
    return MinSpikeThreshold(
        sorting=recording,
        min_spike_threshold=min_spike_threshold,
    )
