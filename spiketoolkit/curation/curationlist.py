from .threshold_min_num_spikes import threshold_min_num_spikes, ThresholdMinNumSpike
from .threshold_min_SNR import threshold_min_SNR, ThresholdMinSNR
from .threshold_max_ISI import threshold_max_ISI, ThresholdMaxISI
from .CurationSortingExtractor import CurationSortingExtractor


curation_full_list = [
    ThresholdMinNumSpike,
    ThresholdMinSNR,
    ThresholdMaxISI
]

installed_curation_list = [c for c in curation_full_list if c.installed]
curation_dict = {c_class.curator_name: c_class for c_class in curation_full_list}
