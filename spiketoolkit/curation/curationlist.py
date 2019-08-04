from .threshold_min_num_spikes import threshold_min_num_spikes, ThresholdMinNumSpikes
from .threshold_min_snr import threshold_min_snr, ThresholdMinSNR
from .threshold_max_isi_violations import threshold_max_isi_violations, ThresholdMaxISIViolations
from .CurationSortingExtractor import CurationSortingExtractor


curation_full_list = [
    ThresholdMinNumSpikes,
    ThresholdMinSNR,
    ThresholdMaxISIViolations
]

installed_curation_list = [c for c in curation_full_list if c.installed]
curation_dict = {c_class.curator_name: c_class for c_class in curation_full_list}
