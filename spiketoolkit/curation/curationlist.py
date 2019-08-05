from .CurationSortingExtractor import CurationSortingExtractor
from .threshold_num_spikes import threshold_num_spikes, ThresholdNumSpikes
from .threshold_firing_rate import threshold_firing_rate, ThresholdFiringRate
from .threshold_snr import threshold_snr, ThresholdSNR
from .threshold_isi_violations import threshold_isi_violations, ThresholdISIViolations

curation_full_list = [
    ThresholdNumSpikes,
    ThresholdFiringRate,
    ThresholdSNR,
    ThresholdISIViolations
]

installed_curation_list = [c for c in curation_full_list if c.installed]
curation_dict = {c_class.curator_name: c_class for c_class in curation_full_list}
