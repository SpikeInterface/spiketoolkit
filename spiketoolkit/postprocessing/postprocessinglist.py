from .threshold_min_num_spikes import threshold_min_num_spikes, ThresholdMinNumSpike
from .threshold_min_SNR import threshold_min_SNR, ThresholdMinSNR
from .threshold_max_ISI import threshold_max_ISI, ThresholdMaxISI
from .postprocessing_tools import get_units_waveforms, get_units_templates, get_units_max_channels, compute_units_pca_scores, \
    export_to_phy, set_units_properties_by_max_channels_properties

postprocessors_full_list = [
    ThresholdMinNumSpike,
    ThresholdMinSNR,
    ThresholdMaxISI
]

installed_postprocessors_list = [pp for pp in postprocessors_full_list if pp.installed]
