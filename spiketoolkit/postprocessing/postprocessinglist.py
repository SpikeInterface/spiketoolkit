from .threshold_min_num_spikes import threshold_min_num_spikes, ThresholdMinNumSpike
from .threshold_min_SNR import threshold_min_SNR, ThresholdMinSNR
from .threshold_max_ISI import threshold_max_ISI, ThresholdMaxISI
from .postprocessing_tools import get_unit_waveforms, get_unit_template, get_unit_max_channel, compute_pca_scores, export_to_phy

postprocessors_full_list = [
    ThresholdMinNumSpike,
    ThresholdMinSNR,
    ThresholdMaxISI
]

installed_postprocessors_list = [pp for pp in postprocessors_full_list if pp.installed]
