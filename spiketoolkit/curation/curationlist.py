from .curationsortingextractor import CurationSortingExtractor
from .threshold_num_spikes import threshold_num_spikes, ThresholdNumSpikes
from .threshold_firing_rate import threshold_firing_rate, ThresholdFiringRate
from .threshold_presence_ratio import threshold_presence_ratio, ThresholdPresenceRatio
from .threshold_isi_violations import threshold_isi_violations, ThresholdISIViolations
from .threshold_snr import threshold_snr, ThresholdSNR
from .threshold_silhouette_score import (
    threshold_silhouette_score,
    ThresholdSilhouetteScore,
)
from .threshold_d_primes import threshold_d_primes, ThresholdDPrimes
from .threshold_l_ratios import threshold_l_ratios, ThresholdLRatios
from .threshold_amplitude_cutoff import (
    threshold_amplitude_cutoff,
    ThresholdAmplitudeCutoff,
)

curation_full_list = [
    ThresholdNumSpikes,
    ThresholdFiringRate,
    ThresholdPresenceRatio,
    ThresholdISIViolations,
    ThresholdSNR,
    ThresholdSilhouetteScore,
    ThresholdDPrimes,
    ThresholdLRatios,
    ThresholdAmplitudeCutoff,
]

installed_curation_list = [c for c in curation_full_list if c.installed]
curation_dict = {c_class.curator_name: c_class for c_class in curation_full_list}
