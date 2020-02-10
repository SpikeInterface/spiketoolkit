from .curationsortingextractor import CurationSortingExtractor
from .threshold_num_spikes import threshold_num_spikes, ThresholdNumSpikes
from .threshold_firing_rates import threshold_firing_rates, ThresholdFiringRates
from .threshold_presence_ratios import threshold_presence_ratios, ThresholdPresenceRatios
from .threshold_isi_violations import threshold_isi_violations, ThresholdISIViolations
from .threshold_snrs import threshold_snrs, ThresholdSNRs
from .threshold_silhouette_scores import (
    threshold_silhouette_scores,
    ThresholdSilhouetteScores,
)
from .threshold_d_primes import threshold_d_primes, ThresholdDPrimes
from .threshold_l_ratios import threshold_l_ratios, ThresholdLRatios
from .threshold_amplitude_cutoffs import (
    threshold_amplitude_cutoffs,
    ThresholdAmplitudeCutoffs,
)

curation_full_list = [
    ThresholdNumSpikes,
    ThresholdFiringRates,
    ThresholdPresenceRatios,
    ThresholdISIViolations,
    ThresholdSNRs,
    ThresholdSilhouetteScores,
    ThresholdDPrimes,
    ThresholdLRatios,
    ThresholdAmplitudeCutoffs,
]

installed_curation_list = [c for c in curation_full_list if c.installed]
curation_dict = {c_class.curator_name: c_class for c_class in curation_full_list}
