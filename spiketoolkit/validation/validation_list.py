from .quality_metric_classes.metric_data import MetricData
from .quality_metric_classes.num_spikes import NumSpikes
from .quality_metric_classes.amplitude_cutoff import AmplitudeCutoff
from .quality_metric_classes.silhouette_score import SilhouetteScore
from .quality_metric_classes.d_prime import DPrime
from .quality_metric_classes.l_ratio import LRatio
from .quality_metric_classes.isolation_distance import IsolationDistance
from .quality_metric_classes.firing_rate import FiringRate
from .quality_metric_classes.presence_ratio import PresenceRatio
from .quality_metric_classes.isi_violation import ISIViolation
from .quality_metric_classes.snr import SNR
from .quality_metric_classes.nearest_neighbor import NearestNeighbor
from .quality_metric_classes.drift_metric import DriftMetric

from .quality_metrics import (
    compute_num_spikes,
    compute_firing_rates,
    compute_presence_ratios,
    compute_isi_violations,
    compute_amplitude_cutoffs,
    compute_snrs,
    compute_drift_metrics,
    compute_silhouette_scores,
    compute_isolation_distances,
    compute_l_ratios,
    compute_d_primes,
    compute_nn_metrics,
    compute_metrics,
)

from .quality_metric_classes.utils.validation_tools import (
    get_all_metric_data,
    get_pca_metric_data,
    get_amplitude_metric_data,
    get_spike_times_metrics_data,
)

from .quality_metric_classes.parameter_dictionaries import (
    get_recording_params,
    get_amplitude_params,
    get_pca_scores_params,
    get_epoch_params,
    get_feature_params,
    update_param_dicts,
)

curation_full_list = [
    NumSpikes,
    FiringRate,
    PresenceRatio,
    ISIViolation,
    SNR,
    AmplitudeCutoff,
    DriftMetric,
    SilhouetteScore,
    DPrime,
    LRatio,
    IsolationDistance,
    NearestNeighbor,
]

installed_curation_list = [c for c in curation_full_list if c.installed]
curation_dict = {c_class.curator_name: c_class for c_class in curation_full_list}
