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
