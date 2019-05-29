from .bandpass_filter import bandpass_filter, BandpassFilterRecording
from .notch_filter import notch_filter, NotchFilterRecording
from .whiten import whiten, WhitenRecording
from .common_reference import common_reference, CommonReferenceRecording
from .resample import resample, ResampledRecording
from .rectify import rectify, RectifyRecording
from .remove_artifacts import remove_artifacts, RemoveArtifactsRecording
from .transform_traces import transform_traces, TransformTracesRecording
from .remove_bad_channels import remove_bad_channels, RemoveBadChannelsRecording
from .normalize_by_quantile import normalize_by_quantile, NormalizeByQuantileRecording

preprocessers_full_list = [
    BandpassFilterRecording,
    NotchFilterRecording,
    WhitenRecording,
    CommonReferenceRecording,
    ResampledRecording,
    RectifyRecording,
    RemoveArtifactsRecording,
    RemoveBadChannelsRecording,
    TransformTracesRecording,
    NormalizeByQuantileRecording
]

installed_preprocessers_list = [pp for pp in preprocessers_full_list if pp.installed]
