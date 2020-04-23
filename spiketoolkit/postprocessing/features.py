"""
Uses the functions in SpikeInterface/spikefeatures to compute
unit template features
"""

import pandas
import spikefeatures as sf
from scipy.signal import resample_poly
from .postprocessing_tools import get_unit_templates, get_unit_max_channels
from .utils import update_all_param_dicts_with_kwargs, select_max_channels_from_templates
import numpy as np


def get_quality_metrics_list():
    return features.all_1D_features


def compute_unit_template_features(recording, sorting, unit_ids=None, channel_ids=None, feature_names=None,
                                   max_channels_per_features=1, recovery_slope_window=0.7, upsampling_factor=1,
                                   invert_waveforms=False, as_dataframe=False, **kwargs):
    """
    Use SpikeInterface/spikefeatures to compute features for the unit template.

    These consist of a set of 1D features:
        - peak to valley (peak_to_valley), time between peak and valley
        - halfwidth (halfwidth), width of peak at half its amplitude
        - peak trough ratio (peak_trough_ratio), amplitude of peak over amplitude of trough
        - repolarization slope (repolarization_slope), slope between trough and return to base
        - recovery slope (recovery_slope), slope after peak towards baseline

    And 2D features:
        - unit_spread
        - propagation velocity
        To be implemented

    The metrics are computed on 'negative' waveforms, if templates are saved as
    positive, pass keyword 'invert_waveforms'.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor
    sorting: SortingExtractor
        The sorting extractor
    unit_ids: list
        List of unit ids to compute features
    channel_ids: list
        List of channels ids to compute templates on which features are computed
    feature_names: list
        List of feature names to be computed. If None, all features are computed
    max_channels_per_features: int
        Maximum number of channels to compute features on (default 1). If channel_ids is used, this parameter
        is ignored
    upsampling_factor: int
        Factor with which to upsample the template resolution (default 1)
    invert_waveforms: bool
        Invert templates before computing features (default False)
    recovery_slope_window: float
        Window after peak in ms wherein to compute recovery slope (default 0.7)
    as_dataframe: bool
        IfTrue, output is returned as a pandas dataframe, otherwise as a dictionary
    **kwargs: Keyword arguments
        A dictionary with default values can be retrieved with:
        st.postprocessing.get_waveforms_params():
            grouping_property: str
                Property to group channels. E.g. if the recording extractor has the 'group' property and
                'grouping_property' is 'group', then waveforms are computed group-wise.
            ms_before: float
                Time period in ms to cut waveforms before the spike events
            ms_after: float
                Time period in ms to cut waveforms after the spike events
            dtype: dtype
                The numpy dtype of the waveforms
            compute_property_from_recording: bool
                If True and 'grouping_property' is given, the property of each unit is assigned as the corresponding
                property of the recording extractor channel on which the average waveform is the largest
            max_channels_per_waveforms: int or None
                Maximum channels per waveforms to return. If None, all channels are returned
            n_jobs: int
                Number of parallel jobs (default 1)
            max_spikes_per_unit: int
                The maximum number of spikes to extract per unit
            memmap: bool
                If True, waveforms are saved as memmap object (recommended for long recordings with many channels)
            seed: int
                Random seed for extracting random waveforms
            save_property_or_features: bool
                If True (default), waveforms are saved as features of the sorting extractor object
            recompute_info: bool
                If True, waveforms are recomputed (default False)
            verbose: bool
                If True output is verbose


    Returns
    -------
    features: dict or pandas.DataFrame
        The computed features as a dictionary or a pandas.DataFrame (if as_dataframe is True)
    """

    # ------------------- SETUP ------------------------------
    if isinstance(unit_ids, (int, np.integer)):
        unit_ids = [unit_ids]
    elif unit_ids is None:
        unit_ids = sorting.get_unit_ids()
    elif not isinstance(unit_ids, (list, np.ndarray)):
        raise Exception("unit_ids is is invalid")
    if isinstance(channel_ids, (int, np.integer)):
        channel_ids = [channel_ids]

    if channel_ids is None:
        channel_ids = recording.get_channel_ids()

    assert np.all([u in sorting.get_unit_ids() for u in unit_ids]), "Invalid unit_ids"
    assert np.all([ch in recording.get_channel_ids() for ch in channel_ids]), "Invalid channel_ids"

    params_dict = update_all_param_dicts_with_kwargs(kwargs)
    save_property_or_features = params_dict['save_property_or_features']

    if feature_names is None:
        feature_names = sf.all_1D_features
    else:
        bad_features = []
        for m in feature_names:
            if m not in sf.all_1D_features:
                bad_features.append(m)
        if len(bad_features) > 0:
            raise ValueError(f"Improper feature names: {str(bad_features)}. The following features names can be "
                             f"calculated: {str(sf.all_1D_features)}")

    templates = np.array(get_unit_templates(recording, sorting, unit_ids=unit_ids, channel_ids=channel_ids,
                                            mode='median', **kwargs))

    # -------------------- PROCESS TEMPLATES -----------------------------
    if upsampling_factor > 1:
        upsampling_factor = int(upsampling_factor)
        processed_templates = resample_poly(templates, up=upsampling_factor, down=1, axis=2)
        resampled_fs = recording.get_sampling_frequency() * upsampling_factor
    else:
        processed_templates = templates
        resampled_fs = recording.get_sampling_frequency()

    if invert_waveforms:
        processed_templates = -processed_templates

    features_dict = dict()
    for feat in feature_names:
        features_dict[feat] = []
    # --------------------- COMPUTE FEATURES ------------------------------
    for unit_id, unit in enumerate(unit_ids):
        template = processed_templates[unit_id]
        max_channel_idxs = select_max_channels_from_templates(template, recording, max_channels_per_features)
        template_channels = template[max_channel_idxs]
        if len(template_channels.shape) == 1:
            template_channels = template_channels[np.newaxis, :]
        feat_list = sf.calculate_features(waveforms=template_channels,
                                          sampling_frequency=resampled_fs,
                                          feature_names=feature_names,
                                          recovery_slope_window=recovery_slope_window)

        for feat, feat_val in feat_list.items():
            features_dict[feat].append(feat_val)

    # ---------------------- DEAL WITH OUTPUT -------------------------
    if save_property_or_features:
        for feat_name, feat_val in features_dict.items():
            sorting.set_units_property(unit_ids=unit_ids,
                                       property_name=feat_name,
                                       values=feat_val)
    if as_dataframe:
        features = pandas.DataFrame.from_dict(features_dict)
        features = features.rename(index={original_idx: unit_ids[i] for
                                          i, original_idx in enumerate(range(len(features)))})
    else:
        features = features_dict
    return features
