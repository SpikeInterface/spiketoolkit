"""
Uses the functions in SpikeInterface/spikefeatures to compute
unit template features
"""

import pandas as pd
from spikefeatures import features
from scipy.signal import resample
from .postprocessing_tools import get_unit_templates, get_unit_max_channels


def compute_unit_template_features(
        sorting,
        recording,
        unit_ids=None,
        feature_names=None,
        save_property_or_features=False,
        recovery_slope_window=0.7,
        upsampling_factor=1,
        invert_waveforms=False,
        spread_threshold=None,
        site_range=None,
        site_spacing=None,
        epoch_name=None,
        channel_map=None,
        return_dict=False,
        verbose=False,

):
    """
    Use SpikeInterface/spikefeatures to compute features for the unit
    template. These consist of a set of 1D features:
    - peak to valley (peak_to_valley), time between peak and valley
    - halfwidth (halfwidth), width of peak at half its amplitude
    - peak trough ratio (peak_trough_ratio), amplitude of peak over amplitude of trough
    - repolarization slope (repolarization_slope), slope between trough and return to base
    - recovery slope (recovery_slope), slope after peak towards baseline

    And 2D features:

    The metrics are computed on 'negative' waveforms, if templates are saved as
    positive, pass keyword 'invert_waveforms'.

    Recording should be bandpassfiltered  # TODO: make this a requirement?

    Parameters:
    -------
    sorting: SortingExtractor
        The sorting result to be evaluated.
    recording: RecordingExtractor
        The given recording extractor from which to extract amplitudes
    unit_ids: list
        List of unit ids to compute metric for. If not specified, all units are used
    return_dict: bool
        If True, will return dict of metrics (default False)
    feature_names: list
        list of feature names to be computed
    upsampling_factor : int
        factor with which to upsample the template resolution (default 1)

    TODO:
    save_property_or_features=False,
    recovery_slope_window=0.7,
    invert_waveforms=False,
    spread_threshold=None,
    site_range=None,
    site_spacing=None,
    epoch_name=None,
    channel_map=None,
    verbose=False,

    Returns

    -------
    metrics : pandas.DataFrame
        table containing all metrics:
            rows: unit_ids
            columns: metrics

    OR
    TODO
    """

    # ------------------- SETUP ------------------------------
    if unit_ids is None:
        unit_ids = sorting.get_unit_ids()

    max_channels = get_unit_max_channels(
        sorting=sorting,
        recording=recording,
        unit_ids=unit_ids,
    )

    if feature_names is None:
        feature_names = features.all_1D_features

    if 'template' not in sorting.get_shared_unit_property_names():
        all_templates = get_unit_templates(
            sorting=sorting,
            recording=recording,
            save_property_or_features=save_property_or_features,
            unit_ids=unit_ids,
            mode='mean',
            verbose=verbose,
        )
    else:
        all_templates = [sorting.get_unit_property(uid, 'template') for uid in unit_ids]


    # --------------------- COMPUTE FEATURES ------------------------------
    all_template_features = pd.DataFrame(columns=feature_names)
    for unit_index, unit in enumerate(unit_ids):
        unit_template = all_templates[unit_index]

        if invert_waveforms:
            unit_template = -unit_template

        upsampled_template_shape = unit_template.shape[1] * upsampling_factor
        resampled_unit_templates = resample(unit_template, upsampled_template_shape, axis=1)

        resampled_fs = sorting.get_sampling_frequency() * upsampling_factor

        unit_all_chans_features = features.calculate_features(  # this guy takes 2D matrices, instead of computing the
            waveforms=resampled_unit_templates,                 # features per unit, for each channel, we can also first
            sampling_frequency=resampled_fs,                    # extract the main waveform per unit, and pass this to
            feature_names=feature_names,                        # calculate features
            recovery_slope_window=recovery_slope_window,
        )
        all_template_features.loc[unit_index] = unit_all_chans_features.loc[max_channels[unit_index]]


    # ---------------------- DEAL WITH OUTPUT -------------------------
    if save_property_or_features:
        for feature_name in all_template_features.columns:
            sorting.set_units_property(unit_ids=unit_ids,
                                       property_name=feature_name,
                                       values=all_template_features[feature_name].values)

    return all_template_features


