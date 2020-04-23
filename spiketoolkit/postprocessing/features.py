"""
Uses the functions in SpikeInterface/spikefeatures to compute
unit template features
"""

import pandas as pd
from spikefeatures import features
from scipy.signal import resample
from .postprocessing_tools import get_unit_templates, get_unit_max_channels
import numpy as np


def compute_unit_template_features(
        sorting,
        recording,
        unit_ids=None,
        channel_ids=None,
        feature_names=None,
        save_property_or_features=False,
        recovery_slope_window=0.7,
        upsampling_factor=1,
        invert_waveforms=False,
        as_dataframe=False,
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
    channel_ids : list
        if None, max channel for the unit is detected and used. (not used when using template properties)
    return_dict: bool
        If True, will return dict of metrics (default False)
    feature_names: list
        list of feature names to be computed
    upsampling_factor : int
        factor with which to upsample the template resolution (default 1)
    invert_waveforms : bool
        invert templates before computing features (default False)
    recovery_slope_window : float
        window after peak in ms wherein to compute recovery slope (default 0.7)
    use_sorting_property_template : bool
        use templates saved as unit properties (default True)

    save_property_or_features=False,
    verbose=False,

    Returns

    -------
    metrics : pandas.DataFrame
        table containing all metrics:
            rows: unit_ids
            columns: metrics

    OR

    metrics : dict
        dict with metrics as keywords, values are list with
        metric per unit_id
    """

    # ------------------- SETUP ------------------------------
    if unit_ids is None:
        unit_ids = sorting.get_unit_ids()

    if channel_ids is None:
        channel_ids = get_unit_max_channels(
            sorting=sorting,
            recording=recording,
            unit_ids=unit_ids,
        )
    else:
        assert len(channel_ids) == len(unit_ids), f'#channel ids should be # unit ids'

    if feature_names is None:
        feature_names = features.all_1D_features

    input_templates = []  # a list with templates per unit to compute features for
    if 'template' not in sorting.get_shared_unit_property_names(unit_ids=unit_ids):
        for uid, chid in zip(unit_ids, channel_ids):
            input_templates.append(get_unit_templates(
                sorting=sorting,
                recording=recording,
                save_property_or_features=False,
                unit_ids=[uid],
                channel_ids=chid,
                mode='median',
                verbose=verbose,
                memmap=False,
            ))

    else:
        for uid, chid in zip(unit_ids, channel_ids):
            unit_template = sorting.get_unit_property(uid, 'template')
            if unit_template.shape[0] == 1:
                input_templates.append(unit_template)  # use the one channel is the same as provided channel id
            else:
                assert unit_template.shape[0] == len(recording.get_channel_ids())
                template_for_chan = np.expand_dims(unit_template[chid, :], axis=0)
                input_templates.append(template_for_chan)

    # -------------------- PROCESS TEMPLATES -----------------------------
    proc_templates = []  # processed templates

    upsampled_template_shape = input_templates[0].shape[1] * upsampling_factor
    resampled_fs = sorting.get_sampling_frequency() * upsampling_factor
    for template in input_templates:
        proc_template = resample(template, upsampled_template_shape, axis=1)

        if invert_waveforms:
            proc_template = -proc_template

        proc_templates.append(proc_template)

    # --------------------- COMPUTE FEATURES ------------------------------

    features_df = pd.DataFrame(columns=feature_names + ['unit_id', 'channel_id'])
    df_idx = 0

    for unit_index, unit in enumerate(unit_ids):
        unit_features = features.calculate_features(
            waveforms=proc_templates[unit_index],
            sampling_frequency=resampled_fs,
            feature_names=feature_names,
            recovery_slope_window=recovery_slope_window,
        )
        for i, uf_idx in enumerate(unit_features.index):
            features_df.loc[df_idx] = unit_features.iloc[uf_idx]
            features_df.at[df_idx, 'channel_id'] = channel_ids[unit_index]
            features_df.at[df_idx, 'unit_id'] = unit
            df_idx += 1

    # ---------------------- DEAL WITH OUTPUT -------------------------
    if save_property_or_features:
        for feature_name in features_df.columns:
            sorting.set_units_property(unit_ids=unit_ids,
                                       property_name=feature_name,
                                       values=features_df[feature_name].values)

    if as_dataframe:
        return features_df
    else:
        features_dict = dict()
        for feature_name in feature_names:
            features_dict[feature_name] = features_df[feature_name].values
        return features_dict


