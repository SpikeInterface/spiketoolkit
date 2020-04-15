"""
This code is based on: https://github.com/AllenInstitute/ecephys_spike_sorting
"""

import numpy as np
import pandas as pd
from .postprocessing_tools import get_unit_templates
from ..postprocessing import waveform_features_library as waveform_features
from ..postprocessing.waveform_features_library import all_1D_features


def compute_waveform_features(
        sorting,
        recording=None,
        feature_names=None,
        unit_ids=None,
        as_dict=False,
        as_dataframe=False,
        save_property_or_features=False
):
    # Handle input parameters
    # unit ids needs to be a list
    if isinstance(unit_ids, (int, np.integer)):
        unit_ids = [unit_ids]
    elif unit_ids is None:
        unit_ids = sorting.get_unit_ids()
    elif not isinstance(unit_ids, (list, np.ndarray)):
        raise Exception("unit_ids should be an int or list")

    # extract templates, either from sorting or recording
    if 'template' not in sorting.get_shared_unit_property_names():
        if recording is None:
            raise ValueError("Add 'template as unit property (st.postprocessing.get_unit_templates), or pass"
                             "recording as kw argument")
        all_templates = get_unit_templates(
            sorting=sorting,
            recording=recording,
            save_property_or_features=save_property_or_features,
            unit_ids=unit_ids,
            mode='mean',
        )
    else:
        all_templates = [sorting.get_unit_property(uid, 'template') for uid in unit_ids]

    # check given feature names
    if feature_names is None:
        feature_names = all_1D_features
    else:
        for name in feature_names:
            assert name in all_1D_features, f'{name} not in {all_1D_features}'

    # Extract features
    all_features = []
    for unit_index, unit in enumerate(unit_ids):
        all_unit_features = []
        unit_template = all_templates[unit_index]

        # TODO: parameter handling
        feature_params = dict(
            waveform=unit_template,
            fs=sorting.get_sampling_frequency(),
            max_time_after_trough=1,
        )

        for feature_name in feature_names:
            if feature_name == 'recovery_slope':
                feature_params['window'] = 0.7
            feature_values = getattr(waveform_features, feature_name)(**feature_params)
            all_unit_features.append(feature_values)

            if save_property_or_features:
                sorting.set_unit_property(unit, feature_name, feature_values)

        all_features.append(all_unit_features)

    # handle output
    if as_dict and as_dataframe:
        print('set either as_dict or as_dataframe True, returning dict')
        as_dataframe = False

    if as_dict:
        feature_dict = dict()
        for i, uid in enumerate(unit_ids):
            feature_dict[uid] = dict()
            for j, fname in enumerate(feature_names):
                feature_dict[uid][fname] = all_features[i][j]
        return feature_dict
    elif as_dataframe:
        feature_dict = dict()
        for i, uid in enumerate(unit_ids):
            unit_frame = pd.DataFrame()
            for j, fname in enumerate(feature_names):
                for chan_nr, feature_value in enumerate(all_features[i][j]):
                    unit_frame.at[chan_nr, fname] = feature_value

            feature_dict[uid] = unit_frame

        return feature_dict
    else:
        return all_features


# # ==========================================================
#
# # EXTRACTING 2D FEATURES
#
# # ==========================================================
#
#
# def compute_unit_template_spreads(recording, sorting, unit_ids=None, channel_ids=None,
#                                   mode='median', _waveforms=None, **kwargs):
#     pass

# def compute_2D_features(waveform, timestamps, peak_channel, spread_threshold=0.12, site_range=16, site_spacing=10e-6):
#     """
#     Compute features of 2D waveform (channels x samples)
#     Inputs:
#     ------
#     waveform : numpy.ndarray (N channels x M samples)
#     timestamps : numpy.ndarray (M samples)
#     peak_channel : int
#     spread_threshold : float
#     site_range: int
#     site_spacing : float
#     Outputs:
#     --------
#     amplitude : uV
#     spread : um
#     velocity_above : s / m
#     velocity_below : s / m
#     """
#
#     assert site_range % 2 == 0  # must be even
#
#     sites_to_sample = np.arange(-site_range, site_range + 1, 2) + peak_channel
#
#     sites_to_sample = sites_to_sample[(sites_to_sample > 0) * (sites_to_sample < waveform.shape[0])]
#
#     wv = waveform[sites_to_sample, :]
#
#     # smoothed_waveform = np.zeros((wv.shape[0]-1,wv.shape[1]))
#     # for i in range(wv.shape[0]-1):
#     #    smoothed_waveform[i,:] = np.mean(wv[i:i+2,:],0)
#
#     trough_idx = np.argmin(wv, 1)
#     trough_amplitude = np.min(wv, 1)
#
#     peak_idx = np.argmax(wv, 1)
#     peak_amplitude = np.max(wv, 1)
#
#     duration = np.abs(timestamps[peak_idx] - timestamps[trough_idx])
#
#     overall_amplitude = peak_amplitude - trough_amplitude
#     amplitude = np.max(overall_amplitude)
#     max_chan = np.argmax(overall_amplitude)
#
#     points_above_thresh = np.where(overall_amplitude > (amplitude * spread_threshold))[0]
#
#     if len(points_above_thresh) > 1:
#         points_above_thresh = points_above_thresh[isnot_outlier(points_above_thresh)]
#
#     spread = len(points_above_thresh) * site_spacing * 1e6
#
#     channels = sites_to_sample - peak_channel
#     channels = channels[points_above_thresh]
#
#     trough_times = timestamps[trough_idx] - timestamps[trough_idx[max_chan]]
#     trough_times = trough_times[points_above_thresh]
#
#     velocity_above, velocity_below = get_velocity(channels, trough_times, site_spacing)
#
#     return amplitude, spread, velocity_above, velocity_below
#
#
# # ==========================================================
#
# # HELPER FUNCTIONS:
#
# # ==========================================================
#
#
# def get_velocity(channels, times, distance_between_channels=10e-6):
#     """
#     Calculate slope of trough time above and below soma.
#     Inputs:
#     -------
#     channels : np.ndarray
#         Channel index relative to soma
#     times : np.ndarray
#         Trough time relative to peak channel
#     distance_between_channels : float
#         Distance between channels (m)
#     Outputs:
#     --------
#     velocity_above : float
#         Inverse of velocity of spike propagation above the soma (s / m)
#     velocity_below : float
#         Inverse of velocity of spike propagation below the soma (s / m)
#     """
#
#     above_soma = channels >= 0
#     below_soma = channels <= 0
#
#     if np.sum(above_soma) > 1:
#         slope_above, intercept, r_value, p_value, std_err = linregress(channels[above_soma], times[above_soma])
#         velocity_above = slope_above / distance_between_channels
#     else:
#         velocity_above = np.nan
#
#     if np.sum(below_soma) > 1:
#         slope_below, intercept, r_value, p_value, std_err = linregress(channels[below_soma], times[below_soma])
#         velocity_below = slope_below / distance_between_channels
#     else:
#         velocity_below = np.nan
#
#     return velocity_above, velocity_below
#
#
# def isnot_outlier(points, thresh=1.5):
#     """
#     Returns a boolean array with True if points are outliers and False
#     otherwise.
#     Parameters:
#     -----------
#         points : An numobservations by numdimensions array of observations
#         thresh : The modified z-score to use as a threshold. Observations with
#             a modified z-score (based on the median absolute deviation) greater
#             than this value will be classified as outliers.
#     Returns:
#     --------
#         mask : A numobservations-length boolean array.
#     References:
#     ----------
#         Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
#         Handle Outliers", The ASQC Basic References in Quality Control:
#         Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
#     """
#
#     if len(points.shape) == 1:
#         points = points[:, None]
#
#     median = np.median(points, axis=0)
#
#     diff = np.sum((points - median) ** 2, axis=-1)
#     diff = np.sqrt(diff)
#
#     med_abs_deviation = np.median(diff)
#
#     modified_z_score = 0.6745 * diff / med_abs_deviation
#
#     return modified_z_score <= thresh
#

