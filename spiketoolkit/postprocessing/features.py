'''
This code is based on: https://github.com/AllenInstitute/ecephys_spike_sorting
'''


import numpy as np
import random
import pandas as pd

from scipy.stats import linregress
from scipy.signal import resample

from .postprocessing_tools import get_unit_templates


# def compute_unit_template_features(waveforms,
#                                cluster_id,
#                                peak_channel,
#                                channel_map,
#                                sample_rate,
#                                upsampling_factor,
#                                spread_threshold,
#                                site_range,
#                                site_spacing,
#                                epoch_name):
#     """
#     Calculate metrics for an array of waveforms.
#     Metrics come from Jia et al. (2019) High-density extracellular probes reveal
#     dendritic backpropagation and facilitate neuron classification. J Neurophys
#     https://doi.org/10.1152/jn.00680.2018
#     Inputs:
#     -------
#     waveforms : numpy.ndarray (num_spikes x num_channels x num_samples)
#         Can include NaN values for missing spikes
#     cluster_id : int
#         ID for cluster
#     peak_channel : int
#         Location of waveform peak
#     channel_map : numpy.ndarray
#         Channels used for spike sorting
#     sample_rate : float
#         Sample rate in Hz
#     upsampling_factor : float
#         Relative rate at which to upsample the spike waveform
#     spread_threshold : float
#         Threshold for computing spread of 2D waveform
#     site_range : float
#         Number of sites to use for 2D waveform metrics
#     site_spacing : float
#         Average vertical distance between sites (m)
#     epoch_name : str
#         Name of epoch for which these waveforms originated
#     Outputs:
#     -------
#     metrics : pandas.DataFrame
#         Single-row table containing all metrics
#     """
#
#     snr = compute_snr(waveforms[:, peak_channel, :])
#
#     mean_2D_waveform = np.squeeze(np.nanmean(waveforms[:, channel_map, :], 0))
#     local_peak = np.argmin(np.abs(channel_map - peak_channel))
#
#     num_samples = waveforms.shape[2]
#     new_sample_count = int(num_samples * upsampling_factor)
#
#     mean_1D_waveform = resample(
#         mean_2D_waveform[local_peak, :], new_sample_count)
#
#     timestamps = np.linspace(0, num_samples / sample_rate, new_sample_count)
#
#     duration = compute_unit_template_duration(mean_1D_waveform, timestamps)
#     halfwidth = compute_unit_template_halfwidth(mean_1D_waveform, timestamps)
#     PT_ratio = compute_unit_template_PT_ratio(mean_1D_waveform)
#     repolarization_slope = compute_unit_template_repolarization_slope(
#         mean_1D_waveform, timestamps)
#     recovery_slope = compute_unit_template_recovery_slope(
#         mean_1D_waveform, timestamps)
#
#     amplitude, spread, velocity_above, velocity_below = compute_2D_features(
#         mean_2D_waveform, timestamps, local_peak, spread_threshold, site_range, site_spacing)
#
#     data = [[cluster_id, epoch_name, peak_channel, snr, duration, halfwidth, PT_ratio, repolarization_slope,
#              recovery_slope, amplitude, spread, velocity_above, velocity_below]]
#
#     metrics = pd.DataFrame(data,
#                            columns=['cluster_id', 'epoch_name', 'peak_channel', 'snr', 'duration', 'halfwidth',
#                                     'PT_ratio', 'repolarization_slope', 'recovery_slope', 'amplitude',
#                                     'spread', 'velocity_above', 'velocity_below'])
#
#     return metrics

def add_template(sorting, recording, overwrite=False):
    if 'template' in sorting.get_shared_unit_property_names() and not overwrite:
        print('template allready in sorting, use "overwrite" to overwrite')
        return
    elif 'template' in sorting.get_shared_unit_property_names() and overwrite:
        print('overwriting existing template')

    unit_ids = sorting.get_unit_ids()

    templates = get_unit_templates(
        recording=recording,
        sorting=sorting,
        mode='median'
    )
    for i, uid in enumerate(unit_ids):
        sorting.set_unit_property(uid, 'template', templates[i])


# ==========================================================

# EXTRACTING 1D FEATURES

# ==========================================================


# def compute_unit_template_duration(waveform, timestamps):
#     """
#     Duration (in seconds) between peak and trough
#     Inputs:
#     ------
#     waveform : numpy.ndarray (N samples)
#     timestamps : numpy.ndarray (N samples)
#     Outputs:
#     --------
#     duration : waveform duration in milliseconds
#     """
#
#     trough_idx = np.argmin(waveform)
#     peak_idx = np.argmax(waveform)
#
#     # to avoid detecting peak before trough
#     if waveform[peak_idx] > np.abs(waveform[trough_idx]):
#         duration = timestamps[peak_idx:][np.where(waveform[peak_idx:] == np.min(waveform[peak_idx:]))[0][0]] - \
#                    timestamps[peak_idx]
#     else:
#         duration = timestamps[trough_idx:][np.where(waveform[trough_idx:] == np.max(waveform[trough_idx:]))[0][0]] - \
#                    timestamps[trough_idx]
#
#     return duration * 1e3


def compute_trough_to_peak_duration(sorting,
                                    unit_ids=None,
                                    invert_template=False,
                                    save_property_or_features=True,
                                    return_through_and_peak_idx=False):

    if isinstance(unit_ids, (int, np.integer)):
        unit_ids = [unit_ids]
    elif unit_ids is None:
        unit_ids = sorting.get_unit_ids()
    elif not isinstance(unit_ids, (list, np.ndarray)):
        raise Exception("unit_ids should be an int or list")

    assert 'template' in sorting.get_shared_unit_property_names(), 'add template to unit properties'

    durations = []
    peaks = []
    throughs = []

    for uid in unit_ids:
        templates = sorting.get_unit_property(uid, 'template')
        unit_durations = []
        unit_peaks = []
        unit_throughs = []

        for i, channel_template in enumerate(templates):
            if invert_template:
                channel_template = -channel_template

            trough_idx = np.argmin(channel_template)
            unit_throughs.append(trough_idx)
            peak_idx = np.argmax(channel_template)
            unit_peaks.append(peak_idx)

            dur = (peak_idx - trough_idx) * (1 / sorting.get_sampling_frequency())
            unit_durations.append(dur)  # exclude negative values?

        durations.append(unit_durations)
        peaks.append(unit_peaks)
        throughs.append(unit_throughs)

    if save_property_or_features:
        for i, uid in enumerate(unit_ids):
            sorting.set_unit_property(uid, 'trough_to_peak_duration', durations[i])
            sorting.set_unit_property(uid, 'peak_index_in_template', peaks[i])
            sorting.set_unit_property(uid, 'through_index_in_template', throughs[i])

    if return_through_and_peak_idx:
        return durations, throughs, peaks
    else:
        return durations


# def compute_unit_template_halfwidths(waveform, timestamps):
#     """
#     Spike width (in seconds) at half max amplitude
#     Inputs:
#     ------
#     waveform : numpy.ndarray (N samples)
#     timestamps : numpy.ndarray (N samples)
#     Outputs:
#     --------
#     halfwidth : waveform halfwidth in milliseconds
#     """
#
#     trough_idx = np.argmin(waveform)
#     peak_idx = np.argmax(waveform)
#
#     try:
#         if waveform[peak_idx] > np.abs(waveform[trough_idx]):
#             threshold = waveform[peak_idx] * 0.5
#             thresh_crossing_1 = np.min(
#                 np.where(waveform[:peak_idx] > threshold)[0])
#             thresh_crossing_2 = np.min(
#                 np.where(waveform[peak_idx:] < threshold)[0]) + peak_idx
#         else:
#             threshold = waveform[trough_idx] * 0.5
#             thresh_crossing_1 = np.min(
#                 np.where(waveform[:trough_idx] < threshold)[0])
#             thresh_crossing_2 = np.min(
#                 np.where(waveform[trough_idx:] > threshold)[0]) + trough_idx
#
#         halfwidth = (timestamps[thresh_crossing_2] - timestamps[thresh_crossing_1])
#
#     except ValueError:
#
#         halfwidth = np.nan
#
#     return halfwidth * 1e3
#
#
# def compute_unit_template_pt_ratios(waveform):
#     """
#     Peak-to-trough ratio of 1D waveform
#     Inputs:
#     ------
#     waveform : numpy.ndarray (N samples)
#     Outputs:
#     --------
#     PT_ratio : waveform peak-to-trough ratio
#     """
#
#     trough_idx = np.argmin(waveform)
#
#     peak_idx = np.argmax(waveform)
#
#     PT_ratio = np.abs(waveform[peak_idx] / waveform[trough_idx])
#
#     return PT_ratio
#
#
# def compute_unit_template_repolarization_slopes(waveform, timestamps, window=20):
#     """
#     Spike repolarization slope (after maximum deflection point)
#     Inputs:
#     ------
#     waveform : numpy.ndarray (N samples)
#     timestamps : numpy.ndarray (N samples)
#     window : int
#         Window (in samples) for linear regression
#     Outputs:
#     --------
#     repolarization_slope : slope of return to baseline (V / s)
#     """
#
#     max_point = np.argmax(np.abs(waveform))
#
#     waveform = - waveform * (np.sign(waveform[max_point]))  # invert if we're using the peak
#
#     repolarization_slope = linregress(timestamps[max_point:max_point + window], waveform[max_point:max_point + window])[
#         0]
#
#     return repolarization_slope * 1e-6
#
#
# def compute_unit_template_recovery_slopes(waveform, timestamps, window=20):
#     """
#     Spike recovery slope (after repolarization)
#     Inputs:
#     ------
#     waveform : numpy.ndarray (N samples)
#     timestamps : numpy.ndarray (N samples)
#     window : int
#         Window (in samples) for linear regression
#     Outputs:
#     --------
#     recovery_slope : slope of recovery period (V / s)
#     """
#
#     max_point = np.argmax(np.abs(waveform))
#
#     waveform = - waveform * (np.sign(waveform[max_point]))  # invert if we're using the peak
#
#     peak_idx = np.argmax(waveform[max_point:]) + max_point
#
#     recovery_slope = linregress(timestamps[peak_idx:peak_idx + window], waveform[peak_idx:peak_idx + window])[0]
#
#     return recovery_slope * 1e-6
#
#
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

