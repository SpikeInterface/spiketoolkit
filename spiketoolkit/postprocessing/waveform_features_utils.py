"""
Functions based on
https://github.com/AllenInstitute/ecephys_spike_sorting/blob/master/ecephys_spike_sorting/modules/mean_waveforms/waveform_metrics.py
15/04/2020
"""


import numpy as np
from scipy.stats import linregress

all_1D_features = ['peak_to_valley', 'halfwidth', 'peak_trough_ratio',
                   'repolarization_slope', 'recovery_slope']


def peak_to_valley(waveform, fs, max_time_after_trough):
    trough_idx, peak_idx = _get_trough_and_peak_idx(waveform, fs, max_time_after_trough)
    ptv = (peak_idx - trough_idx) * (1/fs)
    return ptv


def halfwidth(waveform, fs, max_time_after_trough):
    trough_idx, peak_idx = _get_trough_and_peak_idx(waveform, fs, max_time_after_trough)
    hw = np.empty(waveform.shape[0])

    for i, pkidx in enumerate(peak_idx):
        cross_pre_pk, cross_post_pk = _get_halfwidth_crossing(waveform[i, :], pkidx)
        if cross_pre_pk is None:
            hw[i] = np.nan
        else:
            hw[i] = (cross_post_pk - cross_pre_pk+pkidx) * (1/fs)

    return hw


def peak_trough_ratio(waveform, fs, max_time_after_trough):
    trough_idx, peak_idx = _get_trough_and_peak_idx(waveform, fs, max_time_after_trough)
    ptratio = np.empty(trough_idx.shape[0])
    for i, (thidx, pkidx) in enumerate(zip(trough_idx, peak_idx)):
        ptratio[i] = waveform[i, pkidx] / waveform[i, thidx]
    return ptratio


# TODO: is this usefull? we allready have TTP duration and ratio
def repolarization_slope(waveform, fs, max_time_after_trough):
    """

    Parameters
    ----------
    waveform : numpy.ndarray [n_template x n_samples]
    fs : samplerate in Hz

    Returns
    -------
    rslope: slope between trough and peak ( d[Units waveform] / dt, time [s])

    """
    trough_idx, peak_idx = _get_trough_and_peak_idx(waveform, fs, max_time_after_trough)
    rslope = np.empty(waveform.shape[0])
    time = np.arange(0, waveform.shape[1]) * (1/fs)  # in s
    for i in range(waveform.shape[0]):
        if peak_idx[i] < trough_idx[i]:
            print('trough larger than peak')
            continue
        if trough_idx[i].size == 0 or peak_idx[i].size == 0:
            continue
        rslope[i] = linregress(time[trough_idx[i]:peak_idx[i]], waveform[i, trough_idx[i]: peak_idx[i]])[0]
    return rslope


def recovery_slope(waveform, fs, max_time_after_trough, window):
    """

    Parameters
    ----------
    waveform : numpy.ndarray [n_template x n_samples]
    fs : samplerate in Hz
    window : duration after peak to include for regression

    Returns
    -------

    """
    _, peak_idx = _get_trough_and_peak_idx(waveform, fs, max_time_after_trough)
    rslope = np.empty(waveform.shape[0])
    time = np.arange(0, waveform.shape[1]) * (1/fs)  # in s

    for i in range(waveform.shape[0]):
        max_idx = int(peak_idx[i] + ((window/1000)*fs))
        max_idx = np.min([max_idx, waveform.shape[1]])
        slope = _get_slope(time[peak_idx[i]:max_idx], waveform[i, peak_idx[i]:max_idx])
        rslope[i] = slope[0]
    return rslope


def _get_slope(time, waveform):
    slope = linregress(time, waveform)
    return slope


def _get_trough_and_peak_idx(waveform, fs, max_time_after_trough):
    trough_idx = np.argmin(waveform, axis=1)
    peak_idx = np.empty(trough_idx.shape, dtype=int)

    time = np.arange(0, waveform.shape[1]) * (1/fs) * 1000  # in ms
    for i, tridx in enumerate(trough_idx):
        if tridx == waveform.shape[1]-1:
            continue
        constrained_idx = np.where(
            (time > time[tridx]) &
            (time < time[tridx] + max_time_after_trough)
        )[0]
        idx = np.argmax(waveform[i, constrained_idx])
        peak_idx[i] = constrained_idx[idx]

    return trough_idx, peak_idx


def _get_halfwidth_crossing(waveform, peak_index):
    waveform = waveform - np.median(waveform)  # TODO add median?
    threshold = waveform[peak_index] * 0.5
    cross_pre_pk = np.where(waveform[:peak_index] > threshold)[0]
    cross_post_pk = np.where(waveform[peak_index:] < threshold)[0]
    if len(cross_pre_pk) == 0 or len(cross_post_pk) == 0:
        return None, None
    else:
        return cross_pre_pk[0], cross_post_pk[0] + peak_index

#
#
#
#
# TODO: add check for waveform?
# def _check_wave_form():


# ------------------------- SOURCE IMPLEMENTATIONS --------------------------------

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