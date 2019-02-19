import numpy as np

'''
This module implements a number of biophysical metrics to validate spike sorting
results.
'''


def getISIRatio(unit_spike_train, sampling_frequency):
    '''This function calculates the ratio between the frequency of spikes present
    within 0- to 2-ms (refractory period) interspike interval (ISI) and those at 0- to 20-ms
    interval. Taken from:

     "Large-scale, high-density (up to 512 channels) recording of local circuits
     in behaving animals" - Antal Berényi, et al.

    Parameters
    ----------
    unit_spike_train: array_like
        1D array of spike times in frames (sorted in ascending chronological order)
    sampling_frequency: float
        The sampling frequency of recording

    Returns
    ----------
    ISI_ratio: float
        The ratio between the frequency of spikes present within 0- to 2-ms ISI
        and those at 0- to 20-ms interval.
    '''
    ref_frame_period = sampling_frequency*0.002
    long_interval = sampling_frequency*0.02

    ISIs = np.diff(unit_spike_train)
    num_ref_violations = float(sum(ISIs<ref_frame_period))
    num_longer_interval = float(sum(ISIs<long_interval))

    ISI_ratio = num_ref_violations / num_longer_interval
    return ISI_ratio
