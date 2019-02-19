import numpy as np

'''
This module implements a number of biophysical metrics to validate spike sorting
results. These are taken from the "Proposal for web-based spike sorting validation"
white paper written by Alex Barnett, Jeremy Magland, and James Jun
'''


def getISIViolations(unit_spike_train, sampling_frequency, ref_period=0.002, min_ISI=0.0):
    '''This function calculates an estimated false positive rate of the spikes
    in the given spike train using the number of refractory violations:

    Parameters
    ----------
    unit_spike_train: array_like
        1D array of spike times in frames (sorted in ascending chronological order)
    ref_period: float
        The estimated time (in seconds) of the refractory period of the unit
    sampling_frequency: float
        The sampling frequency of recording
    min_ISI: float
        The minimum possible ISI time (in seconds)based on the cutoff window.
        This parameter is for single electrode recordings where there is a cutoff
        window for detection, this is not defined for multi-electrode recordings.
        Therefore, the default value is 0, but should be set to about 1ms for for
        most single electrode recordings.

    Returns
    ----------
    fp_rate: float
        The estimated false positive rate
    num_violations: int
        The number of refractory/minISI violations in the given spike train

    '''
    ref_frame_period = sampling_frequency*ref_period
    min_frame_ISI = sampling_frequency*min_ISI

    ISIs = np.diff(unit_spike_train)
    num_spikes = unit_spike_train.shape[0]
    num_violations = sum(ISIs<ref_frame_period)

    violation_time = 2*num_spikes*(ref_period-min_ISI)
    total_rate = num_spikes/(unit_spike_train[-1]/sampling_frequency)
    violation_rate = num_violations/violation_time
    fp_rate = violation_rate/total_rate
    return fp_rate, num_violations
