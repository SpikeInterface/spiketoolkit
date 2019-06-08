import numpy as np

'''
This module implements a number of biophysical metrics to validate spike sorting
results.
'''


def compute_ISI_violation_ratio(sorting, sampling_frequency, unit_ids=None, save_as_property=True,
                                ref_period_ms=2, long_period_ms=20):
    '''This function calculates the ratio between the frequency of spikes present
    within 0- to 2-ms (ref_period_ms) interspike interval (ISI) and those at 0- to 20-ms (long_period_ms)
    interval. It then returns the ratios and also adds a property, ISI_ratio, for
    the passed in sorting extractor. Taken from:

     "Large-scale, high-density (up to 512 channels) recording of local circuits
     in behaving animals" - Antal Berényi, et al.

    Parameters
    ----------
    unit_ids: list
        List of unit ids for which to get ISIratios
    sorting: SortingExtractor
        SortingExtractor for the results file being analyzed
    sampling_frequency: float
        The sampling frequency of recording
    save_as_property: boolean
        If True, this will save the ISI_ratio as a property in the given
        sorting extractor.
    ref_period_ms: float
        Refractory period in ms (default 2 ms)
    long_period_ms: float
        Long period in ms (default 20 ms)

    Returns
    ----------
    isi_ratio_list: list of floats
        A list of ratios for each unit passed into this function. Each ratio is
        the ratio between the frequency of spikes present within 0- to 2-ms ISI
        and those at 0- to 20-ms interval for the corresponding spike train.
    '''
    isi_ratio_list = []
    if unit_ids is None:
        unit_ids = sorting.get_unit_ids()
    for unit_id in unit_ids:
        unit_spike_train = sorting.get_unit_spike_train(unit_id)
        ref_frame_period = sampling_frequency * ref_period_ms / 1000.
        long_interval = sampling_frequency * long_period_ms / 1000.

        ISIs = np.diff(unit_spike_train)
        num_ref_violations = float(sum(ISIs < ref_frame_period))
        num_longer_interval = float(sum(ISIs < long_interval))

        if num_longer_interval > 0:
            ISI_ratio = num_ref_violations / num_longer_interval
        else:
            ISI_ratio = 0

        if save_as_property:
            sorting.set_unit_property(unit_id, 'ISI_violation_ratio', ISI_ratio)
        isi_ratio_list.append(ISI_ratio)
    return isi_ratio_list
