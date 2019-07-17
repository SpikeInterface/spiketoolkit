import numpy as np
import spiketoolkit as st
import spikemetrics.metrics as metrics
from spikemetrics.common import Epoch 

#Quality metrics from Allen Institute can't do individual units so we calculate metrics all units to get 
#the metrics for only a few units.

def compute_firing_rates(sorting, sampling_frequency, unit_ids=None, epoch_tuple=None):
    '''
    Computes and returns the spike times in seconds and also returns 
    the cluster_ids needed for quality metrics (all within given epoch)
    
    Parameters
    ----------
    sorting: SortingExtractor
        The sorting extractor.
    sampling_frequency: float
        The sampling frequency of the recording.
    unit_ids: list
        List of unit ids to compute firing rates for. If not specified, all units are used
    epoch_tuple: int tuple
        A tuple with a start and end frame for the epoch in question.

    Returns
    ----------
    firing_rates: np.array
        The firing rates of the sorted units.
    '''

    if unit_ids is None or unit_ids == []:
        unit_ids = sorting.get_unit_ids()
        unit_indices = np.arange(len(unit_ids))
    else:
        unit_indices = []
        sorting_unit_ids = np.asarray(sorting.get_unit_ids())
        for unit_id in unit_ids:
            index, = np.where(sorting_unit_ids == unit_id)
            if len(index) != 0:
                unit_indices.append(index[0])
    spike_times, spike_clusters  = st.validation.validation_tools.get_firing_times_ids(sorting, sampling_frequency)
    total_units = len(sorting.get_unit_ids()) 
    if epoch_tuple is None:
        epoch = (0, np.inf)
    else:
        epoch = (epoch_tuple[0]/sampling_frequency, epoch_tuple[1]/sampling_frequency)
    in_epoch = np.logical_and(spike_times > epoch[0], spike_times < epoch[1])
    firing_rates_all, _ = metrics.calculate_firing_rate(spike_times[in_epoch], spike_clusters[in_epoch], total_units)
    firing_rates_list = []
    for i in unit_indices:
        firing_rates_list.append(firing_rates_all[i])
    firing_rates = np.asarray(firing_rates_list)
    return firing_rates

def compute_unit_SNR(recording, sorting, unit_ids=None, save_as_property=True, mode='mad',
                     seconds=10, max_num_waveforms=1000, apply_filter=False, freq_min=300, freq_max=6000):
    '''
    Computes signal-to-noise ratio (SNR) of the average waveforms on the largest channel.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor
    sorting: SortingExtractor
        The sorting extractor
    unit_ids: list
        List of unit ids to compute SNR for. If not specified, all units are used
    save_as_property: bool
        If True (defult), the computed SNR is added as a unit property to the sorting extractor as 'snr'
    mode: str
        Mode to compute noise SNR ('mad' | 'std' - default 'mad')
    seconds: float
        Number of seconds to compute noise level from (default 10)
    max_num_waveforms: int
        Maximum number of waveforms to cpmpute templates from (default 1000)
    apply_filter: bool
        If True, recording is filtered before computing noise level
    freq_min: float
        High-pass frequency for optional filter (default 300 Hz)
    freq_max: float
        Low-pass frequency for optional filter (default 6000 Hz)

    Returns
    -------
    snr_list: np.array
        List of computed SNRs

    '''
    if unit_ids is None:
        unit_ids = sorting.get_unit_ids()
    if apply_filter:
        recording_f = st.preprocessing.bandpass_filter(recording=recording, freq_min=freq_min, freq_max=freq_max,
                                                       cache=True)
    else:
        recording_f = recording
    channel_noise_levels = _compute_channel_noise_levels(recording=recording_f, mode=mode, seconds=seconds)
    templates = st.postprocessing.get_unit_templates(recording_f, sorting, unit_ids=unit_ids,
                                                    max_num_waveforms=max_num_waveforms,
                                                    mode='median')
    max_channels = st.postprocessing.get_unit_max_channels(recording, sorting, unit_ids=unit_ids,
                                                          max_num_waveforms=max_num_waveforms, peak='both',
                                                          mode='median')
    snr_list = []
    for i, unit_id in enumerate(sorting.get_unit_ids()):
        max_channel_idx = recording.get_channel_ids().index(max_channels[i])
        snr = _compute_template_SNR(templates[i], channel_noise_levels, max_channel_idx)
        if save_as_property:
            sorting.set_unit_property(unit_id, 'snr', snr)
        snr_list.append(snr)
    return np.asarray(snr_list)


def _compute_template_SNR(template, channel_noise_levels, max_channel_idx):
    '''
    Computes SNR on the channel with largest amplitude

    Parameters
    ----------
    template: np.array
        Template (n_elec, n_timepoints)
    channel_noise_levels: list
        Noise levels for the different channels
    max_channel_idx: int
        Index of channel with largest templaye

    Returns
    -------
    snr: float
        Signal-to-noise ratio for the template
    '''
    snr = np.max(np.abs(template[max_channel_idx])) / channel_noise_levels[max_channel_idx]
    return snr


def _compute_channel_noise_levels(recording, mode='mad', seconds=10):
    '''
    Computes noise level channel-wise

    Parameters
    ----------
    recording: RecordingExtractor
        The recording ectractor object
    mode: str
        'std' or 'mad' (default
    seconds: float
        Number of seconds to compute SNR from

    Returns
    -------
    moise_levels: list
        Noise levels for each channel
    '''
    M = recording.get_num_channels()
    n_frames = int(seconds * recording.get_sampling_frequency())

    if n_frames > recording.get_num_frames():
        start_frame = 0
        end_frame = recording.get_num_frames()
    else:
        start_frame = np.random.randint(0, recording.get_num_frames() - n_frames)
        end_frame = start_frame + n_frames

    X = recording.get_traces(start_frame=start_frame, end_frame=end_frame)
    noise_levels = []
    for ch in range(M):
        if mode == 'std':
            noise_level = np.std(X[ch, :])
        elif mode == 'mad':
            noise_level = np.median(np.abs(X[ch, :])/0.6745)
        else:
            raise Exception("'mode' can be 'std' or 'mad'")
        noise_levels.append(noise_level)
    return noise_levels
