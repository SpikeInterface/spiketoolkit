import scipy.signal as ss
from joblib import Parallel, delayed
import spikeextractors as se
import itertools
import numpy as np


def detect_spikes(recording, channel_ids=None, detect_threshold=5, n_pad_ms=2, upsample=1, detect_sign=-1,
                  min_diff_samples=5, align=True, n_jobs=1, verbose=False):
    '''
    Detects spikes per channel.
    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object
    channel_ids: list or None
        List of channels to perform detection. If None all channels are used
    detect_threshold: float
        Threshold in MAD to detect peaks
    n_pad_ms: float
        Time in ms to find absolute peak around detected peak
    upsample: int
        The detected waveforms are upsampled 'upsample' times (default=1)
    detect_sign: int
        Sign of the detection: -1 (negative), 1 (positive), 0 (both)
    min_diff_samples: int
        Minimum interval to skip consecutive spikes (default=5)
    parallel: bool
        If True, each channel is run in parallel
    n_jobs: int
        Number of jobs when parallel
    Returns
    -------
    sorting_detected: SortingExtractor
        The sorting extractor object with the detected spikes. Unit ids are the same as channel ids and units have the
        'channel' property to specify which channel they correspond to
    '''
    spike_times = []
    labels = []
    n_pad_samples = int(n_pad_ms * recording.get_sampling_frequency() / 1000)

    if channel_ids is None:
        channel_ids = recording.get_channel_ids()
    else:
        assert np.all([ch in recording.get_channel_ids() for ch in channel_ids]), "Not all 'channel_ids' are in the" \
                                                                                  "recording."

    if not recording.check_if_dumpable():
        if n_jobs > 1:
            n_jobs = 0
            print("RecordingExtractor is not dumpable and can't be processedin parallel")
            rec_arg = recording
        else:
            rec_arg = recording
    else:
        rec_arg = recording.make_serialized_dict()

    if n_jobs > 1:
        output = Parallel(n_jobs=n_jobs)(delayed(_detect_and_align_peaks_single_channel)
                                         (rec_arg, ch, detect_threshold, detect_sign,
                                          n_pad_samples, upsample, min_diff_samples, align, verbose)
                                         for ch in channel_ids)
        for o in output:
            spike_times.append(o[0])
            labels.append(o[1])
    else:
        for ch in channel_ids:
            peak_times, label = _detect_and_align_peaks_single_channel(recording, ch, detect_threshold, detect_sign,
                                                                       n_pad_samples, upsample, min_diff_samples,
                                                                       align, verbose)
            spike_times.append(peak_times)
            labels.append(label)

    # create sorting extractor
    sorting = se.NumpySortingExtractor()
    labels_flat = np.array(list(itertools.chain(*labels)))
    times_flat = np.array(list(itertools.chain(*spike_times)))
    sorting.set_times_labels(times=times_flat, labels=labels_flat)

    for u in sorting.get_unit_ids():
        sorting.set_unit_property(u, 'channel', u)

    return sorting


def _detect_and_align_peaks_single_channel(rec_arg, channel, n_std, detect_sign, n_pad, upsample, min_diff_samples,
                                           align, verbose):
    if verbose:
        print(f'Detecting spikes on channel {channel}')
    if isinstance(rec_arg, dict):
        recording = se.load_extractor_from_dict(rec_arg)
    else:
        recording = rec_arg
    trace = np.squeeze(recording.get_traces(channel_ids=channel))
    if detect_sign == -1:
        thresh = -n_std * np.median(np.abs(trace) / 0.6745)
        idx_spikes = np.where(trace < thresh)[0]
    elif detect_sign == 1:
        thresh = n_std * np.median(np.abs(trace) / 0.6745)
        idx_spikes = np.where(trace > thresh)[0]
    else:
        thresh = n_std * np.median(np.abs(trace) / 0.6745)
        idx_spikes = np.where((trace > thresh) | (trace < -thresh))[0]
    intervals = np.diff(idx_spikes)
    sp_times = []

    for i_t, diff in enumerate(intervals):
        if diff > min_diff_samples or i_t == len(intervals) - 1:
            idx_spike = idx_spikes[i_t]

            if align:
                if idx_spike - n_pad > 0 and idx_spike + n_pad < len(trace):
                    spike = trace[idx_spike - n_pad:idx_spike + n_pad]
                    t_spike = np.arange(idx_spike - n_pad, idx_spike + n_pad)
                elif idx_spike - n_pad < 0:
                    spike = trace[:idx_spike + n_pad]
                    spike = np.pad(spike, (np.abs(idx_spike - n_pad), 0), 'constant')
                    t_spike = np.arange(idx_spike + n_pad)
                    t_spike = np.pad(t_spike, (np.abs(idx_spike - n_pad), 0), 'constant')
                elif idx_spike + n_pad > len(trace):
                    spike = trace[idx_spike - n_pad:]
                    spike = np.pad(spike, (0, idx_spike + n_pad - len(trace)), 'constant')
                    t_spike = np.arange(idx_spike - n_pad, len(trace))
                    t_spike = np.pad(t_spike, (0, idx_spike + n_pad - len(trace)), 'constant')

                if upsample > 1:
                    spike_up = ss.resample(spike, int(upsample * len(spike)))
                    t_spike_up = np.linspace(t_spike[0], t_spike[-1], num=len(spike_up))
                else:
                    spike_up = spike
                    t_spike_up = t_spike
                if detect_sign == -1:
                    peak_idx = np.argmin(spike_up)
                elif detect_sign == 1:
                    peak_idx = np.argmax(spike_up)
                else:
                    peak_idx = np.argmax(np.abs(spike_up))

                min_time_up = t_spike_up[peak_idx]
                sp_times.append(int(min_time_up))
            else:
                sp_times.append(idx_spike)

    labels = [channel] * len(sp_times)

    return sp_times, labels
