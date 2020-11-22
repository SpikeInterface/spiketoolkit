import scipy.signal as ss
from joblib import Parallel, delayed
import spikeextractors as se
from ..postprocessing.postprocessing_tools import divide_recording_into_time_chunks
import itertools
from tqdm import tqdm
import numpy as np


def detect_spikes(recording, channel_ids=None, detect_threshold=5, n_pad_ms=2, upsample=1, detect_sign=-1,
                  min_diff_samples=5, align=True, start_frame=None, end_frame=None, n_jobs=1,
                  chunk_size=None, chunk_mb=500, verbose=False):
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
    align: bool
        If True, spike times are aligned on the peak
    start_frame: int
        Start frame for detection
    end_frame: int
        End frame end frame for detection
    n_jobs: int
        Number of jobs when parallel
    chunk_size: int
        Size of chunks in number of samples. If None, it is automatically calculated
    chunk_mb: int
        Size of chunks in Mb (default 500 Mb)
    verbose: bool
                If True output is verbose

    Returns
    -------
    sorting_detected: SortingExtractor
        The sorting extractor object with the detected spikes. Unit ids are the same as channel ids and units have the
        'channel' property to specify which channel they correspond to. The sorting extractor also has the `spike_rate`
        and `spike_amplitude` properties.
    '''
    n_pad_samples = int(n_pad_ms * recording.get_sampling_frequency() / 1000)

    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = recording.get_num_frames()

    if channel_ids is None:
        channel_ids = recording.get_channel_ids()
    else:
        assert np.all([ch in recording.get_channel_ids() for ch in channel_ids]), "Not all 'channel_ids' are in the" \
                                                                                  "recording."

    if n_jobs is None:
        n_jobs = 1
    if n_jobs == 0:
        n_jobs = 1

    if start_frame != 0 or end_frame != recording.get_num_frames():
        recording_sub = se.SubRecordingExtractor(recording, start_frame=start_frame, end_frame=end_frame)
    else:
        recording_sub = recording

    num_frames = recording_sub.get_num_frames()

    # set chunk size
    if chunk_size is not None:
        chunk_size = int(chunk_size)
    elif chunk_mb is not None:
        n_bytes = np.dtype(recording.get_dtype()).itemsize
        max_size = int(chunk_mb * 1e6)  # set Mb per chunk
        chunk_size = max_size // (recording.get_num_channels() * n_bytes)

    if n_jobs > 1:
        chunk_size /= n_jobs

    # chunk_size = num_bytes_per_chunk / num_bytes_per_frame
    chunks = divide_recording_into_time_chunks(
        num_frames=num_frames,
        chunk_size=chunk_size,
        padding_size=0
    )
    n_chunk = len(chunks)

    if verbose:
        print(f"Number of chunks: {len(chunks)} - Number of jobs: {n_jobs}")

    if verbose and n_jobs == 1:
        chunk_iter = tqdm(range(n_chunk), ascii=True, desc="Detecting spikes in chunks")
    else:
        chunk_iter = range(n_chunk)

    if not recording_sub.check_if_dumpable():
        if n_jobs > 1:
            n_jobs = 1
            print("RecordingExtractor is not dumpable and can't be processed in parallel")
        rec_arg = recording_sub
    else:
        if n_jobs > 1:
            rec_arg = recording_sub.dump_to_dict()
        else:
            rec_arg = recording_sub

    all_channel_times = [[] for ii in range(len(channel_ids))]
    all_channel_amps = [[] for ii in range(len(channel_ids))]

    if n_jobs > 1:
        output = Parallel(n_jobs=n_jobs)(delayed(_detect_and_align_peaks_chunk)
                                         (ii, rec_arg, chunks, channel_ids, detect_threshold,
                                                              detect_sign, n_pad_samples, upsample, min_diff_samples,
                                                              align, verbose)
                                         for ii in chunk_iter)
        for ii, (times_ii, amps_ii) in enumerate(output):
            for i, ch in enumerate(channel_ids):
                times = times_ii[i]
                amps = amps_ii[i]
                all_channel_amps[i].append(amps)
                all_channel_times[i].append(times)
    else:
        for ii in chunk_iter:
            times_ii, amps_ii = _detect_and_align_peaks_chunk(ii, rec_arg, chunks, channel_ids, detect_threshold,
                                                              detect_sign, n_pad_samples, upsample, min_diff_samples,
                                                              align, False)

            for i, ch in enumerate(channel_ids):
                times = times_ii[i]
                amps = amps_ii[i]
                all_channel_amps[i].append(amps)
                all_channel_times[i].append(times)

    if len(chunks) > 1:
        times_list = []
        amp_list = []
        for i_ch in range(len(channel_ids)):
            times_concat = np.concatenate([all_channel_times[i_ch][ch] for ch in range(len(chunks))],
                                           axis=0)
            times_list.append(times_concat)
            amps_concat = np.concatenate([all_channel_amps[i_ch][ch] for ch in range(len(chunks))],
                                          axis=0)
            amp_list.append(amps_concat)
    else:
        times_list = [times[0] for times in all_channel_times]
        amp_list = [amps[0] for amps in all_channel_amps]

    labels_list = [[ch] * len(times) for (ch, times) in zip(channel_ids, times_list)]

    # create sorting extractor
    sorting = se.NumpySortingExtractor()
    labels_flat = np.array(list(itertools.chain(*labels_list)))
    times_flat = np.array(list(itertools.chain(*times_list)))
    sorting.set_times_labels(times=times_flat, labels=labels_flat)
    sorting.set_sampling_frequency(recording.get_sampling_frequency())

    duration = (end_frame - start_frame) / recording.get_sampling_frequency()

    for i_u, u in enumerate(sorting.get_unit_ids()):
        sorting.set_unit_property(u, 'channel', u)
        amps = amp_list[i_u]
        if len(amps) > 0:
            sorting.set_unit_property(u, 'spike_amplitude', np.median(amp_list[i_u]))
        else:
            sorting.set_unit_property(u, 'spike_amplitude', 0)
        sorting.set_unit_property(u, 'spike_rate', len(sorting.get_unit_spike_train(u)) / duration)

    return sorting


def _detect_and_align_peaks_chunk(ii, rec_arg, chunks, channel_ids, detect_threshold, detect_sign, n_pad, upsample,
                                  min_diff_samples, align, verbose):
    chunk = chunks[ii]

    if verbose:
        print(f"Chunk {ii + 1}: detecting spikes")
    if isinstance(rec_arg, dict):
        recording = se.load_extractor_from_dict(rec_arg)
    else:
        recording = rec_arg

    traces = recording.get_traces(start_frame=chunk['istart'],
                                  end_frame=chunk['iend'])

    sp_times = [[] for ii in range(len(channel_ids))]
    sp_amplitudes = [[] for ii in range(len(channel_ids))]

    for i, ch in enumerate(channel_ids):
        trace = traces[i]
        if detect_sign == -1:
            thresh = -detect_threshold * np.median(np.abs(trace) / 0.6745)
            idx_spikes = np.where(trace < thresh)[0]
        elif detect_sign == 1:
            thresh = detect_threshold * np.median(np.abs(trace) / 0.6745)
            idx_spikes = np.where(trace > thresh)[0]
        else:
            thresh = detect_threshold * np.median(np.abs(trace) / 0.6745)
            idx_spikes = np.where((trace > thresh) | (trace < -thresh))[0]
        intervals = np.diff(idx_spikes)

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
                        peak_val = np.min(spike_up)
                    elif detect_sign == 1:
                        peak_idx = np.argmax(spike_up)
                        peak_val = np.max(spike_up)
                    else:
                        peak_idx = np.argmax(np.abs(spike_up))
                        peak_val = np.max(np.abs(spike_up))

                    min_time_up = t_spike_up[peak_idx]
                    sp_times[i].append(int(min_time_up))
                    sp_amplitudes[i].append(peak_val)
                else:
                    sp_times[i].append(idx_spike)
                    sp_amplitudes[i].append(trace[idx_spike])

    return sp_times, sp_amplitudes
