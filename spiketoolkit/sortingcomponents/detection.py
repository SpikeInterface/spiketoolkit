from joblib import Parallel, delayed
import spikeextractors as se
from ..postprocessing.postprocessing_tools import divide_recording_into_time_chunks
import itertools
from tqdm import tqdm
import numpy as np


def detect_spikes(recording, channel_ids=None, detect_threshold=5, detect_sign=-1,
                  n_shifts=2, n_snippets_for_threshold=10, snippet_size_sec=1,
                  start_frame=None, end_frame=None, n_jobs=1, joblib_backend='loky',
                  chunk_size=None, chunk_mb=500, verbose=False):
    '''
    Detects spikes per channel. Spikes are detected as threshold crossings and the threshold is in terms of the median
    average deviation (MAD). The MAD is computed by taking 'n_snippets_for_threshold' snippets of the recordings
    of 'snippet_size_sec' seconds uniformly distributed between 'start_frame' and 'end_frame'.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object
    channel_ids: list or None
        List of channels to perform detection. If None all channels are used
    detect_threshold: float
        Threshold in median absolute deviations (MAD) to detect peaks
    n_shifts: int
        Number of shifts to find peak. E.g. if n_shift is 2, a peak is detected (if detect_sign is 'negative') if
        a sample is below the threshold, the two samples before are higher than the sample, and the two samples after
        the sample are higher than the sample.
    n_snippets_for_threshold: int
        Number of snippets to use to compute channel-wise thresholds
    snippet_size_sec: float
        Length of each snippet in seconds
    detect_sign: int
        Sign of the detection: -1 (negative), 1 (positive), 0 (both)
    start_frame: int
        Start frame for detection
    end_frame: int
        End frame end frame for detection
    n_jobs: int
        Number of jobs for parallelization. Default is None (no parallelization)
    joblib_backend: str
        The backend for joblib. Default is 'loky'
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

    snippet_len = int(snippet_size_sec * recording.get_sampling_frequency())
    reference_frames = np.linspace(snippet_len+1, recording.get_num_frames() - snippet_len,
                                   n_snippets_for_threshold)
    snippets = recording.get_snippets(reference_frames=reference_frames, snippet_len=snippet_len)
    traces_mad = np.concatenate(snippets, 1)
    thresholds = detect_threshold * np.median(np.abs(traces_mad) / 0.6745, 1)[:, None]

    if n_jobs > 1:
        output = Parallel(n_jobs=n_jobs, backend=joblib_backend)(delayed(_detect_and_align_peaks_chunk)
                                                                 (ii, rec_arg, chunks, channel_ids, thresholds,
                                                                  detect_sign,
                                                                  n_shifts, verbose)
                                                                 for ii in chunk_iter)
        for ii, (times_ii, amps_ii) in enumerate(output):
            for i, ch in enumerate(channel_ids):
                times = times_ii[i]
                amps = amps_ii[i]
                all_channel_amps[i].append(amps)
                all_channel_times[i].append(times)
    else:
        for ii in chunk_iter:
            times_ii, amps_ii = _detect_and_align_peaks_chunk(ii, rec_arg, chunks, channel_ids, thresholds,
                                                              detect_sign, n_shifts, False)

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


def _detect_and_align_peaks_chunk(ii, rec_arg, chunks, channel_ids, thresholds, detect_sign, n_shifts,
                                  verbose):
    chunk = chunks[ii]

    if verbose:
        print(f"Chunk {ii + 1}: detecting spikes")
    if isinstance(rec_arg, dict):
        recording = se.load_extractor_from_dict(rec_arg)
    else:
        recording = rec_arg

    traces = recording.get_traces(start_frame=chunk['istart'],
                                  end_frame=chunk['iend'])

    if detect_sign == -1:
        traces = -traces
    elif detect_sign == 0:
        traces = np.abs(traces)

    sig_center = traces[:, n_shifts:-n_shifts]
    peak_mask = sig_center > thresholds
    for i in range(n_shifts):
        peak_mask &= sig_center > traces[:, i:i + sig_center.shape[1]]
        peak_mask &= sig_center >= traces[:, n_shifts + i + 1:n_shifts + i + 1 + sig_center.shape[1]]

    # find peaks
    peak_chan_ind, peak_sample_ind = np.nonzero(peak_mask)
    # correct for time shift
    peak_sample_ind += n_shifts

    sp_times = []
    sp_amplitudes = []

    for ch in range(len(channel_ids)):
        peak_times = peak_sample_ind[np.where(peak_chan_ind == ch)]
        sp_times.append(peak_sample_ind[np.where(peak_chan_ind == ch)] + chunk['istart'])
        sp_amplitudes.append(traces[ch, peak_times])

    return sp_times, sp_amplitudes
