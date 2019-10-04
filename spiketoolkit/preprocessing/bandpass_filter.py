from .filterrecording import FilterRecording
import numpy as np
from scipy import special
import spikeextractors as se

try:
    import scipy.signal as ss
    HAVE_BFR = True
except ImportError:
    HAVE_BFR = False


class BandpassFilterRecording(FilterRecording):

    preprocessor_name = 'BandpassFilter'
    installed = HAVE_BFR  # check at class level if installed or not
    preprocessor_gui_params = [
        {'name': 'freq_min', 'type': 'float', 'value': 300.0, 'default': 300.0, 'title': "High-pass frequency"},
        {'name': 'freq_max', 'type': 'float', 'value': 6000.0, 'default': 6000.0, 'title': "Low-pass frequency"},
        {'name': 'freq_wid', 'type': 'float', 'value': 1000.0, 'default': 1000.0, 'title':
            "Width of the filter (when type is 'fft')"},
        {'name': 'type', 'type': 'str', 'value': 'fft', 'default': 'fft', 'title': "Filter type ('fft' or 'butter')"},
        {'name': 'order', 'type': 'int', 'value': 3, 'default': 3, 'title': "Order of the filter (if 'butter')"},
        {'name': 'chunk_size', 'type': 'int', 'value': 30000, 'default': 30000, 'title':
            "Chunk size for the filter."},
        {'name': 'cache_chunks', 'type': 'bool', 'value': False, 'default': False, 'title':
            "If True fileterd chunk traces are computed and cached in memory"},
    ]
    installation_mesg = "To use the BandpassFilterRecording, install scipy: \n\n pip install scipy\n\n"  # err

    def __init__(self, recording, freq_min=300, freq_max=6000, freq_wid=1000, type='fft', order=3,
                 chunk_size=30000, cache_chunks=False):
        assert HAVE_BFR, "To use the BandpassFilterRecording, install scipy: \n\n pip install scipy\n\n"
        self._freq_min = freq_min
        self._freq_max = freq_max
        self._freq_wid = freq_wid
        self._type = type
        self._order = order
        self._chunk_size = chunk_size

        if self._type == 'butter':
            fn = recording.get_sampling_frequency() / 2.
            band = np.array([self._freq_min, self._freq_max]) / fn

            self._b, self._a = ss.butter(self._order, band, btype='bandpass')

            if not np.all(np.abs(np.roots(self._a)) < 1):
                raise ValueError('Filter is not stable')
        FilterRecording.__init__(self, recording=recording, chunk_size=chunk_size, cache_chunks=cache_chunks)
        self.copy_channel_properties(recording)

    def filter_chunk(self, *, start_frame, end_frame):
        padding = 3000
        i1 = start_frame - padding
        i2 = end_frame + padding
        padded_chunk = self._read_chunk(i1, i2)
        filtered_padded_chunk = self._do_filter(padded_chunk)
        return filtered_padded_chunk[:, start_frame - i1:end_frame - i1]

    def _do_filter(self, chunk):
        sampling_frequency = self._recording.get_sampling_frequency()
        M = chunk.shape[0]
        chunk2 = chunk
        # Do the actual filtering with a DFT with real input
        if self._type == 'fft':
            chunk_fft = np.fft.rfft(chunk2)
            kernel = _create_filter_kernel(
                chunk2.shape[1],
                sampling_frequency,
                self._freq_min, self._freq_max, self._freq_wid
            )
            kernel = kernel[0:chunk_fft.shape[1]]  # because this is the DFT of real data
            chunk_fft = chunk_fft * np.tile(kernel, (M, 1))
            chunk_filtered = np.fft.irfft(chunk_fft)
        elif self._type == 'butter':
            chunk_filtered = ss.filtfilt(self._b, self._a, chunk2, axis=1)

        return chunk_filtered

    def _read_chunk(self, i1, i2):
        M = len(self._recording.get_channel_ids())
        N = self._recording.get_num_frames()
        if i1 < 0:
            i1b = 0
        else:
            i1b = i1
        if i2 > N:
            i2b = N
        else:
            i2b = i2
        ret = np.zeros((M, i2 - i1))
        ret[:, i1b - i1:i2b - i1] = self._recording.get_traces(start_frame=i1b, end_frame=i2b)

        return ret


def _create_filter_kernel(N, sampling_frequency, freq_min, freq_max, freq_wid=1000):
    # Matches ahb's code /matlab/processors/ms_bandpass_filter.m
    # improved ahb, changing tanh to erf, correct -3dB pts  6/14/16
    T = N / sampling_frequency  # total time
    df = 1 / T  # frequency grid
    relwid = 3.0  # relative bottom-end roll-off width param, kills low freqs by factor 1e-5.

    k_inds = np.arange(0, N)
    k_inds = np.where(k_inds <= (N + 1) / 2, k_inds, k_inds - N)

    fgrid = df * k_inds
    absf = np.abs(fgrid)

    val = np.ones(fgrid.shape)
    if freq_min != 0:
        val = val * (1 + special.erf(relwid * (absf - freq_min) / freq_min)) / 2
        val = np.where(np.abs(k_inds) < 0.1, 0, val)  # kill DC part exactly
    if freq_max != 0:
        val = val * (1 - special.erf((absf - freq_max) / freq_wid)) / 2;
    val = np.sqrt(val)  # note sqrt of filter func to apply to spectral intensity not ampl
    return val


def bandpass_filter(recording, freq_min=300, freq_max=6000, freq_wid=1000, type='fft', order=3,
                    chunk_size=30000, cache_to_file=False, cache_chunks=False):
    '''
    Performs a lazy filter on the recording extractor traces.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to be filtered.
    freq_min: int or float
        High-pass cutoff frequency.
    freq_max: int or float
        Low-pass cutoff frequency.
    freq_wid: int or float
        Width of the filter (when type is 'fft').
    type: str
        'fft' or 'butter'. The 'fft' filter uses a kernel in the frequency domain. The 'butter' filter uses
        scipy butter and filtfilt functions.
    order: int
        Order of the filter (if 'butter').
    chunk_size: int
        The chunk size to be used for the filtering.
    cache_to_file: bool (default False).
        If True, filtered traces are computed and cached all at once on disk in temp file 
    cache_chunks: bool (default False).
        If True then each chunk is cached in memory (in a dict)
    Returns
    -------
    filter_recording: BandpassFilterRecording
        The filtered recording extractor object
    '''
    if cache_to_file:
        assert not cache_chunks, 'if cache_to_file cache_chunks should be False'
    
    bpf_recording = BandpassFilterRecording(
        recording=recording,
        freq_min=freq_min,
        freq_max=freq_max,
        freq_wid=freq_wid,
        type=type,
        order=order,
        chunk_size=chunk_size,
        cache_chunks=cache_chunks,
    )
    if cache_to_file:
        return se.CacheRecordingExtractor(bpf_recording, chunk_size=chunk_size)
    else:
        return bpf_recording
