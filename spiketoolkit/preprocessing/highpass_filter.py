from .filterrecording import FilterRecording
import numpy as np
import scipy.signal as ss
from scipy import special
import spikeextractors as se


class HighpassFilterRecording(FilterRecording):
    preprocessor_name = 'HighpassFilter'

    def __init__(self, recording, freq_min=300, freq_wid=1000, filter_type='butter', order=1,
                 chunk_size=30000, cache_chunks=False, dtype=None):
        self._freq_min = freq_min
        self._freq_wid = freq_wid
        self._type = filter_type
        self._order = order
        self._chunk_size = chunk_size
        self._padding = 3000

        if self._type == 'butter':
            fn = recording.get_sampling_frequency() / 2.
            band = self._freq_min / fn

            self._b, self._a = ss.butter(self._order, band, btype='highpass')

            if not np.all(np.abs(np.roots(self._a)) < 1):
                raise ValueError('Filter is not stable')
        elif self._type == 'fft':
            self._kernel = _create_filter_kernel(
                self._chunk_size+2*self._padding,
                recording.get_sampling_frequency(),
                self._freq_min, self._freq_wid
            )
            self._kernel = self._kernel[0:(chunk_size+2*self._padding)//2+1]  # because this is the DFT of real data
        else:
            raise NotImplementedError('filter type {} not implemented.'.format(filter_type))
            
        FilterRecording.__init__(self, recording=recording, chunk_size=chunk_size, cache_chunks=cache_chunks,
                                 dtype=dtype)
        self.is_filtered = True
        self._kwargs = {'recording': recording.make_serialized_dict(), 'freq_min': freq_min,
                        'freq_wid': freq_wid, 'filter_type': filter_type, 'order': order,
                        'chunk_size': chunk_size, 'cache_chunks': cache_chunks}

    def filter_chunk(self, *, start_frame, end_frame, channel_ids, return_scaled):
        padding = 3000
        i1 = start_frame - self._padding
        i2 = end_frame + self._padding
        padded_chunk = self._read_chunk(i1, i2, channel_ids, return_scaled)
        filtered_padded_chunk = self._do_filter(padded_chunk)
        return filtered_padded_chunk[:, start_frame - i1:end_frame - i1]

    def _do_filter(self, chunk):
        sampling_frequency = self._recording.get_sampling_frequency()
        M = chunk.shape[0]
        chunk2 = chunk
        # Do the actual filtering with a DFT with real input
        if self._type == 'fft':
            chunk_fft = np.fft.rfft(chunk2)
            chunk_fft = chunk_fft * np.tile(self._kernel, (M, 1))
            chunk_filtered = np.fft.irfft(chunk_fft)
        elif self._type == 'butter':
            chunk_filtered = ss.filtfilt(self._b, self._a, chunk2, axis=1)

        return chunk_filtered


def _create_filter_kernel(N, sampling_frequency, freq_min, freq_wid=1000):
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
    val = np.sqrt(val)  # note sqrt of filter func to apply to spectral intensity not ampl
    return val


def highpass_filter(recording, freq_min=300, freq_wid=1000, filter_type='butter', order=1,
                    chunk_size=30000, cache_chunks=False, dtype=None):
    '''
    Performs a lazy filter on the recording extractor traces.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to be filtered.
    freq_min: int or float
        High-pass cutoff frequency.
    freq_wid: int or float
        Width of the filter (when type is 'fft').
    filter_type: str
        'fft' or 'butter'. The 'fft' filter uses a kernel in the frequency domain. The 'butter' filter uses
        scipy butter and filtfilt functions.
    order: int
        Order of the filter (if 'butter').
    chunk_size: int
        The chunk size to be used for the filtering.
    cache_chunks: bool (default False).
        If True then each chunk is cached in memory (in a dict)
    dtype: dtype
        The dtype of the traces

    Returns
    -------
    filter_recording: HighpassFilterRecording
        The filtered recording extractor object
    '''
    hp_recording = HighpassFilterRecording(
        recording=recording,
        freq_min=freq_min,
        freq_wid=freq_wid,
        filter_type=filter_type,
        order=order,
        chunk_size=chunk_size,
        cache_chunks=cache_chunks,
        dtype=dtype
    )
    return hp_recording
