from .filterrecording import FilterRecording
import numpy as np
from scipy import special
from scipy.signal import butter, filtfilt

class BandpassFilterRecording(FilterRecording):
    def __init__(self, recording, freq_min=300, freq_max=6000, freq_wid=1000, type='fft', order=3):
        FilterRecording.__init__(self, recording=recording, chunk_size=3000 * 10)
        self._recording = recording
        self._freq_min = freq_min
        self._freq_max = freq_max
        self._freq_wid = freq_wid
        self._type = type
        self._order = order
        self.copy_channel_properties(recording)

    def filter_chunk(self, *, start_frame, end_frame):
        padding = 3000
        i1 = start_frame - padding
        i2 = end_frame + padding
        padded_chunk = self._read_chunk(i1, i2)
        filtered_padded_chunk = self._do_filter(padded_chunk)
        return filtered_padded_chunk[:, start_frame - i1:end_frame - i1]

    def _create_filter_kernel(self, N, samplerate, freq_min, freq_max, freq_wid=1000):
        # Matches ahb's code /matlab/processors/ms_bandpass_filter.m
        # improved ahb, changing tanh to erf, correct -3dB pts  6/14/16
        T = N / samplerate  # total time
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

    def _do_filter(self, chunk):
        samplerate = self._recording.get_sampling_frequency()
        M = chunk.shape[0]
        chunk2 = chunk
        # Do the actual filtering with a DFT with real input
        if self._type == 'fft':
            chunk_fft = np.fft.rfft(chunk2)
            kernel = self._create_filter_kernel(
                chunk2.shape[1],
                samplerate,
                self._freq_min, self._freq_max, self._freq_wid
            )
            kernel = kernel[0:chunk_fft.shape[1]]  # because this is the DFT of real data
            chunk_fft = chunk_fft * np.tile(kernel, (M, 1))
            chunk_filtered = np.fft.irfft(chunk_fft)
        elif self._type == 'butter':
            fn = self.get_sampling_frequency() / 2.
            band = np.array([self._freq_min, self._freq_max]) / fn

            b, a = butter(self._order, band, btype='bandpass')

            if np.all(np.abs(np.roots(a)) < 1) and np.all(np.abs(np.roots(a)) < 1):
                chunk_filtered = filtfilt(b, a, chunk2, axis=1)
            else:
                raise ValueError('Filter is not stable')
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


def bandpass_filter(recording, freq_min=300, freq_max=6000, freq_wid=1000, type='fft', order=3):
    return BandpassFilterRecording(
        recording=recording,
        freq_min=freq_min,
        freq_max=freq_max,
        freq_wid=freq_wid,
        type=type,
        order=order
    )
