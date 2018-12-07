from .filterrecording import FilterRecording
import numpy as np
from scipy import special

class BandpassFilterRecording(FilterRecording):
    def __init__(self, *, recording, freq_min, freq_max, freq_wid):
        FilterRecording.__init__(self, recording=recording, chunk_size=3000 * 10)
        self._recording = recording
        self._freq_min = freq_min
        self._freq_max = freq_max
        self._freq_wid = freq_wid
        self.copyChannelProperties(recording)

    def filterChunk(self, *, start_frame, end_frame):
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
        relwid = 3.0;  # relative bottom-end roll-off width param, kills low freqs by factor 1e-5.

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
        samplerate = self._recording.getSamplingFrequency()
        M = chunk.shape[0]
        chunk2 = chunk
        # Subtract off the mean of each channel unless we are doing only a low-pass filter
        # if self._freq_min!=0:
        #    for m in range(M):
        #        chunk2[m,:]=chunk2[m,:]-np.mean(chunk2[m,:])
        # Do the actual filtering with a DFT with real input
        chunk_fft = np.fft.rfft(chunk2)
        kernel = self._create_filter_kernel(
            chunk2.shape[1],
            samplerate,
            self._freq_min, self._freq_max, self._freq_wid
        )
        kernel = kernel[0:chunk_fft.shape[1]]  # because this is the DFT of real data
        chunk_fft = chunk_fft * np.tile(kernel, (M, 1))
        chunk_filtered = np.fft.irfft(chunk_fft)
        return chunk_filtered

    def _read_chunk(self, i1, i2):
        M = len(self._recording.getChannelIds())
        N = self._recording.getNumFrames()
        if i1 < 0:
            i1b = 0
        else:
            i1b = i1
        if i2 > N:
            i2b = N
        else:
            i2b = i2
        ret = np.zeros((M, i2 - i1))
        ret[:, i1b - i1:i2b - i1] = self._recording.getTraces(start_frame=i1b, end_frame=i2b)
        return ret


def bandpass_filter(recording, freq_min=300, freq_max=6000, freq_wid=1000, resample=None):
    return BandpassFilterRecording(
        recording=recording,
        freq_min=freq_min,
        freq_max=freq_max,
        freq_wid=freq_wid,
    )
