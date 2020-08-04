from .filterrecording import FilterRecording
import spikeextractors as se
import numpy as np
import scipy.signal as ss


class NotchFilterRecording(FilterRecording):

    preprocessor_name = 'NotchFilter'
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # error message when not installed

    def __init__(self, recording, freq=3000, q=30, chunk_size=30000, cache_chunks=False):
        self._freq = freq
        self._q = q
        fn = 0.5 * float(recording.get_sampling_frequency())
        self._b, self._a = ss.iirnotch(self._freq / fn, self._q)

        if not np.all(np.abs(np.roots(self._a)) < 1):
            raise ValueError('Filter is not stable')
        FilterRecording.__init__(self, recording=recording, chunk_size=chunk_size, cache_chunks=cache_chunks)
        self.copy_channel_properties(recording)
        self.is_filtered = self._recording.is_filtered

        self._kwargs = {'recording': recording.make_serialized_dict(), 'freq': freq,
                        'q': q, 'chunk_size': chunk_size, 'cache_chunks': cache_chunks}

    def filter_chunk(self, *, start_frame, end_frame, channel_ids):
        padding = 3000
        i1 = start_frame - padding
        i2 = end_frame + padding
        padded_chunk = self._read_chunk(i1, i2, channel_ids)
        filtered_padded_chunk = self._do_filter(padded_chunk)
        return filtered_padded_chunk[:, start_frame - i1:end_frame - i1]

    def _do_filter(self, chunk):
        chunk_filtered = ss.filtfilt(self._b, self._a, chunk, axis=1)

        return chunk_filtered


def notch_filter(recording, freq=3000, q=30, chunk_size=30000, cache_to_file=False, cache_chunks=False):
    '''
    Performs a notch filter on the recording extractor traces using scipy iirnotch function.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to be notch-filtered.
    freq: int or float
        The target frequency of the notch filter.
    q: int
        The quality factor of the notch filter.
    chunk_size: int
        The chunk size to be used for the filtering.
    cache_to_file: bool (default False).
        If True, filtered traces are computed and cached all at once on disk in temp file 
    cache_chunks: bool (default False).
        If True then each chunk is cached in memory (in a dict)
    Returns
    -------
    filter_recording: NotchFilterRecording
        The notch-filtered recording extractor object
    '''

    if cache_to_file:
        assert not cache_chunks, 'if cache_to_file cache_chunks should be False'
    
    notch_recording =  NotchFilterRecording(
        recording=recording,
        freq=freq,
        q=q,
        chunk_size=chunk_size,
        cache_chunks=cache_chunks,
    )
    if cache_to_file:
        return se.CacheRecordingExtractor(notch_recording, chunk_size=chunk_size)
    else:
        return notch_recording
