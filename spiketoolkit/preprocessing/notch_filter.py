from .filterrecording import FilterRecording
import spikeextractors as se
import numpy as np

try:
    import scipy.signal as ss
    HAVE_NFR = True
except ImportError:
    HAVE_NFR = False


class NotchFilterRecording(FilterRecording):

    preprocessor_name = 'NotchFilter'
    installed = HAVE_NFR  # check at class level if installed or not
    preprocessor_gui_params = [
        {'name': 'freq', 'type': 'float', 'value':3000.0, 'default':3000.0, 'title': "Frequency"},
        {'name': 'q', 'type': 'int', 'value':30, 'default':30, 'title': "Quality factor"},
        {'name': 'chunk_size', 'type': 'int', 'value': 30000, 'default': 30000, 'title':
            "Chunk size for the filter."},
        {'name': 'cache_chunks', 'type': 'bool', 'value': False, 'default': False, 'title':
            "If True filtered traces are computed and cached"},
    ]
    installation_mesg = "To use the NotchFilterRecording, install scipy: \n\n pip install scipy\n\n"  # error message when not installed

    def __init__(self, recording, freq=3000, q=30, chunk_size=30000, cache_chunks=False):
        assert HAVE_NFR, "To use the NotchFilterRecording, install scipy: \n\n pip install scipy\n\n"
        self._freq = freq
        self._q = q
        fn = 0.5 * float(recording.get_sampling_frequency())
        self._b, self._a = ss.iirnotch(self._freq / fn, self._q)

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
        chunk_filtered = ss.filtfilt(self._b, self._a, chunk, axis=1)

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
