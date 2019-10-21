from .filterrecording import FilterRecording
import numpy as np


class WhitenRecording(FilterRecording):

    preprocessor_name = 'Whiten'
    installed = True  # check at class level if installed or not
    preprocessor_gui_params = [
        {'name': 'chunk_size', 'type': 'int', 'value': 30000, 'default': 30000, 'title':
            "Chunk size for the filter."},
        {'name': 'cache_chunks', 'type': 'bool', 'value': False, 'default': False, 'title':
            "If True filtered traces are computed and cached"},
         {'name': 'seed', 'type': 'int', 'value': 0, 'default': 0, 
          'title': "Random seed for reproducibility."},
    ]
    installation_mesg = ""  # err

    def __init__(self, recording, chunk_size=30000, cache_chunks=False, seed=0):
        self._recording = recording
        self._whitening_matrix = self._compute_whitening_matrix(seed=seed)
        FilterRecording.__init__(self, recording=recording, chunk_size=chunk_size, cache_chunks=cache_chunks)

    def _get_random_data_for_whitening(self, num_chunks=50, chunk_size=500, seed=0):
        N = self._recording.get_num_frames()
        random_ints = np.random.RandomState(seed=seed).randint(0, N - chunk_size, size=num_chunks)
        chunk_list = []
        for ff in random_ints:
            chunk = self._recording.get_traces(start_frame=ff,
                                               end_frame=ff + chunk_size)
            chunk_list.append(chunk)
        return np.concatenate(chunk_list, axis=1)

    def _compute_whitening_matrix(self, seed):
        data = self._get_random_data_for_whitening(seed=seed)
        
        # center the data
        data = data - np.mean(data, axis=1, keepdims=True)
        
        # Original by Jeremy
        AAt = data @ np.transpose(data)
        AAt = AAt / data.shape[1]
        U, S, Ut = np.linalg.svd(AAt, full_matrices=True)
        W = (U @ np.diag(1 / np.sqrt(S))) @ Ut
        
        # proposed by Alessio
        # AAt = data @ data.T / data.shape[1]
        # D, V = np.linalg.eig(AAt)
        # W = np.dot(np.diag(1.0 / np.sqrt(D + 1e-10)), V)
        
        return W

    def filter_chunk(self, *, start_frame, end_frame):
        chunk = self._recording.get_traces(start_frame=start_frame, end_frame=end_frame)
        chunk = chunk - np.mean(chunk, axis=1, keepdims=True)
        chunk2 = self._whitening_matrix @ chunk
        return chunk2


def whiten(recording, chunk_size=30000, cache_chunks=False, seed=0):
    '''
    Whitens the recording extractor traces.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to be whitened.
    chunk_size: int
        The chunk size to be used for the filtering.
    cache_chunks: bool
        If True, filtered traces are computed and cached all at once (default False).
    seed: int
        Random seed for reproducibility
    Returns
    -------
    whitened_recording: WhitenRecording
        The whitened recording extractor

    '''
    return WhitenRecording(
        recording=recording,
        chunk_size=chunk_size,
        cache_chunks=cache_chunks,
        seed=seed
    )
