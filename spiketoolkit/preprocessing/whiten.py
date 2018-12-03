from .filterrecording import FilterRecording
import numpy as np


class WhitenRecording(FilterRecording):
    def __init__(self, *, recording):
        FilterRecording.__init__(self, recording=recording, chunk_size=1000)
        self._recording = recording
        self._whitening_matrix = self._compute_whitening_matrix()

    def _get_random_data_for_whitening(self, num_chunks=50, chunk_size=500):
        N = self._recording.getNumFrames()
        list = []
        for i in range(num_chunks):
            ff = np.random.randint(0, N - chunk_size)
            chunk = self._recording.getTraces(start_frame=ff, end_frame=ff + chunk_size)
            list.append(chunk)
        return np.concatenate(list, axis=1)

    def _compute_whitening_matrix(self):
        data = self._get_random_data_for_whitening()
        AAt = data @ np.transpose(data)
        AAt = AAt / data.shape[1]
        U, S, Ut = np.linalg.svd(AAt, full_matrices=True)
        W = (U @ np.diag(1 / np.sqrt(S))) @ Ut
        return W

    def filterChunk(self, *, start_frame, end_frame):
        chunk = self._recording.getTraces(start_frame=start_frame, end_frame=end_frame)
        chunk2 = self._whitening_matrix @ chunk
        return chunk2


def whiten(recording):
    return WhitenRecording(
        recording=recording
    )
