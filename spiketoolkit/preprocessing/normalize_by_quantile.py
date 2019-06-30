from spikeextractors import RecordingExtractor
import numpy as np


class NormalizeByQuantileRecording(RecordingExtractor):

    preprocessor_name = 'NormalizeByQuantileRecording'
    installed = True  # check at class level if installed or not
    _gui_params = [
        {'name': 'scale', 'type': 'float',
            'title': "Scale for the output distribution"},
        {'name': 'median', 'type': 'float',
            'title': "Median for the output distribution"},
        {'name': 'q1', 'type': 'float',
            'title': "Lower quantile used for measuring the scale"},
        {'name': 'q2', 'type': 'float',
            'title': "Upper quantile used for measuring the scale"},
    ]
    installation_mesg = ""  # err

    def __init__(self, recording, scale=1.0, median=0.0, q1=0.01, q2=0.99):
        RecordingExtractor.__init__(self)
        if not isinstance(recording, RecordingExtractor):
            raise ValueError("'recording' must be a RecordingExtractor")
        self._recording = recording

        random_data = self._get_random_data_for_scaling().ravel()
        loc_q1, pre_median, loc_q2 = np.quantile(random_data, q=[q1, 0.5, q2])
        pre_scale = abs(loc_q2 - loc_q1)

        self._scalar = scale / pre_scale
        self._offset = median - pre_median * self._scalar
        self.copy_channel_properties(recording=self._recording)

    def _get_random_data_for_scaling(self, num_chunks=50, chunk_size=500):
        np.random.seed(0)
        N = self._recording.get_num_frames()
        list = []
        for i in range(num_chunks):
            ff = np.random.randint(0, N - chunk_size)
            chunk = self._recording.get_traces(start_frame=ff,
                                               end_frame=ff + chunk_size)
            list.append(chunk)
        return np.concatenate(list, axis=1)

    def get_sampling_frequency(self):
        return self._recording.get_sampling_frequency()

    def get_num_frames(self):
        return self._recording.get_num_frames()

    def get_channel_ids(self):
        return self._recording.get_channel_ids()

    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_frames()
        if channel_ids is None:
            channel_ids = self.get_channel_ids()
        traces = self._recording.get_traces(channel_ids=channel_ids,
                                            start_frame=start_frame,
                                            end_frame=end_frame)
        return traces * self._scalar + self._offset


def normalize_by_quantile(recording, scale=1.0, median=0.0, q1=0.01, q2=0.99):
    '''
    Rescale the traces from the given recording extractor with a scalar
    and offset. First, the median and quantiles of the distribution are estimated.
    Then the distribution is rescaled and offset so that the scale given by the
    distance between the quantiles (1st and 99th by default) is set to `scale`,
    and the median is set to the given median.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to be transformed
    scalar: float
        Scale for the output distribution
    median: float
        Median for the output distribution
    q1: float (default 0.01)
        Lower quantile used for measuring the scale
    q1: float (default 0.99)
        Upper quantile used for measuring the scale
    Returns
    -------
    rescaled_traces: NormalizeByQuantileRecording
        The rescaled traces recording extractor object
    '''
    return NormalizeByQuantileRecording(
        recording=recording, scale=scale, median=median, q1=q1, q2=q2
    )
