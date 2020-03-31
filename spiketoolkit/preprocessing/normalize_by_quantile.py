from spikeextractors import RecordingExtractor
import numpy as np
from spikeextractors.extraction_tools import check_get_traces_args

class NormalizeByQuantileRecording(RecordingExtractor):

    preprocessor_name = 'NormalizeByQuantile'
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # err

    def __init__(self, recording, scale=1.0, median=0.0, q1=0.01, q2=0.99, seed=0):
        if not isinstance(recording, RecordingExtractor):
            raise ValueError("'recording' must be a RecordingExtractor")
        self._recording = recording

        random_data = self._get_random_data_for_scaling(seed=seed).ravel()
        loc_q1, pre_median, loc_q2 = np.quantile(random_data, q=[q1, 0.5, q2])
        pre_scale = abs(loc_q2 - loc_q1)

        self._scalar = scale / pre_scale
        self._offset = median - pre_median * self._scalar
        RecordingExtractor.__init__(self)
        self.copy_channel_properties(recording=self._recording)

        self._kwargs = {'recording': recording.make_serialized_dict(), 'scale': scale, 'median': median,
                        'q1': q1, 'q2': q2, 'seed': seed}

    def _get_random_data_for_scaling(self, num_chunks=50, chunk_size=500, seed=0):
        N = self._recording.get_num_frames()
        random_ints = np.random.RandomState(seed=seed).randint(0, N - chunk_size, size=num_chunks)
        chunk_list = []
        for ff in random_ints:
            chunk = self._recording.get_traces(start_frame=ff,
                                               end_frame=ff + chunk_size)
            chunk_list.append(chunk)
        return np.concatenate(chunk_list, axis=1)

    def get_sampling_frequency(self):
        return self._recording.get_sampling_frequency()

    def get_num_frames(self):
        return self._recording.get_num_frames()

    def get_channel_ids(self):
        return self._recording.get_channel_ids()

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        traces = self._recording.get_traces(channel_ids=channel_ids,
                                            start_frame=start_frame,
                                            end_frame=end_frame)
        return traces * self._scalar + self._offset


def normalize_by_quantile(recording, scale=1.0, median=0.0, q1=0.01, q2=0.99, seed=0):
    '''
    Rescale the traces from the given recording extractor with a scalar
    and offset. First, the median and quantiles of the distribution are estimated.
    Then the distribution is rescaled and offset so that the scale is given by the
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
        Upper quantile used for measuring the 
    seed: int
        Random seed for reproducibility
    Returns
    -------
    rescaled_traces: NormalizeByQuantileRecording
        The rescaled traces recording extractor object
    '''
    return NormalizeByQuantileRecording(
        recording=recording, 
        scale=scale, 
        median=median, 
        q1=q1, 
        q2=q2,
        seed=seed
    )
