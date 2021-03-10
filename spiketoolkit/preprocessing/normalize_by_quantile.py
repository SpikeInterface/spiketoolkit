from spikeextractors import RecordingExtractor
import numpy as np
from .basepreprocessorrecording import BasePreprocessorRecordingExtractor
from spikeextractors.extraction_tools import check_get_traces_args


class NormalizeByQuantileRecording(BasePreprocessorRecordingExtractor):
    preprocessor_name = 'NormalizeByQuantile'

    def __init__(self, recording, scale=1.0, median=0.0, q1=0.01, q2=0.99, seed=0):
        BasePreprocessorRecordingExtractor.__init__(self, recording)

        random_data = self._get_random_data_for_scaling(seed=seed).ravel()
        loc_q1, pre_median, loc_q2 = np.quantile(random_data, q=[q1, 0.5, q2])
        pre_scale = abs(loc_q2 - loc_q1)

        self._scalar = scale / pre_scale
        self._offset = median - pre_median * self._scalar
        self.has_unscaled = False
        self._kwargs = {'recording': recording.make_serialized_dict(), 'scale': scale, 'median': median,
                        'q1': q1, 'q2': q2, 'seed': seed}

    def _get_random_data_for_scaling(self, num_chunks=50, chunk_size=500, seed=0):
        N = self._recording.get_num_frames()
        random_ints = np.random.RandomState(seed=seed).randint(0, N - chunk_size, size=num_chunks)
        chunk_list = []
        for ff in np.sort(random_ints):
            chunk = self._recording.get_traces(start_frame=ff,
                                               end_frame=ff + chunk_size)
            chunk_list.append(chunk)
        return np.concatenate(chunk_list, axis=1)

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None, return_scaled=True):
        assert return_scaled, "'normalize_by_quantile' only supports return_scaled=True"

        traces = self._recording.get_traces(channel_ids=channel_ids,
                                            start_frame=start_frame,
                                            end_frame=end_frame,
                                            return_scaled=return_scaled)
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
