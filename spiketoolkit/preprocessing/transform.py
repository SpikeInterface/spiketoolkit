from spikeextractors import RecordingExtractor
from spikeextractors.extraction_tools import check_get_traces_args
import numpy as np


class TransformRecording(RecordingExtractor):

    preprocessor_name = 'Transform'
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # err

    def __init__(self, recording, scalar=1., offset=0., dtype=None):
        if not isinstance(recording, RecordingExtractor):
            raise ValueError("'recording' must be a RecordingExtractor")
        self._scalar = scalar
        self._offset = offset
        if dtype is None:
            self._dtype = recording.get_dtype()
        else:
            self._dtype = dtype
        RecordingExtractor.__init__(self)
        self._recording = recording
        self.copy_channel_properties(recording=self._recording)

        self._kwargs = {'recording': recording.make_serialized_dict(), 'scalar': scalar, 'offset': offset,
                        'dtype': dtype}

    def get_sampling_frequency(self):
        return self._recording.get_sampling_frequency()

    def get_num_frames(self):
        return self._recording.get_num_frames()

    def get_channel_ids(self):
        return self._recording.get_channel_ids()

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        traces = self._recording.get_traces(channel_ids=channel_ids, start_frame=start_frame, end_frame=end_frame)
        if isinstance(self._scalar, (int, float, np.integer, np.float)):
            traces = traces*self._scalar
        else:
            if len(self._scalar) == len(channel_ids):
                scalar = np.array(self._scalar)
            else:
                channel_idxs = np.array([self._recording.get_channel_ids().index(ch) for ch in channel_ids])
                scalar = np.array(self._scalar)[channel_idxs]
            traces = traces * scalar[:, np.newaxis]
        if isinstance(self._offset, (int, float, np.integer, np.float)):
            traces = traces + self._offset
        else:
            if len(self._offset) == len(channel_ids):
                offset = np.array(self._offset)
            else:
                channel_idxs = np.array([self._recording.get_channel_ids().index(ch) for ch in channel_ids])
                offset = np.array(self._offset)[channel_idxs]
            traces = traces + offset[:, np.newaxis]
        return traces.astype(self._dtype)


def transform(recording, scalar=1, offset=0):
    '''
    Transforms the traces from the given recording extractor with a scalar
    and offset. New traces = traces*scalar + offset.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to be transformed
    scalar: float
        Scalar for the traces of the recording extractor
    offset: float
        Offset for the traces of the recording extractor
    Returns
    -------
    transform_traces: TransformTracesRecording
        The transformed traces recording extractor object
    '''
    return TransformRecording(
        recording=recording, scalar=scalar, offset=offset
    )
