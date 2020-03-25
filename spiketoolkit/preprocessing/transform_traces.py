from spikeextractors import RecordingExtractor
import numpy as np

class TransformTracesRecording(RecordingExtractor):

    preprocessor_name = 'TransformTraces'
    installed = True  # check at class level if installed or not
    preprocessor_gui_params = [
        {'name': 'scalar', 'type': 'float or array', 'title': "Scalar for all traces or each channel "
                                                              "of the recording extractor"},
        {'name': 'offset', 'type': 'float or array', 'title': "Offset for all traces or each channel"
                                                              "of the recording extractor"},
    ]
    installation_mesg = ""  # err

    def __init__(self, recording, scalar=1, offset=0):
        if not isinstance(recording, RecordingExtractor):
            raise ValueError("'recording' must be a RecordingExtractor")
        self._recording = recording
        self._scalar = scalar
        self._offset = offset
        RecordingExtractor.__init__(self)
        self.copy_channel_properties(recording=self._recording)

        self._kwargs = {'recording': recording.make_serialized_dict(), 'scalar': scalar, 'offset': offset}

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
        traces = self._recording.get_traces(channel_ids=channel_ids, start_frame=start_frame, end_frame=end_frame)
        if isinstance(self._scalar, (int, float, np.integer, np.float)):
            traces = traces*self._scalar
        else:
            if len(self._scalar) == len(channel_ids):
                scalar = self._scalar
            else:
                channel_idxs = np.array([self._recording.get_channel_ids().index(ch) for ch in channel_ids])
                scalar = self._scalar[channel_idxs]
            traces = np.array([t * s for (t, s) in zip(traces, scalar)])
        if isinstance(self._offset, (int, float, np.integer, np.float)):
            traces = traces + self._offset
        else:
            if len(self._offset) == len(channel_ids):
                offset = self._offset
            else:
                channel_idxs = np.array([self._recording.get_channel_ids().index(ch) for ch in channel_ids])
                offset = self._offset[channel_idxs]
            traces = np.array([t + o for (t, o) in zip(traces, offset)])
        return traces


def transform_traces(recording, scalar=1, offset=0):
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
    return TransformTracesRecording(
        recording=recording, scalar=scalar, offset=offset
    )
