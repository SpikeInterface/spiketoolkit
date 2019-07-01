from spikeextractors import RecordingExtractor
import numpy as np


class ClipTracesRecording(RecordingExtractor):

    preprocessor_name = 'ClipTracesRecording'
    installed = True  # check at class level if installed or not
    _gui_params = [
        {'name': 'a_min', 'type': 'float',
            'title': "Minimum value. If `None`, clipping is not performed on lower interval edge."},
        {'name': 'a_max', 'type': 'float',
            'title': "Maximum value. If `None`, clipping is not performed on upper interval edge."},
    ]
    installation_mesg = ""  # err

    def __init__(self, recording, a_min=None, a_max=None):
        RecordingExtractor.__init__(self)
        if not isinstance(recording, RecordingExtractor):
            raise ValueError("'recording' must be a RecordingExtractor")
        self._recording = recording
        self._a_min = a_min
        self._a_max = a_max
        self.copy_channel_properties(recording=self._recording)

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
        if self._a_min is not None:
            traces[traces<a_min] = self._a_min
        if self._a_max is not None:
            traces[traces>a_max] = self._a_max
        return traces


def clip_traces(recording, a_min=None, a_max=None):
    '''
    Limit the values of the data between a_min and a_max. Values exceeding the
    range will be set to the minimum or maximum, respectively.
    
    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to be transformed
    a_min: float or `None` (default `None`)
        Minimum value. If `None`, clipping is not performed on lower
        interval edge.
    a_max: float or `None` (default `None`)
        Maximum value. If `None`, clipping is not performed on upper
        interval edge.

    Returns
    -------
    rescaled_traces: ClipTracesRecording
        The clipped traces recording extractor object
    '''
    return ClipTracesRecording(
        recording = recording, a_min=a_min, a_max=a_max
    )
