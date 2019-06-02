from spikeextractors import RecordingExtractor
import numpy as np

class RectifyRecording(RecordingExtractor):

    preprocessor_name = 'RectifyRecording'
    installed = True  # check at class level if installed or not
    _gui_params = []
    installation_mesg = ""  # err

    def __init__(self, recording):
        RecordingExtractor.__init__(self)
        self._recording = recording
        self.copy_channel_properties(recording)

    def get_sampling_frequency(self):
        return self._recording.get_sampling_frequency()

    def get_num_frames(self):
        return self._recording.get_num_frames()

    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_frames()
        if channel_ids is None:
            channel_ids = self.get_channel_ids()
        return np.abs(self._recording.get_traces(channel_ids=channel_ids, start_frame=start_frame, end_frame=end_frame))

    def get_channel_ids(self):
        return self._recording.get_channel_ids()


def rectify(recording):
    '''
    Rectifies the recording extractor traces. It is useful, in combination with 'resample', to compute multi-unit
    activity (MUA).

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object to be rectified

    Returns
    -------
    rectified_recording: RectifyRecording
        The rectified recording extractor object

    '''
    return RectifyRecording(
        recording=recording
    )
