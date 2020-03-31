from spikeextractors import RecordingExtractor
from spikeextractors.extraction_tools import check_get_traces_args
import numpy as np


class RectifyRecording(RecordingExtractor):

    preprocessor_name = 'Rectify'
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # err

    def __init__(self, recording):
        self._recording = recording
        RecordingExtractor.__init__(self)
        self.copy_channel_properties(recording)

        self._kwargs = {'recording': recording.make_serialized_dict()}

    def get_sampling_frequency(self):
        return self._recording.get_sampling_frequency()

    def get_num_frames(self):
        return self._recording.get_num_frames()

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
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
