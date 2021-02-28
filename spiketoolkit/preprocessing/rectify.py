from spikeextractors import RecordingExtractor
from spikeextractors.extraction_tools import check_get_traces_args
from .basepreprocessorrecording import BasePreprocessorRecordingExtractor
import numpy as np


class RectifyRecording(BasePreprocessorRecordingExtractor):
    preprocessor_name = 'Rectify'

    def __init__(self, recording):
        BasePreprocessorRecordingExtractor.__init__(self, recording)
        self._kwargs = {'recording': recording.make_serialized_dict()}

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None, return_scaled=True):
        return np.abs(self._recording.get_traces(channel_ids=channel_ids, start_frame=start_frame, end_frame=end_frame,
                                                 return_scaled=return_scaled))


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
