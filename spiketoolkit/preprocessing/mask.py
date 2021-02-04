from spikeextractors import RecordingExtractor
from spikeextractors.extraction_tools import check_get_traces_args
import numpy as np
from .basepreprocessorrecording import BasePreprocessorRecordingExtractor


class MaskRecording(BasePreprocessorRecordingExtractor):
    preprocessor_name = 'Mask'

    def __init__(self, recording, mask):
        if not isinstance(recording, RecordingExtractor):
            raise ValueError("'recording' must be a RecordingExtractor")
        self._mask = mask
        assert len(mask) == recording.get_num_frames(), "'mask' should be a boolean array with length of " \
                                                        "number of frames"
        assert np.array(mask).dtype in (bool, np.bool), "'mask' should be a boolean array"
        BasePreprocessorRecordingExtractor.__init__(self, recording)
        self._kwargs = {'recording': recording.make_serialized_dict(), 'mask': mask}

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        traces = self._recording.get_traces(channel_ids=channel_ids,
                                            start_frame=start_frame,
                                            end_frame=end_frame)

        traces = traces.copy()  # takes care of memmap objects
        traces[:, ~self._mask[start_frame:end_frame]] = 0.0
        return traces


def mask(recording, mask):
    '''
    Apply a boolean mask to the recording, where False elements of the mask case the associated recording frames to
    be set to 0

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to be transformed
    mask: list or numpy array
        Boolean values of the same length as the recording

    Returns
    -------
    masked_traces: MaskTracesRecording
        The masked traces recording extractor object
    '''
    return MaskRecording(
        recording=recording, mask=mask
    )
