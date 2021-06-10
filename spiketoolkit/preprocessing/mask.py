from spikeextractors import RecordingExtractor
from spikeextractors.extraction_tools import check_get_traces_args
import numpy as np
from .basepreprocessorrecording import BasePreprocessorRecordingExtractor


class MaskRecording(BasePreprocessorRecordingExtractor):
    preprocessor_name = 'Mask'

    def __init__(self, recording, bool_mask):
        if not isinstance(recording, RecordingExtractor):
            raise ValueError("'recording' must be a RecordingExtractor")
        self._mask = bool_mask
        assert len(bool_mask) == recording.get_num_frames(), "'bool_mask' should be a boolean array with length of " \
                                                             "number of frames"
        assert np.array(bool_mask).dtype in (bool, np.bool), "'bool_mask' should be a boolean array"
        self.is_dumpable = False
        BasePreprocessorRecordingExtractor.__init__(self, recording)
        self._kwargs = {'recording': recording.make_serialized_dict(), 'bool_mask': bool_mask}

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None, return_scaled=True):
        traces = self._recording.get_traces(channel_ids=channel_ids,
                                            start_frame=start_frame,
                                            end_frame=end_frame,
                                            return_scaled=return_scaled)

        traces = traces.copy()  # takes care of memmap objects
        traces[:, ~self._mask[start_frame:end_frame]] = 0.0
        return traces


def mask(recording, bool_mask):
    '''
    Apply a boolean mask to the recording, where False elements of the mask cause the associated recording frames to
    be set to 0

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to be transformed
    bool_mask: list or numpy array
        Boolean values of the same length as the recording

    Returns
    -------
    masked_traces: MaskTracesRecording
        The masked traces recording extractor object
    '''
    return MaskRecording(
        recording=recording, bool_mask=bool_mask
    )
