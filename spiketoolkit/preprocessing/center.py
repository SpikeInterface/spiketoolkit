from spikeextractors import RecordingExtractor
from .transform import TransformRecording
import numpy as np


class CenterRecording(TransformRecording):

    preprocessor_name = 'Center'
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # err

    def __init__(self, recording, mode):
        if not isinstance(recording, RecordingExtractor):
            raise ValueError("'recording' must be a RecordingExtractor")
        self._recording = recording
        self._scalar = 1
        self._mode = mode
        assert self._mode in ['mean', 'median'], "'mode' can be 'mean' or 'median'"
        if self._mode == 'mean':
            self._offset = -np.mean(self._recording.get_traces(), axis=1)
        else:
            self._offset = -np.median(self._recording.get_traces(), axis=1)
        TransformRecording.__init__(self, self._recording, scalar=self._scalar, offset=self._offset)
        self.copy_channel_properties(recording=self._recording)
        self._kwargs = {'recording': recording.make_serialized_dict(), 'mode': self._mode}


def center(recording, mode='median'):
    '''
    Removes the offset of the traces channel by channel.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to be transformed
    mode: str
        'median' (default) or 'mean'
    Returns
    -------
    rcenter: CenterRecording
        The output recording extractor object
    '''
    return CenterRecording(recording=recording, mode=mode)
