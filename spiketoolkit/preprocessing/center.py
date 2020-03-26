from spikeextractors import RecordingExtractor
from .transform import TransformRecording


class CenterRecording(TransformRecording):

    preprocessor_name = 'Center'
    installed = True  # check at class level if installed or not
    preprocessor_gui_params = []
    installation_mesg = ""  # err

    def __init__(self, recording):
        if not isinstance(recording, RecordingExtractor):
            raise ValueError("'recording' must be a RecordingExtractor")
        self._recording = recording
        self._scalar = 1
        self._offset = -self._recording.get_traces().mean(axis=1)
        TransformRecording.__init__(self, self._recording, scalar=self._scalar, offset=self._offset)
        self.copy_channel_properties(recording=self._recording)
        self._kwargs = {'recording': recording.make_serialized_dict(), 'scalar': self._scalar, 'offset': self._offset}


def center(recording):
    '''
    Removes the offset of the traces channel by channel.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to be transformed
    Returns
    -------
    rcenter: CenterRecording
        The output recording extractor object
    '''
    return CenterRecording(recording=recording)
