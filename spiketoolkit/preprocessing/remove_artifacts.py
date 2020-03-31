from spikeextractors import RecordingExtractor
from spikeextractors.extraction_tools import check_get_traces_args
import numpy as np

class RemoveArtifactsRecording(RecordingExtractor):

    preprocessor_name = 'RemoveArtifacts'
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # err

    def __init__(self, recording, triggers, ms_before=0.5, ms_after=3):
        if not isinstance(recording, RecordingExtractor):
            raise ValueError("'recording' must be a RecordingExtractor")
        self._recording = recording
        self._triggers = np.array(triggers)
        self._ms_before = ms_before
        self._ms_after = ms_after
        RecordingExtractor.__init__(self)
        self.copy_channel_properties(recording=self._recording)

        self._kwargs = {'recording': recording.make_serialized_dict(), 'triggers': triggers,
                        'ms_before': ms_before, 'ms_after': ms_after}

    def get_sampling_frequency(self):
        return self._recording.get_sampling_frequency()

    def get_num_frames(self):
        return self._recording.get_num_frames()

    def get_channel_ids(self):
        return self._recording.get_channel_ids()

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        traces = self._recording.get_traces(channel_ids=channel_ids, start_frame=start_frame, end_frame=end_frame)
        triggers = self._triggers[(self._triggers > start_frame) & (self._triggers < end_frame)] - start_frame

        pad = [int(self._ms_before * self.get_sampling_frequency() / 1000),
               int(self._ms_after * self.get_sampling_frequency() / 1000)]

        for trig in triggers:
            if trig - pad[0] > 0 and trig + pad[1] < end_frame - start_frame:
                traces[:, trig - pad[0]:trig + pad[1]] = 0
            elif trig - pad[0] <= 0 and trig + pad[1] >= end_frame - start_frame:
                traces = 0
            elif trig - pad[0] <= 0:
                traces[:, :trig + pad[1]] = 0
            elif trig + pad[1] >= end_frame - start_frame:
                traces[:, trig - pad[0]:] = 0
        return traces


def remove_artifacts(recording, triggers, ms_before=0.5, ms_after=3):
    '''
    Removes stimulation artifacts from recording extractor traces. Artifact periods are zeroed-out.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to remove artifacts from
    triggers: list
        List of int with the stimulation trigger frames
    ms_before: float
        Time interval in ms to remove before the trigger events
    ms_after: float
        Time interval in ms to remove after the trigger events

    Returns
    -------
    removed_recording: RemoveArtifactsRecording
        The recording extractor after artifact removal

    '''
    return RemoveArtifactsRecording(
        recording=recording, triggers=triggers, ms_before=ms_before, ms_after=ms_after
    )
