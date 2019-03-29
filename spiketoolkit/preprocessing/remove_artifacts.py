from spikeextractors import RecordingExtractor
import numpy as np

class RemoveArtifactsRecording(RecordingExtractor):
    def __init__(self, recording, triggers, ms_before=0.5, ms_after=3):
        '''

        Parameters
        ----------
        recording
        reference
        groups
        '''
        RecordingExtractor.__init__(self)
        if not isinstance(recording, RecordingExtractor):
            raise ValueError("'recording' must be a RecordingExtractor")
        self._recording = recording
        self._triggers = triggers
        self._ms_before = ms_before
        self._ms_after = ms_after
        self.copyChannelProperties(recording=self._recording)

    def getSamplingFrequency(self):
        return self._recording.getSamplingFrequency()

    def getNumFrames(self):
        return self._recording.getNumFrames()

    def getChannelIds(self):
        return self._recording.getChannelIds()

    def getTraces(self, channel_ids=None, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.getNumFrames()
        if channel_ids is None:
            channel_ids = self.getChannelIds()
        traces = self._recording.getTraces(channel_ids=channel_ids, start_frame=start_frame, end_frame=end_frame)
        triggers = self._triggers[(self._triggers > start_frame) & (self._triggers < end_frame)] - start_frame

        pad = [int(self._ms_before * self.getSamplingFrequency() / 1000),
               int(self._ms_after * self.getSamplingFrequency() / 1000)]

        for trig in triggers:
            if trig - pad[0] > 0 and trig + pad[1] < end_frame - start_frame:
                traces[:, trig - pad[0]:trig + pad[1]] = 0
            elif trig - pad[0] > 0:
                traces[:, :trig + pad[1]] = 0
            elif trig + pad[1] < end_frame - start_frame:
                traces[:, trig - pad[0]:] = 0
        return traces


def remove_artifacts(recording, triggers, ms_before=0.5, ms_after=3):
    return RemoveArtifactsRecording(
        recording=recording, triggers=triggers, ms_before=ms_before, ms_after=ms_after
    )
