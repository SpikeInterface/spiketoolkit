from spikeextractors import RecordingExtractor
import numpy as np

class RectifyRecording(RecordingExtractor):
    def __init__(self, *, recording):
        RecordingExtractor.__init__(self)
        self._recording = recording
        self.copyChannelProperties(recording)

    def getSamplingFrequency(self):
        return self._recording.getSamplingFrequency()

    def getNumFrames(self):
        return self._recording.getNumFrames()

    def getTraces(self, channel_ids=None, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.getNumFrames()
        if channel_ids is None:
            channel_ids = self.getChannelIds()
        return np.abs(self._recording.getTraces(channel_ids=channel_ids, start_frame=start_frame, end_frame=end_frame))

    def getChannelIds(self):
        return self._recording.getChannelIds()


def rectify(recording):
    return RectifyRecording(
        recording=recording
    )
