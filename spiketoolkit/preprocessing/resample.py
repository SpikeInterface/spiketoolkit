from spikeextractors import RecordingExtractor
import numpy as np
from scipy import special, signal

class ResampledRecording(RecordingExtractor):
    def __init__(self, *, recording, resample_rate):
        RecordingExtractor.__init__(self)
        self._recording = recording
        self._resample_rate = resample_rate
        self.copyChannelProperties(recording)

    def getSamplingFrequency(self):
        return self._resample_rate

    def getNumFrames(self):
        return int(self._recording.getNumFrames() / self._recording.getSamplingFrequency() * self._resample_rate)

    def getTraces(self, channel_ids=None, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame_not_sampled = 0
            start_frame_sampled = 0
        else:
            start_frame_not_sampled = int(start_frame / self.getSamplingFrequency() *
                                          self._recording.getSamplingFrequency())
            start_frame_sampled = start_frame
        if end_frame is None:
            end_frame_not_sampled = self._recording.getNumFrames()
            end_frame_sampled = self.getNumFrames()
        else:
            end_frame_not_sampled = int(end_frame / self.getSamplingFrequency() * self._recording.getSamplingFrequency())
            end_frame_sampled = end_frame
        if channel_ids is None:
            channel_ids = self.getChannelIds()
        traces = self._recording.getTraces(start_frame=start_frame_not_sampled,
                                           end_frame=end_frame_not_sampled,
                                           channel_ids=channel_ids)
        if np.mod(self._recording.getSamplingFrequency(), self._resample_rate) == 0:
            print('Decimate')
            traces_resampled = signal.decimate(traces,
                                               q=int(self._recording.getSamplingFrequency() / self._resample_rate),
                                               axis=1)
        else:
            traces_resampled = signal.resample(traces, int(end_frame_sampled - start_frame_sampled), axis=1)
        return traces_resampled

    def getChannelIds(self):
        return self._recording.getChannelIds()


def resample(recording, resample_rate):
    return ResampledRecording(
        recording=recording,
        resample_rate=resample_rate
    )
