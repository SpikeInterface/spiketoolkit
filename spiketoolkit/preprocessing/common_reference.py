from spikeextractors import RecordingExtractor
import numpy as np

class CommonReferenceRecording(RecordingExtractor):
    def __init__(self, recording, reference='median', groups=None):
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
        if reference != 'median' and reference != 'average':
            raise ValueError("'reference' must be either 'median' or 'average'")
        self._recording = recording
        self._ref = reference
        self._groups = groups
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
        if self._ref == 'median':
            if self._groups is None:
                return self._recording.getTraces(channel_ids=channel_ids, start_frame=start_frame, end_frame=end_frame) \
                       - np.median(self._recording.getTraces(channel_ids=channel_ids, start_frame=start_frame,
                                                             end_frame=end_frame), axis=0, keepdims=True)
            else:
                return np.vstack(np.array([self._recording.getTraces(channel_ids=split_group,
                                                                     start_frame=start_frame, end_frame=end_frame)
                                           - np.median(self._recording.getTraces(channel_ids=split_group,
                                                                                 start_frame=start_frame,
                                                                                 end_frame=end_frame),
                                                       axis=0, keepdims=True) for split_group in self._groups]))
        elif self._ref == 'average':
            if self._groups is None:
                return self._recording.getTraces(channel_ids=channel_ids, start_frame=start_frame, end_frame=end_frame) \
                       - np.mean(self._recording.getTraces(channel_ids=channel_ids, start_frame=start_frame,
                                                           end_frame=end_frame), axis=0, keepdims=True)
            else:
                return np.vstack(np.array([self._recording.getTraces(channel_ids=split_group,
                                                                     start_frame=start_frame, end_frame=end_frame)
                                           - np.mean(self._recording.getTraces(channel_ids=split_group,
                                                                               start_frame=start_frame,
                                                                               end_frame=end_frame),
                                                     axis=0, keepdims=True) for split_group in self._groups]))


def common_reference(recording, reference='median', groups=None):
    return CommonReferenceRecording(
        recording=recording, reference=reference, groups=groups
    )