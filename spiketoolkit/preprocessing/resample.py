from spikeextractors import RecordingExtractor
import numpy as np
from scipy import special, signal

class ResampledRecording(RecordingExtractor):
    def __init__(self, recording, resample_rate):
        RecordingExtractor.__init__(self)
        self._recording = recording
        self._resample_rate = resample_rate
        self.copy_channel_properties(recording)

    def get_sampling_frequency(self):
        return self._resample_rate

    def get_num_frames(self):
        return int(self._recording.get_num_frames() / self._recording.get_sampling_frequency() * self._resample_rate)

    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame_not_sampled = 0
            start_frame_sampled = 0
        else:
            start_frame_not_sampled = int(start_frame / self.get_sampling_frequency() *
                                          self._recording.get_sampling_frequency())
            start_frame_sampled = start_frame
        if end_frame is None:
            end_frame_not_sampled = self._recording.get_num_frames()
            end_frame_sampled = self.get_num_frames()
        else:
            end_frame_not_sampled = int(end_frame / self.get_sampling_frequency() * self._recording.get_sampling_frequency())
            end_frame_sampled = end_frame
        if channel_ids is None:
            channel_ids = self.get_channel_ids()
        traces = self._recording.get_traces(start_frame=start_frame_not_sampled,
                                           end_frame=end_frame_not_sampled,
                                           channel_ids=channel_ids)
        if np.mod(self._recording.get_sampling_frequency(), self._resample_rate) == 0:
            print('Decimate')
            traces_resampled = signal.decimate(traces,
                                               q=int(self._recording.get_sampling_frequency() / self._resample_rate),
                                               axis=1)
        else:
            traces_resampled = signal.resample(traces, int(end_frame_sampled - start_frame_sampled), axis=1)
        return traces_resampled

    def get_channel_ids(self):
        return self._recording.get_channel_ids()


def resample(recording, resample_rate):
    return ResampledRecording(
        recording=recording,
        resample_rate=resample_rate
    )
