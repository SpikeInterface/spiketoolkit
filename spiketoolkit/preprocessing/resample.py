from spikeextractors import RecordingExtractor
import numpy as np

try:
    from scipy import special, signal
    HAVE_RR = True
except ImportError:
    HAVE_RR = False

class ResampledRecording(RecordingExtractor):

    preprocessor_name = 'ResampledRecording'
    installed = HAVE_RR  # check at class level if installed or not
    _gui_params = [
        {'name': 'resample_rate', 'type': 'float', 'title': "The resampling frequency"},
    ]
    installation_mesg = "To use the ResampledRecording, install scipy: \n\n pip install scipy\n\n"  # err


    def __init__(self, recording, resample_rate):
        assert HAVE_RR, "To use the BandpassFilterRecording, install scipy: \n\n pip install scipy\n\n"
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
            traces_resampled = signal.decimate(traces,
                                               q=int(self._recording.get_sampling_frequency() / self._resample_rate),
                                               axis=1)
        else:
            traces_resampled = signal.resample(traces, int(end_frame_sampled - start_frame_sampled), axis=1)
        return traces_resampled

    def get_channel_ids(self):
        return self._recording.get_channel_ids()


def resample(recording, resample_rate):
    '''
    Resamples the recording extractor traces. If the resampling rate is multiple of the sampling rate, the faster
    scipy decimate function is used.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to be resampled
    resample_rate: int or float
        The resampling frequency

    Returns
    -------
    resampled_recording: ResampledRecording
        The resampled recording extractor

    '''
    return ResampledRecording(
        recording=recording,
        resample_rate=resample_rate
    )
