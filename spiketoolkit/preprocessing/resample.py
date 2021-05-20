from spikeextractors import RecordingExtractor
from spikeextractors.extraction_tools import check_get_traces_args
from .basepreprocessorrecording import BasePreprocessorRecordingExtractor
import numpy as np
from warnings import warn

try:
    from scipy import special, signal

    HAVE_RR = True
except ImportError:
    HAVE_RR = False


class ResampleRecording(BasePreprocessorRecordingExtractor):
    preprocessor_name = 'Resample'
    installed = HAVE_RR  # check at class level if installed or not
    installation_mesg = "To use the ResampleRecording, install scipy: \n\n pip install scipy\n\n"  # err

    def __init__(self, recording, resample_rate):
        assert HAVE_RR, "To use the ResampleRecording, install scipy: \n\n pip install scipy\n\n"
        self._resample_rate = resample_rate
        BasePreprocessorRecordingExtractor.__init__(self, recording, copy_times=False)
        self._dtype = recording.get_dtype()

        if recording._times is not None:
            # resample timestamps uniformly
            warn("Timestamps will be resampled uniformly. Non-uniform timestamps will be lost due to resampling.")
            resampled_times = np.linspace(recording._times[0], recording._times[-1], self.get_num_frames())
            self.set_times(resampled_times)

        self._kwargs = {'recording': recording.make_serialized_dict(), 'resample_rate': resample_rate}

    def get_sampling_frequency(self):
        return self._resample_rate

    def get_num_frames(self):
        return int(self._recording.get_num_frames() / self._recording.get_sampling_frequency() * self._resample_rate)

    # avoid filtering one sample
    def get_dtype(self, return_scaled=True):
        return self._dtype

    # need to override frame_to_time and time_to_frame because self._recording might not have "times"
    def frame_to_time(self, frames):
        if self._times is not None:
            return np.round(frames / self.get_sampling_frequency(), 6)
        else:
            return self._recording.time_to_frame(frames)

    def time_to_frame(self, times):
        if self._times is not None:
            return np.round(times * self.get_sampling_frequency()).astype('int64')
        else:
            return self._recording.time_to_frame(times)


    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None, return_scaled=True):
        start_frame_not_sampled = int(start_frame / self.get_sampling_frequency() *
                                      self._recording.get_sampling_frequency())
        start_frame_sampled = start_frame
        end_frame_not_sampled = int(end_frame / self.get_sampling_frequency() *
                                    self._recording.get_sampling_frequency())
        end_frame_sampled = end_frame
        traces = self._recording.get_traces(start_frame=start_frame_not_sampled,
                                            end_frame=end_frame_not_sampled,
                                            channel_ids=channel_ids,
                                            return_scaled=return_scaled)
        traces_resampled = signal.resample(traces, int(end_frame_sampled - start_frame_sampled), axis=1)

        return traces_resampled.astype(self._dtype)


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
    resampled_recording: ResampleRecording
        The resample recording extractor

    '''
    return ResampleRecording(
        recording=recording,
        resample_rate=resample_rate
    )
