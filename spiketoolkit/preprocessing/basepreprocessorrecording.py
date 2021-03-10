from spikeextractors import RecordingExtractor
from spikeextractors.extraction_tools import check_get_traces_args


class BasePreprocessorRecordingExtractor(RecordingExtractor):
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # err

    def __init__(self, recording):
        assert isinstance(recording, RecordingExtractor), "'recording' must be a RecordingExtractor"
        RecordingExtractor.__init__(self)
        self._recording = recording
        self.copy_channel_properties(recording)
        self.copy_epochs(recording)
        self.copy_times(recording)

        # avoid rescaling twice
        self.set_channel_gains(1)
        self.set_channel_offsets(0)

        self.is_filtered = recording.is_filtered
        if hasattr(recording, "has_unscaled"):
            self.has_unscaled = recording.has_unscaled
        else:
            self.has_unscaled = False

    def get_channel_ids(self):
        return self._recording.get_channel_ids()

    def get_num_frames(self):
        return self._recording.get_num_frames()

    def get_sampling_frequency(self):
        return self._recording.get_sampling_frequency()

    def time_to_frame(self, time):
        return self._recording.time_to_frame(time)

    def frame_to_time(self, frame):
        return self._recording.frame_to_time(frame)

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None, return_scaled=True):
        raise NotImplementedError

