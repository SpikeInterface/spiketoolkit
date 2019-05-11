from spikeextractors import RecordingExtractor, SubRecordingExtractor
import numpy as np

class RemoveBadChannelsRecording(RecordingExtractor):

    preprocessor_name = 'RemoveBadChannelsRecording'
    installed = True  # check at class level if installed or not
    _gui_params = [
        {'name': 'bad_channels', 'type': 'list', 'title': "List of bad channels or 'auto'"},
        {'name': 'bad_threshold', 'type': 'float', 'title': "Threshold in number of sd to remove channels ('auto')"},
        {'name': 'seconds', 'type': 'float', 'title': "Number of seconds to compute standard deviation ('auto)"},
    ]
    installation_mesg = ""  # err

    def __init__(self, recording, bad_channels, bad_threshold, seconds):
        RecordingExtractor.__init__(self)
        if not isinstance(recording, RecordingExtractor):
            raise ValueError("'recording' must be a RecordingExtractor")
        self._recording = recording
        self._bad_channels = bad_channels
        self._bad_threshold = bad_threshold
        self._seconds = seconds
        self._initialize_subrecording_extractor()
        self.copy_channel_properties(recording=self._subrecording)


    def get_sampling_frequency(self):
        return self._subrecording.get_sampling_frequency()

    def get_num_frames(self):
        return self._subrecording.get_num_frames()

    def get_channel_ids(self):
        return self._subrecording.get_channel_ids()

    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_frames()
        if channel_ids is None:
            channel_ids = self.get_channel_ids()
        traces = self._subrecording.get_traces(channel_ids=channel_ids, start_frame=start_frame, end_frame=end_frame)
        return traces

    def _initialize_subrecording_extractor(self):
        if isinstance(self._bad_channels, (list, np.ndarray)):
            active_channels = []
            for chan in self._recording.get_channel_ids():
                if chan not in self._bad_channels:
                    active_channels.append(chan)
            self._subrecording = SubRecordingExtractor(self._recording, channel_ids=active_channels)
        elif self._bad_channels == 'auto':
            start_frame = self._recording.get_num_frames() // 2
            end_frame = int(start_frame + self._seconds * self._recording.get_sampling_frequency())
            traces = self._recording.get_traces(start_frame=start_frame, end_frame=end_frame)
            stds = np.std(traces, axis=1)
            bad_channels = [ch for ch, std in enumerate(stds) if std > self._bad_threshold * np.median(stds)]
            active_channels = []
            for chan in self._recording.get_channel_ids():
                if chan not in bad_channels:
                    active_channels.append(chan)
            self._subrecording = SubRecordingExtractor(self._recording, channel_ids=active_channels)
        else:
            self._subrecording = self._recording


def remove_bad_channels(recording, bad_channels, bad_threshold=2, seconds=10):
    '''
    Remove bad channels from the recording extractor.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object
    bad_channels: list or 'auto'
        List of bad channels (int) or 'auto' for automatic removal based on standard deviation
    bad_threshold: float
        If 'auto' is used, the threshold for the standard deviation over which channels are removed
    seconds: float
        If 'auto' is used, the number of seconds used to compute standard deviations

    Returns
    -------
    remove_bad_channels_recording: RemoveBadChannelsRecording
        The recording extractor without bad channels

    '''
    return RemoveBadChannelsRecording(recording=recording, bad_channels=bad_channels,
                                      bad_threshold=bad_threshold, seconds=seconds)
