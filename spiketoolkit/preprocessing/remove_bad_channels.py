from spikeextractors import RecordingExtractor, SubRecordingExtractor
from spikeextractors.extraction_tools import check_get_traces_args
import numpy as np

class RemoveBadChannelsRecording(RecordingExtractor):

    preprocessor_name = 'RemoveBadChannels'
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # err

    def __init__(self, recording, bad_channel_ids, bad_threshold, seconds, verbose):
        if not isinstance(recording, RecordingExtractor):
            raise ValueError("'recording' must be a RecordingExtractor")
        self._recording = recording
        self._bad_channel_ids = bad_channel_ids
        self._bad_threshold = bad_threshold
        self._seconds = seconds
        self.verbose = verbose
        self._initialize_subrecording_extractor()
        RecordingExtractor.__init__(self)
        self.copy_channel_properties(recording=self._subrecording)

        self._kwargs = {'recording': recording.make_serialized_dict(), 'bad_channel_ids': bad_channel_ids,
                        'bad_threshold': bad_threshold, 'seconds': seconds, 'verbose': verbose}

    def get_sampling_frequency(self):
        return self._subrecording.get_sampling_frequency()

    def get_num_frames(self):
        return self._subrecording.get_num_frames()

    def get_channel_ids(self):
        return self._subrecording.get_channel_ids()

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        traces = self._subrecording.get_traces(channel_ids=channel_ids, start_frame=start_frame, end_frame=end_frame)
        return traces

    def _initialize_subrecording_extractor(self):
        if isinstance(self._bad_channel_ids, (list, np.ndarray)):
            active_channels = []
            for chan in self._recording.get_channel_ids():
                if chan not in self._bad_channel_ids:
                    active_channels.append(chan)
            self._subrecording = SubRecordingExtractor(self._recording, channel_ids=active_channels)
        elif self._bad_channel_ids is None:
            start_frame = self._recording.get_num_frames() // 2
            end_frame = int(start_frame + self._seconds * self._recording.get_sampling_frequency())
            if end_frame > self._recording.get_num_frames():
                end_frame = self._recording.get_num_frames()
            traces = self._recording.get_traces(start_frame=start_frame, end_frame=end_frame)
            stds = np.std(traces, axis=1)
            bad_channel_ids = [ch for ch, std in enumerate(stds) if std > self._bad_threshold * np.median(stds)]
            if self.verbose:
                print('Automatically removing channels:', bad_channel_ids)
            active_channels = []
            for chan in self._recording.get_channel_ids():
                if chan not in bad_channel_ids:
                    active_channels.append(chan)
            self._subrecording = SubRecordingExtractor(self._recording, channel_ids=active_channels)
        else:
            self._subrecording = self._recording
        self.active_channels = self._subrecording.get_channel_ids()


def remove_bad_channels(recording, bad_channel_ids=None, bad_threshold=2, seconds=10, verbose=False):
    '''
    Remove bad channels from the recording extractor.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object
    bad_channel_ids: list
        List of bad channel ids (int). If None, automatic removal will be done based on standard deviation.
    bad_threshold: float
        If automatic is used, the threshold for the standard deviation over which channels are removed
    seconds: float
        If automatic is used, the number of seconds used to compute standard deviations
    verbose: bool
        If True, output is verbose

    Returns
    -------
    remove_bad_channels_recording: RemoveBadChannelsRecording
        The recording extractor without bad channels

    '''
    return RemoveBadChannelsRecording(recording=recording, bad_channel_ids=bad_channel_ids,
                                      bad_threshold=bad_threshold, seconds=seconds, verbose=verbose)
