from spikeextractors import RecordingExtractor
import numpy as np

class CommonReferenceRecording(RecordingExtractor):

    preprocessor_name = 'CommonReferenceRecording'
    installed = True  # check at class level if installed or not
    _gui_params = [
        {'name': 'reference', 'type': 'str', 'value':'median', 'default':'median', 'title': "Reference type ('median', 'average', or 'single')"},
        {'name': 'groups', 'type': 'int_list', 'value':None, 'default':None, 'title': "List of lists containins the channels for splitting the reference"},
        {'name': 'ref_channel', 'type': 'int/int_list', 'value':None, 'default':None, 'title': "All channels are referenced to 'ref_channel(s)"},
        {'name': 'verbose', 'type': 'bool', 'value':False, 'default':False, 'title': "If True, then the function will be verbose"}
    ]
    installation_mesg = ""  # err

    def __init__(self, recording, reference='median', groups=None, ref_channel=None, verbose=False):
        RecordingExtractor.__init__(self)
        if not isinstance(recording, RecordingExtractor):
            raise ValueError("'recording' must be a RecordingExtractor")
        if reference != 'median' and reference != 'average' and reference != 'single':
            raise ValueError("'reference' must be either 'median' or 'average'")
        self._recording = recording
        self._ref = reference
        self._groups = groups
        self._ref_channel = ref_channel
        self.verbose = verbose
        self.copy_channel_properties(recording=self._recording)

    def get_sampling_frequency(self):
        return self._recording.get_sampling_frequency()

    def get_num_frames(self):
        return self._recording.get_num_frames()

    def get_channel_ids(self):
        return self._recording.get_channel_ids()

    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_frames()
        if channel_ids is None:
            channel_ids = self.get_channel_ids()
        if self._ref == 'median':
            if self._groups is None:
                if self.verbose:
                    print('Common median reference using all channels')
                return self._recording.get_traces(channel_ids=channel_ids, start_frame=start_frame, end_frame=end_frame) \
                       - np.median(self._recording.get_traces(channel_ids=channel_ids, start_frame=start_frame,
                                                              end_frame=end_frame), axis=0, keepdims=True)
            else:
                new_groups = []
                for g in self._groups:
                    new_chans = []
                    for chan in g:
                        if chan in channel_ids:
                            new_chans.append(chan)
                    new_groups.append(new_chans)
                if self.verbose:
                    print('Common median in groups: ', new_groups)
                return np.vstack(np.array([self._recording.get_traces(channel_ids=split_group,
                                                                     start_frame=start_frame, end_frame=end_frame)
                                           - np.median(self._recording.get_traces(channel_ids=split_group,
                                                                                 start_frame=start_frame,
                                                                                 end_frame=end_frame),
                                                       axis=0, keepdims=True) for split_group in new_groups]))
        elif self._ref == 'average':
            if self.verbose:
                print('Common average reference using all channels')
            if self._groups is None:
                return self._recording.get_traces(channel_ids=channel_ids, start_frame=start_frame, end_frame=end_frame) \
                       - np.mean(self._recording.get_traces(channel_ids=channel_ids, start_frame=start_frame,
                                                           end_frame=end_frame), axis=0, keepdims=True)
            else:
                new_groups = []
                for g in self._groups:
                    new_chans = []
                    for chan in g:
                        if chan in self._recording.get_channel_ids():
                            new_chans.append(chan)
                    new_groups.append(new_chans)
                if self.verbose:
                    print('Common average in groups: ', new_groups)
                return np.vstack(np.array([self._recording.get_traces(channel_ids=split_group,
                                                                     start_frame=start_frame, end_frame=end_frame)
                                           - np.mean(self._recording.get_traces(channel_ids=split_group,
                                                                               start_frame=start_frame,
                                                                               end_frame=end_frame),
                                                     axis=0, keepdims=True) for split_group in new_groups]))

        elif self._ref == 'single':
            assert self._ref_channel is not None, "With 'single' reference, provide 'ref_channel'"
            if self._groups is None:
                assert isinstance(self._ref_channel, (int, np.integer)), "'ref_channel' must be int"
                if self.verbose:
                    print('Reference to channel', self._ref_channel)
                return self._recording.get_traces(channel_ids=channel_ids, start_frame=start_frame, end_frame=end_frame) \
                       - self._recording.get_traces(channel_ids=self._ref_channel, start_frame=start_frame,
                                                   end_frame=end_frame)
            else:
                assert len(self._ref_channel) == len(self._groups), "'ref_channel' and 'groups' must have the " \
                                                                    "same length"
                new_groups = []
                for g in self._groups:
                    new_chans = []
                    for chan in g:
                        if chan in self._recording.get_channel_ids():
                            new_chans.append(chan)
                    new_groups.append(new_chans)
                if self.verbose:
                    print('Reference', new_groups, 'to channels', self._ref_channel)
                return np.vstack(np.array([self._recording.get_traces(channel_ids=split_group,
                                                                      start_frame=start_frame, end_frame=end_frame)
                                           - self._recording.get_traces(channel_ids=ref, start_frame=start_frame,
                                                                        end_frame=end_frame)
                                           for (split_group, ref) in zip(new_groups, self._ref_channel)]))


def common_reference(recording, reference='median', groups=None, ref_channel=None, verbose=False):
    '''
    Re-references the recording extractor traces.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to be re-referenced
    reference: str
        'median', 'average', or 'single'.
        If 'median', common median reference (CMR) is implemented (the median of
        the selected channels is removed for each timestamp).
        If 'average', common average reference (CAR) is implemented (the mean of the selected channels is removed
        for each timestamp).
        If 'single', the selected channel(s) is remove from all channels.
    groups: list
        List of lists containins the channels for splitting the reference. The CMR, CAR, or referencing with respect to
        single channels are applied group-wise. It is useful when dealing with different channel groups, e.g. multiple
        tetrodes.
    ref_channel: int or list
        If no 'groups' are specified, all channels are referenced to 'ref_channel'. If 'groups' is provided, then a
        list of channels to be applied to each group is expected.
    verbose: bool
        If True, output is verbose

    Returns
    -------
    referenced_recording: CommonReferenceRecording
        The re-referenced recording extractor object
    '''
    return CommonReferenceRecording(
        recording=recording, reference=reference, groups=groups, ref_channel=ref_channel, verbose=verbose
    )
