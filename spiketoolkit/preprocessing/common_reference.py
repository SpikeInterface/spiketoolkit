from spikeextractors import RecordingExtractor
import numpy as np
from copy import deepcopy
from spikeextractors.extraction_tools import check_get_traces_args


class CommonReferenceRecording(RecordingExtractor):
    preprocessor_name = 'CommonReference'
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # err

    def __init__(self, recording, reference='median', groups=None, ref_channels=None, dtype=None, verbose=False):
        if not isinstance(recording, RecordingExtractor):
            raise ValueError("'recording' must be a RecordingExtractor")
        if reference != 'median' and reference != 'average' and reference != 'single':
            raise ValueError("'reference' must be either 'median' or 'average'")
        self._recording = recording
        self._ref = reference
        self._groups = groups
        if self._ref == 'single':
            assert ref_channels is not None, "With 'single' reference, provide 'ref_channels'"
            if self._groups is not None:
                assert len(ref_channels) == len(self._groups), "'ref_channel' and 'groups' must have the " \
                                                               "same length"
            else:
                if isinstance(ref_channels, (list, np.ndarray)):
                    assert len(ref_channels) == 1, "'ref_channel' with no 'groups' can be int or a list of one element"
                else:
                    assert isinstance(ref_channels, (int, np.integer)), "'ref_channels' must be int"
                    ref_channels = [ref_channels]
        self._ref_channel = ref_channels
        if dtype is None:
            self._dtype = recording.get_dtype()
        else:
            self._dtype = dtype
        self.verbose = verbose
        RecordingExtractor.__init__(self)
        self.copy_channel_properties(recording=self._recording)

        # update dump dict
        self._kwargs = {'recording': recording.make_serialized_dict(), 'reference': reference, 'groups': groups,
                        'ref_channels': ref_channels, 'dtype': dtype, 'verbose': verbose}

    def get_sampling_frequency(self):
        return self._recording.get_sampling_frequency()

    def get_num_frames(self):
        return self._recording.get_num_frames()

    def get_channel_ids(self):
        return self._recording.get_channel_ids()

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None):
        channel_idxs = np.array([self.get_channel_ids().index(ch) for ch in channel_ids])
        if self._ref == 'median':
            if self._groups is None:
                if self.verbose:
                    print('Common median reference using all channels')
                traces = self._recording.get_traces(start_frame=start_frame, end_frame=end_frame)
                traces = traces - np.median(traces, axis=0, keepdims=True)
                return traces[channel_idxs].astype(self._dtype)
            else:
                new_groups = []
                for g in self._groups:
                    new_chans = []
                    for chan in g:
                        if chan in self._recording.get_channel_ids():
                            new_chans.append(chan)
                    new_groups.append(new_chans)
                if self.verbose:
                    print('Common median in groups: ', new_groups)
                traces = np.vstack(np.array([self._recording.get_traces(channel_ids=split_group,
                                                                        start_frame=start_frame, end_frame=end_frame)
                                             - np.median(self._recording.get_traces(channel_ids=split_group,
                                                                                    start_frame=start_frame,
                                                                                    end_frame=end_frame),
                                                         axis=0, keepdims=True) for split_group in new_groups]))
                return traces[channel_idxs].astype(self._dtype)
        elif self._ref == 'average':
            if self.verbose:
                print('Common average reference using all channels')
            if self._groups is None:
                traces = self._recording.get_traces(start_frame=start_frame, end_frame=end_frame)
                traces = traces - np.mean(traces, axis=0, keepdims=True)
                return traces[channel_idxs].astype(self._dtype)
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
                traces = np.vstack(np.array([self._recording.get_traces(channel_ids=split_group,
                                                                        start_frame=start_frame, end_frame=end_frame)
                                             - np.mean(self._recording.get_traces(channel_ids=split_group,
                                                                                  start_frame=start_frame,
                                                                                  end_frame=end_frame),
                                                       axis=0, keepdims=True) for split_group in new_groups]))
                return traces[channel_idxs].astype(self._dtype)
        elif self._ref == 'single':
            if self._groups is None:
                if self.verbose:
                    print('Reference to channel', self._ref_channel)
                traces = self._recording.get_traces(channel_ids=channel_ids, start_frame=start_frame,
                                                    end_frame=end_frame) \
                         - self._recording.get_traces(channel_ids=self._ref_channel, start_frame=start_frame,
                                                      end_frame=end_frame)
                return traces.astype(self._dtype)
            else:
                new_groups = []
                for g in self._groups:
                    new_chans = []
                    for chan in g:
                        if chan in self._recording.get_channel_ids():
                            new_chans.append(chan)
                    new_groups.append(new_chans)
                if self.verbose:
                    print('Reference', new_groups, 'to channels', self._ref_channel)
                traces = np.vstack(np.array([self._recording.get_traces(channel_ids=split_group,
                                                                        start_frame=start_frame, end_frame=end_frame)
                                             - self._recording.get_traces(channel_ids=[ref], start_frame=start_frame,
                                                                          end_frame=end_frame)
                                             for (split_group, ref) in zip(new_groups, self._ref_channel)]))
                return traces[channel_idxs].astype(self._dtype)


def common_reference(recording, reference='median', groups=None, ref_channels=None, dtype=None, verbose=False):
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
    ref_channels: list or int
        If no 'groups' are specified, all channels are referenced to 'ref_channels'. If 'groups' is provided, then a
        list of channels to be applied to each group is expected. If 'single' reference, a list of one channel  or an
        int is expected.
    dtype: str
        dtype of the returned traces. If None, dtype is maintained
    verbose: bool
        If True, output is verbose

    Returns
    -------
    referenced_recording: CommonReferenceRecording
        The re-referenced recording extractor object
    '''
    return CommonReferenceRecording(
        recording=recording, reference=reference, groups=groups, ref_channels=ref_channels, dtype=dtype, verbose=verbose
    )
