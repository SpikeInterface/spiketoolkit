from spikeextractors import RecordingExtractor
import numpy as np

class CommonReferenceRecording(RecordingExtractor):
    def __init__(self, recording, reference='median', groups=None, ref_channel=None):
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
        if reference != 'median' and reference != 'average' and reference != 'single':
            raise ValueError("'reference' must be either 'median' or 'average'")
        self._recording = recording
        self._ref = reference
        self._groups = groups
        self._ref_channel = ref_channel
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
                print('Common median in groups: ', new_groups)
                return np.vstack(np.array([self._recording.get_traces(channel_ids=split_group,
                                                                     start_frame=start_frame, end_frame=end_frame)
                                           - np.median(self._recording.get_traces(channel_ids=split_group,
                                                                                 start_frame=start_frame,
                                                                                 end_frame=end_frame),
                                                       axis=0, keepdims=True) for split_group in new_groups]))
        elif self._ref == 'average':
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
                print('Common average in groups: ', new_groups)
                return np.vstack(np.array([self._recording.get_traces(channel_ids=split_group,
                                                                     start_frame=start_frame, end_frame=end_frame)
                                           - self._recording.get_traces(channel_ids=ref, start_frame=start_frame,
                                                                       end_frame=end_frame)
                                           for (split_group, ref) in zip(new_groups, self._ref_channel)]))


def common_reference(recording, reference='median', groups=None, ref_channel=None):
    return CommonReferenceRecording(
        recording=recording, reference=reference, groups=groups, ref_channel=ref_channel
    )
