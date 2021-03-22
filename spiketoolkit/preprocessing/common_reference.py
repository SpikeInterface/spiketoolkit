from spikeextractors import RecordingExtractor
import numpy as np
from .basepreprocessorrecording import BasePreprocessorRecordingExtractor
from spikeextractors.extraction_tools import check_get_traces_args

from ..utils import get_closest_channels


class CommonReferenceRecording(BasePreprocessorRecordingExtractor):
    preprocessor_name = 'CommonReference'

    def __init__(self, recording, reference='median', groups=None, ref_channels=None,
                 local_radius=(2, 8), dtype=None, verbose=False):

        if not isinstance(recording, RecordingExtractor):
            raise ValueError("'recording' must be a RecordingExtractor")
        if reference not in ['median', 'average', 'single', 'local']:
            raise ValueError("'reference' must be either 'median', 'average', 'single' or 'local'")
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
        elif self._ref == 'local':
            assert groups is None, "With 'local' CAR, the group option should not be used."

        self._ref_channel = ref_channels
        self._local_radius = local_radius
        if dtype is None:
            self._dtype = recording.get_dtype()
        else:
            self._dtype = dtype
        self.verbose = verbose
        BasePreprocessorRecordingExtractor.__init__(self, recording)
        self._kwargs = {'recording': recording.make_serialized_dict(), 'reference': reference, 'groups': groups,
                        'ref_channels': ref_channels, 'local_radius': local_radius,
                        'dtype': dtype, 'verbose': verbose}

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None, return_scaled=True):

        selected_groups, selected_channels = self._create_channel_groups(channel_ids)
        traces = None

        if self._ref == 'median':
            if self.verbose:
                if self._groups is None:
                    print('Common median reference using all channels')
                else:
                    print('Common median in groups: ', selected_groups)

            traces = np.vstack(np.array([self._recording.get_traces(channel_ids=split_channel,
                                                                    start_frame=start_frame, end_frame=end_frame,
                                                                    return_scaled=return_scaled)
                                         - np.median(self._recording.get_traces(channel_ids=split_group,
                                                                                start_frame=start_frame,
                                                                                end_frame=end_frame,
                                                                                return_scaled=return_scaled),
                                                     axis=0, keepdims=True) for (split_channel, split_group) in
                                         zip(selected_channels, selected_groups)]))
        elif self._ref == 'average':
            if self.verbose:
                if self._groups is None:
                    print('Common average reference using all channels')
                else:
                    print('Common average in groups: ', selected_groups)

            traces = np.vstack(np.array([self._recording.get_traces(channel_ids=split_channel,
                                                                    start_frame=start_frame,
                                                                    end_frame=end_frame,
                                                                    return_scaled=return_scaled)
                                         - np.mean(self._recording.get_traces(channel_ids=split_group,
                                                                              start_frame=start_frame,
                                                                              end_frame=end_frame,
                                                                              return_scaled=return_scaled),
                                                   axis=0, keepdims=True) for (split_channel, split_group) in
                                         zip(selected_channels, selected_groups)]))
        elif self._ref == 'single':
            if self.verbose:
                if self._groups is None:
                    print('Reference to channel', self._ref_channel)
                else:
                    print('Reference', selected_groups, 'to channels', self._ref_channel)

            traces = np.vstack(np.array([self._recording.get_traces(channel_ids=split_channel,
                                                                    start_frame=start_frame, end_frame=end_frame,
                                                                    return_scaled=return_scaled)
                                         - self._recording.get_traces(channel_ids=[ref], start_frame=start_frame,
                                                                      end_frame=end_frame,
                                                                      return_scaled=return_scaled)
                                         for (split_channel, ref) in zip(selected_channels, self._ref_channel)]))

        elif self._ref == 'local':
            if self.verbose:
                print(
                    'Local Common average using as reference channels in a ring-shape region with radius: ' + self._local_radius)

            neighrest_id, distances = get_closest_channels(self._recording, channel_ids)
            traces = self._recording.get_traces(channel_ids=channel_ids, start_frame=start_frame,
                                                end_frame=end_frame,
                                                return_scaled=return_scaled) \
                     - np.vstack(np.array([np.average(
                self._recording.get_traces(
                    channel_ids=neighrest_id[id, np.logical_and(self._local_radius[0] < distances[id],
                                                                distances[id] <= self._local_radius[1])],
                    start_frame=start_frame, end_frame=end_frame, return_scaled=return_scaled), axis=0)
                for id in range(len(channel_ids))]))

        return np.array(traces).astype(self._dtype)

    def _create_channel_groups(self, channel_ids):
        selected_groups = []
        selected_channels = []
        if self._groups:
            for g in self._groups:
                new_chans = []
                for chan in g:
                    if chan in self._recording.get_channel_ids():
                        new_chans.append(chan)
                selected_groups.append(new_chans)
                selected_channels.append([ch for ch in channel_ids if ch in new_chans])
        else:
            selected_groups = [self._recording.get_channel_ids()]
            selected_channels = [channel_ids]
        return selected_groups, selected_channels


def common_reference(recording, reference='median', groups=None, ref_channels=None, local_radius=(2, 8), dtype=None,
                     verbose=False):
    '''
    Re-references the recording extractor traces.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to be re-referenced
    reference: str
        'median', 'average', 'single' or 'local'
        If 'median', common median reference (CMR) is implemented (the median of
        the selected channels is removed for each timestamp).
        If 'average', common average reference (CAR) is implemented (the mean of the selected channels is removed
        for each timestamp).
        If 'single', the selected channel(s) is remove from all channels.
        If 'local', an average CAR is implemented with only k channels selected the nearest outside of a radius around each channel
    groups: list
        List of lists containing the channels for splitting the reference. The CMR, CAR, or referencing with respect to
        single channels are applied group-wise. However, this is not applied for the local CAR.
        It is useful when dealing with different channel groups, e.g. multiple tetrodes.
    ref_channels: list or int
        If no 'groups' are specified, all channels are referenced to 'ref_channels'. If 'groups' is provided, then a
        list of channels to be applied to each group is expected. If 'single' reference, a list of one channel  or an
        int is expected.
    local_radius: tuple(int, int)
        Use in the local CAR implementation as the selecting annulus (exclude radius, include radius)
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
        recording=recording, reference=reference, groups=groups, ref_channels=ref_channels, local_radius=local_radius,
        dtype=dtype, verbose=verbose
    )
