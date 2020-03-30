from spikeextractors import RecordingExtractor
from .transform import TransformRecording
import numpy as np


class CenterRecording(TransformRecording):

    preprocessor_name = 'Center'
    installed = True  # check at class level if installed or not
    preprocessor_gui_params = []
    installation_mesg = ""  # err

    def __init__(self, recording, mode, seconds, n_snippets):
        if not isinstance(recording, RecordingExtractor):
            raise ValueError("'recording' must be a RecordingExtractor")
        self._recording = recording
        self._scalar = 1
        self._mode = mode
        assert self._mode in ['mean', 'median'], "'mode' can be 'mean' or 'median'"

        # use 10 snippets of equal duration equally distributed on the recording
        n_snippets = int(n_snippets)
        assert n_snippets > 0, "'n_snippets' must be positive"
        snip_len = seconds / n_snippets * recording.get_sampling_frequency()

        if seconds * recording.get_sampling_frequency() >= recording.get_num_frames():
            traces = self._recording.get_traces()
        else:
            # skip initial and final part
            # TODO test if this is possible
            snip_start = np.linspace(snip_len, recording.get_num_frames()-2*snip_len, n_snippets)
            for i, snip in enumerate(snip_start):
                if i == 0:
                    traces = self._recording.get_traces(start_frame=snip, end_frame=snip+snip_len)
                else:
                    traces = np.vstack((traces, self._recording.get_traces(start_frame=snip, end_frame=snip+snip_len)))

        if self._mode == 'mean':
            self._offset = -np.mean(traces, axis=1)
        else:
            self._offset = -np.median(traces, axis=1)
        TransformRecording.__init__(self, self._recording, scalar=self._scalar, offset=self._offset)
        self.copy_channel_properties(recording=self._recording)
        self._kwargs = {'recording': recording.make_serialized_dict(), 'mode': self._mode}


def center(recording, mode='median', seconds=10, n_snippets=10):
    '''
    Removes the offset of the traces channel by channel.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to be transformed
    mode: str
        'median' (default) or 'mean'
    seconds: float
        Number of seconds used to compute center
    n_snippets: int
        Number of snippets in which the total 'seconds' are divided spannign the recording duration

    Returns
    -------
    rcenter: CenterRecording
        The output recording extractor object
    '''
    return CenterRecording(recording=recording, mode=mode, seconds=seconds, n_snippets=n_snippets)
