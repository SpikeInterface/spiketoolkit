from spikeextractors import RecordingExtractor
from .transform import TransformRecording
import numpy as np


class CenterRecording(TransformRecording):

    preprocessor_name = 'Center'
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # err

    def __init__(self, recording, mode, seconds, n_snippets):
        if not isinstance(recording, RecordingExtractor):
            raise ValueError("'recording' must be a RecordingExtractor")
        self._recording = recording
        self._scalar = 1
        self._mode = mode
        self._seconds = seconds
        self._n_snippets = n_snippets
        assert self._mode in ['mean', 'median'], "'mode' can be 'mean' or 'median'"

        # use n_snippets of equal duration equally distributed on the recording
        n_snippets = int(n_snippets)
        assert n_snippets > 0, "'n_snippets' must be positive"
        snip_len = seconds / n_snippets * recording.get_sampling_frequency()

        if seconds * recording.get_sampling_frequency() >= recording.get_num_frames():
            traces = self._recording.get_traces()
        else:
            # skip initial and final part
            snip_start = np.linspace(snip_len // 2, recording.get_num_frames()-int(1.5*snip_len), n_snippets)
            traces_snippets = self._recording.get_snippets(reference_frames=snip_start, snippet_len=snip_len)
            traces_snippets = traces_snippets.swapaxes(0, 1)
            traces = traces_snippets.reshape((traces_snippets.shape[0],
                                              traces_snippets.shape[1] * traces_snippets.shape[2]))
        if self._mode == 'mean':
            self._offset = -np.mean(traces, axis=1)
        else:
            self._offset = -np.median(traces, axis=1)
        dtype = str(self._recording.get_dtype())
        if 'uint' in dtype:
            if 'numpy' in dtype:
                dtype = str(dtype).replace("<class '", "").replace("'>", "")
                # drop 'numpy'
                dtype = dtype.split('.')[1]
            dtype = dtype[1:]
        TransformRecording.__init__(self, self._recording, scalar=self._scalar, offset=self._offset, dtype=dtype)
        self.copy_channel_properties(recording=self._recording)
        self._kwargs = {'recording': recording.make_serialized_dict(), 'mode': mode, 'seconds': seconds,
                        'n_snippets': n_snippets}


def center(recording, mode='median', seconds=10., n_snippets=10):
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
        Number of snippets in which the total 'seconds' are divided spanning the recording duration

    Returns
    -------
    center: CenterRecording
        The output recording extractor object
    '''
    return CenterRecording(recording=recording, mode=mode, seconds=seconds, n_snippets=n_snippets)
