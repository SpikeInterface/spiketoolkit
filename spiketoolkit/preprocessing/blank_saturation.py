from spikeextractors import RecordingExtractor
import numpy as np


class BlankSaturationRecording(RecordingExtractor):
    preprocessor_name = 'BlankSaturation'
    installed = True  # check at class level if installed or not
    preprocessor_gui_params = [
        {'name': 'threshold', 'type': 'float',
         'title': "Scale for the output distribution"},
        {'name': 'seed', 'type': 'int', 'value': 0, 'default': 0, 
         'title': "Random seed for reproducibility."},
    ]
    installation_mesg = ""  # err

    def __init__(self, recording, threshold=None, seed=0):
        if not isinstance(recording, RecordingExtractor):
            raise ValueError("'recording' must be a RecordingExtractor")
        self._recording = recording
        random_data = self._get_random_data_for_scaling(seed=seed).ravel()
        q = np.quantile(random_data, [0.001, 0.5, 1 - 0.001])
        if 2 * q[1] - q[0] - q[2] < 2 * np.min([q[1] - q[0], q[2] - q[1]]):
            print('Warning, narrow signal range suggests artefact-free data.')
        self._median = q[1]
        if threshold is None:
            if np.abs(q[1] - q[0]) > np.abs(q[1] - q[2]):
                self._threshold = q[0]
                self._lower = True
            else:
                self._threshold = q[2]
                self._lower = False
        else:
            self._threshold = threshold
            if q[1] - threshold < 0:
                self._lower = False
            else:
                self._lower = True
        RecordingExtractor.__init__(self)
        self.copy_channel_properties(recording=self._recording)

    def _get_random_data_for_scaling(self, num_chunks=50, chunk_size=500, seed=0):
        N = self._recording.get_num_frames()
        random_ints = np.random.RandomState(seed=seed).randint(0, N - chunk_size, size=num_chunks)
        chunk_list = []
        for ff in random_ints:
            chunk = self._recording.get_traces(start_frame=ff,
                                               end_frame=ff + chunk_size)
            chunk_list.append(chunk)
        return np.concatenate(chunk_list, axis=1)

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
        traces = self._recording.get_traces(channel_ids=channel_ids,
                                            start_frame=start_frame,
                                            end_frame=end_frame)
        if self._lower:
            traces[traces <= self._threshold] = self._median
        else:
            traces[traces >= self._threshold] = self._median
        return traces


def blank_saturation(recording, threshold=None, seed=0):
    '''
    Find and remove parts of the signal with extereme values. Some arrays
    may produce these when amplifiers enter saturation, typically for
    short periods of time. To remove these artefacts, values below or above 
    a threshold are set to the median signal value.
    The threshold is either be estimated automatically, using the lower and upper 
    0.1 signal percentile with the largest deviation from the median, or specificed.
    Use this function with caution, as it may clip uncontaminated signals. A warning is
    printed if the data range suggests no artefacts.
    
    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to be transformed
        Minimum value. If `None`, clipping is not performed on lower
        interval edge.
    threshold: float or 'None' (default `None`)
        Threshold value (in absolute units) for saturation artifacts.
        If `None`, the threshold will be determined from the 0.1 signal percentile.
    seed: int
        Random seed for reproducibility
    Returns
    -------
    rescaled_traces: BlankSaturationRecording
        The filtered traces recording extractor object
    '''
    return BlankSaturationRecording(
        recording=recording, 
        threshold=threshold,
        seed=seed
    )
