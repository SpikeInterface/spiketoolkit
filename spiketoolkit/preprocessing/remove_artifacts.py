from spikeextractors import RecordingExtractor
from spikeextractors.extraction_tools import check_get_traces_args
from .basepreprocessorrecording import BasePreprocessorRecordingExtractor
import numpy as np
from scipy.interpolate import interp1d


class RemoveArtifactsRecording(BasePreprocessorRecordingExtractor):
    preprocessor_name = 'RemoveArtifacts'

    def __init__(self, recording, triggers, ms_before=0.5, ms_after=3.0, mode='zeros', fit_sample_spacing=1.):
        self._triggers = np.array(triggers)
        self._ms_before = ms_before
        self._ms_after = ms_after
        self._mode = mode
        self._fit_sample_spacing = fit_sample_spacing
        BasePreprocessorRecordingExtractor.__init__(self, recording)
        self._kwargs = {'recording': recording.make_serialized_dict(), 'triggers': triggers,
                        'ms_before': ms_before, 'ms_after': ms_after, 'mode': mode,
                        'fit_sample_spacing': fit_sample_spacing}

    @check_get_traces_args
    def get_traces(self, channel_ids=None, start_frame=None, end_frame=None, return_scaled=True):
        traces = self._recording.get_traces(channel_ids=channel_ids,
                                            start_frame=start_frame,
                                            end_frame=end_frame,
                                            return_scaled=return_scaled)
        triggers = self._triggers[(self._triggers > start_frame) & (self._triggers < end_frame)] - start_frame

        pad = [int(self._ms_before * self.get_sampling_frequency() / 1000),
               int(self._ms_after * self.get_sampling_frequency() / 1000)]

        traces = traces.copy()
        if self._mode == 'zeros':
            for trig in triggers:
                if trig - pad[0] > 0 and trig + pad[1] < end_frame - start_frame:
                    traces[:, trig - pad[0]:trig + pad[1]] = 0
                elif trig - pad[0] <= 0 and trig + pad[1] >= end_frame - start_frame:
                    traces = 0
                elif trig - pad[0] <= 0:
                    traces[:, :trig + pad[1]] = 0
                elif trig + pad[1] >= end_frame - start_frame:
                    traces[:, trig - pad[0]:] = 0
        else:
            sample_freq = self._recording.get_sampling_frequency()

            # generate indices for evenly spaced fit points before and after gap
            fit_sample_range = int(((sample_freq / 1000) * self._fit_sample_spacing * 2) + 1)
            fit_sample_interval = int(self._fit_sample_spacing * (sample_freq / 1000))

            fit_samples = np.array(range(0, fit_sample_range, fit_sample_interval))
            rev_fit_samples = fit_sample_range - fit_samples
            triggers = np.array(triggers).astype(int)
            for trig in triggers:
                pre_data_end_idx = trig - pad[0] - 1
                post_data_start_idx = trig + pad[1] + 1

                # Generate fit points from the sample points determined
                pre_idx = pre_data_end_idx - rev_fit_samples + 1
                post_idx = post_data_start_idx + fit_samples

                # Get indices of the gap to fill
                gap_idx = np.array(range(pre_data_end_idx + 1, post_data_start_idx + 0))

                # Make sure we are not going out of bounds
                gap_idx = gap_idx[gap_idx >= 0]
                gap_idx = gap_idx[gap_idx < len(traces[0])]

                # correct for out of bounds indices on both sides:
                if np.max(post_idx) >= len(traces[0]):
                    post_idx = post_idx[post_idx < len(traces[0])]

                if np.min(pre_idx) < 0:
                    pre_idx = pre_idx[pre_idx >= 0]

                # fit x values                
                all_idx = np.hstack((pre_idx, post_idx))

                # fit y values
                interp_traces = traces[:, all_idx]

                # Get the median value from 5 samples around each fit point
                # for robustness to noise / small fluctuations
                pre_vals = np.empty((0, len(traces)), 'int32')
                for idx in iter(pre_idx):
                    if idx == pre_idx[-1]:
                        idxs = np.array(range(idx - 3, idx + 1))
                    else:
                        idxs = np.array(range(idx - 2, idx + 3))

                    if np.min(idx) < 0:
                        idx = idx[idx >= 0]

                    median_vals = np.median(traces[:, idxs], axis=1)
                    pre_vals = np.vstack((pre_vals, median_vals))

                post_vals = np.empty((0, len(traces)), 'int32')
                for idx in iter(post_idx):
                    if idx == post_idx[0]:
                        idxs = np.array(range(idx, idx + 4))
                    else:
                        idxs = np.array(range(idx - 2, idx + 3))

                    if np.max(idx) >= len(traces[0]):
                        idx = idx[idx < len(traces[0])]

                    median_vals = np.median(traces[:, idxs], axis=1)
                    post_vals = np.vstack((post_vals, median_vals))

                interp_traces = np.vstack((pre_vals, post_vals)).T

                if self._mode == 'cubic' and len(all_idx) >= 5:
                    # Enough fit points present on either side to do cubic spline fit:
                    interp_function = interp1d(all_idx, interp_traces, self._mode,
                                               bounds_error=False,
                                               fill_value='extrapolate')
                    traces[:, gap_idx] = interp_function(gap_idx)
                elif self._mode == 'linear' and len(all_idx) >= 2:
                    # Enough fit points present for a linear fit
                    interp_function = interp1d(all_idx, interp_traces, self._mode, bounds_error=False,
                                               fill_value='extrapolate')
                    traces[:, gap_idx] = interp_function(gap_idx)
                elif len(pre_idx) > len(post_idx):
                    # not enough fit points, fill with nearest neighbour on side with the most data points
                    traces[:, gap_idx] = np.repeat(traces[:, pre_idx[-1]] * np.ones((1, 1)), len(gap_idx), 0).T
                elif len(post_idx) > len(pre_idx):
                    # not enough fit points, fill with nearest neighbour on side with the most data points
                    traces[:, gap_idx] = np.repeat(traces[:, post_idx[0]] * np.ones((1, 1)), len(gap_idx), 0).T
                elif len(all_idx) > 0:
                    # not enough fit points, both sides tied for most data points, fill with last pre value
                    traces[:, gap_idx] = np.repeat(traces[:, pre_idx[-1]] * np.ones((1, 1)), len(gap_idx), 0).T
                else:
                    # No data to interpolate from on either side of gap;
                    # Fill with zeros
                    traces[:, gap_idx] = 0

        return traces


def remove_artifacts(recording, triggers, ms_before=0.5, ms_after=3, mode='zeros', fit_sample_spacing=1.):
    '''
    Removes stimulation artifacts from recording extractor traces. By default, 
    artifact periods are zeroed-out (mode = 'zeros'). This is only recommended 
    for traces that are centered around zero (e.g. through a prior highpass
    filter); if this is not the case, linear and cubic interpolation modes are
    also available, controlled by the 'mode' input argument.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor to remove artifacts from
    triggers: list
        List of int with the stimulation trigger frames
    ms_before: float
        Time interval in ms to remove before the trigger events
    ms_after: float
        Time interval in ms to remove after the trigger events
    mode: str
        Determines what artifacts are replaced by. Can be one of the following:
            
        - 'zeros' (default): Artifacts are replaced by zeros.
        
        - 'linear': Replacement are obtained through Linear interpolation between
           the trace before and after the artifact.
           If the trace starts or ends with an artifact period, the gap is filled
           with the closest available value before or after the artifact.
        
        - 'cubic': Cubic spline interpolation between the trace before and after
           the artifact, referenced to evenly spaced fit points before and after
           the artifact. This is an option thatcan be helpful if there are
           significant LFP effects around the time of the artifact, but visual
           inspection of fit behaviour with your chosen settings is recommended.
           The spacing of fit points is controlled by 'fit_sample_spacing', with
           greater spacing between points leading to a fit that is less sensitive
           to high frequency fluctuations but at the cost of a less smooth
           continuation of the trace.
           If the trace starts or ends with an artifact, the gap is filled with
           the closest available value before or after the artifact.
    fit_sample_spacing: float
        Determines the spacing (in ms) of reference points for the cubic spline
        fit if mode = 'cubic'. Default = 1ms. Note: The actual fit samples are 
        the median of the 5 data points around the time of each sample point to
        avoid excessive influence from hyper-local fluctuations.
        

    Returns
    -------
    removed_recording: RemoveArtifactsRecording
        The recording extractor after artifact removal

    '''
    return RemoveArtifactsRecording(
        recording=recording, triggers=triggers, ms_before=ms_before,
        ms_after=ms_after, mode=mode, fit_sample_spacing=fit_sample_spacing)
