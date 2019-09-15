from .thresholdcurator import ThresholdCurator
import spiketoolkit as st


class ThresholdSNR(ThresholdCurator):

    curator_name = 'ThresholdMinSNR'
    installed = True  # check at class level if installed or not
    curator_gui_params = [
        {'name': 'threshold', 'type': 'float', 'value':5.0, 'default':5.0, 'title': "The threshold for the given metric."},
        {'name': 'threshold_sign', 'type': 'str', 'value':'less', 'default':'less', 'title': "If 'less', will threshold any metric less than the given threshold. If 'greater', will threshold any metric greater than the given threshold."},
        {'name': 'snr_mode', 'type': 'str', 'value':'mad', 'default':'mad', 'title': "Mode to compute noise SNR ('mad' | 'std' - default 'mad')"},
        {'name': 'snr_noise_duration', 'type': 'float', 'value':10.0, 'default':10.0, 'title': "Number of seconds to compute noise level from (default 10.0)."},
        {'name': 'max_snr_waveforms', 'type': 'float', 'value':1000, 'default':1000, 'title': "Maximum number of waveforms to compute templates from (default 1000)."},
        {'name': 'seed', 'type': 'int', 'value':0, 'default':0, 'title': "Random seed for computing SNR."},
   ]
    installation_mesg = "" # err

    def __init__(self, sorting, recording, threshold=5.0, threshold_sign='less', snr_mode='mad', snr_noise_duration=10.0, \
                 max_snr_waveforms=1000, recompute_waveform_info=True, save_features_props=False, metric_calculator=None, seed=0):
        metric_name = 'snr'
        if metric_calculator is None:
            self._metric_calculator = st.validation.MetricCalculator(sorting, sampling_frequency=recording.get_sampling_frequency(), \
                                                                     unit_ids=None, epoch_tuples=None, epoch_names=None)
            self._metric_calculator.set_recording(recording)
            self._metric_calculator.compute_snrs(snr_mode, snr_noise_duration, max_snr_waveforms, recompute_waveform_info=recompute_waveform_info, \
                                                 save_features_props=save_features_props, seed=seed)
        else:
            self._metric_calculator = metric_calculator
            if metric_name not in self._metric_calculator.get_metrics_dict().keys():
                self._metric_calculator.set_recording(recording)
                self._metric_calculator.compute_snrs(snr_mode, snr_noise_duration, max_snr_waveforms, recompute_waveform_info=recompute_waveform_info, \
                                                     save_features_props=save_features_props, seed=seed)
        snrs_epoch = self._metric_calculator.get_metrics_dict()[metric_name][0] 

        ThresholdCurator.__init__(self, sorting=sorting, metrics_epoch=snrs_epoch)
        self.threshold_sorting(threshold=threshold, threshold_sign=threshold_sign)


def threshold_snr(sorting, recording, threshold=5.0, threshold_sign='less', snr_mode='mad', snr_noise_duration=10.0, \
                  max_snr_waveforms=1000, metric_calculator=None):
    '''
    Excludes units based on snr.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting extractor to be thresholded.
    recording: RecordingExtractor
        The recording extractor to compute SNR with.
    threshold:
        The threshold for the given metric.
    threshold_sign: str
        If 'less', will threshold any metric less than the given threshold.
        If 'greater', will threshold any metric greater than the given threshold.
    mode: str
        Mode to compute noise SNR ('mad' | 'std' - default 'mad')
    noise_duration: float
        Number of seconds to compute noise level from (default 10.0)
    max_snr_waveforms: int
        Maximum number of waveforms to compute templates from (default 1000)
    metric_calculator: MetricCalculator
        A metric calculator can be passed in with cached metrics.
    Returns
    -------
    thresholded_sorting: ThresholdSNR
        The thresholded sorting extractor

    '''
    return ThresholdSNR(
        sorting=sorting, 
        recording=recording,
        threshold=threshold, 
        threshold_sign=threshold_sign, 
        snr_mode=snr_mode, 
        snr_noise_duration=snr_noise_duration,
        max_snr_waveforms=max_snr_waveforms,
        metric_calculator=metric_calculator
    )
