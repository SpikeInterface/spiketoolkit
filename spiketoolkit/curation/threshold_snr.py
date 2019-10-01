from .thresholdcurator import ThresholdCurator
import spiketoolkit as st


class ThresholdSNR(ThresholdCurator):
    curator_name = 'ThresholdMinSNR'
    installed = True  # check at class level if installed or not
    curator_gui_params = [
        {'name': 'threshold', 'type': 'float', 'value': 5.0, 'default': 5.0,
         'title': "The threshold for the given metric."},
        {'name': 'threshold_sign', 'type': 'str', 'value': 'less', 'default': 'less',
         'title': "If 'less', will threshold any metric less than the given threshold. "
                  "If 'greater', will threshold any metric greater than the given threshold."},
        {'name': 'snr_mode', 'type': 'str', 'value': 'mad', 'default': 'mad',
         'title': "Mode to compute noise SNR ('mad' | 'std' - default 'mad')"},
        {'name': 'snr_noise_duration', 'type': 'float', 'value': 10.0, 'default': 10.0,
         'title': "Number of seconds to compute noise level from (default 10.0)."},
        {'name': 'max_snr_spikes_per_unit', 'type': 'float', 'value': 1000, 'default': 1000,
         'title': "Maximum number of waveforms to compute templates from (default 1000)."},
        {'name': 'apply_filter', 'type': 'bool', 'value': True, 'default': True,
         'title': "If True, recording is bandpass-filtered."},
        {'name': 'freq_min', 'type': 'float', 'value': 300.0, 'default': 300.0, 
        'title': "High-pass frequency for optional filter (default 300 Hz)."},
        {'name': 'freq_max', 'type': 'float', 'value': 6000.0, 'default': 6000.0, 
        'title': "Low-pass frequency for optional filter (default 6000 Hz)."},
        {'name': 'seed', 'type': 'int', 'value': 0, 'default': 0, 'title': "Random seed for computing SNR."},
    ]
    installation_mesg = ""  # err

    def __init__(self, sorting, recording, threshold=5.0, threshold_sign='less', snr_mode='mad',
                 snr_noise_duration=10.0, max_snr_spikes_per_unit=1000, recompute_info=True,
                 apply_filter=True, freq_min=300, freq_max=6000, save_features_props=False, 
                 metric_calculator=None, seed=0):
        metric_name = 'snr'
        if metric_calculator is None:
            self._metric_calculator = st.validation.MetricCalculator(sorting,
                                                                     sampling_frequency=recording.get_sampling_frequency(),
                                                                     unit_ids=None, epoch_tuples=None, epoch_names=None)
            self._metric_calculator.set_recording(recording, apply_filter=apply_filter, freq_min=freq_min, freq_max=freq_max)
            self._metric_calculator.compute_snrs(snr_mode, snr_noise_duration, max_snr_spikes_per_unit,
                                                 recompute_info=recompute_info,
                                                 save_features_props=save_features_props, seed=seed)
        else:
            self._metric_calculator = metric_calculator
            if metric_name not in self._metric_calculator.get_metrics_dict().keys():
                self._metric_calculator.set_recording(recording, apply_filter=apply_filter, freq_min=freq_min, freq_max=freq_max)
                self._metric_calculator.compute_snrs(snr_mode, snr_noise_duration, max_snr_spikes_per_unit,
                                                     recompute_info=recompute_info,
                                                     save_features_props=save_features_props, seed=seed)
        snrs_epoch = self._metric_calculator.get_metrics_dict()[metric_name][0]

        ThresholdCurator.__init__(self, sorting=sorting, metrics_epoch=snrs_epoch)
        self.threshold_sorting(threshold=threshold, threshold_sign=threshold_sign)


def threshold_snr(sorting, recording, threshold=5.0, threshold_sign='less', snr_mode='mad', snr_noise_duration=10.0,
                  max_snr_spikes_per_unit=1000, recompute_info=True, apply_filter=True, freq_min=300, freq_max=6000,
                  save_features_props=False, metric_calculator=None, seed=0):
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
        If 'less_or_equal', will threshold any metric less than or equal to the given threshold.
        If 'greater', will threshold any metric greater than the given threshold.
        If 'greater_or_equal', will threshold any metric greater than or equal to the given threshold.
    snr_mode: str
        Mode to compute noise SNR ('mad' | 'std' - default 'mad')
    snr_noise_duration: float
        Number of seconds to compute noise level from (default 10.0)
    max_snr_spikes_per_unit: int
        Maximum number of spikes to compute templates from (default 1000)
    recompute_info: bool
        If True, waveforms are recomputed
    apply_filter: bool
        If True, recording is bandpass-filtered.
    freq_min: float
        High-pass frequency for optional filter (default 300 Hz).
    freq_max: float
        Low-pass frequency for optional filter (default 6000 Hz).
    save_features_props: bool
        If True, waveforms and templates are saved as sorting features/properties
    metric_calculator: MetricCalculator
        A metric calculator can be passed in with cached metrics.
    seed: int
        Random seed for reproducibility
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
        max_snr_spikes_per_unit=max_snr_spikes_per_unit,
        recompute_info=recompute_info,
        apply_filter=apply_filter, 
        freq_min=freq_min, 
        freq_max=freq_max,
        save_features_props=save_features_props,
        metric_calculator=metric_calculator,
        seed=seed
    )
