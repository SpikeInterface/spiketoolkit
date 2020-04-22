from spiketoolkit.validation.quality_metric_classes.metric_data import MetricData
from spiketoolkit.validation.quality_metric_classes.amplitude_cutoff import AmplitudeCutoff
from spiketoolkit.validation.quality_metric_classes.silhouette_score import SilhouetteScore
from spiketoolkit.validation.quality_metric_classes.num_spikes import NumSpikes
from spiketoolkit.validation.quality_metric_classes.d_prime import DPrime
from spiketoolkit.validation.quality_metric_classes.l_ratio import LRatio
from spiketoolkit.validation.quality_metric_classes.firing_rate import FiringRate
from spiketoolkit.validation.quality_metric_classes.presence_ratio import PresenceRatio
from spiketoolkit.validation.quality_metric_classes.isi_violation import ISIViolation
from spiketoolkit.validation.quality_metric_classes.snr import SNR
from spiketoolkit.validation.quality_metric_classes.isolation_distance import IsolationDistance
from spiketoolkit.validation.quality_metric_classes.nearest_neighbor import NearestNeighbor
from spiketoolkit.validation.quality_metric_classes.drift_metric import DriftMetric
from spiketoolkit.validation.quality_metric_classes.parameter_dictionaries import update_all_param_dicts_with_kwargs


def threshold_num_spikes(
        sorting,
        threshold,
        threshold_sign,
        sampling_frequency=None,
        **kwargs
):
    """
    Computes and thresholds the num spikes in the sorted dataset with the given sign and value.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated
    threshold: int or float
        The threshold for the given metric
    threshold_sign: str
        If 'less', will threshold any metric less than the given threshold
        If 'less_or_equal', will threshold any metric less than or equal to the given threshold
        If 'greater', will threshold any metric greater than the given threshold
        If 'greater_or_equal', will threshold any metric greater than or equal to the given threshold
    sampling_frequency:
        The sampling frequency of the result. If None, will check to see if sampling frequency is in sorting extractor
    **kwargs: keyword arguments
        Keyword arguments among the following:
            save_property_or_features: bool
                If True, the metric is saved as sorting property
            verbose: bool
                If True, will be verbose in metric computation

    Returns
    ----------
    threshold sorting extractor
    """
    params_dict = update_all_param_dicts_with_kwargs(kwargs)

    md = MetricData(sorting=sorting, sampling_frequency=sampling_frequency, recording=None, apply_filter=False,
                    duration_in_frames=None, freq_min=300.0, freq_max=6000.0, unit_ids=None, verbose=params_dict['verbose'])

    ns = NumSpikes(metric_data=md)
    threshold_sorting = ns.threshold_metric(threshold, threshold_sign, **kwargs)
    return threshold_sorting


def threshold_firing_rates(
        sorting,
        threshold,
        threshold_sign,
        duration_in_frames,
        sampling_frequency=None,
        **kwargs
):
    """
    Computes and thresholds the firing rates in the sorted dataset with the given sign and value.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated
    threshold: int or float
        The threshold for the given metric
    threshold_sign: str
        If 'less', will threshold any metric less than the given threshold
        If 'less_or_equal', will threshold any metric less than or equal to the given threshold
        If 'greater', will threshold any metric greater than the given threshold
        If 'greater_or_equal', will threshold any metric greater than or equal to the given threshold
    duration_in_frames: int
        Length of recording (in frames).
    sampling_frequency:
        The sampling frequency of the result. If None, will check to see if sampling frequency is in sorting extractor
    **kwargs: keyword arguments
        Keyword arguments among the following:
            save_property_or_features: bool
                If True, the metric is saved as sorting property
            verbose: bool
                If True, will be verbose in metric computation

    Returns
    ----------
    threshold sorting extractor
    """
    params_dict = update_all_param_dicts_with_kwargs(kwargs)

    md = MetricData(sorting=sorting, sampling_frequency=sampling_frequency, recording=None, apply_filter=False,
                    duration_in_frames=duration_in_frames, freq_min=300.0, freq_max=6000.0, unit_ids=None, 
                    verbose=params_dict['verbose'])

    fr = FiringRate(metric_data=md)
    threshold_sorting = fr.threshold_metric(threshold, threshold_sign, **kwargs)
    return threshold_sorting


def threshold_presence_ratios(
        sorting,
        threshold,
        threshold_sign,
        duration_in_frames,
        sampling_frequency=None,
        **kwargs
):
    """
    Computes and thresholds the presence ratios in the sorted dataset with the given sign and value.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated
    threshold: int or float
        The threshold for the given metric
    threshold_sign: str
        If 'less', will threshold any metric less than the given threshold
        If 'less_or_equal', will threshold any metric less than or equal to the given threshold
        If 'greater', will threshold any metric greater than the given threshold
        If 'greater_or_equal', will threshold any metric greater than or equal to the given threshold
    duration_in_frames: int
        Length of recording (in frames).
    sampling_frequency:
        The sampling frequency of the result. If None, will check to see if sampling frequency is in sorting extractor
    **kwargs: keyword arguments
        Keyword arguments among the following:
            save_property_or_features: bool
                If True, the metric is saved as sorting property
            verbose: bool
                If True, will be verbose in metric computation

    Returns
    ----------
    threshold sorting extractor
    """
    params_dict = update_all_param_dicts_with_kwargs(kwargs)

    md = MetricData(sorting=sorting, sampling_frequency=sampling_frequency, recording=None, apply_filter=False,
                    duration_in_frames=duration_in_frames, freq_min=300.0, freq_max=6000.0, unit_ids=None, 
                    verbose=params_dict['verbose'])

    pr = PresenceRatio(metric_data=md)
    threshold_sorting = pr.threshold_metric(threshold, threshold_sign, **kwargs)
    return threshold_sorting


def threshold_isi_violations(
        sorting,
        threshold,
        threshold_sign,
        duration_in_frames,
        isi_threshold=ISIViolation.params['isi_threshold'],
        min_isi=ISIViolation.params['min_isi'],
        sampling_frequency=None,
        **kwargs
):
    """
    Computes and thresholds the isi violations in the sorted dataset with the given sign and value.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated.
    threshold: int or float
        The threshold for the given metric.
    threshold_sign: str
        If 'less', will threshold any metric less than the given threshold
        If 'less_or_equal', will threshold any metric less than or equal to the given threshold
        If 'greater', will threshold any metric greater than the given threshold
        If 'greater_or_equal', will threshold any metric greater than or equal to the given threshold
    duration_in_frames: int
        Length of recording (in frames).
    isi_threshold: float
        The isi threshold for calculating isi violations.
    min_isi: float
        The minimum expected isi value.
    sampling_frequency:
        The sampling frequency of the result. If None, will check to see if sampling frequency is in sorting extractor.
    **kwargs: keyword arguments
        Keyword arguments among the following:
            save_property_or_features: bool
                If True, the metric is saved as sorting property
            verbose: bool
                If True, will be verbose in metric computation

    Returns
    ----------
    threshold sorting extractor
    """
    params_dict = update_all_param_dicts_with_kwargs(kwargs)

    md = MetricData(sorting=sorting, sampling_frequency=sampling_frequency, recording=None, apply_filter=False,
                    duration_in_frames=duration_in_frames, freq_min=300.0, freq_max=6000.0, unit_ids=None, 
                    verbose=params_dict['verbose'])

    iv = ISIViolation(metric_data=md)
    threshold_sorting = iv.threshold_metric(threshold, threshold_sign, isi_threshold, min_isi, **kwargs)
    return threshold_sorting


def threshold_amplitude_cutoffs(
        sorting,
        recording,
        threshold,
        threshold_sign,
        **kwargs
):
    """
    Computes and thresholds the amplitude cutoffs in the sorted dataset with the given sign and value.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated
    recording: RecordingExtractor
        The given recording extractor
    threshold: int or float
        The threshold for the given metric
    threshold_sign: str
        If 'less', will threshold any metric less than the given threshold
        If 'less_or_equal', will threshold any metric less than or equal to the given threshold
        If 'greater', will threshold any metric greater than the given threshold
        If 'greater_or_equal', will threshold any metric greater than or equal to the given threshold
    **kwargs: keyword arguments
        Keyword arguments among the following:
            method: str
                If 'absolute' (default), amplitudes are absolute amplitudes in uV are returned
                If 'relative', amplitudes are returned as ratios between waveform amplitudes and template amplitudes
            peak: str
                If maximum channel has to be found among negative peaks ('neg'), positive ('pos') or
                both ('both' - default)
            frames_before: int
                Frames before peak to compute amplitude
            frames_after: int
                Frames after peak to compute amplitude
            apply_filter: bool
                If True, recording is bandpass-filtered
            freq_min: float
                High-pass frequency for optional filter (default 300 Hz)
            freq_max: float
                Low-pass frequency for optional filter (default 6000 Hz)
            save_property_or_features: bool
                If true, it will save features in the sorting extractor
            recompute_info: bool
                    If True, waveforms are recomputed
            max_spikes_per_unit: int
                The maximum number of spikes to extract per unit

    Returns
    ----------
    threshold sorting extractor
    """
    params_dict = update_all_param_dicts_with_kwargs(kwargs)

    md = MetricData(sorting=sorting, sampling_frequency=recording.get_sampling_frequency(), recording=recording,
                    apply_filter=params_dict["apply_filter"], freq_min=params_dict["freq_min"],
                    duration_in_frames=None, freq_max=params_dict["freq_max"], unit_ids=None, verbose=params_dict['verbose'])
    md.compute_amplitudes(**kwargs)

    ac = AmplitudeCutoff(metric_data=md)
    threshold_sorting = ac.threshold_metric(threshold, threshold_sign, **kwargs)
    return threshold_sorting


def threshold_snrs(
        sorting,
        recording,
        threshold,
        threshold_sign,
        snr_mode=SNR.params['snr_mode'],
        snr_noise_duration=SNR.params['snr_noise_duration'],
        max_spikes_per_unit_for_snr=SNR.params['max_spikes_per_unit_for_snr'],
        template_mode=SNR.params['template_mode'],
        max_channel_peak=SNR.params['max_channel_peak'],
        **kwargs
):
    """
    Computes and thresholds the snrs in the sorted dataset with the given sign and value.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated.
    recording: RecordingExtractor
        The given recording extractor
    threshold: int or float
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
    max_spikes_per_unit_for_snr: int
        Maximum number of spikes to compute templates from (default 1000)
    template_mode: str
        Use 'mean' or 'median' to compute templates
    max_channel_peak: str
        If maximum channel has to be found among negative peaks ('neg'), positive ('pos') or both ('both' - default)
    **kwargs: keyword arguments
        Keyword arguments among the following:
            method: str
                If 'absolute' (default), amplitudes are absolute amplitudes in uV are returned.
                If 'relative', amplitudes are returned as ratios between waveform amplitudes and template amplitudes
            peak: str
                If maximum channel has to be found among negative peaks ('neg'), positive ('pos') or
                both ('both' - default)
            frames_before: int
                Frames before peak to compute amplitude
            frames_after: int
                Frames after peak to compute amplitude
            apply_filter: bool
                If True, recording is bandpass-filtered
            freq_min: float
                High-pass frequency for optional filter (default 300 Hz)
            freq_max: float
                Low-pass frequency for optional filter (default 6000 Hz)
            grouping_property: str
                Property to group channels. E.g. if the recording extractor has the 'group' property and
                'grouping_property' is 'group', then waveforms are computed group-wise.
            ms_before: float
                Time period in ms to cut waveforms before the spike events
            ms_after: float
                Time period in ms to cut waveforms after the spike events
            dtype: dtype
                The numpy dtype of the waveforms
            compute_property_from_recording: bool
                If True and 'grouping_property' is given, the property of each unit is assigned as the corresponding
                property of the recording extractor channel on which the average waveform is the largest
            max_channels_per_waveforms: int or None
                Maximum channels per waveforms to return. If None, all channels are returned
            n_jobs: int
                Number of parallel jobs (default 1)
            memmap: bool
                If True, waveforms are saved as memmap object (recommended for long recordings with many channels)
            save_property_or_features: bool
                If true, it will save features in the sorting extractor
            recompute_info: bool
                    If True, waveforms are recomputed
            max_spikes_per_unit: int
                The maximum number of spikes to extract per unit
            seed: int
                Random seed for reproducibility
            verbose: bool
                If True, will be verbose in metric computation

    Returns
    ----------
    threshold sorting extractor
    """
    params_dict = update_all_param_dicts_with_kwargs(kwargs)

    md = MetricData(sorting=sorting, sampling_frequency=recording.get_sampling_frequency(), recording=recording,
                    apply_filter=params_dict["apply_filter"], freq_min=params_dict["freq_min"],
                    duration_in_frames=None, freq_max=params_dict["freq_max"], unit_ids=None, verbose=params_dict['verbose'])

    snr = SNR(metric_data=md)
    threshold_sorting = snr.threshold_metric(threshold, threshold_sign, snr_mode, snr_noise_duration,
                                             max_spikes_per_unit_for_snr, template_mode, max_channel_peak, **kwargs)
    return threshold_sorting


def threshold_silhouette_scores(
        sorting,
        recording,
        threshold,
        threshold_sign,
        max_spikes_for_silhouette=SilhouetteScore.params['max_spikes_for_silhouette'],
        **kwargs
):
    """
    Computes and thresholds the silhouette scores in the sorted dataset with the given sign and value.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated
    recording: RecordingExtractor
        The given recording extractor
    threshold: int or float
        The threshold for the given metric
    threshold_sign: str
        If 'less', will threshold any metric less than the given threshold
        If 'less_or_equal', will threshold any metric less than or equal to the given threshold
        If 'greater', will threshold any metric greater than the given threshold
        If 'greater_or_equal', will threshold any metric greater than or equal to the given threshold
    max_spikes_for_silhouette: int
        Max spikes to be used for silhouette metric
    **kwargs: keyword arguments
        Keyword arguments among the following:
            method: str
                If 'absolute' (default), amplitudes are absolute amplitudes in uV are returned.
                If 'relative', amplitudes are returned as ratios between waveform amplitudes and template amplitudes
            peak: str
                If maximum channel has to be found among negative peaks ('neg'), positive ('pos') or
                both ('both' - default)
            frames_before: int
                Frames before peak to compute amplitude
            frames_after: int
                Frames after peak to compute amplitude
            apply_filter: bool
                If True, recording is bandpass-filtered
            freq_min: float
                High-pass frequency for optional filter (default 300 Hz)
            freq_max: float
                Low-pass frequency for optional filter (default 6000 Hz)
            grouping_property: str
                Property to group channels. E.g. if the recording extractor has the 'group' property and
                'grouping_property' is 'group', then waveforms are computed group-wise.
            ms_before: float
                Time period in ms to cut waveforms before the spike events
            ms_after: float
                Time period in ms to cut waveforms after the spike events
            dtype: dtype
                The numpy dtype of the waveforms
            compute_property_from_recording: bool
                If True and 'grouping_property' is given, the property of each unit is assigned as the corresponding
                property of the recording extractor channel on which the average waveform is the largest
            max_channels_per_waveforms: int or None
                Maximum channels per waveforms to return. If None, all channels are returned
            n_jobs: int
                Number of parallel jobs (default 1)
            memmap: bool
                If True, waveforms are saved as memmap object (recommended for long recordings with many channels)
            save_property_or_features: bool
                If true, it will save features in the sorting extractor
            recompute_info: bool
                    If True, waveforms are recomputed
            max_spikes_per_unit: int
                The maximum number of spikes to extract per unit
            seed: int
                Random seed for reproducibility
            verbose: bool
                If True, will be verbose in metric computation

    Returns
    ----------
    threshold sorting extractor
    """
    params_dict = update_all_param_dicts_with_kwargs(kwargs)

    md = MetricData(sorting=sorting, sampling_frequency=recording.get_sampling_frequency(), recording=recording,
                    apply_filter=params_dict["apply_filter"], freq_min=params_dict["freq_min"],
                    duration_in_frames=None, freq_max=params_dict["freq_max"], unit_ids=None, verbose=params_dict['verbose'])

    md.compute_pca_scores(**kwargs)

    silhouette_score = SilhouetteScore(metric_data=md)
    threshold_sorting = silhouette_score.threshold_metric(threshold, threshold_sign, max_spikes_for_silhouette,
                                                          **kwargs)
    return threshold_sorting


def threshold_d_primes(
        sorting,
        recording,
        threshold,
        threshold_sign,
        num_channels_to_compare=DPrime.params['num_channels_to_compare'],
        max_spikes_per_cluster=DPrime.params['max_spikes_per_cluster'],
        **kwargs
):
    """
    Computes and thresholds the d primes in the sorted dataset with the given sign and value.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated
    recording: RecordingExtractor
        The given recording extractor
    threshold: int or float
        The threshold for the given metric
    threshold_sign: str
        If 'less', will threshold any metric less than the given threshold
        If 'less_or_equal', will threshold any metric less than or equal to the given threshold
        If 'greater', will threshold any metric greater than the given threshold
        If 'greater_or_equal', will threshold any metric greater than or equal to the given threshold
    num_channels_to_compare: int
        The number of channels to be used for the PC extraction and comparison
    max_spikes_per_cluster: int
        Max spikes to be used from each unit
    **kwargs: keyword arguments
        Keyword arguments among the following:
            method: str
                If 'absolute' (default), amplitudes are absolute amplitudes in uV are returned.
                If 'relative', amplitudes are returned as ratios between waveform amplitudes and template amplitudes
            peak: str
                If maximum channel has to be found among negative peaks ('neg'), positive ('pos') or
                both ('both' - default)
            frames_before: int
                Frames before peak to compute amplitude
            frames_after: int
                Frames after peak to compute amplitude
            apply_filter: bool
                If True, recording is bandpass-filtered
            freq_min: float
                High-pass frequency for optional filter (default 300 Hz)
            freq_max: float
                Low-pass frequency for optional filter (default 6000 Hz)
            grouping_property: str
                Property to group channels. E.g. if the recording extractor has the 'group' property and
                'grouping_property' is 'group', then waveforms are computed group-wise.
            ms_before: float
                Time period in ms to cut waveforms before the spike events
            ms_after: float
                Time period in ms to cut waveforms after the spike events
            dtype: dtype
                The numpy dtype of the waveforms
            compute_property_from_recording: bool
                If True and 'grouping_property' is given, the property of each unit is assigned as the corresponding
                property of the recording extractor channel on which the average waveform is the largest
            max_channels_per_waveforms: int or None
                Maximum channels per waveforms to return. If None, all channels are returned
            n_jobs: int
                Number of parallel jobs (default 1)
            memmap: bool
                If True, waveforms are saved as memmap object (recommended for long recordings with many channels)
            save_property_or_features: bool
                If true, it will save features in the sorting extractor
            recompute_info: bool
                    If True, waveforms are recomputed
            max_spikes_per_unit: int
                The maximum number of spikes to extract per unit
            seed: int
                Random seed for reproducibility
            verbose: bool
                If True, will be verbose in metric computation

    Returns
    ----------
    threshold sorting extractor
    """
    params_dict = update_all_param_dicts_with_kwargs(kwargs)

    md = MetricData(sorting=sorting, sampling_frequency=recording.get_sampling_frequency(), recording=recording,
                    apply_filter=params_dict["apply_filter"], freq_min=params_dict["freq_min"],
                    duration_in_frames=None, freq_max=params_dict["freq_max"], unit_ids=None, verbose=params_dict['verbose'])

    md.compute_pca_scores(**kwargs)

    d_prime = DPrime(metric_data=md)
    threshold_sorting = d_prime.threshold_metric(threshold, threshold_sign, num_channels_to_compare,
                                                 max_spikes_per_cluster, **kwargs)
    return threshold_sorting


def threshold_l_ratios(
        sorting,
        recording,
        threshold,
        threshold_sign,
        num_channels_to_compare=LRatio.params['num_channels_to_compare'],
        max_spikes_per_cluster=LRatio.params['max_spikes_per_cluster'],
        **kwargs
):
    """
    Computes and thresholds the l ratios in the sorted dataset with the given sign and value.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated.
    recording: RecordingExtractor
        The given recording extractor
    threshold: int or float
        The threshold for the given metric.
    threshold_sign: str
        If 'less', will threshold any metric less than the given threshold.
        If 'less_or_equal', will threshold any metric less than or equal to the given threshold.
        If 'greater', will threshold any metric greater than the given threshold.
        If 'greater_or_equal', will threshold any metric greater than or equal to the given threshold.
    num_channels_to_compare: int
        The number of channels to be used for the PC extraction and comparison
    max_spikes_per_cluster: int
        Max spikes to be used from each unit
    **kwargs: keyword arguments
        Keyword arguments among the following:
            method: str
                If 'absolute' (default), amplitudes are absolute amplitudes in uV are returned.
                If 'relative', amplitudes are returned as ratios between waveform amplitudes and template amplitudes
            peak: str
                If maximum channel has to be found among negative peaks ('neg'), positive ('pos') or
                both ('both' - default)
            frames_before: int
                Frames before peak to compute amplitude
            frames_after: int
                Frames after peak to compute amplitude
            apply_filter: bool
                If True, recording is bandpass-filtered
            freq_min: float
                High-pass frequency for optional filter (default 300 Hz)
            freq_max: float
                Low-pass frequency for optional filter (default 6000 Hz)
            grouping_property: str
                Property to group channels. E.g. if the recording extractor has the 'group' property and
                'grouping_property' is 'group', then waveforms are computed group-wise.
            ms_before: float
                Time period in ms to cut waveforms before the spike events
            ms_after: float
                Time period in ms to cut waveforms after the spike events
            dtype: dtype
                The numpy dtype of the waveforms
            compute_property_from_recording: bool
                If True and 'grouping_property' is given, the property of each unit is assigned as the corresponding
                property of the recording extractor channel on which the average waveform is the largest
            max_channels_per_waveforms: int or None
                Maximum channels per waveforms to return. If None, all channels are returned
            n_jobs: int
                Number of parallel jobs (default 1)
            memmap: bool
                If True, waveforms are saved as memmap object (recommended for long recordings with many channels)
            save_property_or_features: bool
                If true, it will save features in the sorting extractor
            recompute_info: bool
                    If True, waveforms are recomputed
            max_spikes_per_unit: int
                The maximum number of spikes to extract per unit
            seed: int
                Random seed for reproducibility
            verbose: bool
                If True, will be verbose in metric computation

    Returns
    ----------
    threshold sorting extractor
    """
    params_dict = update_all_param_dicts_with_kwargs(kwargs)

    md = MetricData(sorting=sorting, sampling_frequency=recording.get_sampling_frequency(), recording=recording,
                    apply_filter=params_dict["apply_filter"], freq_min=params_dict["freq_min"],
                    duration_in_frames=None, freq_max=params_dict["freq_max"], unit_ids=None, verbose=params_dict['verbose'])

    md.compute_pca_scores(**kwargs)

    l_ratio = LRatio(metric_data=md)
    threshold_sorting = l_ratio.threshold_metric(threshold, threshold_sign, num_channels_to_compare,
                                                 max_spikes_per_cluster, **kwargs)
    return threshold_sorting


def threshold_isolation_distances(
        sorting,
        recording,
        threshold,
        threshold_sign,
        num_channels_to_compare=IsolationDistance.params['num_channels_to_compare'],
        max_spikes_per_cluster=IsolationDistance.params['max_spikes_per_cluster'],
        **kwargs
):
    """
    Computes and thresholds the isolation distances in the sorted dataset with the given sign and value.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated.
    recording: RecordingExtractor
        The given recording extractor
    threshold: int or float
        The threshold for the given metric.
    threshold_sign: str
        If 'less', will threshold any metric less than the given threshold.
        If 'less_or_equal', will threshold any metric less than or equal to the given threshold.
        If 'greater', will threshold any metric greater than the given threshold.
        If 'greater_or_equal', will threshold any metric greater than or equal to the given threshold.
    num_channels_to_compare: int
        The number of channels to be used for the PC extraction and comparison
    max_spikes_per_cluster: int
        Max spikes to be used from each unit
    **kwargs: keyword arguments
        Keyword arguments among the following:
            method: str
                If 'absolute' (default), amplitudes are absolute amplitudes in uV are returned.
                If 'relative', amplitudes are returned as ratios between waveform amplitudes and template amplitudes
            peak: str
                If maximum channel has to be found among negative peaks ('neg'), positive ('pos') or
                both ('both' - default)
            frames_before: int
                Frames before peak to compute amplitude
            frames_after: int
                Frames after peak to compute amplitude
            apply_filter: bool
                If True, recording is bandpass-filtered
            freq_min: float
                High-pass frequency for optional filter (default 300 Hz)
            freq_max: float
                Low-pass frequency for optional filter (default 6000 Hz)
            grouping_property: str
                Property to group channels. E.g. if the recording extractor has the 'group' property and
                'grouping_property' is 'group', then waveforms are computed group-wise.
            ms_before: float
                Time period in ms to cut waveforms before the spike events
            ms_after: float
                Time period in ms to cut waveforms after the spike events
            dtype: dtype
                The numpy dtype of the waveforms
            compute_property_from_recording: bool
                If True and 'grouping_property' is given, the property of each unit is assigned as the corresponding
                property of the recording extractor channel on which the average waveform is the largest
            max_channels_per_waveforms: int or None
                Maximum channels per waveforms to return. If None, all channels are returned
            n_jobs: int
                Number of parallel jobs (default 1)
            memmap: bool
                If True, waveforms are saved as memmap object (recommended for long recordings with many channels)
            save_property_or_features: bool
                If true, it will save features in the sorting extractor
            recompute_info: bool
                    If True, waveforms are recomputed
            max_spikes_per_unit: int
                The maximum number of spikes to extract per unit
            seed: int
                Random seed for reproducibility
            verbose: bool
                If True, will be verbose in metric computation

    Returns
    ----------
    threshold sorting extractor
    """
    params_dict = update_all_param_dicts_with_kwargs(kwargs)

    md = MetricData(sorting=sorting, sampling_frequency=recording.get_sampling_frequency(), recording=recording,
                    apply_filter=params_dict["apply_filter"], freq_min=params_dict["freq_min"],
                    duration_in_frames=None, freq_max=params_dict["freq_max"], unit_ids=None, verbose=params_dict['verbose'])

    md.compute_pca_scores(**kwargs)

    isolaiton_distance = IsolationDistance(metric_data=md)
    threshold_sorting = isolaiton_distance.threshold_metric(threshold, threshold_sign, num_channels_to_compare,
                                                            max_spikes_per_cluster, **kwargs)
    return threshold_sorting


def threshold_nn_metrics(
        sorting,
        recording,
        threshold,
        threshold_sign,
        metric_name="nn_hit_rate",
        num_channels_to_compare=NearestNeighbor.params['num_channels_to_compare'],
        max_spikes_per_cluster=NearestNeighbor.params['max_spikes_per_cluster'],
        max_spikes_for_nn=NearestNeighbor.params['max_spikes_for_nn'],
        n_neighbors=NearestNeighbor.params['n_neighbors'],
        **kwargs
):
    """
    Computes and thresholds the specified nearest neighbor metric for the sorted dataset with the given sign and value.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated.
    recording: RecordingExtractor
        The given recording extractor
    threshold: int or float
        The threshold for the given metric.
    threshold_sign: str
        If 'less', will threshold any metric less than the given threshold.
        If 'less_or_equal', will threshold any metric less than or equal to the given threshold.
        If 'greater', will threshold any metric greater than the given threshold.
        If 'greater_or_equal', will threshold any metric greater than or equal to the given threshold.
    metric_name: str
        The name of the nearest neighbor metric to be thresholded (either "nn_hit_rate" or "nn_miss_rate").
    num_channels_to_compare: int
        The number of channels to be used for the PC extraction and comparison
    max_spikes_per_cluster: int
        Max spikes to be used from each unit
    max_spikes_for_nn: int
        Max spikes to be used for nearest-neighbors calculation.
    n_neighbors: int
        Number of neighbors to compare.
    **kwargs: keyword arguments
        Keyword arguments among the following:
            method: str
                If 'absolute' (default), amplitudes are absolute amplitudes in uV are returned.
                If 'relative', amplitudes are returned as ratios between waveform amplitudes and template amplitudes
            peak: str
                If maximum channel has to be found among negative peaks ('neg'), positive ('pos') or
                both ('both' - default)
            frames_before: int
                Frames before peak to compute amplitude
            frames_after: int
                Frames after peak to compute amplitude
            apply_filter: bool
                If True, recording is bandpass-filtered
            freq_min: float
                High-pass frequency for optional filter (default 300 Hz)
            freq_max: float
                Low-pass frequency for optional filter (default 6000 Hz)
            grouping_property: str
                Property to group channels. E.g. if the recording extractor has the 'group' property and
                'grouping_property' is 'group', then waveforms are computed group-wise.
            ms_before: float
                Time period in ms to cut waveforms before the spike events
            ms_after: float
                Time period in ms to cut waveforms after the spike events
            dtype: dtype
                The numpy dtype of the waveforms
            compute_property_from_recording: bool
                If True and 'grouping_property' is given, the property of each unit is assigned as the corresponding
                property of the recording extractor channel on which the average waveform is the largest
            max_channels_per_waveforms: int or None
                Maximum channels per waveforms to return. If None, all channels are returned
            n_jobs: int
                Number of parallel jobs (default 1)
            memmap: bool
                If True, waveforms are saved as memmap object (recommended for long recordings with many channels)
            save_property_or_features: bool
                If true, it will save features in the sorting extractor
            recompute_info: bool
                    If True, waveforms are recomputed
            max_spikes_per_unit: int
                The maximum number of spikes to extract per unit
            seed: int
                Random seed for reproducibility
            verbose: bool
                If True, will be verbose in metric computation

    Returns
    ----------
    threshold sorting extractor
    """
    params_dict = update_all_param_dicts_with_kwargs(kwargs)

    md = MetricData(sorting=sorting, sampling_frequency=recording.get_sampling_frequency(), recording=recording,
                    apply_filter=params_dict["apply_filter"], freq_min=params_dict["freq_min"],
                    duration_in_frames=None, freq_max=params_dict["freq_max"], unit_ids=None, verbose=params_dict['verbose'])

    md.compute_pca_scores(**kwargs)

    nn = NearestNeighbor(metric_data=md)
    threshold_sorting = nn.threshold_metric(threshold, threshold_sign, metric_name, num_channels_to_compare,
                                            max_spikes_per_cluster, max_spikes_for_nn, n_neighbors, **kwargs)
    return threshold_sorting


def threshold_drift_metrics(
        sorting,
        recording,
        threshold,
        threshold_sign,
        metric_name="max_drift",
        drift_metrics_interval_s=DriftMetric.params['drift_metrics_interval_s'],
        drift_metrics_min_spikes_per_interval=DriftMetric.params['drift_metrics_min_spikes_per_interval'],
        **kwargs
):
    """
    Computes and thresholds the specified drift metric for the sorted dataset with the given sign and value.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated.
    recording: RecordingExtractor
        The given recording extractor
    threshold: int or float
        The threshold for the given metric.
    threshold_sign: str
        If 'less', will threshold any metric less than the given threshold.
        If 'less_or_equal', will threshold any metric less than or equal to the given threshold.
        If 'greater', will threshold any metric greater than the given threshold.
        If 'greater_or_equal', will threshold any metric greater than or equal to the given threshold.
    metric_name: str
        The name of the nearest neighbor metric to be thresholded (either "max_drift" or "cumulative_drift").
    drift_metrics_interval_s: float
        Time period for evaluating drift.
    drift_metrics_min_spikes_per_interval: int
        Minimum number of spikes for evaluating drift metrics per interval.
    **kwargs: keyword arguments
        Keyword arguments among the following:
            method: str
                If 'absolute' (default), amplitudes are absolute amplitudes in uV are returned.
                If 'relative', amplitudes are returned as ratios between waveform amplitudes and template amplitudes
            peak: str
                If maximum channel has to be found among negative peaks ('neg'), positive ('pos') or
                both ('both' - default)
            frames_before: int
                Frames before peak to compute amplitude
            frames_after: int
                Frames after peak to compute amplitude
            apply_filter: bool
                If True, recording is bandpass-filtered
            freq_min: float
                High-pass frequency for optional filter (default 300 Hz)
            freq_max: float
                Low-pass frequency for optional filter (default 6000 Hz)
            grouping_property: str
                Property to group channels. E.g. if the recording extractor has the 'group' property and
                'grouping_property' is 'group', then waveforms are computed group-wise.
            ms_before: float
                Time period in ms to cut waveforms before the spike events
            ms_after: float
                Time period in ms to cut waveforms after the spike events
            dtype: dtype
                The numpy dtype of the waveforms
            compute_property_from_recording: bool
                If True and 'grouping_property' is given, the property of each unit is assigned as the corresponding
                property of the recording extractor channel on which the average waveform is the largest
            max_channels_per_waveforms: int or None
                Maximum channels per waveforms to return. If None, all channels are returned
            n_jobs: int
                Number of parallel jobs (default 1)
            memmap: bool
                If True, waveforms are saved as memmap object (recommended for long recordings with many channels)
            save_property_or_features: bool
                If true, it will save features in the sorting extractor
            recompute_info: bool
                    If True, waveforms are recomputed
            max_spikes_per_unit: int
                The maximum number of spikes to extract per unit
            seed: int
                Random seed for reproducibility
            verbose: bool
                If True, will be verbose in metric computation

    Returns
    ----------
    threshold sorting extractor
    """
    params_dict = update_all_param_dicts_with_kwargs(kwargs)

    md = MetricData(sorting=sorting, sampling_frequency=recording.get_sampling_frequency(), recording=recording,
                    apply_filter=params_dict["apply_filter"], freq_min=params_dict["freq_min"],
                    duration_in_frames=None, freq_max=params_dict["freq_max"], unit_ids=None, verbose=params_dict['verbose'])

    md.compute_pca_scores(**kwargs)

    dm = DriftMetric(metric_data=md)
    threshold_sorting = dm.threshold_metric(threshold, threshold_sign, metric_name, drift_metrics_interval_s,
                                            drift_metrics_min_spikes_per_interval, **kwargs)
    return threshold_sorting
