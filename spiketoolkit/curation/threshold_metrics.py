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
from spiketoolkit.validation.quality_metric_classes.parameter_dictionaries import get_recording_params, \
    get_amplitude_params, get_pca_scores_params, get_feature_params, update_param_dicts, update_param_dicts_with_kwargs


def threshold_num_spikes(
        sorting,
        threshold,
        threshold_sign,
        sampling_frequency=None,
        save_as_property=True,
        verbose=NumSpikes.params['verbose'],
):
    """
    Computes and thresholds the num spikes in the sorted dataset with the given sign and value.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated.
    threshold: int or float
        The threshold for the given metric.
    threshold_sign: str
        If 'less', will threshold any metric less than the given threshold.
        If 'less_or_equal', will threshold any metric less than or equal to the given threshold.
        If 'greater', will threshold any metric greater than the given threshold.
        If 'greater_or_equal', will threshold any metric greater than or equal to the given threshold.
    sampling_frequency:
        The sampling frequency of the result. If None, will check to see if sampling frequency is in sorting extractor.
    save_as_property: bool
        If True, the metric is saved as sorting property
    verbose: bool
        If True, will be verbose in metric computation.
    Returns
    ----------
    threshold sorting extractor
    """
    md = MetricData(
        sorting=sorting,
        sampling_frequency=sampling_frequency,
        recording=None,
        apply_filter=False,
        freq_min=300.0,
        freq_max=6000.0,
        unit_ids=None,
        epoch_tuples=None,
        epoch_names=None,
        verbose=verbose,
    )

    ns = NumSpikes(metric_data=md)
    threshold_sorting = ns.threshold_metric(threshold, threshold_sign, save_as_property)
    return threshold_sorting


def threshold_firing_rates(
        sorting,
        threshold,
        threshold_sign,
        sampling_frequency=None,
        save_as_property=True,
        verbose=FiringRate.params['verbose'],
):
    """
    Computes and thresholds the firing rates in the sorted dataset with the given sign and value.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated.
    threshold: int or float
        The threshold for the given metric.
    threshold_sign: str
        If 'less', will threshold any metric less than the given threshold.
        If 'less_or_equal', will threshold any metric less than or equal to the given threshold.
        If 'greater', will threshold any metric greater than the given threshold.
        If 'greater_or_equal', will threshold any metric greater than or equal to the given threshold.
    sampling_frequency:
        The sampling frequency of the result. If None, will check to see if sampling frequency is in sorting extractor.
    save_as_property: bool
        If True, the metric is saved as sorting property
    verbose: bool
        If True, will be verbose in metric computation.
    Returns
    ----------
    threshold sorting extractor
    """
    md = MetricData(
        sorting=sorting,
        sampling_frequency=sampling_frequency,
        recording=None,
        apply_filter=False,
        freq_min=300.0,
        freq_max=6000.0,
        unit_ids=None,
        epoch_tuples=None,
        epoch_names=None,
        verbose=verbose,
    )

    fr = FiringRate(metric_data=md)
    threshold_sorting = fr.threshold_metric(threshold, threshold_sign, save_as_property)
    return threshold_sorting


def threshold_presence_ratios(
        sorting,
        threshold,
        threshold_sign,
        sampling_frequency=None,
        save_as_property=True,
        verbose=PresenceRatio.params['verbose'],
):
    """
    Computes and thresholds the presence ratios in the sorted dataset with the given sign and value.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated.
    threshold: int or float
        The threshold for the given metric.
    threshold_sign: str
        If 'less', will threshold any metric less than the given threshold.
        If 'less_or_equal', will threshold any metric less than or equal to the given threshold.
        If 'greater', will threshold any metric greater than the given threshold.
        If 'greater_or_equal', will threshold any metric greater than or equal to the given threshold.
    sampling_frequency:
        The sampling frequency of the result. If None, will check to see if sampling frequency is in sorting extractor.
    save_as_property: bool
        If True, the metric is saved as sorting property
    verbose: bool
        If True, will be verbose in metric computation.
    Returns
    ----------
    threshold sorting extractor
    """
    md = MetricData(
        sorting=sorting,
        sampling_frequency=sampling_frequency,
        recording=None,
        apply_filter=False,
        freq_min=300.0,
        freq_max=6000.0,
        unit_ids=None,
        epoch_tuples=None,
        epoch_names=None,
        verbose=verbose,
    )

    pr = PresenceRatio(metric_data=md)
    threshold_sorting = pr.threshold_metric(threshold, threshold_sign, save_as_property)
    return threshold_sorting


def threshold_isi_violations(
        sorting,
        threshold,
        threshold_sign,
        isi_threshold=ISIViolation.params['isi_threshold'],
        min_isi=ISIViolation.params['min_isi'],
        sampling_frequency=None,
        save_as_property=True,
        verbose=ISIViolation.params['verbose']
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
        If 'less', will threshold any metric less than the given threshold.
        If 'less_or_equal', will threshold any metric less than or equal to the given threshold.
        If 'greater', will threshold any metric greater than the given threshold.
        If 'greater_or_equal', will threshold any metric greater than or equal to the given threshold.
    isi_threshold: float
        The isi threshold for calculating isi violations.
    min_isi: float
        The minimum expected isi value.
    sampling_frequency:
        The sampling frequency of the result. If None, will check to see if sampling frequency is in sorting extractor.
    save_as_property: bool
        If True, the metric is saved as sorting property
    verbose: bool
        If True, will be verbose in metric computation.
    Returns
    ----------
    threshold sorting extractor
    """
    md = MetricData(
        sorting=sorting,
        sampling_frequency=sampling_frequency,
        recording=None,
        apply_filter=False,
        freq_min=300.0,
        freq_max=6000.0,
        unit_ids=None,
        epoch_tuples=None,
        epoch_names=None,
        verbose=verbose,
    )

    iv = ISIViolation(metric_data=md)
    threshold_sorting = iv.threshold_metric(threshold, threshold_sign, isi_threshold, min_isi, save_as_property)
    return threshold_sorting


def threshold_amplitude_cutoffs(
        sorting,
        recording,
        threshold,
        threshold_sign,
        save_as_property=True,
        seed=AmplitudeCutoff.params['seed'],
        verbose=AmplitudeCutoff.params['verbose'],
        **kwargs
):
    """
    Computes and thresholds the amplitude cutoffs in the sorted dataset with the given sign and value.

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
    save_as_property: bool
        If True, the metric is saved as sorting property
    seed: int
        Random seed for reproducibility
    verbose: bool
        If True, will be verbose in metric computation.
    **kwargs: keyword arguments
        Keyword arguments among the following:
            amp_method: str
                If 'absolute' (default), amplitudes are absolute amplitudes in uV are returned.
                If 'relative', amplitudes are returned as ratios between waveform amplitudes and template amplitudes.
            amp_peak: str
                If maximum channel has to be found among negative peaks ('neg'), positive ('pos') or both ('both' - default)
            amp_frames_before: int
                Frames before peak to compute amplitude.
            amp_frames_after: int
                Frames after peak to compute amplitude.
            apply_filter: bool
                If True, recording is bandpass-filtered.
            freq_min: float
                High-pass frequency for optional filter (default 300 Hz).
            freq_max: float
                Low-pass frequency for optional filter (default 6000 Hz).
            save_features_props: bool
                If true, it will save features in the sorting extractor.
            recompute_info: bool
                    If True, waveforms are recomputed
            max_spikes_per_unit: int
                The maximum number of spikes to extract per unit.
    Returns
    ----------
    threshold sorting extractor
    """
    rp_dict, ap_dict, pca_dict, ep_dict, fp_dict = update_param_dicts_with_kwargs(kwargs)

    md = MetricData(
        sorting=sorting,
        sampling_frequency=recording.get_sampling_frequency(),
        recording=recording,
        apply_filter=rp_dict["apply_filter"],
        freq_min=rp_dict["freq_min"],
        freq_max=rp_dict["freq_max"],
        unit_ids=None,
        epoch_tuples=None,
        epoch_names=None,
        verbose=verbose
    )
    md.compute_amplitudes(
        amp_method=ap_dict["amp_method"],
        amp_peak=ap_dict["amp_peak"],
        amp_frames_before=ap_dict["amp_frames_before"],
        amp_frames_after=ap_dict["amp_frames_after"],
        max_spikes_per_unit=fp_dict["max_spikes_per_unit"],
        save_features_props=fp_dict['save_features_props'],
        recompute_info=fp_dict['recompute_info'],
        seed=seed,
    )
    ac = AmplitudeCutoff(metric_data=md)
    threshold_sorting = ac.threshold_metric(threshold, threshold_sign, save_as_property)
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
        save_as_property=True,
        seed=SNR.params['seed'],
        verbose=SNR.params['verbose'],
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
    recompute_info: bool
            If True, waveforms are recomputed
    save_as_property: bool
        If True, the metric is saved as sorting property
    seed: int
        Random seed for reproducibility
    verbose: bool
        If True, will be verbose in metric computation.
    **kwargs: keyword arguments
        Keyword arguments among the following:
            amp_method: str
                If 'absolute' (default), amplitudes are absolute amplitudes in uV are returned.
                If 'relative', amplitudes are returned as ratios between waveform amplitudes and template amplitudes.
            amp_peak: str
                If maximum channel has to be found among negative peaks ('neg'), positive ('pos') or both ('both' - default)
            amp_frames_before: int
                Frames before peak to compute amplitude.
            amp_frames_after: int
                Frames after peak to compute amplitude.
            apply_filter: bool
                If True, recording is bandpass-filtered.
            freq_min: float
                High-pass frequency for optional filter (default 300 Hz).
            freq_max: float
                Low-pass frequency for optional filter (default 6000 Hz).
            save_features_props: bool
                If true, it will save features in the sorting extractor.
            recompute_info: bool
                    If True, waveforms are recomputed
            max_spikes_per_unit: int
                The maximum number of spikes to extract per unit.

    Returns
    ----------
    threshold sorting extractor
    """
    rp_dict, ap_dict, pca_dict, ep_dict, fp_dict = update_param_dicts_with_kwargs(kwargs)

    md = MetricData(
        sorting=sorting,
        sampling_frequency=recording.get_sampling_frequency(),
        recording=recording,
        apply_filter=rp_dict["apply_filter"],
        freq_min=rp_dict["freq_min"],
        freq_max=rp_dict["freq_max"],
        unit_ids=None,
        epoch_tuples=None,
        epoch_names=None,
        verbose=verbose
    )

    snr = SNR(metric_data=md)
    threshold_sorting = snr.threshold_metric(threshold, threshold_sign, snr_mode, snr_noise_duration,
                                             max_spikes_per_unit_for_snr, template_mode, max_channel_peak,
                                             fp_dict['save_features_props'], fp_dict['recompute_info'], seed,
                                             save_as_property)
    return threshold_sorting


def threshold_silhouette_scores(
        sorting,
        recording,
        threshold,
        threshold_sign,
        max_spikes_for_silhouette=SilhouetteScore.params['max_spikes_for_silhouette'],
        save_as_property=True,
        seed=SilhouetteScore.params['seed'],
        verbose=SilhouetteScore.params['verbose'],
        **kwargs
):
    """
    Computes and thresholds the silhouette scores in the sorted dataset with the given sign and value.

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
    max_spikes_for_silhouette: int
        Max spikes to be used for silhouette metric.
    save_as_property: bool
        If True, the metric is saved as sorting property
    seed: int
        Random seed for reproducibility
    verbose: bool
        If True, will be verbose in metric computation.
    **kwargs: keyword arguments
        Keyword arguments among the following:
            amp_method: str
                If 'absolute' (default), amplitudes are absolute amplitudes in uV are returned.
                If 'relative', amplitudes are returned as ratios between waveform amplitudes and template amplitudes.
            amp_peak: str
                If maximum channel has to be found among negative peaks ('neg'), positive ('pos') or both ('both' - default)
            amp_frames_before: int
                Frames before peak to compute amplitude.
            amp_frames_after: int
                Frames after peak to compute amplitude.
            apply_filter: bool
                If True, recording is bandpass-filtered.
            freq_min: float
                High-pass frequency for optional filter (default 300 Hz).
            freq_max: float
                Low-pass frequency for optional filter (default 6000 Hz).
            save_features_props: bool
                If true, it will save features in the sorting extractor.
            recompute_info: bool
                    If True, waveforms are recomputed
            max_spikes_per_unit: int
                The maximum number of spikes to extract per unit.
    Returns
    ----------
    threshold sorting extractor
    """
    rp_dict, ap_dict, pca_dict, ep_dict, fp_dict = update_param_dicts_with_kwargs(kwargs)

    md = MetricData(
        sorting=sorting,
        sampling_frequency=recording.get_sampling_frequency(),
        recording=recording,
        apply_filter=rp_dict["apply_filter"],
        freq_min=rp_dict["freq_min"],
        freq_max=rp_dict["freq_max"],
        unit_ids=None,
        epoch_tuples=None,
        epoch_names=None,
        verbose=verbose
    )

    md.compute_pca_scores(
        n_comp=pca_dict["n_comp"],
        ms_before=pca_dict["ms_before"],
        ms_after=pca_dict["ms_after"],
        dtype=pca_dict["dtype"],
        max_spikes_per_unit=fp_dict["max_spikes_per_unit"],
        max_spikes_for_pca=pca_dict["max_spikes_for_pca"],
        save_features_props=fp_dict['save_features_props'],
        recompute_info=fp_dict['recompute_info'],
        seed=seed,
    )

    silhouette_score = SilhouetteScore(metric_data=md)
    threshold_sorting = silhouette_score.threshold_metric(
        threshold, threshold_sign, max_spikes_for_silhouette, seed, save_as_property)
    return threshold_sorting


def threshold_d_primes(
        sorting,
        recording,
        threshold,
        threshold_sign,
        num_channels_to_compare=DPrime.params['num_channels_to_compare'],
        max_spikes_per_cluster=DPrime.params['max_spikes_per_cluster'],
        save_as_property=True,
        seed=DPrime.params['seed'],
        verbose=DPrime.params['verbose'],
        **kwargs
):
    """
    Computes and thresholds the d primes in the sorted dataset with the given sign and value.

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
    save_as_property: bool
        If True, the metric is saved as sorting property
    seed: int
        Random seed for reproducibility
    verbose: bool
        If True, will be verbose in metric computation.
    **kwargs: keyword arguments
        Keyword arguments among the following:
            amp_method: str
                If 'absolute' (default), amplitudes are absolute amplitudes in uV are returned.
                If 'relative', amplitudes are returned as ratios between waveform amplitudes and template amplitudes.
            amp_peak: str
                If maximum channel has to be found among negative peaks ('neg'), positive ('pos') or both ('both' - default)
            amp_frames_before: int
                Frames before peak to compute amplitude.
            amp_frames_after: int
                Frames after peak to compute amplitude.
            apply_filter: bool
                If True, recording is bandpass-filtered.
            freq_min: float
                High-pass frequency for optional filter (default 300 Hz).
            freq_max: float
                Low-pass frequency for optional filter (default 6000 Hz).
            save_features_props: bool
                If true, it will save features in the sorting extractor.
            recompute_info: bool
                    If True, waveforms are recomputed
            max_spikes_per_unit: int
                The maximum number of spikes to extract per unit.

    Returns
    ----------
    threshold sorting extractor
    """
    rp_dict, ap_dict, pca_dict, ep_dict, fp_dict = update_param_dicts_with_kwargs(kwargs)

    md = MetricData(
        sorting=sorting,
        sampling_frequency=recording.get_sampling_frequency(),
        recording=recording,
        apply_filter=rp_dict["apply_filter"],
        freq_min=rp_dict["freq_min"],
        freq_max=rp_dict["freq_max"],
        unit_ids=None,
        epoch_tuples=None,
        epoch_names=None,
        verbose=verbose
    )

    md.compute_pca_scores(
        n_comp=pca_dict["n_comp"],
        ms_before=pca_dict["ms_before"],
        ms_after=pca_dict["ms_after"],
        dtype=pca_dict["dtype"],
        max_spikes_per_unit=fp_dict["max_spikes_per_unit"],
        max_spikes_for_pca=pca_dict["max_spikes_for_pca"],
        save_features_props=fp_dict['save_features_props'],
        recompute_info=fp_dict['recompute_info'],
        seed=seed,
    )

    d_prime = DPrime(metric_data=md)
    threshold_sorting = d_prime.threshold_metric(threshold, threshold_sign, num_channels_to_compare,
                                                 max_spikes_per_cluster, seed, save_as_property)
    return threshold_sorting


def threshold_l_ratios(
        sorting,
        recording,
        threshold,
        threshold_sign,
        num_channels_to_compare=LRatio.params['num_channels_to_compare'],
        max_spikes_per_cluster=LRatio.params['max_spikes_per_cluster'],
        save_as_property=True,
        seed=LRatio.params['seed'],
        verbose=LRatio.params['verbose'],
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
    save_as_property: bool
        If True, the metric is saved as sorting property
    seed: int
        Random seed for reproducibility
    verbose: bool
        If True, will be verbose in metric computation.
    **kwargs: keyword arguments
        Keyword arguments among the following:
            amp_method: str
                If 'absolute' (default), amplitudes are absolute amplitudes in uV are returned.
                If 'relative', amplitudes are returned as ratios between waveform amplitudes and template amplitudes.
            amp_peak: str
                If maximum channel has to be found among negative peaks ('neg'), positive ('pos') or both ('both' - default)
            amp_frames_before: int
                Frames before peak to compute amplitude.
            amp_frames_after: int
                Frames after peak to compute amplitude.
            apply_filter: bool
                If True, recording is bandpass-filtered.
            freq_min: float
                High-pass frequency for optional filter (default 300 Hz).
            freq_max: float
                Low-pass frequency for optional filter (default 6000 Hz).
            save_features_props: bool
                If true, it will save features in the sorting extractor.
            recompute_info: bool
                    If True, waveforms are recomputed
            max_spikes_per_unit: int
                The maximum number of spikes to extract per unit.

    Returns
    ----------
    threshold sorting extractor
    """
    rp_dict, ap_dict, pca_dict, ep_dict, fp_dict = update_param_dicts_with_kwargs(kwargs)

    md = MetricData(
        sorting=sorting,
        sampling_frequency=recording.get_sampling_frequency(),
        recording=recording,
        apply_filter=rp_dict["apply_filter"],
        freq_min=rp_dict["freq_min"],
        freq_max=rp_dict["freq_max"],
        unit_ids=None,
        epoch_tuples=None,
        epoch_names=None,
        verbose=verbose
    )

    md.compute_pca_scores(
        n_comp=pca_dict["n_comp"],
        ms_before=pca_dict["ms_before"],
        ms_after=pca_dict["ms_after"],
        dtype=pca_dict["dtype"],
        max_spikes_per_unit=fp_dict["max_spikes_per_unit"],
        max_spikes_for_pca=pca_dict["max_spikes_for_pca"],
        save_features_props=fp_dict['save_features_props'],
        recompute_info=fp_dict['recompute_info'],
        seed=seed,
    )

    l_ratio = LRatio(metric_data=md)
    threshold_sorting = l_ratio.threshold_metric(threshold, threshold_sign, num_channels_to_compare,
                                                 max_spikes_per_cluster, seed, save_as_property)
    return threshold_sorting


def threshold_isolation_distances(
        sorting,
        recording,
        threshold,
        threshold_sign,
        num_channels_to_compare=IsolationDistance.params['num_channels_to_compare'],
        max_spikes_per_cluster=IsolationDistance.params['max_spikes_per_cluster'],
        save_as_property=True,
        seed=IsolationDistance.params['seed'],
        verbose=IsolationDistance.params['verbose'],
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
    save_as_property: bool
        If True, the metric is saved as sorting property
    seed: int
        Random seed for reproducibility
    verbose: bool
        If True, will be verbose in metric computation.
    **kwargs: keyword arguments
        Keyword arguments among the following:
            amp_method: str
                If 'absolute' (default), amplitudes are absolute amplitudes in uV are returned.
                If 'relative', amplitudes are returned as ratios between waveform amplitudes and template amplitudes.
            amp_peak: str
                If maximum channel has to be found among negative peaks ('neg'), positive ('pos') or both ('both' - default)
            amp_frames_before: int
                Frames before peak to compute amplitude.
            amp_frames_after: int
                Frames after peak to compute amplitude.
            apply_filter: bool
                If True, recording is bandpass-filtered.
            freq_min: float
                High-pass frequency for optional filter (default 300 Hz).
            freq_max: float
                Low-pass frequency for optional filter (default 6000 Hz).
            save_features_props: bool
                If true, it will save features in the sorting extractor.
            recompute_info: bool
                    If True, waveforms are recomputed
            max_spikes_per_unit: int
                The maximum number of spikes to extract per unit.

    Returns
    ----------
    threshold sorting extractor
    """
    rp_dict, ap_dict, pca_dict, ep_dict, fp_dict = update_param_dicts_with_kwargs(kwargs)

    md = MetricData(
        sorting=sorting,
        sampling_frequency=recording.get_sampling_frequency(),
        recording=recording,
        apply_filter=rp_dict["apply_filter"],
        freq_min=rp_dict["freq_min"],
        freq_max=rp_dict["freq_max"],
        unit_ids=None,
        epoch_tuples=None,
        epoch_names=None,
        verbose=verbose
    )

    md.compute_pca_scores(
        n_comp=pca_dict["n_comp"],
        ms_before=pca_dict["ms_before"],
        ms_after=pca_dict["ms_after"],
        dtype=pca_dict["dtype"],
        max_spikes_per_unit=fp_dict["max_spikes_per_unit"],
        max_spikes_for_pca=pca_dict["max_spikes_for_pca"],
        save_features_props=fp_dict['save_features_props'],
        recompute_info=fp_dict['recompute_info'],
        seed=seed,
    )

    isolaiton_distance = IsolationDistance(metric_data=md)
    threshold_sorting = isolaiton_distance.threshold_metric(threshold, threshold_sign, num_channels_to_compare,
                                                            max_spikes_per_cluster, seed, save_as_property)
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
        save_as_property=True,
        seed=NearestNeighbor.params['seed'],
        verbose=NearestNeighbor.params['verbose'],
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
    save_as_property: bool
        If True, the metric is saved as sorting property
    seed: int
        Random seed for reproducibility
    verbose: bool
        If True, will be verbose in metric computation.
    **kwargs: keyword arguments
        Keyword arguments among the following:
            amp_method: str
                If 'absolute' (default), amplitudes are absolute amplitudes in uV are returned.
                If 'relative', amplitudes are returned as ratios between waveform amplitudes and template amplitudes.
            amp_peak: str
                If maximum channel has to be found among negative peaks ('neg'), positive ('pos') or both ('both' - default)
            amp_frames_before: int
                Frames before peak to compute amplitude.
            amp_frames_after: int
                Frames after peak to compute amplitude.
            apply_filter: bool
                If True, recording is bandpass-filtered.
            freq_min: float
                High-pass frequency for optional filter (default 300 Hz).
            freq_max: float
                Low-pass frequency for optional filter (default 6000 Hz).
            save_features_props: bool
                If true, it will save features in the sorting extractor.
            recompute_info: bool
                    If True, waveforms are recomputed
            max_spikes_per_unit: int
                The maximum number of spikes to extract per unit.

    Returns
    ----------
    threshold sorting extractor
    """
    rp_dict, ap_dict, pca_dict, ep_dict, fp_dict = update_param_dicts_with_kwargs(kwargs)

    md = MetricData(
        sorting=sorting,
        sampling_frequency=recording.get_sampling_frequency(),
        recording=recording,
        apply_filter=rp_dict["apply_filter"],
        freq_min=rp_dict["freq_min"],
        freq_max=rp_dict["freq_max"],
        unit_ids=None,
        epoch_tuples=None,
        epoch_names=None,
        verbose=verbose
    )
    md.compute_pca_scores(
        n_comp=pca_dict["n_comp"],
        ms_before=pca_dict["ms_before"],
        ms_after=pca_dict["ms_after"],
        dtype=pca_dict["dtype"],
        max_spikes_per_unit=fp_dict["max_spikes_per_unit"],
        max_spikes_for_pca=pca_dict["max_spikes_for_pca"],
        save_features_props=fp_dict['save_features_props'],
        recompute_info=fp_dict['recompute_info'],
        seed=seed,
    )

    nn = NearestNeighbor(metric_data=md)
    threshold_sorting = nn.threshold_metric(threshold, threshold_sign, metric_name, num_channels_to_compare,
                                            max_spikes_per_cluster, max_spikes_for_nn, n_neighbors, seed,
                                            save_as_property)
    return threshold_sorting


def threshold_drift_metrics(
        sorting,
        recording,
        threshold,
        threshold_sign,
        metric_name="max_drift",
        drift_metrics_interval_s=DriftMetric.params['drift_metrics_interval_s'],
        drift_metrics_min_spikes_per_interval=DriftMetric.params['drift_metrics_min_spikes_per_interval'],
        save_as_property=True,
        seed=DriftMetric.params['seed'],
        verbose=DriftMetric.params['verbose'],
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
    save_as_property: bool
        If True, the metric is saved as sorting property
    seed: int
        Random seed for reproducibility
    verbose: bool
        If True, will be verbose in metric computation.
    **kwargs: keyword arguments
        Keyword arguments among the following:
            amp_method: str
                If 'absolute' (default), amplitudes are absolute amplitudes in uV are returned.
                If 'relative', amplitudes are returned as ratios between waveform amplitudes and template amplitudes.
            amp_peak: str
                If maximum channel has to be found among negative peaks ('neg'), positive ('pos') or both ('both' - default)
            amp_frames_before: int
                Frames before peak to compute amplitude.
            amp_frames_after: int
                Frames after peak to compute amplitude.
            apply_filter: bool
                If True, recording is bandpass-filtered.
            freq_min: float
                High-pass frequency for optional filter (default 300 Hz).
            freq_max: float
                Low-pass frequency for optional filter (default 6000 Hz).
            save_features_props: bool
                If true, it will save features in the sorting extractor.
            recompute_info: bool
                    If True, waveforms are recomputed
            max_spikes_per_unit: int
                The maximum number of spikes to extract per unit.

    Returns
    ----------
    threshold sorting extractor
    """
    rp_dict, ap_dict, pca_dict, ep_dict, fp_dict = update_param_dicts_with_kwargs(kwargs)

    md = MetricData(
        sorting=sorting,
        sampling_frequency=recording.get_sampling_frequency(),
        recording=recording,
        apply_filter=rp_dict["apply_filter"],
        freq_min=rp_dict["freq_min"],
        freq_max=rp_dict["freq_max"],
        unit_ids=None,
        epoch_tuples=None,
        epoch_names=None,
        verbose=verbose
    )

    md.compute_pca_scores(
        n_comp=pca_dict["n_comp"],
        ms_before=pca_dict["ms_before"],
        ms_after=pca_dict["ms_after"],
        dtype=pca_dict["dtype"],
        max_spikes_per_unit=fp_dict["max_spikes_per_unit"],
        max_spikes_for_pca=pca_dict["max_spikes_for_pca"],
        save_features_props=fp_dict['save_features_props'],
        recompute_info=fp_dict['recompute_info'],
        seed=seed,
    )

    dm = DriftMetric(metric_data=md)
    threshold_sorting = dm.threshold_metric(threshold, threshold_sign, metric_name, drift_metrics_interval_s,
                                            drift_metrics_min_spikes_per_interval, save_as_property)
    return threshold_sorting
