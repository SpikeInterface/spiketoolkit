from spiketoolkit.validation.metric_data import MetricData
from spiketoolkit.validation.amplitude_cutoff import AmplitudeCutoff
from spiketoolkit.validation.silhouette_score import SilhouetteScore


def threshold_amplitude_cutoffs(
    sorting,
    recording,
    threshold,
    threshold_sign,
    epoch=None,
    recording_params_dict=MetricData.recording_params_dict,
    amplitude_params_dict=MetricData.amplitude_params_dict,
    metric_scope_params_dict=MetricData.metric_scope_params_dict,
    save_features_props=False,
    seed=0,
):
    """
    Computes and returns the amplitude cutoffs for the sorted dataset.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated.
    recording: RecordingExtractor
        The given recording extractor from which to extract amplitudes
    threshold: int or float
        The threshold for the given metric.
    threshold_sign: str
        If 'less', will threshold any metric less than the given threshold.
        If 'less_or_equal', will threshold any metric less than or equal to the given threshold.
        If 'greater', will threshold any metric greater than the given threshold.
        If 'greater_or_equal', will threshold any metric greater than or equal to the given threshold.
    epoch:
        The threshold will be applied to the specified epoch.
        If epoch is None, then it will default to the first epoch.
    amplitude_params_dict: dict
        This dictionary should contain any subset of the following parameters:
            amp_method: str
                If 'absolute' (default), amplitudes are absolute amplitudes in uV are returned.
                If 'relative', amplitudes are returned as ratios between waveform amplitudes and template amplitudes.
            amp_peak: str
                If maximum channel has to be found among negative peaks ('neg'), positive ('pos') or both ('both' - default)
            amp_frames_before: int
                Frames before peak to compute amplitude.
            amp_frames_after: int
                Frames after peak to compute amplitude.
    recording_params_dict: dict
        This dictionary should contain any subset of the following parameters:
            apply_filter: bool
                If True, recording is bandpass-filtered.
            freq_min: float
                High-pass frequency for optional filter (default 300 Hz).
            freq_max: float
                Low-pass frequency for optional filter (default 6000 Hz).
    quality_metric_params_dict: dict
        This dictionary should contain any subset of the following parameters:
            unit_ids: list
                List of unit ids to compute metric for. If not specified, all units are used
            epoch_tuples: list
                A list of tuples with a start and end time for each epoch
            epoch_names: list
                A list of strings for the names of the given epochs.
    save_features_props: bool
        If true, it will save amplitudes in the sorting extractor.
    seed: int
        Random seed for reproducibility
    Returns
    ----------
    amplitude_cutoffs_epochs: list of lists
        The amplitude cutoffs of the sorted units in the given epochs.
    """
    rp_dict = recording_params_dict.copy()
    ap_dict = amplitude_params_dict.copy()
    ms_dict = metric_scope_params_dict.copy()
    if ms_dict["unit_ids"] is None:
        ms_dict["unit_ids"] = sorting.get_unit_ids()

    md = MetricData(
        sorting=sorting,
        recording=recording,
        apply_filter=rp_dict["apply_filter"],
        freq_min=rp_dict["freq_min"],
        freq_max=rp_dict["freq_max"],
        unit_ids=ms_dict["unit_ids"],
        epoch_tuples=ms_dict["epoch_tuples"],
        epoch_names=ms_dict["epoch_names"],
    )
    md.compute_amplitudes(
        amp_method=ap_dict["amp_method"],
        amp_peak=ap_dict["amp_peak"],
        amp_frames_before=ap_dict["amp_frames_before"],
        amp_frames_after=ap_dict["amp_frames_after"],
        save_features_props=save_features_props,
        seed=seed,
    )
    ac = AmplitudeCutoff(metric_data=md)
    threshold_sorting = ac.threshold_metric(threshold, threshold_sign, epoch)
    return threshold_sorting


def threshold_silhouette_scores(
    sorting,
    recording,
    threshold,
    threshold_sign,
    epoch=None,
    recording_params_dict=MetricData.recording_params_dict,
    pca_scores_params_dict=MetricData.pca_scores_params_dict,
    metric_scope_params_dict=MetricData.metric_scope_params_dict,
    save_features_props=False,
    seed=0,
):
    """
    Computes and returns the silhouette scores in the sorted dataset.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated.

    recording: RecordingExtractor
        The given recording extractor from which to extract amplitudes

    threshold: int or float
        The threshold for the given metric.

    threshold_sign: str
        If 'less', will threshold any metric less than the given threshold.
        If 'less_or_equal', will threshold any metric less than or equal to the given threshold.
        If 'greater', will threshold any metric greater than the given threshold.
        If 'greater_or_equal', will threshold any metric greater than or equal to the given threshold.

    epoch:
        The threshold will be applied to the specified epoch.
        If epoch is None, then it will default to the first epoch.

    recording_params_dict: dict
        This dictionary should contain any subset of the following parameters:
            apply_filter: bool
                If True, recording is bandpass-filtered.
            freq_min: float
                High-pass frequency for optional filter (default 300 Hz).
            freq_max: float
                Low-pass frequency for optional filter (default 6000 Hz).

    pca_scores_params_dict: dict
        This dictionary should contain any subset of the following parameters:
            ms_before: float
                Time period in ms to cut waveforms before the spike events
            ms_after: float
                Time period in ms to cut waveforms after the spike events
            dtype: dtype
                The numpy dtype of the waveforms
            max_spikes_per_unit: int
                The maximum number of spikes to extract per unit.
            max_spikes_for_pca: int
                The maximum number of spikes to use to compute PCA.

    metric_scope_params_dict: dict
        This dictionary should contain any subset of the following parameters:
            unit_ids: list
                unit ids to compute metric for. If not specified, all units are used
            epoch_tuples: list
                A list of tuples with a start and end time for each epoch
            epoch_names: list
                A list of strings for the names of the given epochs.

    save_features_props: bool
        If true, it will save amplitudes in the sorting extractor.

    save_as_property: bool
        If True, the metric is saved as sorting property

    seed: int
        Random seed for reproducibility

    Returns
    ----------
    threshold sorting extractor
    """

    recording_params_dict = MetricData.recording_params_dict.copy()

    pca_scores_params_dict = MetricData.pca_scores_params_dict.copy()

    metric_scope_params_dict = MetricData.metric_scope_params_dict.copy()

    if metric_scope_params_dict["unit_ids"] is None:
        metric_scope_params_dict["unit_ids"] = sorting.get_unit_ids()

    metric_data = MetricData(
        sorting=sorting,
        recording=recording,
        apply_filter=recording_params_dict["apply_filter"],
        freq_min=recording_params_dict["freq_min"],
        freq_max=recording_params_dict["freq_max"],
        unit_ids=metric_scope_params_dict["unit_ids"],
        epoch_tuples=metric_scope_params_dict["epoch_tuples"],
        epoch_names=metric_scope_params_dict["epoch_names"],
    )

    metric_data.compute_pca_scores(
        n_comp=pca_scores_params_dict["n_comp"],
        ms_before=pca_scores_params_dict["ms_before"],
        ms_after=pca_scores_params_dict["ms_after"],
        dtype=pca_scores_params_dict["dtype"],
        max_spikes_per_unit=pca_scores_params_dict["max_spikes_per_unit"],
        max_spikes_for_pca=pca_scores_params_dict["max_spikes_for_pca"],
        save_features_props=save_features_props,
        seed=seed,
    )

    silhouette_score = SilhouetteScore(metric_data=metric_data)
    threshold_sorting = silhouette_score.threshold_metric(
        threshold, threshold_sign, epoch)

    return threshold_sorting
