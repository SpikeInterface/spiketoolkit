from .thresholdcurator import ThresholdCurator
import spiketoolkit as st

cgps = {
    "threshold": {
        "type": "float",
        "title": "The threshold for the given metric.",
    },
    "threshold_sign": {
        "type": "str",
        "title": (
            "If 'less', will threshold any metric less than the"
            " given threshold. If 'greater', will threshold any"
            " metric greater than the given threshold."
        ),
    },
    "num_channels_to_compare": {
        "type": "int",
        "value": 13,
        "default": 13,
        "title": ("The number of channels to be used for the PC extraction"
                  " and comparison"),
    },
    "max_spikes_per_cluster": {
        "type": "int",
        "value": 500,
        "default": 500,
        "title": "Max spikes to be used from each unit",
    },
    "n_comp": {
        "type": "int",
        "value": 3,
        "default": 3,
        "title": "n_compFeatures in template-gui format.",
    },
    "ms_before": {
        "type": "float",
        "value": 1.0,
        "default": 1.0,
        "title": ("Time period in ms to cut waveforms before the spike"
                  " events."),
    },
    "ms_after": {
        "type": "float",
        "value": 2.0,
        "default": 2.0,
        "title": ("Time period in ms to cut waveforms after the spike"
                  " events."),
    },
    "dtype": {
        "type": "dtype",
        "value": None,
        "default": None,
        "title": ("The numpy dtype of the waveforms."),
    },
    "max_spikes_per_unit": {
        "type": "int",
        "value": 300,
        "default": 300,
        "title": ("The maximum number of spikes to extract (default is"
                  " np.inf)."),
    },
    "recompute_info": {
        "type": "bool",
        "value": True,
        "default": True,
        "title": ("If True, will always re-extract waveforms."),
    },
    "max_spikes_for_pca": {
        "type": "int",
        "value": 100_000,
        "default": 100_000,
        "title": ("The maximum number of spikes to use to compute PCA"
                  " (default is np.inf)"),
    },
    "apply_filter": {
        "type": "bool",
        "value": True,
        "default": True,
        "title": ("If True, recording is bandpass-filtered."),
    },
    "freq_min": {
        "type": "float",
        "value": 300.0,
        "default": 300.0,
        "title": ("High-pass frequency for optional filter (default"
                  " 300 Hz)."),
    },
    "freq_max": {
        "type": "float",
        "value": 6000.0,
        "default": 6000.0,
        "title": ("Low-pass frequency for optional filter (default"
                  " 6000 Hz)."),
    },
    "save_features_props": {
        "type": "bool",
        "value": False,
        "default": False,
        "title": ("If True, save all features and properties in the"
                  " sorting extractor."),
    },
    "save_as_property": {
        "type": "bool",
        "value": True,
        "default": True,
        "title": ("If True, the metric is saved as sorting property."),
    },
    "seed": {
        "type": "int",
        "value": 0,
        "default": 0,
        "title": "Random seed for reproducibility.",
    },
}


class ThresholdLRatios(ThresholdCurator):
    curator_name = "ThresholdLRatios"
    installed = True
    installation_mesg = ""  # err

    # To avoid duplication, build curator_gui_params from cgps
    curator_gui_params = [dict(value, name=key) for key, value in cgps.items()]

    def __init__(
        self,
        sorting,
        recording,
        threshold,
        threshold_sign,
        num_channels_to_compare=cgps["num_channels_to_compare"]["default"],
        max_spikes_per_cluster=cgps["max_spikes_per_cluster"]["default"],
        n_comp=cgps["n_comp"]["default"],
        ms_before=cgps["ms_before"]["default"],
        ms_after=cgps["ms_after"]["default"],
        dtype=cgps["dtype"]["default"],
        max_spikes_per_unit=cgps["max_spikes_per_unit"]["default"],
        recompute_info=cgps["recompute_info"]["default"],
        max_spikes_for_pca=cgps["max_spikes_for_pca"]["default"],
        apply_filter=cgps["apply_filter"]["default"],
        freq_min=cgps["freq_min"]["default"],
        freq_max=cgps["freq_max"]["default"],
        save_features_props=cgps["save_features_props"]["default"],
        save_as_property=cgps["save_as_property"]["default"],
        metric_calculator=None,
        seed=cgps["seed"]["default"],
    ):

        metric_name = "l_ratio"

        if metric_calculator is None:
            self._metric_calculator = st.validation.MetricCalculator(
                sorting,
                sampling_frequency=recording.get_sampling_frequency(),
                unit_ids=None,
                epoch_tuples=None,
                epoch_names=None,
            )
        else:
            self._metric_calculator = metric_calculator

        if (
            metric_name not in self._metric_calculator.get_metrics_dict().keys()):      # noqa: E501
            self._metric_calculator.compute_pca_scores(
                recording=recording,
                n_comp=n_comp,
                ms_before=ms_before,
                ms_after=ms_after,
                dtype=dtype,
                max_spikes_per_unit=max_spikes_per_unit,
                recompute_info=recompute_info,
                max_spikes_for_pca=max_spikes_for_pca,
                apply_filter=apply_filter,
                freq_min=freq_min,
                freq_max=freq_max,
                save_features_props=save_features_props,
                seed=seed,
            )
            self._metric_calculator.compute_l_ratios(
                num_channels_to_compare=cgps["num_channels_to_compare"]["default"],     # noqa: E501
                max_spikes_per_cluster=cgps["max_spikes_per_cluster"]["default"],       # noqa: E501
                seed=cgps["seed"]["default"],
            )

        l_ratio_epochs = self._metric_calculator.get_metrics_dict()[metric_name][0]    # noqa: E501

        ThresholdCurator.__init__(self, sorting=sorting, metrics_epoch=l_ratio_epochs)  # noqa: E501

        self.threshold_sorting(threshold=threshold, threshold_sign=threshold_sign)      # noqa: E501


def threshold_l_ratios(
    sorting,
    recording,
    threshold,
    threshold_sign,
    num_channels_to_compare=cgps["num_channels_to_compare"]["default"],
    max_spikes_per_cluster=cgps["max_spikes_per_cluster"]["default"],
    n_comp=cgps["n_comp"]["default"],
    ms_before=cgps["ms_before"]["default"],
    ms_after=cgps["ms_after"]["default"],
    dtype=cgps["dtype"]["default"],
    max_spikes_per_unit=cgps["max_spikes_per_unit"]["default"],
    recompute_info=cgps["recompute_info"]["default"],
    max_spikes_for_pca=cgps["max_spikes_for_pca"]["default"],
    apply_filter=cgps["apply_filter"]["default"],
    freq_min=cgps["freq_min"]["default"],
    freq_max=cgps["freq_max"]["default"],
    save_features_props=cgps["save_features_props"]["default"],
    save_as_property=cgps["save_as_property"]["default"],
    metric_calculator=None,
    seed=cgps["seed"]["default"],
):
    """
    Excludes units based on the lda-based metric, d-prime.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated.
    recording: RecordingExtractor
        The given recording extractor from which to extract amplitudes.
    num_channels_to_compare: int
        The number of channels to be used for the PC extraction and comparison
    max_spikes_per_cluster: int
        Max spikes to be used from each unit
    n_comp: int
        n_compFeatures in template-gui format
    ms_before: float
        Time period in ms to cut waveforms before the spike events
    ms_after: float
        Time period in ms to cut waveforms after the spike events
    dtype: dtype
        The numpy dtype of the waveforms
    max_spikes_per_unit: int
        The maximum number of spikes to extract (default is np.inf)
    recompute_info: bool
        If True, will always re-extract waveforms.
    max_spikes_for_pca: int
        The maximum number of spikes to use to compute PCA (default is np.inf)
    apply_filter: bool
        If True, recording is bandpass-filtered.
    freq_min: float
        High-pass frequency for optional filter (default 300 Hz).
    freq_max: float
        Low-pass frequency for optional filter (default 6000 Hz).
    save_features_props: bool
        If True, save all features and properties in the sorting extractor.
    unit_ids: list
        List of unit ids to compute metric for. If not specified, all units are used
    epoch_tuples: list
        A list of tuples with a start and end time for each epoch.
    epoch_names: list
        A list of strings for the names of the given epochs.
    save_as_property: bool
        If True, the metric is saved as sorting property
    metric_calculator: MetricCalculator
        A metric calculator can be passed in with cached metrics.
    seed: int
        Random seed for reproducibility

    Returns
    -------
    thresholded_sorting: ThresholdLRatios
        The thresholded sorting extractor

    """
    return ThresholdLRatios(
        sorting=sorting,
        recording=recording,
        threshold=threshold,
        threshold_sign=threshold_sign,
        num_channels_to_compare=num_channels_to_compare,
        max_spikes_per_cluster=max_spikes_per_cluster,
        n_comp=n_comp,
        ms_before=ms_before,
        ms_after=ms_after,
        dtype=dtype,
        max_spikes_per_unit=max_spikes_per_unit,
        recompute_info=recompute_info,
        max_spikes_for_pca=max_spikes_for_pca,
        apply_filter=apply_filter,
        freq_min=freq_min,
        freq_max=freq_max,
        save_features_props=save_features_props,
        save_as_property=save_as_property,
        metric_calculator=None,
        seed=seed,
    )
