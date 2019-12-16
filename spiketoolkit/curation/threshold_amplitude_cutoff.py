from .thresholdcurator import ThresholdCurator
import spiketoolkit as st

cgps = {
    "threshold": {
        "type": "float",
        "title": "The threshold for the given metric.",
    },
    "threshold_sign": {
        "type": "str",
        "title": ("If 'less', will threshold any metric less than the"
                  " given threshold. If 'greater', will threshold any"
                  " metric greater than the given threshold."),
    },
    "amp_method": {
        "type": "str",
        "value": "absolute",
        "default": "absolute",
        "title": ("If 'absolute' (default), amplitudes are absolute amplitudes in uV are"
                  " returned. If 'relative', amplitudes returned as ratios between waveform"
                  " amplitudes and template amplitudes."),
    },
    "amp_peak": {
        "type": "str",
        "value": "both",
        "default": "both",
        "title": ("If maximum channel has to be found among negative peaks ('neg'),"
                  "positive ('pos') or both ('both' - default)"),
    },
    "amp_frames_before": {
        "type": "int",
        "value": 3,
        "default": 3,
        "title": "Frames before peak to compute amplitude.",
    },
    "amp_frames_after": {
        "type": "int",
        "value": 3,
        "default": 3,
        "title": "Frames after peak to compute amplitude.",
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


class ThresholdAmplitudeCutoff(ThresholdCurator):
    curator_name = "ThresholdAmplitudeCutoff"
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
        amp_method=cgps["amp_method"]["default"],
        amp_peak=cgps["amp_peak"]["default"],
        amp_frames_before=cgps["amp_frames_before"]["default"],
        amp_frames_after=cgps["amp_frames_after"]["default"],
        apply_filter=cgps["apply_filter"]["default"],
        freq_min=cgps["freq_min"]["default"],
        freq_max=cgps["freq_max"]["default"],
        save_features_props=cgps["save_features_props"]["default"],
        save_as_property=cgps["save_as_property"]["default"],
        metric_calculator=None,
        seed=cgps["seed"]["default"],
    ):

        metric_name = "amplitude_cutoff"

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

        if metric_name not in self._metric_calculator.get_metrics_dict().keys():
            self._metric_calculator.compute_amplitudes(
                recording=recording,
                amp_method=amp_method,
                amp_peak=amp_peak,
                amp_frames_before=amp_frames_before,
                amp_frames_after=amp_frames_after,
                apply_filter=apply_filter,
                freq_min=freq_min,
                freq_max=freq_max,
                save_features_props=save_features_props,
                seed=seed
            )
            self._metric_calculator.compute_amplitude_cutoffs()

        amplitude_cutoff_epochs = self._metric_calculator.get_metrics_dict()[metric_name][0]
        ThresholdCurator.__init__(self, sorting=sorting, metrics_epoch=amplitude_cutoff_epochs)

        self.threshold_sorting(threshold=threshold, threshold_sign=threshold_sign)


def threshold_amplitude_cutoff(
    sorting,
    recording,
    threshold,
    threshold_sign,
    amp_method=cgps["amp_method"]["default"],
    amp_peak=cgps["amp_peak"]["default"],
    amp_frames_before=cgps["amp_frames_before"]["default"],
    amp_frames_after=cgps["amp_frames_after"]["default"],
    apply_filter=cgps["apply_filter"]["default"],
    freq_min=cgps["freq_min"]["default"],
    freq_max=cgps["freq_max"]["default"],
    save_features_props=cgps["save_features_props"]["default"],
    save_as_property=cgps["save_as_property"]["default"],
    metric_calculator=None,
    seed=cgps["seed"]["default"],
):
    """
    Computes and returns the amplitude cutoffs for the sorted dataset.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated.
    recording: RecordingExtractor
        The given recording extractor from which to extract amplitudes
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
        If true, it will save amplitudes in the sorting extractor.
    unit_ids: list
        List of unit ids to compute metric for. If not specified, all units are used
    epoch_tuples: list
        A list of tuples with a start and end time for each epoch
    epoch_names: list
        A list of strings for the names of the given epochs.
    seed: int
        Random seed for reproducibility
    save_as_property: bool
        If True, the metric is saved as sorting property
    metric_calculator: MetricCalculator
        A metric calculator can be passed in with cached metrics.
    seed: int
        Random seed for reproducibility

    Returns
    ----------
    thresholded_sorting: amplitude_cutoff
        The thresholded sorting extractor
    """

    return ThresholdAmplitudeCutoff(
        sorting=sorting,
        recording=recording,
        threshold=threshold,
        threshold_sign=threshold_sign,
        amp_method=amp_method,
        amp_peak=amp_peak,
        amp_frames_before=amp_frames_before,
        amp_frames_after=amp_frames_after,
        apply_filter=apply_filter,
        freq_min=freq_min,
        freq_max=freq_max,
        save_features_props=save_features_props,
        save_as_property=save_as_property,
        metric_calculator=None,
        seed=seed,
    )
