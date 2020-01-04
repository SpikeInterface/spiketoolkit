import spiketoolkit as st
from .metric_data import MetricData
from .amplitude_cutoff import AmplitudeCutoff

def compute_amplitude_cutoffs(
    sorting,
    recording,
    recording_params_dict=MetricData.recording_params_dict,
    amplitude_params_dict=MetricData.amplitude_params_dict,
    metric_scope_params_dict=MetricData.metric_scope_params_dict,
    save_features_props=False,
    save_as_property = True,
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
    save_as_property: bool
        If True, the metric is saved as sorting property
    seed: int
        Random seed for reproducibility
    Returns
    ----------
    amplitude_cutoffs_epochs: list of lists
        The amplitude cutoffs of the sorted units in the given epochs.
    """
    rp_dict = dict(MetricData.recording_params_dict.copy(), **recording_params_dict)
    ap_dict = dict(MetricData.amplitude_params_dict.copy(), **amplitude_params_dict)
    ms_dict = dict(MetricData.metric_scope_params_dict.copy(), **metric_scope_params_dict)
    if ms_dict['unit_ids'] is None:
        ms_dict['unit_ids'] = sorting.get_unit_ids()

    mc = MetricData(sorting=sorting, recording=recording, apply_filter=rp_dict['apply_filter'],
                    freq_min=rp_dict['freq_min'], freq_max=rp_dict['freq_max'], unit_ids=ms_dict['unit_ids'], 
                    epoch_tuples=ms_dict['epoch_tuples'], epoch_names=ms_dict['epoch_names'])
    mc.compute_amplitudes(amp_method=ap_dict['amp_method'], amp_peak=ap_dict['amp_peak'],
                          amp_frames_before=ap_dict['amp_frames_before'], amp_frames_after=ap_dict['amp_frames_after'],
                          save_features_props=save_features_props, seed=seed)
    ac = AmplitudeCutoff(metric_data=mc)
    amplitude_cutoffs_epochs = ac.compute_metric()

    if save_as_property:
        if ms_dict['epoch_tuples'] is None:
            amplitude_cutoffs = amplitude_cutoffs_epochs[0]
            for i_u, u in enumerate(ms_dict['unit_ids']):
                sorting.set_unit_property(u, "amplitude_cutoff", amplitude_cutoffs[i_u])
        else:
            raise NotImplementedError(
                "Quality metrics cannot be saved as properties if 'epochs_tuples' are given."
            )
    return amplitude_cutoffs_epochs

