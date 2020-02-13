from .quality_metric_classes.metric_data import MetricData
from .quality_metric_classes.amplitude_cutoff import AmplitudeCutoff
from .quality_metric_classes.silhouette_score import SilhouetteScore
from .quality_metric_classes.num_spikes import NumSpikes
from .quality_metric_classes.firing_rate import FiringRate
from .quality_metric_classes.d_prime import DPrime
from .quality_metric_classes.l_ratio import LRatio
from .quality_metric_classes.presence_ratio import PresenceRatio
from .quality_metric_classes.isi_violation import ISIViolation
from .quality_metric_classes.snr import SNR
from .quality_metric_classes.isolation_distance import IsolationDistance
from .quality_metric_classes.nearest_neighbor import NearestNeighbor
from .quality_metric_classes.drift_metric import DriftMetric   
from .parameter_dictionaries import get_recording_params, get_amplitude_params, get_pca_scores_params, get_metric_scope_params, update_param_dicts

def compute_num_spikes(
    sorting,
    sampling_frequency=None,
    metric_scope_params=get_metric_scope_params(),
    save_as_property=True
):
    """
    Computes and returns the num spikes for the sorted dataset.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated.
    sampling_frequency:
        The sampling frequency of the result. If None, will check to see if sampling frequency is in sorting extractor.
    metric_scope_params: dict
        This dictionary should contain any subset of the following parameters:
            unit_ids: list
                List of unit ids to compute metric for. If not specified, all units are used
            epoch_tuples: list
                A list of tuples with a start and end time for each epoch
            epoch_names: list
                A list of strings for the names of the given epochs.
    save_as_property: bool
        If True, the metric is saved as sorting property
    Returns
    ----------
    num_spikes_epochs: list of lists
        The num spikes of the sorted units in the given epochs.
    """
    ms_dict, = update_param_dicts(metric_scope_params=metric_scope_params)

    if ms_dict["unit_ids"] is None:
        ms_dict["unit_ids"] = sorting.get_unit_ids()

    md = MetricData(
        sorting=sorting,
        sampling_frequency=sampling_frequency,
        unit_ids=ms_dict["unit_ids"],
        epoch_tuples=ms_dict["epoch_tuples"],
        epoch_names=ms_dict["epoch_names"],
    )

    ns = NumSpikes(metric_data=md)
    num_spikes_epochs = ns.compute_metric(save_as_property)
    return num_spikes_epochs


def compute_firing_rates(
    sorting,
    sampling_frequency=None,
    metric_scope_params=get_metric_scope_params(),
    save_as_property=True
):
    """
    Computes and returns the firing rates for the sorted dataset.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated.
    sampling_frequency:
        The sampling frequency of the result. If None, will check to see if sampling frequency is in sorting extractor.
    metric_scope_params: dict
        This dictionary should contain any subset of the following parameters:
            unit_ids: list
                List of unit ids to compute metric for. If not specified, all units are used
            epoch_tuples: list
                A list of tuples with a start and end time for each epoch
            epoch_names: list
                A list of strings for the names of the given epochs.
    save_as_property: bool
        If True, the metric is saved as sorting property
    Returns
    ----------
    firing_rate_epochs: list of lists
        The firing rates of the sorted units in the given epochs.
    """
    ms_dict, = update_param_dicts(metric_scope_params=metric_scope_params)

    if ms_dict["unit_ids"] is None:
        ms_dict["unit_ids"] = sorting.get_unit_ids()

    md = MetricData(
        sorting=sorting,
        sampling_frequency=sampling_frequency,
        unit_ids=ms_dict["unit_ids"],
        epoch_tuples=ms_dict["epoch_tuples"],
        epoch_names=ms_dict["epoch_names"],
    )

    fr = FiringRate(metric_data=md)
    firing_rate_epochs = fr.compute_metric(save_as_property)
    return firing_rate_epochs


def compute_presence_ratios(
    sorting,
    sampling_frequency=None,
    metric_scope_params=get_metric_scope_params(),
    save_as_property=True
):
    """
    Computes and returns the presence ratios for the sorted dataset.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated.
    sampling_frequency:
        The sampling frequency of the result. If None, will check to see if sampling frequency is in sorting extractor.
    metric_scope_params: dict
        This dictionary should contain any subset of the following parameters:
            unit_ids: list
                List of unit ids to compute metric for. If not specified, all units are used
            epoch_tuples: list
                A list of tuples with a start and end time for each epoch
            epoch_names: list
                A list of strings for the names of the given epochs.
    save_as_property: bool
        If True, the metric is saved as sorting property
    Returns
    ----------
    presence_ratio_epochs: list of lists
        The presence ratios of the sorted units in the given epochs.
    """
    ms_dict, = update_param_dicts(metric_scope_params=metric_scope_params)

    if ms_dict["unit_ids"] is None:
        ms_dict["unit_ids"] = sorting.get_unit_ids()

    md = MetricData(
        sorting=sorting,
        sampling_frequency=sampling_frequency,
        unit_ids=ms_dict["unit_ids"],
        epoch_tuples=ms_dict["epoch_tuples"],
        epoch_names=ms_dict["epoch_names"],
    )

    pr = PresenceRatio(metric_data=md)
    presence_ratio_epochs = pr.compute_metric(save_as_property)
    return presence_ratio_epochs


def compute_isi_violations(
    sorting,
    isi_threshold=0.0015, 
    min_isi=0.000166,
    sampling_frequency=None,
    metric_scope_params=get_metric_scope_params(),
    save_as_property=True
):
    """
    Computes and returns the isi violations for the sorted dataset.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated.
    isi_threshold: float
        The isi threshold for calculating isi violations.
    min_isi: float
        The minimum expected isi value.
    sampling_frequency:
        The sampling frequency of the result. If None, will check to see if sampling frequency is in sorting extractor.
    metric_scope_params: dict
        This dictionary should contain any subset of the following parameters:
            unit_ids: list
                List of unit ids to compute metric for. If not specified, all units are used
            epoch_tuples: list
                A list of tuples with a start and end time for each epoch
            epoch_names: list
                A list of strings for the names of the given epochs.
    save_as_property: bool
        If True, the metric is saved as sorting property
    Returns
    ----------
    isi_violation_epochs: list of lists
        The isi violations of the sorted units in the given epochs.
    """
    ms_dict, = update_param_dicts(metric_scope_params=metric_scope_params)

    if ms_dict["unit_ids"] is None:
        ms_dict["unit_ids"] = sorting.get_unit_ids()

    md = MetricData(
        sorting=sorting,
        sampling_frequency=sampling_frequency,
        unit_ids=ms_dict["unit_ids"],
        epoch_tuples=ms_dict["epoch_tuples"],
        epoch_names=ms_dict["epoch_names"],
    )

    iv = ISIViolation(metric_data=md)
    isi_violation_epochs = iv.compute_metric(isi_threshold, min_isi, save_as_property)
    return isi_violation_epochs


def compute_amplitude_cutoffs(
    sorting,
    recording,
    recording_params=get_recording_params(),
    amplitude_params=get_amplitude_params(),
    metric_scope_params=get_metric_scope_params(),
    save_features_props=False,
    save_as_property=True,
    seed=None,
):
    """
    Computes and returns the amplitude cutoffs for the sorted dataset.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated.
    recording: RecordingExtractor
        The given recording extractor from which to extract amplitudes
    amplitude_params: dict
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
    recording_params: dict
        This dictionary should contain any subset of the following parameters:
            apply_filter: bool
                If True, recording is bandpass-filtered.
            freq_min: float
                High-pass frequency for optional filter (default 300 Hz).
            freq_max: float
                Low-pass frequency for optional filter (default 6000 Hz).
    metric_scope_params: dict
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
    save_as_property: bool
        If True, the metric is saved as sorting property
    Returns
    ----------
    amplitude_cutoffs_epochs: list of lists
        The amplitude cutoffs of the sorted units in the given epochs.
    """

    rp_dict, ap_dict, ms_dict = update_param_dicts(recording_params=recording_params, 
                                                   amplitude_params=amplitude_params, 
                                                   metric_scope_params=metric_scope_params)

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
    amplitude_cutoffs_epochs = ac.compute_metric(save_as_property)
    return amplitude_cutoffs_epochs

def compute_snrs(
    sorting,
    recording,
    snr_mode="mad",
    snr_noise_duration=10.0,
    max_spikes_per_unit_for_snr=1000,
    template_mode="median", 
    max_channel_peak="both", 
    recording_params=get_recording_params(),
    metric_scope_params=get_metric_scope_params(),
    recompute_info=True,
    save_features_props=True,
    save_as_property=True,
    seed=None,
):
    """
    Computes and returns the snrs in the sorted dataset.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated.

    recording: RecordingExtractor
        The given recording extractor from which to extract amplitudes

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

    pca_scores_params: dict
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

    recording_params: dict
        This dictionary should contain any subset of the following parameters:
            apply_filter: bool
                If True, recording is bandpass-filtered.
            freq_min: float
                High-pass frequency for optional filter (default 300 Hz).
            freq_max: float
                Low-pass frequency for optional filter (default 6000 Hz).

    metric_scope_params: dict
        This dictionary should contain any subset of the following parameters:
            unit_ids: list
                unit ids to compute metric for. If not specified, all units are used
            epoch_tuples: list
                A list of tuples with a start and end time for each epoch
            epoch_names: list
                A list of strings for the names of the given epochs.

    recompute_info: bool
            If True, waveforms are recomputed

    save_features_props: bool
        If true, it will save amplitudes in the sorting extractor.

    seed: int
        Random seed for reproducibility

    save_as_property: bool
        If True, the metric is saved as sorting property

    Returns
    ----------
    snr_epochs: list of lists
        The snrs of the sorted units in the given epochs.
    """
    rp_dict, ms_dict = update_param_dicts(recording_params=recording_params, 
                                          metric_scope_params=metric_scope_params)

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

    snr = SNR(metric_data=md)
    snr_epochs = snr.compute_metric(snr_mode, snr_noise_duration, max_spikes_per_unit_for_snr, 
                                    template_mode, max_channel_peak, save_features_props,
                                    recompute_info, seed, save_as_property)
    return snr_epochs

def compute_silhouette_scores(
    sorting,
    recording,
    max_spikes_for_silhouette=10000,
    pca_scores_params=get_pca_scores_params(),
    recording_params=get_recording_params(),
    metric_scope_params=get_metric_scope_params(),
    save_features_props=False,
    save_as_property=True,
    seed=None,
):
    """
    Computes and returns the silhouette scores in the sorted dataset.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated.

    recording: RecordingExtractor
        The given recording extractor from which to extract amplitudes

    max_spikes_for_silhouette: int
        Max spikes to be used for silhouette metric.

    pca_scores_params: dict
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

    recording_params: dict
        This dictionary should contain any subset of the following parameters:
            apply_filter: bool
                If True, recording is bandpass-filtered.
            freq_min: float
                High-pass frequency for optional filter (default 300 Hz).
            freq_max: float
                Low-pass frequency for optional filter (default 6000 Hz).

    metric_scope_params: dict
        This dictionary should contain any subset of the following parameters:
            unit_ids: list
                unit ids to compute metric for. If not specified, all units are used
            epoch_tuples: list
                A list of tuples with a start and end time for each epoch
            epoch_names: list
                A list of strings for the names of the given epochs.

    save_features_props: bool
        If true, it will save amplitudes in the sorting extractor.

    seed: int
        Random seed for reproducibility

    save_as_property: bool
        If True, the metric is saved as sorting property

    Returns
    ----------
    silhouette_score_epochs: list of lists
        The sihouette scores of the sorted units in the given epochs.
    """
    rp_dict, ps_dict, ms_dict = update_param_dicts(recording_params=recording_params, 
                                                   pca_scores_params=pca_scores_params, 
                                                   metric_scope_params=metric_scope_params)

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

    md.compute_pca_scores(
        n_comp=ps_dict["n_comp"],
        ms_before=ps_dict["ms_before"],
        ms_after=ps_dict["ms_after"],
        dtype=ps_dict["dtype"],
        max_spikes_per_unit=ps_dict["max_spikes_per_unit"],
        max_spikes_for_pca=ps_dict["max_spikes_for_pca"],
        save_features_props=save_features_props,
        seed=seed,
    )

    silhouette_score = SilhouetteScore(metric_data=md)
    silhouette_score_epochs = silhouette_score.compute_metric(max_spikes_for_silhouette, seed, save_as_property)
    return silhouette_score_epochs


def compute_d_primes(
    sorting,
    recording,
    num_channels_to_compare=13,
    max_spikes_per_cluster=500,
    pca_scores_params=get_pca_scores_params(),
    recording_params=get_recording_params(),
    metric_scope_params=get_metric_scope_params(),
    save_features_props=False,
    save_as_property=True,
    seed=None,
):
    """
    Computes and returns the d primes in the sorted dataset.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated.

    recording: RecordingExtractor
        The given recording extractor from which to extract amplitudes

    num_channels_to_compare: int
        The number of channels to be used for the PC extraction and comparison
        
    max_spikes_per_cluster: int
        Max spikes to be used from each unit

    pca_scores_params: dict
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

    recording_params: dict
        This dictionary should contain any subset of the following parameters:
            apply_filter: bool
                If True, recording is bandpass-filtered.
            freq_min: float
                High-pass frequency for optional filter (default 300 Hz).
            freq_max: float
                Low-pass frequency for optional filter (default 6000 Hz).

    metric_scope_params: dict
        This dictionary should contain any subset of the following parameters:
            unit_ids: list
                unit ids to compute metric for. If not specified, all units are used
            epoch_tuples: list
                A list of tuples with a start and end time for each epoch
            epoch_names: list
                A list of strings for the names of the given epochs.

    save_features_props: bool
        If true, it will save amplitudes in the sorting extractor.

    seed: int
        Random seed for reproducibility

    save_as_property: bool
        If True, the metric is saved as sorting property

    Returns
    ----------
    d_prime_epochs: list of lists
        The d primes of the sorted units in the given epochs.
    """
    rp_dict, ps_dict, ms_dict = update_param_dicts(recording_params=recording_params, 
                                                   pca_scores_params=pca_scores_params, 
                                                   metric_scope_params=metric_scope_params)

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

    md.compute_pca_scores(
        n_comp=ps_dict["n_comp"],
        ms_before=ps_dict["ms_before"],
        ms_after=ps_dict["ms_after"],
        dtype=ps_dict["dtype"],
        max_spikes_per_unit=ps_dict["max_spikes_per_unit"],
        max_spikes_for_pca=ps_dict["max_spikes_for_pca"],
        save_features_props=save_features_props,
        seed=seed,
    )

    d_prime = DPrime(metric_data=md)
    d_prime_epochs = d_prime.compute_metric(num_channels_to_compare, max_spikes_per_cluster, seed, save_as_property)
    return d_prime_epochs

def compute_l_ratios(
    sorting,
    recording,
    num_channels_to_compare=13,
    max_spikes_per_cluster=500,
    pca_scores_params=get_pca_scores_params(),
    recording_params=get_recording_params(),
    metric_scope_params=get_metric_scope_params(),
    save_features_props=False,
    save_as_property=True,
    seed=None,
):
    """
    Computes and returns the l ratios in the sorted dataset.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated.

    recording: RecordingExtractor
        The given recording extractor from which to extract amplitudes

    num_channels_to_compare: int
        The number of channels to be used for the PC extraction and comparison
        
    max_spikes_per_cluster: int
        Max spikes to be used from each unit

    pca_scores_params: dict
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

    recording_params: dict
        This dictionary should contain any subset of the following parameters:
            apply_filter: bool
                If True, recording is bandpass-filtered.
            freq_min: float
                High-pass frequency for optional filter (default 300 Hz).
            freq_max: float
                Low-pass frequency for optional filter (default 6000 Hz).

    metric_scope_params: dict
        This dictionary should contain any subset of the following parameters:
            unit_ids: list
                unit ids to compute metric for. If not specified, all units are used
            epoch_tuples: list
                A list of tuples with a start and end time for each epoch
            epoch_names: list
                A list of strings for the names of the given epochs.

    save_features_props: bool
        If true, it will save amplitudes in the sorting extractor.

    seed: int
        Random seed for reproducibility

    save_as_property: bool
        If True, the metric is saved as sorting property

    Returns
    ----------
    l_ratio_epochs: list of lists
        The l ratios of the sorted units in the given epochs.
    """
    rp_dict, ps_dict, ms_dict = update_param_dicts(recording_params=recording_params, 
                                                   pca_scores_params=pca_scores_params, 
                                                   metric_scope_params=metric_scope_params)

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

    md.compute_pca_scores(
        n_comp=ps_dict["n_comp"],
        ms_before=ps_dict["ms_before"],
        ms_after=ps_dict["ms_after"],
        dtype=ps_dict["dtype"],
        max_spikes_per_unit=ps_dict["max_spikes_per_unit"],
        max_spikes_for_pca=ps_dict["max_spikes_for_pca"],
        save_features_props=save_features_props,
        seed=seed,
    )

    l_ratio = LRatio(metric_data=md)
    l_ratio_epochs = l_ratio.compute_metric(num_channels_to_compare, max_spikes_per_cluster, seed, save_as_property)
    return l_ratio_epochs


def compute_isolation_distances(
    sorting,
    recording,
    num_channels_to_compare=13,
    max_spikes_per_cluster=500,
    pca_scores_params=get_pca_scores_params(),
    recording_params=get_recording_params(),
    metric_scope_params=get_metric_scope_params(),
    save_features_props=False,
    save_as_property=True,
    seed=None,
):
    """
    Computes and returns the isolation distances in the sorted dataset.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated.

    recording: RecordingExtractor
        The given recording extractor from which to extract amplitudes

    num_channels_to_compare: int
        The number of channels to be used for the PC extraction and comparison
        
    max_spikes_per_cluster: int
        Max spikes to be used from each unit

    pca_scores_params: dict
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

    recording_params: dict
        This dictionary should contain any subset of the following parameters:
            apply_filter: bool
                If True, recording is bandpass-filtered.
            freq_min: float
                High-pass frequency for optional filter (default 300 Hz).
            freq_max: float
                Low-pass frequency for optional filter (default 6000 Hz).

    metric_scope_params: dict
        This dictionary should contain any subset of the following parameters:
            unit_ids: list
                unit ids to compute metric for. If not specified, all units are used
            epoch_tuples: list
                A list of tuples with a start and end time for each epoch
            epoch_names: list
                A list of strings for the names of the given epochs.

    save_features_props: bool
        If true, it will save amplitudes in the sorting extractor.

    seed: int
        Random seed for reproducibility

    save_as_property: bool
        If True, the metric is saved as sorting property

    Returns
    ----------
    isolation_distance_epochs: list of lists
        The isolation distances of the sorted units in the given epochs.
    """
    rp_dict, ps_dict, ms_dict = update_param_dicts(recording_params=recording_params, 
                                                   pca_scores_params=pca_scores_params, 
                                                   metric_scope_params=metric_scope_params)

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

    md.compute_pca_scores(
        n_comp=ps_dict["n_comp"],
        ms_before=ps_dict["ms_before"],
        ms_after=ps_dict["ms_after"],
        dtype=ps_dict["dtype"],
        max_spikes_per_unit=ps_dict["max_spikes_per_unit"],
        max_spikes_for_pca=ps_dict["max_spikes_for_pca"],
        save_features_props=save_features_props,
        seed=seed,
    )

    isolation_distance = IsolationDistance(metric_data=md)
    isolation_distance_epochs = isolation_distance.compute_metric(num_channels_to_compare, max_spikes_per_cluster, seed, save_as_property)
    return isolation_distance_epochs

def compute_nn_metrics(
    sorting,
    recording,
    num_channels_to_compare=13,
    max_spikes_per_cluster=500,
    max_spikes_for_nn=10000,
    n_neighbors=4,
    pca_scores_params=get_pca_scores_params(),
    recording_params=get_recording_params(),
    metric_scope_params=get_metric_scope_params(),
    save_features_props=False,
    save_as_property=True,
    seed=None,
):
    """
    Computes and returns the nearest neighbor metrics in the sorted dataset.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated.

    recording: RecordingExtractor
        The given recording extractor from which to extract amplitudes

    num_channels_to_compare: int
        The number of channels to be used for the PC extraction and comparison
        
    max_spikes_per_cluster: int
        Max spikes to be used from each unit

    max_spikes_for_nn: int
        Max spikes to be used for nearest-neighbors calculation.
    
    n_neighbors: int
        Number of neighbors to compare.

    pca_scores_params: dict
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

    recording_params: dict
        This dictionary should contain any subset of the following parameters:
            apply_filter: bool
                If True, recording is bandpass-filtered.
            freq_min: float
                High-pass frequency for optional filter (default 300 Hz).
            freq_max: float
                Low-pass frequency for optional filter (default 6000 Hz).

    metric_scope_params: dict
        This dictionary should contain any subset of the following parameters:
            unit_ids: list
                unit ids to compute metric for. If not specified, all units are used
            epoch_tuples: list
                A list of tuples with a start and end time for each epoch
            epoch_names: list
                A list of strings for the names of the given epochs.

    save_features_props: bool
        If true, it will save amplitudes in the sorting extractor.

    seed: int
        Random seed for reproducibility

    save_as_property: bool
        If True, the metric is saved as sorting property

    Returns
    ----------
    nn_metrics_epochs: list of lists
        The nearest neighbor metrics of the sorted units in the given epochs.
    """
    rp_dict, ps_dict, ms_dict = update_param_dicts(recording_params=recording_params, 
                                                   pca_scores_params=pca_scores_params, 
                                                   metric_scope_params=metric_scope_params)

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

    md.compute_pca_scores(
        n_comp=ps_dict["n_comp"],
        ms_before=ps_dict["ms_before"],
        ms_after=ps_dict["ms_after"],
        dtype=ps_dict["dtype"],
        max_spikes_per_unit=ps_dict["max_spikes_per_unit"],
        max_spikes_for_pca=ps_dict["max_spikes_for_pca"],
        save_features_props=save_features_props,
        seed=seed,
    )

    nn = NearestNeighbor(metric_data=md)
    nn_metrics_epochs = nn.compute_metric(num_channels_to_compare, max_spikes_per_cluster, 
                                          max_spikes_for_nn, n_neighbors, seed, save_as_property)
    return nn_metrics_epochs

def compute_drift_metrics(
    sorting,
    recording,
    drift_metrics_interval_s=51,
    drift_metrics_min_spikes_per_interval=10,
    pca_scores_params=get_pca_scores_params(),
    recording_params=get_recording_params(),
    metric_scope_params=get_metric_scope_params(),
    save_features_props=False,
    save_as_property=True,
    seed=None,
):
    """
    Computes and returns the drift metrics in the sorted dataset.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated.

    recording: RecordingExtractor
        The given recording extractor from which to extract amplitudes

    drift_metrics_interval_s: float
        Time period for evaluating drift.

    drift_metrics_min_spikes_per_interval: int
        Minimum number of spikes for evaluating drift metrics per interval.

    pca_scores_params: dict
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

    recording_params: dict
        This dictionary should contain any subset of the following parameters:
            apply_filter: bool
                If True, recording is bandpass-filtered.
            freq_min: float
                High-pass frequency for optional filter (default 300 Hz).
            freq_max: float
                Low-pass frequency for optional filter (default 6000 Hz).

    metric_scope_params: dict
        This dictionary should contain any subset of the following parameters:
            unit_ids: list
                unit ids to compute metric for. If not specified, all units are used
            epoch_tuples: list
                A list of tuples with a start and end time for each epoch
            epoch_names: list
                A list of strings for the names of the given epochs.

    save_features_props: bool
        If true, it will save amplitudes in the sorting extractor.

    seed: int
        Random seed for reproducibility

    save_as_property: bool
        If True, the metric is saved as sorting property

    Returns
    ----------
    dm_metrics_epochs: list of lists
        The drift metrics of the sorted units in the given epochs.
    """
    rp_dict, ps_dict, ms_dict = update_param_dicts(recording_params=recording_params, 
                                                   pca_scores_params=pca_scores_params, 
                                                   metric_scope_params=metric_scope_params)

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

    md.compute_pca_scores(
        n_comp=ps_dict["n_comp"],
        ms_before=ps_dict["ms_before"],
        ms_after=ps_dict["ms_after"],
        dtype=ps_dict["dtype"],
        max_spikes_per_unit=ps_dict["max_spikes_per_unit"],
        max_spikes_for_pca=ps_dict["max_spikes_for_pca"],
        save_features_props=save_features_props,
        seed=seed,
    )

    dm = DriftMetric(metric_data=md)
    dm_metrics_epochs = dm.compute_metric(drift_metrics_interval_s, drift_metrics_min_spikes_per_interval, save_as_property)
    return dm_metrics_epochs

#     def compute_metrics(
#     sorting,
#     recording=None,
#     sampling_frequency=None,
#     isi_threshold=0.0015,
#     min_isi=0.000166,
#     snr_mode="mad",
#     snr_noise_duration=10.0,
#     max_spikes_per_unit_for_snr=1000,
#     drift_metrics_interval_s=51,
#     drift_metrics_min_spikes_per_interval=10,
#     max_spikes_for_silhouette=10000,
#     num_channels_to_compare=13,
#     max_spikes_per_cluster=500,
#     max_spikes_for_nn=10000,
#     n_neighbors=4,
#     max_spikes_per_unit=300,
#     recompute_info=True,
#     save_features_props=False,
#     metric_names=None,
#     unit_ids=None,
#     epoch_tuples=None,
#     epoch_names=None,
#     return_dataframe=False,
#     seed=None,
# ):
#     """
#     Computes and returns all specified metrics for the sorted dataset.

#     Parameters
#     ----------
#     sorting: SortingExtractor
#         The sorting result to be evaluated.
#     sampling_frequency:
#         The sampling frequency of the result. If None, will check to see if sampling frequency is in sorting extractor.
#     recording: RecordingExtractor
#         The given recording extractor from which to extract amplitudes. If None, certain metrics cannot be computed.
#     isi_threshold: float
#         The isi threshold for calculating isi violations.
#     min_isi: float
#         The minimum expected isi value.
#     snr_mode: str
#         Mode to compute noise SNR ('mad' | 'std' - default 'mad')
#     snr_noise_duration: float
#         Number of seconds to compute noise level from (default 10.0)
#     max_spikes_per_unit_for_snr: int
#         Maximum number of spikes to compute templates from (default 1000)
#     drift_metrics_interval_s: float
#         Time period for evaluating drift.
#     drift_metrics_min_spikes_per_interval: int
#         Minimum number of spikes for evaluating drift metrics per interval.
#     max_spikes_for_silhouette: int
#         Max spikes to be used for silhouette metric
#     num_channels_to_compare: int
#         The number of channels to be used for the PC extraction and comparison.
#     max_spikes_per_cluster: int
#         Max spikes to be used from each unit to compute metrics.
#     max_spikes_for_nn: int
#         Max spikes to be used for nearest-neighbors calculation.
#     n_neighbors: int
#         Number of neighbors to compare for  nearest-neighbors calculation.
#     max_spikes_per_unit: int
#         The maximum number of spikes to extract (default is np.inf)
#     recompute_info: bool
#         If True, will always re-extract waveforms.
#     save_features_props: bool
#         If True, save all features and properties in the sorting extractor.
#     metrics_names: list
#         The list of metric names to be computed. Available metrics are: 'firing_rate', 'num_spikes', 'isi_viol',
#             'presence_ratio', 'amplitude_cutoff', 'max_drift', 'cumulative_drift', 'silhouette_score',
#             'isolation_distance', 'l_ratio', 'd_prime', 'nn_hit_rate', 'nn_miss_rate', 'snr'. If None, all metrics are
#             computed.
#     return_dataframe: bool
#         If True, this function will return a dataframe of the metrics.
#     seed: int
#         Random seed for reproducibility

#     Returns
#     ----------
#     metrics_epochs : list of lists
#         List of metrics data. The list consists of lists of metric data for each given epoch.
#     OR
#     metrics_df: pandas.DataFrame
#         A pandas dataframe of the cached metrics
#     """
#     rp_dict, ps_dict, ms_dict = update_param_dicts(recording_params=recording_params, 
#                                                 pca_scores_params=pca_scores_params, 
#                                                 metric_scope_params=metric_scope_params)
#     metrics_epochs = []
#     all_metrics_list = [
#         "firing_rate",
#         "num_spikes",
#         "isi_viol",
#         "presence_ratio",
#         "amplitude_cutoff",
#         "max_drift",
#         "cumulative_drift",
#         "silhouette_score",
#         "isolation_distance",
#         "l_ratio",
#         "d_prime",
#         "nn_hit_rate",
#         "nn_miss_rate",
#         "snr",
#     ]

#     if metric_names is None:
#         metric_names = all_metrics_list
#     else:
#         bad_metrics = []
#         for m in metric_names:
#             if m not in all_metrics_list:
#                 bad_metrics.append(m)
#         if len(bad_metrics) > 0:
#             raise ValueError("Wrong metrics name: " + str(bad_metrics))

#     if recording is not None:
#         sampling_frequency = recording.get_sampling_frequency()

#     md = MetricData(
#         sorting=sorting,
#         recording=sampling_frequency,
#         apply_filter=rp_dict["apply_filter"],
#         freq_min=rp_dict["freq_min"],
#         freq_max=rp_dict["freq_max"],
#         unit_ids=ms_dict["unit_ids"],
#         epoch_tuples=ms_dict["epoch_tuples"],
#         epoch_names=ms_dict["epoch_names"],
#     )

#     if (
#         "max_drift" in metric_names
#         or "cumulative_drift" in metric_names
#         or "silhouette_score" in metric_names
#         or "isolation_distance" in metric_names
#         or "l_ratio" in metric_names
#         or "d_prime" in metric_names
#         or "nn_hit_rate" in metric_names
#         or "nn_miss_rate" in metric_names
#     ):
#         if recording is None:
#             raise ValueError(
#                 "The recording cannot be None when computing max_drift, cumulative_drift, "
#                 "silhouette_score isolation_distance, l_ratio, d_prime, nn_hit_rate, amplitude_cutoff, "
#                 "or nn_miss_rate."
#             )
#         else:
#             metric_calculator.compute_all_metric_data(
#                 recording=recording,
#                 n_comp=n_comp,
#                 ms_before=ms_before,
#                 ms_after=ms_after,
#                 dtype=dtype,
#                 max_spikes_per_unit=max_spikes_per_unit,
#                 amp_method=amp_method,
#                 amp_peak=amp_peak,
#                 amp_frames_before=amp_frames_before,
#                 amp_frames_after=amp_frames_after,
#                 recompute_info=recompute_info,
#                 max_spikes_for_pca=max_spikes_for_pca,
#                 apply_filter=apply_filter,
#                 freq_min=freq_min,
#                 freq_max=freq_max,
#                 save_features_props=save_features_props,
#                 seed=seed,
#             )
#     elif "amplitude_cutoff" in metric_names:
#         if recording is None:
#             raise ValueError(
#                 "The recording cannot be None when computing amplitude cutoffs."
#             )
#         else:
#             metric_calculator.compute_amplitudes(
#                 recording=recording,
#                 amp_method=amp_method,
#                 amp_peak=amp_peak,
#                 amp_frames_before=amp_frames_before,
#                 amp_frames_after=amp_frames_after,
#                 apply_filter=apply_filter,
#                 freq_min=freq_min,
#                 freq_max=freq_max,
#                 save_features_props=save_features_props,
#                 seed=seed,
#             )
#     elif "snr" in metric_names:
#         if recording is None:
#             raise ValueError("The recording cannot be None when computing snr.")
#         else:
#             metric_calculator.set_recording(
#                 recording,
#                 apply_filter=apply_filter,
#                 freq_min=freq_min,
#                 freq_max=freq_max,
#             )

#     if "num_spikes" in metric_names:
#         num_spikes_epochs = metric_calculator.compute_num_spikes()
#         metrics_epochs.append(num_spikes_epochs)

#     if "firing_rate" in metric_names:
#         firing_rates_epochs = metric_calculator.compute_firing_rates()
#         metrics_epochs.append(firing_rates_epochs)

#     if "presence_ratio" in metric_names:
#         presence_ratios_epochs = metric_calculator.compute_presence_ratios()
#         metrics_epochs.append(presence_ratios_epochs)

#     if "isi_viol" in metric_names:
#         isi_violations_epochs = metric_calculator.compute_isi_violations(
#             isi_threshold=isi_threshold, min_isi=min_isi
#         )
#         metrics_epochs.append(isi_violations_epochs)

#     if "amplitude_cutoff" in metric_names:
#         amplitude_cutoffs_epochs = metric_calculator.compute_amplitude_cutoffs()
#         metrics_epochs.append(amplitude_cutoffs_epochs)

#     if "snr" in metric_names:
#         snrs_epochs = metric_calculator.compute_snrs(
#             snr_mode=snr_mode,
#             snr_noise_duration=snr_noise_duration,
#             max_spikes_per_unit_for_snr=max_spikes_per_unit_for_snr,
#         )
#         metrics_epochs.append(snrs_epochs)

#     if "max_drift" in metric_names or "cumulative_drift" in metric_names:
#         (
#             max_drifts_epochs,
#             cumulative_drifts_epochs,
#         ) = metric_calculator.compute_drift_metrics(
#             drift_metrics_interval_s=drift_metrics_interval_s,
#             drift_metrics_min_spikes_per_interval=drift_metrics_min_spikes_per_interval,
#         )
#         if "max_drift" in metric_names:
#             metrics_epochs.append(max_drifts_epochs)
#         if "cumulative_drift" in metric_names:
#             metrics_epochs.append(cumulative_drifts_epochs)

#     if "silhouette_score" in metric_names:
#         silhouette_scores_epochs = metric_calculator.compute_silhouette_scores(
#             max_spikes_for_silhouette=max_spikes_for_silhouette, seed=seed
#         )
#         metrics_epochs.append(silhouette_scores_epochs)

#     if "isolation_distance" in metric_names:
#         isolation_distances_epochs = metric_calculator.compute_isolation_distances(
#             num_channels_to_compare=num_channels_to_compare,
#             max_spikes_per_cluster=max_spikes_per_cluster,
#             seed=seed,
#         )
#         metrics_epochs.append(isolation_distances_epochs)

#     if "l_ratio" in metric_names:
#         l_ratios_epochs = metric_calculator.compute_l_ratios(
#             num_channels_to_compare=num_channels_to_compare,
#             max_spikes_per_cluster=max_spikes_per_cluster,
#             seed=seed,
#         )
#         metrics_epochs.append(l_ratios_epochs)

#     if "d_prime" in metric_names:
#         d_primes_epochs = metric_calculator.compute_d_primes(
#             num_channels_to_compare=num_channels_to_compare,
#             max_spikes_per_cluster=max_spikes_per_cluster,
#             seed=seed,
#         )
#         metrics_epochs.append(d_primes_epochs)

#     if "nn_hit_rate" in metric_names or "nn_miss_rate" in metric_names:
#         (
#             nn_hit_rates_epochs,
#             nn_miss_rates_epochs,
#         ) = metric_calculator.compute_nn_metrics(
#             num_channels_to_compare=num_channels_to_compare,
#             max_spikes_per_cluster=max_spikes_per_cluster,
#             max_spikes_for_nn=max_spikes_for_nn,
#             n_neighbors=n_neighbors,
#             seed=seed,
#         )
#         if "nn_hit_rate" in metric_names:
#             metrics_epochs.append(nn_hit_rates_epochs)
#         if "nn_miss_rate" in metric_names:
#             metrics_epochs.append(nn_miss_rates_epochs)

#     if return_dataframe:
#         metrics_df = metric_calculator.get_metrics_df()
#         return metrics_df
#     else:
#         return metrics_epochs