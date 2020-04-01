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
from .quality_metric_classes.parameter_dictionaries import update_all_param_dicts_with_kwargs
from collections import OrderedDict
from copy import deepcopy


# All parameter values are stored in the class definitions


def compute_num_spikes(
        sorting,
        sampling_frequency=None,
        unit_ids=None,
        **kwargs
):
    """
    Computes and returns the num spikes for the sorted dataset.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated
    sampling_frequency: float
        The sampling frequency of the result. If None, will check to see if sampling frequency is in sorting extractor
    unit_ids: list
        List of unit ids to compute metric for. If not specified, all units are used
    **kwargs: keyword arguments
        Keyword arguments among the following:
            epoch_tuples: list
                A list of tuples with a start and end time for each epoch
            epoch_names: list
                A list of strings for the names of the given epochs
            save_property_or_features: bool
                If True, the metric is saved as sorting property
            verbose: bool
                If True, will be verbose in metric computation

    Returns
    ----------
    num_spikes_epochs: list of lists
        The num spikes of the sorted units in the given epochs
    """
    params_dict = update_all_param_dicts_with_kwargs(kwargs)

    if unit_ids is None:
        unit_ids = sorting.get_unit_ids()

    md = MetricData(sorting=sorting, sampling_frequency=sampling_frequency, recording=None,
                    apply_filter=False, freq_min=params_dict["freq_min"],
                    freq_max=params_dict["freq_max"], unit_ids=unit_ids, epoch_tuples=params_dict["epoch_tuples"],
                    epoch_names=params_dict["epoch_names"], verbose=params_dict['verbose'])

    ns = NumSpikes(metric_data=md)
    num_spikes_epochs = ns.compute_metric(**kwargs)
    return num_spikes_epochs


def compute_firing_rates(
        sorting,
        sampling_frequency=None,
        unit_ids=None,
        **kwargs
):
    """
    Computes and returns the firing rates for the sorted dataset.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated
    sampling_frequency: float
        The sampling frequency of the result. If None, will check to see if sampling frequency is in sorting extractor
    unit_ids: list
        List of unit ids to compute metric for. If not specified, all units are used
    **kwargs: keyword arguments
        Keyword arguments among the following:
            epoch_tuples: list
                A list of tuples with a start and end time for each epoch
            epoch_names: list
                A list of strings for the names of the given epochs
            save_property_or_features: bool
                If True, the metric is saved as sorting property
            verbose: bool
                If True, will be verbose in metric computation

    Returns
    ----------
    firing_rate_epochs: list of lists
        The firing rates of the sorted units in the given epochs
    """
    params_dict = update_all_param_dicts_with_kwargs(kwargs)

    if unit_ids is None:
        unit_ids = sorting.get_unit_ids()

    md = MetricData(sorting=sorting, sampling_frequency=sampling_frequency, recording=None,
                    apply_filter=False, freq_min=params_dict["freq_min"],
                    freq_max=params_dict["freq_max"], unit_ids=unit_ids, epoch_tuples=params_dict["epoch_tuples"],
                    epoch_names=params_dict["epoch_names"], verbose=params_dict['verbose'])

    fr = FiringRate(metric_data=md)
    firing_rate_epochs = fr.compute_metric(**kwargs)
    return firing_rate_epochs


def compute_presence_ratios(
        sorting,
        sampling_frequency=None,
        unit_ids=None,
        **kwargs
):
    """
    Computes and returns the presence ratios for the sorted dataset.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated.
    sampling_frequency: float
        The sampling frequency of the result. If None, will check to see if sampling frequency is in sorting extractor
    unit_ids: list
        List of unit ids to compute metric for. If not specified, all units are used
    **kwargs: keyword arguments
        Keyword arguments among the following:
            epoch_tuples: list
                A list of tuples with a start and end time for each epoch
            epoch_names: list
                A list of strings for the names of the given epochs
            save_property_or_features: bool
                If True, the metric is saved as sorting property
            verbose: bool
                If True, will be verbose in metric computation

    Returns
    ----------
    presence_ratio_epochs: list of lists
        The presence ratios of the sorted units in the given epochs
    """
    params_dict = update_all_param_dicts_with_kwargs(kwargs)

    if unit_ids is None:
        unit_ids = sorting.get_unit_ids()

    md = MetricData(sorting=sorting, sampling_frequency=sampling_frequency, recording=None,
                    apply_filter=False, freq_min=params_dict["freq_min"],
                    freq_max=params_dict["freq_max"], unit_ids=unit_ids, epoch_tuples=params_dict["epoch_tuples"],
                    epoch_names=params_dict["epoch_names"], verbose=params_dict['verbose'])

    pr = PresenceRatio(metric_data=md)
    presence_ratio_epochs = pr.compute_metric(**kwargs)
    return presence_ratio_epochs


def compute_isi_violations(
        sorting,
        isi_threshold=ISIViolation.params['isi_threshold'],
        min_isi=ISIViolation.params['min_isi'],
        sampling_frequency=None,
        unit_ids=None,
        **kwargs

):
    """
    Computes and returns the isi violations for the sorted dataset.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated
    isi_threshold: float
        The isi threshold for calculating isi violations
    min_isi: float
        The minimum expected isi value
    sampling_frequency: float
        The sampling frequency of the result. If None, will check to see if sampling frequency is in sorting extractor
    unit_ids: list
        List of unit ids to compute metric for. If not specified, all units are used
    **kwargs: keyword arguments
        Keyword arguments among the following:
            epoch_tuples: list
                A list of tuples with a start and end time for each epoch
            epoch_names: list
                A list of strings for the names of the given epochs
            save_property_or_features: bool
                If True, the metric is saved as sorting property
            verbose: bool
                If True, will be verbose in metric computation

    Returns
    ----------
    isi_violation_epochs: list of lists
        The isi violations of the sorted units in the given epochs
    """
    params_dict = update_all_param_dicts_with_kwargs(kwargs)

    if unit_ids is None:
        unit_ids = sorting.get_unit_ids()

    md = MetricData(sorting=sorting, sampling_frequency=sampling_frequency, recording=None,
                    apply_filter=False, freq_min=params_dict["freq_min"],
                    freq_max=params_dict["freq_max"], unit_ids=unit_ids, epoch_tuples=params_dict["epoch_tuples"],
                    epoch_names=params_dict["epoch_names"], verbose=params_dict['verbose'])

    iv = ISIViolation(metric_data=md)
    isi_violation_epochs = iv.compute_metric(isi_threshold, min_isi, **kwargs)
    return isi_violation_epochs


def compute_amplitude_cutoffs(
        sorting,
        recording,
        unit_ids=None,
        **kwargs
):
    """
    Computes and returns the amplitude cutoffs for the sorted dataset.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated.
    recording: RecordingExtractor
        The given recording extractor from which to extract amplitudes
    unit_ids: list
        List of unit ids to compute metric for. If not specified, all units are used
    **kwargs: keyword arguments
        Keyword arguments among the following:
            apply_filter: bool
                If True, recording is bandpass-filtered.
            freq_min: float
                High-pass frequency for optional filter (default 300 Hz)
            freq_max: float
                Low-pass frequency for optional filter (default 6000 Hz)
            save_property_or_features: bool
                If true, it will save amplitudes in the sorting extractor
            recompute_info: bool
                    If True, waveforms are recomputed
            max_spikes_per_unit: int
                The maximum number of spikes to extract per unit
            method: str
                If 'absolute' (default), amplitudes are absolute amplitudes in uV are returned.
                If 'relative', amplitudes are returned as ratios between waveform amplitudes and template amplitudes.
            peak: str
                If maximum channel has to be found among negative peaks ('neg'), positive ('pos') or
                both ('both' - default)
            frames_before: int
                Frames before peak to compute amplitude
            frames_after: float
                Frames after peak to compute amplitude
            save_property_or_features: bool
                If True, the metric is saved as sorting property
            seed: int
                Random seed for reproducibility
            verbose: bool
                If True, will be verbose in metric computation
            epoch_tuples: list
                A list of tuples with a start and end time for each epoch
            epoch_names: list
                A list of strings for the names of the given epochs

    Returns
    ----------
    amplitude_cutoffs_epochs: list of lists
        The amplitude cutoffs of the sorted units in the given epochs
    """
    params_dict = update_all_param_dicts_with_kwargs(kwargs)

    if unit_ids is None:
        unit_ids = sorting.get_unit_ids()

    md = MetricData(sorting=sorting, sampling_frequency=recording.get_sampling_frequency(), recording=recording,
                    apply_filter=params_dict["apply_filter"], freq_min=params_dict["freq_min"],
                    freq_max=params_dict["freq_max"], unit_ids=unit_ids, epoch_tuples=params_dict["epoch_tuples"],
                    epoch_names=params_dict["epoch_names"], verbose=params_dict['verbose'])

    md.compute_amplitudes(**kwargs)
    ac = AmplitudeCutoff(metric_data=md)
    amplitude_cutoffs_epochs = ac.compute_metric(**kwargs)
    return amplitude_cutoffs_epochs


def compute_snrs(
        sorting,
        recording,
        snr_mode=SNR.params['snr_mode'],
        snr_noise_duration=SNR.params['snr_noise_duration'],
        max_spikes_per_unit_for_snr=SNR.params['max_spikes_per_unit_for_snr'],
        template_mode=SNR.params['template_mode'],
        max_channel_peak=SNR.params['max_channel_peak'],
        unit_ids=None,
        **kwargs
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
    unit_ids: list
        List of unit ids to compute metric for. If not specified, all units are used
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
            epoch_tuples: list
                A list of tuples with a start and end time for each epoch
            epoch_names: list
                A list of strings for the names of the given epochs

    Returns
    ----------
    snr_epochs: list of lists
        The snrs of the sorted units in the given epochs
    """
    params_dict = update_all_param_dicts_with_kwargs(kwargs)

    if unit_ids is None:
        unit_ids = sorting.get_unit_ids()

    md = MetricData(sorting=sorting, sampling_frequency=recording.get_sampling_frequency(), recording=recording,
                    apply_filter=params_dict["apply_filter"], freq_min=params_dict["freq_min"],
                    freq_max=params_dict["freq_max"], unit_ids=unit_ids, epoch_tuples=params_dict["epoch_tuples"],
                    epoch_names=params_dict["epoch_names"], verbose=params_dict['verbose'])

    snr = SNR(metric_data=md)
    snr_epochs = snr.compute_metric(snr_mode, snr_noise_duration, max_spikes_per_unit_for_snr,
                                    template_mode, max_channel_peak, **kwargs)
    return snr_epochs


def compute_silhouette_scores(
        sorting,
        recording,
        max_spikes_for_silhouette=SilhouetteScore.params['max_spikes_for_silhouette'],
        unit_ids=None,
        **kwargs
):
    """
    Computes and returns the silhouette scores in the sorted dataset.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated
    recording: RecordingExtractor
        The given recording extractor from which to extract amplitudes
    max_spikes_for_silhouette: int
        Max spikes to be used for silhouette metric
    unit_ids: list
        List of unit ids to compute metric for. If not specified, all units are used
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
            epoch_tuples: list
                A list of tuples with a start and end time for each epoch
            epoch_names: list
                A list of strings for the names of the given epochs

    Returns
    ----------
    silhouette_score_epochs: list of lists
        The sihouette scores of the sorted units in the given epochs
    """
    params_dict = update_all_param_dicts_with_kwargs(kwargs)

    if unit_ids is None:
        unit_ids = sorting.get_unit_ids()

    md = MetricData(sorting=sorting, sampling_frequency=recording.get_sampling_frequency(), recording=recording,
                    apply_filter=params_dict["apply_filter"], freq_min=params_dict["freq_min"],
                    freq_max=params_dict["freq_max"], unit_ids=unit_ids, epoch_tuples=params_dict["epoch_tuples"],
                    epoch_names=params_dict["epoch_names"], verbose=params_dict['verbose'])

    md.compute_pca_scores(**kwargs)

    silhouette_score = SilhouetteScore(metric_data=md)
    silhouette_score_epochs = silhouette_score.compute_metric(max_spikes_for_silhouette, **kwargs)
    return silhouette_score_epochs


def compute_d_primes(
        sorting,
        recording,
        num_channels_to_compare=DPrime.params['num_channels_to_compare'],
        max_spikes_per_cluster=DPrime.params['max_spikes_per_cluster'],
        unit_ids=None,
        **kwargs
):
    """
    Computes and returns the d primes in the sorted dataset.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated
    recording: RecordingExtractor
        The given recording extractor from which to extract amplitudes
    num_channels_to_compare: int
        The number of channels to be used for the PC extraction and comparison
    max_spikes_per_cluster: int
        Max spikes to be used from each unit
    unit_ids: list
        List of unit ids to compute metric for. If not specified, all units are used
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
            epoch_tuples: list
                A list of tuples with a start and end time for each epoch
            epoch_names: list
                A list of strings for the names of the given epochs

    Returns
    ----------
    d_prime_epochs: list of lists
        The d primes of the sorted units in the given epochs
    """
    params_dict = update_all_param_dicts_with_kwargs(kwargs)

    if unit_ids is None:
        unit_ids = sorting.get_unit_ids()

    md = MetricData(sorting=sorting, sampling_frequency=recording.get_sampling_frequency(), recording=recording,
                    apply_filter=params_dict["apply_filter"], freq_min=params_dict["freq_min"],
                    freq_max=params_dict["freq_max"], unit_ids=unit_ids, epoch_tuples=params_dict["epoch_tuples"],
                    epoch_names=params_dict["epoch_names"], verbose=params_dict['verbose'])

    md.compute_pca_scores(**kwargs)

    d_prime = DPrime(metric_data=md)
    d_prime_epochs = d_prime.compute_metric(num_channels_to_compare, max_spikes_per_cluster, **kwargs)
    return d_prime_epochs


def compute_l_ratios(
        sorting,
        recording,
        num_channels_to_compare=LRatio.params['num_channels_to_compare'],
        max_spikes_per_cluster=LRatio.params['max_spikes_per_cluster'],
        unit_ids=None,
        **kwargs
):
    """
    Computes and returns the l ratios in the sorted dataset.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated
    recording: RecordingExtractor
        The given recording extractor from which to extract amplitudes
    num_channels_to_compare: int
        The number of channels to be used for the PC extraction and comparison
    max_spikes_per_cluster: int
        Max spikes to be used from each unit
    unit_ids: list
        List of unit ids to compute metric for. If not specified, all units are used
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
            epoch_tuples: list
                A list of tuples with a start and end time for each epoch
            epoch_names: list
                A list of strings for the names of the given epochs

    Returns
    ----------
    l_ratio_epochs: list of lists
        The l ratios of the sorted units in the given epochs
    """
    params_dict = update_all_param_dicts_with_kwargs(kwargs)

    if unit_ids is None:
        unit_ids = sorting.get_unit_ids()

    md = MetricData(sorting=sorting, sampling_frequency=recording.get_sampling_frequency(), recording=recording,
                    apply_filter=params_dict["apply_filter"], freq_min=params_dict["freq_min"],
                    freq_max=params_dict["freq_max"], unit_ids=unit_ids, epoch_tuples=params_dict["epoch_tuples"],
                    epoch_names=params_dict["epoch_names"], verbose=params_dict['verbose'])

    md.compute_pca_scores(**kwargs)

    l_ratio = LRatio(metric_data=md)
    l_ratio_epochs = l_ratio.compute_metric(num_channels_to_compare, max_spikes_per_cluster, **kwargs)
    return l_ratio_epochs


def compute_isolation_distances(
        sorting,
        recording,
        num_channels_to_compare=IsolationDistance.params['num_channels_to_compare'],
        max_spikes_per_cluster=IsolationDistance.params['max_spikes_per_cluster'],
        unit_ids=None,
        **kwargs
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
    unit_ids: list
        List of unit ids to compute metric for. If not specified, all units are used
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
            epoch_tuples: list
                A list of tuples with a start and end time for each epoch
            epoch_names: list
                A list of strings for the names of the given epochs

    Returns
    ----------
    isolation_distance_epochs: list of lists
        The isolation distances of the sorted units in the given epochs
    """
    params_dict = update_all_param_dicts_with_kwargs(kwargs)

    if unit_ids is None:
        unit_ids = sorting.get_unit_ids()

    md = MetricData(sorting=sorting, sampling_frequency=recording.get_sampling_frequency(), recording=recording,
                    apply_filter=params_dict["apply_filter"], freq_min=params_dict["freq_min"],
                    freq_max=params_dict["freq_max"], unit_ids=unit_ids, epoch_tuples=params_dict["epoch_tuples"],
                    epoch_names=params_dict["epoch_names"], verbose=params_dict['verbose'])

    md.compute_pca_scores(**kwargs)

    isolation_distance = IsolationDistance(metric_data=md)
    isolation_distance_epochs = isolation_distance.compute_metric(num_channels_to_compare, max_spikes_per_cluster,
                                                                  **kwargs)
    return isolation_distance_epochs


def compute_nn_metrics(
        sorting,
        recording,
        num_channels_to_compare=NearestNeighbor.params['num_channels_to_compare'],
        max_spikes_per_cluster=NearestNeighbor.params['max_spikes_per_cluster'],
        max_spikes_for_nn=NearestNeighbor.params['max_spikes_for_nn'],
        n_neighbors=NearestNeighbor.params['n_neighbors'],
        unit_ids=None,
        **kwargs
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
        Max spikes to be used for nearest-neighbors calculation
    n_neighbors: int
        Number of neighbors to compare
    unit_ids: list
        List of unit ids to compute metric for. If not specified, all units are used
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
            epoch_tuples: list
                A list of tuples with a start and end time for each epoch
            epoch_names: list
                A list of strings for the names of the given epochs

    Returns
    ----------
    nn_metrics_epochs: list of lists
        The nearest neighbor metrics of the sorted units in the given epochs
    """
    params_dict = update_all_param_dicts_with_kwargs(kwargs)

    if unit_ids is None:
        unit_ids = sorting.get_unit_ids()

    md = MetricData(sorting=sorting, sampling_frequency=recording.get_sampling_frequency(), recording=recording,
                    apply_filter=params_dict["apply_filter"], freq_min=params_dict["freq_min"],
                    freq_max=params_dict["freq_max"], unit_ids=unit_ids, epoch_tuples=params_dict["epoch_tuples"],
                    epoch_names=params_dict["epoch_names"], verbose=params_dict['verbose'])

    md.compute_pca_scores(**kwargs)

    nn = NearestNeighbor(metric_data=md)
    nn_metrics_epochs = nn.compute_metric(num_channels_to_compare, max_spikes_per_cluster,
                                          max_spikes_for_nn, n_neighbors, **kwargs)
    return nn_metrics_epochs


def compute_drift_metrics(
        sorting,
        recording,
        drift_metrics_interval_s=DriftMetric.params['drift_metrics_interval_s'],
        drift_metrics_min_spikes_per_interval=DriftMetric.params['drift_metrics_min_spikes_per_interval'],
        unit_ids=None,
        **kwargs
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
    unit_ids: list
        List of unit ids to compute metric for. If not specified, all units are used
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
            epoch_tuples: list
                A list of tuples with a start and end time for each epoch
            epoch_names: list
                A list of strings for the names of the given epochs

    Returns
    ----------
    dm_metrics_epochs: list of lists
        The drift metrics of the sorted units in the given epochs
    """
    params_dict = update_all_param_dicts_with_kwargs(kwargs)

    if unit_ids is None:
        unit_ids = sorting.get_unit_ids()

    md = MetricData(sorting=sorting, sampling_frequency=recording.get_sampling_frequency(), recording=recording,
                    apply_filter=params_dict["apply_filter"], freq_min=params_dict["freq_min"],
                    freq_max=params_dict["freq_max"], unit_ids=unit_ids, epoch_tuples=params_dict["epoch_tuples"],
                    epoch_names=params_dict["epoch_names"], verbose=params_dict['verbose'])

    md.compute_pca_scores(**kwargs)

    dm = DriftMetric(metric_data=md)
    dm_metrics_epochs = dm.compute_metric(drift_metrics_interval_s, drift_metrics_min_spikes_per_interval, **kwargs)
    return dm_metrics_epochs


def compute_metrics(
        sorting,
        recording=None,
        metric_names=None,
        unit_ids=None,
        return_dict=False,
        isi_threshold=ISIViolation.params['isi_threshold'],
        min_isi=ISIViolation.params['min_isi'],
        snr_mode=SNR.params['snr_mode'],
        snr_noise_duration=SNR.params['snr_noise_duration'],
        max_spikes_per_unit_for_snr=SNR.params['max_spikes_per_unit_for_snr'],
        template_mode=SNR.params['template_mode'],
        max_channel_peak=SNR.params['max_channel_peak'],
        drift_metrics_interval_s=DriftMetric.params['drift_metrics_interval_s'],
        drift_metrics_min_spikes_per_interval=DriftMetric.params['drift_metrics_min_spikes_per_interval'],
        max_spikes_for_silhouette=SilhouetteScore.params['max_spikes_for_silhouette'],
        num_channels_to_compare=13,
        max_spikes_per_cluster=500,
        max_spikes_for_nn=NearestNeighbor.params['max_spikes_for_nn'],
        n_neighbors=NearestNeighbor.params['n_neighbors'],
        **kwargs
):
    """
    Computes and returns all specified metrics for the sorted dataset.
    
    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated.
    recording: RecordingExtractor
        The given recording extractor from which to extract amplitudes
    metric_names: list
        List of metric names to be computed
    unit_ids: list
        List of unit ids to compute metric for. If not specified, all units are used
    return_dict: bool
        If True, will return dict of metrics
    isi_threshold: float
        The isi threshold for calculating isi violations
    min_isi: float
        The minimum expected isi value
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
    drift_metrics_interval_s: float
        Time period for evaluating drift.
    drift_metrics_min_spikes_per_interval: int
        Minimum number of spikes for evaluating drift metrics per interval
    max_spikes_for_silhouette: int
        Max spikes to be used for silhouette metric
    num_channels_to_compare: int
        The number of channels to be used for the PC extraction and comparison
    max_spikes_per_cluster: int
        Max spikes to be used from each unit
    max_spikes_for_nn: int
        Max spikes to be used for nearest-neighbors calculation
    n_neighbors: int
        Number of neighbors to compare
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
            epoch_tuples: list
                A list of tuples with a start and end time for each epoch
            epoch_names: list
                A list of strings for the names of the given epochs

    Returns
    ----------
    metrics_epochs : list of lists
        List of metrics data. The list consists of lists of metric data for each given epoch
    OR
    metrics_dict: OrderedDict
        Dict of metrics data. The dict consists of lists of metric data for each given epoch for each metric
    
    """
    params_dict = update_all_param_dicts_with_kwargs(kwargs)

    metrics_epochs = []
    metrics_dict = OrderedDict()
    all_metrics_list = ["num_spikes", "firing_rate", "presence_ratio", "isi_viol", "amplitude_cutoff", "snr",
                        "max_drift", "cumulative_drift", "silhouette_score", "isolation_distance", "l_ratio",
                        "d_prime", "nn_hit_rate", "nn_miss_rate"]

    if metric_names is None:
        metric_names = all_metrics_list
    else:
        bad_metrics = []
        for m in metric_names:
            if m not in all_metrics_list:
                bad_metrics.append(m)
        if len(bad_metrics) > 0:
            raise ValueError("Wrong metrics name: " + str(bad_metrics))

    if unit_ids is None:
        unit_ids = sorting.get_unit_ids()

    md = MetricData(sorting=sorting, sampling_frequency=recording.get_sampling_frequency(), recording=recording,
                    apply_filter=params_dict["apply_filter"], freq_min=params_dict["freq_min"],
                    freq_max=params_dict["freq_max"], unit_ids=unit_ids, epoch_tuples=params_dict["epoch_tuples"],
                    epoch_names=params_dict["epoch_names"], verbose=params_dict['verbose'])

    if "max_drift" in metric_names or "cumulative_drift" in metric_names or "silhouette_score" in metric_names \
        or "isolation_distance" in metric_names or "l_ratio" in metric_names or "d_prime" in metric_names \
            or "nn_hit_rate" in metric_names or "nn_miss_rate" in metric_names:
        if recording is None:
            raise ValueError("The recording cannot be None when computing max_drift, cumulative_drift, "
                             "silhouette_score isolation_distance, l_ratio, d_prime, nn_hit_rate, amplitude_cutoff.")
        else:
            md.compute_pca_scores(**kwargs)

    if "amplitude_cutoff" in metric_names:
        if recording is None:
            raise ValueError("The recording cannot be None when computing amplitude cutoffs.")
        else:
            md.compute_amplitudes(**kwargs)
    if "snr" in metric_names:
        if recording is None:
            raise ValueError("The recording cannot be None when computing snr.")

    if "num_spikes" in metric_names:
        ns = NumSpikes(metric_data=md)
        num_spikes_epochs = ns.compute_metric(**kwargs)
        metrics_epochs.append(num_spikes_epochs)
        metrics_dict['num_spikes'] = num_spikes_epochs

    if "firing_rate" in metric_names:
        fr = FiringRate(metric_data=md)
        firing_rate_epochs = fr.compute_metric(**kwargs)
        metrics_epochs.append(firing_rate_epochs)
        metrics_dict['firing_rate'] = firing_rate_epochs

    if "presence_ratio" in metric_names:
        pr = PresenceRatio(metric_data=md)
        presence_ratio_epochs = pr.compute_metric(**kwargs)
        metrics_epochs.append(presence_ratio_epochs)
        metrics_dict['presence_ratio'] = presence_ratio_epochs

    if "isi_viol" in metric_names:
        iv = ISIViolation(metric_data=md)
        isi_violation_epochs = iv.compute_metric(isi_threshold, min_isi, **kwargs)
        metrics_epochs.append(isi_violation_epochs)
        metrics_dict['isi_viol'] = isi_violation_epochs

    if "amplitude_cutoff" in metric_names:
        ac = AmplitudeCutoff(metric_data=md)
        amplitude_cutoffs_epochs = ac.compute_metric(**kwargs)
        metrics_epochs.append(amplitude_cutoffs_epochs)
        metrics_dict['amplitude_cutoff'] = amplitude_cutoffs_epochs

    if "snr" in metric_names:
        snr = SNR(metric_data=md)
        snr_epochs = snr.compute_metric(snr_mode, snr_noise_duration, max_spikes_per_unit_for_snr,
                                        template_mode, max_channel_peak, **kwargs)
        metrics_epochs.append(snr_epochs)
        metrics_dict['snr'] = snr_epochs

    if "max_drift" in metric_names or "cumulative_drift" in metric_names:
        dm = DriftMetric(metric_data=md)
        dm_metrics_epochs = dm.compute_metric(drift_metrics_interval_s, drift_metrics_min_spikes_per_interval, **kwargs)
        max_drifts_epochs = []
        cumulative_drifts_epochs = []
        for dm_metric in dm_metrics_epochs:
            max_drifts_epochs.append(dm_metric[0])
            cumulative_drifts_epochs.append(dm_metric[1])
        if "max_drift" in metric_names:
            metrics_epochs.append(max_drifts_epochs)
            metrics_dict['max_drift'] = max_drifts_epochs
        if "cumulative_drift" in metric_names:
            metrics_epochs.append(cumulative_drifts_epochs)
            metrics_dict['cumulative_drift'] = cumulative_drifts_epochs

    if "silhouette_score" in metric_names:
        silhouette_score = SilhouetteScore(metric_data=md)
        silhouette_score_epochs = silhouette_score.compute_metric(max_spikes_for_silhouette, **kwargs)
        metrics_epochs.append(silhouette_score_epochs)
        metrics_dict['silhouette_score'] = silhouette_score_epochs

    if "isolation_distance" in metric_names:
        isolation_distance = IsolationDistance(metric_data=md)
        isolation_distance_epochs = isolation_distance.compute_metric(num_channels_to_compare, max_spikes_per_cluster,
                                                                      **kwargs)
        metrics_epochs.append(isolation_distance_epochs)
        metrics_dict['isolation_distance'] = isolation_distance_epochs

    if "l_ratio" in metric_names:
        l_ratio = LRatio(metric_data=md)
        l_ratio_epochs = l_ratio.compute_metric(num_channels_to_compare, max_spikes_per_cluster, **kwargs)
        metrics_epochs.append(l_ratio_epochs)
        metrics_dict['l_ratio'] = l_ratio_epochs

    if "d_prime" in metric_names:
        d_prime = DPrime(metric_data=md)
        d_prime_epochs = d_prime.compute_metric(num_channels_to_compare, max_spikes_per_cluster, **kwargs)
        metrics_epochs.append(d_prime_epochs)
        metrics_dict['d_prime'] = d_prime_epochs

    if "nn_hit_rate" in metric_names or "nn_miss_rate" in metric_names:
        nn = NearestNeighbor(metric_data=md)
        nn_metrics_epochs = nn.compute_metric(num_channels_to_compare, max_spikes_per_cluster,
                                              max_spikes_for_nn, n_neighbors, **kwargs)
        nn_hit_rates_epochs = []
        nn_miss_rates_epochs = []
        for nn_metric in nn_metrics_epochs:
            nn_hit_rates_epochs.append(nn_metric[0])
            nn_miss_rates_epochs.append(nn_metric[1])
        if "nn_hit_rate" in metric_names:
            metrics_epochs.append(nn_hit_rates_epochs)
            metrics_dict['nn_hit_rate'] = nn_hit_rates_epochs
        if "nn_miss_rate" in metric_names:
            metrics_epochs.append(nn_miss_rates_epochs)
            metrics_dict['nn_miss_rate'] = nn_miss_rates_epochs

    if return_dict:
        return metrics_dict
    else:
        return metrics_epochs
