import numpy as np
import spiketoolkit as st
import spikemetrics.metrics as metrics
from spikemetrics.utils import Epoch 

#Quality metrics from Allen Institute can't do individual units so we calculate metrics all units to get 
#the metrics for only a few units.

def compute_firing_rates(sorting, sampling_frequency, unit_ids=None, epoch_tuple=None):
    '''
    Computes and returns the spike times in seconds and also returns 
    the cluster_ids needed for quality metrics (all within given epoch)
    
    Parameters
    ----------
    sorting: SortingExtractor
        The sorting extractor.
    sampling_frequency: float
        The sampling frequency of the recording.
    unit_ids: list
        List of unit ids to compute metric for. If not specified, all units are used
    epoch_tuple: int tuple
        A tuple with a start and end frame for the epoch in question.

    Returns
    ----------
    firing_rates: np.array
        The firing rates of the sorted units in the given epoch.
    '''
    if unit_ids is None or unit_ids == []:
        unit_ids = sorting.get_unit_ids()
        unit_indices = np.arange(len(unit_ids))
    else:
        unit_indices = _get_unit_indices(sorting, unit_ids)

    spike_times, spike_clusters  = st.validation.validation_tools.get_firing_times_ids(sorting, sampling_frequency)
    total_units = len(sorting.get_unit_ids()) 
    in_epoch = _get_in_epoch(spike_times, epoch_tuple, sampling_frequency)
    firing_rates_all, _ = metrics.calculate_firing_rate_and_spikes(spike_times[in_epoch], spike_clusters[in_epoch], total_units)
    firing_rates_list = []
    for i in unit_indices:
        firing_rates_list.append(firing_rates_all[i])
    firing_rates = np.asarray(firing_rates_list)

    return firing_rates

def compute_num_spikes(sorting, sampling_frequency, unit_ids=None, epoch_tuple=None):
    '''
    Computes and returns the spike times in seconds and also returns 
    the cluster_ids needed for quality metrics (all within given epoch)
    
    Parameters
    ----------
    sorting: SortingExtractor
        The sorting extractor.
    sampling_frequency: float
        The sampling frequency of the recording.
    unit_ids: list
        List of unit ids to compute metric for. If not specified, all units are used
    epoch_tuple: int tuple
        A tuple with a start and end frame for the epoch in question.

    Returns
    ----------
    num_spikes: np.array
        The spike counts of the sorted units in the given epoch.
    '''
    if unit_ids is None or unit_ids == []:
        unit_ids = sorting.get_unit_ids()
        unit_indices = np.arange(len(unit_ids))
    else:
        unit_indices = _get_unit_indices(sorting, unit_ids)

    spike_times, spike_clusters  = st.validation.validation_tools.get_firing_times_ids(sorting, sampling_frequency)
    total_units = len(sorting.get_unit_ids()) 
    in_epoch = _get_in_epoch(spike_times, epoch_tuple, sampling_frequency)
    _, num_spikes_all = metrics.calculate_firing_rate_and_spikes(spike_times[in_epoch], spike_clusters[in_epoch], total_units)
    num_spikes_list = []
    for i in unit_indices:
        num_spikes_list.append(num_spikes_all[i])
    num_spikes = np.asarray(num_spikes_list)

    return num_spikes

def compute_isi_violations(sorting, sampling_frequency, isi_threshold=0.0015, min_isi=0.000166, \
                           unit_ids=None, epoch_tuple=None):
    '''
    Computes and returns the ISI violations for the given units and parameters.
    
    Parameters
    ----------
    sorting: SortingExtractor
        The sorting extractor.
    sampling_frequency: float
        The sampling frequency of the recording.
    isi_threshold: float
        The isi threshold for calculating isi violations.
    min_isi: float
        The minimum expected isi value.
    unit_ids: list
        List of unit ids to compute metric for. If not specified, all units are used
    epoch_tuple: int tuple
        A tuple with a start and end frame for the epoch in question.

    Returns
    ----------
    isi_violations: np.array
        The isi violations of the sorted units in the given epoch.
    '''
    if unit_ids is None or unit_ids == []:
        unit_ids = sorting.get_unit_ids()
        unit_indices = np.arange(len(unit_ids))
    else:
        unit_indices = _get_unit_indices(sorting, unit_ids)

    spike_times, spike_clusters  = st.validation.validation_tools.get_firing_times_ids(sorting, sampling_frequency)
    total_units = len(sorting.get_unit_ids()) 
    in_epoch = _get_in_epoch(spike_times, epoch_tuple, sampling_frequency)
    isi_violations_all = metrics.calculate_isi_violations(spike_times[in_epoch], spike_clusters[in_epoch], total_units, \
                                                          isi_threshold=isi_threshold, min_isi=min_isi)
    isi_violations_list = []
    for i in unit_indices:
        isi_violations_list.append(isi_violations_all[i])
    isi_violations = np.asarray(isi_violations_list)

    return isi_violations

def compute_presence_ratios(sorting, sampling_frequency, unit_ids=None, epoch_tuple=None):
    '''
    Computes and returns the presence ratios for the given units.
    
    Parameters
    ----------
    sorting: SortingExtractor
        The sorting extractor.
    sampling_frequency: float
        The sampling frequency of the recording.
    unit_ids: list
        List of unit ids to compute metric for. If not specified, all units are used
    epoch_tuple: int tuple
        A tuple with a start and end frame for the epoch in question.

    Returns
    ----------
    presence_ratios: np.array
        The presence ratios violations of the sorted units in the given epoch.
        calculate_presence_ratio
    '''

    if unit_ids is None or unit_ids == []:
        unit_ids = sorting.get_unit_ids()
        unit_indices = np.arange(len(unit_ids))
    else:
        unit_indices = _get_unit_indices(sorting, unit_ids)

    spike_times, spike_clusters  = st.validation.validation_tools.get_firing_times_ids(sorting, sampling_frequency)
    total_units = len(sorting.get_unit_ids()) 
    in_epoch = _get_in_epoch(spike_times, epoch_tuple, sampling_frequency)
    presence_ratios_all = metrics.calculate_presence_ratio(spike_times[in_epoch], spike_clusters[in_epoch], total_units)
    presence_ratios_list = []
    for i in unit_indices:
        presence_ratios_list.append(presence_ratios_all[i])
    presence_ratios = np.asarray(presence_ratios_list)

    return presence_ratios

def compute_drift_metrics(recording, sorting, drift_metrics_interval_s=51, drift_metrics_min_spikes_per_interval=10, \
                          nPC=3, ms_before=1., ms_after=2., dtype=None, max_num_waveforms=np.inf, \
                          max_num_pca_waveforms=np.inf, save_waveforms=False, unit_ids=None, epoch_tuple=None):
    '''
    Computes and returns the drift metrics for the sorted dataset.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor
    sorting: SortingExtractor
        The sorting extractor
    drift_metrics_interval_s: float
        Time period for evaluating drift.
    drift_metrics_min_spikes_per_interval: int
        Minimum number of spikes for evaluating drift metrics per interval. 
    nPC: int
        nPCFeatures in template-gui format
    ms_before: float
        Time period in ms to cut waveforms before the spike events
    ms_after: float
        Time period in ms to cut waveforms after the spike events
    dtype: dtype
        The numpy dtype of the waveforms
    max_num_waveforms: int
        The maximum number of waveforms to extract (default is np.inf)
    max_num_pca_waveforms: int
        The maximum number of waveforms to use to compute PCA (default is np.inf)
    save_waveforms: bool
        If True, waveforms are saved as waveforms.npy
    verbose: bool
        If True output is verbose
    unit_ids: list
        List of unit ids to compute metric for. If not specified, all units are used
    epoch_tuple: int tuple
        A tuple with a start and end frame for the epoch in question.

    Returns
    ----------
    max_drifts: np.array
        The max drift of the given units
    cumulative_drifts: np.array
        The cumulative drifts of the given unit
    '''

    if unit_ids is None or unit_ids == []:
        unit_ids = sorting.get_unit_ids()
        unit_indices = np.arange(len(unit_ids))
    else:
        unit_indices = _get_unit_indices(sorting, unit_ids)

    spike_times, spike_clusters, amplitudes, channel_map, \
    pc_features, pc_feature_ind  = st.validation.validation_tools.get_quality_metric_data(recording, sorting, nPC=nPC, ms_before=ms_before, \
                                                                                          ms_after=ms_after, dtype=dtype, max_num_waveforms=np.inf, \
                                                                                          max_num_pca_waveforms=max_num_waveforms, \
                                                                                          save_waveforms=save_waveforms)
    total_units = len(sorting.get_unit_ids()) 
    in_epoch = _get_in_epoch(spike_times, epoch_tuple, recording.get_sampling_frequency())
    
    max_drifts_all, cumulative_drifts_all = metrics.calculate_drift_metrics(spike_times[in_epoch],
                                                                            spike_clusters[in_epoch],
                                                                            total_units,
                                                                            pc_features[in_epoch,:,:],
                                                                            pc_feature_ind,
                                                                            drift_metrics_interval_s,
                                                                            drift_metrics_min_spikes_per_interval)
    max_drifts_list = []
    cumulative_drifts_list = []
    for i in unit_indices:
        max_drifts_list.append(max_drifts_all[i])
        cumulative_drifts_list.append(cumulative_drifts_all[i])
    
    max_drifts = np.asarray(max_drifts_list)
    cumulative_drifts = np.asarray(cumulative_drifts_list)

    return max_drifts, cumulative_drifts

def compute_silhouette_score(recording, sorting, max_spikes_for_silhouette=10000, nPC=3, ms_before=1., 
                             ms_after=2., dtype=None, max_num_waveforms=np.inf, max_num_pca_waveforms=np.inf, 
                             save_waveforms=False, unit_ids=None, epoch_tuple=None):
    '''
    Computes and returns the silhouette scores for each unit in the sorted dataset.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor
    sorting: SortingExtractor
        The sorting extractor
    max_spikes_for_silhouette: int
        Max spikes to be used for silhouette metric
    nPC: int
        nPCFeatures in template-gui format
    ms_before: float
        Time period in ms to cut waveforms before the spike events
    ms_after: float
        Time period in ms to cut waveforms after the spike events
    dtype: dtype
        The numpy dtype of the waveforms
    max_num_waveforms: int
        The maximum number of waveforms to extract (default is np.inf)
    max_num_pca_waveforms: int
        The maximum number of waveforms to use to compute PCA (default is np.inf)
    save_waveforms: bool
        If True, waveforms are saved as waveforms.npy
    verbose: bool
        If True output is verbose
    unit_ids: list
        List of unit ids to compute metric for. If not specified, all units are used
    epoch_tuple: int tuple
        A tuple with a start and end frame for the epoch in question.

    Returns
    ----------
    max_drifts: np.array
        The max drift of the given units
    cumulative_drifts: np.array
        The cumulative drifts of the given unit
    '''

    if unit_ids is None or unit_ids == []:
        unit_ids = sorting.get_unit_ids()
        unit_indices = np.arange(len(unit_ids))
    else:
        unit_indices = _get_unit_indices(sorting, unit_ids)

    spike_times, spike_clusters, amplitudes, channel_map, \
    pc_features, pc_feature_ind  = st.validation.validation_tools.get_quality_metric_data(recording, sorting, nPC=nPC, ms_before=ms_before, \
                                                                                          ms_after=ms_after, dtype=dtype, max_num_waveforms=np.inf, \
                                                                                          max_num_pca_waveforms=max_num_waveforms, \
                                                                                          save_waveforms=save_waveforms)    
    total_units = len(sorting.get_unit_ids()) 
    in_epoch = _get_in_epoch(spike_times, epoch_tuple, recording.get_sampling_frequency())
    spikes_in_epoch = np.sum(in_epoch)
    spikes_for_silhouette = min(spikes_in_epoch, max_spikes_for_silhouette)
    
    silhouette_scores_all = metrics.calculate_silhouette_score(spike_clusters[in_epoch],
                                                               total_units,
                                                               pc_features[in_epoch,:,:],
                                                               pc_feature_ind,
                                                               spikes_for_silhouette)
    silhouette_scores_list = []
    for i in unit_indices:
        silhouette_scores_list.append(silhouette_scores_all[i])
    silhouette_scores = np.asarray(silhouette_scores_list)
    return silhouette_scores

def compute_isolations_distances(recording, sorting, num_channels_to_compare=13, max_spikes_for_unit=500, nPC=3, \
                                 ms_before=1., ms_after=2., dtype=None, max_num_waveforms=np.inf, max_num_pca_waveforms=np.inf, \
                                 save_waveforms=False, unit_ids=None, epoch_tuple=None, seed=0):
    '''
    Computes and returns the mahalanobis metric, isolation distance, for the sorted dataset.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor
    sorting: SortingExtractor
        The sorting extractor
    num_channels_to_compare: int
        The number of channels to be used for the PC extraction and comparison
    max_spikes_for_unit: int
        Max spikes to be used from each unit
    nPC: int 
        nPCFeatures in template-gui format
    ms_before: float
        Time period in ms to cut waveforms before the spike events
    ms_after: float
        Time period in ms to cut waveforms after the spike events
    dtype: dtype
        The numpy dtype of the waveforms
    max_num_waveforms: int
        The maximum number of waveforms to extract (default is np.inf)
    max_num_pca_waveforms: int
        The maximum number of waveforms to use to compute PCA (default is np.inf)
    save_waveforms: bool
        If True, waveforms are saved as waveforms.npy
    verbose: bool
        If True output is verbose
    unit_ids: list
        List of unit ids to compute metric for. If not specified, all units are used
    epoch_tuple: int tuple
        A tuple with a start and end frame for the epoch in question.
    seed: int
        Random seed for extracting pc features.

    Returns
    ----------
    isolation_distances: np.array
        Returns the isolation distances of each specified unit
    '''

    if unit_ids is None or unit_ids == []:
        unit_ids = sorting.get_unit_ids()
        unit_indices = np.arange(len(unit_ids))
    else:
        unit_indices = _get_unit_indices(sorting, unit_ids)

    spike_times, spike_clusters, amplitudes, channel_map, \
    pc_features, pc_feature_ind  = st.validation.validation_tools.get_quality_metric_data(recording, sorting, nPC=nPC, ms_before=ms_before, \
                                                                                          ms_after=ms_after, dtype=dtype, max_num_waveforms=np.inf, \
                                                                                          max_num_pca_waveforms=max_num_waveforms, \
                                                                                          save_waveforms=save_waveforms)
    total_units = len(sorting.get_unit_ids()) 
    in_epoch = _get_in_epoch(spike_times, epoch_tuple, recording.get_sampling_frequency())
    
    isolation_distances_all = metrics.calculate_pc_metrics(spike_clusters=spike_clusters[in_epoch],
                                                           total_units=total_units,   
                                                           channel_map=channel_map,
                                                           pc_features=pc_features[in_epoch,:,:],
                                                           pc_feature_ind=pc_feature_ind,
                                                           num_channels_to_compare=num_channels_to_compare,
                                                           max_spikes_for_cluster=max_spikes_for_unit,
                                                           spikes_for_nn=None,
                                                           n_neighbors=None,
                                                           metric_names=['isolation_distance'],
                                                           seed=seed)[0]
    isolation_distances_list = []
    for i in unit_indices:
        isolation_distances_list.append(isolation_distances_all[i])
    isolation_distances = np.asarray(isolation_distances_list)
    return isolation_distances


def compute_l_ratios(recording, sorting, num_channels_to_compare=13, max_spikes_for_unit=500, nPC=3, \
                     ms_before=1., ms_after=2., dtype=None, max_num_waveforms=np.inf, max_num_pca_waveforms=np.inf, \
                     save_waveforms=False, unit_ids=None, epoch_tuple=None, seed=0):
    '''
    Computes and returns the mahalanobis metric, l-ratio, for the sorted dataset.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor
    sorting: SortingExtractor
        The sorting extractor
    num_channels_to_compare: int
        The number of channels to be used for the PC extraction and comparison
    max_spikes_for_unit: int
        Max spikes to be used from each unit
    nPC: int 
        nPCFeatures in template-gui format
    ms_before: float
        Time period in ms to cut waveforms before the spike events
    ms_after: float
        Time period in ms to cut waveforms after the spike events
    dtype: dtype
        The numpy dtype of the waveforms
    max_num_waveforms: int
        The maximum number of waveforms to extract (default is np.inf)
    max_num_pca_waveforms: int
        The maximum number of waveforms to use to compute PCA (default is np.inf)
    save_waveforms: bool
        If True, waveforms are saved as waveforms.npy
    verbose: bool
        If True output is verbose
    unit_ids: list
        List of unit ids to compute metric for. If not specified, all units are used
    epoch_tuple: int tuple
        A tuple with a start and end frame for the epoch in question.
    seed: int
        Random seed for extracting pc features.

    Returns
    ----------
    l_ratios: np.array
        Returns the L ratios of each specified unit
    '''

    if unit_ids is None or unit_ids == []:
        unit_ids = sorting.get_unit_ids()
        unit_indices = np.arange(len(unit_ids))
    else:
        unit_indices = _get_unit_indices(sorting, unit_ids)

    spike_times, spike_clusters, amplitudes, channel_map, \
    pc_features, pc_feature_ind  = st.validation.validation_tools.get_quality_metric_data(recording, sorting, nPC=nPC, ms_before=ms_before, \
                                                                                          ms_after=ms_after, dtype=dtype, max_num_waveforms=np.inf, \
                                                                                          max_num_pca_waveforms=max_num_waveforms, \
                                                                                          save_waveforms=save_waveforms)
    total_units = len(sorting.get_unit_ids()) 
    in_epoch = _get_in_epoch(spike_times, epoch_tuple, recording.get_sampling_frequency())
    
    l_ratios_all = metrics.calculate_pc_metrics(spike_clusters=spike_clusters[in_epoch],
                                                total_units=total_units,   
                                                channel_map=channel_map,
                                                pc_features=pc_features[in_epoch,:,:],
                                                pc_feature_ind=pc_feature_ind,
                                                num_channels_to_compare=num_channels_to_compare,
                                                max_spikes_for_cluster=max_spikes_for_unit,
                                                spikes_for_nn=None,
                                                n_neighbors=None,
                                                metric_names=['l_ratio'],
                                                seed=seed)[1]
    l_ratios_list = []
    for i in unit_indices:
        l_ratios_list.append(l_ratios_all[i])
    l_ratios = np.asarray(l_ratios_list)
    return l_ratios

def compute_d_primes(recording, sorting, num_channels_to_compare=13, max_spikes_for_unit=500, nPC=3, \
                     ms_before=1., ms_after=2., dtype=None, max_num_waveforms=np.inf, max_num_pca_waveforms=np.inf, \
                     save_waveforms=False, unit_ids=None, epoch_tuple=None, seed=0):
    '''
    Computes and returns the lda-based metric, d prime, for the sorted dataset.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor
    sorting: SortingExtractor
        The sorting extractor
    num_channels_to_compare: int
        The number of channels to be used for the PC extraction and comparison
    max_spikes_for_unit: int
        Max spikes to be used from each unit
    nPC: int 
        nPCFeatures in template-gui format
    ms_before: float
        Time period in ms to cut waveforms before the spike events
    ms_after: float
        Time period in ms to cut waveforms after the spike events
    dtype: dtype
        The numpy dtype of the waveforms
    max_num_waveforms: int
        The maximum number of waveforms to extract (default is np.inf)
    max_num_pca_waveforms: int
        The maximum number of waveforms to use to compute PCA (default is np.inf)
    save_waveforms: bool
        If True, waveforms are saved as waveforms.npy
    verbose: bool
        If True output is verbose
    unit_ids: list
        List of unit ids to compute metric for. If not specified, all units are used
    epoch_tuple: int tuple
        A tuple with a start and end frame for the epoch in question.
    seed: int
        Random seed for extracting pc features.

    Returns
    ----------
    d_primes: np.array
        Returns the d primes of each specified unit
    '''

    if unit_ids is None or unit_ids == []:
        unit_ids = sorting.get_unit_ids()
        unit_indices = np.arange(len(unit_ids))
    else:
        unit_indices = _get_unit_indices(sorting, unit_ids)

    spike_times, spike_clusters, amplitudes, channel_map, \
    pc_features, pc_feature_ind  = st.validation.validation_tools.get_quality_metric_data(recording, sorting, nPC=nPC, ms_before=ms_before, \
                                                                                          ms_after=ms_after, dtype=dtype, max_num_waveforms=np.inf, \
                                                                                          max_num_pca_waveforms=max_num_waveforms, \
                                                                                          save_waveforms=save_waveforms)
    total_units = len(sorting.get_unit_ids()) 
    in_epoch = _get_in_epoch(spike_times, epoch_tuple, recording.get_sampling_frequency())
    
    d_primes_all = metrics.calculate_pc_metrics(spike_clusters=spike_clusters[in_epoch],
                                                total_units=total_units,   
                                                channel_map=channel_map,
                                                pc_features=pc_features[in_epoch,:,:],
                                                pc_feature_ind=pc_feature_ind,
                                                num_channels_to_compare=num_channels_to_compare,
                                                max_spikes_for_cluster=max_spikes_for_unit,
                                                spikes_for_nn=None,
                                                n_neighbors=None,
                                                metric_names=['d_prime'],
                                                seed=seed)[2]
    d_primes_list = []
    for i in unit_indices:
        d_primes_list.append(d_primes_all[i])
    d_primes = np.asarray(d_primes_list)
    return d_primes

def compute_unit_SNR(recording, sorting, unit_ids=None, save_as_property=True, mode='mad',
                     seconds=10, max_num_waveforms=1000, apply_filter=False, freq_min=300, freq_max=6000):
    '''
    Computes signal-to-noise ratio (SNR) of the average waveforms on the largest channel.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor
    sorting: SortingExtractor
        The sorting extractor
    unit_ids: list
        List of unit ids to compute SNR for. If not specified, all units are used
    save_as_property: bool
        If True (defult), the computed SNR is added as a unit property to the sorting extractor as 'snr'
    mode: str
        Mode to compute noise SNR ('mad' | 'std' - default 'mad')
    seconds: float
        Number of seconds to compute noise level from (default 10)
    max_num_waveforms: int
        Maximum number of waveforms to cpmpute templates from (default 1000)
    apply_filter: bool
        If True, recording is filtered before computing noise level
    freq_min: float
        High-pass frequency for optional filter (default 300 Hz)
    freq_max: float
        Low-pass frequency for optional filter (default 6000 Hz)

    Returns
    -------
    snr_list: np.array
        List of computed SNRs

    '''
    if unit_ids is None:
        unit_ids = sorting.get_unit_ids()
    if apply_filter:
        recording_f = st.preprocessing.bandpass_filter(recording=recording, freq_min=freq_min, freq_max=freq_max,
                                                       cache=True)
    else:
        recording_f = recording
    channel_noise_levels = _compute_channel_noise_levels(recording=recording_f, mode=mode, seconds=seconds)
    templates = st.postprocessing.get_unit_templates(recording_f, sorting, unit_ids=unit_ids,
                                                    max_num_waveforms=max_num_waveforms,
                                                    mode='median')
    max_channels = st.postprocessing.get_unit_max_channels(recording, sorting, unit_ids=unit_ids,
                                                          max_num_waveforms=max_num_waveforms, peak='both',
                                                          mode='median')
    snr_list = []
    for i, unit_id in enumerate(sorting.get_unit_ids()):
        max_channel_idx = recording.get_channel_ids().index(max_channels[i])
        snr = _compute_template_SNR(templates[i], channel_noise_levels, max_channel_idx)
        if save_as_property:
            sorting.set_unit_property(unit_id, 'snr', snr)
        snr_list.append(snr)
    return np.asarray(snr_list)


def _compute_template_SNR(template, channel_noise_levels, max_channel_idx):
    '''
    Computes SNR on the channel with largest amplitude

    Parameters
    ----------
    template: np.array
        Template (n_elec, n_timepoints)
    channel_noise_levels: list
        Noise levels for the different channels
    max_channel_idx: int
        Index of channel with largest templaye

    Returns
    -------
    snr: float
        Signal-to-noise ratio for the template
    '''
    snr = np.max(np.abs(template[max_channel_idx])) / channel_noise_levels[max_channel_idx]
    return snr


def _compute_channel_noise_levels(recording, mode='mad', seconds=10):
    '''
    Computes noise level channel-wise

    Parameters
    ----------
    recording: RecordingExtractor
        The recording ectractor object
    mode: str
        'std' or 'mad' (default
    seconds: float
        Number of seconds to compute SNR from

    Returns
    -------
    moise_levels: list
        Noise levels for each channel
    '''
    M = recording.get_num_channels()
    n_frames = int(seconds * recording.get_sampling_frequency())

    if n_frames > recording.get_num_frames():
        start_frame = 0
        end_frame = recording.get_num_frames()
    else:
        start_frame = np.random.randint(0, recording.get_num_frames() - n_frames)
        end_frame = start_frame + n_frames

    X = recording.get_traces(start_frame=start_frame, end_frame=end_frame)
    noise_levels = []
    for ch in range(M):
        if mode == 'std':
            noise_level = np.std(X[ch, :])
        elif mode == 'mad':
            noise_level = np.median(np.abs(X[ch, :])/0.6745)
        else:
            raise Exception("'mode' can be 'std' or 'mad'")
        noise_levels.append(noise_level)
    return noise_levels

def _get_unit_indices(sorting, unit_ids):
    unit_indices = []
    sorting_unit_ids = np.asarray(sorting.get_unit_ids())
    for unit_id in unit_ids:
        index, = np.where(sorting_unit_ids == unit_id)
        if len(index) != 0:
            unit_indices.append(index[0])
    return unit_indices

def _get_in_epoch(spike_times, epoch_tuple, sampling_frequency):
    if epoch_tuple is None:
        epoch = (0, np.inf)
    else:
        epoch = (epoch_tuple[0]/sampling_frequency, epoch_tuple[1]/sampling_frequency)
    in_epoch = np.logical_and(spike_times > epoch[0], spike_times < epoch[1])
    return in_epoch
