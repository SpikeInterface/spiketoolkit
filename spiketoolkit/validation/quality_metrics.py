import spiketoolkit as st


def compute_num_spikes(sorting, sampling_frequency=None, unit_ids=None, epoch_tuples=None, epoch_names=None,
                       save_as_property=True):
    '''
    Computes and returns the spike times in seconds and also returns the cluster_ids needed for quality metrics.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated.
    sampling_frequency:
        The sampling frequency of the result. If None, will check to see if sampling frequency is in sorting extractor
    unit_ids: list
        List of unit ids to compute metric for. If not specified, all units are used
    epoch_tuples: list
        A list of tuples with a start and end time for each epoch
    epoch_names: list
        A list of strings for the names of the given epochs
    save_as_property: bool
        If True, the metric is saved as sorting property

    Returns
    ----------
    num_spikes_epochs: list
        The spike counts of the sorted units in the given epochs.
    '''

    if unit_ids is None:
        unit_ids = sorting.get_unit_ids()

    metric_calculator = st.validation.MetricCalculator(sorting, sampling_frequency=sampling_frequency,
                                                       unit_ids=unit_ids,
                                                       epoch_tuples=epoch_tuples, epoch_names=epoch_names)
    num_spikes_epochs = metric_calculator.compute_num_spikes()

    if save_as_property:
        if epoch_tuples is None:
            for i_u, u in enumerate(unit_ids):
                sorting.set_unit_property(u, 'num_spikes', num_spikes_epochs[i_u])
        else:
            raise NotImplementedError("Quality metrics cannot be saved as properties if 'epochs_tuples' are given.")

    return num_spikes_epochs


def compute_firing_rates(sorting, sampling_frequency=None, unit_ids=None, epoch_tuples=None, epoch_names=None,
                         save_as_property=True):
    '''
    Computes and returns the spike times in seconds and also returns the cluster_ids needed for quality metrics.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated.
    sampling_frequency:
        The sampling frequency of the result. If None, will check to see if sampling frequency is in sorting extractor
    unit_ids: list
        List of unit ids to compute metric for. If not specified, all units are used
    epoch_tuples: list
        A list of tuples with a start and end time for each epoch
    epoch_names: list
        A list of strings for the names of the given epochs
    save_as_property: bool
        If True, the metric is saved as sorting property

    Returns
    ----------
    firing_rates_epochs: list
        The firing rates of the sorted units in the given epochs.
    '''

    if unit_ids is None:
        unit_ids = sorting.get_unit_ids()

    metric_calculator = st.validation.MetricCalculator(sorting, sampling_frequency=sampling_frequency,
                                                       unit_ids=unit_ids,
                                                       epoch_tuples=epoch_tuples, epoch_names=epoch_names)
    firings_rates_epochs = metric_calculator.compute_firing_rates()

    if save_as_property:
        if epoch_tuples is None:
            for i_u, u in enumerate(unit_ids):
                sorting.set_unit_property(u, 'firing_rate', firings_rates_epochs[i_u])
        else:
            raise NotImplementedError("Quality metrics cannot be saved as properties if 'epochs_tuples' are given.")

    return firings_rates_epochs


def compute_presence_ratios(sorting, sampling_frequency=None, unit_ids=None, epoch_tuples=None, epoch_names=None,
                            save_as_property=True):
    '''
    Computes and returns the presence ratios.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated.
    sampling_frequency:
        The sampling frequency of the result. If None, will check to see if sampling frequency is in sorting extractor
    unit_ids: list
        List of unit ids to compute metric for. If not specified, all units are used
    epoch_tuples: list
        A list of tuples with a start and end time for each epoch
    epoch_names: list
        A list of strings for the names of the given epochs
    save_as_property: bool
        If True, the metric is saved as sorting property

    Returns
    ----------
    presence_ratios_epochs: list
        The presence ratios violations of the sorted units in the given epochs.
    '''

    if unit_ids is None:
        unit_ids = sorting.get_unit_ids()

    metric_calculator = st.validation.MetricCalculator(sorting, sampling_frequency=sampling_frequency,
                                                       unit_ids=unit_ids,
                                                       epoch_tuples=epoch_tuples, epoch_names=epoch_names)
    presence_ratios_epochs = metric_calculator.compute_presence_ratios()

    if save_as_property:
        if epoch_tuples is None:
            for i_u, u in enumerate(unit_ids):
                sorting.set_unit_property(u, 'presence_ratio', presence_ratios_epochs[i_u])
        else:
            raise NotImplementedError("Quality metrics cannot be saved as properties if 'epochs_tuples' are given.")

    return presence_ratios_epochs


def compute_isi_violations(sorting, sampling_frequency=None, isi_threshold=0.0015, min_isi=0.000166, unit_ids=None,
                           epoch_tuples=None, epoch_names=None, save_as_property=True):
    '''
    Computes and returns the ISI violations for the given parameters.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated.
    sampling_frequency:
        The sampling frequency of the result. If None, will check to see if sampling frequency is in sorting extractor
    isi_threshold: float
        The isi threshold for calculating isi violations
    min_isi: float
        The minimum expected isi value
    unit_ids: list
        List of unit ids to compute metric for. If not specified, all units are used
    epoch_tuples: list
        A list of tuples with a start and end time for each epoch
    epoch_names: list
        A list of strings for the names of the given epochs
    save_as_property: bool
        If True, the metric is saved as sorting property

    Returns
    ----------
    isi_violations_epochs: list
        The isi violations of the sorted units in the given epochs.
    '''

    if unit_ids is None:
        unit_ids = sorting.get_unit_ids()

    metric_calculator = st.validation.MetricCalculator(sorting, sampling_frequency=sampling_frequency,
                                                       unit_ids=unit_ids,
                                                       epoch_tuples=epoch_tuples, epoch_names=epoch_names)
    isi_violations_epochs = metric_calculator.compute_isi_violations(isi_threshold=isi_threshold, min_isi=min_isi)

    if save_as_property:
        if epoch_tuples is None:
            for i_u, u in enumerate(unit_ids):
                sorting.set_unit_property(u, 'isi_violation', isi_violations_epochs[i_u])
        else:
            raise NotImplementedError("Quality metrics cannot be saved as properties if 'epochs_tuples' are given.")

    return isi_violations_epochs


def compute_amplitude_cutoffs(sorting, recording, amp_method='absolute', amp_peak='both', amp_frames_before=3,
                              amp_frames_after=3, apply_filter=True, freq_min=300, freq_max=6000, save_features_props=False, 
                              unit_ids=None, epoch_tuples=None, epoch_names=None, save_as_property=True, seed=0):
    '''
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

    Returns
    ----------
    amplitude_cutoffs_epochs: list
        The amplitude cutoffs of the sorted units in the given epochs.
    '''

    if unit_ids is None:
        unit_ids = sorting.get_unit_ids()

    metric_calculator = st.validation.MetricCalculator(sorting, sampling_frequency=recording.get_sampling_frequency(),
                                                       unit_ids=unit_ids,
                                                       epoch_tuples=epoch_tuples, epoch_names=epoch_names)
    metric_calculator.compute_amplitudes(recording=recording, amp_method=amp_method, amp_peak=amp_peak,
                                         amp_frames_before=amp_frames_before,
                                         amp_frames_after=amp_frames_after, apply_filter=apply_filter, 
                                         freq_min=freq_min, freq_max=freq_max, 
                                         save_features_props=save_features_props, seed=seed)
    amplitude_cutoffs_epochs = metric_calculator.compute_amplitude_cutoffs()

    if save_as_property:
        if epoch_tuples is None:
            for i_u, u in enumerate(unit_ids):
                sorting.set_unit_property(u, 'amplitude_cutoff', amplitude_cutoffs_epochs[i_u])
        else:
            raise NotImplementedError("Quality metrics cannot be saved as properties if 'epochs_tuples' are given.")

    return amplitude_cutoffs_epochs


def compute_snrs(sorting, recording, snr_mode='mad', snr_noise_duration=10.0, max_spikes_per_unit_for_snr=1000,
                 recompute_info=True, apply_filter=True, freq_min=300, freq_max=6000, save_features_props=False, 
                 unit_ids=None, epoch_tuples=None, epoch_names=None, save_as_property=True, seed=0):
    '''
    Computes and stores snrs for the sorted units.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated.
    recording: RecordingExtractor
        The given recording extractor from which to extract amplitudes.
    snr_mode: str
            Mode to compute noise SNR ('mad' | 'std' - default 'mad')
    snr_noise_duration: float
        Number of seconds to compute noise level from (default 10.0)
    max_spikes_per_unit_for_snr: int
        Maximum number of spikes to compute templates from (default 1000)
    unit_ids: list
        List of unit ids to compute metric for. If not specified, all units are used
    recompute_info: bool
        If True, waveforms are recomputed
    apply_filter: bool
        If True, recording is bandpass-filtered.
    freq_min: float
        High-pass frequency for optional filter (default 300 Hz).
    freq_max: float
        Low-pass frequency for optional filter (default 6000 Hz).
    save_features_props: bool
        If True, waveforms and templates are saved as properties and features of the sorting extractor
    epoch_tuples: list
        A list of tuples with a start and end time for each epoch.
    epoch_names: list
        A list of strings for the names of the given epochs.
    save_as_property: bool
        If True, the metric is saved as sorting property
    seed: int
        Random seed for reproducibility.

    Returns
    ----------
    snrs_epochs: list
       The snrs of the sorted units in the given epochs.
    '''

    if unit_ids is None:
        unit_ids = sorting.get_unit_ids()

    metric_calculator = st.validation.MetricCalculator(sorting, sampling_frequency=recording.get_sampling_frequency(),
                                                       unit_ids=unit_ids,
                                                       epoch_tuples=epoch_tuples, epoch_names=epoch_names)
    metric_calculator.set_recording(recording, apply_filter=apply_filter, freq_min=freq_min, freq_max=freq_max)
    snrs_epochs = metric_calculator.compute_snrs(snr_mode=snr_mode, snr_noise_duration=snr_noise_duration,
                                                 max_spikes_per_unit_for_snr=max_spikes_per_unit_for_snr,
                                                 recompute_info=recompute_info,
                                                 save_features_props=save_features_props, seed=seed)

    if save_as_property:
        if epoch_tuples is None:
            for i_u, u in enumerate(unit_ids):
                sorting.set_unit_property(u, 'snr', snrs_epochs[i_u])
        else:
            raise NotImplementedError("Quality metrics cannot be saved as properties if 'epochs_tuples' are given.")
    return snrs_epochs


def compute_drift_metrics(sorting, recording, drift_metrics_interval_s=51, drift_metrics_min_spikes_per_interval=10,
                          n_comp=3, ms_before=1., ms_after=2., dtype=None, max_spikes_per_unit=300, recompute_info=True,
                          max_spikes_for_pca=1e5, apply_filter=True, freq_min=300, freq_max=6000, save_features_props=False, 
                          unit_ids=None, epoch_tuples=None, epoch_names=None, save_as_property=True, seed=0):
    '''
    Computes and returns the drift metrics for the sorted dataset.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated.
    recording: RecordingExtractor
        The given recording extractor from which to extract amplitudes.
    drift_metrics_interval_s: float
        Time period for evaluating drift.
    drift_metrics_min_spikes_per_interval: int
        Minimum number of spikes for evaluating drift metrics per interval.
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
    seed: int
        Random seed for reproducibility

    Returns
    ----------
    max_drifts_epochs: list
        The max drift of the given units over the specified epochs
    cumulative_drifts_epochs: list
        The cumulative drifts of the given units over the specified epochs
    '''
    if unit_ids is None:
        unit_ids = sorting.get_unit_ids()

    metric_calculator = st.validation.MetricCalculator(sorting, sampling_frequency=recording.get_sampling_frequency(),
                                                       unit_ids=unit_ids,
                                                       epoch_tuples=epoch_tuples, epoch_names=epoch_names)
    metric_calculator.compute_pca_scores(recording=recording, n_comp=n_comp, ms_before=ms_before, ms_after=ms_after,
                                         dtype=dtype,
                                         max_spikes_per_unit=max_spikes_per_unit,
                                         recompute_info=recompute_info,
                                         max_spikes_for_pca=max_spikes_for_pca,
                                         apply_filter=apply_filter, freq_min=freq_min, freq_max=freq_max,
                                         save_features_props=save_features_props, seed=seed)
    max_drifts_epochs, cumulative_drifts_epochs = metric_calculator.compute_drift_metrics(
        drift_metrics_interval_s=drift_metrics_interval_s,
        drift_metrics_min_spikes_per_interval=drift_metrics_min_spikes_per_interval)

    if save_as_property:
        if epoch_tuples is None:
            for i_u, u in enumerate(unit_ids):
                sorting.set_unit_property(u, 'max_drift', max_drifts_epochs[i_u])
                sorting.set_unit_property(u, 'cumulative_drift', cumulative_drifts_epochs[i_u])
        else:
            raise NotImplementedError("Quality metrics cannot be saved as properties if 'epochs_tuples' are given.")

    return max_drifts_epochs, cumulative_drifts_epochs


def compute_silhouette_scores(sorting, recording, max_spikes_for_silhouette=10000, n_comp=3, ms_before=1., ms_after=2.,
                              dtype=None, max_spikes_per_unit=300, recompute_info=True,
                              max_spikes_for_pca=1e5, apply_filter=True, freq_min=300, freq_max=6000, save_features_props=False, 
                              unit_ids=None, epoch_tuples=None, epoch_names=None, save_as_property=True, seed=0):
    '''
    Computes and returns the silhouette scores in the sorted dataset.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated.
    recording: RecordingExtractor
        The given recording extractor from which to extract amplitudes.
    max_spikes_for_silhouette: int
        Max spikes to be used for silhouette metric
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
    seed: int
        Random seed for reproducibility

    Returns
    ----------
    silhouette_scores_epochs: list
        The silhouette scores of the given units for the specified epochs.
    '''
    if unit_ids is None:
        unit_ids = sorting.get_unit_ids()

    metric_calculator = st.validation.MetricCalculator(sorting, sampling_frequency=recording.get_sampling_frequency(),
                                                       unit_ids=unit_ids,
                                                       epoch_tuples=epoch_tuples, epoch_names=epoch_names)
    metric_calculator.compute_pca_scores(recording=recording, n_comp=n_comp, ms_before=ms_before, ms_after=ms_after,
                                         dtype=dtype,
                                         max_spikes_per_unit=max_spikes_per_unit,
                                         recompute_info=recompute_info,
                                         max_spikes_for_pca=max_spikes_for_pca,
                                         apply_filter=apply_filter, freq_min=freq_min, freq_max=freq_max,
                                         save_features_props=save_features_props, seed=seed)
    silhouette_scores_epochs = metric_calculator.compute_silhouette_scores(
        max_spikes_for_silhouette=max_spikes_for_silhouette, seed=seed)

    if save_as_property:
        if epoch_tuples is None:
            for i_u, u in enumerate(unit_ids):
                sorting.set_unit_property(u, 'silhouette_score', silhouette_scores_epochs[i_u])
        else:
            raise NotImplementedError("Quality metrics cannot be saved as properties if 'epochs_tuples' are given.")

    return silhouette_scores_epochs


def compute_isolation_distances(sorting, recording, num_channels_to_compare=13, max_spikes_per_cluster=500, n_comp=3,
                                ms_before=1., ms_after=2.,
                                dtype=None, max_spikes_per_unit=300, recompute_info=True, max_spikes_for_pca=1e5,
                                apply_filter=True, freq_min=300, freq_max=6000, save_features_props=False,
                                unit_ids=None, epoch_tuples=None, epoch_names=None, save_as_property=True, seed=0):
    '''
    Computes and returns the mahalanobis metric, isolation distance, for the sorted dataset.

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
    seed: int
        Random seed for reproducibility

    Returns
    ----------
    isolation_distances_epochs: list
        Returns the isolation distances of each specified unit for the given epochs.
    '''
    if unit_ids is None:
        unit_ids = sorting.get_unit_ids()

    metric_calculator = st.validation.MetricCalculator(sorting, sampling_frequency=recording.get_sampling_frequency(),
                                                       unit_ids=unit_ids,
                                                       epoch_tuples=epoch_tuples, epoch_names=epoch_names)
    metric_calculator.compute_pca_scores(recording=recording, n_comp=n_comp, ms_before=ms_before, ms_after=ms_after,
                                         dtype=dtype,
                                         max_spikes_per_unit=max_spikes_per_unit,
                                         recompute_info=recompute_info,
                                         max_spikes_for_pca=max_spikes_for_pca,
                                         apply_filter=apply_filter, freq_min=freq_min, freq_max=freq_max,
                                         save_features_props=save_features_props, seed=seed)
    isolation_distances_epochs = metric_calculator.compute_isolation_distances(
        num_channels_to_compare=num_channels_to_compare,
        max_spikes_per_cluster=max_spikes_per_cluster, seed=seed)

    if save_as_property:
        if epoch_tuples is None:
            for i_u, u in enumerate(unit_ids):
                sorting.set_unit_property(u, 'isolation_distance', isolation_distances_epochs[i_u])
        else:
            raise NotImplementedError("Quality metrics cannot be saved as properties if 'epochs_tuples' are given.")

    return isolation_distances_epochs


def compute_l_ratios(sorting, recording, num_channels_to_compare=13, max_spikes_per_cluster=500, n_comp=3, ms_before=1.,
                     ms_after=2., dtype=None, max_spikes_per_unit=300, recompute_info=True,
                     max_spikes_for_pca=1e5, apply_filter=True, freq_min=300, freq_max=6000, save_features_props=False, 
                     unit_ids=None, epoch_tuples=None, epoch_names=None, save_as_property=True, seed=0):
    '''
    Computes and returns the mahalanobis metric, l-ratio, for the sorted dataset.

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
    seed: int
        Random seed for reproducibility

    Returns
    ----------
    l_ratios_epochs: list
        Returns the L ratios of each specified unit for the given epochs
    '''
    if unit_ids is None:
        unit_ids = sorting.get_unit_ids()

    metric_calculator = st.validation.MetricCalculator(sorting, sampling_frequency=recording.get_sampling_frequency(),
                                                       unit_ids=unit_ids,
                                                       epoch_tuples=epoch_tuples, epoch_names=epoch_names)
    metric_calculator.compute_pca_scores(recording=recording, n_comp=n_comp, ms_before=ms_before, ms_after=ms_after,
                                         dtype=dtype,
                                         max_spikes_per_unit=max_spikes_per_unit,
                                         recompute_info=recompute_info,
                                         max_spikes_for_pca=max_spikes_for_pca,
                                         apply_filter=apply_filter, freq_min=freq_min, freq_max=freq_max,
                                         save_features_props=save_features_props, seed=seed)
    l_ratios_epochs = metric_calculator.compute_l_ratios(num_channels_to_compare=num_channels_to_compare,
                                                         max_spikes_per_cluster=max_spikes_per_cluster, seed=seed)

    if save_as_property:
        if epoch_tuples is None:
            for i_u, u in enumerate(unit_ids):
                sorting.set_unit_property(u, 'l_ratio', l_ratios_epochs[i_u])
        else:
            raise NotImplementedError("Quality metrics cannot be saved as properties if 'epochs_tuples' are given.")

    return l_ratios_epochs


def compute_d_primes(sorting, recording, num_channels_to_compare=13, max_spikes_per_cluster=500, n_comp=3, ms_before=1.,
                     ms_after=2., dtype=None, max_spikes_per_unit=300, recompute_info=True,
                     max_spikes_for_pca=1e5, apply_filter=True, freq_min=300, freq_max=6000,
                     save_features_props=False, unit_ids=None, epoch_tuples=None, epoch_names=None,
                     save_as_property=True, seed=0):
    '''
    Computes and returns the lda-based metric, d-prime, for the sorted dataset.

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
    seed: int
        Random seed for reproducibility

    Returns
    ----------
    d_primes_epochs: list
        Returns the d primes of each specified unit for the given epochs.
    '''
    if unit_ids is None:
        unit_ids = sorting.get_unit_ids()

    metric_calculator = st.validation.MetricCalculator(sorting, sampling_frequency=recording.get_sampling_frequency(),
                                                       unit_ids=unit_ids,
                                                       epoch_tuples=epoch_tuples, epoch_names=epoch_names)
    metric_calculator.compute_pca_scores(recording=recording, n_comp=n_comp, ms_before=ms_before, ms_after=ms_after,
                                         dtype=dtype,
                                         max_spikes_per_unit=max_spikes_per_unit,
                                         recompute_info=recompute_info,
                                         max_spikes_for_pca=max_spikes_for_pca,
                                         apply_filter=apply_filter, freq_min=freq_min, freq_max=freq_max,
                                         save_features_props=save_features_props, seed=seed)
    d_primes_epochs = metric_calculator.compute_d_primes(num_channels_to_compare=num_channels_to_compare,
                                                         max_spikes_per_cluster=max_spikes_per_cluster, seed=seed)

    if save_as_property:
        if epoch_tuples is None:
            for i_u, u in enumerate(unit_ids):
                sorting.set_unit_property(u, 'd_prime', d_primes_epochs[i_u])
        else:
            raise NotImplementedError("Quality metrics cannot be saved as properties if 'epochs_tuples' are given.")

    return d_primes_epochs


def compute_nn_metrics(sorting, recording, num_channels_to_compare=13, max_spikes_per_cluster=500,
                       max_spikes_for_nn=10000, n_neighbors=4, n_comp=3, ms_before=1., ms_after=2., 
                       dtype=None, max_spikes_per_unit=300, recompute_info=True, max_spikes_for_pca=1e5, 
                       apply_filter=True, freq_min=300, freq_max=6000, save_features_props=False,
                       unit_ids=None, epoch_tuples=None, epoch_names=None, save_as_property=True, seed=0):
    '''
    Computes and returns the nearest neighbor metrics for the sorted dataset.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated.
    recording: RecordingExtractor
        The given recording extractor from which to extract amplitudes.
    num_channels_to_compare: int
        The number of channels to be used for the PC extraction and comparison.
    max_spikes_per_cluster: int
        Max spikes to be used from each unit.
    max_spikes_for_nn: int
        Max spikes to be used for nearest-neighbors calculation.
    n_neighbors: int
        Number of neighbors to compare.
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
    seed: int
        Random seed for reproducibility

    Returns
    ----------
    nn_hit_rates_epochs: np.array
        The nearest neighbor hit rates for each specified unit.
    nn_miss_rates_epochs: np.array
        The nearest neighbor miss rates for each specified unit.
    '''
    if unit_ids is None:
        unit_ids = sorting.get_unit_ids()

    metric_calculator = st.validation.MetricCalculator(sorting, sampling_frequency=recording.get_sampling_frequency(),
                                                       unit_ids=unit_ids,
                                                       epoch_tuples=epoch_tuples, epoch_names=epoch_names)
    metric_calculator.compute_pca_scores(recording=recording, n_comp=n_comp, ms_before=ms_before, ms_after=ms_after,
                                         dtype=dtype,
                                         max_spikes_per_unit=max_spikes_per_unit,
                                         recompute_info=recompute_info,
                                         max_spikes_for_pca=max_spikes_for_pca,
                                         apply_filter=apply_filter, freq_min=freq_min, freq_max=freq_max,
                                         save_features_props=save_features_props, seed=seed)
    nn_hit_rates_epochs, nn_miss_rates_epochs = metric_calculator.compute_nn_metrics(
        num_channels_to_compare=num_channels_to_compare,
        max_spikes_per_cluster=max_spikes_per_cluster,
        max_spikes_for_nn=max_spikes_for_nn, n_neighbors=n_neighbors,
        seed=seed)

    if save_as_property:
        if epoch_tuples is None:
            for i_u, u in enumerate(unit_ids):
                sorting.set_unit_property(u, 'nn_hit_rates', nn_hit_rates_epochs[i_u])
                sorting.set_unit_property(u, 'nn_miss_rates', nn_miss_rates_epochs[i_u])
        else:
            raise NotImplementedError("Quality metrics cannot be saved as properties if 'epochs_tuples' are given.")
    return nn_hit_rates_epochs, nn_miss_rates_epochs


def compute_metrics(sorting, recording=None, sampling_frequency=None, isi_threshold=0.0015, min_isi=0.000166,
                    snr_mode='mad', snr_noise_duration=10.0, max_spikes_per_unit_for_snr=1000,
                    drift_metrics_interval_s=51, drift_metrics_min_spikes_per_interval=10,
                    max_spikes_for_silhouette=10000, num_channels_to_compare=13, max_spikes_per_cluster=500,
                    max_spikes_for_nn=10000, n_neighbors=4, n_comp=3, ms_before=1., ms_after=2., dtype=None,
                    max_spikes_per_unit=300,  amp_method='absolute', amp_peak='both', amp_frames_before=3, 
                    amp_frames_after=3, recompute_info=True,  max_spikes_for_pca=1e5, apply_filter=True, 
                    freq_min=300, freq_max=6000, save_features_props=False, metric_names=None, unit_ids=None, 
                    epoch_tuples=None, epoch_names=None, return_dataframe=False, seed=0):
    '''
    Computes and returns all specified metrics for the sorted dataset.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting result to be evaluated.
    sampling_frequency:
        The sampling frequency of the result. If None, will check to see if sampling frequency is in sorting extractor.
    recording: RecordingExtractor
        The given recording extractor from which to extract amplitudes. If None, certain metrics cannot be computed.
    isi_threshold: float
        The isi threshold for calculating isi violations.
    min_isi: float
        The minimum expected isi value.
    snr_mode: str
        Mode to compute noise SNR ('mad' | 'std' - default 'mad')
    snr_noise_duration: float
        Number of seconds to compute noise level from (default 10.0)
    max_spikes_per_unit_for_snr: int
        Maximum number of spikes to compute templates from (default 1000)
    drift_metrics_interval_s: float
        Time period for evaluating drift.
    drift_metrics_min_spikes_per_interval: int
        Minimum number of spikes for evaluating drift metrics per interval.
    max_spikes_for_silhouette: int
        Max spikes to be used for silhouette metric
    num_channels_to_compare: int
        The number of channels to be used for the PC extraction and comparison.
    max_spikes_per_cluster: int
        Max spikes to be used from each unit to compute metrics.
    max_spikes_for_nn: int
        Max spikes to be used for nearest-neighbors calculation.
    n_neighbors: int
        Number of neighbors to compare for  nearest-neighbors calculation.
    max_spikes_per_unit: int
        The maximum number of spikes to extract (default is np.inf)
    amp_method: str
        If 'absolute' (default), amplitudes are absolute amplitudes in uV are returned.
        If 'relative', amplitudes are returned as ratios between waveform amplitudes and template amplitudes.
    amp_peak: str
        If maximum channel has to be found among negative peaks ('neg'), positive ('pos') or both ('both' - default)
    amp_frames_before: int
        Frames before peak to compute amplitude
    amp_frames_after: float
        Frames after peak to compute amplitude
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
    n_comp: int
        n_compFeatures in template-gui format
    ms_before: float
        Time period in ms to cut waveforms before the spike events
    ms_after: float
        Time period in ms to cut waveforms after the spike events
    dtype: dtype
        The numpy dtype of the waveforms
    metrics_names: list
        The list of metric names to be computed. Available metrics are: 'firing_rate', 'num_spikes', 'isi_viol',
            'presence_ratio', 'amplitude_cutoff', 'max_drift', 'cumulative_drift', 'silhouette_score',
            'isolation_distance', 'l_ratio', 'd_prime', 'nn_hit_rate', 'nn_miss_rate', 'snr'. If None, all metrics are
            computed.
    unit_ids: list
        List of unit ids to compute metric for. If not specified, all units are used
    epoch_tuples: list
        A list of tuples with a start and end time for each epoch.
    epoch_names: list
        A list of strings for the names of the given epochs.
    return_dataframe: bool
        If True, this function will return a dataframe of the metrics.
    seed: int
        Random seed for reproducibility

    Returns
    ----------
    metrics_epochs : list
        List of metrics data. The list consists of lists of metric data for each given epoch.
    OR
    metrics_df: pandas.DataFrame
        A pandas dataframe of the cached metrics
    '''
    metrics_epochs = []
    all_metrics_list = ['firing_rate', 'num_spikes', 'isi_viol', 'presence_ratio', 'amplitude_cutoff', 'max_drift',
                        'cumulative_drift', 'silhouette_score', 'isolation_distance', 'l_ratio', 'd_prime',
                        'nn_hit_rate', 'nn_miss_rate', 'snr']

    if metric_names is None:
        metric_names = all_metrics_list
    else:
        bad_metrics = []
        for m in metric_names:
            if m not in all_metrics_list:
                bad_metrics.append(m)
        if len(bad_metrics) > 0:
            raise ValueError("Wrong metrics name: " + str(bad_metrics))

    if recording is not None:
        sampling_frequency = recording.get_sampling_frequency()

    metric_calculator = st.validation.MetricCalculator(sorting, sampling_frequency=sampling_frequency,
                                                       unit_ids=unit_ids,
                                                       epoch_tuples=epoch_tuples, epoch_names=epoch_names)

    if 'max_drift' in metric_names or 'cumulative_drift' in metric_names or 'silhouette_score' in metric_names \
            or 'isolation_distance' in metric_names or 'l_ratio' in metric_names or 'd_prime' in metric_names \
            or 'nn_hit_rate' in metric_names or 'nn_miss_rate' in metric_names:
        if recording is None:
            raise ValueError("The recording cannot be None when computing max_drift, cumulative_drift, "
                             "silhouette_score isolation_distance, l_ratio, d_prime, nn_hit_rate, amplitude_cutoff, "
                             "or nn_miss_rate.")
        else:
            metric_calculator.compute_all_metric_data(recording=recording, n_comp=n_comp, ms_before=ms_before,
                                                      ms_after=ms_after, dtype=dtype,
                                                      max_spikes_per_unit=max_spikes_per_unit, amp_method=amp_method,
                                                      amp_peak=amp_peak,
                                                      amp_frames_before=amp_frames_before,
                                                      amp_frames_after=amp_frames_after,
                                                      recompute_info=recompute_info,
                                                      max_spikes_for_pca=max_spikes_for_pca,
                                                      apply_filter=apply_filter, freq_min=freq_min, freq_max=freq_max,
                                                      save_features_props=save_features_props, seed=seed)
    elif 'amplitude_cutoff' in metric_names:
        if recording is None:
            raise ValueError("The recording cannot be None when computing amplitude cutoffs.")
        else:
            metric_calculator.compute_amplitudes(recording=recording, amp_method=amp_method, amp_peak=amp_peak,
                                                 amp_frames_before=amp_frames_before,
                                                 amp_frames_after=amp_frames_after, apply_filter=apply_filter, 
                                                 freq_min=freq_min, freq_max=freq_max, 
                                                 save_features_props=save_features_props, seed=seed)
    elif 'snr' in metric_names:
        if recording is None:
            raise ValueError("The recording cannot be None when computing snr.")
        else:
            metric_calculator.set_recording(recording, apply_filter=apply_filter, freq_min=freq_min, freq_max=freq_max)

    if 'num_spikes' in metric_names:
        num_spikes_epochs = metric_calculator.compute_num_spikes()
        metrics_epochs.append(num_spikes_epochs)

    if 'firing_rate' in metric_names:
        firing_rates_epochs = metric_calculator.compute_firing_rates()
        metrics_epochs.append(firing_rates_epochs)

    if 'presence_ratio' in metric_names:
        presence_ratios_epochs = metric_calculator.compute_presence_ratios()
        metrics_epochs.append(presence_ratios_epochs)

    if 'isi_viol' in metric_names:
        isi_violations_epochs = metric_calculator.compute_isi_violations(isi_threshold=isi_threshold, min_isi=min_isi)
        metrics_epochs.append(isi_violations_epochs)

    if 'amplitude_cutoff' in metric_names:
        amplitude_cutoffs_epochs = metric_calculator.compute_amplitude_cutoffs()
        metrics_epochs.append(amplitude_cutoffs_epochs)

    if 'snr' in metric_names:
        snrs_epochs = metric_calculator.compute_snrs(snr_mode=snr_mode, snr_noise_duration=snr_noise_duration,
                                                     max_spikes_per_unit_for_snr=max_spikes_per_unit_for_snr)
        metrics_epochs.append(snrs_epochs)

    if 'max_drift' in metric_names or 'cumulative_drift' in metric_names:
        max_drifts_epochs, cumulative_drifts_epochs = metric_calculator.compute_drift_metrics(
            drift_metrics_interval_s=drift_metrics_interval_s,
            drift_metrics_min_spikes_per_interval=drift_metrics_min_spikes_per_interval)
        if 'max_drift' in metric_names:
            metrics_epochs.append(max_drifts_epochs)
        if 'cumulative_drift' in metric_names:
            metrics_epochs.append(cumulative_drifts_epochs)

    if 'silhouette_score' in metric_names:
        silhouette_scores_epochs = metric_calculator.compute_silhouette_scores(
            max_spikes_for_silhouette=max_spikes_for_silhouette, seed=seed)
        metrics_epochs.append(silhouette_scores_epochs)

    if 'isolation_distance' in metric_names:
        isolation_distances_epochs = metric_calculator.compute_isolation_distances(
            num_channels_to_compare=num_channels_to_compare, max_spikes_per_cluster=max_spikes_per_cluster,
            seed=seed)
        metrics_epochs.append(isolation_distances_epochs)

    if 'l_ratio' in metric_names:
        l_ratios_epochs = metric_calculator.compute_l_ratios(num_channels_to_compare=num_channels_to_compare,
                                                             max_spikes_per_cluster=max_spikes_per_cluster, seed=seed)
        metrics_epochs.append(l_ratios_epochs)

    if 'd_prime' in metric_names:
        d_primes_epochs = metric_calculator.compute_d_primes(num_channels_to_compare=num_channels_to_compare,
                                                             max_spikes_per_cluster=max_spikes_per_cluster, seed=seed)
        metrics_epochs.append(d_primes_epochs)

    if 'nn_hit_rate' in metric_names or 'nn_miss_rate' in metric_names:
        nn_hit_rates_epochs, nn_miss_rates_epochs = metric_calculator.compute_nn_metrics(
            num_channels_to_compare=num_channels_to_compare, max_spikes_per_cluster=max_spikes_per_cluster,
            max_spikes_for_nn=max_spikes_for_nn, n_neighbors=n_neighbors, seed=seed)
        if 'nn_hit_rate' in metric_names:
            metrics_epochs.append(nn_hit_rates_epochs)
        if 'nn_miss_rate' in metric_names:
            metrics_epochs.append(nn_miss_rates_epochs)

    if return_dataframe:
        metrics_df = metric_calculator.get_metrics_df()
        return metrics_df
    else:
        return metrics_epochs
