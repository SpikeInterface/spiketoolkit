from spiketoolkit.postprocessing.postprocessing_tools import _get_quality_metric_data, _get_pca_metric_data, \
    _get_spike_times_clusters, _get_amp_metric_data
import spikeextractors as se
import numpy as np


def get_spike_times_metrics_data(sorting, sampling_frequency):
    '''
    Computes and returns the spike times in seconds and also returns
    along with cluster_ids needed for quality metrics

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting extractor
    sampling_frequency: float
        The sampling frequency of the recording

    Returns
    -------
    spike_times: numpy.ndarray (num_spikes x 0)
        Spike times in seconds
    spike_clusters: numpy.ndarray (num_spikes x 0)
        Cluster IDs for each spike time
    '''
    if not isinstance(sorting, se.SortingExtractor):
        raise AttributeError()
    if len(sorting.get_unit_ids()) == 0:
        raise Exception("No units in the sorting result, can't compute any metric information.")

    # spike times.npy and spike clusters.npy
    spike_times, spike_clusters = _get_spike_times_clusters(sorting)

    spike_times = np.squeeze((spike_times / sampling_frequency))
    spike_clusters = np.squeeze(spike_clusters.astype(int))

    return spike_times, spike_clusters


def get_pca_metric_data(recording, sorting, **kwargs):
    '''
    Computes and returns all data needed to compute all the quality metrics from SpikeMetrics

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor
    sorting: SortingExtractor
        The sorting extractor
    n_comp: int
        n_compFeatures in template-gui format
    recompute_info: bool
        If True, will always re-extract waveforms
    save_property_or_features: bool
        If True, save all features and properties in the sorting extractor
    verbose: bool
        If True output is verbose
    **wf_args: Keyword arguments
        Keyword arguments for waveforms. A dictionary with default values can be retrieved with:
        st.postprocessing.get_waveforms_params():
            grouping_property: str
                Property to group channels. E.g. if the recording extractor has the 'group' property and
                'grouping_property' is 'group', then waveforms are computed group-wise.
            ms_before: float
                Time period in ms to cut waveforms before the spike events
            ms_after: float
                Time period in ms to cut waveforms after the spike events
            dtype: dtype
                The numpy dtype of the waveforms
            max_spikes_per_unit: int
                The maximum number of spikes to extract per unit.
            compute_property_from_recording: bool
                If True and 'grouping_property' is given, the property of each unit is assigned as the corresponding
                property of the recording extractor channel on which the average waveform is the largest
            seed: int
                Random seed for extracting random waveforms
            n_jobs: int
                Number of parallel jobs (default 1)
            memmap: bool
                If True, waveforms are saved as memmap object (recommended for long recordings with many channels)
            max_channels_per_waveforms: int or None
                Maximum channels per waveforms to return. If None, all channels are returned

    Returns
    -------
    spike_times: numpy.ndarray (num_spikes x 0)
        Spike times in seconds
    spike_clusters: numpy.ndarray (num_spikes x 0)
        Cluster IDs for each spike time
    pc_features: numpy.ndarray (num_spikes x num_pcs x num_channels)
        Pre-computed PCs for blocks of channels around each spike
    pc_feature_ind: numpy.ndarray (num_units x num_channels)
        Channel indices of PCs for each unit
    '''
    if not isinstance(recording, se.RecordingExtractor) or not isinstance(sorting, se.SortingExtractor):
        raise AttributeError()
    if len(sorting.get_unit_ids()) == 0:
        raise Exception("No units in the sorting result, can't compute any metric information.")

    spike_times, spike_times_pca, spike_clusters, \
    spike_clusters_pca, pc_features, pc_feature_ind = _get_pca_metric_data(recording, sorting, **kwargs)

    return np.squeeze(recording.frame_to_time(spike_times)), np.squeeze(recording.frame_to_time(spike_times_pca)),\
           np.squeeze(spike_clusters),  np.squeeze(spike_clusters_pca), pc_features, pc_feature_ind


def get_amplitude_metric_data(recording, sorting, **kwargs):
    '''
    Computes and returns all data needed to compute all the quality metrics from SpikeMetrics

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor
    sorting: SortingExtractor
        The sorting extractor
    **kwargs: Keyword arguments
        Keyword arguments for amplitudes. A dictionary with default values can be retrieved with:
        st.postprocessing.get_amplitude_params():
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
            max_spikes_per_unit: int
                The maximum number of amplitudes to extract for each unit(default is np.inf). If less than np.inf,
                the amplitudes will be returned from a random permutation of the spikes.
            recompute_info: bool
                If True, will always re-extract waveforms
            save_property_or_features: bool
                If True, save all features and properties in the sorting extractor
            seed: int
                    Random seed for reproducibility
            memmap: bool
                If True, amplitudes are saved as memmap object (recommended for long recordings with many channels)

    Returns
    -------
    spike_times: numpy.ndarray (num_spikes x 0)
        Spike times in seconds
    spike_clusters: numpy.ndarray (num_spikes x 0)
        Cluster IDs for each spike time
    amplitudes: numpy.ndarray (num_spikes x 0)
        Amplitude value for each spike time
    '''
    if not isinstance(recording, se.RecordingExtractor) or not isinstance(sorting, se.SortingExtractor):
        raise AttributeError()
    if len(sorting.get_unit_ids()) == 0:
        raise Exception("No units in the sorting result, can't compute any metric information.")

    spike_times, spike_times_amp, spike_clusters, \
    spike_clusters_amp, amplitudes = _get_amp_metric_data(recording, sorting, **kwargs)

    return np.squeeze(recording.frame_to_time(spike_times)), np.squeeze(recording.frame_to_time(spike_times_amp)),\
           np.squeeze(spike_clusters), np.squeeze(spike_clusters_amp), np.squeeze(amplitudes)
