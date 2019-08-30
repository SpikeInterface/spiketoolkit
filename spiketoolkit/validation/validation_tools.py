import spiketoolkit as st
import spikeextractors as se
import numpy as np


def get_firing_times_ids(sorting, sampling_frequency):
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
        Spike times in frames
    spike_clusters: numpy.ndarray (num_spikes x 0)
        Cluster IDs for each spike time
    '''
    if not isinstance(sorting, se.SortingExtractor):
        raise AttributeError()
    if len(sorting.get_unit_ids()) == 0:
        raise Exception("No units in the sorting result, can't compute any metric information.")

    # spike times.npy and spike clusters.npy
    spike_times = np.array([])
    spike_clusters = np.array([])

    for i_u, unit_id in enumerate(sorting.get_unit_ids()):
        spike_train = sorting.get_unit_spike_train(unit_id)
        cl = [i_u] * len(sorting.get_unit_spike_train(unit_id))
        spike_times = np.concatenate((spike_times, np.array(spike_train)))
        spike_clusters = np.concatenate((spike_clusters, np.array(cl)))

    sorting_idxs = np.argsort(spike_times)
    spike_times = spike_times[sorting_idxs, np.newaxis]
    spike_clusters = spike_clusters[sorting_idxs, np.newaxis]

    spike_times = (spike_times / sampling_frequency).flatten('F')
    spike_clusters = spike_clusters.astype(int).flatten('F')

    return spike_times, spike_clusters


def get_quality_metric_data(recording, sorting, nPC=3, ms_before=1., ms_after=2., dtype=None, amp_method='absolute',
                            amp_peak='both',
                            amp_frames_before=3, amp_frames_after=3, max_num_waveforms=np.inf,
                            max_num_pca_waveforms=np.inf, \
                            recompute_waveform_info=True, save_features_props=False, verbose=False, seed=0):
    '''
    Computes and returns all data needed to compute all the quality metrics from SpikeMetrics

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor
    sorting: SortingExtractor
        The sorting extractor
    nPC: int
        nPCFeatures in template-gui format
    ms_before: float
        Time period in ms to cut waveforms before the spike events
    ms_after: float
        Time period in ms to cut waveforms after the spike events
    dtype: dtype
        The numpy dtype of the waveforms
    amp_method: str
        If 'absolute' (default), amplitudes are absolute amplitudes in uV are returned.
        If 'relative', amplitudes are returned as ratios between waveform amplitudes and template amplitudes.
    amp_peak: str
        If maximum channel has to be found among negative peaks ('neg'), positive ('pos') or both ('both' - default)
    frames_before: int
        Frames before peak to compute amplitude
    frames_after: float
        Frames after peak to compute amplitude
    max_num_waveforms: int
        The maximum number of waveforms to extract (default is np.inf)
    max_num_pca_waveforms: int
        The maximum number of waveforms to use to compute PCA (default is np.inf)
    recompute_waveform_info: bool
        If True, will always re-extract waveforms.
    save_features_props: bool
        If True, save all features and properties in the sorting extractor.
    verbose: bool
        If True output is verbose
    seed: int
            Random seed for reproducibility

    Returns
    -------
    spike_times: numpy.ndarray (num_spikes x 0)
        Spike times in frames
    spike_clusters: numpy.ndarray (num_spikes x 0)
        Cluster IDs for each spike time
    amplitudes: numpy.ndarray (num_spikes x 0)
        Amplitude value for each spike time
    pc_features: numpy.ndarray (num_spikes x num_pcs x num_channels)
        Pre-computed PCs for blocks of channels around each spike
    pc_feature_ind: numpy.ndarray (num_units x num_channels)
        Channel indices of PCs for each unit
    '''
    if not isinstance(recording, se.RecordingExtractor) or not isinstance(sorting, se.SortingExtractor):
        raise AttributeError()
    if len(sorting.get_unit_ids()) == 0:
        raise Exception("No units in the sorting result, can't compute any metric information.")

    spike_times, spike_clusters, amplitudes, _, pc_features, pc_feature_ind \
        = st.postprocessing.postprocessing_tools._get_quality_metric_data(recording, sorting, nPC=nPC,
                                                                          ms_before=ms_before, ms_after=ms_after,
                                                                          dtype=dtype, amp_method=amp_method,
                                                                          amp_peak=amp_peak,
                                                                          amp_frames_before=amp_frames_before,
                                                                          amp_frames_after=amp_frames_after,
                                                                          max_num_waveforms=max_num_waveforms,
                                                                          max_num_pca_waveforms=max_num_pca_waveforms,
                                                                          recompute_waveform_info=recompute_waveform_info,
                                                                          save_features_props=save_features_props,
                                                                          verbose=verbose, seed=seed)
    return recording.frame_to_time(spike_times).flatten('F'), spike_clusters.astype(int).flatten('F'), \
           amplitudes.flatten('F'), pc_features, pc_feature_ind
