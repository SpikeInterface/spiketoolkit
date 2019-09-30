from ..postprocessing.postprocessing_tools import _get_quality_metric_data, _get_pca_metric_data, \
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
        Spike times in frames
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


def get_pca_metric_data(recording, sorting, n_comp=3, ms_before=1., ms_after=2., dtype=None, max_spikes_per_unit=np.inf,
                        max_spikes_for_pca=np.inf, recompute_info=True, save_features_props=False,
                        verbose=False, seed=0):
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
    recompute_info: bool
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
    pc_features: numpy.ndarray (num_spikes x num_pcs x num_channels)
        Pre-computed PCs for blocks of channels around each spike
    pc_feature_ind: numpy.ndarray (num_units x num_channels)
        Channel indices of PCs for each unit
    '''
    if not isinstance(recording, se.RecordingExtractor) or not isinstance(sorting, se.SortingExtractor):
        raise AttributeError()
    if len(sorting.get_unit_ids()) == 0:
        raise Exception("No units in the sorting result, can't compute any metric information.")

    spike_times, spike_clusters, pc_features, pc_feature_ind = _get_pca_metric_data(recording, sorting, n_comp=n_comp,
                                                                                    ms_before=ms_before,
                                                                                    ms_after=ms_after,
                                                                                    dtype=dtype,
                                                                                    max_spikes_per_unit=
                                                                                    max_spikes_per_unit,
                                                                                    max_spikes_for_pca=
                                                                                    max_spikes_for_pca,
                                                                                    recompute_info=recompute_info,
                                                                                    save_features_props=
                                                                                    save_features_props,
                                                                                    verbose=verbose, seed=seed)

    return np.squeeze(recording.frame_to_time(spike_times)), np.squeeze(spike_clusters), pc_features, pc_feature_ind


def get_amplitude_metric_data(recording, sorting, amp_method='absolute', amp_peak='both', amp_frames_before=3,
                              amp_frames_after=3, max_spikes_per_unit=np.inf, recompute_info=True,
                              save_features_props=False, seed=0):
    '''
    Computes and returns all data needed to compute all the quality metrics from SpikeMetrics

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor
    sorting: SortingExtractor
        The sorting extractor
    dtype: dtype
        The numpy dtype of the waveforms
    amp_method: str
        If 'absolute' (default), amplitudes are absolute amplitudes in uV are returned.
        If 'relative', amplitudes are returned as ratios between waveform amplitudes and template amplitudes.
    amp_peak: str
        If maximum channel has to be found among negative peaks ('neg'), positive ('pos') or both ('both' - default)
    amp_frames_before: int
        Frames before peak to compute amplitude
    amp_frames_after: int
        Frames after peak to compute amplitude
    max_spikes_per_unit: int
        The maximum number of spikes to extract per unit.
    recompute_info: bool
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
    '''
    if not isinstance(recording, se.RecordingExtractor) or not isinstance(sorting, se.SortingExtractor):
        raise AttributeError()
    if len(sorting.get_unit_ids()) == 0:
        raise Exception("No units in the sorting result, can't compute any metric information.")

    spike_times, spike_clusters, amplitudes = _get_amp_metric_data(recording, sorting,
                                                                   amp_method=amp_method,
                                                                   amp_peak=amp_peak,
                                                                   amp_frames_before=amp_frames_before,
                                                                   amp_frames_after=amp_frames_after,
                                                                   max_spikes_per_unit=max_spikes_per_unit,
                                                                   save_features_props=save_features_props,
                                                                   recompute_info=recompute_info,
                                                                   seed=seed)

    return np.squeeze(recording.frame_to_time(spike_times)), np.squeeze(spike_clusters), np.squeeze(amplitudes)


def get_all_metric_data(recording, sorting, n_comp=3, ms_before=1., ms_after=2., dtype=None, amp_method='absolute',
                        amp_peak='both', amp_frames_before=3, amp_frames_after=3, max_spikes_per_unit=np.inf,
                        max_spikes_for_pca=np.inf, recompute_info=True, save_features_props=False,
                        verbose=False, seed=0):
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
    amp_frames_before: int
        Frames before peak to compute amplitude
    amp_frames_after: int
        Frames after peak to compute amplitude
    max_spikes_per_unit: int
        The maximum number of spikes to extract per unit.
    max_spikes_for_pca: int
        The maximum number of spikes to use to compute PCA.
    recompute_info: bool
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
    spike_times:amps: numpy.ndarray (num_spikes x 0)
        Spike times in frames for amplitudes
    spike_times_pca: numpy.ndarray (num_spikes x 0)
        Spike times in frames for pca
    spike_clusters: numpy.ndarray (num_spikes x 0)
        Cluster IDs for each spike time
    spike_clusters_amps: numpy.ndarray (num_spikes x 0)
        Cluster IDs for each spike time in amplitudes
    spike_clusters_pca: numpy.ndarray (num_spikes x 0)
        Cluster IDs for each spike time in pca
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

    spike_times, spike_times_amps, spike_times_pca, spike_clusters, spike_clusters_amps, spike_clusters_pca, \
    amplitudes, pc_features, pc_feature_ind = _get_quality_metric_data(
        recording, sorting, n_comp=n_comp,
        ms_before=ms_before, ms_after=ms_after,
        dtype=dtype, amp_method=amp_method,
        amp_peak=amp_peak,
        amp_frames_before=amp_frames_before,
        amp_frames_after=amp_frames_after,
        max_spikes_per_unit=max_spikes_per_unit,
        max_spikes_for_pca=max_spikes_for_pca,
        recompute_info=recompute_info,
        save_features_props=save_features_props,
        verbose=verbose, seed=seed)

    return np.squeeze(recording.frame_to_time(spike_times)), np.squeeze(recording.frame_to_time(spike_times_amps)), \
           np.squeeze(recording.frame_to_time(spike_times_pca)), np.squeeze(spike_clusters), \
           np.squeeze(spike_clusters_amps), np.squeeze(spike_clusters_pca), \
           np.squeeze(amplitudes), pc_features, pc_feature_ind
