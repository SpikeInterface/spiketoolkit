import numpy as np
from spikeextractors import RecordingExtractor, SortingExtractor
import spiketoolkit as st
import spikemetrics.metrics as metrics
from spikemetrics.utils import Epoch, printProgressBar
import pandas as pd
from collections import defaultdict, OrderedDict
from .validation_tools import get_all_metric_data, get_pca_metric_data, get_amplitude_metric_data, \
    get_spike_times_metrics_data
import copy
from spiketoolkit.curation.thresholdcurator import ThresholdCurator


class MetricCalculator:
    def __init__(self, sorting, recording=None, sampling_frequency=None, apply_filter=True, freq_min=300, freq_max=6000, 
                 unit_ids=None, epoch_tuples=None, epoch_names=None, verbose=False):
        '''
        Computes and stores inital data along with the unit ids and epochs to be used for computing metrics.

        Parameters
        ----------
        sorting: SortingExtractor
            The sorting extractor to be evaluated.
        recording: RecordingExtractor
            The recording extractor to be stored. If None, the recording extractor can be added later.
        sampling_frequency:
            The sampling frequency of the result. If None, will check to see if sampling frequency is in sorting extractor.
        apply_filter: bool
            If True, recording is bandpass-filtered.
        freq_min: float
            High-pass frequency for optional filter (default 300 Hz).
        freq_max: float
            Low-pass frequency for optional filter (default 6000 Hz).
        unit_ids: list
            List of unit ids to compute metric for. If not specified, all units are used
        epoch_tuples: list
            A list of tuples with a start and end time for each epoch
        epoch_names: list
            A list of strings for the names of the given epochs.
        verbose: bool
            If True, progress bar is displayed
        '''
        if sampling_frequency is None and sorting.get_sampling_frequency() is None:
            raise ValueError("Please pass in a sampling frequency (your SortingExtractor does not have one specified)")
        elif sampling_frequency is None:
            self._sampling_frequency = sorting.get_sampling_frequency()
        else:
            self._sampling_frequency = sampling_frequency

        # only use units with spikes to avoid breaking metric calculation
        num_spikes_per_unit = [len(sorting.get_unit_spike_train(unit_id)) for unit_id in sorting.get_unit_ids()]
        sorting = ThresholdCurator(sorting=sorting, metrics_epoch=num_spikes_per_unit)
        sorting.threshold_sorting(0, 'less_or_equal')

        if unit_ids is None:
            unit_ids = sorting.get_unit_ids()
        else:
            unit_ids = set(unit_ids)
            unit_ids = list(unit_ids.intersection(sorting.get_unit_ids()))

        if len(unit_ids) == 0:
            raise ValueError("No units found.")

        spike_times, spike_clusters = get_spike_times_metrics_data(sorting, self._sampling_frequency)
        assert isinstance(sorting, SortingExtractor), "'sorting' must be  a SortingExtractor object"
        self._sorting = sorting
        self._set_unit_ids(unit_ids)
        self._set_epochs(epoch_tuples, epoch_names)
        self._spike_times = spike_times
        self._spike_clusters = spike_clusters
        self._total_units = len(unit_ids)
        self._unit_indices = _get_unit_indices(self._sorting, unit_ids)
        # To compute this data, need to call all metric data
        self._amplitudes = None
        self._pc_features = None
        self._pc_feature_ind = None
        self._spike_clusters_pca = None
        self._spike_clusters_amps = None
        self._spike_times_pca = None
        self._spike_times_amps = None
        # Dictionary of cached metrics
        self.metrics = {}
        self.verbose = verbose

        for epoch in self._epochs:
            self._sorting.add_epoch(epoch_name=epoch[0], start_frame=epoch[1] * self._sampling_frequency,
                                    end_frame=epoch[2] * self._sampling_frequency)
        if recording is not None:
            assert isinstance(recording, RecordingExtractor), "'sorting' must be  a RecordingExtractor object"
            self.set_recording(recording, apply_filter=apply_filter, freq_min=freq_min, freq_max=freq_max)
        else:
            self._recording = None

    def set_recording(self, recording, apply_filter=True, freq_min=300, freq_max=6000):
        '''
        Sets given recording extractor

        Parameters
        ----------
        recording: RecordingExtractor
            The recording extractor to be stored.
        apply_filter: bool
            If True, recording is bandpass-filtered.
        freq_min: float
            High-pass frequency for optional filter (default 300 Hz).
        freq_max: float
            Low-pass frequency for optional filter (default 6000 Hz).
        '''
        if apply_filter:
            self._is_filtered = True
            recording = st.preprocessing.bandpass_filter(recording=recording, freq_min=freq_min, freq_max=freq_max, cache_to_file=True)
        else:
            self._is_filtered = False
        self._recording = recording
        for epoch in self._epochs:
            self._recording.add_epoch(epoch_name=epoch[0], start_frame=epoch[1] * self._sampling_frequency,
                                      end_frame=epoch[2] * self._sampling_frequency)
    
    def is_filtered(self):
        return self._is_filtered

    def compute_amplitudes(self, recording=None, amp_method='absolute', amp_peak='both', amp_frames_before=3, amp_frames_after=3, 
                           apply_filter=True, freq_min=300, freq_max=6000, save_features_props=False, seed=0):
        '''
        Computes and stores amplitudes for the amplitude cutoff metric

        Parameters
        ----------
        recording: RecordingExtractor
            The given recording extractor from which to extract amplitudes.
        amp_method: str
            If 'absolute' (default), amplitudes are absolute amplitudes in uV are returned.
            If 'relative', amplitudes are returned as ratios between waveform amplitudes and template amplitudes.
        amp_peak: str
            If maximum channel has to be found among negative peaks ('neg'), positive ('pos') or both ('both' - default)
        amp_frames_before: int
            Frames before peak to compute amplitude
        amp_frames_after: int
            Frames after peak to compute amplitude
        apply_filter: bool
            If True, recording is bandpass-filtered.
        freq_min: float
            High-pass frequency for optional filter (default 300 Hz).
        freq_max: float
            Low-pass frequency for optional filter (default 6000 Hz).
        save_features_props: bool
            If true, it will save amplitudes in the sorting extractor.
        seed: int
            Random seed for reproducibility
        '''
        if recording is None:
            if self._recording is None:
                raise ValueError(
                    "No recording given. Either call store_recording or pass a recording into this function")
        else:
            self.set_recording(recording, apply_filter=apply_filter, freq_min=freq_min, freq_max=freq_max)

        spike_times, spike_clusters, amplitudes = get_amplitude_metric_data(self._recording, self._sorting,
                                                                            amp_method=amp_method,
                                                                            save_features_props=save_features_props,
                                                                            amp_peak=amp_peak,
                                                                            amp_frames_before=amp_frames_before,
                                                                            amp_frames_after=amp_frames_after,
                                                                            seed=seed)
        self._amplitudes = amplitudes
        self._spike_clusters_amps = spike_clusters
        self._spike_times_amps = spike_times

    def compute_pca_scores(self, recording=None, n_comp=3, ms_before=1., ms_after=2., dtype=None,
                           max_spikes_per_unit=300, recompute_info=True, max_spikes_for_pca=1e5, 
                           apply_filter=True, freq_min=300, freq_max=6000, save_features_props=False, seed=0):
        '''
        Computes and stores pca for the metrics computation

        Parameters
        ----------
        recording: RecordingExtractor
            The recording extractor
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
        recompute_info: bool
            If True, will always re-extract waveforms.
        max_spikes_for_pca: int
            The maximum number of spikes to use to compute PCA.
        apply_filter: bool
            If True, recording is bandpass-filtered.
        freq_min: float
            High-pass frequency for optional filter (default 300 Hz).
        freq_max: float
            Low-pass frequency for optional filter (default 6000 Hz).
        save_features_props: bool
            If true, it will save amplitudes in the sorting extractor.
        seed: int
            Random seed for reproducibility
        '''
        if recording is None:
            if self._recording is None:
                raise ValueError(
                    "No recording given. Either call store_recording or pass a recording into this function")
        else:
            self.set_recording(recording, apply_filter=apply_filter, freq_min=freq_min, freq_max=freq_max)

        spike_times, spike_clusters, pc_features, pc_feature_ind = get_pca_metric_data(self._recording, self._sorting,
                                                                                       n_comp=n_comp,
                                                                                       ms_before=ms_before,
                                                                                       ms_after=ms_after, dtype=dtype,
                                                                                       max_spikes_per_unit=
                                                                                       max_spikes_per_unit,
                                                                                       max_spikes_for_pca=
                                                                                       max_spikes_for_pca,
                                                                                       recompute_info=
                                                                                       recompute_info,
                                                                                       save_features_props=
                                                                                       save_features_props,
                                                                                       seed=seed)
        self._pc_features = pc_features
        self._spike_clusters_pca = spike_clusters
        self._spike_times_pca = spike_times
        self._pc_feature_ind = pc_feature_ind

    def compute_all_metric_data(self, recording=None, n_comp=3, ms_before=1., ms_after=2., dtype=None,
                                max_spikes_per_unit=300, amp_method='absolute', amp_peak='both',
                                amp_frames_before=3, amp_frames_after=3, recompute_info=True,
                                max_spikes_for_pca=1e5, apply_filter=True, freq_min=300, freq_max=6000,
                                save_features_props=False, seed=0):
        '''
        Computes and stores data for all metrics (all metrics can be run after calling this function).

        Parameters
        ----------
        recording: RecordingExtractor
            The recording extractor
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
        amp_method: str
            If 'absolute' (default), amplitudes are absolute amplitudes in uV are returned.
            If 'relative', amplitudes are returned as ratios between waveform amplitudes and template amplitudes.
        amp_peak: str
            If maximum channel has to be found among negative peaks ('neg'), positive ('pos') or both ('both' - default)
        amp_frames_before: int
            Frames before peak to compute amplitude
        amp_frames_after: int
            Frames after peak to compute amplitude
        recompute_info: bool
            If True, will always re-extract waveforms.
        max_spikes_for_pca: int
            The maximum number of spikes to use to compute PCA.
        apply_filter: bool
            If True, recording is bandpass-filtered.
        freq_min: float
            High-pass frequency for optional filter (default 300 Hz).
        freq_max: float
            Low-pass frequency for optional filter (default 6000 Hz).
        save_features_props: bool
            If True, save all features and properties in the sorting extractor.
        seed: int
            Random seed for reproducibility
        '''
        if recording is None:
            if self._recording is None:
                raise ValueError(
                    "No recording given. Either call store_recording or pass a recording into this function")
        else:
            self.set_recording(recording, apply_filter=apply_filter, freq_min=freq_min, freq_max=freq_max)

        spike_times, spike_times_amps, spike_times_pca, spike_clusters, spike_clusters_amps, spike_clusters_pca, \
        amplitudes, pc_features, pc_feature_ind = get_all_metric_data(
            self._recording, self._sorting, n_comp=n_comp, ms_before=ms_before,
            ms_after=ms_after, dtype=dtype, amp_method=amp_method,
            amp_peak=amp_peak, amp_frames_before=amp_frames_before,
            amp_frames_after=amp_frames_after, max_spikes_per_unit=max_spikes_per_unit,
            max_spikes_for_pca=max_spikes_for_pca, recompute_info=recompute_info,
            save_features_props=save_features_props, seed=seed)

        self._amplitudes = amplitudes
        self._spike_clusters_amps = spike_clusters_amps
        self._spike_times_amps = spike_times_amps
        self._pc_features = pc_features
        self._spike_clusters_pca = spike_clusters_pca
        self._spike_times_pca = spike_times_pca
        self._pc_feature_ind = pc_feature_ind

    def set_amplitudes(self, amplitudes):
        self._amplitudes = amplitudes

    def set_pc_features(self, pc_features):
        self._pc_features = pc_features

    def set_pc_feature_ind(self, pc_feature_ind):
        self._pc_feature_ind = pc_feature_ind

    def _set_epochs(self, epoch_tuples, epoch_names):
        if epoch_tuples is None:
            if epoch_names is None:
                epochs = [("complete_session", 0, np.inf)]
            else:
                raise ValueError("No epoch tuples specified, but names given.")
        else:
            if epoch_names is None:
                epochs = []
                for i, epoch_tuple in enumerate(epoch_tuples):
                    epoch = (str(i), epoch_tuple[0], epoch_tuple[1])
                    epochs.append(epoch)
            else:
                assert len(epoch_names) == len(epoch_tuples), "Make sure the name and epoch lists are equal in length,"
                epochs = []
                for i, epoch_tuple in enumerate(epoch_tuples):
                    epoch = (str(epoch_names[i]), epoch_tuple[0], epoch_tuple[1])
                    epochs.append(epoch)
        self._epochs = epochs

    def _set_unit_ids(self, unit_ids):
        self._unit_ids = unit_ids

    def get_epochs(self):
        return self._epochs

    def get_unit_ids(self):
        return self._unit_ids

    def compute_num_spikes(self):
        '''
        Computes and returns the spike times in seconds and also returns
        the cluster_ids needed for quality metrics.

        Returns
        ----------
        num_spikes_epochs: list
            The spike counts of the sorted units in the given epochs.
        '''
        num_spikes_epochs = []
        for epoch in self._epochs:
            in_epoch = np.logical_and(self._spike_times > epoch[1], self._spike_times < epoch[2])
            _, num_spikes_all = metrics.calculate_firing_rate_and_spikes(self._spike_times[in_epoch],
                                                                         self._spike_clusters[in_epoch],
                                                                         self._total_units, verbose=self.verbose)
            num_spikes_list = []
            for i in self._unit_indices:
                num_spikes_list.append(num_spikes_all[i])
            num_spikes = np.asarray(num_spikes_list)
            num_spikes_epochs.append(num_spikes)

        self.metrics['num_spikes'] = []
        for i, epoch in enumerate(self._epochs):
            self.metrics['num_spikes'].append(num_spikes_epochs[i])
        if len(num_spikes_epochs) == 1:
            num_spikes_epochs = num_spikes_epochs[0]
        return num_spikes_epochs

    def compute_firing_rates(self):
        '''
        Computes and returns the spike times in seconds and also returns
        the cluster_ids needed for quality metrics.

        Returns
        ----------
        firing_rates_epochs: list
            The firing rates of the sorted units in the given epochs.
        '''
        firings_rates_epochs = []
        for epoch in self._epochs:
            in_epoch = np.logical_and(self._spike_times > epoch[1], self._spike_times < epoch[2])
            firing_rates_all, _ = metrics.calculate_firing_rate_and_spikes(self._spike_times[in_epoch],
                                                                           self._spike_clusters[in_epoch],
                                                                           self._total_units, verbose=self.verbose)
            firing_rates_list = []
            for i in self._unit_indices:
                firing_rates_list.append(firing_rates_all[i])
            firing_rates = np.asarray(firing_rates_list)
            firings_rates_epochs.append(firing_rates)

        self.metrics['firing_rate'] = []
        for i, epoch in enumerate(self._epochs):
            self.metrics['firing_rate'].append(firings_rates_epochs[i])
        if len(firings_rates_epochs) == 1:
            firings_rates_epochs = firings_rates_epochs[0]
        return firings_rates_epochs

    def compute_presence_ratios(self):
        '''
        Computes and returns the presence ratios.

        Returns
        ----------
        presence_ratios_epochs: list
            The presence ratios violations of the sorted units in the given epochs.
        '''
        presence_ratios_epochs = []
        for epoch in self._epochs:
            in_epoch = np.logical_and(self._spike_times > epoch[1], self._spike_times < epoch[2])
            presence_ratios_all = metrics.calculate_presence_ratio(self._spike_times[in_epoch],
                                                                   self._spike_clusters[in_epoch], self._total_units,
                                                                   verbose=self.verbose)
            presence_ratios_list = []
            for i in self._unit_indices:
                presence_ratios_list.append(presence_ratios_all[i])
            presence_ratios = np.asarray(presence_ratios_list)
            presence_ratios_epochs.append(presence_ratios)

        self.metrics['presence_ratio'] = []
        for i, epoch in enumerate(self._epochs):
            self.metrics['presence_ratio'].append(presence_ratios_epochs[i])
        if len(presence_ratios_epochs) == 1:
            presence_ratios_epochs = presence_ratios_epochs[0]
        return presence_ratios_epochs

    def compute_isi_violations(self, isi_threshold=0.0015, min_isi=0.000166):
        '''
        Computes and returns the ISI violations for the given parameters.

        Parameters
        ----------
        isi_threshold: float
            The isi threshold for calculating isi violations.
        min_isi: float
            The minimum expected isi value.

        Returns
        ----------
        isi_violations_epochs: list
            The isi violations of the sorted units in the given epochs.
        '''
        isi_violations_epochs = []
        for epoch in self._epochs:
            in_epoch = np.logical_and(self._spike_times > epoch[1], self._spike_times < epoch[2])
            isi_violations_all = metrics.calculate_isi_violations(self._spike_times[in_epoch],
                                                                  self._spike_clusters[in_epoch], self._total_units,
                                                                  isi_threshold=isi_threshold, min_isi=min_isi,
                                                                  verbose=self.verbose)
            isi_violations_list = []
            for i in self._unit_indices:
                isi_violations_list.append(isi_violations_all[i])
            isi_violations = np.asarray(isi_violations_list)
            isi_violations_epochs.append(isi_violations)

        self.metrics['isi_viol'] = []
        for i, epoch in enumerate(self._epochs):
            self.metrics['isi_viol'].append(isi_violations_epochs[i])
        if len(isi_violations_epochs) == 1:
            isi_violations_epochs = isi_violations_epochs[0]
        return isi_violations_epochs

    def compute_amplitude_cutoffs(self):
        '''
        Computes and returns the amplitude cutoffs for the sorted dataset.

        Returns
        ----------
        amplitude_cutoffs_epochs: list
            The amplitude cutoffs of the sorted units in the given epochs.
        '''
        if self._amplitudes is None:
            assert self._recording is not None, "No recording stored. Add a recording first with set_recording"
            self.compute_amplitudes(self._recording)

        amplitude_cutoffs_epochs = []
        for epoch in self._epochs:
            in_epoch = np.logical_and(self._spike_times_amps > epoch[1], self._spike_times_amps < epoch[2])
            amplitude_cutoffs_all = metrics.calculate_amplitude_cutoff(self._spike_clusters_amps[in_epoch],
                                                                       self._amplitudes[in_epoch],
                                                                       self._total_units, verbose=self.verbose)
            amplitude_cutoffs_list = []
            for i in self._unit_indices:
                amplitude_cutoffs_list.append(amplitude_cutoffs_all[i])
            amplitude_cutoffs = np.asarray(amplitude_cutoffs_list)
            amplitude_cutoffs_epochs.append(amplitude_cutoffs)

        self.metrics['amplitude_cutoff'] = []
        for i, epoch in enumerate(self._epochs):
            self.metrics['amplitude_cutoff'].append(amplitude_cutoffs_epochs[i])
        if len(amplitude_cutoffs_epochs) == 1:
            amplitude_cutoffs_epochs = amplitude_cutoffs_epochs[0]
        return amplitude_cutoffs_epochs

    def compute_snrs(self, snr_mode='mad', snr_noise_duration=10.0, max_spikes_per_unit_for_snr=1000,
                     recompute_info=True, save_features_props=False, seed=0):
        '''
        Computes signal-to-noise ratio (SNR) of the average waveforms on the largest channel for sorted dataset.

        Parameters
        ----------
        snr_mode: str
            Mode to compute noise SNR ('mad' | 'std' - default 'mad')
        snr_noise_duration: float
            Number of seconds to compute noise level from (default 10.0)
        max_spikes_per_unit_for_snr: int
            Maximum number of spikes to compute templates from (default 1000)
        recompute_info: bool
            If True, waveforms are recomputed
        save_features_props: bool
            If True, waveforms and templates are saved as sorting features/properties
        seed: int
            Random seed for reproducibility

        Returns
        -------
        snrs_epochs: list
            The snrs of the sorted units in the given epochs.
        '''
        assert self._recording is not None, "No recording stored. Add a recording first with set_recording"

        snrs_epochs = []
        for epoch in self._epochs:
            epoch_recording = self._recording.get_epoch(epoch[0])
            epoch_sorting = self._sorting.get_epoch(epoch[0])
            channel_noise_levels = _compute_channel_noise_levels(recording=epoch_recording, mode=snr_mode,
                                                                 noise_duration=snr_noise_duration)
            templates = st.postprocessing.get_unit_templates(epoch_recording, epoch_sorting, unit_ids=self._unit_ids,
                                                             max_spikes_per_unit=max_spikes_per_unit_for_snr,
                                                             mode='median',
                                                             save_wf_as_features=save_features_props,
                                                             recompute_waveforms=recompute_info,
                                                             save_as_property=save_features_props, seed=seed)

            max_channels = st.postprocessing.get_unit_max_channels(epoch_recording, epoch_sorting,
                                                                   unit_ids=self._unit_ids,
                                                                   max_spikes_per_unit=max_spikes_per_unit_for_snr,
                                                                   peak='both',
                                                                   recompute_templates=recompute_info,
                                                                   save_as_property=save_features_props,
                                                                   mode='median', seed=seed)
            snr_list = []
            for i, unit_id in enumerate(self._unit_ids):
                if self.verbose:
                    printProgressBar(i + 1, len(self._unit_ids))
                max_channel_idx = epoch_recording.get_channel_ids().index(max_channels[i])
                snr = _compute_template_SNR(templates[i], channel_noise_levels, max_channel_idx)
                snr_list.append(snr)
            snrs = np.asarray(snr_list)
            snrs_epochs.append(snrs)
        self.metrics['snr'] = []
        for i, epoch in enumerate(self._epochs):
            self.metrics['snr'].append(snrs_epochs[i])
        if len(snrs_epochs) == 1:
            snrs_epochs = snrs_epochs[0]
        return snrs_epochs

    def compute_drift_metrics(self, drift_metrics_interval_s=51, drift_metrics_min_spikes_per_interval=10):
        '''
        Computes and returns the drift metrics for the sorted dataset.

        Parameters
        ----------
        drift_metrics_interval_s: float
            Time period for evaluating drift.
        drift_metrics_min_spikes_per_interval: int
            Minimum number of spikes for evaluating drift metrics per interval.

        Returns
        ----------
        max_drifts_epochs: list
            The max drift of the given units over the specified epochs
        cumulative_drifts_epochs: list
            The cumulative drifts of the given units over the specified epochs
        '''
        if self._pc_features is None:
            assert self._recording is not None, "No recording stored. Add a recording first with set_recording"
            self.compute_pca_scores(self._recording)

        max_drifts_epochs = []
        cumulative_drifts_epochs = []
        for epoch in self._epochs:
            in_epoch = np.logical_and(self._spike_times_pca > epoch[1], self._spike_times_pca < epoch[2])
            max_drifts_all, cumulative_drifts_all = metrics.calculate_drift_metrics(self._spike_times_pca[in_epoch],
                                                                                    self._spike_clusters_pca[in_epoch],
                                                                                    self._total_units,
                                                                                    self._pc_features[in_epoch, :, :],
                                                                                    self._pc_feature_ind,
                                                                                    drift_metrics_interval_s,
                                                                                    drift_metrics_min_spikes_per_interval,
                                                                                    verbose=self.verbose)
            max_drifts_list = []
            cumulative_drifts_list = []
            for i in self._unit_indices:
                max_drifts_list.append(max_drifts_all[i])
                cumulative_drifts_list.append(cumulative_drifts_all[i])

            max_drifts = np.asarray(max_drifts_list)
            cumulative_drifts = np.asarray(cumulative_drifts_list)
            max_drifts_epochs.append(max_drifts)
            cumulative_drifts_epochs.append(cumulative_drifts)

        self.metrics['max_drift'] = []
        self.metrics['cumulative_drift'] = []
        for i, epoch in enumerate(self._epochs):
            self.metrics['max_drift'].append(max_drifts_epochs[i])
            self.metrics['cumulative_drift'].append(cumulative_drifts_epochs[i])
        if len(max_drifts_epochs) == 1:
            max_drifts_epochs = max_drifts_epochs[0]
            cumulative_drifts_epochs = cumulative_drifts_epochs[0]
        return max_drifts_epochs, cumulative_drifts_epochs

    def compute_silhouette_scores(self, max_spikes_for_silhouette=10000, seed=0):
        '''
        Computes and returns the silhouette scores in the sorted dataset.

        Parameters
        ----------
        max_spikes_for_silhouette: int
            Max spikes to be used for silhouette metric
        seed: int
            A random seed for reproducibility

        Returns
        ----------
        silhouette_scores_epochs: list
            The silhouette scores of the given units for the specified epochs.
        '''
        if self._pc_features is None:
            assert self._recording is not None, "No recording stored. Add a recording first with set_recording"
            self.compute_pca_scores(self._recording)

        silhouette_scores_epochs = []
        for epoch in self._epochs:
            in_epoch = np.logical_and(self._spike_times_pca > epoch[1], self._spike_times_pca < epoch[2])
            spikes_in_epoch = np.sum(in_epoch)
            spikes_for_silhouette = np.min([spikes_in_epoch, max_spikes_for_silhouette])

            silhouette_scores_all = metrics.calculate_silhouette_score(self._spike_clusters_pca[in_epoch],
                                                                       self._total_units,
                                                                       self._pc_features[in_epoch, :, :],
                                                                       self._pc_feature_ind,
                                                                       spikes_for_silhouette,
                                                                       seed=seed, verbose=self.verbose)
            silhouette_scores_list = []
            for i in self._unit_indices:
                silhouette_scores_list.append(silhouette_scores_all[i])
            silhouette_scores = np.asarray(silhouette_scores_list)
            silhouette_scores_epochs.append(silhouette_scores)

        self.metrics['silhouette_score'] = []
        for i, epoch in enumerate(self._epochs):
            self.metrics['silhouette_score'].append(silhouette_scores_epochs[i])
        if len(silhouette_scores_epochs) == 1:
            silhouette_scores_epochs = silhouette_scores_epochs[0]
        return silhouette_scores_epochs

    def compute_isolation_distances(self, num_channels_to_compare=13, max_spikes_per_cluster=500, seed=0):
        '''
        Computes and returns the mahalanobis metric, isolation distance, for the sorted dataset.

        Parameters
        ----------
        num_channels_to_compare: int
            The number of channels to be used for the PC extraction and comparison.
        max_spikes_per_cluster: int
            Max spikes to be used from each unit to compute cluster-based metrics.
        seed: int
            Random seed for extracting pc features.

        Returns
        ----------
        isolation_distances_epochs: list
            Returns the isolation distances of each specified unit for the given epochs.
        '''
        if self._pc_features is None:
            assert self._recording is not None, "No recording stored. Add a recording first with set_recording"
            self.compute_pca_scores(self._recording)

        isolation_distances_epochs = []
        for epoch in self._epochs:
            in_epoch = np.logical_and(self._spike_times_pca > epoch[1], self._spike_times_pca < epoch[2])
            isolation_distances_all = metrics.calculate_pc_metrics(spike_clusters=self._spike_clusters_pca[in_epoch],
                                                                   total_units=self._total_units,
                                                                   pc_features=self._pc_features[in_epoch, :, :],
                                                                   pc_feature_ind=self._pc_feature_ind,
                                                                   num_channels_to_compare=num_channels_to_compare,
                                                                   max_spikes_for_cluster=max_spikes_per_cluster,
                                                                   spikes_for_nn=None,
                                                                   n_neighbors=None,
                                                                   metric_names=['isolation_distance'],
                                                                   seed=seed, verbose=self.verbose)[0]
            isolation_distances_list = []
            for i in self._unit_indices:
                isolation_distances_list.append(isolation_distances_all[i])
            isolation_distances = np.asarray(isolation_distances_list)
            isolation_distances_epochs.append(isolation_distances)

        self.metrics['isolation_distance'] = []
        for i, epoch in enumerate(self._epochs):
            self.metrics['isolation_distance'].append(isolation_distances_epochs[i])
        if len(isolation_distances_epochs) == 1:
            isolation_distances_epochs = isolation_distances_epochs[0]
        return isolation_distances_epochs

    def compute_l_ratios(self, num_channels_to_compare=13, max_spikes_per_cluster=500, seed=0):
        '''
        Computes and returns the mahalanobis metric, l-ratio, for the sorted dataset.

        Parameters
        ----------
        num_channels_to_compare: int
            The number of channels to be used for the PC extraction and comparison.
        max_spikes_per_cluster: int
            Max spikes to be used from each unit to compute cluster-based metrics.
        seed: int
            Random seed for extracting pc features.

        Returns
        ----------
        l_ratios_epochs: list
            Returns the L ratios of each specified unit for the given epochs
        '''
        if self._pc_features is None:
            assert self._recording is not None, "No recording stored. Add a recording first with set_recording"
            self.compute_pca_scores(self._recording)

        l_ratios_epochs = []
        for epoch in self._epochs:
            in_epoch = np.logical_and(self._spike_times_pca > epoch[1], self._spike_times_pca < epoch[2])
            l_ratios_all = metrics.calculate_pc_metrics(spike_clusters=self._spike_clusters_pca[in_epoch],
                                                        total_units=self._total_units,
                                                        pc_features=self._pc_features[in_epoch, :, :],
                                                        pc_feature_ind=self._pc_feature_ind,
                                                        num_channels_to_compare=num_channels_to_compare,
                                                        max_spikes_for_cluster=max_spikes_per_cluster,
                                                        spikes_for_nn=None,
                                                        n_neighbors=None,
                                                        metric_names=['l_ratio'],
                                                        seed=seed, verbose=self.verbose)[1]
            l_ratios_list = []
            for i in self._unit_indices:
                l_ratios_list.append(l_ratios_all[i])
            l_ratios = np.asarray(l_ratios_list)
            l_ratios_epochs.append(l_ratios)

        self.metrics['l_ratio'] = []
        for i, epoch in enumerate(self._epochs):
            self.metrics['l_ratio'].append(l_ratios_epochs[i])
        if len(l_ratios_epochs) == 1:
            l_ratios_epochs = l_ratios_epochs[0]
        return l_ratios_epochs

    def compute_d_primes(self, num_channels_to_compare=13, max_spikes_per_cluster=500, seed=0):
        '''
        Computes and returns the lda-based metric, d-prime, for the sorted dataset.

        Parameters
        ----------
        num_channels_to_compare: int
            The number of channels to be used for the PC extraction and comparison.
        max_spikes_per_cluster: int
            Max spikes to be used from each unit to compute cluster-based metrics.
        seed: int
            Random seed for extracting pc features.

        Returns
        ----------
        d_primes_epochs: list
            Returns the d primes of each specified unit for the given epochs.
        '''
        if self._pc_features is None:
            assert self._recording is not None, "No recording stored. Add a recording first with set_recording"
            self.compute_pca_scores(self._recording)

        d_primes_epochs = []
        for epoch in self._epochs:
            in_epoch = np.logical_and(self._spike_times_pca > epoch[1], self._spike_times_pca < epoch[2])
            d_primes_all = metrics.calculate_pc_metrics(spike_clusters=self._spike_clusters_pca[in_epoch],
                                                        total_units=self._total_units,
                                                        pc_features=self._pc_features[in_epoch, :, :],
                                                        pc_feature_ind=self._pc_feature_ind,
                                                        num_channels_to_compare=num_channels_to_compare,
                                                        max_spikes_for_cluster=max_spikes_per_cluster,
                                                        spikes_for_nn=None,
                                                        n_neighbors=None,
                                                        metric_names=['d_prime'],
                                                        seed=seed, verbose=self.verbose)[2]
            d_primes_list = []
            for i in self._unit_indices:
                d_primes_list.append(d_primes_all[i])
            d_primes = np.asarray(d_primes_list)
            d_primes_epochs.append(d_primes)

        self.metrics['d_prime'] = []
        for i, epoch in enumerate(self._epochs):
            self.metrics['d_prime'].append(d_primes_epochs[i])
        if len(d_primes_epochs) == 1:
            d_primes_epochs = d_primes_epochs[0]
        return d_primes_epochs

    def compute_nn_metrics(self, num_channels_to_compare=13, max_spikes_per_cluster=500, max_spikes_for_nn=10000,
                           n_neighbors=4, seed=0):
        '''
        Computes and returns the nearest neighbor metrics for the sorted dataset.

        Parameters
        ----------
        num_channels_to_compare: int
            The number of channels to be used for the PC extraction and comparison.
        max_spikes_per_cluster: int
            Max spikes to be used from each unit to compute cluster-based metrics.
        max_spikes_for_nn: int
            Max spikes to be used for nearest-neighbors calculation.
        n_neighbors: int
            Number of neighbors to compare.
        seed: int
            Random seed for extracting pc features.

        Returns
        ----------
        nn_hit_rates_epochs: np.array
            The nearest neighbor hit rates for each specified unit.
        nn_miss_rates_epochs: np.array
            The nearest neighbor miss rates for each specified unit.
        '''
        if self._pc_features is None:
            assert self._recording is not None, "No recording stored. Add a recording first " \
                                                "with set_recording or by computing all data for metrics"
            self.compute_pca_scores(self._recording)

        nn_hit_rates_epochs = []
        nn_miss_rates_epochs = []
        for epoch in self._epochs:
            in_epoch = np.logical_and(self._spike_times_pca > epoch[1], self._spike_times_pca < epoch[2])
            spikes_in_epoch = np.sum(in_epoch)
            spikes_for_nn = np.min([spikes_in_epoch, max_spikes_for_nn])

            nn_hit_rates_all, nn_miss_rates_all = metrics.calculate_pc_metrics(
                spike_clusters=self._spike_clusters_pca[in_epoch],
                total_units=self._total_units,
                pc_features=self._pc_features[in_epoch, :, :],
                pc_feature_ind=self._pc_feature_ind,
                num_channels_to_compare=num_channels_to_compare,
                max_spikes_for_cluster=max_spikes_per_cluster,
                spikes_for_nn=spikes_for_nn,
                n_neighbors=n_neighbors,
                metric_names=['nearest_neighbor'],
                seed=seed, verbose=self.verbose)[3:5]
            nn_hit_rates_list = []
            nn_miss_rates_list = []
            for i in self._unit_indices:
                nn_hit_rates_list.append(nn_hit_rates_all[i])
                nn_miss_rates_list.append(nn_miss_rates_all[i])
            nn_hit_rates = np.asarray(nn_hit_rates_list)
            nn_miss_rates = np.asarray(nn_miss_rates_list)
            nn_hit_rates_epochs.append(nn_hit_rates)
            nn_miss_rates_epochs.append(nn_miss_rates)

        self.metrics['nn_hit_rate'] = []
        self.metrics['nn_miss_rate'] = []
        for i, epoch in enumerate(self._epochs):
            self.metrics['nn_hit_rate'].append(nn_hit_rates_epochs[i])
            self.metrics['nn_miss_rate'].append(nn_miss_rates_epochs[i])
        if len(nn_hit_rates_epochs) == 1:
            nn_hit_rates_epochs = nn_hit_rates_epochs[0]
            nn_miss_rates_epochs = nn_miss_rates_epochs[0]
        return nn_hit_rates_epochs, nn_miss_rates_epochs

    def compute_metrics(self, isi_threshold=0.0015, min_isi=0.000166, snr_mode='mad', snr_noise_duration=10.0,
                        max_spikes_per_unit_for_snr=1000, drift_metrics_interval_s=51,
                        drift_metrics_min_spikes_per_interval=10,
                        max_spikes_for_silhouette=10000, num_channels_to_compare=13, max_spikes_per_cluster=500,
                        max_spikes_for_nn=10000, n_neighbors=4, metric_names=None, seed=0):
        '''
        Computes and returns all specified metrics for the sorted dataset.

        Parameters
        ----------
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
            Max spikes to be used from each unit to compute cluster-based metrics.
        max_spikes_for_nn: int
            Max spikes to be used for nearest-neighbors calculation.
        n_neighbors: int
            Number of neighbors to compare for  nearest-neighbors calculation.
        metrics_names: list or None
            The list of metric names to be computed. Available metrics are: 'firing_rate', 'num_spikes', 'isi_viol',
            'presence_ratio', 'amplitude_cutoff', 'max_drift', 'cumulative_drift', 'silhouette_score',
            'isolation_distance', 'l_ratio', 'd_prime', 'nn_hit_rate', 'nn_miss_rate', 'snr'. If None, all metrics are
            computed.
        seed: int
            Random seed for extracting features.

        Returns
        ----------
        metrics_epochs : list
            List of metrics data. The list consists of lists of metric data for each given epoch.
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

        if 'num_spikes' in metric_names:
            num_spikes_epochs = self.compute_num_spikes()
            metrics_epochs.append(num_spikes_epochs)

        if 'firing_rate' in metric_names:
            firing_rates_epochs = self.compute_firing_rates()
            metrics_epochs.append(firing_rates_epochs)

        if 'presence_ratio' in metric_names:
            presence_ratios_epochs = self.compute_presence_ratios()
            metrics_epochs.append(presence_ratios_epochs)

        if 'isi_viol' in metric_names:
            isi_violations_epochs = self.compute_isi_violations(isi_threshold=isi_threshold, min_isi=min_isi)
            metrics_epochs.append(isi_violations_epochs)

        if 'amplitude_cutoff' in metric_names:
            amplitude_cutoffs_epochs = self.compute_amplitude_cutoffs()
            metrics_epochs.append(amplitude_cutoffs_epochs)

        if 'snr' in metric_names:
            snrs_epochs = self.compute_snrs(snr_mode=snr_mode, snr_noise_duration=snr_noise_duration,
                                            max_spikes_per_unit_for_snr=max_spikes_per_unit_for_snr)
            metrics_epochs.append(snrs_epochs)

        if 'max_drift' in metric_names or 'cumulative_drift' in metric_names:
            max_drifts_epochs, cumulative_drifts_epochs = self.compute_drift_metrics(
                drift_metrics_interval_s=drift_metrics_interval_s,
                drift_metrics_min_spikes_per_interval=drift_metrics_min_spikes_per_interval)
            if 'max_drift' in metric_names:
                metrics_epochs.append(max_drifts_epochs)
            if 'cumulative_drift' in metric_names:
                metrics_epochs.append(cumulative_drifts_epochs)

        if 'silhouette_score' in metric_names:
            silhouette_scores_epochs = self.compute_silhouette_scores(
                max_spikes_for_silhouette=max_spikes_for_silhouette, seed=seed)
            metrics_epochs.append(silhouette_scores_epochs)

        if 'isolation_distance' in metric_names:
            isolation_distances_epochs = self.compute_isolation_distances(
                num_channels_to_compare=num_channels_to_compare, max_spikes_per_cluster=max_spikes_per_cluster,
                seed=seed)
            metrics_epochs.append(isolation_distances_epochs)

        if 'l_ratio' in metric_names:
            l_ratios_epochs = self.compute_l_ratios(num_channels_to_compare=num_channels_to_compare,
                                                    max_spikes_per_cluster=max_spikes_per_cluster, seed=seed)
            metrics_epochs.append(l_ratios_epochs)

        if 'd_prime' in metric_names:
            d_primes_epochs = self.compute_d_primes(num_channels_to_compare=num_channels_to_compare,
                                                    max_spikes_per_cluster=max_spikes_per_cluster, seed=seed)
            metrics_epochs.append(d_primes_epochs)

        if 'nn_hit_rate' in metric_names or 'nn_miss_rate' in metric_names:
            nn_hit_rates_epochs, nn_miss_rates_epochs = self.compute_nn_metrics(
                num_channels_to_compare=num_channels_to_compare, max_spikes_per_cluster=max_spikes_per_cluster,
                max_spikes_for_nn=max_spikes_for_nn, n_neighbors=n_neighbors, seed=seed)
            if 'nn_hit_rate' in metric_names:
                metrics_epochs.append(nn_hit_rates_epochs)
            if 'nn_miss_rate' in metric_names:
                metrics_epochs.append(nn_miss_rates_epochs)

        return metrics_epochs

    def get_metrics_dict(self):
        '''
        Return a copy of the cached metric dictionary

        Returns
        ----------
        metrics_copy: dict
            A copy of the metrics dictionary
        '''
        metrics_copy = copy.deepcopy(self.metrics)

        return metrics_copy

    def get_metrics_df(self):
        '''
        Return a dataframe of the cached metric dictionary

        Returns
        ----------
        metrics_df: pandas.DataFrame
            A pandas dataframe of the cached metrics
        '''
        all_unit_ids = self._sorting.get_unit_ids()
        epoch_unit_ids = all_unit_ids * len(self.get_epochs())
        metrics_df = pd.DataFrame(data=OrderedDict({'unit_ids': epoch_unit_ids}))
        for metric_name in self.metrics.keys():
            all_metrics = [0] * len(epoch_unit_ids)
            for i, metric in enumerate(self.metrics[metric_name]):
                unit_ids = self.get_unit_ids()
                for unit_id in all_unit_ids:
                    if unit_id in unit_ids:
                        all_metrics[unit_ids.index(unit_id) + len(all_unit_ids) * i] = metric[unit_ids.index(unit_id)]
                    else:
                        all_metrics[unit_ids.index(unit_id) + len(all_unit_ids) * i] = ''
            metrics_df = pd.concat((metrics_df, pd.DataFrame(data=OrderedDict(({metric_name: all_metrics})))),
                                   sort=False, axis=1)

        epoch_names_all = []
        epoch_starts_all = []
        epoch_ends_all = []
        for epoch in self.get_epochs():
            epoch_name = [epoch[0]] * len(all_unit_ids)
            epoch_start = [epoch[1]] * len(all_unit_ids)
            epoch_end = [epoch[2]] * len(all_unit_ids)
            epoch_names_all = epoch_names_all + epoch_name
            epoch_starts_all = epoch_starts_all + epoch_start
            epoch_ends_all = epoch_ends_all + epoch_end
        metrics_df = pd.concat((metrics_df, pd.DataFrame(data=OrderedDict(({'epoch_name': epoch_names_all})))),
                               sort=False, axis=1)
        metrics_df = pd.concat((metrics_df, pd.DataFrame(data=OrderedDict(({'epoch_start': epoch_starts_all})))),
                               sort=False, axis=1)
        metrics_df = pd.concat((metrics_df, pd.DataFrame(data=OrderedDict(({'epoch_end': epoch_ends_all})))),
                               sort=False, axis=1)

        return metrics_df

    @classmethod
    def get_default_params_dict(self):
        '''
        Returns the default params for all quality metrics.

        Returns
        ----------
        quality_metrics_params: dict
            The default params for all quality metrics.
        '''
        self._quality_metrics_params = {
            "isi_threshold": 0.0015,
            "min_isi": 0.000166,
            "num_channels_to_compare": 13,
            "max_spikes_per_cluster": 500,
            "max_spikes_for_nn": 10000,
            "n_neighbors": 4,
            'n_silhouette': 10000,
            "quality_metrics_output_file": "metrics.csv",
            "drift_metrics_interval_s": 51,
            "drift_metrics_min_spikes_per_interval": 10
        }

        return self._quality_metrics_params

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


def _compute_channel_noise_levels(recording, mode='mad', noise_duration=10.0, seed=0):
    '''
    Computes noise level channel-wise

    Parameters
    ----------
    recording: RecordingExtractor
        The recording ectractor object
    mode: str
        'std' or 'mad' (default
    noise_duration: float
        Number of seconds to compute SNR from

    Returns
    -------
    moise_levels: list
        Noise levels for each channel
    '''
    M = recording.get_num_channels()
    n_frames = int(noise_duration * recording.get_sampling_frequency())

    if n_frames >= recording.get_num_frames():
        start_frame = 0
        end_frame = recording.get_num_frames()
    else:
        start_frame = np.random.RandomState(seed=seed).randint(0, recording.get_num_frames() - n_frames)
        end_frame = start_frame + n_frames

    X = recording.get_traces(start_frame=start_frame, end_frame=end_frame)
    noise_levels = []
    for ch in range(M):
        if mode == 'std':
            noise_level = np.std(X[ch, :])
        elif mode == 'mad':
            noise_level = np.median(np.abs(X[ch, :]) / 0.6745)
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
