import numpy as np
import spiketoolkit as st
import spikemetrics.metrics as metrics
from spikemetrics.utils import Epoch 

class MetricCalculator:
    def __init__(self):
        self._sorting = None
        self._recording = None
        self._sampling_frequency = None
        self._spike_times = None
        self._spike_clusters = None
        self._total_units = None
        self._amplitudes = None
        self._pc_features = None
        self._pc_feature_ind = None

    def setup_spike_based_metric_data(self, sorting, sampling_frequency):
        '''
        Computes and stores spike-based data for all spiking metrics
        '''
        spike_times, spike_clusters =  st.validation.validation_tools.get_firing_times_ids(sorting, sampling_frequency)
        
        self._sorting = sorting
        self._sampling_frequency = sampling_frequency
        self._spike_times = spike_times
        self._spike_clusters = spike_clusters
        self._total_units = len(sorting.get_unit_ids())

    def setup_all_metric_data(self, recording, sorting, nPC=3, ms_before=1., ms_after=2., dtype=None, max_num_waveforms=np.inf, \
                              max_num_pca_waveforms=np.inf, save_features_props=False):
        '''
        Computes and stores data for all metrics
        '''
        spike_times, spike_clusters, amplitudes, \
        pc_features, pc_feature_ind  = st.validation.validation_tools.get_quality_metric_data(recording, sorting, nPC=nPC, ms_before=ms_before, \
                                                                                              ms_after=ms_after, dtype=dtype, max_num_waveforms=np.inf, \
                                                                                              max_num_pca_waveforms=max_num_waveforms, \
                                                                                              save_features_props=save_features_props)
        
        self._sorting = sorting
        self._recording = recording
        self._sampling_frequency = recording.get_sampling_frequency()
        self._spike_times = spike_times
        self._spike_clusters = spike_clusters
        self._total_units = len(sorting.get_unit_ids())
        self._amplitudes = amplitudes
        self._pc_features = pc_features
        self._pc_feature_ind = pc_feature_ind

    def compute_firing_rates(self, unit_ids=None, epoch_tuples=None):
        '''
        Computes and returns the spike times in seconds and also returns 
        the cluster_ids needed for quality metrics (all within given epoch)
        
        Parameters
        ----------
        unit_ids: list
            List of unit ids to compute metric for. If not specified, all units are used
        epoch_tuples: list
            A list of tuples with a start and end frame for the epoch in question.

        Returns
        ----------
        firing_rates: list
            The firing rates of the sorted units in the given epochs.
        '''

        assert self._sorting is not None, "Compute spike-based or all metrics data first"

        if unit_ids is None or unit_ids == []:
            unit_ids = self._sorting.get_unit_ids()
            unit_indices = np.arange(len(unit_ids))
        else:
            unit_indices = _get_unit_indices(self._sorting, unit_ids)
        firings_rates_epochs = []
        epochs = _get_epochs(epoch_tuples, self._sampling_frequency)
        for epoch in epochs:
            in_epoch = np.logical_and(self._spike_times > epoch[0], self._spike_times < epoch[1])
            firing_rates_all, _ = metrics.calculate_firing_rate_and_spikes(self._spike_times[in_epoch], self._spike_clusters[in_epoch], self._total_units)
            firing_rates_list = []
            for i in unit_indices:
                firing_rates_list.append(firing_rates_all[i])
            firing_rates = np.asarray(firing_rates_list)
            firings_rates_epochs.append(firing_rates)

        if len(firings_rates_epochs) == 1:
            firings_rates_epochs = firings_rates_epochs[0]
        return firings_rates_epochs

    def compute_num_spikes(self, unit_ids=None, epoch_tuples=None):
        '''
        Computes and returns the spike times in seconds and also returns 
        the cluster_ids needed for quality metrics (all within given epoch)
        
        Parameters
        ----------
        unit_ids: list
            List of unit ids to compute metric for. If not specified, all units are used
        epoch_tuples: list
            A list of tuples with a start and end frame for the epoch in question.

        Returns
        ----------
        num_spikes_epochs: list
            The spike counts of the sorted units in the given epochs.
        '''

        assert self._sorting is not None, "Compute spike-based or all metrics data first"

        if unit_ids is None or unit_ids == []:
            unit_ids = self._sorting.get_unit_ids()
            unit_indices = np.arange(len(unit_ids))
        else:
            unit_indices = _get_unit_indices(self._sorting, unit_ids)

        num_spikes_epochs = []
        epochs = _get_epochs(epoch_tuples, self._sampling_frequency)
        for epoch in epochs:
            in_epoch = np.logical_and(self._spike_times > epoch[0], self._spike_times < epoch[1])
            _, num_spikes_all = metrics.calculate_firing_rate_and_spikes(self._spike_times[in_epoch], self._spike_clusters[in_epoch], self._total_units)
            num_spikes_list = []
            for i in unit_indices:
                num_spikes_list.append(num_spikes_all[i])
            num_spikes = np.asarray(num_spikes_list)
            num_spikes_epochs.append(num_spikes)
        if len(num_spikes_epochs) == 1:
            num_spikes_epochs = num_spikes_epochs[0]
        return num_spikes_epochs

    def compute_isi_violations(self, isi_threshold=0.0015, min_isi=0.000166, unit_ids=None, epoch_tuples=None):
        '''
        Computes and returns the ISI violations for the given units and parameters.
        
        Parameters
        ----------
        isi_threshold: float
            The isi threshold for calculating isi violations.
        min_isi: float
            The minimum expected isi value.
        unit_ids: list
            List of unit ids to compute metric for. If not specified, all units are used
        epoch_tuples: list
            A list of tuples with a start and end frame for the epoch in question.
            
        Returns
        ----------
        isi_violations_epochs: list
            The isi violations of the sorted units in the given epochs.
        '''

        assert self._sorting is not None, "Compute spike-based or all metrics data first"

        if unit_ids is None or unit_ids == []:
            unit_ids = self._sorting.get_unit_ids()
            unit_indices = np.arange(len(unit_ids))
        else:
            unit_indices = _get_unit_indices(self._sorting, unit_ids)

        isi_violations_epochs = []
        epochs = _get_epochs(epoch_tuples, self._sampling_frequency)
        for epoch in epochs:
            in_epoch = np.logical_and(self._spike_times > epoch[0], self._spike_times < epoch[1])
            isi_violations_all = metrics.calculate_isi_violations(self._spike_times[in_epoch], self._spike_clusters[in_epoch], self._total_units, \
                                                                  isi_threshold=isi_threshold, min_isi=min_isi)
            isi_violations_list = []
            for i in unit_indices:
                isi_violations_list.append(isi_violations_all[i])
            isi_violations = np.asarray(isi_violations_list)
            isi_violations_epochs.append(isi_violations)
        if len(isi_violations_epochs) == 1:
            isi_violations_epochs = isi_violations_epochs[0]
        return isi_violations_epochs

    def compute_presence_ratios(self, unit_ids=None, epoch_tuples=None):
        '''
        Computes and returns the presence ratios for the given units.
        
        Parameters
        ----------
        unit_ids: list
            List of unit ids to compute metric for. If not specified, all units are used
        epoch_tuples: list
            A list of tuples with a start and end frame for the epoch in question.

        Returns
        ----------
        presence_ratios_epochs: list
            The presence ratios violations of the sorted units in the given epochs.
        '''

        assert self._sorting is not None, "Compute spike-based or all metrics data first"

        if unit_ids is None or unit_ids == []:
            unit_ids = self._sorting.get_unit_ids()
            unit_indices = np.arange(len(unit_ids))
        else:
            unit_indices = _get_unit_indices(self._sorting, unit_ids)

        presence_ratios_epochs = []
        epochs = _get_epochs(epoch_tuples, self._sampling_frequency)
        for epoch in epochs:
            in_epoch = np.logical_and(self._spike_times > epoch[0], self._spike_times < epoch[1])
            presence_ratios_all = metrics.calculate_presence_ratio(self._spike_times[in_epoch], self._spike_clusters[in_epoch], self._total_units)
            presence_ratios_list = []
            for i in unit_indices:
                presence_ratios_list.append(presence_ratios_all[i])
            presence_ratios = np.asarray(presence_ratios_list)
            presence_ratios_epochs.append(presence_ratios)
        if len(presence_ratios_epochs) == 1:
            presence_ratios_epochs = presence_ratios_epochs[0]
        return presence_ratios_epochs

    def compute_drift_metrics(self, drift_metrics_interval_s=51, drift_metrics_min_spikes_per_interval=10, \
                              unit_ids=None, epoch_tuples=None):
        '''
        Computes and returns the drift metrics for the sorted dataset.

        Parameters
        ----------
        drift_metrics_interval_s: float
            Time period for evaluating drift.
        drift_metrics_min_spikes_per_interval: int
            Minimum number of spikes for evaluating drift metrics per interval. 
        unit_ids: list
            List of unit ids to compute metric for. If not specified, all units are used
        epoch_tuples: list
            A list of tuples with a start and end frame for the epoch in question.

        Returns
        ----------
        max_drifts_epochs: list
            The max drift of the given units over the specified epochs
        cumulative_drifts_epochs: list
            The cumulative drifts of the given units over the specified epochs
        '''
        assert self._recording is not None, "Compute data for all metrics first"

        if unit_ids is None or unit_ids == []:
            unit_ids = self._sorting.get_unit_ids()
            unit_indices = np.arange(len(unit_ids))
        else:
            unit_indices = _get_unit_indices(self._sorting, unit_ids)

        max_drifts_epochs = []
        cumulative_drifts_epochs = []
        epochs = _get_epochs(epoch_tuples, self._recording.get_sampling_frequency())
        for epoch in epochs:
            in_epoch = np.logical_and(self._spike_times > epoch[0], self._spike_times < epoch[1])
            max_drifts_all, cumulative_drifts_all = metrics.calculate_drift_metrics(self._spike_times[in_epoch],
                                                                                    self._spike_clusters[in_epoch],
                                                                                    self._total_units,
                                                                                    self._pc_features[in_epoch,:,:],
                                                                                    self._pc_feature_ind,
                                                                                    drift_metrics_interval_s,
                                                                                    drift_metrics_min_spikes_per_interval)
            max_drifts_list = []
            cumulative_drifts_list = []
            for i in unit_indices:
                max_drifts_list.append(max_drifts_all[i])
                cumulative_drifts_list.append(cumulative_drifts_all[i])
            
            max_drifts = np.asarray(max_drifts_list)
            cumulative_drifts = np.asarray(cumulative_drifts_list)
            max_drifts_epochs.append(max_drifts)
            cumulative_drifts_epochs.append(cumulative_drifts)
        if len(max_drifts_epochs) == 1:
            max_drifts_epochs = max_drifts_epochs[0]
            cumulative_drifts_epochs = cumulative_drifts_epochs[0]
        return max_drifts_epochs, cumulative_drifts_epochs

    def compute_silhouette_score(self, max_spikes_for_silhouette=10000, unit_ids=None, epoch_tuples=None, seed=0):
        '''
        Computes and returns the silhouette scores for each unit in the sorted dataset.

        Parameters
        ----------
        max_spikes_for_silhouette: int
            Max spikes to be used for silhouette metric
        unit_ids: list
            List of unit ids to compute metric for. If not specified, all units are used
        epoch_tuples: list
            A list of tuples with a start and end frame for the epoch in question.
        seed: int
            A random seed for reproducibility

        Returns
        ----------
        silhouette_scores_epochs: list
            The silhouette scores of the given units for the specified epochs.
        '''
        assert self._recording is not None, "Compute data for all metrics first"

        if unit_ids is None or unit_ids == []:
            unit_ids = self._sorting.get_unit_ids()
            unit_indices = np.arange(len(unit_ids))
        else:
            unit_indices = _get_unit_indices(self._sorting, unit_ids)

        silhouette_scores_epochs = []
        epochs = _get_epochs(epoch_tuples, self._recording.get_sampling_frequency())
        for epoch in epochs:
            in_epoch = np.logical_and(self._spike_times > epoch[0], self._spike_times < epoch[1])
            spikes_in_epoch = np.sum(in_epoch)
            spikes_for_silhouette = min(spikes_in_epoch, max_spikes_for_silhouette)
            
            silhouette_scores_all = metrics.calculate_silhouette_score(self._spike_clusters[in_epoch],
                                                                       self._total_units,
                                                                       self._pc_features[in_epoch,:,:],
                                                                       self._pc_feature_ind,
                                                                       spikes_for_silhouette,
                                                                       seed=seed)
            silhouette_scores_list = []
            for i in unit_indices:
                silhouette_scores_list.append(silhouette_scores_all[i])
            silhouette_scores = np.asarray(silhouette_scores_list)
            silhouette_scores_epochs.append(silhouette_scores)
        if len(silhouette_scores_epochs) == 1:
            silhouette_scores_epochs = silhouette_scores_epochs[0]    
        return silhouette_scores_epochs

    def compute_isolations_distances(self, num_channels_to_compare=13, max_spikes_for_unit=500, unit_ids=None, \
                                     epoch_tuples=None, seed=0):
        '''
        Computes and returns the mahalanobis metric, isolation distance, for the sorted dataset.

        Parameters
        ----------
        num_channels_to_compare: int
            The number of channels to be used for the PC extraction and comparison
        max_spikes_for_unit: int
            Max spikes to be used from each unit
        unit_ids: list
            List of unit ids to compute metric for. If not specified, all units are used
        epoch_tuples: list
            A list of tuples with a start and end frame for the epoch in question.
        seed: int
            Random seed for extracting pc features.

        Returns
        ----------
        isolation_distances_epochs: list
            Returns the isolation distances of each specified unit for the given epochs.
        '''
        assert self._recording is not None, "Compute data for all metrics first"

        if unit_ids is None or unit_ids == []:
            unit_ids = self._sorting.get_unit_ids()
            unit_indices = np.arange(len(unit_ids))
        else:
            unit_indices = _get_unit_indices(self._sorting, unit_ids)

        isolation_distances_epochs = []
        epochs = _get_epochs(epoch_tuples, self._recording.get_sampling_frequency())
        for epoch in epochs:
            in_epoch = np.logical_and(self._spike_times > epoch[0], self._spike_times < epoch[1])
            isolation_distances_all = metrics.calculate_pc_metrics(spike_clusters=self._spike_clusters[in_epoch],
                                                                   total_units=self._total_units,   
                                                                   pc_features=self._pc_features[in_epoch,:,:],
                                                                   pc_feature_ind=self._pc_feature_ind,
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
            isolation_distances_epochs.append(isolation_distances)
        if len(isolation_distances_epochs) == 1:
            isolation_distances_epochs = isolation_distances_epochs[0]    
        return isolation_distances_epochs


    def compute_l_ratios(self, num_channels_to_compare=13, max_spikes_for_unit=500, unit_ids=None, \
                         epoch_tuples=None, seed=0):
        '''
        Computes and returns the mahalanobis metric, l-ratio, for the sorted dataset.

        Parameters
        ----------
        num_channels_to_compare: int
            The number of channels to be used for the PC extraction and comparison
        max_spikes_for_unit: int
            Max spikes to be used from each unit
        epoch_tuples: list
            A list of tuples with a start and end frame for the epoch in question.
        seed: int
            Random seed for extracting pc features.

        Returns
        ----------
        l_ratios_epochs: list
            Returns the L ratios of each specified unit for the given epochs
        '''
        assert self._recording is not None, "Compute data for all metrics first"

        if unit_ids is None or unit_ids == []:
            unit_ids = self._sorting.get_unit_ids()
            unit_indices = np.arange(len(unit_ids))
        else:
            unit_indices = _get_unit_indices(self._sorting, unit_ids)

        l_ratios_epochs = []
        epochs = _get_epochs(epoch_tuples, self._recording.get_sampling_frequency())
        for epoch in epochs:
            in_epoch = np.logical_and(self._spike_times > epoch[0], self._spike_times < epoch[1])
            l_ratios_all = metrics.calculate_pc_metrics(spike_clusters=self._spike_clusters[in_epoch],
                                                        total_units=self._total_units,   
                                                        pc_features=self._pc_features[in_epoch,:,:],
                                                        pc_feature_ind=self._pc_feature_ind,
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
            l_ratios_epochs.append(l_ratios)
        if len(l_ratios_epochs) == 1:
            l_ratios_epochs = l_ratios_epochs[0]
        return l_ratios_epochs

    def compute_d_primes(self, num_channels_to_compare=13, max_spikes_for_unit=500, unit_ids=None, \
                         epoch_tuples=None, seed=0):
        '''
        Computes and returns the lda-based metric, d prime, for the sorted dataset.

        Parameters
        ----------
        num_channels_to_compare: int
            The number of channels to be used for the PC extraction and comparison
        max_spikes_for_unit: int
            Max spikes to be used from each unit
        unit_ids: list
            List of unit ids to compute metric for. If not specified, all units are used
        epoch_tuples: list
            A list of tuples with a start and end frame for the epoch in question.
        seed: int
            Random seed for extracting pc features.

        Returns
        ----------
        d_primes_epochs: list
            Returns the d primes of each specified unit for the given epochs.
        '''
        assert self._recording is not None, "Compute data for all metrics first"

        if unit_ids is None or unit_ids == []:
            unit_ids = self._sorting.get_unit_ids()
            unit_indices = np.arange(len(unit_ids))
        else:
            unit_indices = _get_unit_indices(self._sorting, unit_ids)

        d_primes_epochs = []
        epochs = _get_epochs(epoch_tuples, self._recording.get_sampling_frequency())
        for epoch in epochs:
            in_epoch = np.logical_and(self._spike_times > epoch[0], self._spike_times < epoch[1])    
            d_primes_all = metrics.calculate_pc_metrics(spike_clusters=self._spike_clusters[in_epoch],
                                                        total_units=self._total_units,   
                                                        pc_features=self._pc_features[in_epoch,:,:],
                                                        pc_feature_ind=self._pc_feature_ind,
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
            d_primes_epochs.append(d_primes)
        if len(d_primes_epochs) == 1:
            d_primes_epochs = d_primes_epochs[0]
        return d_primes_epochs

    def compute_nn_metrics(self, num_channels_to_compare=13, max_spikes_for_unit=500, max_spikes_for_nn=10000, \
                           n_neighbors=4, unit_ids=None, epoch_tuples=None, seed=0):
        '''
        Computes and returns the lda-based metric, d prime, for the sorted dataset.

        Parameters
        ----------
        num_channels_to_compare: int
            The number of channels to be used for the PC extraction and comparison.
        max_spikes_for_unit: int
            Max spikes to be used from each unit.
        max_spikes_for_nn: int
            Max spikes to be used for nearest-neighbors calculation.
        unit_ids: list
            List of unit ids to compute metric for. If not specified, all units are used
        epoch_tuples: list
            A list of tuples with a start and end frame for the epoch in question.
        seed: int
            Random seed for extracting pc features.

        Returns
        ----------
        nn_hit_rates: np.array
            The nearest neighbor hit rates for each specified unit.
        nn_miss_rates: np.array
            The nearest neighbor miss rates for each specified unit.
        '''
        assert self._recording is not None, "Compute data for all metrics first"

        if unit_ids is None or unit_ids == []:
            unit_ids = self._sorting.get_unit_ids()
            unit_indices = np.arange(len(unit_ids))
        else:
            unit_indices = _get_unit_indices(self._sorting, unit_ids)

        nn_hit_rates_epochs = []
        nn_miss_rates_epochs = []
        epochs = _get_epochs(epoch_tuples, self._recording.get_sampling_frequency())
        for epoch in epochs:
            in_epoch = np.logical_and(self._spike_times > epoch[0], self._spike_times < epoch[1])  
            spikes_in_epoch = np.sum(in_epoch)
            spikes_for_nn = min(spikes_in_epoch, max_spikes_for_nn)
            
            nn_hit_rates_all, nn_miss_rates_all = metrics.calculate_pc_metrics(spike_clusters=self._spike_clusters[in_epoch],
                                                                               total_units=self._total_units,   
                                                                               pc_features=self._pc_features[in_epoch,:,:],
                                                                               pc_feature_ind=self._pc_feature_ind,
                                                                               num_channels_to_compare=num_channels_to_compare,
                                                                               max_spikes_for_cluster=max_spikes_for_unit,
                                                                               spikes_for_nn=spikes_for_nn,
                                                                               n_neighbors=n_neighbors,
                                                                               metric_names=['nearest_neighbor'],
                                                                               seed=seed)[3:5]
            nn_hit_rates_list = []
            nn_miss_rates_list = []
            for i in unit_indices:
                nn_hit_rates_list.append(nn_hit_rates_all[i])
                nn_miss_rates_list.append(nn_miss_rates_all[i])
            nn_hit_rates = np.asarray(nn_hit_rates_list)
            nn_miss_rates = np.asarray(nn_miss_rates_list)
            nn_hit_rates_epochs.append(nn_hit_rates)
            nn_miss_rates_epochs.append(nn_miss_rates)
        if len(nn_hit_rates_epochs) == 1:
            nn_hit_rates_epochs = nn_hit_rates_epochs[0]
            nn_miss_rates_epochs = nn_miss_rates_epochs[0]
        return nn_hit_rates_epochs, nn_miss_rates_epochs

    # def compute_metrics(self, isi_threshold=0.0015, min_isi=0.000166, drift_metrics_interval_s=51, \
    #                     drift_metrics_min_spikes_per_interval=10, max_spikes_for_silhouette=10000, \
    #                     num_channels_to_compare=13, max_spikes_for_unit=500, max_spikes_for_nn=10000, \
    #                     n_neighbors=4, unit_ids=None, epoch_tuples=None, seed=0):
    #     '''
    #     Computes and returns the lda-based metric, d prime, for the sorted dataset.

    #     Parameters
    #     ----------
    #     isi_threshold: float
    #         The isi threshold for calculating isi violations.
    #     min_isi: float
    #         The minimum expected isi value.
    #     drift_metrics_interval_s: float
    #         Time period for evaluating drift.
    #     drift_metrics_min_spikes_per_interval: int
    #         Minimum number of spikes for evaluating drift metrics per interval. 
    #     max_spikes_for_silhouette: int
    #         Max spikes to be used for silhouette metric
    #     num_channels_to_compare: int
    #         The number of channels to be used for the PC extraction and comparison.
    #     max_spikes_for_unit: int
    #         Max spikes to be used from each unit.
    #     max_spikes_for_nn: int
    #         Max spikes to be used for nearest-neighbors calculation.
    #     unit_ids: list
    #         List of unit ids to compute metric for. If not specified, all units are used
    #     epoch_tuples: list
    #         A list of tuples with a start and end frame for the epoch in question.
    #     seed: int
    #         Random seed for extracting pc features.

    #     Returns
    #     ----------
    #     metrics : pandas.DataFrame
    #     one column for each metric
    #     one row per unit per epoch
    #     '''
    #     assert self._recording is not None, "Compute data for all metrics first"

    #     if unit_ids is None or unit_ids == []:
    #         unit_ids = self._sorting.get_unit_ids()
    #         unit_indices = np.arange(len(unit_ids))
    #     else:
    #         unit_indices = _get_unit_indices(self._sorting, unit_ids)

    #     nn_hit_rates_epochs = []
    #     nn_miss_rates_epochs = []
    #     epochs = _get_epochs(epoch_tuples, self._recording.get_sampling_frequency())
    #     for epoch in epochs:
    #         in_epoch = np.logical_and(self._spike_times > epoch[0], self._spike_times < epoch[1])  
    #         spikes_in_epoch = np.sum(in_epoch)
    #         spikes_for_nn = min(spikes_in_epoch, max_spikes_for_nn)
            
    #         nn_hit_rates_all, nn_miss_rates_all = metrics.calculate_pc_metrics(spike_clusters=self._spike_clusters[in_epoch],
    #                                                                            total_units=self._total_units,   
    #                                                                            pc_features=self._pc_features[in_epoch,:,:],
    #                                                                            pc_feature_ind=self._pc_feature_ind,
    #                                                                            num_channels_to_compare=num_channels_to_compare,
    #                                                                            max_spikes_for_cluster=max_spikes_for_unit,
    #                                                                            spikes_for_nn=spikes_for_nn,
    #                                                                            n_neighbors=n_neighbors,
    #                                                                            metric_names=['nearest_neighbor'],
    #                                                                            seed=seed)[3:5]
    #         nn_hit_rates_list = []
    #         nn_miss_rates_list = []
    #         for i in unit_indices:
    #             nn_hit_rates_list.append(nn_hit_rates_all[i])
    #             nn_miss_rates_list.append(nn_miss_rates_all[i])
    #         nn_hit_rates = np.asarray(nn_hit_rates_list)
    #         nn_miss_rates = np.asarray(nn_miss_rates_list)
    #         nn_hit_rates_epochs.append(nn_hit_rates)
    #         nn_miss_rates_epochs.append(nn_miss_rates)
    #     if len(nn_hit_rates_epochs) == 1:
    #         nn_hit_rates_epochs = nn_hit_rates_epochs[0]
    #         nn_miss_rates_epochs = nn_miss_rates_epochs[0]
    #     return nn_hit_rates_epochs, nn_miss_rates_epochs


    def get_default_params_dict(self):
        '''
        Computes and returns the default params for all quality metrics.

        Returns
        ----------
        quality_metrics_params: dict
        The default params for all quality metrics.
        '''
        self._quality_metrics_params = {
                                        "isi_threshold" : 0.0015,
                                        "min_isi" : 0.000166,
                                        "num_channels_to_compare" : 13,
                                        "max_spikes_for_unit" : 500,
                                        "max_spikes_for_nn" : 10000,
                                        "n_neighbors" : 4,
                                        'n_silhouette' : 10000,
                                        "quality_metrics_output_file" : "metrics.csv",
                                        "drift_metrics_interval_s" : 51,
                                        "drift_metrics_min_spikes_per_interval" : 10
                                       }

        return self._quality_metrics_params

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

def _get_epochs(epoch_tuples, sampling_frequency):
    if epoch_tuples is None:
        epochs = [(0, np.inf)]
    else:
        epochs = []
        for epoch_tuple in epoch_tuples:
            epoch = (epoch_tuple[0]/sampling_frequency, epoch_tuple[1]/sampling_frequency)
            epochs.append(epoch)
    return epochs
