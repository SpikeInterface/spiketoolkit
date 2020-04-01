import numpy as np
import spikemetrics.metrics as metrics
from .utils.thresholdcurator import ThresholdCurator
from .quality_metric import QualityMetric
from collections import OrderedDict
from .parameter_dictionaries import update_all_param_dicts_with_kwargs


class NearestNeighbor(QualityMetric):
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # err
    params = OrderedDict(
        [('num_channels_to_compare', 13), ('max_spikes_per_cluster', 500), ('max_spikes_for_nn', 10000),
         ('n_neighbors', 4)])
    curator_name = "ThresholdNearestNeighbor"

    def __init__(self, metric_data):
        QualityMetric.__init__(self, metric_data, metric_name="nearest_neighbor")

        if not metric_data.has_pca_scores():
            raise ValueError("MetricData object must have pca scores")

    def compute_metric(self, num_channels_to_compare, max_spikes_per_cluster, max_spikes_for_nn,
                       n_neighbors, **kwargs):
        params_dict = update_all_param_dicts_with_kwargs(kwargs)
        save_property_or_features = params_dict['save_property_or_features']
        seed = params_dict['seed']

        nn_hit_rates_epochs = []
        nn_miss_rates_epochs = []
        for epoch in self._metric_data._epochs:
            in_epoch = self._metric_data.get_in_epoch_bool_mask(epoch, self._metric_data._spike_times_pca)
            spikes_in_epoch = np.sum(in_epoch)
            spikes_for_nn = np.min([spikes_in_epoch, max_spikes_for_nn])
            nn_hit_rates_all, nn_miss_rates_all = metrics.calculate_pc_metrics(
                spike_clusters=self._metric_data._spike_clusters_pca[in_epoch],
                total_units=self._metric_data._total_units,
                pc_features=self._metric_data._pc_features[in_epoch, :, :],
                pc_feature_ind=self._metric_data._pc_feature_ind,
                num_channels_to_compare=num_channels_to_compare,
                max_spikes_for_cluster=max_spikes_per_cluster,
                spikes_for_nn=spikes_for_nn,
                n_neighbors=n_neighbors,
                metric_names=["nearest_neighbor"],
                seed=seed,
                verbose=self._metric_data.verbose,
            )[3:5]
            nn_hit_rates_list = []
            nn_miss_rates_list = []
            for i in self._metric_data._unit_indices:
                nn_hit_rates_list.append(nn_hit_rates_all[i])
                nn_miss_rates_list.append(nn_miss_rates_all[i])
            nn_hit_rates = np.asarray(nn_hit_rates_list)
            nn_miss_rates = np.asarray(nn_miss_rates_list)
            nn_hit_rates_epochs.append(nn_hit_rates)
            nn_miss_rates_epochs.append(nn_miss_rates)
        if save_property_or_features:
            self.save_property_or_features(self._metric_data._sorting, nn_hit_rates_epochs, metric_name="nn_hit_rate")
            self.save_property_or_features(self._metric_data._sorting, nn_miss_rates_epochs, metric_name="nn_miss_rate")
        return list(zip(nn_hit_rates_epochs, nn_miss_rates_epochs))

    def threshold_metric(self, threshold, threshold_sign, metric_name, num_channels_to_compare, max_spikes_per_cluster,
                         max_spikes_for_nn, n_neighbors, **kwargs):
        nn_hit_rates_epochs, nn_miss_rates_epochs = \
        self.compute_metric(num_channels_to_compare, max_spikes_per_cluster, max_spikes_for_nn,
                            n_neighbors, **kwargs)[0]
        if metric_name == "nn_hit_rate":
            metrics_epoch = nn_hit_rates_epochs
        elif metric_name == "nn_miss_rate":
            metrics_epoch = nn_miss_rates_epochs
        else:
            raise ValueError("Invalid metric named entered")

        threshold_curator = ThresholdCurator(
            sorting=self._metric_data._sorting, metrics_epoch=metrics_epoch
        )
        threshold_curator.threshold_sorting(
            threshold=threshold, threshold_sign=threshold_sign
        )
        return threshold_curator
