import numpy as np
import spikemetrics.metrics as metrics
from .utils.thresholdcurator import ThresholdCurator
from .quality_metric import QualityMetric
from collections import OrderedDict


class IsolationDistance(QualityMetric):
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # err
    params = OrderedDict([('num_channels_to_compare', 13), ('max_spikes_per_cluster', 500)])
    curator_name = "ThresholdIsolationDistance"

    def __init__(self, metric_data):
        QualityMetric.__init__(self, metric_data, metric_name="isolation_distance")

        if not metric_data.has_pca_scores():
            raise ValueError("MetricData object must have pca scores")

    def compute_metric(self, num_channels_to_compare, max_spikes_per_cluster, seed, save_property_or_features):

        isolation_distances_epochs = []
        for epoch in self._metric_data._epochs:
            in_epoch = self._metric_data.get_in_epoch_bool_mask(epoch, self._metric_data._spike_times_pca)
            isolation_distances_all = metrics.calculate_pc_metrics(
                spike_clusters=self._metric_data._spike_clusters_pca[in_epoch],
                total_units=self._metric_data._total_units,
                pc_features=self._metric_data._pc_features[in_epoch, :, :],
                pc_feature_ind=self._metric_data._pc_feature_ind,
                num_channels_to_compare=num_channels_to_compare,
                max_spikes_for_cluster=max_spikes_per_cluster,
                spikes_for_nn=None,
                n_neighbors=None,
                metric_names=["isolation_distance"],
                seed=seed,
                verbose=self._metric_data.verbose,
            )[0]
            isolation_distances_list = []
            for i in self._metric_data._unit_indices:
                isolation_distances_list.append(isolation_distances_all[i])
            isolation_distances = np.asarray(isolation_distances_list)
            isolation_distances_epochs.append(isolation_distances)
        if save_property_or_features:
            self.save_property_or_features(self._metric_data._sorting, isolation_distances_epochs, self._metric_name)
        return isolation_distances_epochs

    def threshold_metric(self, threshold, threshold_sign, num_channels_to_compare, max_spikes_per_cluster, seed,
                         save_property_or_features):
        isolation_distances_epochs = \
        self.compute_metric(num_channels_to_compare, max_spikes_per_cluster, seed, save_property_or_features=save_property_or_features)[0]
        threshold_curator = ThresholdCurator(
            sorting=self._metric_data._sorting, metrics_epoch=isolation_distances_epochs
        )
        threshold_curator.threshold_sorting(
            threshold=threshold, threshold_sign=threshold_sign
        )
        return threshold_curator
