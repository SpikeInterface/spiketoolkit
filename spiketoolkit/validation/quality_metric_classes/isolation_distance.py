import numpy as np
import spikemetrics.metrics as metrics
from .utils.thresholdcurator import ThresholdCurator
from .quality_metric import QualityMetric
from collections import OrderedDict
from .parameter_dictionaries import update_all_param_dicts_with_kwargs


class IsolationDistance(QualityMetric):
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # err
    params = OrderedDict([('num_channels_to_compare', 13), ('max_spikes_per_cluster', 500)])
    curator_name = "ThresholdIsolationDistance"

    def __init__(self, metric_data):
        QualityMetric.__init__(self, metric_data, metric_name="isolation_distance")

        if not metric_data.has_pca_scores():
            raise ValueError("MetricData object must have pca scores")

    def compute_metric(self, num_channels_to_compare, max_spikes_per_cluster, **kwargs):
        params_dict = update_all_param_dicts_with_kwargs(kwargs)
        save_property_or_features = params_dict['save_property_or_features']
        seed = params_dict['seed']
        isolation_distances_all = metrics.calculate_pc_metrics(
            spike_clusters=self._metric_data._spike_clusters_pca,
            total_units=self._metric_data._total_units,
            pc_features=self._metric_data._pc_features,
            pc_feature_ind=self._metric_data._pc_feature_ind,
            num_channels_to_compare=num_channels_to_compare,
            max_spikes_for_cluster=max_spikes_per_cluster,
            spikes_for_nn=None,
            n_neighbors=None,
            metric_names=["isolation_distance"],
            seed=seed,
            spike_cluster_subset=self._metric_data._unit_indices,
            verbose=self._metric_data.verbose,
        )[0]
        isolation_distances_list = []
        for i in self._metric_data._unit_indices:
            isolation_distances_list.append(isolation_distances_all[i])
        isolation_distances = np.asarray(isolation_distances_list)
        if save_property_or_features:
            self.save_property_or_features(self._metric_data._sorting, isolation_distances, self._metric_name)
        return isolation_distances

    def threshold_metric(self, threshold, threshold_sign, num_channels_to_compare, max_spikes_per_cluster, **kwargs):
        isolation_distances = \
        self.compute_metric(num_channels_to_compare, max_spikes_per_cluster, **kwargs)
        threshold_curator = ThresholdCurator(
            sorting=self._metric_data._sorting, metric=isolation_distances
        )
        threshold_curator.threshold_sorting(
            threshold=threshold, threshold_sign=threshold_sign
        )
        return threshold_curator
