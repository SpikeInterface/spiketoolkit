import numpy as np
import spikemetrics.metrics as metrics
from .utils.thresholdcurator import ThresholdCurator
from .quality_metric import QualityMetric
from collections import OrderedDict
from .parameter_dictionaries import update_all_param_dicts_with_kwargs

class DriftMetric(QualityMetric):
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # err
    params = OrderedDict([('drift_metrics_interval_s', 51), ('drift_metrics_min_spikes_per_interval', 10)])
    curator_name = "ThresholdDriftMetric"

    def __init__(self, metric_data):
        QualityMetric.__init__(self, metric_data, metric_name="drift_metric")

        if not metric_data.has_pca_scores():
            raise ValueError("MetricData object must have pca scores")

    def compute_metric(self, drift_metrics_interval_s, drift_metrics_min_spikes_per_interval, **kwargs):
        params_dict = update_all_param_dicts_with_kwargs(kwargs)
        save_property_or_features = params_dict['save_property_or_features']
        max_drifts_all, cumulative_drifts_all = metrics.calculate_drift_metrics(
            self._metric_data._spike_times_pca,
            self._metric_data._spike_clusters_pca,
            self._metric_data._total_units,
            self._metric_data._pc_features,
            self._metric_data._pc_feature_ind,
            drift_metrics_interval_s,
            drift_metrics_min_spikes_per_interval,
            spike_cluster_subset=self._metric_data._unit_indices,
            verbose=self._metric_data.verbose,
        )
        max_drifts_list = []
        cumulative_drifts_list = []
        for i in self._metric_data._unit_indices:
            max_drifts_list.append(max_drifts_all[i])
            cumulative_drifts_list.append(cumulative_drifts_all[i])
        max_drifts = np.asarray(max_drifts_list)
        cumulative_drifts = np.asarray(cumulative_drifts_list)
        if save_property_or_features:
            self.save_property_or_features(self._metric_data._sorting, max_drifts, metric_name="max_drift")
            self.save_property_or_features(self._metric_data._sorting, cumulative_drifts, metric_name="cumulative_drift")
        return [max_drifts, cumulative_drifts]

    def threshold_metric(self, threshold, threshold_sign, metric_name, drift_metrics_interval_s,
                         drift_metrics_min_spikes_per_interval, **kwargs):
        max_drifts, cumulative_drifts = \
        self.compute_metric(drift_metrics_interval_s, drift_metrics_min_spikes_per_interval,
                            **kwargs)
        if metric_name == "max_drift":
            metric = max_drifts
        elif metric_name == "cumulative_drift":
            metric = cumulative_drifts
        else:
            raise ValueError("Invalid metric named entered")

        threshold_curator = ThresholdCurator(
            sorting=self._metric_data._sorting, metric=metric
        )
        threshold_curator.threshold_sorting(
            threshold=threshold, threshold_sign=threshold_sign
        )
        return threshold_curator
