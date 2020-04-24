from .quality_metric import QualityMetric
import numpy as np
import spikemetrics.metrics as metrics
from .utils.thresholdcurator import ThresholdCurator
from collections import OrderedDict
from .parameter_dictionaries import update_all_param_dicts_with_kwargs


class FiringRate(QualityMetric):
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # err
    params = OrderedDict()
    curator_name = "ThresholdFiringRate"

    def __init__(
            self,
            metric_data,
    ):
        QualityMetric.__init__(self, metric_data, metric_name="firing_rate")

    def compute_metric(self, **kwargs):
        params_dict = update_all_param_dicts_with_kwargs(kwargs)
        save_property_or_features = params_dict['save_property_or_features']
        firing_rate_all = metrics.calculate_firing_rates(
            self._metric_data._spike_times,
            self._metric_data._spike_clusters,
            self._metric_data._total_units,
            duration=self._metric_data._duration_in_frames/self._metric_data._sampling_frequency,
            spike_cluster_subset=self._metric_data._unit_indices,
            verbose=self._metric_data.verbose,
        )
        firing_rate_list = []
        for i in self._metric_data._unit_indices:
            firing_rate_list.append(firing_rate_all[i])
        firing_rate = np.asarray(firing_rate_list)
        if save_property_or_features:
            self.save_property_or_features(self._metric_data._sorting, firing_rate, self._metric_name)
        return firing_rate

    def threshold_metric(self, threshold, threshold_sign, **kwargs):
        firing_rate = self.compute_metric(**kwargs)
        threshold_curator = ThresholdCurator(sorting=self._metric_data._sorting, metric=firing_rate)
        threshold_curator.threshold_sorting(threshold=threshold, threshold_sign=threshold_sign)
        return threshold_curator
