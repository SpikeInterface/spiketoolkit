from .quality_metric import QualityMetric
import numpy as np
import spikemetrics.metrics as metrics
from spiketoolkit.curation.thresholdcurator import ThresholdCurator

class FiringRate(QualityMetric):
    def __init__(
        self,
        metric_data,
    ):
        QualityMetric.__init__(self, metric_data, metric_name="firing_rate")

    def compute_metric(self, save_as_property):
        firing_rate_epochs = []
        for epoch in self._metric_data._epochs:
            in_epoch = np.logical_and(
                self._metric_data._spike_times > epoch[1], self._metric_data._spike_times < epoch[2]
            )
            firing_rate_all,_ = metrics.calculate_firing_rate_and_spikes(
                self._metric_data._spike_times[in_epoch],
                self._metric_data._spike_clusters[in_epoch],
                self._metric_data._total_units,
                verbose=self._metric_data.verbose,
            )
            firing_rate_list = []
            for i in self._metric_data._unit_indices:
                firing_rate_list.append(firing_rate_all[i])
            firing_rate = np.asarray(firing_rate_list)
            firing_rate_epochs.append(firing_rate)
        if save_as_property:
            self.save_as_property(self._metric_data._sorting, firing_rate_epochs, self._metric_name)
        return firing_rate_epochs

    def threshold_metric(self, threshold, threshold_sign, epoch, save_as_property):
        assert (epoch < len(self._metric_data.get_epochs())), "Invalid epoch specified"
        firing_rate_epochs = self.compute_metric(save_as_property=save_as_property)[epoch]
        threshold_curator = ThresholdCurator(sorting=self._metric_data._sorting, metrics_epoch=firing_rate_epochs)
        threshold_curator.threshold_sorting(threshold=threshold, threshold_sign=threshold_sign)
        return threshold_curator
