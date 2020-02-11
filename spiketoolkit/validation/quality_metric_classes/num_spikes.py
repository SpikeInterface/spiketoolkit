from .quality_metric import QualityMetric
import numpy as np
import spikemetrics.metrics as metrics
from spiketoolkit.curation.thresholdcurator import ThresholdCurator

class NumSpikes(QualityMetric):
    def __init__(
        self,
        metric_data,
    ):
        QualityMetric.__init__(self, metric_data, metric_name="num_spikes")

    def compute_metric(self, save_as_property):
        num_spikes_epochs = []
        for epoch in self._metric_data._epochs:
            in_epoch = np.logical_and(
                self._metric_data._spike_times > epoch[1], self._metric_data._spike_times < epoch[2]
            )
            _, num_spikes_all = metrics.calculate_firing_rate_and_spikes(
                self._metric_data._spike_times[in_epoch],
                self._metric_data._spike_clusters[in_epoch],
                self._metric_data._total_units,
                verbose=self._metric_data.verbose,
            )
            num_spikes_list = []
            for i in self._metric_data._unit_indices:
                num_spikes_list.append(num_spikes_all[i])
            num_spikes = np.asarray(num_spikes_list)
            num_spikes_epochs.append(num_spikes)
        if save_as_property:
            self.save_as_property(self._metric_data._sorting, num_spikes_epochs, self._metric_name)
        return num_spikes_epochs

    def threshold_metric(self, threshold, threshold_sign, epoch, save_as_property):
        assert (epoch < len(self._metric_data.get_epochs())), "Invalid epoch specified"
        num_spikes_epochs = self.compute_metric(save_as_property=save_as_property)[epoch]
        threshold_curator = ThresholdCurator(sorting=self._metric_data._sorting, metrics_epoch=num_spikes_epochs)
        threshold_curator.threshold_sorting(threshold=threshold, threshold_sign=threshold_sign)
        return threshold_curator
