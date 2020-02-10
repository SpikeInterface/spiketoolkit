from .quality_metric import QualityMetric
import numpy as np
import spikemetrics.metrics as metrics
from spiketoolkit.curation.thresholdcurator import ThresholdCurator

class PresenceRatio(QualityMetric):
    def __init__(
        self,
        metric_data,
    ):
        QualityMetric.__init__(self, metric_data, metric_name="presence_ratio")

    def compute_metric(self, save_as_property=True):
        presence_ratios_epochs = []
        for epoch in self._metric_data._epochs:
            in_epoch = np.logical_and(
                self._metric_data._spike_times > epoch[1], self._metric_data._spike_times < epoch[2]
            )
            presence_ratios_all = metrics.calculate_presence_ratio(
                self._metric_data._spike_times[in_epoch],
                self._metric_data._spike_clusters[in_epoch],
                self._metric_data._total_units,
                verbose=self._metric_data.verbose,
            )
            presence_ratios_list = []
            for i in self._metric_data._unit_indices:
                presence_ratios_list.append(presence_ratios_all[i])
            presence_ratios = np.asarray(presence_ratios_list)
            presence_ratios_epochs.append(presence_ratios)
        if save_as_property:
            self.save_as_property(self._metric_data._sorting, presence_ratios_epochs)
        return presence_ratios_epochs

    def threshold_metric(self, threshold, threshold_sign, epoch=0, save_as_property=True):
        assert (epoch < len(self._metric_data.get_epochs())), "Invalid epoch specified"
        presence_ratios_epochs = self.compute_metric(save_as_property=save_as_property)[epoch]
        threshold_curator = ThresholdCurator(sorting=self._metric_data._sorting, metrics_epoch=presence_ratios_epochs)
        threshold_curator.threshold_sorting(threshold=threshold, threshold_sign=threshold_sign)
        return threshold_curator
