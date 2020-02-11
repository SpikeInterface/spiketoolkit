from .quality_metric import QualityMetric
import numpy as np
import spikemetrics.metrics as metrics
from spiketoolkit.curation.thresholdcurator import ThresholdCurator

class ISIViolation(QualityMetric):
    def __init__(
        self,
        metric_data,
    ):
        QualityMetric.__init__(self, metric_data, metric_name="isi_viol")

    def compute_metric(self, isi_threshold, min_isi, save_as_property):
        isi_violation_epochs = []
        for epoch in self._metric_data._epochs:
            in_epoch = np.logical_and(
                self._metric_data._spike_times > epoch[1], self._metric_data._spike_times < epoch[2]
            )
            isi_violation_all = metrics.calculate_isi_violations(
                self._metric_data._spike_times[in_epoch],
                self._metric_data._spike_clusters[in_epoch],
                self._metric_data._total_units,
                isi_threshold=isi_threshold,
                min_isi=min_isi,
                verbose=self._metric_data.verbose,
            )
            isi_violation_list = []
            for i in self._metric_data._unit_indices:
                isi_violation_list.append(isi_violation_all[i])
            isi_violation = np.asarray(isi_violation_list)
            isi_violation_epochs.append(isi_violation)
        if save_as_property:
            self.save_as_property(self._metric_data._sorting, isi_violation_epochs, self._metric_name)
        return isi_violation_epochs

    def threshold_metric(self, threshold, threshold_sign, epoch, isi_threshold, min_isi, save_as_property):
        assert (epoch < len(self._metric_data.get_epochs())), "Invalid epoch specified"
        isi_violation_epochs = self.compute_metric(isi_threshold, min_isi, save_as_property=save_as_property)[epoch]
        threshold_curator = ThresholdCurator(sorting=self._metric_data._sorting, metrics_epoch=isi_violation_epochs)
        threshold_curator.threshold_sorting(threshold=threshold, threshold_sign=threshold_sign)
        return threshold_curator
