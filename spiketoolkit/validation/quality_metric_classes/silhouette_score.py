import numpy as np
import spikemetrics.metrics as metrics
from spiketoolkit.curation.thresholdcurator import ThresholdCurator
from .quality_metric import QualityMetric


class SilhouetteScore(QualityMetric):
    def __init__(self, metric_data):
        QualityMetric.__init__(self, metric_data, metric_name="silhouette_score")

        if not metric_data.has_pca_scores():
            raise ValueError("MetricData object must have pca scores")

    def compute_metric(self, max_spikes_for_silhouette=10000, seed=None, save_as_property=True):

        silhouette_scores_epochs = []
        for epoch in self._metric_data._epochs:
            in_epoch = np.logical_and(
                self._metric_data._spike_times_pca > epoch[1],
                self._metric_data._spike_times_pca < epoch[2],
            )
            spikes_in_epoch = np.sum(in_epoch)
            spikes_for_silhouette = np.min([spikes_in_epoch, max_spikes_for_silhouette])

            silhouette_scores_all = metrics.calculate_silhouette_score(
                self._metric_data._spike_clusters_pca[in_epoch],
                self._metric_data._total_units,
                self._metric_data._pc_features[in_epoch, :, :],
                self._metric_data._pc_feature_ind,
                spikes_for_silhouette,
                seed=seed,
                verbose=self._metric_data.verbose,
            )
            silhouette_scores_list = []
            for index in self._metric_data._unit_indices:
                silhouette_scores_list.append(silhouette_scores_all[index])
            silhouette_scores = np.asarray(silhouette_scores_list)
            silhouette_scores_epochs.append(silhouette_scores)
        if save_as_property:
            self.save_as_property(self._metric_data._sorting, silhouette_scores_epochs)
        return silhouette_scores_epochs

    def threshold_metric(self, threshold, threshold_sign, epoch=0, max_spikes_for_silhouette=10000, seed=None, save_as_property=True):

        assert epoch < len(self._metric_data.get_epochs()), "Invalid epoch specified"

        silhouette_scores_epochs = self.compute_metric(max_spikes_for_silhouette, seed, save_as_property=save_as_property)[epoch]
        threshold_curator = ThresholdCurator(
            sorting=self._metric_data._sorting, metrics_epoch=silhouette_scores_epochs
        )
        threshold_curator.threshold_sorting(
            threshold=threshold, threshold_sign=threshold_sign
        )
        return threshold_curator
