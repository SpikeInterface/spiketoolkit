import numpy as np
import spikemetrics.metrics as metrics
from .utils.thresholdcurator import ThresholdCurator
from .quality_metric import QualityMetric
from collections import OrderedDict
from .parameter_dictionaries import update_all_param_dicts_with_kwargs


class SilhouetteScore(QualityMetric):
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # err
    params = OrderedDict([('max_spikes_for_silhouette', 10000)])
    curator_name = "ThresholdSilhouetteScore"

    def __init__(self, metric_data):
        QualityMetric.__init__(self, metric_data, metric_name="silhouette_score")

        if not metric_data.has_pca_scores():
            raise ValueError("MetricData object must have pca scores")

    def compute_metric(self, max_spikes_for_silhouette, **kwargs):
        params_dict = update_all_param_dicts_with_kwargs(kwargs)
        save_property_or_features = params_dict['save_property_or_features']
        seed = params_dict['seed']

        silhouette_scores_epochs = []
        for epoch in self._metric_data._epochs:
            in_epoch = self._metric_data.get_in_epoch_bool_mask(epoch, self._metric_data._spike_times_pca)
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
        if save_property_or_features:
            self.save_property_or_features(self._metric_data._sorting, silhouette_scores_epochs, self._metric_name)
        return silhouette_scores_epochs

    def threshold_metric(self, threshold, threshold_sign, max_spikes_for_silhouette, **kwargs):
        silhouette_scores_epochs = \
        self.compute_metric(max_spikes_for_silhouette, **kwargs)[0]
        threshold_curator = ThresholdCurator(
            sorting=self._metric_data._sorting, metrics_epoch=silhouette_scores_epochs
        )
        threshold_curator.threshold_sorting(
            threshold=threshold, threshold_sign=threshold_sign
        )
        return threshold_curator
