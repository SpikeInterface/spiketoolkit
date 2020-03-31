import numpy as np
import spikemetrics.metrics as metrics
from .utils.thresholdcurator import ThresholdCurator
from .quality_metric import QualityMetric
from collections import OrderedDict
from .parameter_dictionaries import get_recording_gui_params, get_feature_gui_params, get_pca_scores_gui_params

def make_curator_gui_params(params):
    keys = list(params.keys())
    types = [type(params[key]) for key in keys]
    values = [params[key] for key in keys]
    gui_params = [{'name': keys[0], 'type': str(types[0].__name__), 'value': values[0], 'default': values[0], 'title': "Max spikes to be used for silhouette metric."},
                  {'name': keys[1], 'type': 'int', 'value': values[1], 'default': values[1], 'title': "Random seed for reproducibility"},
                  {'name': keys[2], 'type': str(types[2].__name__), 'value': values[2], 'default': values[2], 'title': "If True, will be verbose in metric computation."},]
    curator_gui_params =  [{'name': 'threshold', 'type': 'float', 'title': "The threshold for the given metric."},
                           {'name': 'threshold_sign', 'type': 'str',
                            'title': "If 'less', will threshold any metric less than the given threshold. "
                            "If 'less_or_equal', will threshold any metric less than or equal to the given threshold. "
                            "If 'greater', will threshold any metric greater than the given threshold. "
                            "If 'greater_or_equal', will threshold any metric greater than or equal to the given threshold."}]
    gui_params = curator_gui_params + gui_params + get_recording_gui_params() + get_feature_gui_params() + get_pca_scores_gui_params()
    return gui_params

class SilhouetteScore(QualityMetric):
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # err
    params = OrderedDict([('max_spikes_for_silhouette',10000), ('seed', None), ('verbose', False)])
    curator_name = "ThresholdSilhouetteScore"
    curator_gui_params = make_curator_gui_params(params)
    def __init__(self, metric_data):
        QualityMetric.__init__(self, metric_data, metric_name="silhouette_score")

        if not metric_data.has_pca_scores():
            raise ValueError("MetricData object must have pca scores")

    def compute_metric(self, max_spikes_for_silhouette, seed, save_as_property):

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
        if save_as_property:
            self.save_as_property(self._metric_data._sorting, silhouette_scores_epochs, self._metric_name)
        return silhouette_scores_epochs

    def threshold_metric(self, threshold, threshold_sign, max_spikes_for_silhouette, seed, save_as_property):
        silhouette_scores_epochs = self.compute_metric(max_spikes_for_silhouette, seed, save_as_property=save_as_property)[0]
        threshold_curator = ThresholdCurator(
            sorting=self._metric_data._sorting, metrics_epoch=silhouette_scores_epochs
        )
        threshold_curator.threshold_sorting(
            threshold=threshold, threshold_sign=threshold_sign
        )
        return threshold_curator
