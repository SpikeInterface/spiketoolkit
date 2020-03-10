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
    gui_params = [{'name': keys[0], 'type': str(types[0].__name__), 'value': values[0], 'default': values[0], 'title': "The number of channels to be used for the PC extraction and comparison."},
                  {'name': keys[1], 'type': str(types[1].__name__), 'value': values[1], 'default': values[1], 'title': "Max spikes to be used from each unit"},
                  {'name': keys[2], 'type': 'int', 'value': values[2], 'default': values[2], 'title': "Random seed for reproducibility"},
                  {'name': keys[3], 'type': str(types[3].__name__), 'value': values[3], 'default': values[3], 'title': "If True, will be verbose in metric computation."},]
    curator_gui_params =  [{'name': 'threshold', 'type': 'float', 'title': "The threshold for the given metric."},
                           {'name': 'threshold_sign', 'type': 'str',
                            'title': "If 'less', will threshold any metric less than the given threshold. "
                            "If 'less_or_equal', will threshold any metric less than or equal to the given threshold. "
                            "If 'greater', will threshold any metric greater than the given threshold. "
                            "If 'greater_or_equal', will threshold any metric greater than or equal to the given threshold."}]
    gui_params = curator_gui_params + gui_params + get_recording_gui_params() + get_feature_gui_params() + get_pca_scores_gui_params()
    return gui_params

class LRatio(QualityMetric):
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # err
    params = OrderedDict([('num_channels_to_compare',13), ('max_spikes_per_cluster',500), ('seed',None), ('verbose',False)])
    curator_name = "ThresholdLRatio"
    curator_gui_params = make_curator_gui_params(params)
    def __init__(self, metric_data):
        QualityMetric.__init__(self, metric_data, metric_name="l_ratio")

        if not metric_data.has_pca_scores():
            raise ValueError("MetricData object must have pca scores")

    def compute_metric(self, num_channels_to_compare, max_spikes_per_cluster, seed, save_as_property):

        l_ratios_epochs = []
        for epoch in self._metric_data._epochs:
            start_frame = epoch[1]
            end_frame = epoch[2]
            if start_frame is None:
                start_frame = 0
            if end_frame is None:
                end_frame = np.inf
            in_epoch = np.logical_and(
                self._metric_data._spike_times_pca > start_frame,
                self._metric_data._spike_times_pca < end_frame,
            )
            l_ratios_all = metrics.calculate_pc_metrics(
                spike_clusters=self._metric_data._spike_clusters_pca[in_epoch],
                total_units=self._metric_data._total_units,
                pc_features=self._metric_data._pc_features[in_epoch, :, :],
                pc_feature_ind=self._metric_data._pc_feature_ind,
                num_channels_to_compare=num_channels_to_compare,
                max_spikes_for_cluster=max_spikes_per_cluster,
                spikes_for_nn=None,
                n_neighbors=None,
                metric_names=["l_ratio"],
                seed=seed,
                verbose=self._metric_data.verbose,
            )[1]
            l_ratios_list = []
            for i in self._metric_data._unit_indices:
                l_ratios_list.append(l_ratios_all[i])
            l_ratios = np.asarray(l_ratios_list)
            l_ratios_epochs.append(l_ratios)
        if save_as_property:
            self.save_as_property(self._metric_data._sorting, l_ratios_epochs, self._metric_name)
        return l_ratios_epochs

    def threshold_metric(self, threshold, threshold_sign, num_channels_to_compare, max_spikes_per_cluster, seed, save_as_property):
        l_ratios_epochs = self.compute_metric(num_channels_to_compare, max_spikes_per_cluster, seed, save_as_property=save_as_property)[0]
        threshold_curator = ThresholdCurator(
            sorting=self._metric_data._sorting, metrics_epoch=l_ratios_epochs
        )
        threshold_curator.threshold_sorting(
            threshold=threshold, threshold_sign=threshold_sign
        )
        return threshold_curator
