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
                  {'name': keys[2], 'type': str(types[2].__name__), 'value': values[2], 'default': values[2], 'title': "Max spikes to be used for nearest-neighbors calculation."},
                  {'name': keys[3], 'type': str(types[3].__name__), 'value': values[3], 'default': values[3], 'title': "Number of neighbors to compare."},
                  {'name': keys[4], 'type': 'int', 'value': values[4], 'default': values[4], 'title': "Random seed for reproducibility"},
                  {'name': keys[5], 'type': str(types[5].__name__), 'value': values[5], 'default': values[5], 'title': "If True, will be verbose in metric computation."},]
    curator_gui_params =  [{'name': 'threshold', 'type': 'float', 'title': "The threshold for the given metric."},
                           {'name': 'threshold_sign', 'type': 'str',
                            'title': "If 'less', will threshold any metric less than the given threshold. "
                            "If 'less_or_equal', will threshold any metric less than or equal to the given threshold. "
                            "If 'greater', will threshold any metric greater than the given threshold. "
                            "If 'greater_or_equal', will threshold any metric greater than or equal to the given threshold."},
                           {'name': 'metric_name', 'type': 'str', 'value': "nn_hit_rate", 'default': "nn_hit_rate",
                            'title': "The name of the nearest neighbor metric to be thresholded (either 'nn_hit_rate' or 'nn_miss_rate')."}]
    gui_params = curator_gui_params + gui_params + get_recording_gui_params() + get_feature_gui_params() + get_pca_scores_gui_params()
    return gui_params

class NearestNeighbor(QualityMetric):
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # err
    params = OrderedDict([('num_channels_to_compare',13), ('max_spikes_per_cluster',500), ('max_spikes_for_nn', 10000),
                          ('n_neighbors',4), ('seed',None), ('verbose',False)])
    curator_name = "ThresholdNearestNeighbor"
    curator_gui_params = make_curator_gui_params(params)
    def __init__(self, metric_data):
        QualityMetric.__init__(self, metric_data, metric_name="nearest_neighbor")

        if not metric_data.has_pca_scores():
            raise ValueError("MetricData object must have pca scores")

    def compute_metric(self, num_channels_to_compare, max_spikes_per_cluster, max_spikes_for_nn,
                       n_neighbors, seed, save_as_property):

        nn_hit_rates_epochs = []
        nn_miss_rates_epochs = []
        for epoch in self._metric_data._epochs:
            in_epoch = self._metric_data.get_in_epoch_bool_mask(epoch, self._metric_data._spike_times_pca)
            spikes_in_epoch = np.sum(in_epoch)
            spikes_for_nn = np.min([spikes_in_epoch, max_spikes_for_nn])
            nn_hit_rates_all, nn_miss_rates_all = metrics.calculate_pc_metrics(
                spike_clusters=self._metric_data._spike_clusters_pca[in_epoch],
                total_units=self._metric_data._total_units,
                pc_features=self._metric_data._pc_features[in_epoch, :, :],
                pc_feature_ind=self._metric_data._pc_feature_ind,
                num_channels_to_compare=num_channels_to_compare,
                max_spikes_for_cluster=max_spikes_per_cluster,
                spikes_for_nn=spikes_for_nn,
                n_neighbors=n_neighbors,
                metric_names=["nearest_neighbor"],
                seed=seed,
                verbose=self._metric_data.verbose,
            )[3:5]
            nn_hit_rates_list = []
            nn_miss_rates_list = []
            for i in self._metric_data._unit_indices:
                nn_hit_rates_list.append(nn_hit_rates_all[i])
                nn_miss_rates_list.append(nn_miss_rates_all[i])
            nn_hit_rates = np.asarray(nn_hit_rates_list)
            nn_miss_rates = np.asarray(nn_miss_rates_list)
            nn_hit_rates_epochs.append(nn_hit_rates)
            nn_miss_rates_epochs.append(nn_miss_rates)
        if save_as_property:
            self.save_as_property(self._metric_data._sorting, nn_hit_rates_epochs, metric_name="nn_hit_rate")
            self.save_as_property(self._metric_data._sorting, nn_miss_rates_epochs, metric_name="nn_miss_rate")
        return list(zip(nn_hit_rates_epochs, nn_miss_rates_epochs))

    def threshold_metric(self, threshold, threshold_sign, metric_name, num_channels_to_compare, max_spikes_per_cluster, max_spikes_for_nn,
                         n_neighbors, seed, save_as_property):
        nn_hit_rates_epochs, nn_miss_rates_epochs = self.compute_metric(num_channels_to_compare, max_spikes_per_cluster, max_spikes_for_nn, 
                                                                        n_neighbors, seed, save_as_property=save_as_property)[0]
        if metric_name == "nn_hit_rate":
            metrics_epoch = nn_hit_rates_epochs
        elif metric_name == "nn_miss_rate":
            metrics_epoch = nn_miss_rates_epochs
        else:
            raise ValueError("Invalid metric named entered")
                                                                    
        threshold_curator = ThresholdCurator(
            sorting=self._metric_data._sorting, metrics_epoch=metrics_epoch
        )
        threshold_curator.threshold_sorting(
            threshold=threshold, threshold_sign=threshold_sign
        )
        return threshold_curator
