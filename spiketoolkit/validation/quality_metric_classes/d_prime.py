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

class DPrime(QualityMetric):
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # err
    params = OrderedDict([('num_channels_to_compare',13), ('max_spikes_per_cluster',500), ('seed',None), ('verbose',False)])
    curator_name = "ThresholdDPrime"
    curator_gui_params = make_curator_gui_params(params)
    def __init__(self, metric_data):
        QualityMetric.__init__(self, metric_data, metric_name="d_prime")

        if not metric_data.has_pca_scores():
            raise ValueError("MetricData object must have pca scores")

    def compute_metric(self, num_channels_to_compare, max_spikes_per_cluster, seed, save_as_property):

        d_primes_epochs = []
        for epoch in self._metric_data._epochs:
            in_epoch = np.logical_and(
                self._metric_data._spike_times_pca > epoch[1],
                self._metric_data._spike_times_pca < epoch[2],
            )
            d_primes_all = metrics.calculate_pc_metrics(
                spike_clusters=self._metric_data._spike_clusters_pca[in_epoch],
                total_units=self._metric_data._total_units,
                pc_features=self._metric_data._pc_features[in_epoch, :, :],
                pc_feature_ind=self._metric_data._pc_feature_ind,
                num_channels_to_compare=num_channels_to_compare,
                max_spikes_for_cluster=max_spikes_per_cluster,
                spikes_for_nn=None,
                n_neighbors=None,
                metric_names=["d_prime"],
                seed=seed,
                verbose=self._metric_data.verbose,
            )[2]
            d_primes_list = []
            for i in self._metric_data._unit_indices:
                d_primes_list.append(d_primes_all[i])
            d_primes = np.asarray(d_primes_list)
            d_primes_epochs.append(d_primes)
        if save_as_property:
            self.save_as_property(self._metric_data._sorting, d_primes_epochs, self._metric_name)
        return d_primes_epochs

    def threshold_metric(self, threshold, threshold_sign, num_channels_to_compare, max_spikes_per_cluster, seed, save_as_property):
        d_primes_epochs = self.compute_metric(num_channels_to_compare, max_spikes_per_cluster, seed, save_as_property=save_as_property)[0]
        threshold_curator = ThresholdCurator(
            sorting=self._metric_data._sorting, metrics_epoch=d_primes_epochs
        )
        threshold_curator.threshold_sorting(
            threshold=threshold, threshold_sign=threshold_sign
        )
        return threshold_curator
