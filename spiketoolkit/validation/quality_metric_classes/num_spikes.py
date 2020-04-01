from .quality_metric import QualityMetric
import numpy as np
import spikemetrics.metrics as metrics
from .utils.thresholdcurator import ThresholdCurator
from collections import OrderedDict
from .parameter_dictionaries import update_all_param_dicts_with_kwargs


class NumSpikes(QualityMetric):
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # err
    params = OrderedDict()
    curator_name = "ThresholdNumSpikes"

    def __init__(
            self,
            metric_data,
    ):
        QualityMetric.__init__(self, metric_data, metric_name="num_spikes")

    def compute_metric(self, **kwargs):
        params_dict = update_all_param_dicts_with_kwargs(kwargs)
        save_property_or_features = params_dict['save_property_or_features']

        num_spikes_epochs = []
        for epoch in self._metric_data._epochs:
            in_epoch = self._metric_data.get_in_epoch_bool_mask(epoch, self._metric_data._spike_times)
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
        if save_property_or_features:
            self.save_property_or_features(self._metric_data._sorting, num_spikes_epochs, self._metric_name)
        return num_spikes_epochs

    def threshold_metric(self, threshold, threshold_sign, **kwargs):
        num_spikes_epochs = self.compute_metric(**kwargs)[0]
        threshold_curator = ThresholdCurator(sorting=self._metric_data._sorting, metrics_epoch=num_spikes_epochs)
        threshold_curator.threshold_sorting(threshold=threshold, threshold_sign=threshold_sign)
        return threshold_curator
