from .quality_metric import QualityMetric
import numpy as np
import spikemetrics.metrics as metrics
from .utils.thresholdcurator import ThresholdCurator
from collections import OrderedDict
from .parameter_dictionaries import update_all_param_dicts_with_kwargs


class PresenceRatio(QualityMetric):
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # err
    params = OrderedDict()
    curator_name = "ThresholdPresenceRatio"

    def __init__(
            self,
            metric_data,
    ):
        QualityMetric.__init__(self, metric_data, metric_name="presence_ratio")

    def compute_metric(self, **kwargs):
        params_dict = update_all_param_dicts_with_kwargs(kwargs)
        save_property_or_features = params_dict['save_property_or_features']

        presence_ratios_epochs = []
        for epoch in self._metric_data._epochs:
            in_epoch = self._metric_data.get_in_epoch_bool_mask(epoch, self._metric_data._spike_times)
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
        if save_property_or_features:
            self.save_property_or_features(self._metric_data._sorting, presence_ratios_epochs, self._metric_name)
        return presence_ratios_epochs

    def threshold_metric(self, threshold, threshold_sign, **kwargs):
        presence_ratios_epochs = self.compute_metric(**kwargs)[0]
        threshold_curator = ThresholdCurator(sorting=self._metric_data._sorting, metrics_epoch=presence_ratios_epochs)
        threshold_curator.threshold_sorting(threshold=threshold, threshold_sign=threshold_sign)
        return threshold_curator
