from .quality_metric import QualityMetric
import numpy as np
import spikemetrics.metrics as metrics
from .utils.thresholdcurator import ThresholdCurator
from collections import OrderedDict

def make_curator_gui_params(params):
    keys = list(params.keys())
    types = [type(params[key]) for key in keys]
    values = [params[key] for key in keys]
    gui_params = [{'name': keys[0], 'type': str(types[0].__name__), 'value': values[0], 'default': values[0], 'title': "If True, will be verbose in metric computation."}]
    curator_gui_params =  [{'name': 'threshold', 'type': 'float', 'title': "The threshold for the given metric."},
                           {'name': 'threshold_sign', 'type': 'str',
                            'title': "If 'less', will threshold any metric less than the given threshold. "
                            "If 'less_or_equal', will threshold any metric less than or equal to the given threshold. "
                            "If 'greater', will threshold any metric greater than the given threshold. "
                            "If 'greater_or_equal', will threshold any metric greater than or equal to the given threshold."}]
    gui_params = curator_gui_params + gui_params
    return gui_params

class FiringRate(QualityMetric):
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # err
    params = OrderedDict([('verbose',False)])
    curator_name = "ThresholdFiringRate"
    curator_gui_params = make_curator_gui_params(params)
    def __init__(
        self,
        metric_data,
    ):
        QualityMetric.__init__(self, metric_data, metric_name="firing_rate")

    def compute_metric(self, save_as_property):
        firing_rate_epochs = []
        for epoch in self._metric_data._epochs:
            in_epoch = np.logical_and(
                self._metric_data._spike_times > epoch[1], self._metric_data._spike_times < epoch[2]
            )
            firing_rate_all,_ = metrics.calculate_firing_rate_and_spikes(
                self._metric_data._spike_times[in_epoch],
                self._metric_data._spike_clusters[in_epoch],
                self._metric_data._total_units,
                verbose=self._metric_data.verbose,
            )
            firing_rate_list = []
            for i in self._metric_data._unit_indices:
                firing_rate_list.append(firing_rate_all[i])
            firing_rate = np.asarray(firing_rate_list)
            firing_rate_epochs.append(firing_rate)
        if save_as_property:
            self.save_as_property(self._metric_data._sorting, firing_rate_epochs, self._metric_name)
        return firing_rate_epochs

    def threshold_metric(self, threshold, threshold_sign, save_as_property):
        firing_rate_epochs = self.compute_metric(save_as_property=save_as_property)[0]
        threshold_curator = ThresholdCurator(sorting=self._metric_data._sorting, metrics_epoch=firing_rate_epochs)
        threshold_curator.threshold_sorting(threshold=threshold, threshold_sign=threshold_sign)
        return threshold_curator
