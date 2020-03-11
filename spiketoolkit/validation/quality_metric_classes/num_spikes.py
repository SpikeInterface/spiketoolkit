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

class NumSpikes(QualityMetric):
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # err
    params = OrderedDict([('verbose',False)])
    curator_name = "ThresholdNumSpikes"
    curator_gui_params = make_curator_gui_params(params)
    def __init__(
        self,
        metric_data,
    ):
        QualityMetric.__init__(self, metric_data, metric_name="num_spikes")

    def compute_metric(self, save_as_property):
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
        if save_as_property:
            self.save_as_property(self._metric_data._sorting, num_spikes_epochs, self._metric_name)
        return num_spikes_epochs

    def threshold_metric(self, threshold, threshold_sign, save_as_property):
        num_spikes_epochs = self.compute_metric(save_as_property=save_as_property)[0]
        threshold_curator = ThresholdCurator(sorting=self._metric_data._sorting, metrics_epoch=num_spikes_epochs)
        threshold_curator.threshold_sorting(threshold=threshold, threshold_sign=threshold_sign)
        return threshold_curator