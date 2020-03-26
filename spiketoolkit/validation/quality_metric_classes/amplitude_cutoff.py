from .quality_metric import QualityMetric
import numpy as np
import spikemetrics.metrics as metrics
from .utils.thresholdcurator import ThresholdCurator
from collections import OrderedDict
from .parameter_dictionaries import get_amplitude_gui_params

def make_curator_gui_params(params):
    keys = list(params.keys())
    types = [type(params[key]) for key in keys]
    values = [params[key] for key in keys]
    gui_params = [{'name': keys[0], 'type': 'int', 'value': values[0], 'default': values[0], 'title': "Random seed for reproducibility"},
                  {'name': keys[1], 'type': str(types[1].__name__), 'value': values[1], 'default': values[1], 'title': "If True, will be verbose in metric computation."},]
    curator_gui_params =  [{'name': 'threshold', 'type': 'float', 'title': "The threshold for the given metric."},
                           {'name': 'threshold_sign', 'type': 'str',
                            'title': "If 'less', will threshold any metric less than the given threshold. "
                            "If 'less_or_equal', will threshold any metric less than or equal to the given threshold. "
                            "If 'greater', will threshold any metric greater than the given threshold. "
                            "If 'greater_or_equal', will threshold any metric greater than or equal to the given threshold."}]
    gui_params = curator_gui_params + gui_params + get_amplitude_gui_params()
    return gui_params


class AmplitudeCutoff(QualityMetric):
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # err
    params = OrderedDict([('seed',None), ('verbose',False)])
    curator_name = "ThresholdAmplitudeCutoff"
    curator_gui_params = make_curator_gui_params(params)
    def __init__(
        self,
        metric_data,
    ):
        QualityMetric.__init__(self, metric_data, metric_name="amplitude_cutoff")
        if not metric_data.has_amplitudes():
            raise ValueError("MetricData object must have amplitudes")

    def compute_metric(self, save_as_property):
        amplitude_cutoffs_epochs = []
        for epoch in self._metric_data._epochs:
            in_epoch = self._metric_data.get_in_epoch_bool_mask(epoch, self._metric_data._spike_times_amps)
            amplitude_cutoffs_all = metrics.calculate_amplitude_cutoff(
                self._metric_data._spike_clusters_amps[in_epoch],
                self._metric_data._amplitudes[in_epoch],
                self._metric_data._total_units,
                verbose=self._metric_data.verbose,
            )
            amplitude_cutoffs_list = []
            for i in self._metric_data._unit_indices:
                amplitude_cutoffs_list.append(amplitude_cutoffs_all[i])
            amplitude_cutoffs = np.asarray(amplitude_cutoffs_list)
            amplitude_cutoffs_epochs.append(amplitude_cutoffs)
        if save_as_property:
            self.save_as_property(self._metric_data._sorting, amplitude_cutoffs_epochs, self._metric_name)   
        return amplitude_cutoffs_epochs

    def threshold_metric(self, threshold, threshold_sign, save_as_property):
        amplitude_cutoff_epochs = self.compute_metric(save_as_property=save_as_property)[0]
        threshold_curator = ThresholdCurator(sorting=self._metric_data._sorting,
                                             metrics_epoch=amplitude_cutoff_epochs)
        threshold_curator.threshold_sorting(threshold=threshold, threshold_sign=threshold_sign)
        return threshold_curator
