from .quality_metric import QualityMetric
import numpy as np
import spikemetrics.metrics as metrics
from .utils.thresholdcurator import ThresholdCurator
from collections import OrderedDict

def make_curator_gui_params(params):
    keys = list(params.keys())
    types = [type(params[key]) for key in keys]
    values = [params[key] for key in keys]
    gui_params = [{'name': keys[0], 'type': str(types[0].__name__), 'value': values[0], 'default': values[0], 'title': "The isi threshold for calculating isi violations."},
                  {'name': keys[1], 'type': str(types[1].__name__), 'value': values[1], 'default': values[1], 'title': " The minimum expected isi value."},
                  {'name': keys[2], 'type': str(types[2].__name__), 'value': values[2], 'default': values[2], 'title': "If True, will be verbose in metric computation."}]
    curator_gui_params =  [{'name': 'threshold', 'type': 'float', 'title': "The threshold for the given metric."},
                           {'name': 'threshold_sign', 'type': 'str',
                            'title': "If 'less', will threshold any metric less than the given threshold. "
                            "If 'less_or_equal', will threshold any metric less than or equal to the given threshold. "
                            "If 'greater', will threshold any metric greater than the given threshold. "
                            "If 'greater_or_equal', will threshold any metric greater than or equal to the given threshold."}]
    gui_params = curator_gui_params + gui_params
    return gui_params

class ISIViolation(QualityMetric):
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # err
    params = OrderedDict([('isi_threshold',0.0015),('min_isi',0.000166),('verbose',False)])
    curator_name = "ThresholdISIViolation"
    curator_gui_params = make_curator_gui_params(params)
    def __init__(
        self,
        metric_data,
    ):
        QualityMetric.__init__(self, metric_data, metric_name="isi_viol")

    def compute_metric(self, isi_threshold, min_isi, save_as_property):
        isi_violation_epochs = []
        for epoch in self._metric_data._epochs:
            start_frame = epoch[1]
            end_frame = epoch[2]
            if start_frame is None:
                start_frame = 0
            if end_frame is None:
                end_frame = np.inf
            in_epoch = np.logical_and(
                self._metric_data._spike_times > start_frame, self._metric_data._spike_times < end_frame
            )
            isi_violation_all = metrics.calculate_isi_violations(
                self._metric_data._spike_times[in_epoch],
                self._metric_data._spike_clusters[in_epoch],
                self._metric_data._total_units,
                isi_threshold=isi_threshold,
                min_isi=min_isi,
                verbose=self._metric_data.verbose,
            )
            isi_violation_list = []
            for i in self._metric_data._unit_indices:
                isi_violation_list.append(isi_violation_all[i])
            isi_violation = np.asarray(isi_violation_list)
            isi_violation_epochs.append(isi_violation)
        if save_as_property:
            self.save_as_property(self._metric_data._sorting, isi_violation_epochs, self._metric_name)
        return isi_violation_epochs

    def threshold_metric(self, threshold, threshold_sign, isi_threshold, min_isi, save_as_property):
        isi_violation_epochs = self.compute_metric(isi_threshold, min_isi, save_as_property=save_as_property)[0]
        threshold_curator = ThresholdCurator(sorting=self._metric_data._sorting, metrics_epoch=isi_violation_epochs)
        threshold_curator.threshold_sorting(threshold=threshold, threshold_sign=threshold_sign)
        return threshold_curator
