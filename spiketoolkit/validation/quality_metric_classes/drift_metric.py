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
    gui_params = [{'name': keys[0], 'type': str(types[0].__name__), 'value': values[0], 'default': values[0], 'title': "Time period for evaluating drift."},
                  {'name': keys[1], 'type': str(types[1].__name__), 'value': values[1], 'default': values[1], 'title': "Minimum number of spikes for evaluating drift metrics per interval."},
                  {'name': keys[2], 'type': 'int', 'value': values[2], 'default': values[2], 'title': "Random seed for reproducibility"},
                  {'name': keys[3], 'type': str(types[3].__name__), 'value': values[3], 'default': values[3], 'title': "If True, will be verbose in metric computation."},]
    curator_gui_params =  [{'name': 'threshold', 'type': 'float', 'title': "The threshold for the given metric."},
                           {'name': 'threshold_sign', 'type': 'str',
                            'title': "If 'less', will threshold any metric less than the given threshold. "
                            "If 'less_or_equal', will threshold any metric less than or equal to the given threshold. "
                            "If 'greater', will threshold any metric greater than the given threshold. "
                            "If 'greater_or_equal', will threshold any metric greater than or equal to the given threshold."},
                           {'name': 'metric_name', 'type': 'str', 'value': "max_drift", 'default': "max_drift",
                            'title': "The name of the nearest neighbor metric to be thresholded (either 'max_drift' or 'cumulative_drift')."}]
    gui_params = curator_gui_params + gui_params + get_recording_gui_params() + get_feature_gui_params() + get_pca_scores_gui_params()
    return gui_params

class DriftMetric(QualityMetric):
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # err
    params = OrderedDict([('drift_metrics_interval_s',51), ('drift_metrics_min_spikes_per_interval',10), ('seed',None), ('verbose',False)])
    curator_name = "ThresholdDriftMetric"
    curator_gui_params = make_curator_gui_params(params)
    def __init__(self, metric_data):
        QualityMetric.__init__(self, metric_data, metric_name="drift_metric")

        if not metric_data.has_pca_scores():
            raise ValueError("MetricData object must have pca scores")

    def compute_metric(self, drift_metrics_interval_s, drift_metrics_min_spikes_per_interval, save_as_property):

        max_drifts_epochs = []
        cumulative_drifts_epochs = []
        for epoch in self._metric_data._epochs:
            start_frame = epoch[1]
            end_frame = epoch[2]
            if start_frame is None:
                start_frame = 0
            if end_frame is None:
                end_frame = np.inf
            in_epoch = np.logical_and(
                self._metric_data._spike_times_pca > start_frame, self._metric_data._spike_times_pca < end_frame
            )
            max_drifts_all, cumulative_drifts_all = metrics.calculate_drift_metrics(
                self._metric_data._spike_times_pca[in_epoch],
                self._metric_data._spike_clusters_pca[in_epoch],
                self._metric_data._total_units,
                self._metric_data._pc_features[in_epoch, :, :],
                self._metric_data._pc_feature_ind,
                drift_metrics_interval_s,
                drift_metrics_min_spikes_per_interval,
                verbose=self._metric_data.verbose,
            )
            max_drifts_list = []
            cumulative_drifts_list = []
            for i in self._metric_data._unit_indices:
                max_drifts_list.append(max_drifts_all[i])
                cumulative_drifts_list.append(cumulative_drifts_all[i])
            max_drifts = np.asarray(max_drifts_list)
            cumulative_drifts = np.asarray(cumulative_drifts_list)
            max_drifts_epochs.append(max_drifts)
            cumulative_drifts_epochs.append(cumulative_drifts)
        if save_as_property:
            self.save_as_property(self._metric_data._sorting, max_drifts_epochs, metric_name="max_drift")
            self.save_as_property(self._metric_data._sorting, cumulative_drifts_epochs, metric_name="cumulative_drift")
        return list(zip(max_drifts_epochs, cumulative_drifts_epochs))

    def threshold_metric(self, threshold, threshold_sign, metric_name, drift_metrics_interval_s, 
                         drift_metrics_min_spikes_per_interval, save_as_property):
        max_drifts_epochs, cumulative_drifts_epochs = self.compute_metric(drift_metrics_interval_s, drift_metrics_min_spikes_per_interval, 
                                                                          save_as_property)[0]
        if metric_name == "max_drift":
            metrics_epoch = max_drifts_epochs
        elif metric_name == "cumulative_drift":
            metrics_epoch = cumulative_drifts_epochs
        else:
            raise ValueError("Invalid metric named entered")
                                                                    
        threshold_curator = ThresholdCurator(
            sorting=self._metric_data._sorting, metrics_epoch=metrics_epoch
        )
        threshold_curator.threshold_sorting(
            threshold=threshold, threshold_sign=threshold_sign
        )
        return threshold_curator
