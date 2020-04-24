from .quality_metric import QualityMetric
import numpy as np
import spikemetrics.metrics as metrics
from .utils.thresholdcurator import ThresholdCurator
from collections import OrderedDict
from .parameter_dictionaries import update_all_param_dicts_with_kwargs


class ISIViolation(QualityMetric):
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # err
    params = OrderedDict([('isi_threshold', 0.0015), ('min_isi', None)])
    curator_name = "ThresholdISIViolation"

    def __init__(
            self,
            metric_data,
    ):
        QualityMetric.__init__(self, metric_data, metric_name="isi_violation")

    def compute_metric(self, isi_threshold, min_isi, **kwargs):
        params_dict = update_all_param_dicts_with_kwargs(kwargs)
        save_property_or_features = params_dict['save_property_or_features']
        if min_isi is None:
            min_isi = 1 / (self._metric_data._sampling_frequency) * 0.5
        isi_violation_all = metrics.calculate_isi_violations(
            self._metric_data._spike_times,
            self._metric_data._spike_clusters,
            self._metric_data._total_units,
            isi_threshold=isi_threshold,
            min_isi=min_isi,
            duration=self._metric_data._duration_in_frames/self._metric_data._sampling_frequency,
            spike_cluster_subset=self._metric_data._unit_indices,
            verbose=self._metric_data.verbose,
        )
        isi_violation_list = []
        for i in self._metric_data._unit_indices:
            isi_violation_list.append(isi_violation_all[i])
        isi_violations = np.asarray(isi_violation_list)
        if save_property_or_features:
            self.save_property_or_features(self._metric_data._sorting, isi_violations, self._metric_name)
        return isi_violations

    def threshold_metric(self, threshold, threshold_sign, isi_threshold, min_isi, **kwargs):
        isi_violations = self.compute_metric(isi_threshold, min_isi, **kwargs)
        threshold_curator = ThresholdCurator(sorting=self._metric_data._sorting, metric=isi_violations)
        threshold_curator.threshold_sorting(threshold=threshold, threshold_sign=threshold_sign)
        return threshold_curator
