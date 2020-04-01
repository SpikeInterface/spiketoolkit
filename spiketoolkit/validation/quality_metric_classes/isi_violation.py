from .quality_metric import QualityMetric
import numpy as np
import spikemetrics.metrics as metrics
from .utils.thresholdcurator import ThresholdCurator
from collections import OrderedDict
from .parameter_dictionaries import update_all_param_dicts_with_kwargs


class ISIViolation(QualityMetric):
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # err
    params = OrderedDict([('isi_threshold', 0.0015), ('min_isi', 0.000166)])
    curator_name = "ThresholdISIViolation"

    def __init__(
            self,
            metric_data,
    ):
        QualityMetric.__init__(self, metric_data, metric_name="isi_viol")

    def compute_metric(self, isi_threshold, min_isi, **kwargs):
        params_dict = update_all_param_dicts_with_kwargs(kwargs)
        save_property_or_features = params_dict['save_property_or_features']
        isi_violation_epochs = []
        for epoch in self._metric_data._epochs:
            in_epoch = self._metric_data.get_in_epoch_bool_mask(epoch, self._metric_data._spike_times)
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
        if save_property_or_features:
            self.save_property_or_features(self._metric_data._sorting, isi_violation_epochs, self._metric_name)
        return isi_violation_epochs

    def threshold_metric(self, threshold, threshold_sign, isi_threshold, min_isi, **kwargs):
        isi_violation_epochs = self.compute_metric(isi_threshold, min_isi, **kwargs)[0]
        threshold_curator = ThresholdCurator(sorting=self._metric_data._sorting, metrics_epoch=isi_violation_epochs)
        threshold_curator.threshold_sorting(threshold=threshold, threshold_sign=threshold_sign)
        return threshold_curator
