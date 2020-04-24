from .quality_metric import QualityMetric
import numpy as np
import spikemetrics.metrics as metrics
from .utils.thresholdcurator import ThresholdCurator
from collections import OrderedDict
from .parameter_dictionaries import update_all_param_dicts_with_kwargs


class AmplitudeCutoff(QualityMetric):
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # err
    curator_name = "ThresholdAmplitudeCutoff"
    params = OrderedDict([])

    def __init__(self, metric_data):
        QualityMetric.__init__(self, metric_data, metric_name="amplitude_cutoff")
        if not metric_data.has_amplitudes():
            raise ValueError("MetricData object must have amplitudes")

    def compute_metric(self, **kwargs):
        params_dict = update_all_param_dicts_with_kwargs(kwargs)
        save_property_or_features = params_dict['save_property_or_features']
        amplitude_cutoffs_all = metrics.calculate_amplitude_cutoff(
            self._metric_data._spike_clusters_amps,
            self._metric_data._amplitudes,
            self._metric_data._total_units,
            spike_cluster_subset=self._metric_data._unit_indices,
            verbose=self._metric_data.verbose,
        )
        amplitude_cutoffs_list = []
        for i in self._metric_data._unit_indices:
            amplitude_cutoffs_list.append(amplitude_cutoffs_all[i])
        amplitude_cutoffs = np.asarray(amplitude_cutoffs_list)
        if save_property_or_features:
            self.save_property_or_features(self._metric_data._sorting, amplitude_cutoffs, self._metric_name)
        return amplitude_cutoffs

    def threshold_metric(self, threshold, threshold_sign, **kwargs):
        amplitude_cutoffs = self.compute_metric(**kwargs)
        threshold_curator = ThresholdCurator(sorting=self._metric_data._sorting,
                                             metric=amplitude_cutoffs)
        threshold_curator.threshold_sorting(threshold=threshold, threshold_sign=threshold_sign)
        return threshold_curator
