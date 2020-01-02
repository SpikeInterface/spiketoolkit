from .quality_metric import QualityMetric
import numpy as np
import spikemetrics.metrics as metrics
from spiketoolkit.curation.thresholdcurator import ThresholdCurator

class AmplitudeCutoff(QualityMetric):
    def __init__(
        self,
        sorting,
        recording,
        apply_filter=True,
        freq_min=300,
        freq_max=6000,
        unit_ids=None,
        epoch_tuples=None,
        epoch_names=None,
        verbose=False,
        amp_method="absolute",
        amp_peak="both",
        amp_frames_before=3,
        amp_frames_after=3,
        save_features_props=False,
        seed=0,
    ):

        QualityMetric.__init__(self, sorting=sorting, recording=recording, apply_filter=apply_filter,
                                freq_min=freq_min, freq_max=freq_max, unit_ids=unit_ids, epoch_tuples=epoch_tuples,
                                verbose=verbose)
        self.compute_amplitudes(amp_method=amp_method, amp_peak=amp_peak, amp_frames_before=amp_frames_before, 
                                amp_frames_after=amp_frames_after, save_features_props=save_features_props, seed=seed)

    def compute_metric(self):
        amplitude_cutoffs_epochs = []
        for epoch in self._epochs:
            in_epoch = np.logical_and(
                self._spike_times_amps > epoch[1], self._spike_times_amps < epoch[2]
            )
            amplitude_cutoffs_all = metrics.calculate_amplitude_cutoff(
                self._spike_clusters_amps[in_epoch],
                self._amplitudes[in_epoch],
                self._total_units,
                verbose=self.verbose,
            )
            amplitude_cutoffs_list = []
            for i in self._unit_indices:
                amplitude_cutoffs_list.append(amplitude_cutoffs_all[i])
            amplitude_cutoffs = np.asarray(amplitude_cutoffs_list)
            amplitude_cutoffs_epochs.append(amplitude_cutoffs)

        self.metric["amplitude_cutoff"] = []
        for i, epoch in enumerate(self._epochs):
            self.metric["amplitude_cutoff"].append(amplitude_cutoffs_epochs[i])
        return amplitude_cutoffs_epochs

    def threshold_metric(self, threshold, threshold_sign, epoch=None):
        metric_name = "amplitude_cutoff"
        if metric_name not in self.metric.keys():
            self.compute_metric()
        if epoch is None:
            epoch = 0
        assert (epoch < len(self.get_epochs)), "Invalid epoch specified"
        amplitude_cutoff_epochs = self.metric[metric_name][epoch]
        tc = ThresholdCurator(sorting=self._sorting, metrics_epoch=amplitude_cutoff_epochs)
        tc.threshold_sorting(threshold=threshold, threshold_sign=threshold_sign)
        return tc
