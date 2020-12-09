import numpy as np
from copy import copy
from .utils.thresholdcurator import ThresholdCurator
from .quality_metric import QualityMetric
import spiketoolkit as st
from spikemetrics.utils import printProgressBar
from collections import OrderedDict
from sklearn.neighbors import NearestNeighbors
from .parameter_dictionaries import update_all_param_dicts_with_kwargs


class NoiseOverlap(QualityMetric):
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # err
    params = OrderedDict([('max_spikes_per_unit_for_noise_overlap', 1000), ('num_features', 10),
                          ('num_knn', 6)])
    curator_name = "ThresholdNoiseOverlaps"

    def __init__(self, metric_data):
        QualityMetric.__init__(self, metric_data, metric_name="noise_overlap")

        if not metric_data.has_recording():
            raise ValueError("MetricData object must have a recording")

    def compute_metric(self, max_spikes_per_unit_for_noise_overlap, num_features, num_knn, **kwargs):
        params_dict = update_all_param_dicts_with_kwargs(kwargs)
        save_property_or_features = params_dict['save_property_or_features']
        seed = params_dict['seed']

        waveforms = st.postprocessing.get_unit_waveforms(
            self._metric_data._recording,
            self._metric_data._sorting,
            unit_ids=self._metric_data._unit_ids,
            max_spikes_per_unit=max_spikes_per_unit_for_noise_overlap,
            **kwargs
        )

        if seed is not None:
            np.random.seed(seed)

        noise_overlaps = []
        for i_u, unit in enumerate(self._metric_data._unit_ids):
            if self._metric_data.verbose:
                printProgressBar(i_u + 1, len(self._metric_data._unit_ids))
            wfs = waveforms[i_u]
            times = self._metric_data._sorting.get_unit_spike_train(unit_id=unit)

            if len(wfs) > max_spikes_per_unit_for_noise_overlap:
                selecte_idxs = np.random.choice(times, size=max_spikes_per_unit_for_noise_overlap)
                wfs = wfs[selecte_idxs]

            # get clip_size from waveforms shape
            clip_size = wfs.shape[-1]

            num_clips = len(wfs)
            min_time = np.min(times)
            max_time = np.max(times)
            times_control = np.random.choice(np.arange(min_time, max_time), size=num_clips)
            clips = copy(wfs)
            clips_control = np.stack(self._metric_data._recording.get_snippets(snippet_len=clip_size,
                                                                               reference_frames=times_control))
            template = np.median(wfs, axis=0)
            max_ind = np.unravel_index(np.argmax(np.abs(template)), template.shape)
            chmax = max_ind[0]
            tmax = max_ind[1]
            max_val = template[chmax, tmax]
            weighted_clips_control = np.zeros(clips_control.shape)
            weights = np.zeros(num_clips)
            for j in range(num_clips):
                clip0 = clips_control[j, :, :]
                val0 = clip0[chmax, tmax]
                weight0 = val0 * max_val
                weights[j] = weight0
                weighted_clips_control[j, :, :] = clip0 * weight0

            noise_template = np.sum(weighted_clips_control, axis=0)
            noise_template = noise_template / np.sum(np.abs(noise_template)) * np.sum(np.abs(template))

            for j in range(num_clips):
                clips[j, :, :] = _subtract_clip_component(clips[j, :, :], noise_template)
                clips_control[j, :, :] = _subtract_clip_component(clips_control[j, :, :], noise_template)

            all_clips = np.concatenate([clips, clips_control], axis=0)
            num_channels_wfs = all_clips.shape[1]
            num_samples_wfs = all_clips.shape[2]
            all_features = _compute_pca_features(all_clips.reshape((num_clips * 2,
                                                                    num_channels_wfs * num_samples_wfs)), num_features)

            distances, indices = NearestNeighbors(n_neighbors=num_knn + 1, algorithm='auto').fit(
                all_features.T).kneighbors()

            group_id = np.zeros((num_clips * 2))
            group_id[0:num_clips] = 1
            group_id[num_clips:] = 2
            num_match = 0
            total = 0
            for j in range(num_clips * 2):
                for k in range(1, num_knn + 1):
                    ind = indices[j][k]
                    if group_id[j] == group_id[ind]:
                        num_match = num_match + 1
                    total = total + 1
            pct_match = num_match / total
            noise_overlap = 1 - pct_match
            noise_overlaps.append(noise_overlap)
        noise_overlaps = np.asarray(noise_overlaps)
        if save_property_or_features:
            self.save_property_or_features(self._metric_data._sorting, noise_overlaps, self._metric_name)
        return noise_overlaps

    def threshold_metric(self, threshold, threshold_sign, max_spikes_per_unit_for_noise_overlap,
                         num_features, num_knn, **kwargs):
        noise_overlaps = self.compute_metric(max_spikes_per_unit_for_noise_overlap, num_features, num_knn, **kwargs)
        threshold_curator = ThresholdCurator(sorting=self._metric_data._sorting, metric=noise_overlaps)
        threshold_curator.threshold_sorting(threshold=threshold, threshold_sign=threshold_sign)
        return threshold_curator


def _compute_pca_features(X, num_components):
    u, s, vt = np.linalg.svd(X)
    return u[:, :num_components].T


def _subtract_clip_component(clip1, component):
    V1 = clip1.flatten()
    V2 = component.flatten()
    V1 = V1 - np.mean(V1)
    V2 = V2 - np.mean(V2)
    V1 = V1 - V2 * np.dot(V1, V2) / np.dot(V2, V2)
    return V1.reshape(clip1.shape)
