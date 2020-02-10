import copy
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
from spikeextractors import RecordingExtractor, SortingExtractor

import spikemetrics.metrics as metrics
import spiketoolkit as st
from spikemetrics.utils import Epoch, printProgressBar
from spiketoolkit.curation.thresholdcurator import ThresholdCurator

from .validation_tools import (
    get_all_metric_data,
    get_amplitude_metric_data,
    get_pca_metric_data,
    get_spike_times_metrics_data,
)

# Baseclass for each quality metric

class MetricData:
    recording_params_dict = {'apply_filter':True,
                             'freq_min':300,
                             'freq_max':6000,}
    amplitude_params_dict = {'amp_method':"absolute",
                             'amp_peak':"both",
                             'amp_frames_before':3,
                             'amp_frames_after':3}
    pca_scores_params_dict = {'n_comp':3,
                              'ms_before':1.0,
                              'ms_after':2.0,
                              'dtype':None,
                              'max_spikes_per_unit':300,
                              'max_spikes_for_pca':1e5}
    metric_scope_params_dict = {'unit_ids':None,
                                'epoch_tuples':None,
                                'epoch_names':None}
    def __init__(
        self,
        sorting,
        recording=None,
        sampling_frequency=None,
        apply_filter=True,
        freq_min=300,
        freq_max=6000,
        unit_ids=None,
        epoch_tuples=None,
        epoch_names=None,
        verbose=False,
    ):
        """
        Computes and stores inital data along with the unit ids and epochs to be used for computing metrics.

        Parameters
        ----------
        sorting: SortingExtractor
            The sorting extractor to be evaluated.
        recording: RecordingExtractor
            The recording extractor to be stored. If None, the recording extractor can be added later.
        sampling_frequency:
            The sampling frequency of the result. If None, will check to see if sampling frequency is in sorting extractor.
        apply_filter: bool
            If True, recording is bandpass-filtered.
        freq_min: float
            High-pass frequency for optional filter (default 300 Hz).
        freq_max: float
            Low-pass frequency for optional filter (default 6000 Hz).
        unit_ids: list
            List of unit ids to compute metric for. If not specified, all units are used
        epoch_tuples: list
            A list of tuples with a start and end time for each epoch
        epoch_names: list
            A list of strings for the names of the given epochs.
        verbose: bool
            If True, progress bar is displayed
        """
        if sampling_frequency is None and sorting.get_sampling_frequency() is None:
            raise ValueError(
                "Please pass in a sampling frequency (your SortingExtractor does not have one specified)"
            )
        elif sampling_frequency is None:
            self._sampling_frequency = sorting.get_sampling_frequency()
        else:
            self._sampling_frequency = sampling_frequency

        # only use units with spikes to avoid breaking metric calculation
        num_spikes_per_unit = [
            len(sorting.get_unit_spike_train(unit_id))
            for unit_id in sorting.get_unit_ids()
        ]
        sorting = ThresholdCurator(sorting=sorting, metrics_epoch=num_spikes_per_unit)
        sorting.threshold_sorting(0, "less_or_equal")

        if unit_ids is None:
            unit_ids = sorting.get_unit_ids()
        else:
            unit_ids = set(unit_ids)
            unit_ids = list(unit_ids.intersection(sorting.get_unit_ids()))

        if len(unit_ids) == 0:
            raise ValueError("No units found.")

        spike_times, spike_clusters = get_spike_times_metrics_data(
            sorting, self._sampling_frequency
        )
        assert isinstance(
            sorting, SortingExtractor
        ), "'sorting' must be  a SortingExtractor object"
        self._sorting = sorting
        self._set_unit_ids(unit_ids)
        self._set_epochs(epoch_tuples, epoch_names)
        self._spike_times = spike_times
        self._spike_clusters = spike_clusters
        self._total_units = len(unit_ids)
        self._unit_indices = _get_unit_indices(self._sorting, unit_ids)
        # To compute this data, need to call all metric data
        self._amplitudes = None
        self._pc_features = None
        self._pc_feature_ind = None
        self._spike_clusters_pca = None
        self._spike_clusters_amps = None
        self._spike_times_pca = None
        self._spike_times_amps = None
        self.verbose = verbose

        for epoch in self._epochs:
            self._sorting.add_epoch(
                epoch_name=epoch[0],
                start_frame=epoch[1] * self._sampling_frequency,
                end_frame=epoch[2] * self._sampling_frequency,
            )
        if recording is not None:
            assert isinstance(
                recording, RecordingExtractor
            ), "'sorting' must be  a RecordingExtractor object"
            self.set_recording(
                recording,
                apply_filter=apply_filter,
                freq_min=freq_min,
                freq_max=freq_max,
            )
        else:
            self._recording = None

    def set_recording(self, recording, apply_filter=True, freq_min=300, freq_max=6000):
        """
        Sets given recording extractor

        Parameters
        ----------
        recording: RecordingExtractor
            The recording extractor to be stored.
        apply_filter: bool
            If True, recording is bandpass-filtered.
        freq_min: float
            High-pass frequency for optional filter (default 300 Hz).
        freq_max: float
            Low-pass frequency for optional filter (default 6000 Hz).
        """
        if apply_filter:
            self._is_filtered = True
            recording = st.preprocessing.bandpass_filter(
                recording=recording,
                freq_min=freq_min,
                freq_max=freq_max,
                cache_to_file=True,
            )
        else:
            self._is_filtered = False
        self._recording = recording
        for epoch in self._epochs:
            self._recording.add_epoch(
                epoch_name=epoch[0],
                start_frame=epoch[1] * self._sampling_frequency,
                end_frame=epoch[2] * self._sampling_frequency,
            )

    def is_filtered(self):
        return self._is_filtered

    def has_recording(self):
        return self._recording is not None

    def has_amplitudes(self):
        return self._amplitudes is not None

    def has_pca_scores(self):
        return self._pc_features is not None

    def compute_amplitudes(
        self,
        amp_method="absolute",
        amp_peak="both",
        amp_frames_before=3,
        amp_frames_after=3,
        save_features_props=False,
        seed=None,
    ):
        """
        Computes and stores amplitudes for the amplitude cutoff metric

        Parameters
        ----------
        amp_method: str
            If 'absolute' (default), amplitudes are absolute amplitudes in uV are returned.
            If 'relative', amplitudes are returned as ratios between waveform amplitudes and template amplitudes.
        amp_peak: str
            If maximum channel has to be found among negative peaks ('neg'), positive ('pos') or both ('both' - default)
        amp_frames_before: int
            Frames before peak to compute amplitude
        amp_frames_after: int
            Frames after peak to compute amplitude
        save_features_props: bool
            If true, it will save amplitudes in the sorting extractor.
        seed: int
            Random seed for reproducibility
        """
        spike_times, spike_clusters, amplitudes = get_amplitude_metric_data(
            self._recording,
            self._sorting,
            amp_method=amp_method,
            save_features_props=save_features_props,
            amp_peak=amp_peak,
            amp_frames_before=amp_frames_before,
            amp_frames_after=amp_frames_after,
            seed=seed,
        )
        self._amplitudes = amplitudes
        self._spike_clusters_amps = spike_clusters
        self._spike_times_amps = spike_times

    def compute_pca_scores(
        self,
        n_comp=3,
        ms_before=1.0,
        ms_after=2.0,
        dtype=None,
        max_spikes_per_unit=300,
        recompute_info=True,
        max_spikes_for_pca=1e5,
        save_features_props=False,
        seed=None,
    ):
        """
        Computes and stores pca for the metrics computation

        Parameters
        ----------
        n_comp: int
            n_compFeatures in template-gui format
        ms_before: float
            Time period in ms to cut waveforms before the spike events
        ms_after: float
            Time period in ms to cut waveforms after the spike events
        dtype: dtype
            The numpy dtype of the waveforms
        max_spikes_per_unit: int
            The maximum number of spikes to extract per unit.
        recompute_info: bool
            If True, will always re-extract waveforms.
        max_spikes_for_pca: int
            The maximum number of spikes to use to compute PCA.
        save_features_props: bool
            If true, it will save amplitudes in the sorting extractor.
        seed: int
            Random seed for reproducibility
        """

        spike_times, spike_clusters, pc_features, pc_feature_ind = get_pca_metric_data(
            self._recording,
            self._sorting,
            n_comp=n_comp,
            ms_before=ms_before,
            ms_after=ms_after,
            dtype=dtype,
            max_spikes_per_unit=max_spikes_per_unit,
            max_spikes_for_pca=max_spikes_for_pca,
            recompute_info=recompute_info,
            save_features_props=save_features_props,
            seed=seed,
        )
        self._pc_features = pc_features
        self._spike_clusters_pca = spike_clusters
        self._spike_times_pca = spike_times
        self._pc_feature_ind = pc_feature_ind

    def compute_all_metric_data(
        self,
        n_comp=3,
        ms_before=1.0,
        ms_after=2.0,
        dtype=None,
        max_spikes_per_unit=300,
        amp_method="absolute",
        amp_peak="both",
        amp_frames_before=3,
        amp_frames_after=3,
        recompute_info=True,
        max_spikes_for_pca=1e5,
        save_features_props=False,
        seed=0,
    ):
        """
        Computes and stores data for all metrics (all metrics can be run after calling this function).

        Parameters
        ----------
        n_comp: int
            n_compFeatures in template-gui format
        ms_before: float
            Time period in ms to cut waveforms before the spike events
        ms_after: float
            Time period in ms to cut waveforms after the spike events
        dtype: dtype
            The numpy dtype of the waveforms
        max_spikes_per_unit: int
            The maximum number of spikes to extract per unit.
        amp_method: str
            If 'absolute' (default), amplitudes are absolute amplitudes in uV are returned.
            If 'relative', amplitudes are returned as ratios between waveform amplitudes and template amplitudes.
        amp_peak: str
            If maximum channel has to be found among negative peaks ('neg'), positive ('pos') or both ('both' - default)
        amp_frames_before: int
            Frames before peak to compute amplitude
        amp_frames_after: int
            Frames after peak to compute amplitude
        recompute_info: bool
            If True, will always re-extract waveforms.
        max_spikes_for_pca: int
            The maximum number of spikes to use to compute PCA.
        save_features_props: bool
            If True, save all features and properties in the sorting extractor.
        seed: int
            Random seed for reproducibility
        """

        (
            spike_times,
            spike_times_amps,
            spike_times_pca,
            spike_clusters,
            spike_clusters_amps,
            spike_clusters_pca,
            amplitudes,
            pc_features,
            pc_feature_ind,
        ) = get_all_metric_data(
            self._recording,
            self._sorting,
            n_comp=n_comp,
            ms_before=ms_before,
            ms_after=ms_after,
            dtype=dtype,
            amp_method=amp_method,
            amp_peak=amp_peak,
            amp_frames_before=amp_frames_before,
            amp_frames_after=amp_frames_after,
            max_spikes_per_unit=max_spikes_per_unit,
            max_spikes_for_pca=max_spikes_for_pca,
            recompute_info=recompute_info,
            save_features_props=save_features_props,
            seed=seed,
        )

        self._amplitudes = amplitudes
        self._spike_clusters_amps = spike_clusters_amps
        self._spike_times_amps = spike_times_amps
        self._pc_features = pc_features
        self._spike_clusters_pca = spike_clusters_pca
        self._spike_times_pca = spike_times_pca
        self._pc_feature_ind = pc_feature_ind

    def set_amplitudes(self, amplitudes):
        self._amplitudes = amplitudes

    def set_pc_features(self, pc_features):
        self._pc_features = pc_features

    def set_pc_feature_ind(self, pc_feature_ind):
        self._pc_feature_ind = pc_feature_ind

    def _set_epochs(self, epoch_tuples, epoch_names):
        if epoch_tuples is None:
            if epoch_names is None:
                epochs = [("complete_session", 0, np.inf)]
            else:
                raise ValueError("No epoch tuples specified, but names given.")
        else:
            if epoch_names is None:
                epochs = []
                for i, epoch_tuple in enumerate(epoch_tuples):
                    epoch = (str(i), epoch_tuple[0], epoch_tuple[1])
                    epochs.append(epoch)
            else:
                assert len(epoch_names) == len(
                    epoch_tuples
                ), "Make sure the name and epoch lists are equal in length,"
                epochs = []
                for i, epoch_tuple in enumerate(epoch_tuples):
                    epoch = (str(epoch_names[i]), epoch_tuple[0], epoch_tuple[1])
                    epochs.append(epoch)
        self._epochs = epochs

    def _set_unit_ids(self, unit_ids):
        self._unit_ids = unit_ids

    def get_epochs(self):
        return self._epochs

    def get_unit_ids(self):
        return self._unit_ids

    #implemented by quality metric subclasses
    def compute_metric(self):
        pass

    def threshold_metric(self, threshold, threshold_sign, epoch=None):
        '''
        Parameters
        ----------
        threshold:
            The threshold for the given metric.
        threshold_sign: str
            If 'less', will threshold any metric less than the given threshold.
            If 'less_or_equal', will threshold any metric less than or equal to the given threshold.
            If 'greater', will threshold any metric greater than the given threshold.
            If 'greater_or_equal', will threshold any metric greater than or equal to the given threshold.
        epoch:
            The threshold will be applied to the specified epoch. 
            If epoch is None, then it will default to the first epoch. 
        Returns
        -------
        tc: ThresholdCurator
            The thresholded sorting extractor.
        '''
        pass


def _get_unit_indices(sorting, unit_ids):
    unit_indices = []
    sorting_unit_ids = np.asarray(sorting.get_unit_ids())
    for unit_id in unit_ids:
        (index,) = np.where(sorting_unit_ids == unit_id)
        if len(index) != 0:
            unit_indices.append(index[0])
    return unit_indices
