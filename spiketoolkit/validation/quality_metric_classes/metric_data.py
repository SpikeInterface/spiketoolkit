import copy
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd
from copy import deepcopy
from spikeextractors import RecordingExtractor, SortingExtractor

import spikemetrics.metrics as metrics
from spiketoolkit.preprocessing.bandpass_filter import bandpass_filter
from spikemetrics.utils import Epoch, printProgressBar

from .utils.validation_tools import get_amplitude_metric_data, get_pca_metric_data, get_spike_times_metrics_data


# Baseclass for each quality metric
class MetricData:
    def __init__(
        self,
        sorting,
        recording,
        sampling_frequency,
        apply_filter,
        freq_min,
        freq_max,
        unit_ids,
        epoch_tuples,
        epoch_names,
        verbose,
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
            A list of tuples with a start and end time for each epoch (in seconds)
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

        # checks to see if any units have no spikes (will break metric calculation)
        for unit_id in sorting.get_unit_ids():
            if len(sorting.get_unit_spike_train(unit_id)) == 0:
                raise ValueError("Spike trains must have none zero length. Please remove all zero length spike trains")

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
            start_frame = epoch[1]
            end_frame = epoch[2]
            if start_frame is not None:
                start_frame = start_frame * self._sampling_frequency
            if end_frame is not None:
                end_frame = end_frame * self._sampling_frequency
            self._sorting.add_epoch(
                epoch_name=epoch[0],
                start_frame=start_frame,
                end_frame=end_frame,
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

    def set_recording(self, recording, apply_filter, freq_min, freq_max):
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
        if apply_filter and not recording.is_filtered:
            recording_filter = bandpass_filter(
                recording=recording,
                freq_min=freq_min,
                freq_max=freq_max,
                cache_to_file=True,
            )
        else:
            recording_filter = recording
        self._recording = recording_filter
        for epoch in self._epochs:
            start_frame = epoch[1]
            end_frame = epoch[2]
            if start_frame is not None:
                start_frame = start_frame * self._sampling_frequency
            if end_frame is not None:
                end_frame = end_frame * self._sampling_frequency
            self._recording.add_epoch(
                epoch_name=epoch[0],
                start_frame=start_frame,
                end_frame=end_frame,
            )

    def is_filtered(self):
        return self._recording.is_filtered

    def has_recording(self):
        return self._recording is not None

    def has_amplitudes(self):
        return self._amplitudes is not None

    def has_pca_scores(self):
        return self._pc_features is not None

    def compute_amplitudes(self, **kwargs):
        """
        Computes and stores amplitudes for the amplitude cutoff metric

        Parameters
        ----------
        method: str
            If 'absolute' (default), amplitudes are absolute amplitudes in uV are returned.
            If 'relative', amplitudes are returned as ratios between waveform amplitudes and template amplitudes.
        peak: str
            If maximum channel has to be found among negative peaks ('neg'), positive ('pos') or both ('both' - default)
        frames_before: int
            Frames before peak to compute amplitude
        frames_after: int
            Frames after peak to compute amplitude
        max_spikes_per_unit: int
            The maximum number of spikes to use to compute amplitudes.
        save_property_or_features: bool
            If true, it will save amplitudes in the sorting extractor.
        recompute_info: bool
            If True, will always re-extract waveforms.
        seed: int
            Random seed for reproducibility
        """
        spike_times, spike_times_amp, spike_clusters, spike_clusters_amp, amplitudes = get_amplitude_metric_data(
            self._recording, self._sorting, **kwargs)
        self._amplitudes = amplitudes
        self._spike_clusters_amps = spike_clusters
        self._spike_times_amps = spike_times_amp
        self._spike_clusters_amps = spike_clusters_amp

    def compute_pca_scores(self, **kwargs):
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
        save_property_or_features: bool
            If true, it will save amplitudes in the sorting extractor.
        seed: int
            Random seed for reproducibility
        """

        spike_times, spike_times_pca, spike_clusters, spike_clusters_pca, pc_features, \
        pc_feature_ind = get_pca_metric_data(self._recording, self._sorting, **kwargs)
        self._pc_features = pc_features
        self._spike_clusters_pca = spike_clusters
        self._spike_times_pca = spike_times_pca
        self._spike_clusters_pca = spike_clusters_pca
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
                epochs = [("complete_session", 0, None)]
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

    def get_in_epoch_bool_mask(self, epoch, spike_times):
        start_frame = epoch[1]
        end_frame = epoch[2]
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.inf
        in_epoch = np.logical_and(
            spike_times >= start_frame, spike_times < end_frame
        )
        return in_epoch


def _get_unit_indices(sorting, unit_ids):
    unit_indices = []
    sorting_unit_ids = np.asarray(sorting.get_unit_ids())
    for unit_id in unit_ids:
        (index,) = np.where(sorting_unit_ids == unit_id)
        if len(index) != 0:
            unit_indices.append(index[0])
    return unit_indices
