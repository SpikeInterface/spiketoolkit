import numpy as np
from spikeextractors import RecordingExtractor, SortingExtractor
import spiketoolkit as st
import pandas as pd
from collections import defaultdict, OrderedDict
from .postprocessing_tools import get_unit_waveforms, get_unit_templates, get_unit_max_channels
import copy


class WaveformMetricCalculator:
    def __init__(self, sorting, recording=None, sampling_frequency=None, apply_filter=True, freq_min=300, freq_max=6000,
                 unit_ids=None, epoch_tuples=None, epoch_names=None, max_spikes_per_unit=300, verbose=False):
        '''
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
        '''
        if sampling_frequency is None and sorting.get_sampling_frequency() is None:
            raise ValueError("Please pass in a sampling frequency (your SortingExtractor does not have one specified)")
        elif sampling_frequency is None:
            self._sampling_frequency = sorting.get_sampling_frequency()
        else:
            self._sampling_frequency = sampling_frequency

        # only use units with spikes to avoid breaking metric calculation
        num_spikes_per_unit = [len(sorting.get_unit_spike_train(unit_id)) for unit_id in sorting.get_unit_ids()]

        if unit_ids is None:
            unit_ids = sorting.get_unit_ids()
        else:
            unit_ids = set(unit_ids)
            unit_ids = list(unit_ids.intersection(sorting.get_unit_ids()))

        if len(unit_ids) == 0:
            raise ValueError("No units found.")

        spike_times, spike_clusters = get_spike_times_metrics_data(sorting, self._sampling_frequency)
        assert isinstance(sorting, SortingExtractor), "'sorting' must be  a SortingExtractor object"
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
        # Dictionary of cached metrics
        self.metrics = {}
        self.verbose = verbose

        for epoch in self._epochs:
            self._sorting.add_epoch(epoch_name=epoch[0], start_frame=epoch[1] * self._sampling_frequency,
                                    end_frame=epoch[2] * self._sampling_frequency)
        if recording is not None:
            assert isinstance(recording, RecordingExtractor), "'sorting' must be  a RecordingExtractor object"
            self.set_recording(recording, apply_filter=apply_filter, freq_min=freq_min, freq_max=freq_max)
        else:
            self._recording = None

    def set_recording(self, recording, apply_filter=True, freq_min=300, freq_max=6000):
        '''
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
        '''
        if apply_filter:
            self._is_filtered = True
            recording = st.preprocessing.bandpass_filter(recording=recording, freq_min=freq_min, freq_max=freq_max,
                                                         cache_to_file=True)
        else:
            self._is_filtered = False
        self._recording = recording
        for epoch in self._epochs:
            self._recording.add_epoch(epoch_name=epoch[0], start_frame=epoch[1] * self._sampling_frequency,
                                      end_frame=epoch[2] * self._sampling_frequency)

    def is_filtered(self):
        return self._is_filtered

    def compute_amplitudes(self, recording=None, amp_method='absolute', amp_peak='both', amp_frames_before=3, amp_frames_after=3,
                           apply_filter=True, freq_min=300, freq_max=6000, save_features_props=False, seed=0):
        '''
        Computes and stores amplitudes for the amplitude cutoff metric

        Parameters
        ----------
        recording: RecordingExtractor
            The given recording extractor from which to extract amplitudes.
        amp_method: str
            If 'absolute' (default), amplitudes are absolute amplitudes in uV are returned.
            If 'relative', amplitudes are returned as ratios between waveform amplitudes and template amplitudes.
        amp_peak: str
            If maximum channel has to be found among negative peaks ('neg'), positive ('pos') or both ('both' - default)
        amp_frames_before: int
            Frames before peak to compute amplitude
        amp_frames_after: int
            Frames after peak to compute amplitude
        apply_filter: bool
            If True, recording is bandpass-filtered.
        freq_min: float
            High-pass frequency for optional filter (default 300 Hz).
        freq_max: float
            Low-pass frequency for optional filter (default 6000 Hz).
        save_features_props: bool
            If true, it will save amplitudes in the sorting extractor.
        seed: int
            Random seed for reproducibility
        '''
        if recording is None:
            if self._recording is None:
                raise ValueError(
                    "No recording given. Either call store_recording or pass a recording into this function")
        else:
            self.set_recording(recording, apply_filter=apply_filter, freq_min=freq_min, freq_max=freq_max)

        spike_times, spike_clusters, amplitudes = get_amplitude_metric_data(self._recording, self._sorting,
                                                                            amp_method=amp_method,
                                                                            save_features_props=save_features_props,
                                                                            amp_peak=amp_peak,
                                                                            amp_frames_before=amp_frames_before,
                                                                            amp_frames_after=amp_frames_after,
                                                                            seed=seed)
        self._amplitudes = amplitudes
        self._spike_clusters_amps = spike_clusters
        self._spike_times_amps = spike_times


    def compute_duration(self):
        pass

    def compute_fwhm(self):
        pass

    def compute_peak_trough_ratio(self):
        pass

    def compute_repolarization_slope(self):
        pass

    def compute_recovery_slope(self):
        pass

    def compute_1d_metrics(self):
        pass

    def compute_waveform_spread(self):
        pass

    def compute_velocity_above_soma(self):
        pass

    def compute_velocity_below_soma(self):
        pass
