import numpy as np
from .utils.thresholdcurator import ThresholdCurator
from .quality_metric import QualityMetric
import spiketoolkit as st
from spikemetrics.utils import printProgressBar
from collections import OrderedDict
from .parameter_dictionaries import update_all_param_dicts_with_kwargs


class SNR(QualityMetric):
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # err
    params = OrderedDict([('snr_mode', "mad"), ('snr_noise_duration', 10.0), ('max_spikes_per_unit_for_snr', 1000),
                          ('template_mode', "median"), ('max_channel_peak', "both")])
    curator_name = "ThresholdSNR"

    def __init__(self, metric_data):
        QualityMetric.__init__(self, metric_data, metric_name="snr")

        if not metric_data.has_recording():
            raise ValueError("MetricData object must have a recording")

    def compute_metric(self, snr_mode, snr_noise_duration, max_spikes_per_unit_for_snr,
                       template_mode, max_channel_peak, **kwargs):
        params_dict = update_all_param_dicts_with_kwargs(kwargs)
        save_property_or_features = params_dict['save_property_or_features']
        seed = params_dict['seed']
        channel_noise_levels = _compute_channel_noise_levels(
            recording=self._metric_data._recording,
            mode=snr_mode,
            noise_duration=snr_noise_duration,
            seed=seed,
        )
        templates = st.postprocessing.get_unit_templates(
            self._metric_data._recording,
            self._metric_data._sorting,
            unit_ids=self._metric_data._unit_ids,
            max_spikes_per_unit=max_spikes_per_unit_for_snr,
            mode=template_mode, **kwargs
        )
        max_channels = st.postprocessing.get_unit_max_channels(
            self._metric_data._recording,
            self._metric_data._sorting,
            unit_ids=self._metric_data._unit_ids,
            max_spikes_per_unit=max_spikes_per_unit_for_snr,
            peak=max_channel_peak,
            mode=template_mode, **kwargs
        )
        snr_list = []
        for i, unit_id in enumerate(self._metric_data._unit_ids):
            if self._metric_data.verbose:
                printProgressBar(i + 1, len(self._metric_data._unit_ids))
            max_channel_idx = self._metric_data._recording.get_channel_ids().index(max_channels[i])
            snr = _compute_template_SNR(templates[i], channel_noise_levels, max_channel_idx)
            snr_list.append(snr)
        snrs = np.asarray(snr_list)
        if save_property_or_features:
            self.save_property_or_features(self._metric_data._sorting, snrs, self._metric_name)
        return snrs

    def threshold_metric(self, threshold, threshold_sign, snr_mode, snr_noise_duration, max_spikes_per_unit_for_snr,
                         template_mode, max_channel_peak, **kwargs):
        snrs = self.compute_metric(snr_mode, snr_noise_duration, max_spikes_per_unit_for_snr,
                                   template_mode, max_channel_peak, **kwargs)
        threshold_curator = ThresholdCurator(sorting=self._metric_data._sorting, metric=snrs)
        threshold_curator.threshold_sorting(threshold=threshold, threshold_sign=threshold_sign)
        return threshold_curator


def _compute_template_SNR(template, channel_noise_levels, max_channel_idx):
    """
    Computes SNR on the channel with largest amplitude

    Parameters
    ----------
    template: np.array
        Template (n_elec, n_timepoints)
    channel_noise_levels: list
        Noise levels for the different channels
    max_channel_idx: int
        Index of channel with largest templaye

    Returns
    -------
    snr: float
        Signal-to-noise ratio for the template
    """
    snr = np.max(np.abs(template[max_channel_idx]))/ channel_noise_levels[max_channel_idx]
    return snr


def _compute_channel_noise_levels(recording, mode, noise_duration, seed):
    """
    Computes noise level channel-wise

    Parameters
    ----------
    recording: RecordingExtractor
        The recording ectractor object
    mode: str
        'std' or 'mad' (default
    noise_duration: float
        Number of seconds to compute SNR from

    Returns
    -------
    moise_levels: list
        Noise levels for each channel
    """
    M = recording.get_num_channels()
    n_frames = int(noise_duration * recording.get_sampling_frequency())

    if n_frames >= recording.get_num_frames():
        start_frame = 0
        end_frame = recording.get_num_frames()
    else:
        start_frame = np.random.RandomState(seed=seed).randint(0, recording.get_num_frames() - n_frames)
        end_frame = start_frame + n_frames

    X = recording.get_traces(start_frame=start_frame, end_frame=end_frame)
    noise_levels = []
    for ch in range(M):
        if mode == "std":
            noise_level = np.std(X[ch, :])
        elif mode == "mad":
            noise_level = np.median(np.abs(X[ch, :]) / 0.6745)
        else:
            raise Exception("'mode' can be 'std' or 'mad'")
        noise_levels.append(noise_level)
    return noise_levels
