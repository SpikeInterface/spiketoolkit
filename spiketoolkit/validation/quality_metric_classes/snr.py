import numpy as np
import spikemetrics.metrics as metrics
from .utils.thresholdcurator import ThresholdCurator
from .quality_metric import QualityMetric
import spiketoolkit as st
from spikemetrics.utils import Epoch, printProgressBar
from collections import OrderedDict
from .parameter_dictionaries import get_recording_gui_params, get_feature_gui_params

def make_curator_gui_params(params):
    keys = list(params.keys())
    types = [type(params[key]) for key in keys]
    values = [params[key] for key in keys]
    gui_params = [{'name': keys[0], 'type': str(types[0].__name__), 'value': values[0], 'default': values[0], 'title': "Mode to compute noise SNR ('mad' | 'std' - default 'mad')"},
                  {'name': keys[1], 'type': str(types[1].__name__), 'value': values[1], 'default': values[1], 'title': "Number of seconds to compute noise level from (default 10.0)"},
                  {'name': keys[2], 'type': str(types[2].__name__), 'value': values[2], 'default': values[2], 'title': "Maximum number of spikes to compute templates from (default 1000)"},
                  {'name': keys[3], 'type': str(types[3].__name__), 'value': values[3], 'default': values[3], 'title': "Use 'mean' or 'median' to compute templates"},
                  {'name': keys[4], 'type': str(types[4].__name__), 'value': values[4], 'default': values[4], 'title': "If maximum channel has to be found among negative peaks ('neg'), positive ('pos') or both ('both' - default)"},
                  {'name': keys[5], 'type': 'int', 'value': values[5], 'default': values[5], 'title': "Random seed for reproducibility"},
                  {'name': keys[6], 'type': str(types[6].__name__), 'value': values[6], 'default': values[6], 'title': "If True, will be verbose in metric computation."},]
    curator_gui_params =  [{'name': 'threshold', 'type': 'float', 'title': "The threshold for the given metric."},
                           {'name': 'threshold_sign', 'type': 'str',
                            'title': "If 'less', will threshold any metric less than the given threshold. "
                            "If 'less_or_equal', will threshold any metric less than or equal to the given threshold. "
                            "If 'greater', will threshold any metric greater than the given threshold. "
                            "If 'greater_or_equal', will threshold any metric greater than or equal to the given threshold."}]
    gui_params = curator_gui_params + gui_params + get_recording_gui_params() + get_feature_gui_params()
    return gui_params

class SNR(QualityMetric):
    installed = True  # check at class level if installed or not
    installation_mesg = ""  # err
    params = OrderedDict([('snr_mode',"mad"), ('snr_noise_duration',10.0), ('max_spikes_per_unit_for_snr',1000), 
                          ('template_mode', "median"), ('max_channel_peak', "both"), ('seed',None), ('verbose',False)])
    curator_name = "ThresholdSNR"
    curator_gui_params = make_curator_gui_params(params)
    def __init__(self, metric_data):
        QualityMetric.__init__(self, metric_data, metric_name="snr")

        if not metric_data.has_recording():
            raise ValueError("MetricData object must have a recording")

    def compute_metric(self, snr_mode, snr_noise_duration, max_spikes_per_unit_for_snr, 
                       template_mode, max_channel_peak, save_features_props,
                       recompute_info, seed, save_as_property):

        snrs_epochs = []
        for epoch in self._metric_data._epochs:
            epoch_recording = self._metric_data._recording.get_epoch(epoch[0])
            epoch_sorting = self._metric_data._sorting.get_epoch(epoch[0])
            channel_noise_levels = _compute_channel_noise_levels(
                recording=epoch_recording,
                mode=snr_mode,
                noise_duration=snr_noise_duration,
                seed=seed,
            )

            templates = st.postprocessing.get_unit_templates(
                epoch_recording,
                epoch_sorting,
                unit_ids=self._metric_data._unit_ids,
                max_spikes_per_unit=max_spikes_per_unit_for_snr,
                mode=template_mode,
                save_wf_as_features=save_features_props,
                recompute_waveforms=recompute_info,
                save_as_property=save_features_props,
                seed=seed,
            )

            max_channels = st.postprocessing.get_unit_max_channels(
                epoch_recording,
                epoch_sorting,
                unit_ids=self._metric_data._unit_ids,
                max_spikes_per_unit=max_spikes_per_unit_for_snr,
                peak=max_channel_peak,
                recompute_templates=recompute_info,
                save_as_property=save_features_props,
                mode=template_mode,
                seed=seed,
            )
            snr_list = []
            for i, unit_id in enumerate(self._metric_data._unit_ids):
                if self._metric_data.verbose:
                    printProgressBar(i + 1, len(self._metric_data._unit_ids))
                max_channel_idx = epoch_recording.get_channel_ids().index(
                    max_channels[i]
                )
                snr = _compute_template_SNR(
                    templates[i], channel_noise_levels, max_channel_idx
                )
                snr_list.append(snr)
            snrs = np.asarray(snr_list)
            snrs_epochs.append(snrs)
        if save_as_property:
            self.save_as_property(self._metric_data._sorting, snrs_epochs, self._metric_name)
        return snrs_epochs

    def threshold_metric(self, threshold, threshold_sign, snr_mode, snr_noise_duration, max_spikes_per_unit_for_snr, 
                         template_mode, max_channel_peak, save_features_props, recompute_info, 
                         seed, save_as_property):
        snrs_epochs = self.compute_metric(snr_mode, snr_noise_duration, max_spikes_per_unit_for_snr, 
                                          template_mode, max_channel_peak, save_features_props,
                                          recompute_info, seed, save_as_property)[0]
        threshold_curator = ThresholdCurator(
            sorting=self._metric_data._sorting, metrics_epoch=snrs_epochs
        )
        threshold_curator.threshold_sorting(
            threshold=threshold, threshold_sign=threshold_sign
        )
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
    snr = (
        np.max(np.abs(template[max_channel_idx]))
        / channel_noise_levels[max_channel_idx]
    )
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
        start_frame = np.random.RandomState(seed=seed).randint(
            0, recording.get_num_frames() - n_frames
        )
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