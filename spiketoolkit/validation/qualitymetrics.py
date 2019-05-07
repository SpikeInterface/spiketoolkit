import numpy as np
import spiketoolkit as st


def compute_unit_SNR(recording, sorting, unit_ids=None, save_as_property=True):
    '''
    Computes signal-to-noise ratio (SNR) of the average waveforms on the largest channel.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor
    sorting: SortingExtractor
        The sorting extraxtor
    unit_ids: list
        List of unit ids to compute SNR for. If not specified, all units are used
    save_as_property: bool
        If True (defult), the computed SNR is added as a unit property to the sorting extractor as 'snr'

    Returns
    -------
    snr_list: list
        List of computed SNRs

    '''
    if unit_ids is None:
        unit_ids = sorting.get_unit_ids()
    channel_noise_levels = _compute_channel_noise_levels(recording=recording)
    templates = st.postprocessing.postprocessing_tools.get_unit_template(recording, sorting, unit_ids=unit_ids)
    snr_list = []
    for i, unit_id in enumerate(sorting.get_unit_ids()):
        snr = _compute_template_SNR(templates[i], channel_noise_levels)
        if save_as_property:
            sorting.set_unit_property(unit_id, 'snr', snr)
        snr_list.append(snr)
    return snr_list


def _compute_template_SNR(template, channel_noise_levels):
    channel_snrs = []
    for ch in range(template.shape[0]):
        channel_snrs.append((np.max(template[ch, :]) - np.min(template[ch, :])) / channel_noise_levels[ch])
    return np.max(channel_snrs)

def _compute_channel_noise_levels(recording):
    M = recording.get_num_channels()
    X = recording.get_traces(start_frame=0, end_frame=np.minimum(1000, recording.get_num_frames()))
    ret = []
    for ch in range(M):
        noise_level = np.std(X[ch, :])
        ret.append(noise_level)
    return ret
