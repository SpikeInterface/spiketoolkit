import numpy as np
import spiketoolkit as st


def computeUnitSNR(recording, sorting, unit_ids=None, save_as_property=True):
    '''

    Parameters
    ----------
    recording
    sorting
    unit_ids
    save_as_property

    Returns
    -------

    '''
    if unit_ids is None:
        unit_ids = sorting.getUnitIds()
    channel_noise_levels = _computeChannelNoiseLevels(recording=recording)
    if unit_ids is not None:
        templates = st.postprocessing.getUnitTemplate(recording, sorting, unit_ids=unit_ids)
    else:
        templates = st.postprocessing.getUnitTemplate(recording, sorting)
    snr_list = []
    for i, unit_id in enumerate(sorting.getUnitIds()):
        snr = _computeTemplateSNR(templates[i], channel_noise_levels)
        if save_as_property:
            sorting.setUnitProperty(unit_id, 'snr', snr)
        snr_list.append(snr)
    return snr_list


def _computeTemplateSNR(template, channel_noise_levels):
    channel_snrs = []
    for ch in range(template.shape[0]):
        channel_snrs.append((np.max(template[ch, :]) - np.min(template[ch, :])) / channel_noise_levels[ch])
    return np.max(channel_snrs)

def _computeChannelNoiseLevels(recording):
    M = recording.getNumChannels()
    X = recording.getTraces(start_frame=0, end_frame=np.minimum(1000, recording.getNumFrames()))
    ret = []
    for ch in range(M):
        noise_level = np.std(X[ch, :])
        ret.append(noise_level)
    return ret