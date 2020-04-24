from collections import OrderedDict
import spikeextractors as se
import numpy as np

waveforms_params_dict = OrderedDict([('grouping_property', None), ('ms_before', 3.), ('ms_after', 3.), ('dtype', None),
                                     ('compute_property_from_recording', False),
                                     ('n_jobs', None), ('max_channels_per_waveforms', None)])

amplitudes_params_dict = OrderedDict([('method', 'absolute'), ('peak', 'both'), ('frames_before', 3),
                                      ('frames_after', 3)])

pca_params_dict = OrderedDict([('n_comp', 3), ('by_electrode', True), ('max_spikes_for_pca', 10000),
                               ('whiten', False)])

common_params_dict = OrderedDict([('max_spikes_per_unit', 300), ('recompute_info', False),
                                  ('save_property_or_features', True), ('memmap', True), ('seed', 0),
                                  ('verbose', False), ('joblib_backend', 'loky')])


def get_waveforms_params():
    return waveforms_params_dict.copy()


def get_amplitudes_params():
    return amplitudes_params_dict.copy()


def get_pca_params():
    return pca_params_dict.copy()


def get_common_params():
    return common_params_dict.copy()


def get_postprocessing_params():
    '''
    Returns all available keyword argument params

    Returns
    -------
    all_params: dict
        Dictionary with all available keyword arguments for postprocessing module
    '''
    all_params = {}
    all_params.update(get_waveforms_params())
    all_params.update(get_amplitudes_params())
    all_params.update(get_pca_params())
    all_params.update(get_common_params())

    return all_params


def update_all_param_dicts_with_kwargs(kwargs):
    all_params = get_postprocessing_params()

    if np.any([k in all_params.keys() for k in kwargs.keys()]):
        for k in kwargs.keys():
            if k in all_params.keys():
                all_params[k] = kwargs[k]

    return all_params


def select_max_channels_from_waveforms(wf, recording, max_channels):
    template = np.mean(wf, axis=0)
    # select based on adjacency
    if max_channels < recording.get_num_channels():
        if 'location' in recording.get_shared_channel_property_names():
            max_channel_idx = np.unravel_index(np.argmax(np.abs(template)),
                                               template.shape)[0]
            locs = recording.get_channel_locations()
            loc_max = locs[max_channel_idx]
            distances = [np.linalg.norm(l - loc_max) for l in locs]
            max_channel_idxs = np.argsort(distances)[:max_channels]
        else:  # select based on amplitude
            peak_idx = np.unravel_index(np.argmax(np.abs(template)),
                                        template.shape)[1]
            max_channel_idxs = np.argsort(np.abs(
                template[:, peak_idx]))[::-1][:max_channels]
    else:
        max_channel_idxs = np.arange(recording.get_num_channels())

    return max_channel_idxs


def select_max_channels_from_templates(template, recording, max_channels):
    # select based on adjacency
    if max_channels < recording.get_num_channels():
        if 'location' in recording.get_shared_channel_property_names():
            max_channel_idx = np.unravel_index(np.argmax(np.abs(template)),
                                               template.shape)[0]
            locs = recording.get_channel_locations()
            loc_max = locs[max_channel_idx]
            distances = [np.linalg.norm(l - loc_max) for l in locs]
            max_channel_idxs = np.argsort(distances)[:max_channels]
        else:  # select based on amplitude
            peak_idx = np.unravel_index(np.argmax(np.abs(template)),
                                        template.shape)[1]
            max_channel_idxs = np.argsort(np.abs(
                template[:, peak_idx]))[::-1][:max_channels]
    else:
        max_channel_idxs = np.arange(recording.get_num_channels())

    return max_channel_idxs


def get_max_channels_per_waveforms(recording, grouping_property, channel_ids, max_channels_per_waveforms):
    if grouping_property is None:
        if max_channels_per_waveforms is None:
            n_channels = len(channel_ids)
        elif max_channels_per_waveforms >= len(channel_ids):
            n_channels = len(channel_ids)
        else:
            n_channels = max_channels_per_waveforms
    else:
        rec = se.SubRecordingExtractor(recording, channel_ids=channel_ids)
        rec_groups = np.array(rec.get_channel_groups())
        groups, count = np.unique(rec_groups, return_counts=True)
        if max_channels_per_waveforms is None:
            n_channels = np.max(count)
        elif max_channels_per_waveforms >= np.max(count):
            n_channels = np.max(count)
        else:
            n_channels = max_channels_per_waveforms
    return n_channels
