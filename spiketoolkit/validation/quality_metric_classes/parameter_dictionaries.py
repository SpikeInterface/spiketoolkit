from collections import OrderedDict
import numpy as np

recording_params_dict = OrderedDict([('apply_filter', True), ('freq_min', 300.0), ('freq_max', 6000.0)])

feature_params_dict = OrderedDict(
    [('max_spikes_per_unit', 300), ('recompute_info', False), ('save_features_props', True)])

amplitude_params_dict = OrderedDict(
    [('amp_method', "absolute"), ('amp_peak', "both"), ('amp_frames_before', 3), ('amp_frames_after', 3)])

pca_scores_params_dict = OrderedDict(
    [('n_comp', 3), ('ms_before', 1.0), ('ms_after', 2.0), ('dtype', None), ('max_spikes_for_pca', 100000)])

epoch_params_dict = OrderedDict([('epoch_tuples', None), ('epoch_names', None)])


def get_recording_params():
    return recording_params_dict.copy()


def get_amplitude_params():
    return amplitude_params_dict.copy()


def get_pca_scores_params():
    return pca_scores_params_dict.copy()


def get_epoch_params():
    return epoch_params_dict.copy()


def get_feature_params():
    return feature_params_dict.copy()

def get_kwargs_params():
    '''
    Returns all available keyword argument params

    Returns
    -------
    all_params: dict
        Dictionary with all available keyword arguments for validation and curation functions.
    '''
    all_params = {}
    all_params.update(get_recording_params())
    all_params.update(get_amplitude_params())
    all_params.update(get_pca_scores_params())
    all_params.update(get_epoch_params())
    all_params.update(get_feature_params())

    return all_params


def update_param_dicts_with_kwargs(kwargs):
    recording_params = get_recording_params()
    amplitude_params = get_amplitude_params()
    pca_scores_params = get_pca_scores_params()
    epoch_params = get_epoch_params()
    feature_params = get_feature_params()

    if np.any([k in recording_params.keys() for k in kwargs.keys()]):
        for k in kwargs.keys():
            if k in recording_params.keys():
                recording_params[k] = kwargs[k]
    elif np.any([k in amplitude_params.keys() for k in kwargs.keys()]):
        for k in kwargs.keys():
            if k in amplitude_params.keys():
                amplitude_params[k] = kwargs[k]
    elif np.any([k in pca_scores_params.keys() for k in kwargs.keys()]):
        for k in kwargs.keys():
            if k in pca_scores_params.keys():
                pca_scores_params[k] = kwargs[k]
    elif np.any([k in epoch_params.keys() for k in kwargs.keys()]):
        for k in kwargs.keys():
            if k in epoch_params.keys():
                epoch_params[k] = kwargs[k]
    elif np.any([k in feature_params.keys()for k in kwargs.keys()]):
        for k in kwargs.keys():
            if k in feature_params.keys():
                feature_params[k] = kwargs[k]

    return recording_params, amplitude_params, pca_scores_params, epoch_params, feature_params
