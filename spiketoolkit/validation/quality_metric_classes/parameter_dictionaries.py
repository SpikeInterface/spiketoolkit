from collections import OrderedDict
from spiketoolkit.postprocessing.postprocessing_tools import waveforms_params_dict, pca_params_dict, \
    amplitudes_params_dict, get_amplitudes_params, get_waveforms_params, get_pca_params
import numpy as np

recording_params_dict = OrderedDict([('apply_filter', True), ('freq_min', 300.0), ('freq_max', 6000.0)])
# Defining GUI Params
keys = list(recording_params_dict.keys())
types = [type(recording_params_dict[key]) for key in keys]
values = [recording_params_dict[key] for key in keys]
recording_gui_params = [{'name': keys[0], 'type': str(types[0].__name__), 'value': values[0], 'default': values[0],
                         'title': "If True, apply filter"},
                        {'name': keys[1], 'type': str(types[1].__name__), 'value': values[1], 'default': values[1],
                         'title': "High-pass frequency"},
                        {'name': keys[2], 'type': str(types[2].__name__), 'value': values[2], 'default': values[2],
                         'title': "Low-pass frequency"}]

feature_params_dict = OrderedDict(
    [('max_spikes_per_unit', 300), ('recompute_info', False), ('save_features_props', True)])
# Defining GUI Params
keys = list(feature_params_dict.keys())
types = [type(feature_params_dict[key]) for key in keys]
values = [feature_params_dict[key] for key in keys]
feature_gui_params = [{'name': keys[0], 'type': str(types[0].__name__), 'value': values[0], 'default': values[0],
                       'title': "The maximum number of spikes to extract per unit to compute features."},
                      {'name': keys[1], 'type': str(types[1].__name__), 'value': values[1], 'default': values[1],
                       'title': "If True, will always re-extract waveforms."},
                      {'name': keys[2], 'type': str(types[2].__name__), 'value': values[2], 'default': values[2],
                       'title': "If true, it will save the features in the sorting extractor."}]

# Defining GUI Params
keys = list(amplitudes_params_dict.keys())
types = [type(amplitudes_params_dict[key]) for key in keys]
values = [amplitudes_params_dict[key] for key in keys]
amplitude_gui_params = [{'name': keys[0], 'type': str(types[0].__name__), 'value': values[0], 'default': values[0],
                         'title': "If 'absolute' (default), amplitudes are absolute amplitudes in uV are returned. "
                                  "If 'relative', amplitudes are returned as ratios between waveform amplitudes and "
                                  "template amplitudes."},
                        {'name': keys[1], 'type': str(types[1].__name__), 'value': values[1], 'default': values[1],
                         'title': "If maximum channel has to be found among negative peaks ('neg'), positive ('pos') "
                                  "or both ('both' - default)"},
                        {'name': keys[2], 'type': str(types[2].__name__), 'value': values[2], 'default': values[2],
                         'title': "Frames before peak to compute amplitude"},
                        {'name': keys[3], 'type': str(types[3].__name__), 'value': values[3], 'default': values[3],
                         'title': "Frames after peak to compute amplitude"}]

# Defining GUI Params
keys = list(pca_params_dict.keys())
types = [type(pca_params_dict[key]) for key in keys]
values = [pca_params_dict[key] for key in keys]
pca_scores_gui_params = [{'name': keys[0], 'type': str(types[0].__name__), 'value': values[0], 'default': values[0],
                          'title': "n_compFeatures in template-gui format"},
                         {'name': keys[1], 'type': str(types[1].__name__), 'value': values[1], 'default': values[1],
                          'title': "Time period in ms to cut waveforms before the spike events"},
                         {'name': keys[2], 'type': str(types[2].__name__), 'value': values[2], 'default': values[2],
                          'title': "Time period in ms to cut waveforms after the spike events"},
                         {'name': keys[3], 'type': 'dtype', 'value': values[3], 'default': values[3],
                          'title': "The numpy dtype of the waveforms"},
                         {'name': keys[4], 'type': str(types[4].__name__), 'value': values[4], 'default': values[4],
                          'title': "The maximum number of spikes to use to compute PCA."}]

epoch_params_dict = OrderedDict([('epoch_tuples', None), ('epoch_names', None)])


def get_recording_params():
    return recording_params_dict.copy()


def get_epoch_params():
    return epoch_params_dict.copy()


def get_feature_params():
    return feature_params_dict.copy()


def get_recording_gui_params():
    return recording_gui_params.copy()


def get_amplitude_gui_params():
    return amplitude_gui_params.copy()


def get_pca_scores_gui_params():
    return pca_scores_gui_params.copy()


def get_feature_gui_params():
    return feature_gui_params.copy()

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
    all_params.update(get_amplitudes_params())
    all_params.update(get_pca_params())
    all_params.update(get_epoch_params())
    all_params.update(get_feature_params())

    return all_params


def update_param_dicts_with_kwargs(kwargs):
    recording_params = get_recording_params()
    amplitude_params = get_amplitudes_params()
    pca_scores_params = get_pca_params()
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
