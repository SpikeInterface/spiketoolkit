from collections import OrderedDict
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

amplitude_params_dict = OrderedDict(
    [('amp_method', "absolute"), ('amp_peak', "both"), ('amp_frames_before', 3), ('amp_frames_after', 3)])
# Defining GUI Params
keys = list(amplitude_params_dict.keys())
types = [type(amplitude_params_dict[key]) for key in keys]
values = [amplitude_params_dict[key] for key in keys]
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

pca_scores_params_dict = OrderedDict(
    [('n_comp', 3), ('ms_before', 1.0), ('ms_after', 2.0), ('dtype', None), ('max_spikes_for_pca', 100000)])
# Defining GUI Params
keys = list(pca_scores_params_dict.keys())
types = [type(pca_scores_params_dict[key]) for key in keys]
values = [pca_scores_params_dict[key] for key in keys]
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


def get_amplitude_params():
    return amplitude_params_dict.copy()


def get_pca_scores_params():
    return pca_scores_params_dict.copy()


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


def update_param_dicts(recording_params=None, amplitude_params=None,
                       pca_scores_params=None, epoch_params=None,
                       feature_params=None):
    param_dicts = []
    if recording_params is not None:
        if not set(recording_params.keys()).issubset(
                set(get_recording_params().keys())
        ):
            raise ValueError("Improper parameter entered into the recording param dict.")
        else:
            recording_params = OrderedDict(get_recording_params(), **recording_params)
            param_dicts.append(recording_params)

    if amplitude_params is not None:
        if not set(amplitude_params.keys()).issubset(
                set(get_amplitude_params().keys())
        ):
            raise ValueError("Improper parameter entered into the amplitude param dict.")
        else:
            amplitude_params = OrderedDict(get_amplitude_params(), **amplitude_params)
            param_dicts.append(amplitude_params)

    if pca_scores_params is not None:
        if not set(pca_scores_params.keys()).issubset(
                set(get_pca_scores_params().keys())
        ):
            raise ValueError("Improper parameter entered into the amplitude param dict.")
        else:
            pca_scores_params = OrderedDict(get_pca_scores_params(), **pca_scores_params)
            param_dicts.append(pca_scores_params)

    if epoch_params is not None:
        if not set(epoch_params.keys()).issubset(
                set(get_epoch_params().keys())
        ):
            raise ValueError("Improper parameter entered into the epoch params dict")
        else:
            epoch_params = OrderedDict(get_epoch_params(), **epoch_params)
            param_dicts.append(epoch_params)

    if feature_params is not None:
        if not set(feature_params.keys()).issubset(
                set(get_feature_params().keys())
        ):
            raise ValueError("Improper parameter entered into the feature param dict.")
        else:
            feature_params = OrderedDict(get_feature_params(), **feature_params)
            param_dicts.append(feature_params)

    return param_dicts


def update_param_dicts_with_kwargs(kwargs):
    recording_params = get_recording_params()
    amplitude_params = get_amplitude_params()
    pca_scores_params = get_recording_params()
    epoch_params = get_amplitude_params()
    feature_params = get_recording_params()

    if np.any([k in recording_params.keys() for k in kwargs.keys()]):
        for k in kwargs.keys():
            if k in recording_params.keys():
                recording_params[k] = kwargs[k]
    if np.any([k in amplitude_params.keys() for k in kwargs.keys()]):
        for k in kwargs.keys():
            if k in amplitude_params.keys():
                amplitude_params[k] = kwargs[k]
    if np.any([k in pca_scores_params.keys() for k in kwargs.keys()]):
        for k in kwargs.keys():
            if k in pca_scores_params.keys():
                pca_scores_params[k] = kwargs[k]
    if np.any([k in epoch_params.keys() for k in kwargs.keys()]):
        for k in kwargs.keys():
            if k in epoch_params.keys():
                epoch_params[k] = kwargs[k]
    if np.any([k in feature_params.keys()for k in kwargs.keys()]):
        for k in kwargs.keys():
            if k in feature_params.keys():
                feature_params[k] = kwargs[k]

    return recording_params, amplitude_params, pca_scores_params, epoch_params, feature_params
