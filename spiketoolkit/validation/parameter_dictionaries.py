
recording_params_dict = {'apply_filter':True,
                         'freq_min':300.0,
                         'freq_max':6000.0,}
keys = list(recording_params_dict.keys())
types = [type(recording_params_dict[key]) for key in keys]
values = [recording_params_dict[key] for key in keys]
recording_full_dict = [{'name': keys[0], 'type': str(types[0].__name__), 'value': values[0], 'default': values[0], 'title': "If True, apply filter"},
                       {'name': keys[1], 'type': str(types[1].__name__), 'value': values[1], 'default': values[1], 'title': "High-pass frequency"},
                       {'name': keys[2], 'type': str(types[2].__name__), 'value': values[2], 'default': values[2], 'title': "Low-pass frequency"}]

amplitude_params_dict = {'amp_method':"absolute",
                         'amp_peak':"both",
                         'amp_frames_before':3,
                         'amp_frames_after':3}
keys = list(amplitude_params_dict.keys())
types = [type(amplitude_params_dict[key]) for key in keys]
values = [amplitude_params_dict[key] for key in keys]
amplitude_full_dict = [{'name': keys[0], 'type': str(types[0].__name__), 'value': values[0], 'default': values[0], 'title': "If 'absolute' (default), amplitudes are absolute amplitudes in uV are returned. If 'relative', amplitudes are returned as ratios between waveform amplitudes and template amplitudes."},
                       {'name': keys[1], 'type': str(types[1].__name__), 'value': values[1], 'default': values[1], 'title': "If maximum channel has to be found among negative peaks ('neg'), positive ('pos') or both ('both' - default)"},
                       {'name': keys[2], 'type': str(types[2].__name__), 'value': values[2], 'default': values[2], 'title': "Frames before peak to compute amplitude"},
                       {'name': keys[3], 'type': str(types[3].__name__), 'value': values[3], 'default': values[3], 'title': "Frames after peak to compute amplitude"}]

pca_scores_params_dict = {'n_comp':3,
                          'ms_before':1.0,
                          'ms_after':2.0,
                          'dtype':None,
                          'max_spikes_per_unit':300,
                          'max_spikes_for_pca':1e5}

metric_scope_params_dict = {'unit_ids':None,
                            'epoch_tuples':None,
                            'epoch_names':None}

def get_recording_params():
    return recording_params_dict.copy()

def get_amplitude_params():
    return amplitude_params_dict.copy()

def get_pca_scores_params():
    return pca_scores_params_dict.copy()

def get_metric_scope_params():
    return metric_scope_params_dict.copy()

def update_param_dicts(recording_params=None, amplitude_params=None, 
                       pca_scores_params=None, metric_scope_params=None):

    param_dicts = []
    if recording_params is not None:
        if not set(recording_params.keys()).issubset(
            set(get_recording_params().keys())
        ):
            raise ValueError("Improper parameter entered into the recording param dict.")
        else:
            recording_params = dict(get_recording_params(), **recording_params)
            param_dicts.append(recording_params)

    if amplitude_params is not None:
        if not set(amplitude_params.keys()).issubset(
            set(get_amplitude_params().keys())
        ):
            raise ValueError("Improper parameter entered into the amplitude param dict.")
        else:
            amplitude_params = dict(get_amplitude_params(), **amplitude_params)
            param_dicts.append(amplitude_params)

    if pca_scores_params is not None:
        if not set(pca_scores_params.keys()).issubset(
            set(get_pca_scores_params().keys())
        ):
            raise ValueError("Improper parameter entered into the amplitude param dict.")
        else:
            pca_scores_params = dict(get_pca_scores_params(), **pca_scores_params)
            param_dicts.append(pca_scores_params)

    if metric_scope_params is not None:        
        if not set(metric_scope_params.keys()).issubset(
            set(get_metric_scope_params().keys())
        ):
            raise ValueError("Improper parameter entered into the metric scope param dict.")
        else:
            metric_scope_params = dict(get_metric_scope_params(), **metric_scope_params)
            param_dicts.append(metric_scope_params)

    return param_dicts