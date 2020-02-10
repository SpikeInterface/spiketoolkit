
recording_params_dict = {'apply_filter':True,
                         'freq_min':300,
                         'freq_max':6000,}

amplitude_params_dict = {'amp_method':"absolute",
                         'amp_peak':"both",
                         'amp_frames_before':3,
                         'amp_frames_after':3}

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