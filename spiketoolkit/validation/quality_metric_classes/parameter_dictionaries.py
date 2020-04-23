from collections import OrderedDict
from spiketoolkit.postprocessing.utils import get_amplitudes_params, get_waveforms_params, \
    get_common_params, get_pca_params
import numpy as np

recording_params_dict = OrderedDict([('apply_filter', True), ('freq_min', 300.0), ('freq_max', 6000.0)])

def get_recording_params():
    return recording_params_dict.copy()

def get_validation_params():
    '''
    Returns all available keyword argument params

    Returns
    -------
    all_params: dict
        Dictionary with all available keyword arguments for validation and curation functions.
    '''
    all_params = {}
    all_params.update(get_recording_params())
    all_params.update(get_waveforms_params())
    all_params.update(get_amplitudes_params())
    all_params.update(get_pca_params())
    all_params.update(get_common_params())

    return all_params


def update_all_param_dicts_with_kwargs(kwargs):
    all_params = get_validation_params()

    if np.any([k in all_params.keys() for k in kwargs.keys()]):
        for k in kwargs.keys():
            if k in all_params.keys():
                all_params[k] = kwargs[k]

    return all_params
