import numpy as np
import spiketoolkit as st
import spikeextractors as se
from sklearn.decomposition import PCA
from pathlib import Path
import warnings
import shutil
from joblib import Parallel, delayed
from spikeextractors import RecordingExtractor, SortingExtractor
import csv
from tqdm import tqdm
from copy import copy
import time

from .utils import update_all_param_dicts_with_kwargs, select_max_channels_from_waveforms, \
    divide_recording_into_time_chunks, get_unit_waveforms_for_chunk, get_max_channels_per_waveforms, \
    select_max_channels_from_templates


def get_unit_waveforms(recording, sorting, unit_ids=None, channel_ids=None, return_idxs=False, chunk_size=None,
                       chunk_mb=500, **kwargs):
    """
    Computes the spike waveforms from a recording and sorting extractor.
    The recording is split in chunks (the size in Mb is set with the chunk_mb argument) and all waveforms are extracted
    for each chunk and then re-assembled. If multiple jobs are used (n_jobs > 1), more and smaller chunks are created
    and processed in parallel.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor
    sorting: SortingExtractor
        The sorting extractor
    unit_ids: list
        List of unit ids to extract waveforms
    channel_ids: list
        List of channels ids to compute waveforms from
    return_idxs: bool
        If True, spike indexes and channel indexes are returned
    chunk_size: int
        Size of chunks in number of samples. If None, it is automatically calculated
    chunk_mb: int
        Size of chunks in Mb (default 500 Mb)
    **kwargs: Keyword arguments
        A dictionary with default values can be retrieved with:
        st.postprocessing.get_waveforms_params():
            grouping_property: str
                Property to group channels. E.g. if the recording extractor has the 'group' property and
                'grouping_property' is 'group', then waveforms are computed group-wise.
            ms_before: float
                Time period in ms to cut waveforms before the spike events
            ms_after: float
                Time period in ms to cut waveforms after the spike events
            dtype: dtype
                The numpy dtype of the waveforms
            compute_property_from_recording: bool
                If True and 'grouping_property' is given, the property of each unit is assigned as the corresponding
                property of the recording extractor channel on which the average waveform is the largest
            max_channels_per_waveforms: int or None
                Maximum channels per waveforms to return. If None, all channels are returned.
            n_jobs: int
                Number of parallel jobs (default 1)
            max_spikes_per_unit: int
                The maximum number of spikes to extract per unit.
            memmap: bool
                If True, waveforms are saved as memmap object (recommended for long recordings with many channels)
            seed: int
                Random seed for extracting random waveforms
            save_property_or_features: bool
                If True (default), waveforms are saved as features of the sorting extractor object
            recompute_info: bool
                If True, waveforms are recomputed (default False)
            verbose: bool
                If True output is verbose

    Returns
    -------
    waveforms: list
        List of np.array (n_spikes, n_channels, n_timepoints) containing extracted waveforms for each unit
    spike_indexes: list
        List of spike indexes for which waveforms are computed. Returned if 'return_idxs' is True
    channel_indexes: list
        List of max channel indexes
    """
    if isinstance(unit_ids, (int, np.integer)):
        unit_ids = [unit_ids]
    elif unit_ids is None:
        unit_ids = sorting.get_unit_ids()
    elif not isinstance(unit_ids, (list, np.ndarray)):
        raise Exception("unit_ids is is invalid")
    if isinstance(channel_ids, (int, np.integer)):
        channel_ids = [channel_ids]

    if channel_ids is None:
        channel_ids = recording.get_channel_ids()

    assert np.all([u in sorting.get_unit_ids() for u in unit_ids]), "Invalid unit_ids"
    assert np.all([ch in recording.get_channel_ids() for ch in channel_ids]), "Invalid channel_ids"

    params_dict = update_all_param_dicts_with_kwargs(kwargs)
    grouping_property = params_dict['grouping_property']
    ms_before = params_dict['ms_before']
    ms_after = params_dict['ms_after']
    dtype = params_dict['dtype']
    max_spikes_per_unit = params_dict['max_spikes_per_unit']
    compute_property_from_recording = params_dict['compute_property_from_recording']
    memmap = params_dict['memmap']
    n_jobs = params_dict['n_jobs']
    joblib_backend = params_dict['joblib_backend']
    max_channels_per_waveforms = params_dict['max_channels_per_waveforms']
    seed = params_dict['seed']
    verbose = params_dict['verbose']
    save_property_or_features = params_dict['save_property_or_features']
    recompute_info = params_dict['recompute_info']

    waveform_list = []
    spike_index_list = []
    channel_index_list = []

    if max_channels_per_waveforms is None:
        max_channels_per_waveforms = len(channel_ids)

    if 'waveforms' in sorting.get_shared_unit_spike_feature_names() and not recompute_info:
        for unit_id in unit_ids:
            waveforms = sorting.get_unit_spike_features(unit_id, 'waveforms')
            waveform_list.append(waveforms)
            if len(waveforms) < len(sorting.get_unit_spike_train(unit_id)):
                indexes = sorting.get_unit_spike_features(unit_id, 'waveforms_idxs')
            else:
                indexes = np.arange(len(waveforms))
            if 'waveforms_channel_idxs' in sorting.get_shared_unit_property_names():
                channel_idxs = sorting.get_unit_property(unit_id, 'waveforms_channel_idxs')
            else:
                channel_idxs = np.arange(recording.get_num_channels())
            spike_index_list.append(indexes)
            channel_index_list.append(channel_idxs)
    else:
        if dtype is None:
            dtype = recording.get_dtype()

        if n_jobs is None:
            n_jobs = 1
        if n_jobs == 0:
            n_jobs = 1

        if seed is not None:
            np.random.seed(seed)

        # num_channels = recording.get_num_channels()
        num_frames = recording.get_num_frames()
        fs = recording.get_sampling_frequency()
        n_pad = [int(ms_before * fs / 1000), int(ms_after * fs / 1000)]

        # set chunk size
        if chunk_size is not None:
            chunk_size = int(chunk_size)
        elif chunk_mb is not None:
            n_bytes = np.dtype(recording.get_dtype()).itemsize
            max_size = int(chunk_mb * 1e6)  # set Mb per chunk
            chunk_size = max_size // (recording.get_num_channels() * n_bytes)

        if n_jobs > 1:
            chunk_size /= n_jobs

        # chunk_size = num_bytes_per_chunk / num_bytes_per_frame
        padding_size = 100 + n_pad[0] + n_pad[1]  # a bit excess padding
        chunks = divide_recording_into_time_chunks(
            num_frames=num_frames,
            chunk_size=chunk_size,
            padding_size=padding_size
        )
        n_chunk = len(chunks)

        if verbose:
            print(f"Number of chunks: {len(chunks)} - Number of jobs: {n_jobs}")

        # pre-map memmap files
        n_channels = len(channel_ids)
        if len(channel_ids) < recording.get_num_channels():
            recording = se.SubRecordingExtractor(recording, channel_ids=channel_ids)

        if not recording.check_if_dumpable():
            if n_jobs > 1:
                n_jobs = 1
                print("RecordingExtractor is not dumpable and can't be processed in parallel")
            rec_arg = recording
        else:
            if n_jobs > 1:
                rec_arg = recording.dump_to_dict()
            else:
                rec_arg = recording

        if memmap:
            all_unit_waveforms = []
            for unit_id in unit_ids:
                fname = f'waveforms_{unit_id}.raw'
                len_wf = len(sorting.get_unit_spike_train(unit_id))
                if max_spikes_per_unit is not None:
                    if len_wf > max_spikes_per_unit:
                        len_wf = max_spikes_per_unit
                shape = (len_wf, n_channels, sum(n_pad))
                arr = sorting.allocate_array(shape=shape, dtype=dtype, name=fname, memmap=memmap)
                all_unit_waveforms.append(arr)
        else:
            all_unit_waveforms = [[] for ii in range(len(unit_ids))]

        if verbose and n_jobs == 1:
            chunk_iter = tqdm(range(n_chunk), ascii=True, desc="Extracting waveforms in chunks")
        else:
            chunk_iter = range(n_chunk)

        # Pre-select spikes to include
        spike_times_to_include = []
        if max_spikes_per_unit is not None:
            for i, unit in enumerate(unit_ids):
                spiketrain = sorting.get_unit_spike_train(unit)
                num_spikes = len(spiketrain)
                if num_spikes > max_spikes_per_unit:
                    spike_idxs = np.sort(np.random.permutation(num_spikes)[:max_spikes_per_unit])
                    spike_index_list.append(spike_idxs)
                    spike_times_to_include.append(spiketrain[spike_idxs])
                else:
                    spike_index_list.append(np.arange(num_spikes))
                    spike_times_to_include.append(None)
        else:
            for u in unit_ids:
                spike_index_list.append(None)
                spike_times_to_include.append(None)

        # pre-compute spikes for each chunk
        times_in_all_chunks = []
        start_spike_idxs = []
        n_spikes = np.zeros(len(unit_ids), dtype='int64')
        for chunk in chunks:
            times_in_chunk_units = []
            start_spike_idxs.append(copy(n_spikes))
            for i, unit in enumerate(unit_ids):
                times = sorting.get_unit_spike_train(unit_id=unit)
                times_in_chunk = []
                if spike_times_to_include[i] is not None:
                    spike_times = spike_times_to_include[i]
                    spike_time_idxs = np.where((spike_times >= chunk['istart'])
                                               & (spike_times < chunk['iend']))[0]  # exclude padding

                    if len(spike_time_idxs) > 0:
                        times_in_chunk = spike_times[spike_time_idxs]
                else:
                    spike_time_idxs = np.where((times >= chunk['istart'])
                                               & (times < chunk['iend']))[0]
                    times_in_chunk = times[spike_time_idxs]
                n_spikes[i] += len(times_in_chunk)
                times_in_chunk_units.append(times_in_chunk)
            times_in_all_chunks.append(times_in_chunk_units)

        if n_jobs == 1:
            for ii in chunk_iter:
                unit_waveforms = _extract_waveforms_one_chunk(ii, recording, chunks, unit_ids, n_pad,
                                                              times_in_all_chunks, start_spike_idxs,
                                                              all_unit_waveforms, memmap, dtype, False)

                if not memmap:
                    for i_unit, unit in enumerate(unit_ids):
                        wf = unit_waveforms[i_unit]
                        wf = wf.astype(dtype)
                        all_unit_waveforms[i_unit].append(wf)

        else:
            # waveforms are saved directly to the memmap file if
            unit_waveforms = Parallel(n_jobs=n_jobs, backend=joblib_backend)(
                delayed(_extract_waveforms_one_chunk)(ii, rec_arg, chunks, unit_ids, n_pad,
                                                      times_in_all_chunks, start_spike_idxs,
                                                      all_unit_waveforms, memmap, dtype, verbose, )
                for ii in chunk_iter)

            if not memmap:
                for ii, unit_waveform in enumerate(unit_waveforms):
                    for i_unit, unit in enumerate(unit_ids):
                        wf = unit_waveform[i_unit]
                        wf = wf.astype(dtype)
                        all_unit_waveforms[i_unit].append(wf)

        if memmap:
            waveform_list = all_unit_waveforms
        else:
            # concatenate the results over the chunks
            if len(chunks) > 1:
                waveform_list = []
                for i_unit in range(len(unit_ids)):
                    waveform_concat = np.concatenate([all_unit_waveforms[i_unit][ch] for ch in range(len(chunks))],
                                                     axis=0)
                    waveform_list.append(waveform_concat)
            else:
                waveform_list = [wf[0] for wf in all_unit_waveforms]

        # return correct max channels
        if grouping_property is not None:
            if grouping_property not in recording.get_shared_channel_property_names():
                raise ValueError("'grouping_property' should be a property of recording extractors")
            if compute_property_from_recording:
                compute_sorting_group = True
            elif grouping_property not in sorting.get_shared_unit_property_names():
                warnings.warn('Grouping property not in sorting extractor. Computing it from the recording extractor')
                compute_sorting_group = True
            else:
                compute_sorting_group = False

            waveforms_reduced_channels = []
            channel_groups = np.array([recording.get_channel_property(ch, grouping_property)
                                       for ch in recording.get_channel_ids()])
            unit_groups = []
            if compute_sorting_group:
                # extract unit groups
                for wf in waveform_list:
                    mean_waveforms = np.squeeze(np.mean(wf, axis=0))
                    max_amp_elec = np.unravel_index(mean_waveforms.argmin(), mean_waveforms.shape)[0]
                    unit_group = recording.get_channel_property(recording.get_channel_ids()[max_amp_elec],
                                                                grouping_property)
                    unit_groups.append(unit_group)
            else:
                for u in unit_ids:
                    unit_group = sorting.get_unit_property(u, grouping_property)
                    unit_groups.append(unit_group)

            for (wf, unit_group) in zip(waveform_list, unit_groups):
                channel_unit_group = np.where(channel_groups == unit_group)[0]

                if len(channel_unit_group) < max_channels_per_waveforms:
                    max_channel_idxs = channel_unit_group
                else:
                    subrec = se.SubRecordingExtractor(recording, channel_ids=list(channel_unit_group))
                    max_channel_idxs = select_max_channels_from_waveforms(wf, subrec, max_channels_per_waveforms)

                channel_index_list.append(max_channel_idxs)
                waveform = wf[:, max_channel_idxs]
                # some channels are missing - re-instantiate object
                if memmap:
                    memmap_file = wf.filename
                    if not wf._mmap.closed:
                        wf._mmap.close()
                    del wf
                    Path(memmap_file).unlink()
                    memmap_array = np.memmap(memmap_file, mode='w+', shape=waveform.shape,
                                             dtype=waveform.dtype)
                    memmap_array[:] = waveform
                    del (waveform)
                    waveforms_reduced_channels.append(memmap_array)
                else:
                    waveforms_reduced_channels.append(waveform)
            waveform_list = waveforms_reduced_channels
        else:
            if max_channels_per_waveforms < len(recording.get_channel_ids()):
                waveforms_reduced_channels = []
                for wf in waveform_list:
                    max_channel_idxs = select_max_channels_from_waveforms(wf, recording, max_channels_per_waveforms)
                    channel_index_list.append(max_channel_idxs)
                    waveform = wf[:, max_channel_idxs]
                    # some channels are missing - re-instantiate object
                    if memmap:
                        memmap_file = wf.filename
                        if not wf._mmap.closed:
                            wf._mmap.close()
                        del wf
                        Path(memmap_file).unlink()
                        memmap_array = np.memmap(memmap_file, mode='w+', shape=waveform.shape,
                                                 dtype=waveform.dtype)
                        memmap_array[:] = waveform
                        waveforms_reduced_channels.append(memmap_array)
                    else:
                        waveforms_reduced_channels.append(waveform)
                waveform_list = waveforms_reduced_channels
            else:
                for wf in waveform_list:
                    channel_index_list.append(channel_ids)

        if save_property_or_features:
            for i, unit_id in enumerate(unit_ids):
                sorting.set_unit_spike_features(unit_id, 'waveforms', waveform_list[i], indexes=spike_index_list[i])
                if len(channel_index_list[i]) < recording.get_num_channels():
                    sorting.set_unit_property(unit_id, 'waveforms_channel_idxs', channel_index_list[i])

    if return_idxs:
        return waveform_list, spike_index_list, channel_index_list
    else:
        return waveform_list


def get_unit_templates(recording, sorting, unit_ids=None, channel_ids=None,
                       mode='median', _waveforms=None, **kwargs):
    '''
    Computes the spike templates from a recording and sorting extractor. If waveforms are not found as features,
    they are computed.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor
    sorting: SortingExtractor
        The sorting extractor
    unit_ids: list
        List of unit ids to extract templates
    channel_ids: list
        List of channels ids to compute templates from
    mode: str
        Use 'mean' or 'median' to compute templates
    _waveforms: list
        Pre-computed waveforms to be used for computing templates
    **kwargs: Keyword arguments
        A dictionary with default values can be retrieved with:
        st.postprocessing.get_waveforms_params():
            grouping_property: str
                Property to group channels. E.g. if the recording extractor has the 'group' property and
                'grouping_property' is 'group', then waveforms are computed group-wise.
            ms_before: float
                Time period in ms to cut waveforms before the spike events
            ms_after: float
                Time period in ms to cut waveforms after the spike events
            dtype: dtype
                The numpy dtype of the waveforms
            compute_property_from_recording: bool
                If True and 'grouping_property' is given, the property of each unit is assigned as the corresponding
                property of the recording extractor channel on which the average waveform is the largest
            max_channels_per_waveforms: int or None
                Maximum channels per waveforms to return. If None, all channels are returned
            n_jobs: int
                Number of parallel jobs (default 1)
            max_spikes_per_unit: int
                The maximum number of spikes to extract per unit
            memmap: bool
                If True, waveforms are saved as memmap object (recommended for long recordings with many channels)
            seed: int
                Random seed for extracting random waveforms
            save_property_or_features: bool
                If True (default), waveforms are saved as features of the sorting extractor object
            recompute_info: bool
                If True, waveforms are recomputed (default False)
            verbose: bool
                If True output is verbose

    Returns
    -------
    templates: list
        List of np.array (n_channels, n_timepoints) containing extracted templates for each unit
    '''
    if isinstance(unit_ids, (int, np.integer)):
        unit_ids = [unit_ids]
    elif unit_ids is None:
        unit_ids = sorting.get_unit_ids()
    elif not isinstance(unit_ids, (list, np.ndarray)):
        raise Exception("unit_ids is not a valid in valid")

    assert np.all([u in sorting.get_unit_ids() for u in unit_ids]), "Invalid unit_ids"

    params_dict = update_all_param_dicts_with_kwargs(kwargs)
    recompute_info = params_dict['recompute_info']
    save_property_or_features = params_dict['save_property_or_features']

    template_list = []
    if 'template' in sorting.get_shared_unit_property_names() and not recompute_info:
        for unit_id in unit_ids:
            template = sorting.get_unit_property(unit_id, 'template')
            template_list.append(template)
    else:
        if _waveforms is None:
            waveforms = get_unit_waveforms(recording, sorting, unit_ids, channel_ids, return_idxs=False, **kwargs)
        else:
            waveforms = _waveforms

        for i, unit_id in enumerate(unit_ids):
            wf = waveforms[i]
            if mode == 'mean':
                template = np.mean(wf, axis=0)
            elif mode == 'median':
                template = np.median(wf, axis=0)
            else:
                raise Exception("'mode' can be 'mean' or 'median'")
            if save_property_or_features:
                sorting.set_unit_property(unit_id, 'template', template)

            template_list.append(template)
    return template_list


def get_unit_max_channels(recording, sorting, unit_ids=None, channel_ids=None,
                          max_channels=1, peak='both', mode='median', **kwargs):
    '''
    Computes the spike maximum channels from a recording and sorting extractor. If templates are not found as property,
    they are computed. If templates are computed by group, the max channels refer to the overall channel ids.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor
    sorting: SortingExtractor
        The sorting extractor
    unit_ids: list
        List of unit ids to extract maximum channels
    channel_ids: list
        List of channels ids to compute max_channels from
    max_channels: int
        Number of max channels per units to return (default=1)
    mode: str
        Use 'mean' or 'median' to compute templates
    **kwargs: Keyword arguments
        A dictionary with default values can be retrieved with:
        st.postprocessing.get_waveforms_params():
            grouping_property: str
                Property to group channels. E.g. if the recording extractor has the 'group' property and
                'grouping_property' is 'group', then waveforms are computed group-wise.
            ms_before: float
                Time period in ms to cut waveforms before the spike events
            ms_after: float
                Time period in ms to cut waveforms after the spike events
            dtype: dtype
                The numpy dtype of the waveforms
            compute_property_from_recording: bool
                If True and 'grouping_property' is given, the property of each unit is assigned as the corresponding
                property of the recording extractor channel on which the average waveform is the largest
            max_channels_per_waveforms: int or None
                Maximum channels per waveforms to return. If None, all channels are returned
            n_jobs: int
                Number of parallel jobs (default 1)
            max_spikes_per_unit: int
                The maximum number of spikes to extract per unit
            memmap: bool
                If True, waveforms are saved as memmap object (recommended for long recordings with many channels)
            seed: int
                Random seed for extracting random waveforms
            save_property_or_features: bool
                If True (default), waveforms are saved as features of the sorting extractor object
            recompute_info: bool
                If True, waveforms are recomputed (default False)
            verbose: bool
                If True output is verbose

    Returns
    -------
    max_channels: list
        List of int containing extracted maximum channels for each unit
    '''
    if isinstance(unit_ids, (int, np.integer)):
        unit_ids = [unit_ids]
    elif unit_ids is None:
        unit_ids = sorting.get_unit_ids()
    elif not isinstance(unit_ids, (list, np.ndarray)):
        raise Exception("unit_ids is not a valid in valid")

    assert np.all([u in sorting.get_unit_ids() for u in unit_ids]), "Invalid unit_ids"

    assert max_channels <= recording.get_num_channels(), f"'max_channels' must be less or equal than " \
                                                         f"{recording.get_num_channels()} (number of channels)"

    params_dict = update_all_param_dicts_with_kwargs(kwargs)
    recompute_info = params_dict['recompute_info']
    grouping_property = params_dict['grouping_property']
    save_property_or_features = params_dict['save_property_or_features']

    max_list = []
    if 'max_channel' in sorting.get_shared_unit_property_names() and not recompute_info:
        for unit_id in unit_ids:
            max_channel = sorting.get_unit_property(unit_id, 'max_channel')
            max_list.append(max_channel)
    else:
        for i, unit_id in enumerate(unit_ids):
            if 'template' in sorting.get_unit_property_names(unit_id) and not recompute_info:
                template = sorting.get_unit_property(unit_id, 'template')
            else:
                template = get_unit_templates(recording, sorting, unit_id, channel_ids, mode=mode, **kwargs)[0]
            if channel_ids is None:
                channel_ids = recording.get_channel_ids()
            if max_channels == 1:
                if peak == 'both':
                    max_channel_idxs = np.unravel_index(np.argmax(np.abs(template)),
                                                        template.shape)[0]
                elif peak == 'neg':
                    max_channel_idxs = np.unravel_index(np.argmin(template),
                                                        template.shape)[0]
                elif peak == 'pos':
                    max_channel_idxs = np.unravel_index(np.argmax(template),
                                                        template.shape)[0]
                else:
                    raise ValueError("'peak' can be 'both' (default), 'pos', or 'neg'")
                if grouping_property is not None:
                    assert 'group' in sorting.get_unit_property_names(unit_id), f"Unit {unit_id} does not have the " \
                                                                                f"'group' property "
                    unit_group = sorting.get_unit_property(unit_id, "group")
                    subrecs = se.get_sub_extractors_by_property(recording, "group")
                    subrec_groups = [np.unique(subrec.get_channel_groups()) for subrec in subrecs]
                    subrec_with_unit = subrecs[subrec_groups.index(unit_group)]
                    max_channel = subrec_with_unit.get_channel_ids()[max_channel_idxs]
                else:
                    max_channel = channel_ids[max_channel_idxs]
            else:
                # find peak time
                if peak == 'both':
                    peak_idx = np.unravel_index(np.argmax(np.abs(template)),
                                                template.shape)[1]
                    max_channel_idxs = np.argsort(np.abs(template[:, peak_idx]))[::-1][:max_channels]
                elif peak == 'neg':
                    peak_idx = np.unravel_index(np.argmin(template),
                                                template.shape)[1]
                    max_channel_idxs = np.argsort(template[:, peak_idx])[:max_channels]
                elif peak == 'pos':
                    peak_idx = np.unravel_index(np.argmax(template),
                                                template.shape)[1]
                    max_channel_idxs = np.argsort(template[:, peak_idx])[::-1][:max_channels]
                else:
                    raise ValueError("'peak' can be 'both' (default), 'pos', or 'neg'")
                if grouping_property is not None:
                    assert 'group' in sorting.get_unit_property_names(unit_id), f"Unit {unit_id} does not have the " \
                                                                                f"'group' property "
                    unit_group = sorting.get_unit_property(unit_id, "group")
                    subrecs = se.get_sub_extractors_by_property(recording, "group")
                    subrec_groups = [np.unique(subrec.get_channel_groups()) for subrec in subrecs]
                    subrec_with_unit = subrecs[subrec_groups.index(unit_group)]
                    max_channel = list(np.array(subrec_with_unit.get_channel_ids())[max_channel_idxs])
                else:
                    max_channel = list(np.array(channel_ids)[max_channel_idxs])

            if save_property_or_features:
                sorting.set_unit_property(unit_id, 'max_channel', max_channel)

            max_list.append(max_channel)

    return max_list


def get_unit_amplitudes(recording, sorting, unit_ids=None, channel_ids=None, return_idxs=False, **kwargs):
    '''
    Computes the spike amplitudes from a recording and sorting extractor. Amplitudes can be computed
    in absolute value (uV) or relative to the template amplitude.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor
    sorting: SortingExtractor
        The sorting extractor
    unit_ids: list
        List of unit ids to extract maximum channels
    channel_ids: list
        List of channels ids to compute amplitudes from
    return_idxs: bool
        If True, spike indexes and channel indexes are returned
    **kwargs: Keyword arguments
        A dictionary with default values can be retrieved with:
        st.postprocessing.get_waveforms_params():
            method: str
                If 'absolute' (default), amplitudes are absolute amplitudes in uV are returned.
                If 'relative', amplitudes are returned as ratios between waveform amplitudes and template amplitudes.
            peak: str
                If maximum channel has to be found among negative peaks ('neg'), positive ('pos') or
                both ('both' - default)
            frames_before: int
                Frames before peak to compute amplitude
            frames_after: float
                Frames after peak to compute amplitude
            max_spikes_per_unit: int
                The maximum number of spikes to extract per unit
            memmap: bool
                If True, waveforms are saved as memmap object (recommended for long recordings with many channels)
            seed: int
                Random seed for extracting random waveforms
            save_property_or_features: bool
                If True (default), waveforms are saved as features of the sorting extractor object
            recompute_info: bool
                If True, waveforms are recomputed (default False)
            n_jobs: int
                Number of jobs for parallelization. Default is None (no parallelization)
            joblib_backend: str
                The backend for joblib. Default is 'loky'
            verbose: bool
                If True output is verbose

    Returns
    -------
    amplitudes: list
        List of int containing extracted amplitudes for each unit
    indexes: list
        List of spike indexes for which amplitudes are computed. Returned if 'return_idxs' is True
    '''

    if isinstance(unit_ids, (int, np.integer)):
        unit_ids = [unit_ids]
    elif unit_ids is None:
        unit_ids = sorting.get_unit_ids()
    elif not isinstance(unit_ids, (list, np.ndarray)):
        raise Exception("unit_ids is not a valid in valid")
    assert np.all([u in sorting.get_unit_ids() for u in unit_ids]), "Invalid unit_ids"

    params_dict = update_all_param_dicts_with_kwargs(kwargs)
    method = params_dict['method']
    peak = params_dict['peak']
    frames_before = int(params_dict['frames_before'])
    frames_after = int(params_dict['frames_after'])
    memmap = params_dict['memmap']
    max_spikes_per_unit = params_dict['max_spikes_per_unit']
    save_property_or_features = params_dict['save_property_or_features']
    recompute_info = params_dict['recompute_info']
    ms_before = params_dict['ms_before']
    dtype = recording.get_dtype()
    assert peak in ['neg', 'pos', 'both'], "'peak' can be 'neg', 'pos', or 'both'"

    amp_list = []
    spike_index_list = []
    center_frame = int(ms_before / 1000 * recording.get_sampling_frequency())
    if 'amplitudes' in sorting.get_shared_unit_spike_feature_names() and not recompute_info:
        for unit_id in unit_ids:
            amplitudes = sorting.get_unit_spike_features(unit_id, 'amplitudes')
            amp_list.append(amplitudes)
            if len(amplitudes) < len(sorting.get_unit_spike_train(unit_id)):
                indexes = sorting.get_unit_spike_features(unit_id, 'amplitudes_idxs')
            else:
                indexes = np.arange(len(amplitudes))
            spike_index_list.append(indexes)
    else:
        # pre-construct memmap arrays
        if memmap:
            for unit_id in unit_ids:
                fname = 'amplitudes_' + str(unit_id) + '.raw'
                len_amp = len(sorting.get_unit_spike_train(unit_id))
                if max_spikes_per_unit is not None:
                    if len_amp > max_spikes_per_unit:
                        len_amp = max_spikes_per_unit
                shape = len_amp
                arr = sorting.allocate_array(shape=shape, dtype=dtype, name=fname, memmap=memmap)
                amp_list.append(arr)
        else:
            amp_list = [[] for ii in range(len(unit_ids))]

        waveforms, spike_index_list, channel_index_list = get_unit_waveforms(recording, sorting, unit_ids, channel_ids,
                                                                             return_idxs=True, **kwargs)
        templates = [np.median(wf, 0) for wf in waveforms]
        max_channels = [np.unravel_index(np.argmax(np.abs(t)), t.shape)[0] for t in templates]

        for i, (u, wf) in enumerate(zip(unit_ids, waveforms)):
            wf_cut = wf[:, max_channels[i], center_frame - frames_before:center_frame + frames_after]
            if peak == 'both':
                amps = np.max(np.abs(wf_cut), axis=-1)
                if len(amps.shape) > 1:
                    amps = np.max(wf)
            elif peak == 'neg':
                amps = np.min(wf_cut, axis=-1)
                if len(amps.shape) > 1:
                    amps = np.min(wf, axis=-1)
            else:  # 'pos'
                amps = np.max(wf_cut, axis=-1)
                if len(amps.shape) > 1:
                    amps = np.max(amps, axis=-1)

            if method == 'relative':
                amps /= np.median(amps)
            amps = amps.astype(dtype)

            if memmap:
                amp_list[i] = amps
                del amps
            else:
                amp_list[i] = amps

        if save_property_or_features:
            for i, unit_id in enumerate(unit_ids):
                sorting.set_unit_spike_features(unit_id, 'amplitudes', amp_list[i], indexes=spike_index_list[i])

    if return_idxs:
        return amp_list, spike_index_list
    else:
        return amp_list


def compute_channel_spiking_activity(recording, channel_ids=None, detect_threshold=5, detect_sign=-1, start_frame=None,
                                     end_frame=None, chunk_size=None, chunk_mb=500, **kwargs):
    '''
    Computes spiking rate for each channel.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor
    channel_ids: list
        List of channels ids to compute activity from
    detect_threshold: float
        Detection of threshold in MAD times
    detect_sign: int
        Sign of the detection: -1 (negative), 1 (positive), 0 (both)
    start_frame: int
        Start frame to compute activity
    end_frame: int
        End frame to compute activity
    chunk_size: int
        Size of chunks in number of samples. If None, it is automatically calculated
    chunk_mb: int
        Size of chunks in Mb (default 500 Mb)
    **kwargs: Keyword arguments
        A dictionary with default values can be retrieved with:
        st.postprocessing.get_common_params():
            save_property_or_features: bool
                If True (default), waveforms are saved as features of the sorting extractor object
            recompute_info: bool
                If True, waveforms are recomputed (default False)
            n_jobs: int
                Number of parallel jobs (default 1)
            verbose: bool
                If True output is verbose

    Returns
    -------
    spike_rates: np.array
        Array with spike rate value for each channel
    spike_amplitudes: np.array
        Array with spike amplitude value for each channel
    '''
    params_dict = update_all_param_dicts_with_kwargs(kwargs)
    recompute_info = params_dict['recompute_info']
    save_property_or_features = params_dict['save_property_or_features']
    n_jobs = params_dict['n_jobs']
    joblib_backend = params_dict['joblib_backend']
    verbose = params_dict['verbose']

    if isinstance(channel_ids, (int, np.integer)):
        channel_ids = [channel_ids]

    if channel_ids is None:
        channel_ids = recording.get_channel_ids()

    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = recording.get_num_frames()

    assert np.all([ch in recording.get_channel_ids() for ch in channel_ids]), "Invalid channel_ids"

    spike_rates = np.zeros(len(channel_ids))
    spike_amplitudes = np.zeros(len(channel_ids))

    if 'spike_rate' in recording.get_shared_channel_property_names() and \
            'spike_amplitude' in recording.get_shared_channel_property_names() and not recompute_info:
        for i, ch in enumerate(recording.get_channel_ids()):
            spike_rates[i] = recording.get_channel_property(ch, 'spike_rate')
            spike_amplitudes[i] = recording.get_channel_property(ch, 'spike_amplitude')
    else:
        sort_detect = st.sortingcomponents.detect_spikes(recording, channel_ids=channel_ids,
                                                         detect_threshold=detect_threshold, detect_sign=detect_sign,
                                                         n_jobs=n_jobs, joblib_backend=joblib_backend,
                                                         start_frame=start_frame, end_frame=end_frame,
                                                         chunk_size=chunk_size, chunk_mb=chunk_mb,
                                                         verbose=verbose)

        for i, unit in enumerate(sort_detect.get_unit_ids()):
            spike_rates[i] = sort_detect.get_unit_property(unit, 'spike_rate')
            spike_amplitudes[i] = sort_detect.get_unit_property(unit, 'spike_amplitude')

        if save_property_or_features:
            for i, ch in enumerate(recording.get_channel_ids()):
                recording.set_channel_property(ch, 'spike_rate', spike_rates[i])
                recording.set_channel_property(ch, 'spike_amplitude', spike_amplitudes[i])

    return spike_rates, spike_amplitudes


def compute_unit_centers_of_mass(recording, sorting, unit_ids=None, num_channels=10, **kwargs):
    '''
    Computes the center of mass (COM) of a unit based on the template amplitudes.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor
    sorting: SortingExtractor
        The sorting extractor
    unit_ids: list
        List of unit ids to extract maximum channels
    num_channels: int
        Number of max channels to be used in the calculation of COM
    **kwargs: Keyword arguments
        A dictionary with default values can be retrieved with:
        st.postprocessing.get_waveforms_params():
            method: str
                If 'absolute' (default), amplitudes are absolute amplitudes in uV are returned.
                If 'relative', amplitudes are returned as ratios between waveform amplitudes and template amplitudes.
            peak: str
                If maximum channel has to be found among negative peaks ('neg'), positive ('pos') or
                both ('both' - default)
            frames_before: int
                Frames before peak to compute amplitude
            frames_after: float
                Frames after peak to compute amplitude
            max_spikes_per_unit: int
                The maximum number of spikes to extract per unit
            memmap: bool
                If True, waveforms are saved as memmap object (recommended for long recordings with many channels)
            seed: int
                Random seed for extracting random waveforms
            save_property_or_features: bool
                If True (default), waveforms are saved as features of the sorting extractor object
            recompute_info: bool
                If True, waveforms are recomputed (default False)
            n_jobs: int
                Number of jobs for parallelization. Default is None (no parallelization).
            joblib_backend: str
                The backend for joblib. Default is 'loky'.
            verbose: bool
                If True output is verbose

    Returns
    -------
    centers_of_mass: list
        List of int containing extracted COMs for each unit
    '''

    if isinstance(unit_ids, (int, np.integer)):
        unit_ids = [unit_ids]
    elif unit_ids is None:
        unit_ids = sorting.get_unit_ids()
    elif not isinstance(unit_ids, (list, np.ndarray)):
        raise Exception("unit_ids is not a valid in valid")
    assert np.all([u in sorting.get_unit_ids() for u in unit_ids]), "Invalid unit_ids"

    params_dict = update_all_param_dicts_with_kwargs(kwargs)
    peak = params_dict['peak']
    save_property_or_features = params_dict['save_property_or_features']
    recompute_info = params_dict['recompute_info']
    assert peak in ['neg', 'pos', 'both'], "'peak' can be 'neg', 'pos', or 'both'"

    if num_channels is None:
        num_channels = recording.get_num_channels()
    locations = recording.get_channel_locations()

    coms = []
    for i, unit_id in enumerate(unit_ids):
        if 'template' in sorting.get_unit_property_names(unit_id) and not recompute_info:
            template = sorting.get_unit_property(unit_id, 'template')
        else:
            template = get_unit_templates(recording, sorting, unit_id, **kwargs)[0]
        if peak == 'both':
            amps = np.max(np.abs(template), 1)
        elif peak == 'neg':
            amps = np.min(template, 1)
        elif peak == 'pos':
            amps = np.max(template, 1)

        idxs = np.argsort(amps)[::-1][:num_channels]
        com = np.array([np.sum((amps[idxs] * locations[idxs, 0])) / np.sum(amps[idxs]),
                        np.sum((amps[idxs] * locations[idxs, 1])) / np.sum(amps[idxs])])
        coms.append(com)

    if save_property_or_features:
        for i, unit_id in enumerate(unit_ids):
            sorting.set_unit_property(unit_id, 'com', coms[i])

    return coms


def compute_unit_pca_scores(recording, sorting, unit_ids=None, channel_ids=None, return_idxs=False, _waveforms=None,
                            _spike_index_list=None, _channel_index_list=None, **kwargs):
    '''
    Computes the PCA scores from the unit waveforms. If waveforms are not found as features, they are computed.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor
    sorting: SortingExtractor
        The sorting extractor
    unit_ids: list
        List of unit ids to compute pca scores
    channel_ids: list
        List of channels ids to compute pca from
    return_idxs: list
        List of indexes of used spikes for each unit
    _waveforms: list
        Pre-computed waveforms (optional)
    _spike_index_list: list
        Pre-computed spike indexes for waveforms (optional)
    _channel_index_list: list
        Pre-computed channel indexes for waveforms (optional)
    **kwargs: Keyword arguments
        A dictionary with default values can be retrieved with:
        st.postprocessing.get_waveforms_params():
            n_comp: int
                Number of PCA components (default 3)
            by_electrode: bool
                If True, PCA scores are computed electrode-wise (channel by channel)
            max_spikes_for_pca: int
                The maximum number of spike per unit to use to fit the PCA.
            whiten: bool
                If True, PCA is run with whiten equal True
            grouping_property: str
                Property to group channels. E.g. if the recording extractor has the 'group' property and
                'grouping_property' is 'group', then waveforms are computed group-wise.
            ms_before: float
                Time period in ms to cut waveforms before the spike events
            ms_after: float
                Time period in ms to cut waveforms after the spike events
            dtype: dtype
                The numpy dtype of the waveforms
            compute_property_from_recording: bool
                If True and 'grouping_property' is given, the property of each unit is assigned as the corresponding
                property of the recording extractor channel on which the average waveform is the largest
            max_channels_per_waveforms: int or None
                Maximum channels per waveforms to return. If None, all channels are returned
            n_jobs: int
                Number of parallel jobs (default 1)
            max_spikes_per_unit: int
                The maximum number of spikes to extract per unit
            memmap: bool
                If True, waveforms are saved as memmap object (recommended for long recordings with many channels)
            seed: int
                Random seed for extracting random waveforms
            save_property_or_features: bool
                If True (default), waveforms are saved as features of the sorting extractor object
            recompute_info: bool
                If True, waveforms are recomputed (default False)
            verbose: bool
                If True output is verbose

    Returns
    -------
    pcs_scores: list
        List of np.array containing extracted pca scores.
        If 'by_electrode' is False, the array has shape (n_spikes, n_comp)
        If 'by_electrode' is True, the array has shape (n_spikes, n_channels, n_comp)
    indexes: list
        List of spike indexes for which pca scores are computed. Returned if 'return_idxs' is True
    '''
    if isinstance(unit_ids, (int, np.integer)):
        unit_ids = [unit_ids]
    elif unit_ids is None:
        unit_ids = sorting.get_unit_ids()
    elif not isinstance(unit_ids, (list, np.ndarray)):
        raise Exception("unit_ids is not a valid in valid")
    assert np.all([u in sorting.get_unit_ids() for u in unit_ids]), "Invalid unit_ids"
    unit_ids = list(unit_ids)

    params_dict = update_all_param_dicts_with_kwargs(kwargs)
    max_spikes_for_pca = params_dict['max_spikes_for_pca']
    save_property_or_features = params_dict['save_property_or_features']
    verbose = params_dict['verbose']
    recompute_info = params_dict['recompute_info']
    by_electrode = params_dict['by_electrode']

    pca_scores_list = []
    if 'pca_scores' in sorting.get_shared_unit_spike_feature_names() and not recompute_info:
        spike_index_list = []
        channel_index_list = []
        for unit_id in unit_ids:
            pca_scores = sorting.get_unit_spike_features(unit_id, 'pca_scores')
            pca_scores_list.append(pca_scores)
            if len(pca_scores) < len(sorting.get_unit_spike_train(unit_id)):
                indexes = sorting.get_unit_spike_features(unit_id, 'pca_scores_idxs')
            else:
                indexes = np.arange(len(pca_scores))
            if 'pca_scores_channel_idxs' in sorting.get_shared_unit_property_names() and by_electrode:
                channel_idxs = sorting.get_unit_property(unit_id, 'pca_scores_channel_idxs')
            else:
                channel_idxs = np.arange(recording.get_num_channels())
            spike_index_list.append(indexes)
            channel_index_list.append(channel_idxs)
    else:
        nspikes = []
        if _waveforms is None:
            if verbose:
                print("Computing waveforms")
            waveforms, spike_index_list, channel_index_list = get_unit_waveforms(recording, sorting, unit_ids,
                                                                                 channel_ids,
                                                                                 return_idxs=True, **kwargs)
        else:
            assert _spike_index_list is not None and _channel_index_list is not None, "Provide spike_index_list and " \
                                                                                      "channel_index_list with " \
                                                                                      "waveforms"
            waveforms = _waveforms
            spike_index_list = _spike_index_list
            channel_index_list = _channel_index_list

        # compute len of all waveforms (computed for all units)
        n_waveforms = 0
        n_waveforms_fit = 0
        for wf in waveforms:
            n_spikes = len(wf)
            n_waveforms += n_spikes
            if max_spikes_for_pca is not None:
                n_waveforms_fit += min(n_spikes, max_spikes_for_pca)
            else:
                n_waveforms_fit += n_spikes
        wf_shape = waveforms[0].shape

        memmap = params_dict['memmap']
        seed = params_dict['seed']
        n_comp = params_dict['n_comp']
        whiten = params_dict['whiten']

        dtype = recording.get_dtype()
        # prepare all waveforms
        if by_electrode:
            waveforms_pca_fit = sorting.allocate_array(name='waveforms_pca_fit.raw', dtype=dtype,
                                                       shape=(n_waveforms_fit * wf_shape[1], wf_shape[2]),
                                                       memmap=memmap)
        else:
            waveforms_pca_fit = sorting.allocate_array(name='waveforms_pca_fit.raw', dtype=dtype,
                                                       shape=(n_waveforms_fit, wf_shape[1] * wf_shape[2]),
                                                       memmap=memmap)

        # concatenate all waveforms
        if not isinstance(waveforms, list):
            # single unit
            waveforms = [waveforms]
            spike_index_list = [spike_index_list]

        i_start = 0
        for i_w, wf in enumerate(waveforms):
            if max_spikes_for_pca is not None:
                idxs = np.random.choice(np.arange(wf.shape[0]), min(max_spikes_for_pca, wf.shape[0]), replace=False)
            else:
                idxs = np.arange(wf.shape[0])


            if by_electrode:
                wf_reshaped = wf[idxs].reshape((len(idxs) * wf.shape[1], wf.shape[2]))
                nspikes.append(len(wf) * recording.get_num_channels())
            else:
                wf_reshaped = wf[idxs].reshape((len(idxs), wf.shape[1] * wf.shape[2]))
                nspikes.append(len(wf))
            waveforms_pca_fit[i_start:i_start + wf_reshaped.shape[0]] = wf_reshaped
            i_start += wf_reshaped.shape[0]

        pca = PCA(n_components=n_comp, whiten=whiten, random_state=seed)

        if verbose:
            print("Fitting PCA of %d dimensions on %d waveforms" % (n_comp, n_waveforms_fit))
        pca.fit(
            waveforms_pca_fit[np.random.RandomState(seed=seed).permutation(len(waveforms_pca_fit))[:n_waveforms_fit]])

        if verbose:
            print("Projecting waveforms on PC")
        # project waveforms on principal components
        for unit_id in unit_ids:
            idx_waveform = unit_ids.index(unit_id)
            wf = waveforms[idx_waveform]
            if by_electrode:
                pct = np.dot(wf, pca.components_.T)
            else:
                pct = np.dot(wf.reshape((wf.shape[0], wf.shape[1] * wf.shape[2])), pca.components_.T)
            if whiten:
                pct /= np.sqrt(pca.explained_variance_)
            pca_scores = sorting.allocate_array(array=pct, name='pcascores_' + str(unit_id) + '.raw', memmap=memmap)
            pca_scores_list.append(pca_scores)

        if save_property_or_features:
            for i, unit_id in enumerate(unit_ids):
                sorting.set_unit_spike_features(unit_id, 'pca_scores', pca_scores_list[i], indexes=spike_index_list[i])
                if len(channel_index_list[i]) < recording.get_num_channels():
                    sorting.set_unit_property(unit_id, 'pca_scores_channel_idxs', channel_index_list[i])

    if return_idxs:
        return pca_scores_list, spike_index_list, np.array(channel_index_list)
    else:
        return pca_scores_list


def set_unit_properties_by_max_channel_properties(recording, sorting, property, unit_ids=None, peak='both',
                                                  mode='median', verbose=False, **kwargs):
    '''
    Extracts 'property' from recording channel with largest peak for each unit and saves it as unit property.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor
    sorting: SortingExtractor
        The sorting extractor
    property: str
        Property to compute
    unit_ids: list
        List of unit ids to extract maximum channels
    peak: str
        If maximum channel has to be found among negative peaks ('neg'), positive ('pos') or both ('both' - default)
    mode: str
        Use 'mean' or 'median' to compute templates
    verbose: bool
        If True output is verbose
    **kwargs: Keyword arguments
        A dictionary with default values can be retrieved with:
        st.postprocessing.get_waveforms_params():
            grouping_property: str
                Property to group channels. E.g. if the recording extractor has the 'group' property and
                'grouping_property' is 'group', then waveforms are computed group-wise.
            ms_before: float
                Time period in ms to cut waveforms before the spike events
            ms_after: float
                Time period in ms to cut waveforms after the spike events
            dtype: dtype
                The numpy dtype of the waveforms
            max_spikes_per_unit: int
                The maximum number of spikes to extract per unit
            compute_property_from_recording: bool
                If True and 'grouping_property' is given, the property of each unit is assigned as the corresponding
                property of the recording extractor channel on which the average waveform is the largest
            seed: int
                Random seed for extracting random waveforms
            n_jobs: int
                Number of parallel jobs (default 1)
            memmap: bool
                If True, waveforms are saved as memmap object (recommended for long recordings with many channels)
            max_channels_per_waveforms: int or None
                Maximum channels per waveforms to return. If None, all channels are returned
    '''

    if property not in recording.get_shared_channel_property_names():
        raise Exception("'property' should be in recording properties")

    if isinstance(unit_ids, (int, np.integer)):
        unit_ids = [unit_ids]
    elif unit_ids is None:
        unit_ids = sorting.get_unit_ids()
    elif not isinstance(unit_ids, (list, np.ndarray)):
        raise Exception("unit_ids is not a valid in valid")
    assert np.all([u in sorting.get_unit_ids() for u in unit_ids]), "Invalid unit_ids"

    if 'max_channel' in sorting.get_shared_unit_property_names():
        if verbose:
            print("Using 'template' property")
        max_chan_property = True
    else:
        if verbose:
            print("Computing templates")
        max_chan_property = False

    for i, unit_id in enumerate(unit_ids):
        if property not in sorting.get_unit_property_names(unit_id):
            if max_chan_property:
                max_chan = sorting.get_unit_property(unit_id, 'max_channel')
            else:
                max_chan = get_unit_max_channels(recording, sorting, unit_id, mode=mode, peak=peak,
                                                 **kwargs)[0]
            sorting.set_unit_property(unit_id, property, recording.get_channel_property(max_chan, property))


def export_to_phy(recording, sorting, output_folder, compute_pc_features=True,
                  compute_amplitudes=True, max_channels_per_template=16, copy_binary=True,
                  **kwargs):
    '''
    Exports paired recording and sorting extractors to phy template-gui format.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor
    sorting: SortingExtractor
        The sorting extractor
    output_folder: str
        The output folder where the phy template-gui files are saved
    compute_pc_features: bool
        If True (default), pc features are computed
    compute_amplitudes: bool
        If True (default), waveforms amplitudes are compute
    max_channels_per_template: int or None
        Maximum channels per unit to return. If None, all channels are returned
    copy_binary: bool
        If True, the recording is copied and saved in the phy 'output_folder'. If False and the
        'recording' is a CacheRecordingExtractor or a BinDatRecordingExtractor, then a relative
        link to the file recording location is used. Otherwise, the recording is not copied and the
        recording path is set to 'None'. (default True)
    **kwargs: Keyword arguments
        A dictionary with default values can be retrieved with:
        st.postprocessing.get_waveforms_params():
            n_comp: int
                Number of PCA components (default 3)
            max_spikes_for_pca: int
                The maximum number of spikes per unit to use to fit the PCA.
            whiten: bool
                If True, PCA is run with whiten equal True
            grouping_property: str
                Property to group channels. E.g. if the recording extractor has the 'group' property and
                'grouping_property' is 'group', then waveforms are computed group-wise.
            ms_before: float
                Time period in ms to cut waveforms before the spike events
            ms_after: float
                Time period in ms to cut waveforms after the spike events
            dtype: dtype
                The numpy dtype of the waveforms
            max_spikes_per_unit: int
                The maximum number of spikes to extract per unit
            compute_property_from_recording: bool
                If True and 'grouping_property' is given, the property of each unit is assigned as the corresponding
                property of the recording extractor channel on which the average waveform is the largest
            n_jobs: int
                Number of parallel jobs (default 1)
            joblib_backend: str
                The backend for joblib. Default is 'loky'.
            method: str
                If 'absolute' (default), amplitudes are absolute amplitudes in uV are returned.
                If 'relative', amplitudes are returned as ratios between waveform amplitudes and template amplitudes.
            peak: str
                If maximum channel has to be found among negative peaks ('neg'), positive ('pos') or
                both ('both' - default)
            frames_before: int
                Frames before peak to compute amplitude
            frames_after: float
                Frames after peak to compute amplitude
            recompute_info: bool
                If True, will always re-extract waveforms and templates.
            save_property_or_features: bool
                If True, will store all calculated features and properties
            verbose: bool
                If True output is verbose
            seed: int
                Random seed for extracting random waveforms
            memmap: bool
                If True, waveforms are saved as memmap object (recommended for long recordings with many channels)
            filter_flag: bool
                If False, will not display the warning on non-filtered recording. Default is True.

    '''
    if not isinstance(recording, se.RecordingExtractor) or not isinstance(sorting, se.SortingExtractor):
        raise AttributeError('recording and sorting must be extractor objects')

    empty_flag = False
    for unit_id in sorting.get_unit_ids():
        spikes = sorting.get_unit_spike_train(unit_id)
        if spikes.shape[0] == 0:
            empty_flag = True
    if empty_flag:
        print('Warning: empty units have been removed when being exported to Phy')
        sorting = st.curation.threshold_num_spikes(sorting, 1, "less")

    filter_flag = True if not "filter_flag" in kwargs else kwargs['filter_flag']
    if not recording.is_filtered and filter_flag:
        print("Warning: recording is not filtered! It's recommended to filter the recording before exporting to phy.\n"
              "You can run spiketoolkit.preprocessing.bandpass_filter(recording)")

    if len(sorting.get_unit_ids()) == 0:
        raise Exception("No non-empty units in the sorting result, can't save to phy.")

    output_folder = Path(output_folder).absolute()
    if output_folder.is_dir():
        shutil.rmtree(output_folder)
    output_folder.mkdir()

    params_dict = update_all_param_dicts_with_kwargs(kwargs)
    dtype = params_dict['dtype']
    verbose = params_dict['verbose']

    # save dat file
    if dtype is None:
        dtype = recording.get_dtype()

    if copy_binary:
        rec_path = 'recording.dat'  # Use relative path in this case
        recording.write_to_binary_dat_format(output_folder / rec_path, dtype=dtype)
    elif isinstance(recording, se.CacheRecordingExtractor):
        rec_path = str(Path(recording.filename).absolute())
        dtype = recording.get_dtype()
    elif isinstance(recording, se.BinDatRecordingExtractor):
        rec_path = str(Path(recording._datfile).absolute())
        dtype = recording.get_dtype()
    else:  # don't save recording.dat
        rec_path = 'None'

    dtype = np.dtype(dtype).name

    # write params.py
    with (output_folder / 'params.py').open('w') as f:
        f.write(f"dat_path = r'{str(rec_path)}'\n")
        f.write(f"n_channels_dat = {recording.get_num_channels()}\n")
        f.write(f"dtype = '{str(dtype)}'\n")
        f.write(f"offset = 0\n")
        f.write(f"sample_rate = {recording.get_sampling_frequency()}\n")
        f.write(f"hp_filtered = {recording.is_filtered}")

    if verbose:
        print('Converting to Phy format')
    spike_times, spike_clusters, amplitudes, channel_map, pc_features, pc_feature_ind, \
    spike_templates, templates, templates_ind, similar_templates, channel_map_si, channel_groups, \
    positions = _get_phy_data(recording, sorting, compute_pc_features, compute_amplitudes,
                              max_channels_per_template, **kwargs)

    # Save .tsv metadata
    with (output_folder / 'cluster_group.tsv').open('w') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
        writer.writerow(['cluster_id', 'group'])
        for i, u in enumerate(sorting.get_unit_ids()):
            writer.writerow([i, 'unsorted'])
    if 'group' in sorting.get_shared_unit_property_names():
        with (output_folder / 'cluster_channel_group.tsv').open('w') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
            writer.writerow(['cluster_id', 'ch_group'])
            for i, u in enumerate(sorting.get_unit_ids()):
                writer.writerow([i, sorting.get_unit_property(u, 'group')])
    else:
        with (output_folder / 'cluster_channel_group.tsv').open('w') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
            writer.writerow(['cluster_id', 'chan_group'])
            for i, u in enumerate(sorting.get_unit_ids()):
                writer.writerow([i, 0])

    if verbose:
        print('Saving files')
    if compute_amplitudes:
        np.save(str(output_folder / 'amplitudes.npy'), amplitudes)
    np.save(str(output_folder / 'spike_times.npy'), spike_times)
    np.save(str(output_folder / 'spike_templates.npy'), spike_templates)
    np.save(str(output_folder / 'spike_clusters.npy'), spike_clusters)
    if compute_pc_features:
        np.save(str(output_folder / 'pc_features.npy'), pc_features)
        np.save(str(output_folder / 'pc_feature_ind.npy'), pc_feature_ind)
    np.save(str(output_folder / 'templates.npy'), templates)
    np.save(str(output_folder / 'template_ind.npy'), templates_ind)
    np.save(str(output_folder / 'similar_templates.npy'), similar_templates.astype('float32'))
    np.save(str(output_folder / 'channel_map.npy'), channel_map.astype('uint32'))
    np.save(str(output_folder / 'channel_map_si.npy'), channel_map_si.astype('uint32'))
    np.save(str(output_folder / 'channel_positions.npy'), positions.astype('float32'))
    np.save(str(output_folder / 'channel_groups.npy'), channel_groups.astype('uint32'))

    if verbose:
        print('Saved phy format to: ', output_folder)
        print('Run:\n\nphy template-gui ', str(output_folder / 'params.py'))


def _compute_templates_similarity(templates, template_ind=None):
    similarity = np.zeros((len(templates), len(templates)))
    if template_ind is None:
        template_ind = np.tile(np.arange(templates[0].shape[0]), (len(templates), 1))

    for i, (t_i, t_ind_i) in enumerate(zip(templates, template_ind)):
        for j, (t_j, t_ind_j) in enumerate(zip(templates, template_ind)):
            shared_channel_idxs = [ch for ch in t_ind_i if
                                   ch in t_ind_j and not ch < 0]  # ch<0 is for channels empty, label -1
            if len(shared_channel_idxs) > 0:
                # reorder channels
                reorder_t_ind_i = np.zeros(len(shared_channel_idxs), dtype='int')
                reorder_t_ind_j = np.zeros(len(shared_channel_idxs), dtype='int')
                for s, sc in enumerate(shared_channel_idxs):
                    reorder_t_ind_i[s] = np.where(t_ind_i == sc)[0]
                    reorder_t_ind_j[s] = np.where(t_ind_j == sc)[0]
                t_i_shared = t_i[:, reorder_t_ind_i]
                t_j_shared = t_j[:, reorder_t_ind_j]
                t_i_lin = t_i_shared.reshape(t_i_shared.shape[0] * t_i_shared.shape[1])
                t_j_lin = t_j_shared.reshape(t_i_shared.shape[0] * t_i_shared.shape[1])
                a = np.corrcoef(t_i_lin, t_j_lin)
                # weight similarity based on proportion of shared channels
                sim = np.abs(a[0, 1]) * len(shared_channel_idxs) / len(t_ind_i)
                similarity[i, j] = sim
            else:
                # no channels are shared
                similarity[i, j] = 0
    return similarity


def _get_random_spike_waveforms(recording, sorting, unit, max_spikes_per_unit, snippet_len, channel_ids=None, seed=0):
    st = sorting.get_unit_spike_train(unit_id=unit)
    num_events = len(st)
    if num_events > max_spikes_per_unit:
        event_indexes = np.sort(np.random.RandomState(seed=seed).choice(range(num_events), size=max_spikes_per_unit,
                                                                        replace=False))
    else:
        event_indexes = range(num_events)

    spikes = recording.get_snippets(reference_frames=st[event_indexes].astype('int64'),
                                    snippet_len=snippet_len, channel_ids=channel_ids)
    return spikes, event_indexes


def _get_spike_times_clusters(sorting):
    if not isinstance(sorting, se.SortingExtractor):
        raise AttributeError()
    if len(sorting.get_unit_ids()) == 0:
        raise Exception("No units in the sorting result, can't compute any metric information.")

    # spike times.npy and spike clusters.npy
    spike_times = np.array([])
    spike_clusters = np.array([])

    for i_u, unit_id in enumerate(sorting.get_unit_ids()):
        spike_train = sorting.get_unit_spike_train(unit_id)
        cl = [i_u] * len(sorting.get_unit_spike_train(unit_id))
        spike_times = np.concatenate((spike_times, np.array(spike_train)))
        spike_clusters = np.concatenate((spike_clusters, np.array(cl)))

    sorting_idxs = np.argsort(spike_times)
    spike_times = spike_times[sorting_idxs, np.newaxis]
    spike_clusters = spike_clusters[sorting_idxs, np.newaxis].astype(int)

    return spike_times, spike_clusters


def _get_amp_metric_data(recording, sorting, **kwargs):
    params_dict = update_all_param_dicts_with_kwargs(kwargs)
    memmap = params_dict['recompute_info']

    # amplitudes.npy
    amplitudes_list, amp_idxs = get_unit_amplitudes(recording, sorting, return_idxs=True, **kwargs)

    # compute len of all waveforms (computed for all units)
    n_spikes = 0
    n_amps = 0  # n_pca and n_amps are he same (max_spikes_per_unit)
    for i, (unit_id, amp) in enumerate(zip(sorting.get_unit_ids(), amplitudes_list)):
        n_spikes += len(sorting.get_unit_spike_train(unit_id))
        n_amps += len(amp)

    spike_times = sorting.allocate_array(shape=(n_spikes, 1), dtype=np.uint32, name='spike_times.raw',
                                         memmap=memmap)
    spike_clusters = sorting.allocate_array(shape=(n_spikes, 1), dtype=np.uint32, name='spike_clusters.raw',
                                            memmap=memmap)
    spike_times_amps = sorting.allocate_array(shape=(n_amps, 1), dtype=np.uint32, name='spike_times_amps.raw',
                                              memmap=memmap)
    spike_clusters_amps = sorting.allocate_array(shape=(n_amps, 1), dtype=np.uint32, name='spike_clusters_amps.raw',
                                                 memmap=memmap)
    amplitudes = sorting.allocate_array(shape=(n_amps, 1), dtype=np.float32, name='amplitudes.raw', memmap=memmap)

    i_start_st = 0
    i_start_amp = 0
    for i_u, id in enumerate(sorting.get_unit_ids()):
        st = sorting.get_unit_spike_train(id)
        cl = [i_u] * len(st)
        amp = amplitudes_list[i_u]

        # take care of amps and pca computed on subset of spikes
        if len(amp) < len(st):
            cl_amp = [i_u] * len(amp)
            st_amp = st[amp_idxs[i_u]]
        else:
            cl_amp = [i_u] * len(st)
            st_amp = st

        # assign
        spike_times[i_start_st:i_start_st + len(st)] = st[:, np.newaxis]
        spike_clusters[i_start_st:i_start_st + len(st)] = np.array(cl)[:, np.newaxis]
        spike_times_amps[i_start_amp:i_start_amp + len(st_amp)] = st_amp[:, np.newaxis]
        spike_clusters_amps[i_start_amp:i_start_amp + len(st_amp)] = np.array(cl_amp)[:, np.newaxis]
        amplitudes[i_start_amp:i_start_amp + len(st_amp)] = amp[:, np.newaxis]
        i_start_st += len(st)
        i_start_amp += len(st_amp)

    sorting_idxs = np.argsort(spike_times[:, 0])
    sorting_idxs_amps = np.argsort(spike_times_amps[:, 0])

    spike_times[:] = spike_times[sorting_idxs]
    spike_times_amps[:] = spike_times_amps[sorting_idxs_amps]
    spike_clusters[:] = spike_clusters[sorting_idxs]
    spike_clusters_amps[:] = spike_clusters_amps[sorting_idxs_amps]
    amplitudes[:] = amplitudes[sorting_idxs_amps]

    return spike_times, spike_times_amps, spike_clusters, spike_clusters_amps, amplitudes


def _get_pca_metric_data(recording, sorting, **kwargs):
    params_dict = update_all_param_dicts_with_kwargs(kwargs)
    recompute_info = params_dict['recompute_info']
    memmap = params_dict['recompute_info']
    if recompute_info:
        sorting.clear_units_spike_features(feature_name='waveforms')

    pc_list, pca_idxs, pc_ind = compute_unit_pca_scores(recording, sorting, return_idxs=True, **kwargs)

    # compute len of all waveforms (computed for all units)
    n_spikes = 0
    n_pca = 0  # n_pca and n_amps are he same (max_spikes_per_unit)
    for i, (unit_id, pc) in enumerate(zip(sorting.get_unit_ids(), pc_list)):
        n_spikes += len(sorting.get_unit_spike_train(unit_id))
        n_pca += len(pc)
    pc_shape = pc_list[0].shape

    spike_times = sorting.allocate_array(shape=(n_spikes, 1), dtype=np.uint32, name='spike_times.raw',
                                         memmap=memmap)
    spike_clusters = sorting.allocate_array(shape=(n_spikes, 1), dtype=np.uint32, name='spike_clusters.raw',
                                            memmap=memmap)
    spike_times_pca = sorting.allocate_array(shape=(n_pca, 1), dtype=np.uint32, name='spike_times_pca.raw',
                                             memmap=memmap)
    spike_clusters_pca = sorting.allocate_array(shape=(n_pca, 1), dtype=np.uint32, name='spike_clusters_pca.raw',
                                                memmap=memmap)
    pc_features = sorting.allocate_array(shape=(n_pca, pc_shape[2], pc_shape[1]), dtype=np.float32,
                                         name='pc_features.raw', memmap=memmap)

    i_start_st = 0
    i_start_pc = 0
    for i_u, id in enumerate(sorting.get_unit_ids()):
        st = sorting.get_unit_spike_train(id)
        cl = [i_u] * len(st)
        pc = pc_list[i_u]

        # take care of amps and pca computed on subset of spikes
        if len(pc) < len(st):
            cl_pca = [i_u] * len(pc)
            st_pca = st[pca_idxs[i_u]]
        else:
            cl_pca = [i_u] * len(st)
            st_pca = st

        # assign
        spike_times[i_start_st:i_start_st + len(st)] = st[:, np.newaxis]
        spike_clusters[i_start_st:i_start_st + len(st)] = np.array(cl)[:, np.newaxis]
        spike_times_pca[i_start_pc:i_start_pc + len(st_pca)] = st_pca[:, np.newaxis]
        spike_clusters_pca[i_start_pc:i_start_pc + len(st_pca)] = np.array(cl_pca)[:, np.newaxis]
        pc_features[i_start_pc:i_start_pc + len(st_pca)] = pc.swapaxes(1, 2)
        i_start_st += len(st)
        i_start_pc += len(st_pca)

    sorting_idxs = np.argsort(spike_times[:, 0])
    sorting_idxs_pca = np.argsort(spike_times_pca[:, 0])

    spike_times[:] = spike_times[sorting_idxs]
    spike_times_pca[:] = spike_times_pca[sorting_idxs_pca]
    spike_clusters[:] = spike_clusters[sorting_idxs]
    spike_clusters_pca[:] = spike_clusters_pca[sorting_idxs_pca]
    pc_features[:] = pc_features[sorting_idxs_pca]
    pc_feature_ind = pc_ind

    return spike_times, spike_times_pca, spike_clusters, spike_clusters_pca, pc_features, pc_feature_ind


def _get_quality_metric_data(recording, sorting, n_comp, ms_before, ms_after, dtype, amp_method, amp_peak,
                             amp_frames_before, amp_frames_after, max_spikes_per_unit, max_spikes_for_amplitudes,
                             max_spikes_for_pca, recompute_info, max_channels_per_waveforms,
                             save_property_or_features, n_jobs, joblib_backend, verbose, seed, memmap,
                             compute_pc_features=True, compute_amplitudes=True):
    if recompute_info:
        sorting.clear_units_spike_features(feature_name='waveforms')
        sorting.clear_units_spike_features(feature_name='amplitudes')
        sorting.clear_units_spike_features(feature_name='pca_scores')

    if compute_pc_features or compute_amplitudes:
        max_spikes_per_unit = None

        # check if recomputation is needed
        if 'waveforms' in sorting.get_shared_unit_spike_feature_names():
            unit_ids = sorting.get_unit_ids()
            spike_times = sorting.get_units_spike_train(unit_ids)
            waveforms = [sorting.get_unit_spike_features(u, 'waveforms') for u in unit_ids]

            recompute_info = False
            if np.any([len(wf) < len(times) for (wf, times) in zip(waveforms, spike_times)]):
                print("Recomputing waveforms on all spikes")
                recompute_info = True
            if waveforms[0].shape[1] < recording.get_num_channels():
                print("Recomputing waveforms on all channels")
                recompute_info = True

        if recompute_info:
            sorting.clear_units_property("template")

        waveforms, spike_index_list, channel_index_list = get_unit_waveforms(recording, sorting,
                                                                             max_spikes_per_unit=max_spikes_per_unit,
                                                                             ms_before=ms_before,
                                                                             ms_after=ms_after, dtype=dtype,
                                                                             save_property_or_features=
                                                                             save_property_or_features,
                                                                             verbose=verbose,
                                                                             n_jobs=n_jobs,
                                                                             joblib_backend=joblib_backend,
                                                                             seed=seed,
                                                                             memmap=memmap, return_idxs=True,
                                                                             max_channels_per_waveforms=
                                                                             max_channels_per_waveforms,
                                                                             recompute_info=recompute_info)
    else:
        waveforms, spike_index_list, channel_index_list = None, None, None

    if compute_pc_features:
        # check if recomputation is needed
        if 'pca_scores' in sorting.get_shared_unit_spike_feature_names():
            unit_ids = sorting.get_unit_ids()
            spike_times = sorting.get_units_spike_train(unit_ids)
            pca_scores = [sorting.get_unit_spike_features(u, 'pca_scores') for u in unit_ids]

            if np.any([len(pc) < len(times) for (pc, times) in zip(pca_scores, spike_times)]):
                recompute_info = True
            else:
                recompute_info = False

        # pca scores
        if recompute_info:
            sorting.clear_units_spike_features(feature_name='pca_scores')

        pc_list, pca_idxs, pc_ind = compute_unit_pca_scores(recording, sorting, n_comp=n_comp, by_electrode=True,
                                                            max_spikes_per_unit=max_spikes_per_unit,
                                                            ms_before=ms_before,
                                                            ms_after=ms_after, dtype=dtype,
                                                            save_property_or_features=save_property_or_features,
                                                            max_spikes_for_pca=max_spikes_for_pca,
                                                            verbose=verbose,
                                                            seed=seed,
                                                            memmap=memmap, return_idxs=True,
                                                            max_channels_per_waveforms=max_channels_per_waveforms,
                                                            _waveforms=waveforms, _spike_index_list=spike_index_list,
                                                            _channel_index_list=channel_index_list)
        pc_shape = pc_list[0].shape
    else:
        pc_list, pca_idxs, pc_ind, pc_shape = None, None, None, None

    if compute_amplitudes:
        max_spikes_for_amplitudes = None

        # check if recomputation is needed
        if 'amplitudes' in sorting.get_shared_unit_spike_feature_names():
            unit_ids = sorting.get_unit_ids()
            spike_times = sorting.get_units_spike_train(unit_ids)
            amplitudes = [sorting.get_unit_spike_features(u, 'amplitudes') for u in unit_ids]

            if np.any([len(amp) < len(times) for (amp, times) in zip(amplitudes, spike_times)]):
                recompute_info = True
            else:
                recompute_info = False

        # amplitudes
        if recompute_info:
            sorting.clear_units_spike_features(feature_name='amplitudes')

        amplitudes_list, amp_idxs = get_unit_amplitudes(recording, sorting, method=amp_method,
                                                        save_property_or_features=save_property_or_features,
                                                        peak=amp_peak, max_spikes_per_unit=max_spikes_for_amplitudes,
                                                        frames_before=amp_frames_before, frames_after=amp_frames_after,
                                                        seed=seed, memmap=memmap, n_jobs=n_jobs,
                                                        joblib_backend=joblib_backend, return_idxs=True)
    else:
        amplitudes_list, amp_idxs = None, None

    # templates
    templates = get_unit_templates(recording, sorting,
                                   save_property_or_features=save_property_or_features,
                                   seed=seed, _waveforms=waveforms)

    if channel_index_list is None:
        if max_channels_per_waveforms == recording.get_num_channels():
            channel_index_list = [np.arange(recording.get_num_channels()) for u in sorting.get_unit_ids()]
        else:
            channel_index_list = [select_max_channels_from_templates(temp, recording, max_channels_per_waveforms)
                                  for temp in templates]

    # compute len of all waveforms (computed for all units)
    n_spikes = 0
    for i, unit_id in enumerate(sorting.get_unit_ids()):
        n_spikes += len(sorting.get_unit_spike_train(unit_id))

    n_pca_amps = 0  # n_pca and n_amps are the same (max_spikes_per_unit)
    if compute_pc_features:
        for i, pc in enumerate(pc_list):
            n_pca_amps += len(pc)
    elif compute_amplitudes:
        for i, amp in enumerate(amplitudes_list):
            n_pca_amps += len(amp)

    spike_times = sorting.allocate_array(shape=(n_spikes, 1), dtype=np.uint32, name='spike_times.raw',
                                         memmap=memmap)
    spike_clusters = sorting.allocate_array(shape=(n_spikes, 1), dtype=np.uint32, name='spike_clusters.raw',
                                            memmap=memmap)

    if compute_amplitudes:
        spike_times_amps = sorting.allocate_array(shape=(n_pca_amps, 1), dtype=np.uint32, name='spike_times_amps.raw',
                                                  memmap=memmap)
        spike_clusters_amps = sorting.allocate_array(shape=(n_pca_amps, 1), dtype=np.uint32,
                                                     name='spike_clusters_amps.raw',
                                                     memmap=memmap)
        amplitudes = sorting.allocate_array(shape=(n_pca_amps, 1), dtype=np.float32, name='amplitudes.raw',
                                            memmap=memmap)
    else:
        spike_times_amps, spike_clusters_amps, amplitudes = None, None, None

    if compute_pc_features:
        spike_times_pca = sorting.allocate_array(shape=(n_pca_amps, 1), dtype=np.uint32, name='spike_times_pca.raw',
                                                 memmap=memmap)
        spike_clusters_pca = sorting.allocate_array(shape=(n_pca_amps, 1), dtype=np.uint32,
                                                    name='spike_clusters_pca.raw',
                                                    memmap=memmap)
        pc_features = sorting.allocate_array(shape=(n_pca_amps, pc_shape[2], pc_shape[1]), dtype=np.float32,
                                             name='pc_features.raw', memmap=memmap)
    else:
        spike_times_pca = None
        spike_clusters_pca = None
        pc_features = None

    i_start_st = 0
    i_start_pc = 0
    i_start_amp = 0
    for i_u, id in enumerate(sorting.get_unit_ids()):
        st = sorting.get_unit_spike_train(id)
        cl = [i_u] * len(st)

        # take care of amps and pca computed on subset of spikes
        if compute_pc_features:
            pc = pc_list[i_u]
            if len(pc) < len(st):
                cl_pca = [i_u] * len(pc)
                st_pca = st[pca_idxs[i_u]]
            else:
                cl_pca = [i_u] * len(st)
                st_pca = st
        if compute_amplitudes:
            amp = amplitudes_list[i_u]
            if len(amp) < len(st):
                cl_amp = [i_u] * len(amp)
                st_amp = st[amp_idxs[i_u]]
            else:
                cl_amp = [i_u] * len(st)
                st_amp = st

        # assign
        spike_times[i_start_st:i_start_st + len(st)] = st[:, np.newaxis]
        spike_clusters[i_start_st:i_start_st + len(st)] = np.array(cl)[:, np.newaxis]

        if compute_amplitudes:
            spike_times_amps[i_start_amp:i_start_amp + len(st_amp)] = st_amp[:, np.newaxis]
            spike_clusters_amps[i_start_amp:i_start_amp + len(st_amp)] = np.array(cl_amp)[:, np.newaxis]
            amplitudes[i_start_amp:i_start_amp + len(st_amp)] = amp[:, np.newaxis]
            i_start_amp += len(st_amp)

        if compute_pc_features:
            spike_times_pca[i_start_pc:i_start_pc + len(st_pca)] = st_pca[:, np.newaxis]
            spike_clusters_pca[i_start_pc:i_start_pc + len(st_pca)] = np.array(cl_pca)[:, np.newaxis]
            pc_features[i_start_pc:i_start_pc + len(st_pca)] = pc.swapaxes(1, 2)
            i_start_pc += len(st_pca)

        i_start_st += len(st)

    sorting_idxs = np.argsort(spike_times[:, 0])
    spike_times[:] = spike_times[sorting_idxs]
    spike_clusters[:] = spike_clusters[sorting_idxs]

    if compute_amplitudes:
        sorting_idxs_amps = np.argsort(spike_times_amps[:, 0])
        spike_times_amps[:] = spike_times_amps[sorting_idxs_amps]
        spike_clusters_amps[:] = spike_clusters_amps[sorting_idxs_amps]
        amplitudes[:] = amplitudes[sorting_idxs_amps]

    if compute_pc_features:
        sorting_idxs_pca = np.argsort(spike_times_pca[:, 0])
        spike_times_pca[:] = spike_times_pca[sorting_idxs_pca]
        spike_clusters_pca[:] = spike_clusters_pca[sorting_idxs_pca]
        pc_features[:] = pc_features[sorting_idxs_pca]
    pc_feature_ind = pc_ind

    return spike_times, spike_times_amps, spike_times_pca, spike_clusters, spike_clusters_amps, spike_clusters_pca, \
           amplitudes, pc_features, pc_feature_ind, templates, channel_index_list


def _get_phy_data(recording, sorting, compute_pc_features, compute_amplitudes,
                  max_channels_per_template, **kwargs):
    if not isinstance(recording, se.RecordingExtractor) or not isinstance(sorting, se.SortingExtractor):
        raise AttributeError()
    if len(sorting.get_unit_ids()) == 0:
        raise Exception("No units in the sorting result, can't compute phy information.")

    params_dict = update_all_param_dicts_with_kwargs(kwargs)
    n_comp = params_dict['n_comp']
    max_spikes_for_pca = params_dict['max_spikes_for_pca']
    recompute_info = params_dict['recompute_info']
    save_property_or_features = params_dict['save_property_or_features']
    verbose = params_dict['verbose']
    grouping_property = params_dict['grouping_property']
    ms_before = params_dict['ms_before']
    ms_after = params_dict['ms_after']
    dtype = params_dict['dtype']
    memmap = params_dict['memmap']
    n_jobs = params_dict['n_jobs']
    joblib_backend = params_dict['joblib_backend']
    seed = params_dict['seed']
    amp_method = params_dict['method']
    amp_peak = params_dict['peak']
    amp_frames_before = int(params_dict['frames_before'])
    amp_frames_after = int(params_dict['frames_after'])
    max_spikes_per_unit_amp = np.inf

    if compute_pc_features:
        max_spikes_per_unit_wf = np.inf
    else:
        max_spikes_per_unit_wf = params_dict['max_spikes_per_unit']

    if max_channels_per_template is None:
        max_channels_per_template = recording.get_num_channels()

    if recompute_info:
        sorting.clear_units_property(property_name='template')
        sorting.clear_units_spike_features(feature_name='waveforms')
        sorting.clear_units_spike_features(feature_name='amplitudes')

    # pc_features.npy - [nSpikes, nFeaturesPerChannel, nPCFeatures] single
    if grouping_property in recording.get_shared_channel_property_names():
        groups, num_chans_in_group = np.unique(recording.get_channel_groups(), return_counts=True)
        max_num_chans_in_group = np.max(num_chans_in_group)
        channel_groups = recording.get_channel_groups()
        if max_channels_per_template < recording.get_num_channels():
            print("Disabling 'max_channels_per_template'. Channels are extracted using 'grouping_property'")
            max_channels_per_template = recording.get_num_channels()
    else:
        max_num_chans_in_group = recording.get_num_channels()
        channel_groups = np.array([0] * recording.get_num_channels())

    spike_times, spike_times_amps, spike_times_pca, spike_clusters, spike_clusters_amps, spike_clusters_pca, \
    amplitudes, pc_features, pc_feature_ind, templates, channel_index_list \
        = _get_quality_metric_data(recording, sorting, n_comp=n_comp, ms_before=ms_before, ms_after=ms_after,
                                   dtype=dtype, amp_method=amp_method, amp_peak=amp_peak,
                                   amp_frames_before=amp_frames_before,
                                   amp_frames_after=amp_frames_after, max_spikes_per_unit=max_spikes_per_unit_wf,
                                   max_spikes_for_amplitudes=max_spikes_per_unit_amp,
                                   max_spikes_for_pca=max_spikes_for_pca,
                                   n_jobs=n_jobs, joblib_backend=joblib_backend, recompute_info=recompute_info,
                                   max_channels_per_waveforms=max_channels_per_template,
                                   save_property_or_features=save_property_or_features, verbose=verbose, memmap=memmap,
                                   seed=seed, compute_pc_features=compute_pc_features,
                                   compute_amplitudes=compute_amplitudes)

    channel_map = np.arange(recording.get_num_channels())
    channel_map_si = np.array(recording.get_channel_ids())

    # channel_positions.npy
    if 'location' in recording.get_shared_channel_property_names():
        positions = np.array([recording.get_channel_property(chan, 'location')
                              for chan in recording.get_channel_ids()])
    else:
        if verbose:
            print("'location' property is not available and it will be linear.")
        positions = np.zeros((recording.get_num_channels(), 2))
        positions[:, 1] = np.arange(recording.get_num_channels())

    # templates.npy
    templates = np.array(templates, dtype='float32').swapaxes(1, 2)

    if grouping_property in recording.get_shared_channel_property_names():
        if grouping_property not in sorting.get_shared_unit_property_names():
            set_unit_properties_by_max_channel_properties(recording, sorting, grouping_property, seed=seed)
        # pc_feature_ind = np.zeros((len(sorting.get_unit_ids()), int(max_num_chans_in_group)), dtype=int)
        templates_ind = np.zeros((len(sorting.get_unit_ids()), int(max_num_chans_in_group)), dtype=int)
        templates_red = np.zeros((templates.shape[0], templates.shape[1], int(max_num_chans_in_group)))

        for u_i, u in enumerate(sorting.get_unit_ids()):
            group = sorting.get_unit_property(u, 'group')
            unit_chans_idxs = []
            for ch in recording.get_channel_ids():
                if recording.get_channel_property(ch, 'group') == group:
                    unit_chans_idxs.append(list(channel_map_si).index(ch))
            if len(unit_chans_idxs) == 0:
                raise Exception("Sorting extractor has different property than recording extractor."
                                "They should correspond.")
            if len(unit_chans_idxs) != max_num_chans_in_group:
                # add empty channel
                lacking_channels = max_num_chans_in_group - len(unit_chans_idxs)
                append_chan = [-1] * lacking_channels
                unit_chans = np.array(unit_chans_idxs + append_chan)
                templates_ind[u_i] = unit_chans
                templates_red[u_i, :, :len(unit_chans_idxs)] = templates[u_i, :, unit_chans_idxs].T
            else:
                unit_chans = np.array(unit_chans_idxs)
                templates_ind[u_i] = unit_chans
                templates_red[u_i, :] = templates[u_i, :, unit_chans].T
        templates = templates_red
    elif max_channels_per_template < recording.get_num_channels():
        # waveforms, templates, and pc_scores are computed on the same channels
        templates_ind = np.array(channel_index_list)
        if templates.shape[2] > max_channels_per_template:
            # reduce template based on template indexes
            templates_red = np.zeros((templates.shape[0], templates.shape[1], templates_ind.shape[1]))
            for u_i, u in enumerate(sorting.get_unit_ids()):
                templates_red[u_i] = templates[u_i, :, templates_ind[u_i]].T
            templates = templates_red
    else:
        templates_ind = np.tile(np.arange(recording.get_num_channels()), (len(sorting.get_unit_ids()), 1))

    # Reorder template with amplitude for phy
    templates, templates_ind = _template_descending_order(recording, templates, templates_ind)

    # similar_templates.npy - [nTemplates, nTemplates] single
    similar_templates = _compute_templates_similarity(templates, templates_ind)

    # spike_templates.npy - [nSpikes, ] uint32
    spike_templates = spike_clusters

    return spike_times, spike_clusters, amplitudes, channel_map, pc_features, pc_feature_ind, \
           spike_templates, templates, templates_ind, similar_templates, channel_map_si, channel_groups, positions


def _template_descending_order(recording, templates, templates_ind):
    # Reorder template with amplitude for phy
    for n, template in enumerate(templates):
        max_channel_idx = np.unravel_index(np.argmax(np.abs(template)),
                                           template.shape)[1]
        locs = [i for _ind, i in enumerate(recording.get_channel_locations()) if _ind in templates_ind[n]]
        loc_max = locs[max_channel_idx]
        distances = [np.linalg.norm(l - loc_max) for l in locs]
        max_channel_idxs = np.argsort(distances)

        # If dead channel and grouping_property, template_ind end with -1 (30 line before),
        # fill max_channel_idxs with -1
        if len(max_channel_idxs) < len(templates_ind[n]):
            max_channel_idxs = list(max_channel_idxs) + [-1] * (len(templates_ind[n]) - len(max_channel_idxs))

        templates[n] = templates[n, :, max_channel_idxs].T
        templates_ind[n] = templates_ind[n, max_channel_idxs]
    return templates, templates_ind


def _extract_waveforms_one_chunk(i, rec_arg, chunks, unit_ids, n_pad, times_in_chunk, cumulative_n_spikes,
                                 waveforms_file, memmap, dtype, verbose):
    chunk = chunks[i]
    times_this_chunk = times_in_chunk[i]
    n_spikes = cumulative_n_spikes[i]

    if verbose:
        print(f"Chunk {i + 1}: extracting waveforms")
    if isinstance(rec_arg, dict):
        recording = se.load_extractor_from_dict(rec_arg)
    else:
        recording = rec_arg
    t_start = time.perf_counter()
    # chunk: {istart, iend, istart_with_padding, iend_with_padding} # include padding
    recording_chunk = se.SubRecordingExtractor(
        parent_recording=recording,
        start_frame=chunk['istart_with_padding'],
        end_frame=chunk['iend_with_padding']
    )

    # num_events_in_chunk x num_channels_in_nbhd[unit_id] x len_of_one_snippet
    unit_waveforms = get_unit_waveforms_for_chunk(
        recording=recording_chunk,
        chunk=chunk,
        unit_ids=unit_ids,
        snippet_len=n_pad,
        times_in_chunk=times_this_chunk
    )
    t_stop = time.perf_counter()
    if verbose:
        print(f"Chunk {i + 1}: waveforms extracted in {t_stop - t_start}s")

    if memmap:
        for i_unit, unit in enumerate(unit_ids):
            wf = unit_waveforms[i_unit]
            wf = wf.astype(dtype)

            if len(wf) > 0:
                waveforms_file[i_unit][n_spikes[i_unit]:n_spikes[i_unit] + len(wf)] = wf
        return None
    else:
        return unit_waveforms
