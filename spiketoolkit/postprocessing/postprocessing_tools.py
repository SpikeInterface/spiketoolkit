import numpy as np
import spiketoolkit as st
import spikeextractors as se
from sklearn.decomposition import PCA
from pathlib import Path
import warnings
import tempfile
import shutil
from joblib import Parallel, delayed
from spikeextractors import RecordingExtractor, SortingExtractor
import csv


def get_unit_waveforms(recording, sorting, unit_ids=None, grouping_property=None, channel_ids=None,
                       ms_before=3., ms_after=3., dtype=None, max_spikes_per_unit=300,
                       save_as_features=True, compute_property_from_recording=False, verbose=False,
                       seed=0, memmap=False, n_jobs=None, max_channels_per_waveforms=None, return_idxs=False):
    '''
    Computes the spike waveforms from a recording and sorting extractor.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor
    sorting: SortingExtractor
        The sorting extractor
    unit_ids: list
        List of unit ids to extract waveforms
    grouping_property: str
        Property to group channels. E.g. if the recording extractor has the 'group' property and 'grouping_property' is
        'group', then waveforms are computed group-wise.
    channel_ids: list
        List of channels ids to compute waveforms from
    ms_before: float
        Time period in ms to cut waveforms before the spike events
    ms_after: float
        Time period in ms to cut waveforms after the spike events
    dtype: dtype
        The numpy dtype of the waveforms
    max_spikes_per_unit: int
        The maximum number of spikes to extract per unit.
    save_as_features: bool
        If True (default), waveforms are saved as features of the sorting extractor object
    compute_property_from_recording: bool
        If True and 'grouping_property' is given, the property of each unit is assigned as the corresponding property
        of the recording extractor channel on which the average waveform is the largest
    verbose: bool
        If True output is verbose
    seed: int
        Random seed for extracting random waveforms
    memmap: bool
        If True, waveforms are saved as memmap object (recommended for long recordings with many channels)
    max_channels_per_waveforms: int or None
        Maximum channels per waveforms to return. If None, all channels are returned
    return_idxs: bool
        If True, spike indexes and channel indexes are returned

    Returns
    -------
    waveforms: list
        List of np.array (n_spikes, n_channels, n_timepoints) containing extracted waveforms for each unit
    spike_indexes: list
        List of spike indices for which waveforms are computed. Returned if 'return_idxs' is True
    channel_indexes: list
        List of max channel indexes
    '''
    if isinstance(unit_ids, (int, np.integer)):
        unit_ids = [unit_ids]
    elif unit_ids is None:
        unit_ids = sorting.get_unit_ids()
    elif not isinstance(unit_ids, (list, np.ndarray)):
        raise Exception("unit_ids is not a valid in valid")

    assert np.all([u in sorting.get_unit_ids() for u in unit_ids]), "Invalid unit_ids"

    if dtype is None:
        dtype = recording.get_dtype()

    if n_jobs is None:
        n_jobs = 0

    fs = recording.get_sampling_frequency()
    n_pad = [int(ms_before * fs / 1000), int(ms_after * fs / 1000)]

    waveform_list = []
    spike_index_list = []
    channel_index_list = []

    if n_jobs in [0, 1]:
        for unit in unit_ids:
            waveforms, indices, max_channel_idxs = _compute_one_waveform(unit, recording, sorting, channel_ids,
                                                                         unit_ids, grouping_property,
                                                                         compute_property_from_recording,
                                                                         max_channels_per_waveforms,
                                                                         max_spikes_per_unit, n_pad,
                                                                         dtype, memmap, seed, save_as_features, verbose)

            waveform_list.append(waveforms)
            spike_index_list.append(indices)
            channel_index_list.append(max_channel_idxs)
    else:
        # if parallel and memmap, we need to allocate arrays before
        if memmap:
            memmap_fnames = []
            memmap_shapes = []
            if grouping_property is None:
                if max_channels_per_waveforms is None:
                    n_channels = recording.get_num_channels()
                elif max_channels_per_waveforms >= recording.get_num_channels():
                    n_channels = recording.get_num_channels()
                else:
                    n_channels = max_channels_per_waveforms
            else:
                pass

            for unit_id in unit_ids:
                fname = 'waveforms_' + str(unit_id) + '.raw'
                len_wf = len(sorting.get_unit_spike_train(unit_id))
                if max_spikes_per_unit is not None:
                    if len_wf > max_spikes_per_unit:
                        len_wf = max_spikes_per_unit
                shape = (len_wf, n_channels, sum(n_pad))
                sorting.allocate_array(shape=shape, dtype=dtype, name=fname, memmap=memmap)
                memmap_fnames.append(fname)
                memmap_shapes.append(shape)
            output_list = Parallel(n_jobs=n_jobs)(delayed(_compute_one_waveform)(unit, recording, sorting, channel_ids,
                                                                                 unit_ids, grouping_property,
                                                                                 compute_property_from_recording,
                                                                                 max_channels_per_waveforms,
                                                                                 max_spikes_per_unit, n_pad,
                                                                                 dtype, memmap, seed, save_as_features,
                                                                                 verbose, fname)
                                                  for (unit, fname) in zip(unit_ids, memmap_fnames))

            for i, out in enumerate(output_list):
                waveforms = np.memmap(memmap_fnames[i], dtype=dtype, mode='r+', shape=memmap_shapes[i])
                waveform_list.append(waveforms)
                spike_index_list.append(out[1])
                channel_index_list.append(out[2])
        else:
            output_list = Parallel(n_jobs=n_jobs)(delayed(_compute_one_waveform)(unit, recording, sorting, channel_ids,
                                                                                 unit_ids, grouping_property,
                                                                                 compute_property_from_recording,
                                                                                 max_channels_per_waveforms,
                                                                                 max_spikes_per_unit, n_pad,
                                                                                 dtype, memmap, seed, save_as_features,
                                                                                 verbose, None, )
                                                  for unit in unit_ids)

            for out in output_list:
                raise Exception
                waveform_list.append(out[0])
                spike_index_list.append(out[1])
                channel_index_list.append(out[2])

    # if save_as_features:
    #     if len(indices) < len(sorting.get_unit_spike_train(unit_id)):
    #         features = np.array([None] * len(sorting.get_unit_spike_train(unit_id)))
    #         for i, ind in enumerate(indices):
    #             features[ind] = waveforms[i]
    #     else:
    #         features = waveforms
    #     sorting.set_unit_spike_features(unit_id, 'waveforms', features)

    # if grouping_property is not None:
    #     if grouping_property not in recording.get_shared_channel_property_names():
    #         raise ValueError("'grouping_property' should be a property of recording extractors")
    #     if compute_property_from_recording:
    #         compute_sorting_group = True
    #     elif grouping_property not in sorting.get_shared_unit_property_names():
    #         warnings.warn('Grouping property not in sorting extractor. Computing it from the recording extractor')
    #         compute_sorting_group = True
    #     else:
    #         compute_sorting_group = False
    #     if verbose:
    #         print("Waveforms by property: ", grouping_property)
    #
    #     if not compute_sorting_group:
    #         rec_list, rec_props = recording.get_sub_extractors_by_property(grouping_property,
    #                                                                        return_property_list=True)
    #         sort_list, sort_props = sorting.get_sub_extractors_by_property(grouping_property,
    #                                                                        return_property_list=True)
    #         if len(rec_props) != len(sort_props):
    #             print('Different' + grouping_property + ' numbers: using largest number of ' + grouping_property)
    #             if len(rec_props) > len(sort_props):
    #                 for i_r, rec in enumerate(rec_props):
    #                     if rec not in sort_props:
    #                         print('Inserting None for property ', rec)
    #                         sort_list.insert(i_r, None)
    #             else:
    #                 for i_s, sort in enumerate(sort_props):
    #                     if sort not in rec_props:
    #                         rec_list.insert(i_s, None)
    #         else:
    #             assert len(rec_list) == len(sort_list)
    #
    #         if max_channels_per_waveforms is None:
    #             max_channels_per_waveforms = rec_list[0].get_num_channels()
    #
    #         for i_list, (rec, sort) in enumerate(zip(rec_list, sort_list)):
    #             if sort is not None and rec is not None:
    #                 for i, unit_id in enumerate(unit_ids):
    #                     fs = rec.get_sampling_frequency()
    #                     n_pad = [int(ms_before * fs / 1000), int(ms_after * fs / 1000)]
    #
    #                     if channel_ids is None:
    #                         channel_ids = rec.get_channel_ids()
    #
    #                     if max_spikes_per_unit is None:
    #                         max_spikes = len(sort.get_unit_spike_train(unit_id))
    #                     else:
    #                         max_spikes = max_spikes_per_unit
    #
    #                     if max_channels_per_waveforms is None:
    #                         max_channels_per_waveforms = len(rec.get_channel_ids())
    #
    #                     if verbose:
    #                         print('Waveform ' + str(i + 1) + '/' + str(len(unit_ids)))
    #                     wf, indices = _get_random_spike_waveforms(recording=rec,
    #                                                               sorting=sort,
    #                                                               unit=unit_id,
    #                                                               max_spikes_per_unit=max_spikes,
    #                                                               snippet_len=n_pad,
    #                                                               channel_ids=channel_ids,
    #                                                               seed=seed)
    #                     wf = wf.astype(dtype)
    #
    #                     if max_channels_per_waveforms < len(channel_ids):
    #                         max_channel_idxs = _select_max_channels(wf, rec, max_channels_per_waveforms)
    #                     else:
    #                         max_channel_idxs = np.arange(rec.get_num_channels())
    #                     channel_index_list.append(max_channel_idxs)
    #                     wf = wf[:, max_channel_idxs]
    #
    #                     waveforms = sorting.allocate_array(array=wf, name='waveforms_' + str(unit_id) + '.raw',
    #                                                        memmap=memmap)
    #
    #                     if save_as_features:
    #                         if len(indices) < len(sort.get_unit_spike_train(unit_id)):
    #                             features = np.array([None] * len(sorting.get_unit_spike_train(unit_id)))
    #                             for i, ind in enumerate(indices):
    #                                 features[ind] = waveforms[i]
    #                         else:
    #                             features = waveforms
    #                         sorting.set_unit_spike_features(unit_id, 'waveforms', features)
    #                     waveform_list.append(waveforms)
    #                     spike_index_list.append(indices)
    #     else:
    #         for i, unit_id in enumerate(unit_ids):
    #             if channel_ids is None:
    #                 channel_ids = recording.get_channel_ids()
    #
    #             rec = se.SubRecordingExtractor(recording, channel_ids=channel_ids)
    #             rec_groups = np.array(rec.get_channel_groups())
    #             groups, count = np.unique(rec_groups, return_counts=True)
    #
    #             if max_channels_per_waveforms is None:
    #                 max_channels_per_waveforms = np.max(count)
    #
    #             fs = rec.get_sampling_frequency()
    #             n_pad = [int(ms_before * fs / 1000), int(ms_after * fs / 1000)]
    #
    #             if max_spikes_per_unit is None:
    #                 max_spikes = len(sorting.get_unit_spike_train(unit_id))
    #             else:
    #                 max_spikes = max_spikes_per_unit
    #
    #             if verbose:
    #                 print('Waveform ' + str(i + 1) + '/' + str(len(unit_ids)))
    #             wf, indices = _get_random_spike_waveforms(recording=recording,
    #                                                       sorting=sorting,
    #                                                       unit=unit_id,
    #                                                       max_spikes_per_unit=max_spikes,
    #                                                       snippet_len=n_pad,
    #                                                       channel_ids=channel_ids,
    #                                                       seed=seed)
    #             wf = wf.astype(dtype)
    #
    #             mean_waveforms = np.squeeze(np.mean(wf, axis=0))
    #             max_amp_elec = np.unravel_index(mean_waveforms.argmin(), mean_waveforms.shape)[0]
    #             group = recording.get_channel_property(recording.get_channel_ids()[max_amp_elec], grouping_property)
    #             elec_group = np.where(rec_groups == group)
    #             wf = np.squeeze(wf[:, elec_group, :])
    #
    #             if max_channels_per_waveforms < len(elec_group[0]):
    #                 max_channel_idxs = _select_max_channels(wf, rec, max_channels_per_waveforms)
    #             else:
    #                 max_channel_idxs = np.arange(len(elec_group[0]))
    #             channel_index_list.append(max_channel_idxs)
    #             wf = wf[:, max_channel_idxs]
    #
    #             waveforms = sorting.allocate_array(array=wf, name='waveforms_' + str(unit_id) + '.raw',
    #                                                memmap=memmap)
    #
    #             if save_as_features:
    #                 if len(indices) < len(sorting.get_unit_spike_train(unit_id)):
    #                     features = np.array([None] * len(sorting.get_unit_spike_train(unit_id)))
    #                     for i, ind in enumerate(indices):
    #                         features[ind] = waveforms[i]
    #                 else:
    #                     features = waveforms
    #                 sorting.set_unit_spike_features(unit_id, 'waveforms', features)
    #             waveform_list.append(waveforms)
    #             spike_index_list.append(indices)
    # else:
    #     if channel_ids is None:
    #         channel_ids = recording.get_channel_ids()
    #
    #     if max_channels_per_waveforms is None:
    #         max_channels_per_waveforms = len(channel_ids)
    #
    #     for i, unit_id in enumerate(unit_ids):
    #         fs = recording.get_sampling_frequency()
    #         n_pad = [int(ms_before * fs / 1000), int(ms_after * fs / 1000)]
    #
    #         if max_spikes_per_unit is None:
    #             max_spikes = len(sorting.get_unit_spike_train(unit_id))
    #         else:
    #             max_spikes = max_spikes_per_unit
    #
    #         if verbose:
    #             print('Waveform ' + str(i + 1) + '/' + str(len(unit_ids)))
    #         wf, indices = _get_random_spike_waveforms(recording=recording,
    #                                                   sorting=sorting,
    #                                                   unit=unit_id,
    #                                                   max_spikes_per_unit=max_spikes,
    #                                                   snippet_len=n_pad,
    #                                                   channel_ids=channel_ids,
    #                                                   seed=seed)
    #         wf = wf.astype(dtype)
    #
    #         if max_channels_per_waveforms < len(channel_ids):
    #             max_channel_idxs = _select_max_channels(wf, recording, max_channels_per_waveforms)
    #         else:
    #             max_channel_idxs = np.arange(len(channel_ids))
    #         channel_index_list.append(max_channel_idxs)
    #         wf = wf[:, max_channel_idxs]
    #
    #         waveforms = sorting.allocate_array(array=wf, name='waveforms_' + str(unit_id) + '.raw',
    #                                            memmap=memmap)
    #
    #         if save_as_features:
    #             if len(indices) < len(sorting.get_unit_spike_train(unit_id)):
    #                 features = np.array([None] * len(sorting.get_unit_spike_train(unit_id)))
    #                 for i, ind in enumerate(indices):
    #                     features[ind] = waveforms[i]
    #             else:
    #                 features = waveforms
    #             sorting.set_unit_spike_features(unit_id, 'waveforms', features)
    #         waveform_list.append(waveforms)
    #         spike_index_list.append(indices)

    if return_idxs:
        return waveform_list, spike_index_list, channel_index_list
    else:
        return waveform_list


def get_unit_templates(recording, sorting, unit_ids=None, mode='median', grouping_property=None, save_as_property=True,
                       ms_before=3., ms_after=3., dtype=None, max_spikes_per_unit=300, save_wf_as_features=True,
                       compute_property_from_recording=False, verbose=False, recompute_waveforms=False, seed=0,
                       memmap=False, max_channels_per_waveforms=None, _waveforms=None):
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
    mode: str
        Use 'mean' or 'median' to compute templates
    grouping_property: str
        Property to group channels. E.g. if the recording extractor has the 'group' property and 'grouping_property' is
        'group', then waveforms are computed group-wise.
    save_as_property: bool
        If True (default), templates are saved as property of the sorting extractor object
    save_wf_as_features: bool
        If True (default), waveforms are saved as features of the sorting extractor object
    ms_before: float
        Time period in ms to cut waveforms before the spike events
    ms_after: float
        Time period in ms to cut waveforms after the spike events
    dtype: dtype
        The numpy dtype of the waveforms
    max_spikes_per_unit: int
        The maximum number of spikes to extract per unit.
    compute_property_from_recording: bool
        If True and 'grouping_property' is given, the property of each unit is assigned as the corresponding propery of
        the recording extractor channel on which the average waveform is the largest
    verbose: bool
        If True output is verbose
    recompute_waveforms: bool
        If True, waveforms are recomputed (default False)
    seed: int
        Random seed for extracting random waveforms

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

    if _waveforms is None:
        waveforms = []
        for i, unit_id in enumerate(unit_ids):
            if 'waveforms' in sorting.get_unit_spike_feature_names(unit_id) and not recompute_waveforms:
                wf = sorting.get_unit_spike_features(unit_id, 'waveforms')
                idx_not_none = np.array([i for i in range(len(waveforms)) if waveforms[i] is not None])
                if len(idx_not_none) != len(waveforms):
                    if verbose:
                        print("Using ", len(idx_not_none), " waveforms for unit ", unit_id)
                    wf = np.stack(wf[idx_not_none])
            else:
                wf = get_unit_waveforms(recording, sorting, unit_id, max_spikes_per_unit=max_spikes_per_unit,
                                        ms_before=ms_before, ms_after=ms_after,
                                        save_as_features=save_wf_as_features,
                                        grouping_property=grouping_property, dtype=dtype,
                                        compute_property_from_recording=compute_property_from_recording,
                                        memmap=memmap, max_channels_per_waveforms=max_channels_per_waveforms,
                                        verbose=verbose, seed=seed)[0]
            waveforms.append(wf)
    else:
        waveforms = _waveforms

    template_list = []
    for i, unit_id in enumerate(unit_ids):
        wf = waveforms[i]
        if mode == 'mean':
            template = np.mean(wf, axis=0)
        elif mode == 'median':
            template = np.median(wf, axis=0)
        else:
            raise Exception("'mode' can be 'mean' or 'median'")
        if save_as_property:
            sorting.set_unit_property(unit_id, 'template', template)

        template_list.append(template)
    return template_list


def get_unit_max_channels(recording, sorting, unit_ids=None, max_channels=1, peak='both', mode='median',
                          grouping_property=None, save_as_property=True, ms_before=3., ms_after=3., dtype=None,
                          max_spikes_per_unit=300, compute_property_from_recording=False, verbose=False,
                          recompute_templates=False, seed=0):
    '''
    Computes the spike maximum channels from a recording and sorting extractor. If templates are not found as property,
    they are computed.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor
    sorting: SortingExtractor
        The sorting extractor
    unit_ids: list
        List of unit ids to extract maximum channels
    max_channels: int
        Number of max channels per units to return (default=1)
    mode: str
        Use 'mean' or 'median' to compute templates
    grouping_property: str
        Property to group channels. E.g. if the recording extractor has the 'group' property and 'grouping_property' is
        'group', then waveforms are computed group-wise.
    save_as_property: bool
        If True (default), templates are saved as property of the sorting extractor object
    ms_before: float
        Time period in ms to cut waveforms before the spike events
    ms_after: float
        Time period in ms to cut waveforms after the spike events
    dtype: dtype
        The numpy dtype of the waveforms
    max_spikes_per_unit: int
        The maximum number of spikes to extract per unit.
    compute_property_from_recording: bool
        If True and 'grouping_property' is given, the property of each unit is assigned as the corresponding propery of
        the recording extractor channel on which the average waveform is the largest
    verbose: bool
        If True output is verbose
    recompute_templates: bool
        If True, templates are recomputed (default False)
    seed: int
        Random seed for extracting random waveforms

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

    max_list = []
    for i, unit_id in enumerate(unit_ids):
        if 'template' in sorting.get_unit_property_names(unit_id) and not recompute_templates:
            template = sorting.get_unit_property(unit_id, 'template')
        else:
            template = get_unit_templates(recording, sorting, unit_id, mode=mode,
                                          max_spikes_per_unit=max_spikes_per_unit,
                                          dtype=dtype, ms_before=ms_before, ms_after=ms_after,
                                          grouping_property=grouping_property, save_as_property=save_as_property,
                                          compute_property_from_recording=compute_property_from_recording,
                                          verbose=verbose, seed=seed)[0]
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
            max_channel = recording.get_channel_ids()[max_channel_idxs]
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
                raise Exception("'peak' can be 'neg', 'pos', or 'both'")
            max_channel = list(np.array(recording.get_channel_ids())[max_channel_idxs])

        if save_as_property:
            sorting.set_unit_property(unit_id, 'max_channel', max_channel)

        max_list.append(max_channel)

    return max_list


def get_unit_amplitudes(recording, sorting, unit_ids=None, method='absolute', save_as_features=True, peak='both',
                        frames_before=3, frames_after=3, max_spikes_per_unit=np.inf, seed=0, memmap=False,
                        return_idxs=False):
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
    method: str
        If 'absolute' (default), amplitudes are absolute amplitudes in uV are returned.
        If 'relative', amplitudes are returned as ratios between waveform amplitudes and template amplitudes.
    peak: str
        If maximum channel has to be found among negative peaks ('neg'), positive ('pos') or both ('both' - default)
    save_as_features: bool
        If True (default), amplitudes are saved as features of the sorting extractor object
    frames_before: int
        Frames before peak to compute amplitude
    frames_after: float
        Frames after peak to compute amplitude
    max_spikes_per_unit: int
        The maximum number of amplitudes to extract for each unit(default is np.inf). If less than np.inf,
        the amplitudes will be returned from a random permutation of the spikes.
    seed: int
            Random seed for reproducibility
    memmap: bool
        If True, amplitudes are saved as memmap object (recommended for long recordings with many channels)
    return_idxs: list
        List of indexes of used spikes for each unit

    Returns
    -------
    amplitudes: list
        List of int containing extracted amplitudes for each unit
    indexes: list
        List of spike indices for which amplitudes are computed. Returned if 'return_idxs' is True
    '''
    if isinstance(unit_ids, (int, np.integer)):
        unit_ids = [unit_ids]
    elif unit_ids is None:
        unit_ids = sorting.get_unit_ids()
    elif not isinstance(unit_ids, (list, np.ndarray)):
        raise Exception("unit_ids is not a valid in valid")
    assert np.all([u in sorting.get_unit_ids() for u in unit_ids]), "Invalid unit_ids"

    amp_list = []
    spike_index_list = []
    for i, unit_id in enumerate(unit_ids):
        spike_train = sorting.get_unit_spike_train(unit_id)
        if max_spikes_per_unit < len(spike_train):
            indices = np.random.RandomState(seed=seed).permutation(len(spike_train))[:max_spikes_per_unit]
        else:
            indices = np.arange(len(spike_train))
        spike_train = spike_train[indices]

        snippets = recording.get_snippets(reference_frames=spike_train,
                                          snippet_len=[frames_before, frames_after])
        if peak == 'both':
            amps = np.max(np.abs(snippets), axis=-1)
            if len(amps.shape) > 1:
                amps = np.max(amps, axis=-1)
        elif peak == 'neg':
            amps = np.min(snippets, axis=-1)
            if len(amps.shape) > 1:
                amps = np.min(amps, axis=-1)
        elif peak == 'pos':
            amps = np.max(snippets, axis=-1)
            if len(amps.shape) > 1:
                amps = np.max(amps, axis=-1)
        else:
            raise Exception("'peak' can be 'neg', 'pos', or 'both'")

        if method == 'relative':
            amps /= np.median(amps)

        amplitudes = sorting.allocate_array(array=amps, name='amplitudes_' + str(unit_id) + '.raw',
                                            memmap=memmap)

        if save_as_features:
            if len(indices) < len(spike_train):
                if 'amplitudes' not in sorting.get_unit_spike_feature_names(unit_id):
                    amp_features = np.array([None] * len(sorting.get_unit_spike_train(unit_id)))
                else:
                    amp_features = np.array(sorting.get_unit_spike_features(unit_id, 'amplitudes'))
                for i, ind in enumerate(indices):
                    amp_features[ind] = amplitudes[i]
            else:
                amp_features = amplitudes
            sorting.set_unit_spike_features(unit_id, 'amplitudes', amp_features)
        amp_list.append(amplitudes)
        spike_index_list.append(indices)

    if return_idxs:
        return amp_list, spike_index_list
    else:
        return amp_list


def compute_unit_pca_scores(recording, sorting, unit_ids=None, n_comp=3, by_electrode=False, grouping_property=None,
                            ms_before=3., ms_after=3., dtype=None,
                            max_spikes_per_unit=300, max_spikes_for_pca=10000, max_channels_per_waveforms=None,
                            save_as_features=False, save_waveforms_as_features=False,
                            compute_property_from_recording=False,
                            whiten=False, verbose=False, seed=0, memmap=False, return_idxs=False,
                            _waveforms=None, _spike_index_list=None, _channel_list=None):
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
    n_comp: int
        Number of PCA components (default 3)
    by_electrode: bool
        If True, PCA scores are computed electrode-wise (channel by channel)
    grouping_property: str
        Property to group channels. E.g. if the recording extractor has the 'group' property and 'grouping_property' is
        'group', then waveforms are computed group-wise.
    save_as_features: bool
        If True (default), pca scores are saved as features of the sorting extractor object
    save_waveforms_as_features: bool
        If True, waveforms are saved as features
    ms_before: float
        Time period in ms to cut waveforms before the spike events
    ms_after: float
        Time period in ms to cut waveforms after the spike events
    dtype: dtype
        The numpy dtype of the waveforms
    max_spikes_per_unit: int
        The maximum number of spikes to extract per unit.
    max_spikes_for_pca: int
        The maximum number of spikes to use to compute PCA
    compute_property_from_recording: bool
        If True and 'grouping_property' is given, the property of each unit is assigned as the corresponding propery of
        the recording extractor channel on which the average waveform is the largest
    whiten: bool
        If True, PCA is run with whiten equal True
    verbose: bool
        If True output is verbose
    seed: int
        Random seed for reproducibility
    memmap: bool
        If True, pca_scores are saved as memmap object (recommended for long recordings with many channels)
    return_idxs: list
        List of indexes of used spikes for each unit

    Returns
    -------
    pcs_scores: list
        List of np.array containing extracted pca scores.
        If 'by_electrode' is False, the array has shape (n_spikes, n_comp)
        If 'by_electrode' is True, the array has shape (n_spikes, n_channels, n_comp)
    indexes: list
        List of spike indices for which pca scores are computed. Returned if 'return_idxs' is True
    '''
    if isinstance(unit_ids, (int, np.integer)):
        unit_ids = [unit_ids]
    elif unit_ids is None:
        unit_ids = sorting.get_unit_ids()
    elif not isinstance(unit_ids, (list, np.ndarray)):
        raise Exception("unit_ids is not a valid in valid")
    assert np.all([u in sorting.get_unit_ids() for u in unit_ids]), "Invalid unit_ids"

    if max_spikes_per_unit is None:
        max_spikes_per_unit = np.inf
    if max_spikes_for_pca is None:
        max_spikes_for_pca = np.inf

    nspikes = []
    if _waveforms is None:
        if 'waveforms' in sorting.get_shared_unit_spike_feature_names():
            if verbose:
                print("Using 'waveforms' features")
            waveforms = []
            spike_index_list = []
            channel_list = []
            for unit_id in sorting.get_unit_ids():
                wf = sorting.get_unit_spike_features(unit_id, 'waveforms')
                if 'waveform_ind' in sorting.get_unit_property_names():
                    wf_idxs = sorting.get_unit_spike_features(unit_id, 'waveforms_idxs')
                else:
                    wf_idxs = np.arange(wf.shape[1])
                idxs = np.array([i for i in range(len(wf)) if wf[i] is not None])
                if len(idxs) != len(wf):
                    if verbose:
                        print("Using ", len(idxs), " waveforms for unit ", unit_id)
                    wf = np.array([wf[i] for i in idxs])
                waveforms.append(wf)
                spike_index_list.append(idxs)
                channel_list.append(wf_idxs)
        else:
            if verbose:
                print("Computing waveforms")
            waveforms, spike_index_list, channel_list = get_unit_waveforms(recording, sorting,
                                                                           max_spikes_per_unit=max_spikes_per_unit,
                                                                           ms_before=ms_before, ms_after=ms_after,
                                                                           grouping_property=grouping_property,
                                                                           dtype=dtype,
                                                                           compute_property_from_recording=
                                                                           compute_property_from_recording,
                                                                           max_channels_per_waveforms=
                                                                           max_channels_per_waveforms,
                                                                           save_as_features=save_waveforms_as_features,
                                                                           verbose=verbose, seed=seed, memmap=memmap,
                                                                           return_idxs=True)
    else:
        assert _spike_index_list is not None and _channel_list is not None, "Provide spike_index_list and " \
                                                                            "channel_list with waveforms"
        waveforms = _waveforms
        spike_index_list = _spike_index_list
        channel_list = _channel_list

    # compute len of all waveforms (computed for all units)
    n_waveforms = 0
    for wf in waveforms:
        n_spikes = len(wf)
        n_waveforms += n_spikes
    wf_shape = waveforms[0].shape

    dtype = recording.get_dtype()
    # prepare all waveforms
    if by_electrode:
        all_waveforms = sorting.allocate_array(name='all_waveforms.raw', dtype=dtype,
                                               shape=(n_waveforms * wf_shape[1], wf_shape[2]), memmap=memmap)
    else:
        all_waveforms = sorting.allocate_array(name='all_waveforms.raw', dtype=dtype,
                                               shape=(n_waveforms, wf_shape[1] * wf_shape[2]), memmap=memmap)

    # concatenate all waveforms
    if not isinstance(waveforms, list):
        # single unit
        waveforms = [waveforms]
        spike_index_list = [spike_index_list]

    i_start = 0
    for i_w, wf in enumerate(waveforms):
        if by_electrode:
            wf_reshaped = wf.reshape((wf.shape[0] * wf.shape[1], wf.shape[2]))
            nspikes.append(len(wf) * recording.get_num_channels())
        else:
            wf_reshaped = wf.reshape((wf.shape[0], wf.shape[1] * wf.shape[2]))
            nspikes.append(len(wf))
        all_waveforms[i_start:i_start + wf_reshaped.shape[0]] = wf_reshaped
        i_start += wf_reshaped.shape[0]

    pca = PCA(n_components=n_comp, whiten=whiten, random_state=seed)
    if len(all_waveforms) < max_spikes_for_pca:
        max_spikes_for_pca = n_waveforms
    max_spikes_for_pca = int(max_spikes_for_pca)
    if verbose:
        print("Fitting PCA of %d dimensions on %d waveforms" % (n_comp, max_spikes_for_pca))
    pca.fit(all_waveforms[np.random.RandomState(seed=seed).permutation(len(all_waveforms))[:max_spikes_for_pca]])

    if verbose:
        print("Projecting waveforms on PC")
    pca_scores_list = []
    # project waveforms on principal components
    for unit_id in unit_ids:
        idx_waveform = sorting.get_unit_ids().index(unit_id)
        wf = waveforms[idx_waveform]
        if by_electrode:
            pct = np.dot(wf, pca.components_.T)
        else:
            pct = np.dot(wf.reshape((wf.shape[0], wf.shape[1] * wf.shape[2])), pca.components_.T)
        if whiten:
            pct /= np.sqrt(pca.explained_variance_)
        pca_scores = sorting.allocate_array(array=pct, name='pcascores_' + str(unit_id) + '.raw', memmap=memmap)
        pca_scores_list.append(pca_scores)

    if save_as_features:
        for i, unit_id in enumerate(sorting.get_unit_ids()):
            if len(spike_index_list[i]) < len(sorting.get_unit_spike_train(unit_id)):
                assert spike_index_list[i] is not None, 'Indices are not computed for this unit'
                if 'pca_scores' not in sorting.get_unit_spike_feature_names(unit_id):
                    features = np.array([None] * len(sorting.get_unit_spike_train(unit_id)))
                else:
                    features = np.array(sorting.get_unit_spike_features(unit_id, 'pca_scores'))
                for idx, ind in enumerate(spike_index_list[i]):
                    features[ind] = pca_scores_list[i][idx]
            else:
                features = pca_scores_list[i]
            sorting.set_unit_spike_features(unit_id, 'pca_scores', features)

    if return_idxs:
        return pca_scores_list, spike_index_list, np.array(channel_list)
    else:
        return pca_scores_list


def set_unit_properties_by_max_channel_properties(recording, sorting, property, unit_ids=None, peak='both',
                                                  mode='median', ms_before=3., ms_after=3., dtype=None,
                                                  max_spikes_per_unit=300, verbose=False, seed=0):
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
    ms_before: float
        Time period in ms to cut waveforms before the spike events
    ms_after: float
        Time period in ms to cut waveforms after the spike events
    dtype: dtype
        The numpy dtype of the waveforms
    max_spikes_per_unit: int
        The maximum number of spikes to extract per unit.
    verbose: bool
        If True output is verbose
    seed: int
        Random seed for reproducibility
    '''
    if property not in recording.get_shared_channel_property_names():
        raise Exception("'property' should be in recording properties")

    if isinstance(unit_ids, (int, np.integer)):
        unit_ids = [unit_ids]
    elif unit_ids is None:
        unit_ids = sorting.get_unit_ids()
    elif not isinstance(unit_ids, (list, np.ndarray)):
        raise Exception("unit_ids is not a valid in valid")

    if 'max_channel' in sorting.get_shared_unit_property_names():
        if verbose:
            print("Using 'template' property")
        max_chan_property = True
    else:
        if verbose:
            print("Computing templates")
        max_chan_property = False

    for i, unit_id in enumerate(unit_ids):
        if unit_id not in sorting.get_unit_ids():
            raise Exception("unit_ids is not in valid")
        if property not in sorting.get_unit_property_names(unit_id):
            if max_chan_property:
                max_chan = sorting.get_unit_property(unit_id, 'max_channel')
            else:
                max_chan = get_unit_max_channels(recording, sorting, unit_id, mode=mode, peak=peak,
                                                 max_spikes_per_unit=max_spikes_per_unit, dtype=dtype,
                                                 ms_before=ms_before, ms_after=ms_after, verbose=verbose,
                                                 seed=seed)[0]
            sorting.set_unit_property(unit_id, property, recording.get_channel_property(max_chan, property))


def export_to_phy(recording, sorting, output_folder, n_comp=3, electrode_dimensions=None,
                  grouping_property=None, ms_before=1., ms_after=2., dtype=None, amp_method='absolute', amp_peak='both',
                  amp_frames_before=3, amp_frames_after=3, max_spikes_for_pca=10000, max_channels_per_template=16,
                  recompute_info=True, save_features_props=False, verbose=False, memmap=True, seed=0):
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
    n_comp: int
        n_compFeatures in template-gui format
    electrode_dimensions: list
        If electrode locations are 3D, it indicates the 2D dimensions to use as channel location
    grouping_property: str
        Property to group channels. E.g. if the recording extractor has the 'group' property and 'grouping_property' is
        'group', then waveforms are computed group-wise.
    ms_before: float
        Time period in ms to cut waveforms before the spike events
    ms_after: float
        Time period in ms to cut waveforms after the spike events
    dtype: dtype
        The numpy dtype of the waveforms
    amp_method: str
        If 'absolute' (default), amplitudes are absolute amplitudes in uV are returned.
        If 'relative', amplitudes are returned as ratios between waveform amplitudes and template amplitudes.
    amp_peak: str
        If maximum channel has to be found among negative peaks ('neg'), positive ('pos') or both ('both' - default)
    amp_frames_before: int
        Frames before peak to compute amplitude
    amp_frames_after: int
        Frames after peak to compute amplitude
    max_spikes_for_pca: int
        The maximum number of waveforms to use to compute PCA (default is 10'000)
    max_channels_per_template: int
        The number of channels per template for visualization (default=16)
    recompute_info: bool
        If True, will always re-extract waveforms and templates.
    save_features_props: bool
        If True, will store all calculated features and properties
    verbose: bool
        If True output is verbose
    memmap: bool
        If True, all phy data are memmapped to tmp files (set to True for large recordings)
    seed: int
        Random seed for extracting waveforms and pcs
    '''
    if not isinstance(recording, se.RecordingExtractor) or not isinstance(sorting, se.SortingExtractor):
        raise AttributeError()

    max_spikes_per_unit = np.inf

    empty_flag = False
    for unit_id in sorting.get_unit_ids():
        spikes = sorting.get_unit_spike_train(unit_id)
        if spikes.shape[0] == 0:
            empty_flag = True
    if empty_flag:
        print('Warning: empty units have been removed when being exported to Phy')
        sorting = st.curation.threshold_num_spikes(sorting, 1)

    if len(sorting.get_unit_ids()) == 0:
        raise Exception("No non-empty units in the sorting result, can't save to phy.")

    output_folder = Path(output_folder).absolute()
    if output_folder.is_dir():
        shutil.rmtree(output_folder)
    output_folder.mkdir()

    # save dat file
    if dtype is None:
        dtype = recording.get_dtype()

    recording.write_to_binary_dat_format(output_folder / 'recording.dat', dtype=dtype)

    # write params.py
    with (output_folder / 'params.py').open('w') as f:
        f.write("dat_path =" + "r'" + str(output_folder / 'recording.dat') + "'" + '\n')
        f.write('n_channels_dat = ' + str(recording.get_num_channels()) + '\n')
        f.write("dtype = '" + str(dtype) + "'\n")
        f.write('offset = 0\n')
        f.write('sample_rate = ' + str(recording.get_sampling_frequency()) + '\n')
        f.write('hp_filtered = False')

    spike_times, spike_clusters, amplitudes, channel_map, pc_features, pc_feature_ind, \
    spike_templates, templates, templates_ind, similar_templates, channel_map_si, channel_groups, \
    positions = _get_phy_data(recording, sorting, n_comp, electrode_dimensions, grouping_property, ms_before,
                              ms_after, dtype, amp_method, amp_peak, amp_frames_before, amp_frames_after,
                              max_spikes_per_unit, max_spikes_for_pca, max_channels_per_template,
                              recompute_info, save_features_props,
                              verbose, memmap, seed)

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

    np.save(str(output_folder / 'amplitudes.npy'), amplitudes)
    np.save(str(output_folder / 'spike_times.npy'), spike_times)
    np.save(str(output_folder / 'spike_templates.npy'), spike_templates)
    np.save(str(output_folder / 'spike_clusters.npy'), spike_clusters)
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


def _compute_templates_similarity(templates):
    similarity = np.zeros((len(templates), len(templates)))
    for i, t_i in enumerate(templates):
        for j, t_j in enumerate(templates):
            t_i_lin = t_i.reshape(t_i.shape[0] * t_i.shape[1])
            t_j_lin = t_j.reshape(t_j.shape[0] * t_j.shape[1])
            a = np.corrcoef(t_i_lin, t_j_lin)
            similarity[i, j] = np.abs(a[0, 1])
    return similarity


def _get_random_spike_waveforms(recording, sorting, unit, max_spikes_per_unit, snippet_len, channel_ids=None, seed=0):
    st = sorting.get_unit_spike_train(unit_id=unit)
    num_events = len(st)
    if num_events > max_spikes_per_unit:
        event_indices = np.random.RandomState(seed=seed).choice(range(num_events), size=max_spikes_per_unit,
                                                                replace=False)
    else:
        event_indices = range(num_events)

    spikes = recording.get_snippets(reference_frames=st[event_indices].astype('int64'),
                                    snippet_len=snippet_len, channel_ids=channel_ids)
    return spikes, event_indices


def _get_spike_times_clusters(sorting, memmap=True):
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


def _get_amp_metric_data(recording, sorting, amp_method, amp_peak,
                         amp_frames_before, amp_frames_after, max_spikes_per_unit, recompute_info,
                         save_features_props, seed, memmap=True):
    if recompute_info:
        sorting.clear_units_spike_features(feature_name='amplitudes')

    # amplitudes.npy
    amplitudes_list, amp_idxs = get_unit_amplitudes(recording, sorting, method=amp_method,
                                                    save_as_features=save_features_props, peak=amp_peak,
                                                    max_spikes_per_unit=max_spikes_per_unit,
                                                    frames_before=amp_frames_before, frames_after=amp_frames_after,
                                                    seed=seed, memmap=memmap, return_idxs=True)

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


def _get_pca_metric_data(recording, sorting, n_comp, ms_before, ms_after, dtype, max_spikes_per_unit,
                         max_spikes_for_pca, recompute_info, save_features_props, verbose, seed, memmap=True):
    if recompute_info:
        sorting.clear_units_spike_features(feature_name='waveforms')

    if memmap:
        if sorting.get_tmp_folder() is None:
            tmp_folder = Path(tempfile.mkdtemp())
            sorting.set_tmp_folder(tmp_folder)
        else:
            tmp_folder = sorting.get_tmp_folder()

    pc_list, pca_idxs, pc_ind = compute_unit_pca_scores(recording, sorting, n_comp=n_comp, by_electrode=True,
                                                        max_spikes_per_unit=max_spikes_per_unit, ms_before=ms_before,
                                                        ms_after=ms_after, dtype=dtype,
                                                        save_as_features=save_features_props,
                                                        max_spikes_for_pca=max_spikes_for_pca, verbose=verbose,
                                                        seed=seed,
                                                        memmap=memmap, return_idxs=True)

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
                             amp_frames_before, amp_frames_after, max_spikes_per_unit, max_spikes_for_pca,
                             recompute_info, max_channels_per_waveforms, save_features_props, verbose, seed, memmap):
    if recompute_info:
        sorting.clear_units_spike_features(feature_name='waveforms')
        sorting.clear_units_spike_features(feature_name='amplitudes')

    if memmap:
        if sorting.get_tmp_folder() is None:
            tmp_folder = Path(tempfile.mkdtemp())
            sorting.set_tmp_folder(tmp_folder)
        else:
            tmp_folder = sorting.get_tmp_folder()

    if 'waveforms' not in sorting.get_shared_unit_spike_feature_names():
        waveforms, spike_index_list, channel_list = get_unit_waveforms(recording, sorting,
                                                                       max_spikes_per_unit=max_spikes_per_unit,
                                                                       ms_before=ms_before,
                                                                       ms_after=ms_after, dtype=dtype,
                                                                       save_as_features=save_features_props,
                                                                       verbose=verbose,
                                                                       seed=seed,
                                                                       memmap=memmap, return_idxs=True,
                                                                       max_channels_per_waveforms=
                                                                       max_channels_per_waveforms)
    else:
        waveforms, spike_index_list, channel_list = None, None, None

    # pca scores
    pc_list, pca_idxs, pc_ind = compute_unit_pca_scores(recording, sorting, n_comp=n_comp, by_electrode=True,
                                                        max_spikes_per_unit=max_spikes_per_unit, ms_before=ms_before,
                                                        ms_after=ms_after, dtype=dtype,
                                                        save_as_features=save_features_props,
                                                        max_spikes_for_pca=max_spikes_for_pca, verbose=verbose,
                                                        seed=seed,
                                                        memmap=memmap, return_idxs=True,
                                                        max_channels_per_waveforms=max_channels_per_waveforms,
                                                        _waveforms=waveforms, _spike_index_list=spike_index_list,
                                                        _channel_list=channel_list)
    # amplitudes
    amplitudes_list, amp_idxs = get_unit_amplitudes(recording, sorting, method=amp_method,
                                                    save_as_features=save_features_props, peak=amp_peak,
                                                    max_spikes_per_unit=max_spikes_per_unit,
                                                    frames_before=amp_frames_before, frames_after=amp_frames_after,
                                                    seed=seed, memmap=memmap, return_idxs=True)

    # templates
    templates = get_unit_templates(recording, sorting,
                                   save_as_property=save_features_props, seed=seed, _waveforms=waveforms)

    # compute len of all waveforms (computed for all units)
    n_spikes = 0
    n_pca_amps = 0  # n_pca and n_amps are he same (max_spikes_per_unit)
    for i, (unit_id, amp) in enumerate(zip(sorting.get_unit_ids(), amplitudes_list)):
        n_spikes += len(sorting.get_unit_spike_train(unit_id))
        n_pca_amps += len(amp)
    pc_shape = pc_list[0].shape

    spike_times = sorting.allocate_array(shape=(n_spikes, 1), dtype=np.uint32, name='spike_times.raw',
                                         memmap=memmap)
    spike_clusters = sorting.allocate_array(shape=(n_spikes, 1), dtype=np.uint32, name='spike_clusters.raw',
                                            memmap=memmap)
    spike_times_pca = sorting.allocate_array(shape=(n_pca_amps, 1), dtype=np.uint32, name='spike_times_pca.raw',
                                             memmap=memmap)
    spike_clusters_pca = sorting.allocate_array(shape=(n_pca_amps, 1), dtype=np.uint32, name='spike_clusters_pca.raw',
                                                memmap=memmap)
    pc_features = sorting.allocate_array(shape=(n_pca_amps, pc_shape[2], pc_shape[1]), dtype=np.float32,
                                         name='pc_features.raw', memmap=memmap)
    spike_times_amps = sorting.allocate_array(shape=(n_pca_amps, 1), dtype=np.uint32, name='spike_times_amps.raw',
                                              memmap=memmap)
    spike_clusters_amps = sorting.allocate_array(shape=(n_pca_amps, 1), dtype=np.uint32, name='spike_clusters_amps.raw',
                                                 memmap=memmap)
    amplitudes = sorting.allocate_array(shape=(n_pca_amps, 1), dtype=np.float32, name='amplitudes.raw', memmap=memmap)

    i_start_st = 0
    i_start_pc_amp = 0
    for i_u, id in enumerate(sorting.get_unit_ids()):
        st = sorting.get_unit_spike_train(id)
        cl = [i_u] * len(st)
        pc = pc_list[i_u]
        amp = amplitudes_list[i_u]

        # take care of amps and pca computed on subset of spikes
        if len(pc) < len(st):
            cl_pca = [i_u] * len(pc)
            st_pca = st[pca_idxs[i_u]]
        else:
            cl_pca = [i_u] * len(st)
            st_pca = st

        if len(amp) < len(st):
            cl_amp = [i_u] * len(amp)
            st_amp = st[amp_idxs[i_u]]
        else:
            cl_amp = [i_u] * len(st)
            st_amp = st

        # assign
        spike_times[i_start_st:i_start_st + len(st)] = st[:, np.newaxis]
        spike_clusters[i_start_st:i_start_st + len(st)] = np.array(cl)[:, np.newaxis]
        spike_times_pca[i_start_pc_amp:i_start_pc_amp + len(st_pca)] = st_pca[:, np.newaxis]
        spike_clusters_pca[i_start_pc_amp:i_start_pc_amp + len(st_pca)] = np.array(cl_pca)[:, np.newaxis]
        spike_times_amps[i_start_pc_amp:i_start_pc_amp + len(st_amp)] = st_amp[:, np.newaxis]
        spike_clusters_amps[i_start_pc_amp:i_start_pc_amp + len(st_amp)] = np.array(cl_amp)[:, np.newaxis]
        amplitudes[i_start_pc_amp:i_start_pc_amp + len(st_amp)] = amp[:, np.newaxis]
        pc_features[i_start_pc_amp:i_start_pc_amp + len(st_amp)] = pc.swapaxes(1, 2)
        i_start_st += len(st)
        i_start_pc_amp += len(st_amp)

    sorting_idxs = np.argsort(spike_times[:, 0])
    sorting_idxs_amps = np.argsort(spike_times_amps[:, 0])
    sorting_idxs_pca = np.argsort(spike_times_pca[:, 0])

    spike_times[:] = spike_times[sorting_idxs]
    spike_times_amps[:] = spike_times_amps[sorting_idxs_amps]
    spike_times_pca[:] = spike_times_pca[sorting_idxs_pca]

    spike_clusters[:] = spike_clusters[sorting_idxs]
    spike_clusters_amps[:] = spike_clusters_amps[sorting_idxs_amps]
    spike_clusters_pca[:] = spike_clusters_pca[sorting_idxs_pca]

    amplitudes[:] = amplitudes[sorting_idxs_amps]
    pc_features[:] = pc_features[sorting_idxs_pca]
    pc_feature_ind = pc_ind  # np.tile(np.arange(recording.get_num_channels()), (len(sorting.get_unit_ids()), 1))

    return spike_times, spike_times_amps, spike_times_pca, spike_clusters, spike_clusters_amps, spike_clusters_pca, \
           amplitudes, pc_features, pc_feature_ind, templates


def _get_phy_data(recording, sorting, n_comp, electrode_dimensions, grouping_property,
                  ms_before, ms_after, dtype, amp_method, amp_peak, amp_frames_before,
                  amp_frames_after, max_spikes_per_unit, max_spikes_for_pca, max_channels_per_template,
                  recompute_info, save_features_props, verbose, memmap, seed):
    if not isinstance(recording, se.RecordingExtractor) or not isinstance(sorting, se.SortingExtractor):
        raise AttributeError()
    if len(sorting.get_unit_ids()) == 0:
        raise Exception("No units in the sorting result, can't compute phy information.")

    if max_channels_per_template is None:
        max_channels_per_template = recording.get_num_channels()

    if recompute_info:
        sorting.clear_units_property(property_name='template')
        sorting.clear_units_spike_features(feature_name='waveforms')
        sorting.clear_units_spike_features(feature_name='amplitudes')

    # pc_features.npy - [nSpikes, nFeaturesPerChannel, nPCFeatures] single
    if grouping_property in recording.get_shared_channel_property_names():
        groups, num_chans_in_group = np.unique([recording.get_channel_property(ch, grouping_property)
                                                for ch in recording.get_channel_ids()], return_counts=True)
        max_num_chans_in_group = np.max(num_chans_in_group)
        channel_groups = np.array([recording.get_channel_property(ch, grouping_property)
                                   for ch in recording.get_channel_ids()])
    else:
        max_num_chans_in_group = recording.get_num_channels()
        channel_groups = np.array([0] * recording.get_num_channels())

    spike_times, spike_times_amps, spike_times_pca, spike_clusters, spike_clusters_amps, spike_clusters_pca, \
    amplitudes, pc_features, pc_feature_ind, templates \
        = _get_quality_metric_data(recording, sorting, n_comp=n_comp, ms_before=ms_before, ms_after=ms_after,
                                   dtype=dtype, amp_method=amp_method, amp_peak=amp_peak,
                                   amp_frames_before=amp_frames_before,
                                   amp_frames_after=amp_frames_after, max_spikes_per_unit=max_spikes_per_unit,
                                   max_spikes_for_pca=max_spikes_for_pca,
                                   recompute_info=recompute_info, max_channels_per_waveforms=max_channels_per_template,
                                   save_features_props=save_features_props, verbose=verbose, memmap=memmap, seed=seed)

    channel_map = np.arange(recording.get_num_channels())
    channel_map_si = np.array(recording.get_channel_ids())

    # channel_positions.npy
    if 'location' in recording.get_shared_channel_property_names():
        positions = np.array([recording.get_channel_property(chan, 'location')
                              for chan in recording.get_channel_ids()])
        if electrode_dimensions is not None:
            positions = positions[:, electrode_dimensions]
    else:
        if verbose:
            print("'location' property is not available and it will be linear.")
        positions = np.zeros((recording.get_num_channels(), 2))
        positions[:, 1] = np.arange(recording.get_num_channels())

    # similar_templates.npy - [nTemplates, nTemplates] single
    # templates = get_unit_templates(recording, sorting, save_as_property=save_features_props, seed=seed)
    similar_templates = _compute_templates_similarity(templates)

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
            unit_chans = []
            for ch in recording.get_channel_ids():
                if recording.get_channel_property(ch, 'group') == group:
                    unit_chans.append(list(channel_map_si).index(ch))
            if len(unit_chans) == 0:
                raise Exception("Sorting extractor has different property than recording extractor. "
                                "They should correspond.")
            if len(unit_chans) != max_num_chans_in_group:
                # append closest channel
                if list(channel_map).index(int(np.max(unit_chans))) + 1 < np.max(channel_map):
                    unit_chans.append(list(channel_map).index(int(np.max(unit_chans)) + 1))
                else:
                    unit_chans.append(list(channel_map).index(int(np.min(unit_chans)) - 1))
            unit_chans = np.array(unit_chans)
            templates_ind[u_i] = unit_chans
            templates_red[u_i, :] = templates[u_i, :, unit_chans].T
        templates = templates_red
    elif max_channels_per_template < recording.get_num_channels():
        # waveforms, templates, and pc_scores are computed on the same channels
        templates_ind = pc_feature_ind
    else:
        templates_ind = np.tile(np.arange(recording.get_num_channels()), (len(sorting.get_unit_ids()), 1))

    # spike_templates.npy - [nSpikes, ] uint32
    spike_templates = spike_clusters

    return spike_times, spike_clusters, amplitudes, channel_map, pc_features, pc_feature_ind, \
           spike_templates, templates, templates_ind, similar_templates, channel_map_si, channel_groups, positions


def _select_max_channels(wf, recording, max_channels):
    template = np.mean(wf, axis=0)
    # select based on adjacency
    if max_channels < recording.get_num_channels():
        if 'location' in recording.get_shared_channel_property_names():
            max_channel_idx = np.unravel_index(np.argmax(np.abs(template)),
                                               template.shape)[0]
            locs = recording.get_channel_locations()
            loc_max = locs[recording.get_channel_ids().index(max_channel_idx)]
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


def _compute_one_waveform(unit, recording, sorting, channel_ids, unit_ids, grouping_property,
                          compute_property_from_recording, max_channels_per_waveforms, max_spikes_per_unit,
                          n_pad, dtype, memmap, seed, save_as_features, verbose, memmap_fname=None):
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
        if verbose:
            print("Waveforms by property: ", grouping_property)

        if not compute_sorting_group:
            rec_list, rec_props = recording.get_sub_extractors_by_property(grouping_property,
                                                                           return_property_list=True)
            sort_list, sort_props = sorting.get_sub_extractors_by_property(grouping_property,
                                                                           return_property_list=True)
            if len(rec_props) != len(sort_props):
                print('Different' + grouping_property + ' numbers: using largest number of ' + grouping_property)
                if len(rec_props) > len(sort_props):
                    for i_r, rec in enumerate(rec_props):
                        if rec not in sort_props:
                            print('Inserting None for property ', rec)
                            sort_list.insert(i_r, None)
                else:
                    for i_s, sort in enumerate(sort_props):
                        if sort not in rec_props:
                            rec_list.insert(i_s, None)
            else:
                assert len(rec_list) == len(sort_list)

            if max_channels_per_waveforms is None:
                max_channels_per_waveforms = rec_list[0].get_num_channels()

            for i_list, (rec, sort) in enumerate(zip(rec_list, sort_list)):
                if sort is not None and rec is not None:
                    for i, unit_id in enumerate(unit_ids):
                        if unit == unit_id:
                            if channel_ids is None:
                                channel_ids = rec.get_channel_ids()

                            if max_spikes_per_unit is None:
                                max_spikes = len(sort.get_unit_spike_train(unit_id))
                            else:
                                max_spikes = max_spikes_per_unit

                            if max_channels_per_waveforms is None:
                                max_channels_per_waveforms = len(rec.get_channel_ids())

                            if verbose:
                                print('Waveform ' + str(i + 1) + '/' + str(len(unit_ids)))
                            wf, indices = _get_random_spike_waveforms(recording=rec,
                                                                      sorting=sort,
                                                                      unit=unit_id,
                                                                      max_spikes_per_unit=max_spikes,
                                                                      snippet_len=n_pad,
                                                                      channel_ids=channel_ids,
                                                                      seed=seed)
                            wf = wf.astype(dtype)

                            if max_channels_per_waveforms < len(channel_ids):
                                max_channel_idxs = _select_max_channels(wf, rec, max_channels_per_waveforms)
                            else:
                                max_channel_idxs = np.arange(rec.get_num_channels())
                            wf = wf[:, max_channel_idxs]

                            if memmap_fname is None:
                                waveforms = sorting.allocate_array(array=wf, name='waveforms_' + str(unit_id) + '.raw',
                                                                   memmap=memmap)
                            else:
                                arr = np.memmap(memmap_fname, dtype, mode='r+', shape=wf.shape)
                                arr[:] = wf
                                waveforms = 0

                            # move out
                            if save_as_features:
                                if len(indices) < len(sort.get_unit_spike_train(unit_id)):
                                    features = np.array([None] * len(sorting.get_unit_spike_train(unit_id)))
                                    for i, ind in enumerate(indices):
                                        features[ind] = waveforms[i]
                                else:
                                    features = waveforms
                                sorting.set_unit_spike_features(unit_id, 'waveforms', features)
        else:
            for i, unit_id in enumerate(unit_ids):
                if unit == unit_id:
                    if channel_ids is None:
                        channel_ids = recording.get_channel_ids()

                    rec = se.SubRecordingExtractor(recording, channel_ids=channel_ids)
                    rec_groups = np.array(rec.get_channel_groups())
                    groups, count = np.unique(rec_groups, return_counts=True)

                    if max_channels_per_waveforms is None:
                        max_channels_per_waveforms = np.max(count)

                    if max_spikes_per_unit is None:
                        max_spikes = len(sorting.get_unit_spike_train(unit_id))
                    else:
                        max_spikes = max_spikes_per_unit

                    if verbose:
                        print('Waveform ' + str(i + 1) + '/' + str(len(unit_ids)))
                    wf, indices = _get_random_spike_waveforms(recording=recording,
                                                              sorting=sorting,
                                                              unit=unit_id,
                                                              max_spikes_per_unit=max_spikes,
                                                              snippet_len=n_pad,
                                                              channel_ids=channel_ids,
                                                              seed=seed)
                    wf = wf.astype(dtype)

                    mean_waveforms = np.squeeze(np.mean(wf, axis=0))
                    max_amp_elec = np.unravel_index(mean_waveforms.argmin(), mean_waveforms.shape)[0]
                    group = recording.get_channel_property(recording.get_channel_ids()[max_amp_elec], grouping_property)
                    elec_group = np.where(rec_groups == group)
                    wf = np.squeeze(wf[:, elec_group, :])

                    if max_channels_per_waveforms < len(elec_group[0]):
                        max_channel_idxs = _select_max_channels(wf, rec, max_channels_per_waveforms)
                    else:
                        max_channel_idxs = np.arange(len(elec_group[0]))
                    wf = wf[:, max_channel_idxs]

                    if memmap_fname is None:
                        waveforms = sorting.allocate_array(array=wf, name='waveforms_' + str(unit_id) + '.raw',
                                                           memmap=memmap)
                    else:
                        arr = np.memmap(memmap_fname, dtype, mode='r+', shape=wf.shape)
                        arr[:] = wf
                        waveforms = 0

                    if save_as_features:
                        if len(indices) < len(sorting.get_unit_spike_train(unit_id)):
                            features = np.array([None] * len(sorting.get_unit_spike_train(unit_id)))
                            for i, ind in enumerate(indices):
                                features[ind] = waveforms[i]
                        else:
                            features = waveforms
                        sorting.set_unit_spike_features(unit_id, 'waveforms', features)
    else:
        for i, unit_id in enumerate(unit_ids):
            if unit == unit_id:
                if channel_ids is None:
                    channel_ids = recording.get_channel_ids()

                if max_channels_per_waveforms is None:
                    max_channels_per_waveforms = len(channel_ids)

                if max_spikes_per_unit is None:
                    max_spikes = len(sorting.get_unit_spike_train(unit_id))
                else:
                    max_spikes = max_spikes_per_unit

                if verbose:
                    print('Waveform ' + str(i + 1) + '/' + str(len(unit_ids)))
                wf, indices = _get_random_spike_waveforms(recording=recording,
                                                          sorting=sorting,
                                                          unit=unit_id,
                                                          max_spikes_per_unit=max_spikes,
                                                          snippet_len=n_pad,
                                                          channel_ids=channel_ids,
                                                          seed=seed)
                wf = wf.astype(dtype)

                if max_channels_per_waveforms < len(channel_ids):
                    max_channel_idxs = _select_max_channels(wf, recording, max_channels_per_waveforms)
                else:
                    max_channel_idxs = np.arange(len(channel_ids))
                wf = wf[:, max_channel_idxs]

                if memmap_fname is None:
                    waveforms = sorting.allocate_array(array=wf, name='waveforms_' + str(unit_id) + '.raw',
                                                       memmap=memmap)
                else:
                    arr = np.memmap(sorting.get_tmp_folder() / memmap_fname, dtype, mode='r+', shape=wf.shape)
                    arr[:] = wf
                    waveforms = 0

    return waveforms, np.array(indices), np.array(channel_ids)

