import numpy as np
import spiketoolkit as st
import spikeextractors as se
from sklearn.decomposition import PCA
from pathlib import Path
import shutil
import csv

def get_unit_waveforms(recording, sorting, unit_ids=None, grouping_property=None, start_frame=None, end_frame=None,
                       ms_before=3., ms_after=3., dtype=None, max_num_waveforms=np.inf,
                       save_as_features=True, compute_property_from_recording=False, verbose=False):
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
    start_frame: int
        The start frame for computing waveforms
    end_frame: int
        The end frame for computing waveforms
    ms_before: float
        Time period in ms to cut waveforms before the spike events
    ms_after: float
        Time period in ms to cut waveforms after the spike events
    dtype: dtype
        The numpy dtype of the waveforms
    max_num_waveforms: int
        The maximum number of wavefomrs to extract (default is np.inf)
    save_as_features: bool
        If True (default), waveforms are saved as features of the sorting extractor object
    compute_property_from_recording: bool
        If True and 'grouping_property' is given, the property of each unit is assigned as the corresponding propery of
        the recording extractor channel on which the average waveform is the largest
    verbose: bool
        If True output is verbose

    Returns
    -------
    waveforms: list
        List of np.array (n_spikes, n_channels, n_timepoints) containing extracted waveforms for each unit

    '''
    if isinstance(unit_ids, (int, np.integer)):
        unit_ids = [unit_ids]
    elif unit_ids is None:
        unit_ids = sorting.get_unit_ids()
    elif not isinstance(unit_ids, (list, np.ndarray)):
        raise Exception("unit_ids is not a valid in valid")

    if dtype is None:
        dtype = np.float32

    waveform_list = []
    if grouping_property is not None:
        if grouping_property not in recording.get_channel_property_names():
            raise ValueError("'grouping_property' should be a property of recording extractors")
        if compute_property_from_recording:
            compute_sorting_group = True
        elif grouping_property not in sorting.get_unit_property_names():
            print(grouping_property, ' not in sorting extractor. Computing it from the recording extractor')
            compute_sorting_group = True
        else:
            compute_sorting_group = False
        print("Waveforms by property: ", grouping_property)

        if not compute_sorting_group:
            rec_list, rec_props = se.get_sub_extractors_by_property(recording, grouping_property,
                                                                    return_property_list=True)
            sort_list, sort_props = se.get_sub_extractors_by_property(sorting, grouping_property,
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

            for i_list, (rec, sort) in enumerate(zip(rec_list, sort_list)):
                for i, unit_id in enumerate(unit_ids):
                    # ts_ = time.time()
                    if sort is not None and rec is not None:
                        if unit_id in sort.get_unit_ids():
                            fs = rec.get_sampling_frequency()
                            n_pad = [int(ms_before * fs / 1000), int(ms_after * fs / 1000)]

                            if verbose:
                                print('Waveform ' + str(i + 1) + '/' + str(len(unit_ids)))
                            waveforms, indices = _get_random_spike_waveforms(recording=rec,
                                                                             sorting=sort,
                                                                             unit=unit_id,
                                                                             max_num=max_num_waveforms,
                                                                             snippet_len=n_pad)
                            waveforms = waveforms.swapaxes(0, 2)
                            waveforms = waveforms.swapaxes(1, 2)
                            waveforms = waveforms.astype(dtype)

                            if save_as_features:
                                if len(indices) < len(sort.get_unit_spike_train(unit_id)):
                                    if 'waveforms' not in sorting.get_unit_spike_feature_names(unit_id):
                                        features = np.array([None] * len(sorting.get_unit_spike_train(unit_id)))
                                    else:
                                        features = np.array(sorting.get_unit_spike_features(unit_id, 'waveforms'))
                                    for i, ind in enumerate(indices):
                                        features[ind] = waveforms[i]
                                else:
                                    features = waveforms
                                sorting.set_unit_spike_features(unit_id, 'waveforms', features)
                            waveform_list.append(waveforms)
        else:
            for i, unit_id in enumerate(unit_ids):
                # ts_ = time.time()
                if unit_id in sorting.get_unit_ids():
                    rec_groups = np.array([recording.get_channel_property(ch, grouping_property)
                                           for ch in recording.get_channel_ids()])
                    rec = recording
                    fs = rec.get_sampling_frequency()
                    n_pad = [int(ms_before * fs / 1000), int(ms_after * fs / 1000)]

                    if verbose:
                        print('Waveform ' + str(i + 1) + '/' + str(len(unit_ids)))
                    waveforms, indices = _get_random_spike_waveforms(recording=recording,
                                                                     sorting=sorting,
                                                                     unit=unit_id,
                                                                     max_num=max_num_waveforms,
                                                                     snippet_len=n_pad)
                    waveforms = waveforms.swapaxes(0, 2)
                    waveforms = waveforms.swapaxes(1, 2)
                    waveforms = waveforms.astype(dtype)
                    mean_waveforms = np.squeeze(np.mean(waveforms, axis=0))
                    max_amp_elec = np.unravel_index(mean_waveforms.argmin(), mean_waveforms.shape)[0]
                    group = recording.get_channel_property(recording.get_channel_ids()[max_amp_elec], grouping_property)
                    elec_group = np.where(rec_groups == group)
                    waveforms = np.squeeze(waveforms[:, elec_group, :])
                    if save_as_features:
                        if len(indices) < len(sorting.get_unit_spike_train(unit_id)):
                            if 'waveforms' not in sorting.get_unit_spike_feature_names(unit_id):
                                features = np.array([None] * len(sorting.get_unit_spike_train(unit_id)))
                            else:
                                features = np.array(sorting.get_unit_spike_features(unit_id, 'waveforms'))
                            for i, ind in enumerate(indices):
                                features[ind] = waveforms[i]
                        else:
                            features = waveforms
                        sorting.set_unit_spike_features(unit_id, 'waveforms', features)
                    waveform_list.append(waveforms)
            return waveform_list
    else:
        for i, unit_id in enumerate(unit_ids):
            # ts_ = time.time()
            if unit_id not in sorting.get_unit_ids():
                raise Exception("unit_ids is not in valid")

            fs = recording.get_sampling_frequency()
            n_pad = [int(ms_before * fs / 1000), int(ms_after * fs / 1000)]

            if verbose:
                print('Waveform ' + str(i + 1) + '/' + str(len(unit_ids)))
            waveforms, indices = _get_random_spike_waveforms(recording=recording,
                                                             sorting=sorting,
                                                             unit=unit_id,
                                                             max_num=max_num_waveforms,
                                                             snippet_len=n_pad)
            # print('extract wf: ', time.time() - ts_)
            waveforms = waveforms.swapaxes(0, 2)
            waveforms = waveforms.swapaxes(1, 2)
            waveforms = waveforms.astype(dtype)
            # print('swap wf: ', time.time() - ts_)
            if save_as_features:
                if len(indices) < len(sorting.get_unit_spike_train(unit_id)):
                    if 'waveforms' not in sorting.get_unit_spike_feature_names(unit_id):
                        features = np.array([None] * len(sorting.get_unit_spike_train(unit_id)))
                    else:
                        features = np.array(sorting.get_unit_spike_features(unit_id, 'waveforms'))
                    for i, ind in enumerate(indices):
                        features[ind] = waveforms[i]
                else:
                    features = waveforms
                sorting.set_unit_spike_features(unit_id, 'waveforms', features)
            # print('append feats: ', time.time() - ts_)
            waveform_list.append(waveforms)
            # print('append wf: ', time.time() - ts_)

        if len(waveform_list) == 1:
            return waveform_list[0]
        else:
            return waveform_list


def get_unit_template(recording, sorting, unit_ids=None, grouping_property=None, save_as_property=True,
                      start_frame=None, end_frame=None, ms_before=3., ms_after=3., dtype=None,
                      max_num_waveforms=np.inf, compute_property_from_recording=False,
                      verbose=False):
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
    grouping_property: str
        Property to group channels. E.g. if the recording extractor has the 'group' property and 'grouping_property' is
        'group', then waveforms are computed group-wise.
    save_as_features: bool
        If True (default), templates are saved as property of the sorting extractor object
    start_frame: int
        The start frame for computing waveforms
    end_frame: int
        The end frame for computing waveforms
    ms_before: float
        Time period in ms to cut waveforms before the spike events
    ms_after: float
        Time period in ms to cut waveforms after the spike events
    dtype: dtype
        The numpy dtype of the waveforms
    max_num_waveforms: int
        The maximum number of wavefomrs to extract (default is np.inf)
    compute_property_from_recording: bool
        If True and 'grouping_property' is given, the property of each unit is assigned as the corresponding propery of
        the recording extractor channel on which the average waveform is the largest
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

    if 'waveforms' in sorting.get_unit_spike_feature_names():
        if verbose:
            print("Using 'waveforms' features")
        waveforms_features = True
    else:
        if verbose:
            print("Computing waveforms")
        waveforms_features = False

    template_list = []
    for i, unit_id in enumerate(unit_ids):
        if unit_id not in sorting.get_unit_ids():
            raise Exception("unit_ids is not in valid")
        if waveforms_features:
            waveforms = sorting.get_unit_spike_features(unit_id, 'waveforms')
            idx_not_none = np.array([i for i in range(len(waveforms)) if waveforms[i] is not None])
            if len(idx_not_none) != len(waveforms):
                if verbose:
                    print("Using ", len(idx_not_none), " waveforms for unit ", unit_id)
                waveforms = np.array(waveforms[idx_not_none])
            template = np.mean(waveforms, axis=0)
        else:
            template = np.mean(get_unit_waveforms(recording, sorting, unit_id, start_frame=start_frame,
                                                end_frame=end_frame, max_num_waveforms=max_num_waveforms,
                                                ms_before=ms_before, ms_after=ms_after,
                                                grouping_property=grouping_property,
                                                compute_property_from_recording=compute_property_from_recording,
                                                verbose=verbose)
                               , axis=0)

        if save_as_property:
            sorting.set_unit_property(unit_id, 'template', template)

        template_list.append(template)
    if len(template_list) == 1:
        return template_list[0]
    else:
        return template_list


def get_unit_max_channel(recording, sorting, unit_ids=None, grouping_property=None,
                        save_as_property=True, start_frame=None, end_frame=None,
                        ms_before=3., ms_after=3., dtype=None, max_num_waveforms=np.inf,
                        compute_property_from_recording=False, verbose=False):
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
    grouping_property: str
        Property to group channels. E.g. if the recording extractor has the 'group' property and 'grouping_property' is
        'group', then waveforms are computed group-wise.
    save_as_property: bool
        If True (default), templates are saved as property of the sorting extractor object
    start_frame: int
        The start frame for computing waveforms
    end_frame: int
        The end frame for computing waveforms
    ms_before: float
        Time period in ms to cut waveforms before the spike events
    ms_after: float
        Time period in ms to cut waveforms after the spike events
    dtype: dtype
        The numpy dtype of the waveforms
    max_num_waveforms: int
        The maximum number of wavefomrs to extract (default is np.inf)
    compute_property_from_recording: bool
        If True and 'grouping_property' is given, the property of each unit is assigned as the corresponding propery of
        the recording extractor channel on which the average waveform is the largest
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

    if 'template' in sorting.get_unit_property_names():
        if verbose:
            print("Using 'template' property")
        template_property = True
    else:
        if verbose:
            print("Computing templates")
        template_property = False

    max_list = []
    for i, unit_id in enumerate(unit_ids):
        if unit_id not in sorting.get_unit_ids():
            raise Exception("unit_ids is not in valid")
        if template_property:
            template = sorting.get_unit_property(unit_id, 'template')
        else:
            template = get_unit_template(recording, sorting, unit_id, start_frame=start_frame,
                                       end_frame=end_frame, max_num_waveforms=max_num_waveforms,
                                       ms_before=ms_before, ms_after=ms_after,  grouping_property=grouping_property,
                                       compute_property_from_recording=compute_property_from_recording,
                                       verbose=verbose)
        max_channel = np.unravel_index(np.argmax(np.abs(template)),
                                       template.shape)[0]
        if save_as_property:
            sorting.set_unit_property(unit_id, 'max_channel', max_channel)

        max_list.append(max_channel)

    if len(max_list) == 1:
        return max_list[0]
    else:
        return max_list


def compute_pca_scores(recording, sorting, unit_ids=None, n_comp=3, by_electrode=False, grouping_property=None,
                       start_frame=None, end_frame=None, ms_before=3., ms_after=3., dtype=None,
                       max_num_waveforms=np.inf, save_as_features=True, compute_property_from_recording=False,
                       verbose=False):
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
    start_frame: int
        The start frame for computing waveforms
    end_frame: int
        The end frame for computing waveforms
    ms_before: float
        Time period in ms to cut waveforms before the spike events
    ms_after: float
        Time period in ms to cut waveforms after the spike events
    dtype: dtype
        The numpy dtype of the waveforms
    max_num_waveforms: int
        The maximum number of wavefomrs to extract (default is np.inf)
    compute_property_from_recording: bool
        If True and 'grouping_property' is given, the property of each unit is assigned as the corresponding propery of
        the recording extractor channel on which the average waveform is the largest
    verbose: bool
        If True output is verbose

    Returns
    -------
    pcs_scores: list
        List of np.array containing extracted pca scores.
        If 'by_electrode' is False, the array has shape (n_spikes, n_comp)
        If 'by_electrode' is True, the array has shape (n_spikes, n_channels, n_comp)
    '''
    if isinstance(unit_ids, (int, np.integer)):
        unit_ids = [unit_ids]
    elif unit_ids is None:
        unit_ids = sorting.get_unit_ids()
    elif not isinstance(unit_ids, (list, np.ndarray)):
        raise Exception("unit_ids is not a valid in valid")

    # concatenate all waveforms
    all_waveforms = np.array([])
    nspikes = []
    idx_not_none = None
    if 'waveforms' in sorting.get_unit_spike_feature_names():
        if verbose:
            print("Using 'waveforms' features")
        waveforms = []
        for unit_id in unit_ids:
            wf = sorting.get_unit_spike_features(unit_id, 'waveforms')
            idx_not_none = np.array([i for i in range(len(wf)) if wf[i] is not None])
            if len(idx_not_none) != len(wf):
                if verbose:
                    print("Using ", len(idx_not_none), " waveforms for unit ", unit_id)
                wf = np.array([wf[i] for i in idx_not_none])
            waveforms.append(wf)
    else:
        if verbose:
            print("Computing waveforms")
        waveforms = get_unit_waveforms(recording, sorting)

    if not isinstance(waveforms, list):
        # single unit
        waveforms = [waveforms]

    for i_w, wf in enumerate(waveforms):
        if wf is None:
            wf = get_unit_waveforms(recording, sorting, unit_id, start_frame=start_frame,
                                    end_frame=end_frame, max_num_waveforms=max_num_waveforms,
                                    ms_before=ms_before, ms_after=ms_after,
                                    grouping_property=grouping_property,
                                    compute_property_from_recording=compute_property_from_recording,
                                    verbose=verbose)
        if by_electrode:
            wf_reshaped = wf.reshape((wf.shape[0] * wf.shape[1], wf.shape[2]))
            nspikes.append(len(wf) * recording.get_num_channels())
        else:
            wf_reshaped = wf.reshape((wf.shape[0], wf.shape[1] * wf.shape[2]))
            nspikes.append(len(wf))
        if i_w == 0:
            all_waveforms = wf_reshaped
        else:
            all_waveforms = np.concatenate((all_waveforms, wf_reshaped))
    if verbose:
        print("Fitting PCA of %d dimensions on %d waveforms" % (n_comp, len(all_waveforms)))

    pca = PCA(n_components=n_comp, whiten=True)
    pca.fit_transform(all_waveforms)
    scores = pca.transform(all_waveforms)

    init = 0
    pca_scores = []
    for i_n, nsp in enumerate(nspikes):
        pcascores = scores[init: init + nsp, :]
        init = nsp
        if by_electrode:
            pca_scores.append(pcascores.reshape(nsp // recording.get_num_channels(),
                                                recording.get_num_channels(), n_comp))
        else:
            pca_scores.append(pcascores)

    if save_as_features:
        for i, unit_id in enumerate(sorting.get_unit_ids()):
            if len(pca_scores[i]) < len(sorting.get_unit_spike_train(unit_id)):
                assert idx_not_none is not None
                if 'pca_scores' not in sorting.get_unit_spike_feature_names(unit_id):
                    features = np.array([None] * len(sorting.get_unit_spike_train(unit_id)))
                else:
                    features = np.array(sorting.get_unit_spike_features(unit_id, 'pca_scores'))
                for i, ind in enumerate(idx_not_none):
                    features[ind] = pca_scores[i]
            else:
                features = pca_scores[i]
            sorting.set_unit_spike_features(unit_id, 'pca_scores', features)
    return pca_scores


def export_to_phy(recording, sorting, output_folder, nPCchan=3, nPC=5, electrode_dimensions=None,
                  grouping_property=None, start_frame=None, end_frame=None, ms_before=3., ms_after=3., dtype=None,
                  max_num_waveforms=np.inf, compute_property_from_recording=False, save_waveforms=False,
                  verbose=False):
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
    nPCchan: int
        nFeaturesPerChannel in template-gui format
    nPC: int
        nPCFeatures in template-gui format
    electrode_dimensions: list
        If electrode locations are 3D, it indicates the 2D dimensions to use as channel location
    grouping_property: str
        Property to group channels. E.g. if the recording extractor has the 'group' property and 'grouping_property' is
        'group', then waveforms are computed group-wise.
    start_frame: int
        The start frame for computing waveforms
    end_frame: int
        The end frame for computing waveforms
    ms_before: float
        Time period in ms to cut waveforms before the spike events
    ms_after: float
        Time period in ms to cut waveforms after the spike events
    dtype: dtype
        The numpy dtype of the waveforms
    max_num_waveforms: int
        The maximum number of wavefomrs to extract (default is np.inf)
    compute_property_from_recording: bool
        If True and 'grouping_property' is given, the property of each unit is assigned as the corresponding propery of
        the recording extractor channel on which the average waveform is the largest
    save_waveforms: bool
        If True, waveforms are saved as waveforms.npy
    verbose: bool
        If True output is verbose
    '''
    if not isinstance(recording, se.RecordingExtractor) or not isinstance(sorting, se.SortingExtractor):
        raise AttributeError()
    output_folder = Path(output_folder).absolute()
    if output_folder.is_dir():
        shutil.rmtree(output_folder)
    output_folder.mkdir()

    # save dat file
    se.write_binary_dat_format(recording, output_folder / 'recording.dat', dtype='int16')

    # write params.py
    with (output_folder / 'params.py').open('w') as f:
        f.write("dat_path =" + "'" + str(output_folder / 'recording.dat') + "'" + '\n')
        f.write('n_channels_dat = ' + str(recording.get_num_channels()) + '\n')
        f.write("dtype = 'int16'\n")
        f.write('offset = 0\n')
        f.write('sample_rate = ' + str(recording.get_sampling_frequency()) + '\n')
        f.write('hp_filtered = False')

    # pc_features.npy - [nSpikes, nFeaturesPerChannel, nPCFeatures] single
    if 'group' in recording.get_channel_property_names() and 'group' in sorting.get_unit_property_names():
        groups, num_chans_in_group = np.unique([recording.get_channel_property(ch, 'group')
                                                    for ch in recording.get_channel_ids()], return_counts=True)
        max_num_chans_in_group = np.max(num_chans_in_group)
        channel_groups = np.array([recording.get_channel_property(ch, 'group') for ch in recording.get_channel_ids()])
    else:
        max_num_chans_in_group = recording.get_num_channels()
        channel_groups = np.array([0] * recording.get_num_channels())
    if nPC > max_num_chans_in_group:
        nPC = max_num_chans_in_group
        if verbose:
            print("Changed number of PC to number of channels: ", nPC)
    waveforms = get_unit_waveforms(recording, sorting,
                                   start_frame=start_frame, end_frame=end_frame, max_num_waveforms=max_num_waveforms,
                                   ms_before=ms_before, ms_after=ms_after,
                                   grouping_property=grouping_property,
                                   compute_property_from_recording=compute_property_from_recording,
                                   verbose=verbose)
    pc_scores = compute_pca_scores(recording, sorting, n_comp=nPC, by_electrode=True,
                                   start_frame=start_frame, end_frame=end_frame, max_num_waveforms=max_num_waveforms,
                                   ms_before=ms_before, ms_after=ms_after,
                                   grouping_property=grouping_property,
                                   compute_property_from_recording=compute_property_from_recording,
                                   verbose=verbose)

    # spike times.npy and spike clusters.npy
    spike_times = np.array([])
    spike_clusters = np.array([])
    pc_features = np.array([])

    for i_u, id in enumerate(sorting.get_unit_ids()):
        st = sorting.get_unit_spike_train(id)
        cl = [i_u] * len(sorting.get_unit_spike_train(id))
        pc = pc_scores[i_u]
        spike_times = np.concatenate((spike_times, np.array(st)))
        spike_clusters = np.concatenate((spike_clusters, np.array(cl)))
        if i_u == 0:
            pc_features = np.array(pc)
        else:
            pc_features = np.vstack((pc_features, np.array(pc)))

    sorting_idxs = np.argsort(spike_times)
    spike_times = spike_times[sorting_idxs, np.newaxis]
    spike_clusters = spike_clusters[sorting_idxs, np.newaxis]
    pc_features = pc_features[sorting_idxs, :nPCchan, :]

    # amplitudes.npy
    amplitudes = np.ones((len(spike_times), 1), dtype='int16')

    # channel_map.npy
    channel_map = np.arange(recording.get_num_channels())
    channel_map_si = np.array(recording.get_channel_ids())

    # channel_positions.npy
    if 'location' in recording.get_channel_property_names():
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
    templates = get_unit_template(recording, sorting)

    if not isinstance(templates, list):
        if len(templates.shape) == 2:
            # single unit
            templates = templates.reshape(1, templates.shape[0], templates.shape[1])

    similar_templates = _compute_templates_similarity(templates)

    # templates.npy
    templates = np.array(templates, dtype='float32').swapaxes(1, 2)

    if 'group' in recording.get_channel_property_names() and 'group' in sorting.get_unit_property_names():
        pc_feature_ind = np.zeros((len(sorting.get_unit_ids()), int(max_num_chans_in_group)), dtype=int)
        templates_ind = np.zeros((len(sorting.get_unit_ids()), int(max_num_chans_in_group)), dtype=int)
        templates_red = np.zeros((templates.shape[0], templates.shape[1], int(max_num_chans_in_group)))

        for u_i, u in enumerate(sorting.get_unit_ids()):
            group = sorting.get_unit_property(u, 'group')
            unit_chans = []
            for ch in recording.get_channel_ids():
                if recording.get_channel_property(ch, 'group') == group:
                    unit_chans.append(list(channel_map_si).index(ch))
            if len(unit_chans) != max_num_chans_in_group:
                # append closest channel
                if list(channel_map).index(int(np.max(unit_chans))) + 1 < np.max(channel_map):
                    unit_chans.append(list(channel_map).index(int(np.max(unit_chans)) + 1))
                else:
                    unit_chans.append(list(channel_map).index(int(np.min(unit_chans)) - 1))
            unit_chans = np.array(unit_chans)
            pc_feature_ind[u_i] = unit_chans
            templates_ind[u_i] = unit_chans
            templates_red[u_i, :] = templates[u_i, :, unit_chans].T
        templates = templates_red
    else:
        # pc_feature_ind.npy - [nTemplates, nPCFeatures] uint32
        pc_feature_ind = np.tile(np.arange(nPC), (len(sorting.get_unit_ids()), 1))
        # template_ind.npy
        templates_ind = np.tile(np.arange(recording.get_num_channels()), (len(sorting.get_unit_ids()), 1))

    # print(pc_feature_ind.shape, templates_ind.shape)
    # print(pc_feature_ind[:, 0])
    # spike_templates.npy - [nSpikes, ] uint32
    spike_templates = spike_clusters
    # Save .tsv metadata
    max_amplitudes = [np.min(t) for t in templates]
    second_max_channel = []
    for t in templates:
        second_max_channel.append(np.argsort(np.abs(np.min(t, axis=0)))[::-1][1])

    with (output_folder / 'cluster_amps.tsv').open('w') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
        writer.writerow(['cluster_id', 'max_amp'])
        for i, (u, amp) in enumerate(zip(sorting.get_unit_ids(), max_amplitudes)):
            writer.writerow([i, amp])
    with (output_folder / 'cluster_second_max_chans.tsv').open('w') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
        writer.writerow(['cluster_id', 'sec_channel'])
        for i, (u, ch) in enumerate(zip(sorting.get_unit_ids(), second_max_channel)):
            writer.writerow([i, ch])
    with (output_folder / 'cluster_group.tsv').open('w') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
        writer.writerow(['cluster_id', 'group'])
        for i, u in enumerate(sorting.get_unit_ids()):
            writer.writerow([i, 'unsorted'])
    if 'group' in sorting.get_unit_property_names():
        with (output_folder / 'cluster_channel_group.tsv').open('w') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
            writer.writerow(['cluster_id', 'ch_group'])
            for i, u in enumerate(sorting.get_unit_ids()):
                writer.writerow([i, sorting.get_unit_property(u, 'group')])
    else:
        with (output_folder / 'cluster_channel_group.tsv').open('w') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
            writer.writerow(['cluster_id', 'ch_group'])
            for i, u in enumerate(sorting.get_unit_ids()):
                writer.writerow([i, 0])

    np.save(str(output_folder / 'amplitudes.npy'), amplitudes)
    np.save(str(output_folder / 'spike_times.npy'), spike_times.astype(int))
    np.save(str(output_folder / 'spike_templates.npy'), spike_templates.astype(int))
    np.save(str(output_folder / 'pc_features.npy'), pc_features)
    np.save(str(output_folder / 'pc_feature_ind.npy'), pc_feature_ind.astype(int))
    np.save(str(output_folder / 'templates.npy'), templates, )
    np.save(str(output_folder / 'template_ind.npy'), templates_ind.astype(int))
    np.save(str(output_folder / 'similar_templates.npy'), similar_templates)
    np.save(str(output_folder / 'channel_map.npy'), channel_map.astype(int))
    np.save(str(output_folder / 'channel_map_si.npy'), channel_map_si.astype(int))
    np.save(str(output_folder / 'channel_positions.npy'), positions)
    np.save(str(output_folder / 'channel_groups.npy'), channel_groups)
    # np.save(str(output_folder / 'whitening_mat.npy'), whitening_mat)
    # np.save(str(output_folder / 'whitening_mat_inv.npy'), whitening_mat_inv)

    if save_waveforms:
        np.save(str(output_folder / 'waveforms.npy'), np.array(waveforms))

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


def _compute_whitening_and_inverse(recording):
    white_recording = st.preprocessing.whiten(recording)
    wh_mat = white_recording._whitening_matrix
    wh_mat_inv = np.linalg.inv(wh_mat)
    return wh_mat, wh_mat_inv


def _get_random_spike_waveforms(recording, sorting, unit, max_num, snippet_len, channels=None):
    st = sorting.get_unit_spike_train(unit_id=unit)
    num_events = len(st)
    if num_events > max_num:
        event_indices = np.random.choice(range(num_events), size=max_num, replace=False)
    else:
        event_indices = range(num_events)

    spikes = recording.get_snippets(reference_frames=st[event_indices].astype(int),
                                   snippet_len=snippet_len, channel_ids=channels)
    spikes = np.dstack(tuple(spikes))
    return spikes, event_indices
