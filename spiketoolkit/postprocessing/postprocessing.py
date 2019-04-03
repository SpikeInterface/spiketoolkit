import numpy as np
import spiketoolkit as st
import spikeextractors as se
from sklearn.decomposition import PCA
from pathlib import Path


def get_unit_waveforms(recording, sorting, unit_ids=None, grouping_property=None, start_frame=None, end_frame=None,
                     ms_before=3., ms_after=3., dtype=None, max_num_waveforms=np.inf, filter=False,
                     bandpass=[300, 6000], save_as_features=True, verbose=False, compute_property_from_recording=False):
    '''
    This function returns the spike waveforms from the specified unit_ids from t_start and t_stop
    in the form of a numpy array of spike waveforms.

    Parameters
    ----------
    recording
    sorting
    unit_ids
    grouping_property
    start_frame
    end_frame
    ms_before
    ms_after
    dtype
    max_num_waveforms
    filter
    bandpass
    save_as_features
    verbose
    compute_property_from_recording

    Returns
    -------
    waveforms

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
            rec_list, rec_props = se.get_sub_extractors_by_property(recording, grouping_property, return_property_list=True)
            sort_list, sort_props = se.get_sub_extractors_by_property(sorting, grouping_property, return_property_list=True)
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
                            if not filter:
                                rec = rec
                            else:
                                rec = st.preprocessing.bandpass_filter(recording=rec, freq_min=bandpass[0],
                                                                       freq_max=bandpass[1])

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
                    if not filter:
                        rec = recording
                    else:
                        rec = st.preprocessing.bandpass_filter(recording=recording, freq_min=bandpass[0],
                                                               freq_max=bandpass[1])

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

            if filter:
                recording = st.preprocessing.bandpass_filter(recording=recording, freq_min=bandpass[0],
                                                             freq_max=bandpass[1]).get_traces(start_frame, end_frame)

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


def get_unit_template(recording, sorting, unit_ids=None, grouping_property=None,
                    save_as_property=True, start_frame=None, end_frame=None,
                    ms_before=3., ms_after=3., max_num_waveforms=np.inf, filter=False,
                    bandpass=[300, 6000], compute_property_from_recording=False,
                    verbose=False):
    '''

    Parameters
    ----------
    recording
    sorting
    unit_ids
    grouping_property
    save_as_property
    start_frame
    end_frame
    ms_before
    ms_after
    max_num_waveforms
    filter
    bandpass
    compute_property_from_recording

    Returns
    -------

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
                                                ms_before=ms_before, ms_after=ms_after, filter=filter,
                                                bandpass=bandpass, grouping_property=grouping_property,
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
                      ms_before=3., ms_after=3., max_num_waveforms=np.inf, filter=False, bandpass=[300, 6000],
                      compute_property_from_recording=False, verbose=False):
    '''

    Parameters
    ----------
    recording
    sorting
    unit_ids
    grouping_property
    save_as_property
    start_frame
    end_frame
    ms_before
    ms_after
    max_num_waveforms
    filter
    bandpass
    compute_property_from_recording

    Returns
    -------

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
                                       ms_before=ms_before, ms_after=ms_after, filter=filter,
                                       bandpass=bandpass, grouping_property=grouping_property,
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


def compute_pca_scores(recording, sorting, n_comp=3, grouping_property=None, compute_property_from_recording=False,
                     by_electrode=False, max_num_waveforms=np.inf, save_as_features=True, verbose=False):
    '''

    Parameters
    ----------
    recording
    sorting
    n_comp
    grouping_property
    by_electrode
    max_num_waveforms
    save_as_features

    Returns
    -------

    '''
    # concatenate all waveforms
    all_waveforms = np.array([])
    nspikes = []
    idx_not_none = None
    if 'waveforms' in sorting.get_unit_spike_feature_names():
        if verbose:
            print("Using 'waveforms' features")
        waveforms = []
        for unit_id in sorting.get_unit_ids():
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

    for i_w, wf in enumerate(waveforms):
        if wf is None:
            wf = get_unit_waveforms(recording, sorting, [sorting.get_unit_ids()[i_w]],
                                  max_num_waveforms=max_num_waveforms,
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
    print("Fitting PCA of %d dimensions on %d waveforms" % (n_comp, len(all_waveforms)))

    pca = PCA(n_components=n_comp, whiten=True)
    pca.fit_transform(all_waveforms)
    scores = pca.transform(all_waveforms)

    init = 0
    pca_scores = []
    for i_n, nsp in enumerate(nspikes):
        pcascores = scores[init: init + nsp, :]
        init = nsp + 1
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


def export_to_phy(recording, sorting, output_folder, nPCchan=3, nPC=5, filter=False, electrode_dimensions=None,
                max_num_waveforms=np.inf):
    import spiketoolkit as st
    if not isinstance(recording, se.RecordingExtractor) or not isinstance(sorting, se.SortingExtractor):
        raise AttributeError()
    output_folder = Path(output_folder).absolute()
    if not output_folder.is_dir():
        output_folder.mkdir()

    if filter:
        recording = st.preprocessing.bandpass_filter(recording, freq_min=300, freq_max=6000)

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
    if nPC > recording.get_num_channels():
        nPC = recording.get_num_channels()
        print("Changed number of PC to number of channels: ", nPC)
    pc_scores = compute_pca_scores(recording, sorting, n_comp=nPC, by_electrode=True, max_num_waveforms=max_num_waveforms)

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
    channel_map = np.array(recording.get_channel_ids())

    # channel_positions.npy
    if 'location' in recording.get_channel_property_names():
        positions = np.array([recording.get_channel_property(chan, 'location')
                              for chan in range(recording.get_num_channels())])
        if electrode_dimensions is not None:
            positions = positions[:, electrode_dimensions]
    else:
        print("'location' property is not available and it will be linear.")
        positions = np.zeros((recording.get_num_channels(), 2))
        positions[:, 1] = np.arange(recording.get_num_channels())

    # pc_feature_ind.npy - [nTemplates, nPCFeatures] uint32
    pc_feature_ind = np.tile(np.arange(nPC), (len(sorting.get_unit_ids()), 1))

    # similar_templates.npy - [nTemplates, nTemplates] single
    templates = get_unit_template(recording, sorting)
    similar_templates = _compute_templates_similarity(templates)

    # templates.npy
    templates = np.array(templates, dtype='float32').swapaxes(1, 2)

    # template_ind.npy
    templates_ind = np.tile(np.arange(recording.get_num_channels()), (len(sorting.get_unit_ids()), 1))

    # spike_templates.npy - [nSpikes, ] uint32
    spike_templates = spike_clusters

    # whitening_mat.npy - [nChannels, nChannels] double
    # whitening_mat_inv.npy - [nChannels, nChannels] double
    # whitening_mat, whitening_mat_inv = _compute_whitening_and_inverse(recording)
    whitening_mat = np.eye(recording.get_num_channels())
    whitening_mat_inv = whitening_mat

    np.save(str(output_folder / 'amplitudes.npy'), amplitudes)
    np.save(str(output_folder / 'spike_times.npy'), spike_times.astype(int))
    # np.save(str(output_folder / 'spike_clusters.npy'), spike_clusters.astype(int))
    np.save(str(output_folder / 'spike_templates.npy'), spike_templates.astype(int))
    np.save(str(output_folder / 'pc_features.npy'), pc_features)
    np.save(str(output_folder / 'pc_feature_ind.npy'), pc_feature_ind.astype(int))
    np.save(str(output_folder / 'templates.npy'), templates, )
    # np.save(str(output_folder / 'templates_ind.npy'), templates_ind.astype(int))
    np.save(str(output_folder / 'similar_templates.npy'), similar_templates)
    np.save(str(output_folder / 'channel_map.npy'), channel_map.astype(int))
    np.save(str(output_folder / 'channel_positions.npy'), positions)
    np.save(str(output_folder / 'whitening_mat.npy'), whitening_mat)
    np.save(str(output_folder / 'whitening_mat_inv.npy'), whitening_mat_inv)
    print('Saved phy format to: ', output_folder)
    print('Run:\n\nphy template-gui ', str(output_folder / 'params.py'))


def _compute_template_SNR(template, channel_noise_levels):
    channel_snrs = []
    for ch in range(template.shape[0]):
        channel_snrs.append((np.max(template[ch, :]) - np.min(template[ch, :])) / channel_noise_levels[ch])
    return np.max(channel_snrs)

def _compute_channel_noise_levels(recording):
    M = recording.get_num_channels()
    X = recording.get_traces(start_frame=0, end_frame=np.minimum(1000, recording.get_num_frames()))
    ret = []
    for ch in range(M):
        noise_level = np.std(X[ch, :])
        ret.append(noise_level)
    return ret

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
