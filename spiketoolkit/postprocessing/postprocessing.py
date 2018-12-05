import numpy as np
import spiketoolkit as st
import spikeextractors as se
from sklearn.decomposition import PCA
import time


def getUnitWaveforms(recording, sorting, unit_ids=None, by_property=None, start_frame=None, end_frame=None,
                     ms_before=3., ms_after=3., max_num_waveforms=np.inf, filter=False, bandpass=[300, 6000],
                     save_as_features=True, verbose=False):
    '''This function returns the spike waveforms from the specified unit_ids from t_start and t_stop
    in the form of a numpy array of spike waveforms.

    Parameters
    ----------
    unit_ids: (int or list)
        The unit id or list of unit ids to extract waveforms from
    start_frame: (int)
        The starting frame to extract waveforms
    end_frame: (int)
        The ending frame to extract waveforms
    ms_before: float
        Time in ms to cut out waveform before the peak
    ms_after: float
        Time in ms to cut out waveform after the peak

    Returns
    -------
    waveforms: np.array
        A list of 3D arrays that contain all waveforms between start and end_frame
        Dimensions of each element are: (numm_spikes x num_channels x num_spike_frames)

    '''
    if isinstance(unit_ids, (int, np.integer)):
        unit_ids = [unit_ids]
    elif unit_ids is None:
        unit_ids = sorting.getUnitIds()
    elif not isinstance(unit_ids, (list, np.ndarray)):
        raise Exception("unit_ids is not a valid in valid")

    waveform_list = []
    if by_property is not None:
        if not by_property in sorting.getUnitPropertyNames() \
                or by_property not in recording.getChannelPropertyNames():
            raise ValueError("'by_property' should be aproperty of recording and sorting extractors")

        else:
            print("Waveforms by property: ", by_property)
            rec_list = se.getSubExtractorsByProperty(recording, by_property)
            sort_list = se.getSubExtractorsByProperty(sorting, by_property)
            assert len(rec_list) == len(sort_list)

            # TODO make this in parallel
            for i_list, (rec, sort) in enumerate(zip(rec_list, sort_list)):
                for i, unit_id in enumerate(unit_ids):
                    # ts_ = time.time()
                    if unit_id in sort.getUnitIds():
                        if not filter:
                            recordings = rec.getTraces(start_frame, end_frame)
                        else:
                            recordings = st.filters.bandpass_filter(recording=rec, freq_min=bandpass[0],
                                                                    freq_max=bandpass[1]).getTraces(start_frame, end_frame)
                        # print('check filter: ', time.time() - ts_)

                        fs = rec.getSamplingFrequency()
                        n_pad = [int(ms_before * fs / 1000), int(ms_after * fs / 1000)]

                        if verbose:
                            print('Waveform ' + str(i + 1) + '/' + str(len(unit_ids)))
                        ts = time.time()
                        waveforms, indices = _get_random_spike_waveforms(recording=rec,
                                                                         sorting=sort,
                                                                         unit=unit_id,
                                                                         max_num=max_num_waveforms,
                                                                         snippet_len=n_pad)
                        # print('extract wf: ', time.time() - ts_)
                        waveforms = waveforms.swapaxes(0, 2)
                        waveforms = waveforms.swapaxes(1, 2)
                        # print('swap wf: ', time.time() - ts_)
                        if save_as_features:
                            if len(indices) < len(sort.getUnitSpikeTrain(unit_id)):
                                if 'waveforms' not in sorting.getUnitSpikeFeatureNames(unit_id):
                                    features = np.array([None] * len(sorting.getUnitSpikeTrain(unit_id)))
                                else:
                                    features = np.array(sorting.getUnitSpikeFeatures(unit_id, 'waveforms'))
                                for i, ind in enumerate(indices):
                                    features[ind] = waveforms[i]
                            else:
                                features = waveforms
                            sorting.setUnitSpikeFeatures(unit_id, 'waveforms', features)
                        # print('append feats: ', time.time() - ts_)
                        waveform_list.append(waveforms)
                        # print('append wf: ', time.time() - ts_)
    else:
        # TODO make this in parallel
        for i, unit_id in enumerate(unit_ids):
            # ts_ = time.time()
            if unit_id not in sorting.getUnitIds():
                raise Exception("unit_ids is not in valid")

            if not filter:
                recordings = recording.getTraces(start_frame, end_frame)
            else:
                recordings = st.filters.bandpass_filter(recording=recording, freq_min=bandpass[0],
                                                        freq_max=bandpass[1]).getTraces(start_frame, end_frame)
            # print('check filter: ', time.time() - ts_)

            fs = recording.getSamplingFrequency()
            n_pad = [int(ms_before * fs / 1000), int(ms_after * fs / 1000)]

            if verbose:
                print('Waveform ' + str(i + 1) + '/' + str(len(unit_ids)))
            ts = time.time()
            waveforms, indices = _get_random_spike_waveforms(recording=recording,
                                                             sorting=sorting,
                                                             unit=unit_id,
                                                             max_num=max_num_waveforms,
                                                             snippet_len=n_pad)
            # print('extract wf: ', time.time() - ts_)
            waveforms = waveforms.swapaxes(0, 2)
            waveforms = waveforms.swapaxes(1, 2)
            # print('swap wf: ', time.time() - ts_)
            if save_as_features:
                if len(indices) < len(sorting.getUnitSpikeTrain(unit_id)):
                    if 'waveforms' not in sorting.getUnitSpikeFeatureNames(unit_id):
                        features = np.array([None] * len(sorting.getUnitSpikeTrain(unit_id)))
                    else:
                        features = np.array(sorting.getUnitSpikeFeatures(unit_id, 'waveforms'))
                    for i, ind in enumerate(indices):
                        features[ind] = waveforms[i]
                else:
                    features = waveforms
                sorting.setUnitSpikeFeatures(unit_id, 'waveforms', features)
            # print('append feats: ', time.time() - ts_)
            waveform_list.append(waveforms)
            # print('append wf: ', time.time() - ts_)

        if len(waveform_list) == 1:
            return waveform_list[0]
        else:
            return waveform_list


def getUnitTemplate(recording, sorting, unit_ids=None, by_property=None,
                    save_as_property=True, start_frame=None, end_frame=None,
                    ms_before=3., ms_after=3., max_num_waveforms=np.inf, filter=False, bandpass=[300, 6000]):
    '''

    Parameters
    ----------
    unit_ids
    start_frame
    end_frame

    Returns
    -------

    '''
    if isinstance(unit_ids, (int, np.integer)):
        unit_ids = [unit_ids]
    elif unit_ids is None:
        unit_ids = sorting.getUnitIds()
    elif not isinstance(unit_ids, (list, np.ndarray)):
        raise Exception("unit_ids is not a valid in valid")

    if 'waveforms' in sorting.getUnitSpikeFeatureNames():
        print("Using 'waveforms' features")
        waveforms_features = True
    else:
        print("Computing waveforms")
        waveforms_features = False

    template_list = []
    for i, unit_id in enumerate(unit_ids):
        if unit_id not in sorting.getUnitIds():
            raise Exception("unit_ids is not in valid")
        if waveforms_features:
            waveforms = sorting.getUnitSpikeFeatures(unit_id, 'waveforms')
            idx_not_none = np.array([i for i in range(len(waveforms)) if waveforms[i] is not None])
            if len(idx_not_none) != len(waveforms):
                print("Using ", len(idx_not_none), " waveforms for unit ", unit_id)
                waveforms = np.array(waveforms[idx_not_none])
            template = np.mean(waveforms, axis=0)
        else:
            template = np.mean(getUnitWaveforms(recording, sorting, unit_id, start_frame, end_frame, max_num_waveforms,
                                                ms_before, ms_after, filter, bandpass), axis=0)

        if save_as_property:
            sorting.setUnitProperty(unit_id, 'template', template)

        template_list.append(template)
    if len(template_list) == 1:
        return template_list[0]
    else:
        return template_list


def getUnitMaxChannel(recording, sorting, unit_ids=None, by_property=None,
                      save_as_property=True, start_frame=None, end_frame=None,
                      ms_before=3., ms_after=3., max_num_waveforms=np.inf, filter=False, bandpass=[300, 6000]):
    '''

    Parameters
    ----------
    unit_ids

    Returns
    -------

    '''
    if isinstance(unit_ids, (int, np.integer)):
        unit_ids = [unit_ids]
    elif unit_ids is None:
        unit_ids = sorting.getUnitIds()
    elif not isinstance(unit_ids, (list, np.ndarray)):
        raise Exception("unit_ids is not a valid in valid")

    if 'template' in sorting.getUnitPropertyNames():
        print("Using 'template' property")
        template_property = True
    else:
        print("Computing templates")
        template_property = False

    max_list = []
    for i, unit_id in enumerate(unit_ids):
        if unit_id not in sorting.getUnitIds():
            raise Exception("unit_ids is not in valid")
        if template_property:
            template = sorting.getUnitProperty(unit_id, 'template')
        else:
            template = getUnitTemplate(recording, sorting, unit_id, start_frame, end_frame, max_num_waveforms,
                                       ms_before, ms_after, filter, bandpass)
        max_channel = np.unravel_index(np.argmax(np.abs(template)),
                                       template.shape)[0]
        if save_as_property:
            sorting.setUnitProperty(unit_id, 'max_channel', max_channel)

        max_list.append(max_channel)

    if len(max_list) == 1:
        return max_list[0]
    else:
        return max_list


def computePCAScores(recording, sorting, n_comp=3, by_property=None,
                     elec=False, max_num_waveforms=np.inf, save_as_features=True):
    '''

    Parameters
    ----------
    n_comp

    Returns
    -------

    '''
    # concatenate all waveforms
    all_waveforms = np.array([])
    nspikes = []
    if 'waveforms' in sorting.getUnitSpikeFeatureNames():
        print("Using 'waveforms' features")
        waveforms = []
        for unit_id in sorting.getUnitIds():
            wf = sorting.getUnitSpikeFeatures(unit_id, 'waveforms')
            idx_not_none = np.where(wf != None)[0]
            if len(idx_not_none) > 0:
                waveforms.append(wf[idx_not_none])
    else:
        print("Copmputing waveforms")
        waveforms = getUnitWaveforms(recording, sorting)

    for i_w, wf in enumerate(waveforms):
        if wf is None:
            wf = getUnitWaveforms(sorting.getUnitIds()[i_w], verbose=True)
        if elec:
            wf_reshaped = wf.reshape((wf.shape[0] * wf.shape[1], wf.shape[2]))
            nspikes.append(len(wf) * recording.getNumChannels())
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
        if elec:
            pca_scores.append(pcascores.reshape(nsp // recording.getNumChannels(),
                                                recording.getNumChannels(), n_comp))
        else:
            pca_scores.append(pcascores)

    for i, unit_id in enumerate(sorting.getUnitIds()):
        if save_as_features:
            sorting.setUnitSpikeFeatures(unit_id, 'pca_scores', pca_scores[i])

    return np.array(pca_scores)


def computeUnitSNR(recording, sorting, unit_ids=None, save_as_property=True):
    if unit_ids is None:
        unit_ids = sorting.getUnitIds()
    channel_noise_levels = _computeChannelNoiseLevels(recording=recording)
    if unit_ids is not None:
        templates = getUnitTemplate(recording, sorting, unit_ids=unit_ids)
    else:
        templates = getUnitTemplate(recording, sorting)
    snr_list = []
    for i, unit_id in enumerate(sorting.getUnitIds()):
        snr = _computeTemplateSNR(templates[i], channel_noise_levels)
        if save_as_property:
            sorting.setUnitProperty(unit_id, 'snr', snr)

        snr_list.append(snr)
    return snr_list


def exportToPhy(recording, sorting, output_folder, nPCchan=3, nPC=5, filter=False, electrode_dimensions=None,
                max_num_waveforms=np.inf):
    analyzer = Analyzer(recording, sorting)

    if not isinstance(recording, se.RecordingExtractor) or not isinstance(sorting, se.SortingExtractor):
        raise AttributeError()
    output_folder = os.path.abspath(output_folder)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    if filter:
        recording = bandpass_filter(recording, freq_min=300, freq_max=6000)

    # save dat file
    se.writeBinaryDatFormat(recording, join(output_folder, 'recording.dat'), dtype='int16')

    # write params.py
    with open(join(output_folder, 'params.py'), 'w') as f:
        f.write("dat_path =" + "'" + join(output_folder, 'recording.dat') + "'" + '\n')
        f.write('n_channels_dat = ' + str(recording.getNumChannels()) + '\n')
        f.write("dtype = 'int16'\n")
        f.write('offset = 0\n')
        f.write('sample_rate = ' + str(recording.getSamplingFrequency()) + '\n')
        f.write('hp_filtered = False')

    # pc_features.npy - [nSpikes, nFeaturesPerChannel, nPCFeatures] single
    if nPC > recording.getNumChannels():
        nPC = recording.getNumChannels()
        print("Changed number of PC to number of channels: ", nPC)
    pc_scores = analyzer.computePCAscores(n_comp=nPC, elec=True, max_num_waveforms=max_num_waveforms)

    # spike times.npy and spike clusters.npy
    spike_times = np.array([])
    spike_clusters = np.array([])
    pc_features = np.array([])
    for i_u, id in enumerate(sorting.getUnitIds()):
        st = sorting.getUnitSpikeTrain(id)
        cl = [i_u] * len(sorting.getUnitSpikeTrain(id))
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
    amplitudes = np.ones((len(spike_times), 1))

    # channel_map.npy
    channel_map = np.array(recording.getChannelIds())

    # channel_positions.npy
    if 'location' in recording.getChannelPropertyNames():
        positions = np.array([recording.getChannelProperty(chan, 'location')
                              for chan in range(recording.getNumChannels())])
        if electrode_dimensions is not None:
            positions = positions[:, electrode_dimensions]
    else:
        print("'location' property is not available and it will be linear.")
        positions = np.zeros((recording.getNumChannels(), 2))
        positions[:, 1] = np.arange(recording.getNumChannels())

    # pc_feature_ind.npy - [nTemplates, nPCFeatures] uint32
    pc_feature_ind = np.tile(np.arange(nPC), (len(sorting.getUnitIds()), 1))

    # similar_templates.npy - [nTemplates, nTemplates] single
    templates = analyzer.getUnitTemplate()
    similar_templates = _computeTemplatesSimilarity(templates)

    # templates.npy
    templates = np.array(templates).swapaxes(1, 2)

    # template_ind.npy
    templates_ind = np.tile(np.arange(recording.getNumChannels()), (len(sorting.getUnitIds()), 1))

    # spike_templates.npy - [nSpikes, ] uint32
    spike_templates = spike_clusters

    # whitening_mat.npy - [nChannels, nChannels] double
    # whitening_mat_inv.npy - [nChannels, nChannels] double
    whitening_mat, whitening_mat_inv = _computeWhiteningAndInverse(recording)

    np.save(join(output_folder, 'amplitudes.npy'), amplitudes)
    np.save(join(output_folder, 'spike_times.npy'), spike_times.astype(int))
    # np.save(join(output_folder, 'spike_clusters.npy'), spike_clusters.astype(int))
    np.save(join(output_folder, 'spike_templates.npy'), spike_templates.astype(int))
    np.save(join(output_folder, 'pc_features.npy'), pc_features)
    np.save(join(output_folder, 'pc_feature_ind.npy'), pc_feature_ind.astype(int))
    np.save(join(output_folder, 'templates.npy'), templates)
    np.save(join(output_folder, 'templates_ind.npy'), templates_ind.astype(int))
    np.save(join(output_folder, 'similar_templates.npy'), similar_templates)
    np.save(join(output_folder, 'channel_map.npy'), channel_map.astype(int))
    np.save(join(output_folder, 'channel_positions.npy'), positions)
    np.save(join(output_folder, 'whitening_mat.npy'), whitening_mat)
    np.save(join(output_folder, 'whitening_mat_inv.npy'), whitening_mat_inv)
    print('Saved phy format to: ', output_folder)
    print('Run:\n\nphy template-gui ', join(output_folder, 'params.py'))


def _computeTemplateSNR(template, channel_noise_levels):
    channel_snrs = []
    for ch in range(template.shape[0]):
        channel_snrs.append((np.max(template[ch, :]) - np.min(template[ch, :])) / channel_noise_levels[ch])
    return np.max(channel_snrs)

def _computeChannelNoiseLevels(recording):
    M = recording.getNumChannels()
    X = recording.getTraces(start_frame=0, end_frame=np.minimum(1000, recording.getNumFrames()))
    ret = []
    for ch in range(M):
        noise_level = np.std(X[ch, :])
        ret.append(noise_level)
    return ret

def _computeTemplatesSimilarity(templates):
    similarity = np.zeros((len(templates), len(templates)))
    for i, t_i in enumerate(templates):
        for j, t_j in enumerate(templates):
            t_i_lin = t_i.reshape(t_i.shape[0] * t_i.shape[1])
            t_j_lin = t_j.reshape(t_j.shape[0] * t_j.shape[1])
            a = np.corrcoef(t_i_lin, t_j_lin)
            similarity[i, j] = np.abs(a[0, 1])
    return similarity

def _computeWhiteningAndInverse(recording):
    white_recording = whiten(recording)
    wh_mat = white_recording._whitening_matrix
    wh_mat_inv = np.linalg.inv(wh_mat)
    return wh_mat, wh_mat_inv


def _get_random_spike_waveforms(recording, sorting, unit, max_num, snippet_len, channels=None):
    st = sorting.getUnitSpikeTrain(unit_id=unit)
    num_events = len(st)
    if num_events > max_num:
        event_indices = np.random.choice(range(num_events), size=max_num, replace=False)
    else:
        event_indices = range(num_events)

    spikes = recording.getSnippets(reference_frames=st[event_indices].astype(int),
                                   snippet_len=snippet_len, channel_ids=channels)
    spikes = np.dstack(tuple(spikes))
    return spikes, event_indices

