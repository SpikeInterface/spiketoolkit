import spikeextractors as se
from .analyzer import Analyzer
from .filters import bandpass_filter, whiten
from os.path import join
import os
import numpy as np


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
        f.write("dat_path ="  + "'" + join(output_folder, 'recording.dat') +"'" + '\n')
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
    pc_feature_ind = np.tile(np.arange(nPC), (len(sorting.getUnitIds()),1))

    # similar_templates.npy - [nTemplates, nTemplates] single
    templates = analyzer.getUnitTemplate()
    similar_templates = _computeTemplatesSimilarity(templates)

    # templates.npy
    templates = np.array(templates).swapaxes(1,2)

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


def computeUnitSNR(*, recording, sorting, unit_ids=None, **kwargs):
    analyzer = Analyzer(recording, sorting)
    if unit_ids is None:
        unit_ids = sorting.getUnitIds()
    channel_noise_levels = _computeChannelNoiseLevels(recording=recording)
    if unit_ids is not None:
        templates = analyzer.getUnitTemplate(unit_ids=unit_ids, **kwargs)
    else:
        templates = analyzer.getUnitTemplate(**kwargs)
    ret = []
    for template in templates:
        snr = _computeTemplateSNR(template, channel_noise_levels)
        ret.append(snr)
    return ret


def _computeTemplatesSimilarity(templates):
    similarity = np.zeros((len(templates), len(templates)))
    for i, t_i in enumerate(templates):
        for j, t_j in enumerate(templates):
            t_i_lin = t_i.reshape(t_i.shape[0] * t_i.shape[1])
            t_j_lin = t_j.reshape(t_j.shape[0] * t_j.shape[1])
            a = np.corrcoef(t_i_lin, t_j_lin)
            similarity[i, j] = np.abs(a[0,1])
    return similarity

def _computeWhiteningAndInverse(recording):
    white_recording = whiten(recording)
    wh_mat = white_recording._whitening_matrix
    wh_mat_inv = np.linalg.inv(wh_mat)
    return wh_mat, wh_mat_inv

