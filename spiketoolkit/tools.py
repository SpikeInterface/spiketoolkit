from spikeinterface import RecordingExtractor, SortingExtractor
import spikeinterface as si
from .analyzer import Analyzer
from .filters import bandpass_filter, whiten
from os.path import join
import os
import numpy as np


def exportToPhy(recording, sorting, output_folder, nPC=5, filter=False):

    analyzer = Analyzer(recording, sorting)

    if not isinstance(recording, RecordingExtractor) or not isinstance(sorting, SortingExtractor):
        raise AttributeError()
    output_folder = os.path.abspath(output_folder)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    if filter:
        recording = bandpass_filter(recording, freq_min=300, freq_max=6000)

    # save dat file
    si.writeBinaryDatFormat(recording, join(output_folder, 'recording.dat'))

    # write params.py
    with open(join(output_folder, 'params.py'), 'w') as f:
        f.write('dat_path = ' + join(output_folder, 'recording.dat') + '\n')
        f.write('n_channels_dat = ' + str(recording.getNumChannels()) + '\n')
        f.write('dtype = float32\n')
        f.write('offset = 0\n')
        f.write('sample_rate = ' + str(recording.getSamplingFrequency()) + '\n')
        f.write('hp_filtered = False')

    # pc_features.npy - [nSpikes, nFeaturesPerChannel, nPCFeatures] single
    pc_features = analyzer.computePCAscores(n_comp=nPC, elec=True)

    # spike times.npy and spike clusters.npy
    spike_times = np.array([])
    spike_clusters = np.array([])
    spike_pc_features = np.array([])
    for i_u, id in enumerate(sorting.getUnitIds()):
        st = sorting.getUnitSpikeTrain(id)
        cl = [id] * len(sorting.getUnitSpikeTrain(id))
        pc = pc_features[i_u]
        spike_times = np.concatenate((spike_times, np.array(st)))
        spike_clusters = np.concatenate((spike_clusters, np.array(cl)))
        spike_pc_features = np.concatenate((spike_pc_features, np.array(pc)))
    sorting_idxs = np.argsort(spike_times)
    spike_times = spike_times[sorting_idxs]
    spike_clusters = spike_clusters[sorting_idxs]
    spike_pc_features = spike_pc_features[sorting_idxs]

    # amplitudes.npy
    amplitudes = np.ones(len(spike_times))

    # channel_map.npy
    channel_map = recording.getChannelIds()

    # channel_positions.npy
    if 'location' in recording.getChannelPropertyNames():
        positions = np.array([recording.getChannelProperty(chan, 'location')
                              for chan in range(recording.getNumChannels())])
    else:
        print("'location' property is not available and it will be linear.")
        positions = np.zeros((recording.getNumChannels(), 2))
        positions[:, 1] = np.arange(recording.getNumChannels())


    # pc_feature_ind.npy - [nTemplates, nPCFeatures] uint32

    # similar_templates.npy - [nTemplates, nTemplates] single

    # templates
    templates = analyzer.getUnitTemplate()

    # spike_templates.npy - [nSpikes, ] uint32
    spike_templates = spike_clusters

    # whitening_mat.npy - [nChannels, nChannels] double
    # whitening_mat_inv.npy - [nChannels, nChannels] double
    whitening_mat, whitening_mat_inv = _computeWhiteningAndInverse(recording)

    np.save(join(output_folder, 'spike_times.npy'), spike_times.astype(int))
    np.save(join(output_folder, 'spike_clusters.npy'), spike_clusters.astype(int))

def _computeTemplatesSimilarity(templates):
    similarity = np.zeros((len(templates), len(templates)))
    for i, t_i in enumerate(templates):
        for j, t_j in enumerate(templates):
            if i != j:
                a = np.corrcoef(t_i, t_j)
                raise Exception()

def _computeWhiteningAndInverse(recording):
    white_recording = whiten(recording)
    wh_mat = white_recording._whitening_matrix
    wh_mat_inv = np.linalg.inv(wh_mat)
    return wh_mat, wh_mat_inv

