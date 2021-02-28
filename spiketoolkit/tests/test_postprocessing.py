import numpy as np
import pytest
from spiketoolkit.tests.utils import create_signal_with_known_waveforms, create_dumpable_extractors_from_existing
import spikeextractors as se
from spiketoolkit.postprocessing import get_unit_waveforms, get_unit_templates, get_unit_amplitudes, \
    get_unit_max_channels, set_unit_properties_by_max_channel_properties, compute_unit_pca_scores, export_to_phy, \
    compute_unit_template_features, compute_channel_spiking_activity, compute_unit_centers_of_mass
from spiketoolkit.preprocessing import remove_bad_channels
import pandas
import os
import shutil
from pathlib import Path
import sys

if sys.platform == "win32":
    memmaps = [False]
else:
    memmaps = [False, True]


@pytest.mark.implemented
def test_waveforms():
    n_wf_samples = 100
    n_jobs = [0, 2]
    for n in n_jobs:
        for m in memmaps:
            print('N jobs', n, 'memmap', m)
            folder = 'test'
            if os.path.isdir(folder):
                shutil.rmtree(folder)
            rec, sort, waveforms, templates, max_chans, amps = create_signal_with_known_waveforms(n_waveforms=2,
                                                                                                  n_channels=4,
                                                                                                  n_wf_samples=
                                                                                                  n_wf_samples)
            rec, sort = create_dumpable_extractors_from_existing(folder, rec, sort)
            # get num samples in ms
            ms_cut = n_wf_samples // 2 / rec.get_sampling_frequency() * 1000

            # no group
            wav = get_unit_waveforms(rec, sort, ms_before=ms_cut, ms_after=ms_cut, save_property_or_features=False,
                                     n_jobs=n, memmap=m, recompute_info=True)

            for (w, w_gt) in zip(wav, waveforms):
                assert np.allclose(w, w_gt)
            assert 'waveforms' not in sort.get_shared_unit_spike_feature_names()

            # small chunks
            wav = get_unit_waveforms(rec, sort, ms_before=ms_cut, ms_after=ms_cut, save_property_or_features=False,
                                     n_jobs=n, memmap=m, chunk_mb=5, recompute_info=True)

            for (w, w_gt) in zip(wav, waveforms):
                assert np.allclose(w, w_gt)
            assert 'waveforms' not in sort.get_shared_unit_spike_feature_names()

            # change cut ms
            wav = get_unit_waveforms(rec, sort, ms_before=2, ms_after=2, save_property_or_features=True, n_jobs=n,
                                     memmap=m, recompute_info=True)

            for (w, w_gt) in zip(wav, waveforms):
                _, _, samples = w.shape
                assert np.allclose(w[:, :, samples // 2 - n_wf_samples // 2: samples // 2 + n_wf_samples // 2], w_gt)
            assert 'waveforms' in sort.get_shared_unit_spike_feature_names()

            # by group
            rec.set_channel_groups([0, 0, 1, 1])
            wav = get_unit_waveforms(rec, sort, ms_before=ms_cut, ms_after=ms_cut, grouping_property='group', n_jobs=n,
                                     memmap=m, recompute_info=True)

            for (w, w_gt) in zip(wav, waveforms):
                assert np.allclose(w, w_gt[:, :2]) or np.allclose(w, w_gt[:, 2:])

            # test compute_property_from_recordings
            wav = get_unit_waveforms(rec, sort, ms_before=ms_cut, ms_after=ms_cut, grouping_property='group',
                                     compute_property_from_recording=True, n_jobs=n,
                                     memmap=m, recompute_info=True)
            for (w, w_gt) in zip(wav, waveforms):
                assert np.allclose(w, w_gt[:, :2]) or np.allclose(w, w_gt[:, 2:])

            # test max_spikes_per_unit
            wav = get_unit_waveforms(rec, sort, ms_before=ms_cut, ms_after=ms_cut, max_spikes_per_unit=10,
                                     save_property_or_features=False, n_jobs=n,
                                     memmap=m, recompute_info=True)
            for w in wav:
                assert len(w) <= 10

            # test channels
            wav = get_unit_waveforms(rec, sort, ms_before=ms_cut, ms_after=ms_cut, channel_ids=[0, 1, 2], n_jobs=n,
                                     memmap=m, recompute_info=True)

            for (w, w_gt) in zip(wav, waveforms):
                assert np.allclose(w, w_gt[:, :3])
    shutil.rmtree('test')


@pytest.mark.implemented
def test_templates():
    n_wf_samples = 100
    folder = 'test'
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    rec, sort, waveforms, templates, max_chans, amps = create_signal_with_known_waveforms(n_waveforms=2,
                                                                                          n_channels=4,
                                                                                          n_wf_samples=n_wf_samples)
    rec, sort = create_dumpable_extractors_from_existing(folder, rec, sort)
    # get num samples in ms
    ms_cut = n_wf_samples // 2 / rec.get_sampling_frequency() * 1000

    # no group
    temp = get_unit_templates(rec, sort, ms_before=ms_cut, ms_after=ms_cut, save_property_or_features=False,
                              save_wf_as_features=False, recompute_info=True)

    for (t, t_gt) in zip(temp, templates):
        assert np.allclose(t, t_gt, atol=1)
    assert 'template' not in sort.get_shared_unit_property_names()
    assert 'waveforms' not in sort.get_shared_unit_spike_feature_names()

    # change cut ms
    temp = get_unit_templates(rec, sort, ms_before=2, ms_after=2, save_property_or_features=True,
                              recompute_waveforms=True, recompute_info=True)

    for (t, t_gt) in zip(temp, templates):
        _, samples = t.shape
        assert np.allclose(t[:, samples // 2 - n_wf_samples // 2: samples // 2 + n_wf_samples // 2], t_gt, atol=1)
    assert 'template' in sort.get_shared_unit_property_names()
    assert 'waveforms' in sort.get_shared_unit_spike_feature_names()

    # by group
    rec.set_channel_groups([0, 0, 1, 1])
    temp = get_unit_templates(rec, sort, ms_before=ms_cut, ms_after=ms_cut, grouping_property='group',
                              recompute_info=True)

    for (t, t_gt) in zip(temp, templates):
        assert np.allclose(t, t_gt[:2], atol=1) or np.allclose(t, t_gt[2:], atol=1)
    shutil.rmtree('test')


@pytest.mark.implemented
def test_max_chan():
    n_wf_samples = 100
    folder = 'test'
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    rec, sort, waveforms, templates, max_chans, amps = create_signal_with_known_waveforms(n_waveforms=2,
                                                                                          n_channels=4,
                                                                                          n_wf_samples=n_wf_samples)
    rec, sort = create_dumpable_extractors_from_existing(folder, rec, sort)
    max_channels = get_unit_max_channels(rec, sort, save_property_or_features=False)
    assert np.allclose(np.array(max_chans), np.array(max_channels))
    assert 'max_channel' not in sort.get_shared_unit_property_names()

    max_channels = get_unit_max_channels(rec, sort, save_property_or_features=True, recompute_templates=True,
                                         peak='neg', recompute_info=True)
    assert np.allclose(np.array(max_chans), np.array(max_channels))
    assert 'max_channel' in sort.get_shared_unit_property_names()

    # multiple channels
    max_channels = get_unit_max_channels(rec, sort, max_channels=2,
                                         peak='neg', recompute_info=True)
    assert np.allclose(np.array(max_chans), np.array(max_channels)[:, 0])
    assert np.array(max_channels).shape[1] == 2
    shutil.rmtree('test')


@pytest.mark.implemented
def test_amplitudes():
    n_jobs = [0, 2]
    for n in n_jobs:
        for m in memmaps:
            print('N jobs', n, 'memmap', m)
            n_wf_samples = 100
            folder = 'test'
            if os.path.isdir(folder):
                shutil.rmtree(folder)
            rec, sort, waveforms, templates, max_chans, amps = create_signal_with_known_waveforms(n_waveforms=2,
                                                                                                  n_channels=4,
                                                                                                  n_wf_samples=
                                                                                                  n_wf_samples)
            rec, sort = create_dumpable_extractors_from_existing(folder, rec, sort)

            amp = get_unit_amplitudes(rec, sort, frames_before=50, frames_after=50, save_property_or_features=False,
                                      n_jobs=n, memmap=m)

            for (a, a_gt) in zip(amp, amps):
                assert np.allclose(a, np.abs(a_gt))
            assert 'amplitudes' not in sort.get_shared_unit_spike_feature_names()

            amp = get_unit_amplitudes(rec, sort, frames_before=50, frames_after=50, save_property_or_features=True,
                                      peak='neg', n_jobs=n, memmap=m)

            for (a, a_gt) in zip(amp, amps):
                assert np.allclose(a, a_gt)
            assert 'amplitudes' in sort.get_shared_unit_spike_feature_names()

            # relative
            amp = get_unit_amplitudes(rec, sort, frames_before=50, frames_after=50, save_property_or_features=False,
                                      recompute_info=True, method='relative', n_jobs=n, memmap=m)

            amps_rel = [a / np.median(a) for a in amps]

            for (a, a_gt) in zip(amp, amps_rel):
                assert np.allclose(a, np.abs(a_gt), 0.02)
            shutil.rmtree('test')


@pytest.mark.implemented
def test_spiking_activity():
    n_jobs = [0, 2]
    num_channels = 32
    folder = 'test'
    for n in n_jobs:
        print('N jobs', n)
        if os.path.isdir(folder):
            shutil.rmtree(folder)
        rec, sort = se.example_datasets.toy_example(num_channels=num_channels, dumpable=True, dump_folder=folder)
        rates, amps = compute_channel_spiking_activity(rec, n_jobs=n)

        assert len(rates) == num_channels and len(amps) == num_channels
        assert 'spike_rate' in rec.get_shared_channel_property_names()
        assert 'spike_amplitude' in rec.get_shared_channel_property_names()

        shutil.rmtree(folder)


@pytest.mark.notimplemented
def test_compute_pca_scores():
    num_channels = 32
    folder = 'test'
    n_jobs = [0, 2]
    for n in n_jobs:
        for m in memmaps:
            print('N jobs', n, 'memmap', m)

            if os.path.isdir(folder):
                shutil.rmtree(folder)
            rec, sort = se.example_datasets.toy_example(num_channels=num_channels, dumpable=True, dump_folder=folder)

            pca_scores = compute_unit_pca_scores(rec, sort, n_comp=5, memmap=m, n_jobs=n,
                                                 save_property_or_features=False)
            for pc in pca_scores:
                assert pc.shape[-1] == 5
            assert 'pca_scores' not in sort.get_shared_unit_spike_feature_names()


            pca_scores = compute_unit_pca_scores(rec, sort, unit_ids=np.asarray([1,2]), channel_ids=[0, 1, 2, 3, 4],
                                                 max_channels_per_waveforms=3, n_comp=3, memmap=m, n_jobs=n)
            assert len(pca_scores) == 2
            for pc in pca_scores:
                assert pc.shape[-1] == 3
            assert 'pca_scores' not in sort.get_shared_unit_spike_feature_names()
            assert 'pca_scores_channel_idxs' not in sort.get_shared_unit_property_names()

            pca_scores = compute_unit_pca_scores(rec, sort, channel_ids=[0, 1, 2, 3, 4],
                                                 max_channels_per_waveforms=3, n_comp=3, memmap=m, n_jobs=n)
            for pc in pca_scores:
                assert pc.shape[-1] == 3
            assert 'pca_scores' in sort.get_shared_unit_spike_feature_names()
            assert 'pca_scores_channel_idxs' in sort.get_shared_unit_property_names()

            shutil.rmtree(folder)


@pytest.mark.notimplemented
def test_compute_centers_of_mass():
    num_channels = 32
    folder = 'test'
    n_jobs = [0, 2]
    locations = np.zeros((num_channels, 2))
    radius = 100
    # create circular locations
    for i in np.arange(num_channels):
        angle = 2*np.pi / num_channels * i
        locations[i, 0] = np.cos(angle) * radius
        locations[i, 1] = np.sin(angle) * radius

    for n in n_jobs:
        for m in memmaps:
            print('N jobs', n, 'memmap', m)

            if os.path.isdir(folder):
                shutil.rmtree(folder)
            rec, sort = se.example_datasets.toy_example(num_channels=num_channels, dumpable=True, dump_folder=folder)
            rec.set_channel_locations(locations)

            coms = compute_unit_centers_of_mass(rec, sort, num_channels=None, memmap=m, n_jobs=n,
                                                save_property_or_features=False)
            for com in coms:
                assert np.linalg.norm(com) <= radius
            assert 'com' not in sort.get_shared_unit_property_names()

            coms = compute_unit_centers_of_mass(rec, sort, num_channels=5, memmap=m, n_jobs=n)
            for com in coms:
                assert np.linalg.norm(com) <= radius
            assert 'com' in sort.get_shared_unit_property_names()

            shutil.rmtree(folder)


@pytest.mark.implemented
def test_export_to_phy():
    folder = 'test'
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    rec, sort = se.example_datasets.toy_example(dump_folder=folder, dumpable=True, duration=10, num_channels=8)

    export_to_phy(rec, sort, output_folder='phy')
    rec.set_channel_groups([0, 0, 0, 0, 1, 1, 1, 1])
    export_to_phy(rec, sort, output_folder='phy_group', grouping_property='group', recompute_info=True)
    export_to_phy(rec, sort, output_folder='phy_max_channels', max_channels_per_template=4, recompute_info=True)
    export_to_phy(rec, sort, output_folder='phy_no_feat', grouping_property='group', compute_pc_features=False,
                  recompute_info=True)
    export_to_phy(rec, sort, output_folder='phy_no_amp', compute_amplitudes=False)
    export_to_phy(rec, sort, output_folder='phy_no_amp_feat', compute_amplitudes=False,
                  compute_pc_features=False)
    export_to_phy(rec, sort, output_folder='phy_par', n_jobs=2)

    rec_phy = se.PhyRecordingExtractor('phy')
    rec_phyg = se.PhyRecordingExtractor('phy_group')
    assert np.allclose(rec.get_traces(), rec_phy.get_traces())
    assert np.allclose(rec.get_traces(), rec_phyg.get_traces())
    assert not (Path('phy_no_feat') / 'pc_features.npy').is_file()
    assert not (Path('phy_no_feat') / 'pc_feature_ind.npy').is_file()
    assert not (Path('phy_no_amp') / 'amplitudes.npy').is_file()
    assert not (Path('phy_no_amp_feat') / 'amplitudes.npy').is_file()
    assert not (Path('phy_no_amp_feat') / 'pc_features.npy').is_file()
    assert not (Path('phy_no_amp_feat') / 'pc_feature_ind.npy').is_file()

    sort_phy = se.PhySortingExtractor('phy')
    sort_phyg = se.PhySortingExtractor('phy_group')

    assert np.allclose(sort_phy.get_unit_spike_train(0), sort.get_unit_spike_train(sort.get_unit_ids()[0]))
    assert np.allclose(sort_phyg.get_unit_spike_train(2), sort.get_unit_spike_train(sort.get_unit_ids()[2]))

    rec.set_channel_groups([0, 0, 0, 0, 1, 1, 1, 1])
    recrm = remove_bad_channels(rec, [1, 2, 5])
    export_to_phy(recrm, sort, output_folder='phy_rm', grouping_property='group', recompute_info=True, verbose=True)
    templates_ind = np.load('phy_rm/template_ind.npy')
    assert len(np.where(templates_ind == -1)[0]) > 0  # removed channels are -1

    try:
        shutil.rmtree('test')
        shutil.rmtree('phy')
        shutil.rmtree('phy_group')
        shutil.rmtree('phy_max_channels')
        shutil.rmtree('phy_no_feat')
        shutil.rmtree('phy_no_amp')
        shutil.rmtree('phy_no_amp_feat')
        shutil.rmtree('phy_rm')
        shutil.rmtree('phy_par')
    except:
        print("Could not delete some test folders")


@pytest.mark.implemented
def test_set_unit_properties_by_max_channel_properties():
    folder = 'test'
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    rec, sort = se.example_datasets.toy_example(dump_folder=folder, dumpable=True, duration=10, num_channels=8)

    rec.set_channel_groups([0, 0, 0, 0, 1, 1, 1, 1])
    set_unit_properties_by_max_channel_properties(rec, sort, property='group')
    assert 'group' in sort.get_shared_unit_property_names()
    sort_groups = [sort.get_unit_property(u, 'group') for u in sort.get_unit_ids()]
    assert np.all(np.unique(sort_groups) == [0, 1])
    shutil.rmtree('test')


@pytest.mark.implemented
def test_compute_features():
    rec, sort = se.example_datasets.toy_example(duration=10, num_channels=8)

    features = compute_unit_template_features(rec, sort)
    assert isinstance(features, dict)

    for feat, feat_val in features.items():
        assert len(feat_val) == len(sort.get_unit_ids())
        assert np.all(len(f) == 1 for f in feat_val)

    max_chan_per_features = 2
    features = compute_unit_template_features(rec, sort, max_channels_per_features=max_chan_per_features)
    assert isinstance(features, dict)

    for feat, feat_val in features.items():
        assert len(feat_val) == len(sort.get_unit_ids())
        assert np.all(len(f) == 2 for f in feat_val)

    features = compute_unit_template_features(rec, sort, max_channels_per_features=max_chan_per_features,
                                              upsampling_factor=10)
    assert isinstance(features, dict)

    for feat, feat_val in features.items():
        assert len(feat_val) == len(sort.get_unit_ids())
        assert np.all(len(f) == 2 for f in feat_val)

    features_df = compute_unit_template_features(rec, sort, max_channels_per_features=max_chan_per_features,
                                                 as_dataframe=True)
    assert isinstance(features_df, pandas.DataFrame)
    assert np.all([fk in features.keys() for fk in features_df.keys()])


if __name__ == '__main__':
    test_spiking_activity()
