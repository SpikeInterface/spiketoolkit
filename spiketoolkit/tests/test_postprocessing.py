import numpy as np
import pytest
from .utils import create_signal_with_known_waveforms
import spikeextractors as se
from spiketoolkit.postprocessing import get_unit_waveforms, get_unit_templates, get_unit_amplitudes, \
    get_unit_max_channels, set_unit_properties_by_max_channel_properties, compute_unit_pca_scores, export_to_phy


@pytest.mark.implemented
def test_waveforms():
    n_wf_samples = 100
    rec, sort, waveforms, templates, max_chans, amps = create_signal_with_known_waveforms(n_waveforms=2,
                                                                                          n_channels=4,
                                                                                          n_wf_samples=n_wf_samples)
    # get num samples in ms
    ms_cut = n_wf_samples // 2 / rec.get_sampling_frequency() * 1000

    # no group
    wav = get_unit_waveforms(rec, sort, ms_before=ms_cut, ms_after=ms_cut, save_as_features=False)

    for (w, w_gt) in zip(wav, waveforms):
        assert np.allclose(w, w_gt)
    assert 'waveforms' not in sort.get_shared_unit_spike_feature_names()

    # change cut ms
    wav = get_unit_waveforms(rec, sort, ms_before=2, ms_after=2, save_as_features=True)

    for (w, w_gt) in zip(wav, waveforms):
        _, _, samples = w.shape
        assert np.allclose(w[:, :, samples // 2 - n_wf_samples // 2: samples // 2 + n_wf_samples // 2], w_gt)
    assert 'waveforms' in sort.get_shared_unit_spike_feature_names()

    # by group
    rec.set_channel_groups(rec.get_channel_ids(), [0, 0, 1, 1])
    wav = get_unit_waveforms(rec, sort, ms_before=ms_cut, ms_after=ms_cut, grouping_property='group')

    for (w, w_gt) in zip(wav, waveforms):
        assert np.allclose(w, w_gt[:, :2]) or np.allclose(w, w_gt[:, 2:])

    # test compute_property_from_recordings
    wav = get_unit_waveforms(rec, sort, ms_before=ms_cut, ms_after=ms_cut, grouping_property='group',
                             compute_property_from_recording=True)
    for (w, w_gt) in zip(wav, waveforms):
        assert np.allclose(w, w_gt[:, :2]) or np.allclose(w, w_gt[:, 2:])

    # test max_spikes_per_unit
    wav = get_unit_waveforms(rec, sort, ms_before=ms_cut, ms_after=ms_cut, max_spikes_per_unit=10, save_as_features=False)
    for w in wav:
        assert len(w) <= 10

    # test channels
    wav = get_unit_waveforms(rec, sort, ms_before=ms_cut, ms_after=ms_cut, channel_ids=[0, 1, 2])

    for (w, w_gt) in zip(wav, waveforms):
        assert np.allclose(w, w_gt[:, :3])


@pytest.mark.implemented
def test_templates():
    n_wf_samples = 100
    rec, sort, waveforms, templates, max_chans, amps = create_signal_with_known_waveforms(n_waveforms=2,
                                                                                          n_channels=4,
                                                                                          n_wf_samples=n_wf_samples)
    # get num samples in ms
    ms_cut = n_wf_samples // 2 / rec.get_sampling_frequency() * 1000

    # no group
    temp = get_unit_templates(rec, sort, ms_before=ms_cut, ms_after=ms_cut, save_as_property=False,
                              save_wf_as_features=False)

    for (t, t_gt) in zip(temp, templates):
        assert np.allclose(t, t_gt, atol=1)
    assert 'template' not in sort.get_shared_unit_property_names()
    assert 'waveforms' not in sort.get_shared_unit_spike_feature_names()

    # change cut ms
    temp = get_unit_templates(rec, sort, ms_before=2, ms_after=2, save_as_property=True, recompute_waveforms=True)

    for (t, t_gt) in zip(temp, templates):
        _, samples = t.shape
        assert np.allclose(t[:, samples // 2 - n_wf_samples // 2: samples // 2 + n_wf_samples // 2], t_gt, atol=1)
    assert 'template' in sort.get_shared_unit_property_names()
    assert 'waveforms' in sort.get_shared_unit_spike_feature_names()

    # by group
    rec.set_channel_groups(rec.get_channel_ids(), [0, 0, 1, 1])
    temp = get_unit_templates(rec, sort, ms_before=ms_cut, ms_after=ms_cut, grouping_property='group',
                              recompute_waveforms=True)

    for (t, t_gt) in zip(temp, templates):
        assert np.allclose(t, t_gt[:2], atol=1) or np.allclose(t, t_gt[2:], atol=1)


@pytest.mark.implemented
def test_max_chan():
    n_wf_samples = 100
    rec, sort, waveforms, templates, max_chans, amps = create_signal_with_known_waveforms(n_waveforms=2,
                                                                                          n_channels=4,
                                                                                          n_wf_samples=n_wf_samples)
    max_channels = get_unit_max_channels(rec, sort, save_as_property=False)
    assert np.allclose(np.array(max_chans), np.array(max_channels))
    assert 'max_channel' not in sort.get_shared_unit_property_names()

    max_channels = get_unit_max_channels(rec, sort, save_as_property=True, recompute_templates=True,
                                         peak='neg')
    assert np.allclose(np.array(max_chans), np.array(max_channels))
    assert 'max_channel' in sort.get_shared_unit_property_names()


@pytest.mark.implemented
def test_amplitudes():
    n_wf_samples = 100
    rec, sort, waveforms, templates, max_chans, amps = create_signal_with_known_waveforms(n_waveforms=2,
                                                                                          n_channels=4,
                                                                                          n_wf_samples=n_wf_samples)

    amp = get_unit_amplitudes(rec, sort, frames_before=50, frames_after=50, save_as_features=False)

    for (a, a_gt) in zip(amp, amps):
        assert np.allclose(a, np.abs(a_gt))
    assert 'amps' not in sort.get_shared_unit_spike_feature_names()

    amp = get_unit_amplitudes(rec, sort, frames_before=50, frames_after=50, save_as_features=True, peak='neg')

    for (a, a_gt) in zip(amp, amps):
        assert np.allclose(a, a_gt)
    assert 'amps' in sort.get_shared_unit_spike_feature_names()

    # relative
    amp = get_unit_amplitudes(rec, sort, frames_before=50, frames_after=50, save_as_features=False, method='relative')

    amps_rel = [a / np.median(a) for a in amps]

    for (a, a_gt) in zip(amp, amps_rel):
        assert np.allclose(a, np.abs(a_gt), 0.02)


@pytest.mark.implemented
def test_export_to_phy():
    rec, sort = se.example_datasets.toy_example(duration=10, num_channels=8)

    export_to_phy(rec, sort, output_folder='phy')
    rec.set_channel_groups(rec.get_channel_ids(), [0, 0, 0, 0, 1, 1, 1, 1])
    export_to_phy(rec, sort, output_folder='phy_group', grouping_property='group')

    rec_phy = se.PhyRecordingExtractor('phy')
    rec_phyg = se.PhyRecordingExtractor('phy_group')
    assert np.allclose(rec.get_traces(), rec_phy.get_traces())
    assert np.allclose(rec.get_traces(), rec_phyg.get_traces())

    sort_phy = se.PhySortingExtractor('phy', load_waveforms=True)
    sort_phyg = se.PhySortingExtractor('phy_group', load_waveforms=True)

    assert np.allclose(sort_phy.get_unit_spike_train(0), sort.get_unit_spike_train(sort.get_unit_ids()[0]))
    assert np.allclose(sort_phyg.get_unit_spike_train(2), sort.get_unit_spike_train(sort.get_unit_ids()[2]))
    assert sort_phy.get_unit_spike_features(1, 'waveforms').shape[1] == 8
    assert sort_phyg.get_unit_spike_features(3, 'waveforms').shape[1] == 4


@pytest.mark.implemented
def test_set_unit_properties_by_max_channel_properties():
    rec, sort = se.example_datasets.toy_example(duration=10, num_channels=8)

    rec.set_channel_groups(rec.get_channel_ids(), [0, 0, 0, 0, 1, 1, 1, 1])
    set_unit_properties_by_max_channel_properties(rec, sort, property='group')
    assert 'group' in sort.get_shared_unit_property_names()
    sort_groups = [sort.get_unit_property(u, 'group') for u in sort.get_unit_ids()]
    assert np.all(np.unique(sort_groups) == [0, 1])


@pytest.mark.notimplemented
def test_compute_pca_scores():
    pass
