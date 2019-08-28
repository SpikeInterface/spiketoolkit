import numpy as np
import pytest
from .utils import create_signal_with_known_waveforms
from spiketoolkit.postprocessing import get_unit_waveforms, get_unit_templates, get_unit_amplitudes, \
    get_unit_max_channels, compute_unit_pca_scores, export_to_phy


@pytest.mark.implemented
def test_waveforms():
    n_wf_samples = 100
    rec, sort, waveforms, templates, max_chans = create_signal_with_known_waveforms(n_waveforms=2,
                                                                                    n_channels=4,
                                                                                    n_wf_samples=n_wf_samples)
    # get num samples in ms
    ms_cut = n_wf_samples // 2 / rec.get_sampling_frequency() * 1000

    # no group
    wav = get_unit_waveforms(rec, sort, ms_before=ms_cut, ms_after=ms_cut, save_as_features=False)

    for (w, w_gt) in zip(wav, waveforms):
        assert np.allclose(w, w_gt)
    assert 'waveforms' not in sort.get_unit_spike_feature_names()

    # change cut ms
    wav = get_unit_waveforms(rec, sort, ms_before=2, ms_after=2, save_as_features=True)

    for (w, w_gt) in zip(wav, waveforms):
        _, _, samples = w.shape
        assert np.allclose(w[:, :, samples // 2 - n_wf_samples // 2: samples // 2 + n_wf_samples // 2], w_gt)
    assert 'waveforms' in sort.get_unit_spike_feature_names()

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

    # test max_num_waveforms
    wav = get_unit_waveforms(rec, sort, ms_before=ms_cut, ms_after=ms_cut, max_num_waveforms=10, save_as_features=False)
    for w in wav:
        assert len(w) <= 10

    # test channels
    wav = get_unit_waveforms(rec, sort, ms_before=ms_cut, ms_after=ms_cut, channels=[0, 1, 2])

    for (w, w_gt) in zip(wav, waveforms):
        assert np.allclose(w, w_gt[:, :3])


@pytest.mark.implemented
def test_templates():
    n_wf_samples = 100
    rec, sort, waveforms, templates, max_chans = create_signal_with_known_waveforms(n_waveforms=2,
                                                                                    n_channels=4,
                                                                                    n_wf_samples=n_wf_samples)
    # get num samples in ms
    ms_cut = n_wf_samples // 2 / rec.get_sampling_frequency() * 1000

    # no group
    temp = get_unit_templates(rec, sort, ms_before=ms_cut, ms_after=ms_cut, save_as_property=False,
                              save_wf_as_features=False)

    for (t, t_gt) in zip(temp, templates):
        assert np.allclose(t, t_gt, atol=1)
    assert 'template' not in sort.get_unit_property_names()
    assert 'waveforms' not in sort.get_unit_spike_feature_names()

    # change cut ms
    temp = get_unit_templates(rec, sort, ms_before=2, ms_after=2, save_as_property=True, recompute_waveforms=True)

    for (t, t_gt) in zip(temp, templates):
        _, samples = t.shape
        assert np.allclose(t[:, samples // 2 - n_wf_samples // 2: samples // 2 + n_wf_samples // 2], t_gt, atol=1)
    assert 'template' in sort.get_unit_property_names()
    assert 'waveforms' in sort.get_unit_spike_feature_names()

    # by group
    rec.set_channel_groups(rec.get_channel_ids(), [0, 0, 1, 1])
    temp = get_unit_templates(rec, sort, ms_before=ms_cut, ms_after=ms_cut, grouping_property='group',
                              recompute_waveforms=True)

    for (t, t_gt) in zip(temp, templates):
        assert np.allclose(t, t_gt[:2], atol=1) or np.allclose(t, t_gt[2:], atol=1)


@pytest.mark.undertest
def test_max_chan():
    n_wf_samples = 100
    rec, sort, waveforms, templates, max_chans = create_signal_with_known_waveforms(n_waveforms=2,
                                                                                    n_channels=4,
                                                                                    n_wf_samples=n_wf_samples)
    max_channels = get_unit_max_channels(rec, sort)
    assert np.allclose(np.array(max_chans), np.array(max_channels))



