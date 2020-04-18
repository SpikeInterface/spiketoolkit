import numpy as np
import spikeextractors as se
import pytest
import shutil
from spiketoolkit.tests.utils import check_signal_power_signal1_below_signal2
from spiketoolkit.preprocessing import bandpass_filter, blank_saturation, center, clip, common_reference, \
    normalize_by_quantile, notch_filter, rectify, remove_artifacts, remove_bad_channels, resample, transform, \
    whiten
from spikeextractors.tests.utils import check_dumping


@pytest.mark.implemented
def test_bandpass_filter():
    rec, sort = se.example_datasets.toy_example(dump_folder='test', dumpable=True, duration=10, num_channels=4, seed=0)

    rec_fft = bandpass_filter(rec, freq_min=5000, freq_max=10000, filter_type='fft')

    assert check_signal_power_signal1_below_signal2(rec_fft.get_traces(), rec.get_traces(), freq_range=[1000, 5000],
                                                    fs=rec.get_sampling_frequency())
    assert check_signal_power_signal1_below_signal2(rec_fft.get_traces(), rec.get_traces(), freq_range=[10000, 15000],
                                                    fs=rec.get_sampling_frequency())

    rec_sci = bandpass_filter(rec, freq_min=3000, freq_max=6000, filter_type='butter', order=3)

    assert check_signal_power_signal1_below_signal2(rec_sci.get_traces(), rec.get_traces(), freq_range=[1000, 3000],
                                                    fs=rec.get_sampling_frequency())
    assert check_signal_power_signal1_below_signal2(rec_sci.get_traces(), rec.get_traces(), freq_range=[6000, 10000],
                                                    fs=rec.get_sampling_frequency())

    rec_cache = bandpass_filter(rec, freq_min=3000, freq_max=6000, filter_type='butter', order=3, cache_to_file=True)

    assert check_signal_power_signal1_below_signal2(rec_cache.get_traces(), rec.get_traces(), freq_range=[1000, 3000],
                                                    fs=rec.get_sampling_frequency())
    assert check_signal_power_signal1_below_signal2(rec_cache.get_traces(), rec.get_traces(), freq_range=[6000, 10000],
                                                    fs=rec.get_sampling_frequency())

    traces = rec.get_traces().astype('uint16')
    rec_u = se.NumpyRecordingExtractor(traces, sampling_frequency=rec.get_sampling_frequency())
    rec_fu = bandpass_filter(rec_u, freq_min=5000, freq_max=10000, filter_type='fft')

    assert check_signal_power_signal1_below_signal2(rec_fu.get_traces(), rec_u.get_traces(), freq_range=[1000, 5000],
                                                    fs=rec.get_sampling_frequency())
    assert check_signal_power_signal1_below_signal2(rec_fu.get_traces(), rec_u.get_traces(), freq_range=[10000, 15000],
                                                    fs=rec.get_sampling_frequency())
    assert not str(rec_fu.get_dtype()).startswith('u')

    check_dumping(rec_fft)
    check_dumping(rec_cache)

    shutil.rmtree('test')


@pytest.mark.implemented
def test_bandpass_filter_with_cache():
    rec, sort = se.example_datasets.toy_example(dump_folder='test', dumpable=True, duration=10, num_channels=4, seed=0)

    rec_filtered = bandpass_filter(rec, freq_min=5000, freq_max=10000, cache_to_file=True, chunk_size=10000)

    rec_filtered2 = bandpass_filter(rec, freq_min=5000, freq_max=10000, cache_to_file=True, chunk_size=None)

    rec_filtered3 = bandpass_filter(rec, freq_min=5000, freq_max=10000, cache_chunks=True, chunk_size=10000)
    rec_filtered3.get_traces()
    assert rec_filtered3._filtered_cache_chunks.get('0') is not None

    rec_filtered4 = bandpass_filter(rec, freq_min=5000, freq_max=10000, cache_chunks=True, chunk_size=None)

    assert np.allclose(rec_filtered.get_traces(), rec_filtered2.get_traces(), rtol=1e-02, atol=1e-02)
    assert np.allclose(rec_filtered.get_traces(), rec_filtered3.get_traces(), rtol=1e-02, atol=1e-02)
    assert np.allclose(rec_filtered.get_traces(), rec_filtered4.get_traces(), rtol=1e-02, atol=1e-02)

    check_dumping(rec_filtered)
    check_dumping(rec_filtered2)
    check_dumping(rec_filtered3)

    shutil.rmtree('test')


@pytest.mark.implemented
def test_blank_saturation():
    rec, sort = se.example_datasets.toy_example(dump_folder='test', dumpable=True, duration=10, num_channels=4, seed=0)
    threshold = 2
    rec_bs = blank_saturation(rec, threshold=threshold)

    index_below_threshold = np.where(rec.get_traces() < threshold)

    assert np.all(rec_bs.get_traces()[index_below_threshold] < threshold)

    check_dumping(rec_bs)
    shutil.rmtree('test')


@pytest.mark.implemented
def test_center():
    rec, sort = se.example_datasets.toy_example(dump_folder='test', dumpable=True, duration=10, num_channels=4, seed=0)

    rec_c = center(rec, mode='mean')
    assert np.allclose(np.mean(rec_c.get_traces(), axis=1), 0, atol=0.001)
    check_dumping(rec_c)

    rec_c = center(rec, mode='median')
    assert np.allclose(np.median(rec_c.get_traces(), axis=1), 0, atol=0.001)
    check_dumping(rec_c)

    shutil.rmtree('test')


@pytest.mark.implemented
def test_clip():
    rec, sort = se.example_datasets.toy_example(dump_folder='test', dumpable=True, duration=10, num_channels=4, seed=0)
    threshold = 5
    rec_clip = clip(rec, a_min=-threshold, a_max=threshold)

    index_below_threshold = np.where(rec.get_traces() < -threshold)
    index_above_threshold = np.where(rec.get_traces() > threshold)

    assert np.all(rec_clip.get_traces()[index_below_threshold] == -threshold)
    assert np.all(rec_clip.get_traces()[index_above_threshold] == threshold)

    check_dumping(rec_clip)
    shutil.rmtree('test')


@pytest.mark.implemented
def test_common_reference():
    rec, sort = se.example_datasets.toy_example(dump_folder='test', dumpable=True, duration=10, num_channels=4, seed=0)

    # no groups
    rec_cmr = common_reference(rec, reference='median')
    rec_car = common_reference(rec, reference='average')
    rec_sin = common_reference(rec, reference='single', ref_channels=0)
    rec_cmr_int16 = common_reference(rec, dtype='int16')

    traces = rec.get_traces()
    assert np.allclose(traces, rec_cmr.get_traces() + np.median(traces, axis=0, keepdims=True), atol=0.01)
    assert np.allclose(traces, rec_car.get_traces() + np.mean(traces, axis=0, keepdims=True), atol=0.01)
    assert not np.all(rec_sin.get_traces()[0])
    assert np.allclose(rec_sin.get_traces()[1], traces[1] - traces[0])
    assert 'int16' in str(rec_cmr_int16.get_dtype())

    # test groups
    groups = [[0, 1], [2, 3]]
    rec_cmr_g = common_reference(rec, reference='median', groups=groups)
    rec_car_g = common_reference(rec, reference='average', groups=groups)
    rec_sin_g = common_reference(rec, reference='single', ref_channels=[0, 2], groups=groups)
    rec_cmr_int16_g = common_reference(rec, groups=groups, dtype='int16')

    traces = rec.get_traces()
    assert np.allclose(traces[:2], rec_cmr_g.get_traces()[:2] + np.median(traces[:2], axis=0, keepdims=True), atol=0.01)
    assert np.allclose(traces[2:], rec_cmr_g.get_traces()[2:] + np.median(traces[2:], axis=0, keepdims=True), atol=0.01)
    assert np.allclose(traces[:2], rec_car_g.get_traces()[:2] + np.mean(traces[:2], axis=0, keepdims=True), atol=0.01)
    assert np.allclose(traces[2:], rec_car_g.get_traces()[2:] + np.mean(traces[2:], axis=0, keepdims=True), atol=0.01)
    assert not np.all(rec_sin_g.get_traces()[0])
    assert np.allclose(rec_sin_g.get_traces()[1], traces[1] - traces[0])
    assert not np.all(rec_sin_g.get_traces()[2])
    assert np.allclose(rec_sin_g.get_traces()[3], traces[3] - traces[2])
    assert 'int16' in str(rec_cmr_int16_g.get_dtype())

    check_dumping(rec_cmr)
    check_dumping(rec_car)
    check_dumping(rec_sin)
    check_dumping(rec_cmr_int16)
    shutil.rmtree('test')


@pytest.mark.notimplemented
def test_norm_by_quantile():
    pass


@pytest.mark.implemented
def test_notch_filter():
    rec, sort = se.example_datasets.toy_example(dump_folder='test', dumpable=True, duration=10, num_channels=4, seed=0)

    rec_n = notch_filter(rec, 3000, q=10)

    assert check_signal_power_signal1_below_signal2(rec_n.get_traces(), rec.get_traces(), freq_range=[2900, 3100],
                                                    fs=rec.get_sampling_frequency())

    check_dumping(rec_n)
    shutil.rmtree('test')


@pytest.mark.implemented
def test_rectify():
    rec, sort = se.example_datasets.toy_example(dump_folder='test', dumpable=True, duration=10, num_channels=4, seed=0)

    rec_rect = rectify(rec)

    assert np.allclose(rec_rect.get_traces(), np.abs(rec.get_traces()))

    check_dumping(rec_rect)
    shutil.rmtree('test')


@pytest.mark.implemented
def test_remove_artifacts():
    rec, sort = se.example_datasets.toy_example(dump_folder='test', dumpable=True, duration=10, num_channels=4, seed=0)
    triggers = [30000, 90000]
    ms = 10
    ms_frames = int(ms * rec.get_sampling_frequency() / 1000)

    rec_rmart = remove_artifacts(rec, triggers, ms_before=10, ms_after=10)
    traces_all_0 = rec_rmart.get_traces(start_frame=triggers[0] - ms_frames, end_frame=triggers[0] + ms_frames)
    traces_short_0 = rec_rmart.get_traces(start_frame=triggers[0] - 10, end_frame=triggers[0] + 10)
    traces_all_1 = rec_rmart.get_traces(start_frame=triggers[1] - ms_frames, end_frame=triggers[1] + ms_frames)
    traces_short_1 = rec_rmart.get_traces(start_frame=triggers[1] - 10, end_frame=triggers[1] + 10)

    assert not np.any(traces_all_0)
    assert not np.any(traces_all_1)
    assert not np.any(traces_short_0)
    assert not np.any(traces_short_1)

    check_dumping(rec_rmart)
    shutil.rmtree('test')


@pytest.mark.implemented
def test_remove_bad_channels():
    rec, sort = se.example_datasets.toy_example(dump_folder='test', dumpable=True, duration=10, num_channels=4, seed=0)
    rec_rm = remove_bad_channels(rec, bad_channel_ids=[0])
    assert 0 not in rec_rm.get_channel_ids()

    rec_rm = remove_bad_channels(rec, bad_channel_ids=[1, 2])
    assert 1 not in rec_rm.get_channel_ids() and 2 not in rec_rm.get_channel_ids()

    check_dumping(rec_rm)
    shutil.rmtree('test')

    timeseries = np.random.randn(4, 60000)
    timeseries[1] = 10 * timeseries[1]

    rec_np = se.NumpyRecordingExtractor(timeseries=timeseries, sampling_frequency=30000)
    rec_np.set_channel_locations(np.ones((rec_np.get_num_channels(), 2)))
    se.MdaRecordingExtractor.write_recording(rec_np, 'test')
    rec = se.MdaRecordingExtractor('test')
    rec_rm = remove_bad_channels(rec, bad_channel_ids=None, bad_threshold=2)
    assert 1 not in rec_rm.get_channel_ids()
    check_dumping(rec_rm)

    rec_rm = remove_bad_channels(rec, bad_channel_ids=None, bad_threshold=2, seconds=0.1)
    assert 1 not in rec_rm.get_channel_ids()
    check_dumping(rec_rm)

    rec_rm = remove_bad_channels(rec, bad_channel_ids=None, bad_threshold=2, seconds=10)
    assert 1 not in rec_rm.get_channel_ids()
    check_dumping(rec_rm)
    shutil.rmtree('test')


@pytest.mark.implemented
def test_resample():
    rec, sort = se.example_datasets.toy_example(dump_folder='test', dumpable=True, duration=10, num_channels=4, seed=0)

    resample_rate_low = 0.1 * rec.get_sampling_frequency()
    resample_rate_high = 2 * rec.get_sampling_frequency()

    rec_rsl = resample(rec, resample_rate_low)
    rec_rsh = resample(rec, resample_rate_high)

    assert rec_rsl.get_num_frames() == int(rec.get_num_frames() * 0.1)
    assert rec_rsh.get_num_frames() == int(rec.get_num_frames() * 2)

    check_dumping(rec_rsl)
    check_dumping(rec_rsh)
    shutil.rmtree('test')


@pytest.mark.implemented
def test_transform():
    rec, sort = se.example_datasets.toy_example(dump_folder='test', dumpable=True, duration=10, num_channels=4, seed=0)

    scalar = 3
    offset = 50

    rec_t = transform(rec, scalar=scalar, offset=offset)
    assert np.allclose(rec_t.get_traces(), scalar * rec.get_traces() + offset, atol=0.001)

    scalars = np.random.randn(4)
    offsets = np.random.randn(4)
    rec_t_arr = transform(rec, scalar=scalars, offset=offsets)
    for (tt, to, s, o) in zip(rec_t_arr.get_traces(), rec.get_traces(), scalars, offsets):
        assert np.allclose(tt, s * to + o, atol=0.001)

    check_dumping(rec_t)
    check_dumping(rec_t_arr)
    shutil.rmtree('test')


@pytest.mark.implemented
def test_whiten():
    rec, sort = se.example_datasets.toy_example(dump_folder='test', dumpable=True, duration=10, num_channels=4, seed=0)

    rec_w = whiten(rec)
    cov_w = np.cov(rec_w.get_traces())
    assert np.allclose(cov_w, np.eye(4), atol=0.3)

    # should size should not affect
    rec_w2 = whiten(rec, chunk_size=30000)

    assert np.array_equal(rec_w.get_traces(), rec_w2.get_traces())

    check_dumping(rec_w)
    shutil.rmtree('test')


if __name__ == '__main__':
    test_bandpass_filter()
    test_bandpass_filter_with_cache()
    test_blank_saturation()
    test_clip()
    test_center()
    test_common_reference()
    test_norm_by_quantile()
    test_notch_filter()
    test_rectify()
    test_remove_artifacts()
    test_remove_bad_channels()
    test_resample()
    test_transform()
    test_whiten()
