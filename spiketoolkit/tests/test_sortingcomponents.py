import spikeextractors as se
import spiketoolkit as st
import numpy as np
import shutil


def test_detection():
    folder = 'test'
    rec, sort = se.example_datasets.toy_example(num_channels=4, duration=20, seed=0, dumpable=True, dump_folder=folder)

    # negative
    sort_d_n = st.sortingcomponents.detect_spikes(rec)
    sort_dc_n = st.sortingcomponents.detect_spikes(rec, n_jobs=1, chunk_mb=10)
    sort_dp_n = st.sortingcomponents.detect_spikes(rec, n_jobs=2, chunk_mb=10)

    assert 'channel' in sort_d_n.get_shared_unit_property_names()
    assert 'channel' in sort_dc_n.get_shared_unit_property_names()
    assert 'channel' in sort_dp_n.get_shared_unit_property_names()
    assert 'spike_rate' in sort_d_n.get_shared_unit_property_names()
    assert 'spike_rate' in sort_dc_n.get_shared_unit_property_names()
    assert 'spike_rate' in sort_dp_n.get_shared_unit_property_names()
    assert 'spike_amplitude' in sort_d_n.get_shared_unit_property_names()
    assert 'spike_amplitude' in sort_dc_n.get_shared_unit_property_names()
    assert 'spike_amplitude' in sort_dp_n.get_shared_unit_property_names()

    for u in sort_d_n.get_unit_ids():
        assert np.array_equal(sort_d_n.get_unit_spike_train(u), sort_dp_n.get_unit_spike_train(u))
        assert np.array_equal(sort_d_n.get_unit_spike_train(u), sort_dc_n.get_unit_spike_train(u))

    # positive
    sort_d_p = st.sortingcomponents.detect_spikes(rec, detect_sign=1)
    sort_dc_p = st.sortingcomponents.detect_spikes(rec, detect_sign=1, n_jobs=1, chunk_mb=10)
    sort_dp_p = st.sortingcomponents.detect_spikes(rec, detect_sign=1, n_jobs=2, chunk_mb=10)

    assert 'channel' in sort_d_p.get_shared_unit_property_names()
    assert 'channel' in sort_dc_p.get_shared_unit_property_names()
    assert 'channel' in sort_dp_p.get_shared_unit_property_names()
    assert 'spike_rate' in sort_d_p.get_shared_unit_property_names()
    assert 'spike_rate' in sort_dc_p.get_shared_unit_property_names()
    assert 'spike_rate' in sort_dp_p.get_shared_unit_property_names()
    assert 'spike_amplitude' in sort_d_p.get_shared_unit_property_names()
    assert 'spike_amplitude' in sort_dc_p.get_shared_unit_property_names()
    assert 'spike_amplitude' in sort_dp_p.get_shared_unit_property_names()

    for u in sort_d_p.get_unit_ids():
        assert np.array_equal(sort_d_p.get_unit_spike_train(u), sort_dp_p.get_unit_spike_train(u))
        assert np.array_equal(sort_d_p.get_unit_spike_train(u), sort_dc_p.get_unit_spike_train(u))

    # both
    sort_d_b = st.sortingcomponents.detect_spikes(rec, detect_sign=0)
    sort_dc_b = st.sortingcomponents.detect_spikes(rec, detect_sign=0, n_jobs=2, chunk_mb=10)
    sort_dp_b = st.sortingcomponents.detect_spikes(rec, detect_sign=0, n_jobs=2, chunk_mb=10)

    assert 'channel' in sort_d_b.get_shared_unit_property_names()
    assert 'channel' in sort_dc_b.get_shared_unit_property_names()
    assert 'channel' in sort_dp_b.get_shared_unit_property_names()
    assert 'spike_rate' in sort_d_b.get_shared_unit_property_names()
    assert 'spike_rate' in sort_dc_b.get_shared_unit_property_names()
    assert 'spike_rate' in sort_dp_b.get_shared_unit_property_names()
    assert 'spike_amplitude' in sort_d_b.get_shared_unit_property_names()
    assert 'spike_amplitude' in sort_dc_b.get_shared_unit_property_names()
    assert 'spike_amplitude' in sort_dp_b.get_shared_unit_property_names()

    for u in sort_d_b.get_unit_ids():
        assert np.array_equal(sort_d_b.get_unit_spike_train(u), sort_dp_b.get_unit_spike_train(u))
        assert np.array_equal(sort_d_b.get_unit_spike_train(u), sort_dc_b.get_unit_spike_train(u))

    shutil.rmtree(folder)


if __name__ == '__main__':
    test_detection()
