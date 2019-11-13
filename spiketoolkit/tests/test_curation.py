import spikeextractors as se
import numpy as np
from spiketoolkit.curation import threshold_snr, threshold_firing_rate, threshold_isi_violations, \
    threshold_num_spikes, threshold_presence_ratio, threshold_metrics
from spiketoolkit.validation import compute_snrs, compute_firing_rates, compute_num_spikes

def test_thresh_num_spikes():
    rec, sort = se.example_datasets.toy_example(duration=10, num_channels=4, seed=0)
    s_threshold = 25

    sort_ns = threshold_num_spikes(sort, s_threshold, 'less')
    new_ns = compute_num_spikes(sort_ns, rec.get_sampling_frequency())

    assert np.all(new_ns >= s_threshold)

def test_thresh_snr():
    rec, sort = se.example_datasets.toy_example(duration=10, num_channels=4, seed=0)
    snr_thresh = 4

    sort_snr = threshold_snr(sort, rec, snr_thresh, 'less')
    new_snr = compute_snrs(sort_snr, rec)

    assert np.all(new_snr >= snr_thresh)


def test_thresh_fr():
    rec, sort = se.example_datasets.toy_example(duration=10, num_channels=4, seed=0)
    fr_thresh = 2

    sort_fr = threshold_firing_rate(sort, fr_thresh, 'less')
    new_fr = compute_firing_rates(sort_fr)

    assert np.all(new_fr >= fr_thresh)


def test_thresh_metrics():
    rec, sort = se.example_datasets.toy_example(duration=10, num_channels=4, seed=0)
    fr_thresh = 2
    snr_thresh = 4

    sorting_metrics1 = threshold_metrics(
        sort, rec, 
        metrics=['firing_rate', 'snr'],
        thresholds=[fr_thresh, snr_thresh],
        threshold_signs=['less', 'less'],
        mode='or'
    )

    new_fr = compute_firing_rates(sorting_metrics1)
    new_snr  = compute_snrs(sorting_metrics1, rec)

    assert np.all(new_fr >= fr_thresh) and np.all(new_snr >= snr_thresh)

    sorting_metrics1 = threshold_metrics(
        sort, rec, 
        metrics=['firing_rate', 'snr'],
        thresholds=[fr_thresh, snr_thresh],
        threshold_signs=['less', 'less'],
        mode='and'
    )

    new_fr = compute_firing_rates(sorting_metrics1)
    new_snr  = compute_snrs(sorting_metrics1, rec)

    assert np.all((new_fr >= fr_thresh) + (new_snr >= snr_thresh))


if __name__ == '__main__':
    test_thresh_snr()
    test_thresh_fr()
    test_thresh_metrics()
