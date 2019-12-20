import spikeextractors as se
import numpy as np

from spiketoolkit.curation import (
    threshold_snr,
    threshold_silhouette_score,
    threshold_d_primes,
    threshold_firing_rate,
    # threshold_isi_violations,
    threshold_num_spikes,
    # threshold_presence_ratio,
    threshold_l_ratios,
    threshold_amplitude_cutoff,
)
from spiketoolkit.validation import (
    compute_snrs,
    compute_silhouette_scores,
    compute_d_primes,
    compute_l_ratios,
    compute_firing_rates,
    compute_num_spikes,
    compute_amplitude_cutoffs,
)


def test_thresh_num_spikes():
    rec, sort = se.example_datasets.toy_example(
        duration=10,
        num_channels=4,
        seed=0
    )
    s_threshold = 25

    sort_ns = threshold_num_spikes(sort, s_threshold, 'less')
    new_ns = compute_num_spikes(sort_ns, rec.get_sampling_frequency())[0]

    assert np.all(new_ns >= s_threshold)


def test_thresh_snr():
    rec, sort = se.example_datasets.toy_example(
        duration=10, num_channels=4, seed=0
    )
    snr_thresh = 4

    sort_snr = threshold_snr(sort, rec, snr_thresh, 'less')
    new_snr = compute_snrs(sort_snr, rec)[0]

    assert np.all(new_snr >= snr_thresh)


def test_thresh_silhouette():
    rec, sort = se.example_datasets.toy_example(
        duration=10, num_channels=4, seed=0
    )
    silhouette_thresh = .5

    _ = threshold_silhouette_score(
        sort, rec, silhouette_thresh, "less"
    )
    silhouette = np.asarray(compute_silhouette_scores(sort, rec)[0])
    new_silhouette = silhouette[np.where(silhouette >= silhouette_thresh)]

    assert np.all(new_silhouette >= silhouette_thresh)


def test_thresh_d_primes():
    rec, sort = se.example_datasets.toy_example(
        duration=10, num_channels=4, seed=0
    )
    d_primes_thresh = .5

    sort_d_primes = threshold_d_primes(
        sort, rec, d_primes_thresh, "less"
    )
    new_d_primes = compute_d_primes(sort_d_primes, rec)[0]

    assert np.all(new_d_primes >= d_primes_thresh)


def test_thresh_l_ratios():
    rec, sort = se.example_datasets.toy_example(
        duration=10, num_channels=4, seed=0
    )
    l_ratios_thresh = 0

    sort_l_ratios = threshold_l_ratios(
        sort, rec, l_ratios_thresh, "less"
    )
    new_l_ratios = compute_l_ratios(sort_l_ratios, rec)[0]

    assert np.all(new_l_ratios >= l_ratios_thresh)


def test_thresh_amplitude_cutoff():
    rec, sort = se.example_datasets.toy_example(
        duration=10, num_channels=4, seed=0
    )
    amplitude_cutoff_thresh = 0

    sort_amplitude_cutoff = threshold_amplitude_cutoff(
        sort, rec, amplitude_cutoff_thresh, "less"
    )
    new_amplitude_cutoff = compute_amplitude_cutoffs(sort_amplitude_cutoff, rec)[0]

    assert np.all(new_amplitude_cutoff >= amplitude_cutoff_thresh)


def test_thresh_fr():
    rec, sort = se.example_datasets.toy_example(
        duration=10, num_channels=4, seed=0
     )
    fr_thresh = 2

    sort_fr = threshold_firing_rate(sort, fr_thresh, 'less')
    new_fr = compute_firing_rates(sort_fr)[0]

    assert np.all(new_fr >= fr_thresh)


if __name__ == "__main__":
    test_thresh_silhouette()
    test_thresh_snr()
    test_thresh_fr()
