import spikeextractors as se
import numpy as np

from spiketoolkit.curation import (
    threshold_snrs,
    threshold_silhouette_scores,
    threshold_d_primes,
    threshold_firing_rates,
    threshold_isi_violations,
    threshold_num_spikes,
    threshold_presence_ratios,
    threshold_l_ratios,
    threshold_amplitude_cutoffs,
    threshold_isolation_distances,
    threshold_nn_metrics,
    threshold_drift_metrics,
    get_kwargs_params
)


from spiketoolkit.validation import (
    compute_num_spikes,
    compute_firing_rates,
    compute_presence_ratios,
    compute_isi_violations,
    compute_amplitude_cutoffs,
    compute_snrs,
    compute_drift_metrics,
    compute_silhouette_scores,
    compute_isolation_distances,
    compute_l_ratios,
    compute_d_primes,
    compute_nn_metrics,
    compute_metrics,
)

from spiketoolkit.tests.utils import check_dumping, create_dumpable_sorting, create_dumpable_extractors


def test_thresh_num_spikes():
    sort = create_dumpable_sorting(duration=10, num_channels=4, K=10, seed=0, folder='test')
    s_threshold = 25

    sort_ns = threshold_num_spikes(sort, s_threshold, 'less')
    new_ns = compute_num_spikes(sort_ns, sort.get_sampling_frequency())[0]

    assert np.all(new_ns >= s_threshold)
    check_dumping(sort_ns)


def test_thresh_snrs():
    rec, sort = create_dumpable_extractors(duration=10, num_channels=4, K=10, seed=0, folder='test')

    snr_thresh = 4

    sort_snr = threshold_snrs(sort, rec, snr_thresh, 'less')
    new_snr = compute_snrs(sort_snr, rec)[0]

    assert np.all(new_snr >= snr_thresh)
    check_dumping(sort_snr)


def test_thresh_silhouettes():
    rec, sort = create_dumpable_extractors(duration=10, num_channels=4, K=10, seed=0, folder='test')

    silhouette_thresh = .5

    sort_silhouette = threshold_silhouette_scores(sort, rec, silhouette_thresh, "less", apply_filter=False)
    silhouette = np.asarray(compute_silhouette_scores(sort, rec, apply_filter=False)[0])
    new_silhouette = silhouette[np.where(silhouette >= silhouette_thresh)]

    assert np.all(new_silhouette >= silhouette_thresh)
    check_dumping(sort_silhouette)


def test_thresh_d_primes():
    rec, sort = create_dumpable_extractors(duration=10, num_channels=4, K=10, seed=0, folder='test')

    d_primes_thresh = .5

    sort_d_primes = threshold_d_primes(sort, rec, d_primes_thresh, "less", apply_filter=False)
    new_d_primes = compute_d_primes(sort_d_primes, rec)[0]

    assert np.all(new_d_primes >= d_primes_thresh)
    check_dumping(sort_d_primes)


def test_thresh_l_ratios():
    rec, sort = create_dumpable_extractors(duration=10, num_channels=4, K=10, seed=0, folder='test')

    l_ratios_thresh = 0

    sort_l_ratios = threshold_l_ratios(sort, rec, l_ratios_thresh, "less", apply_filter=False)
    new_l_ratios = compute_l_ratios(sort_l_ratios, rec)[0]

    assert np.all(new_l_ratios >= l_ratios_thresh)
    check_dumping(sort_l_ratios)


def test_thresh_amplitude_cutoffs():
    rec, sort = create_dumpable_extractors(duration=10, num_channels=4, K=10, seed=0, folder='test')

    amplitude_cutoff_thresh = 0

    sort_amplitude_cutoff = threshold_amplitude_cutoffs(sort, rec, amplitude_cutoff_thresh, "less", apply_filter=False)
    new_amplitude_cutoff = compute_amplitude_cutoffs(sort_amplitude_cutoff, rec)[0]

    assert np.all(new_amplitude_cutoff >= amplitude_cutoff_thresh)
    check_dumping(sort_amplitude_cutoff)


def test_thresh_frs():
    sort = create_dumpable_sorting(duration=10, num_channels=4, K=10, seed=0, folder='test')
    fr_thresh = 2

    sort_fr = threshold_firing_rates(sort, fr_thresh, 'less')
    new_fr = compute_firing_rates(sort_fr)[0]

    assert np.all(new_fr >= fr_thresh)
    check_dumping(sort_fr)

def test_kwarg_params():
    print(get_kwargs_params())


if __name__ == "__main__":
    test_thresh_silhouettes()
    # test_thresh_snrs()
    # test_thresh_frs()
    # test_thresh_amplitude_cutoffs()
    # test_thresh_silhouettes()
    # test_thresh_l_ratios()
    # test_thresh_snrs()
    # test_thresh_num_spikes()
