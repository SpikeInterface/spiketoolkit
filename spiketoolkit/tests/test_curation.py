import spikeextractors as se
import numpy as np
import shutil
from spikeextractors.tests.utils import check_dumping

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
    get_curation_params
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
)


def test_thresh_num_spikes():
    rec, sort = se.example_datasets.toy_example(dump_folder='test', dumpable=True, duration=10, num_channels=4, K=10,
                                                seed=0)
    s_threshold = 25

    sort_ns = threshold_num_spikes(sort, s_threshold, 'less')
    new_ns = compute_num_spikes(sort_ns, sort.get_sampling_frequency())

    assert np.all(new_ns >= s_threshold)
    check_dumping(sort_ns)
    shutil.rmtree('test')


def test_thresh_isi_violations():
    rec, sort = se.example_datasets.toy_example(dump_folder='test', dumpable=True, duration=10, num_channels=4, K=10,
                                                seed=0)
    s_threshold = 0.01

    sort_isi = threshold_isi_violations(sort, s_threshold, 'greater', rec.get_num_frames())
    new_isi = compute_isi_violations(sort_isi, rec.get_num_frames(), sort.get_sampling_frequency())

    assert np.all(new_isi <= s_threshold)
    check_dumping(sort_isi)
    shutil.rmtree('test')


def test_thresh_presence_ratios():
    rec, sort = se.example_datasets.toy_example(dump_folder='test', dumpable=True, duration=10, num_channels=4, K=10,
                                                seed=0)
    s_threshold = 0.18

    sort_pr = threshold_presence_ratios(sort, s_threshold, 'less', rec.get_num_frames())
    new_pr = compute_presence_ratios(sort_pr, rec.get_num_frames(), sort.get_sampling_frequency())

    assert np.all(new_pr >= s_threshold)
    check_dumping(sort_pr)
    shutil.rmtree('test')


def test_thresh_amplitude_cutoffs():
    rec, sort = se.example_datasets.toy_example(dump_folder='test', dumpable=True, duration=10, num_channels=4, K=10,
                                                seed=0)

    amplitude_cutoff_thresh = 0

    sort_amplitude_cutoff = threshold_amplitude_cutoffs(sort, rec, amplitude_cutoff_thresh, "less",
                                                        apply_filter=False, seed=0)
    new_amplitude_cutoff = compute_amplitude_cutoffs(sort_amplitude_cutoff, rec, apply_filter=False, seed=0)

    assert np.all(new_amplitude_cutoff >= amplitude_cutoff_thresh)
    check_dumping(sort_amplitude_cutoff)
    shutil.rmtree('test')


def test_thresh_frs():
    rec, sort = se.example_datasets.toy_example(dump_folder='test', dumpable=True, duration=10, num_channels=4, K=10,
                                                seed=0)
    fr_thresh = 2

    sort_fr = threshold_firing_rates(sort, fr_thresh, 'less', rec.get_num_frames())
    new_fr = compute_firing_rates(sort_fr, rec.get_num_frames())

    assert np.all(new_fr >= fr_thresh)
    check_dumping(sort_fr)
    shutil.rmtree('test')


def test_thresh_threshold_drift_metrics():
    rec, sort = se.example_datasets.toy_example(dump_folder='test', dumpable=True, duration=10, num_channels=4, K=10,
                                                seed=0)
    s_threshold = 1

    sort_max = threshold_drift_metrics(sort, rec, s_threshold, 'greater', metric_name="max_drift",
                                       apply_filter=False, seed=0)
    sort_cum = threshold_drift_metrics(sort, rec, s_threshold, 'greater', metric_name="cumulative_drift",
                                       apply_filter=False, seed=0)
    new_max_drift, _ = compute_drift_metrics(sort_max, rec, apply_filter=False, seed=0)
    _, new_cum_drift = compute_drift_metrics(sort_cum, rec, apply_filter=False, seed=0)

    assert np.all(new_max_drift <= s_threshold)
    assert np.all(new_cum_drift <= s_threshold)
    check_dumping(sort_max)
    check_dumping(sort_cum)
    shutil.rmtree('test')


def test_thresh_snrs():
    rec, sort = se.example_datasets.toy_example(dump_folder='test', dumpable=True, duration=10, num_channels=4, K=10,
                                                seed=0)

    snr_thresh = 4

    sort_snr = threshold_snrs(sort, rec, snr_thresh, 'less', apply_filter=False, seed=0)
    new_snr = compute_snrs(sort_snr, rec, apply_filter=False, seed=0)

    assert np.all(new_snr >= snr_thresh)
    check_dumping(sort_snr)
    shutil.rmtree('test')


# PCA-based
def test_thresh_isolation_distances():
    rec, sort = se.example_datasets.toy_example(dump_folder='test', dumpable=True, duration=10, num_channels=4, K=10,
                                                seed=0)
    s_threshold = 200

    iso = compute_isolation_distances(sort, rec,  apply_filter=False, seed=0)
    sort_iso = threshold_isolation_distances(sort, rec, s_threshold, 'less', apply_filter=False, seed=0)

    original_ids = sort.get_unit_ids()
    new_iso = []
    for unit in sort_iso.get_unit_ids():
        new_iso.append(iso[original_ids.index(unit)])
    new_iso = np.array(new_iso)
    assert np.all(new_iso >= s_threshold)
    check_dumping(sort_iso)
    shutil.rmtree('test')


def test_thresh_silhouettes():
    rec, sort = se.example_datasets.toy_example(dump_folder='test', dumpable=True, duration=10, num_channels=4, K=10,
                                                seed=0)
    silhouette_thresh = .5

    silhouette = compute_silhouette_scores(sort, rec, apply_filter=False, seed=0)
    sort_silhouette = threshold_silhouette_scores(sort, rec, silhouette_thresh, "less", apply_filter=False, seed=0)

    original_ids = sort.get_unit_ids()
    new_silhouette = []
    for unit in sort_silhouette.get_unit_ids():
        new_silhouette.append(silhouette[original_ids.index(unit)])
    new_silhouette = np.array(new_silhouette)
    assert np.all(new_silhouette >= silhouette_thresh)
    check_dumping(sort_silhouette)
    shutil.rmtree('test')


def test_thresh_nn_metrics():
    rec, sort = se.example_datasets.toy_example(dump_folder='test', dumpable=True, duration=10, num_channels=4, K=10,
                                                seed=0)
    s_threshold_hit = 0.9
    s_threshold_miss = 0.002

    nn_hit, nn_miss = compute_nn_metrics(sort, rec, apply_filter=False, seed=0)
    sort_hit = threshold_nn_metrics(sort, rec, s_threshold_hit, 'less', metric_name="nn_hit_rate",
                                    apply_filter=False, seed=0)
    sort_miss = threshold_nn_metrics(sort, rec, s_threshold_miss, 'greater', metric_name="nn_miss_rate",
                                     apply_filter=False, seed=0)

    original_ids = sort.get_unit_ids()
    new_nn_hit = []
    for unit in sort_hit.get_unit_ids():
        new_nn_hit.append(nn_hit[original_ids.index(unit)])
    new_nn_miss = []
    for unit in sort_miss.get_unit_ids():
        new_nn_miss.append(nn_miss[original_ids.index(unit)])
    new_nn_hit = np.array(new_nn_hit)
    new_nn_miss = np.array(new_nn_miss)
    assert np.all(new_nn_hit >= s_threshold_hit)
    assert np.all(new_nn_miss <= s_threshold_miss)
    check_dumping(sort_hit)
    check_dumping(sort_miss)
    shutil.rmtree('test')


def test_thresh_d_primes():
    rec, sort = se.example_datasets.toy_example(dump_folder='test', dumpable=True, duration=10, num_channels=4, K=10,
                                                seed=0)
    d_primes_thresh = .5

    d_primes = compute_d_primes(sort, rec, apply_filter=False, seed=0)
    sort_d_primes = threshold_d_primes(sort, rec, d_primes_thresh, "less", apply_filter=False, seed=0)

    original_ids = sort.get_unit_ids()
    new_d_primes = []
    for unit in sort_d_primes.get_unit_ids():
        new_d_primes.append(d_primes[original_ids.index(unit)])
    new_d_primes = np.array(new_d_primes)
    assert np.all(new_d_primes >= d_primes_thresh)
    check_dumping(sort_d_primes)
    shutil.rmtree('test')


def test_thresh_l_ratios():
    rec, sort = se.example_datasets.toy_example(dump_folder='test', dumpable=True, duration=10, num_channels=4, K=10,
                                                seed=0)
    l_ratios_thresh = 0

    l_ratios = compute_l_ratios(sort, rec, apply_filter=False, seed=0)
    sort_l_ratios = threshold_l_ratios(sort, rec, l_ratios_thresh, "less", apply_filter=False, seed=0)

    original_ids = sort.get_unit_ids()
    new_l_ratios = []
    for unit in sort_l_ratios.get_unit_ids():
        new_l_ratios.append(l_ratios[original_ids.index(unit)])
    new_l_ratios = np.array(new_l_ratios)
    assert np.all(new_l_ratios >= l_ratios_thresh)
    check_dumping(sort_l_ratios)
    shutil.rmtree('test')


def test_curation_params():
    print(get_curation_params())


if __name__ == "__main__":
    test_thresh_num_spikes()
    test_thresh_presence_ratios()
    test_thresh_frs()
    test_thresh_isi_violations()

    test_thresh_snrs()
    test_thresh_amplitude_cutoffs()

    test_thresh_silhouettes()
    test_thresh_isolation_distances()
    test_thresh_l_ratios()
    test_thresh_threshold_drift_metrics()
    test_thresh_nn_metrics()
