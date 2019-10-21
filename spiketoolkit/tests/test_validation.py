import spikeextractors as se
from numpy.testing import assert_array_equal
from spiketoolkit.validation import compute_isolation_distances, compute_isi_violations, compute_snrs, \
    compute_amplitude_cutoffs, compute_d_primes, compute_drift_metrics, compute_firing_rates, compute_l_ratios, \
    compute_metrics, compute_nn_metrics, compute_num_spikes, compute_presence_ratios, compute_silhouette_scores, \
    MetricCalculator


def test_calculator():
    rec, sort = se.example_datasets.toy_example(duration=10, num_channels=4)
    mc = MetricCalculator(sort, rec)
    mc.compute_all_metric_data()

    _ = mc.compute_metrics()
    metric_dict = mc.get_metrics_dict()
    assert type(mc._recording._recording).__name__ == 'BandpassFilterRecording' #check if filter by default
    assert 'firing_rate' in metric_dict.keys()
    assert 'num_spikes' in metric_dict.keys()
    assert 'isi_viol' in metric_dict.keys()
    assert 'presence_ratio' in metric_dict.keys()
    assert 'amplitude_cutoff' in metric_dict.keys()
    assert 'max_drift' in metric_dict.keys()
    assert 'cumulative_drift' in metric_dict.keys()
    assert 'silhouette_score' in metric_dict.keys()
    assert 'isolation_distance' in metric_dict.keys()
    assert 'l_ratio' in metric_dict.keys()
    assert 'd_prime' in metric_dict.keys()
    assert 'nn_hit_rate' in metric_dict.keys()
    assert 'nn_miss_rate' in metric_dict.keys()
    assert 'snr' in metric_dict.keys()
    assert mc.is_filtered()


def test_functions():
    rec, sort = se.example_datasets.toy_example(duration=10, num_channels=4)

    firing_rates = compute_firing_rates(sort)
    num_spikes = compute_num_spikes(sort)
    isi = compute_isi_violations(sort)
    presence = compute_presence_ratios(sort)

    amp_cutoff = compute_amplitude_cutoffs(sort, rec)

    max_drift, cum_drift = compute_drift_metrics(sort, rec)
    silh = compute_silhouette_scores(sort, rec)
    iso = compute_isolation_distances(sort, rec)
    l_ratio = compute_l_ratios(sort, rec)
    dprime = compute_d_primes(sort, rec)
    nn_hit, nn_miss = compute_nn_metrics(sort, rec)

    snr = compute_snrs(sort, rec)
