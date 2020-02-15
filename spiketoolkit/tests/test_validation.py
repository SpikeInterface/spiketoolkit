import spikeextractors as se
from numpy.testing import assert_array_equal
from spiketoolkit.validation import compute_isolation_distances, compute_isi_violations, compute_snrs, \
    compute_amplitude_cutoffs, compute_d_primes, compute_drift_metrics, compute_firing_rates, compute_l_ratios, \
    compute_metrics, compute_nn_metrics, compute_num_spikes, compute_presence_ratios, compute_silhouette_scores \

def test_functions():
    rec, sort = se.example_datasets.toy_example(duration=10, num_channels=4, seed=0)

    firing_rates = compute_firing_rates(sort)
    num_spikes = compute_num_spikes(sort)
    isi = compute_isi_violations(sort)
    presence = compute_presence_ratios(sort)

    amp_cutoff = compute_amplitude_cutoffs(sort, rec)

    max_drift, cum_drift = compute_drift_metrics(sort, rec)[0]
    silh = compute_silhouette_scores(sort, rec)
    iso = compute_isolation_distances(sort, rec)
    l_ratio = compute_l_ratios(sort, rec)
    dprime = compute_d_primes(sort, rec)
    nn_hit, nn_miss = compute_nn_metrics(sort, rec)[0]

    snr = compute_snrs(sort, rec)
