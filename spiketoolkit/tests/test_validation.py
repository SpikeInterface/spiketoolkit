import spikeextractors as se
import numpy as np
from spiketoolkit.validation import compute_isolation_distances, compute_isi_violations, compute_snrs, \
    compute_amplitude_cutoffs, compute_d_primes, compute_drift_metrics, compute_firing_rates, compute_l_ratios, \
    compute_metrics, compute_nn_metrics, compute_num_spikes, compute_presence_ratios, compute_silhouette_scores, \
    get_validation_params


def test_functions():
    rec, sort = se.example_datasets.toy_example(duration=10, num_channels=4, seed=0)

    firing_rates = compute_firing_rates(sort, seed=0)[0]
    num_spikes = compute_num_spikes(sort, seed=0)[0]
    isi = compute_isi_violations(sort, seed=0)[0]
    presence = compute_presence_ratios(sort, seed=0)[0]
    amp_cutoff = compute_amplitude_cutoffs(sort, rec, seed=0)[0]
    max_drift, cum_drift = compute_drift_metrics(sort, rec, seed=0, memmap=False)[0]
    silh = compute_silhouette_scores(sort, rec, seed=0)[0]
    iso = compute_isolation_distances(sort, rec, seed=0)[0]
    l_ratio = compute_l_ratios(sort, rec, seed=0)[0]
    dprime = compute_d_primes(sort, rec, seed=0)[0]
    nn_hit, nn_miss = compute_nn_metrics(sort, rec, seed=0)[0]
    snr = compute_snrs(sort, rec, seed=0)[0]
    metrics = compute_metrics(sort, rec, return_dict=True, seed=0)

    assert np.allclose(metrics['firing_rate'][0], firing_rates)
    assert np.allclose(metrics['num_spikes'][0], num_spikes)
    assert np.allclose(metrics['isi_viol'][0], isi)
    assert np.allclose(metrics['amplitude_cutoff'][0], amp_cutoff)
    assert np.allclose(metrics['presence_ratio'][0], presence)
    assert np.allclose(metrics['silhouette_score'][0], silh)
    assert np.allclose(metrics['isolation_distance'][0], iso)
    assert np.allclose(metrics['l_ratio'][0], l_ratio)
    assert np.allclose(metrics['d_prime'][0], dprime)
    assert np.allclose(metrics['snr'][0], snr)
    assert np.allclose(metrics['max_drift'][0], max_drift)
    assert np.allclose(metrics['cumulative_drift'][0], cum_drift)
    assert np.allclose(metrics['nn_hit_rate'][0], nn_hit)
    assert np.allclose(metrics['nn_miss_rate'][0], nn_miss)


def test_validation_params():
    print(get_validation_params())


if __name__ == '__main__':
    test_functions()