import spikeextractors as se
import numpy as np
from spiketoolkit.validation import compute_isolation_distances, compute_isi_violations, compute_snrs, \
    compute_amplitude_cutoffs, compute_d_primes, compute_drift_metrics, compute_firing_rates, compute_l_ratios, \
    compute_quality_metrics, compute_nn_metrics, compute_num_spikes, compute_presence_ratios, compute_silhouette_scores, \
    get_validation_params


def test_functions():
    rec, sort = se.example_datasets.toy_example(duration=10, num_channels=4, seed=0)

    firing_rates = compute_firing_rates(sort, rec.get_num_frames(), seed=0)
    num_spikes = compute_num_spikes(sort, seed=0)
    isi = compute_isi_violations(sort, rec.get_num_frames(), seed=0)
    presence = compute_presence_ratios(sort, rec.get_num_frames(), seed=0)
    amp_cutoff = compute_amplitude_cutoffs(sort, rec, seed=0)
    max_drift, cum_drift = compute_drift_metrics(sort, rec, seed=0, memmap=False)
    silh = compute_silhouette_scores(sort, rec, seed=0)
    iso = compute_isolation_distances(sort, rec, seed=0)
    l_ratio = compute_l_ratios(sort, rec, seed=0)
    dprime = compute_d_primes(sort, rec, seed=0)
    nn_hit, nn_miss = compute_nn_metrics(sort, rec, seed=0)
    snr = compute_snrs(sort, rec, seed=0)
    metrics = compute_quality_metrics(sort, rec, return_dict=True, seed=0)

    assert np.allclose(metrics['firing_rate'], firing_rates)
    assert np.allclose(metrics['num_spikes'], num_spikes)
    assert np.allclose(metrics['isi_violation'], isi)
    assert np.allclose(metrics['amplitude_cutoff'], amp_cutoff)
    assert np.allclose(metrics['presence_ratio'], presence)
    assert np.allclose(metrics['silhouette_score'], silh)
    assert np.allclose(metrics['isolation_distance'], iso)
    assert np.allclose(metrics['l_ratio'], l_ratio)
    assert np.allclose(metrics['d_prime'], dprime)
    assert np.allclose(metrics['snr'], snr)
    assert np.allclose(metrics['max_drift'], max_drift)
    assert np.allclose(metrics['cumulative_drift'], cum_drift)
    assert np.allclose(metrics['nn_hit_rate'], nn_hit)
    assert np.allclose(metrics['nn_miss_rate'], nn_miss)


def test_validation_params():
    print(get_validation_params())


if __name__ == '__main__':
    test_functions()