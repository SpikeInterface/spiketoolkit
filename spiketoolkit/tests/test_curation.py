import spikeextractors as se
import numpy as np
from spiketoolkit.curation import (
    threshold_snr,
    threshold_silhouette_score,
    threshold_firing_rate,
    threshold_isi_violations,
    threshold_num_spikes,
    threshold_presence_ratio,
)
from spiketoolkit.validation import (
    compute_snrs,
    compute_silhouette_scores,
    compute_firing_rates,
    compute_num_spikes,
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

    sort_silhouette = threshold_silhouette_score(
        sort, rec, silhouette_thresh, "less"
    )
    silhouette = np.asarray(compute_silhouette_scores(sort, rec)[0])
    new_silhouette = silhouette[np.where(silhouette >= silhouette_thresh)]

    assert np.all(new_silhouette >= silhouette_thresh)


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
