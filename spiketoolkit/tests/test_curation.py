import spikeextractors as se
import numpy as np
from spiketoolkit.curation import threshold_snr, threshold_firing_rate, threshold_isi_violations, \
    threshold_num_spikes, threshold_presence_ratio
from spiketoolkit.validation import compute_snrs, compute_firing_rates


def test_thresh_snr():
    rec, sort = se.example_datasets.toy_example(duration=10, num_channels=4)
    snr_thresh = 4

    sort_snr = threshold_snr(sort, rec, snr_thresh, 'less')
    new_snr = compute_snrs(sort_snr, rec)

    assert np.all(new_snr > snr_thresh)


def test_thresh_fr():
    rec, sort = se.example_datasets.toy_example(duration=10, num_channels=4)
    fr_thresh = 2

    sort_fr = threshold_firing_rate(sort, fr_thresh, 'less')
    new_fr = compute_firing_rates(sort_fr)

    assert np.all(new_fr > fr_thresh)


if __name__ == '__main__':
    test_thresh_snr()
    test_thresh_fr()
