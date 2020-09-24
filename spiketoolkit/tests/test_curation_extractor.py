import numpy as np
import os, sys
import unittest
import tempfile
import shutil
#
#
# def append_to_path(dir0):  # A convenience function
#     if dir0 not in sys.path:
#         sys.path.append(dir0)
#
#
# append_to_path(os.getcwd() + '/..')
import spikeextractors as se
import spiketoolkit as st
from spikeextractors.tests.utils import check_dumping


def test_curation_sorting_extractor():
    rec, sort = se.example_datasets.toy_example(dump_folder='test', dumpable=True, duration=10, num_channels=4,
                                                K=3, seed=0)

    # Dummy features for testing merging and splitting of features
    sort.set_unit_spike_features(1, 'f_int', range(0 + 1, len(sort.get_unit_spike_train(1)) + 1))
    sort.set_unit_spike_features(2, 'f_int', range(0, len(sort.get_unit_spike_train(2))))
    sort.set_unit_spike_features(2, 'bad_features', np.repeat(1, len(sort.get_unit_spike_train(2))))
    sort.set_unit_spike_features(3, 'f_int', range(0, len(sort.get_unit_spike_train(3))))

    CSX = st.curation.CurationSortingExtractor(parent_sorting=sort)
    merged_unit_id = CSX.merge_units(unit_ids=[1, 2])
    assert np.allclose(merged_unit_id, 4)
    original_spike_train = np.concatenate((sort.get_unit_spike_train(1), sort.get_unit_spike_train(2)))
    indices_sort = np.argsort(original_spike_train)
    original_spike_train = original_spike_train[indices_sort]
    original_features = np.concatenate(
        (sort.get_unit_spike_features(1, 'f_int'), sort.get_unit_spike_features(2, 'f_int')))
    original_features = original_features[indices_sort]
    assert np.allclose(CSX.get_unit_spike_train(4), original_spike_train)
    assert np.allclose(CSX.get_unit_spike_features(4, 'f_int'), original_features)
    assert CSX.get_unit_spike_feature_names(4) == ['f_int']
    assert np.allclose(CSX.get_sampling_frequency(), sort.get_sampling_frequency())

    unit_ids_split = CSX.split_unit(unit_id=3, indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert np.allclose(unit_ids_split[0], 5)
    assert np.allclose(unit_ids_split[1], 6)
    original_spike_train = sort.get_unit_spike_train(3)
    original_features = sort.get_unit_spike_features(3, 'f_int')
    split_spike_train_1 = CSX.get_unit_spike_train(5)
    split_spike_train_2 = CSX.get_unit_spike_train(6)
    split_features_1 = CSX.get_unit_spike_features(5, 'f_int')
    split_features_2 = CSX.get_unit_spike_features(6, 'f_int')
    assert np.allclose(original_spike_train[:10], split_spike_train_1)
    assert np.allclose(original_spike_train[10:], split_spike_train_2)
    assert np.allclose(original_features[:10], split_features_1)
    assert np.allclose(original_features[10:], split_features_2)

    check_dumping(CSX)


if __name__ == '__main__':
    test_curation_sorting_extractor()
