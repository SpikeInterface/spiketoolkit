import numpy as np
import os, sys
import unittest
import tempfile
import shutil


def append_to_path(dir0):  # A convenience function
    if dir0 not in sys.path:
        sys.path.append(dir0)


append_to_path(os.getcwd() + '/..')
import spikeextractors as se
import spiketoolkit as st


class TestCuration(unittest.TestCase):
    def setUp(self):
        self.RX, self.SX, self.SX2, self.example_info = self._create_example()
        self.test_dir = tempfile.mkdtemp()
        # self.test_dir = '.'

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)
        # pass

    def _create_example(self):
        channel_ids = [0, 1, 2, 3]
        num_channels = 4
        num_frames = 10000
        sampling_frequency = 30000
        X = np.random.normal(0, 1, (num_channels, num_frames))
        geom = np.random.normal(0, 1, (num_channels, 2))
        X = (X * 100).astype(int)
        RX = se.NumpyRecordingExtractor(timeseries=X, sampling_frequency=sampling_frequency, geom=geom)
        SX = se.NumpySortingExtractor()
        spike_times = [200, 300, 400]
        train1 = np.sort(np.rint(np.random.uniform(0, num_frames, spike_times[0])).astype(int))
        SX.add_unit(unit_id=1, times=train1)
        SX.add_unit(unit_id=2, times=np.sort(np.random.uniform(0, num_frames, spike_times[1])))
        SX.add_unit(unit_id=3, times=np.sort(np.random.uniform(0, num_frames, spike_times[2])))
        SX.set_unit_property(unit_id=1, property_name='stablility', value=80)
        SX.set_sampling_frequency(sampling_frequency)
        SX2 = se.NumpySortingExtractor()
        spike_times2 = [100, 150, 450]
        train2 = np.rint(np.random.uniform(0, num_frames, spike_times[0])).astype(int)
        SX2.add_unit(unit_id=3, times=train2)
        SX2.add_unit(unit_id=4, times=np.random.uniform(0, num_frames, spike_times2[1]))
        SX2.add_unit(unit_id=5, times=np.random.uniform(0, num_frames, spike_times2[2]))
        SX2.set_unit_property(unit_id=4, property_name='stablility', value=80)
        RX.set_channel_property(channel_id=0, property_name='location', value=(0, 0))
        example_info = dict(
            channel_ids=channel_ids,
            num_channels=num_channels,
            num_frames=num_frames,
            sampling_frequency=sampling_frequency,
            unit_ids=[1, 2, 3],
            train1=train1,
            unit_prop=80,
            channel_prop=(0, 0)
        )

        return (RX, SX, SX2, example_info)

    def test_curation_sorting_extractor(self):
        #Dummy features for testing merging and splitting of features
        self.SX.set_unit_spike_features(1, 'f_int', range(0 + 1, len(self.SX.get_unit_spike_train(1)) + 1))
        self.SX.set_unit_spike_features(2, 'f_int', range(0, len(self.SX.get_unit_spike_train(2))))
        self.SX.set_unit_spike_features(2, 'bad_features', np.repeat(1, len(self.SX.get_unit_spike_train(2))))
        self.SX.set_unit_spike_features(3, 'f_int', range(0, len(self.SX.get_unit_spike_train(3))))

        CSX = st.curation.CurationSortingExtractor(
            parent_sorting=self.SX
        )
        CSX.merge_units(unit_ids=[1, 2])
        original_spike_train = np.concatenate((self.SX.get_unit_spike_train(1), self.SX.get_unit_spike_train(2)))
        indices_sort = np.argsort(original_spike_train)
        original_spike_train = original_spike_train[indices_sort]
        original_features = np.concatenate((self.SX.get_unit_spike_features(1, 'f_int'), self.SX.get_unit_spike_features(2, 'f_int')))
        original_features = original_features[indices_sort]
        self.assertTrue(np.array_equal(CSX.get_unit_spike_train(4), original_spike_train))
        self.assertTrue(np.array_equal(CSX.get_unit_spike_features(4, 'f_int'), original_features))
        self.assertTrue(np.array_equal(np.asarray(CSX.get_unit_spike_feature_names(4)), np.asarray(['f_int'])))
        self.assertEqual(CSX.get_sampling_frequency(), self.SX.get_sampling_frequency())

        CSX.split_unit(unit_id=3, indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        original_spike_train = self.SX.get_unit_spike_train(3)
        original_features = self.SX.get_unit_spike_features(3, 'f_int')
        split_spike_train_1 = CSX.get_unit_spike_train(5)
        split_spike_train_2 = CSX.get_unit_spike_train(6)
        split_features_1 = CSX.get_unit_spike_features(5, 'f_int')
        split_features_2 = CSX.get_unit_spike_features(6, 'f_int')
        self.assertTrue(np.array_equal(original_spike_train[:10], split_spike_train_1))
        self.assertTrue(np.array_equal(original_spike_train[10:], split_spike_train_2))
        self.assertTrue(np.array_equal(original_features[:10], split_features_1))
        self.assertTrue(np.array_equal(original_features[10:], split_features_2))

if __name__ == '__main__':
    unittest.main()