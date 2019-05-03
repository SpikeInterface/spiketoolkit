import unittest
import spikeextractors as se


class SorterCommonTestSuite:
    """
    This class run some basic for a sorter class.
    This is the minimal test suite for each sorter class:
      * run once
      * run with several groups
      * run with several groups in thread
    """
    SorterClass = None

    def test_on_toy(self):

        recording, sorting_gt = se.example_datasets.toy_example(num_channels=4, duration=60)

        params = self.SorterClass.default_params()

        sorter = self.SorterClass(recording=recording, output_folder=None,
                                  grouping_property=None, parallel=False, debug=False)
        sorter.set_params(**params)
        sorter.run()
        sorting = sorter.get_result()

        for unit_id in sorting.get_unit_ids():
            print('unit #', unit_id, 'nb', len(sorting.get_unit_spike_train(unit_id)))
        del sorting

    def test_several_groups(self):

        # run sorter with several groups in paralel or not
        recording, sorting_gt = se.example_datasets.toy_example(num_channels=8, duration=30)

        # make 2 artificial groups
        for ch_id in range(0, 4):
            recording.set_channel_property(ch_id, 'group', 0)
        for ch_id in range(4, 8):
            recording.set_channel_property(ch_id, 'group', 1)


        params = self.SorterClass.default_params()

        for parallel in [False, True]:
            sorter = self.SorterClass(recording=recording, output_folder=None,
                                      grouping_property='group', parallel=parallel, debug=False)
            sorter.set_params(**params)
            sorter.run()
            sorting = sorter.get_result()
            del sorting
    
    def test_with_BinDatRecordingExtractor(self):
        # some sorter (TDC, KS, KS2, ...) work by default with the raw binary
        # format as input to avoid copy when the recording is already this format
        
        recording, sorting_gt = se.example_datasets.toy_example(num_channels=2, duration=10)

        # create a raw dat file and prb file
        raw_filename = 'raw_file.dat'
        prb_filename = 'raw_file.prb'

        samplerate = recording.get_sampling_frequency()
        traces = recording.get_traces().astype('float32')
        with open(raw_filename, mode='wb') as f:
            f.write(traces.T.tobytes())

        se.save_probe_file(recording, prb_filename, format='spyking_circus')

        recording = se.BinDatRecordingExtractor(raw_filename, samplerate, 2, 'float32', frames_first=True, offset=0)
        se.load_probe_file(recording, prb_filename)
        
        params = self.SorterClass.default_params()
        sorter = self.SorterClass(recording=recording, output_folder=None)
        sorter.set_params(**params)
        sorter.run()
        sorting = sorter.get_result()
        
