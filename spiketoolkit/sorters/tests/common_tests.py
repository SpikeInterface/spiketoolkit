import spikeextractors as se


class SorterCommonTestSuite:
    """
    This class run some basic for a sorter class.
    This is the minimal test suite for each sorter class:
      * run once
      * run with several groups
      * run with several groups in thread
    """
    SorterCLass = None
    
    def test_on_toy(self):
        recording, sorting_gt = se.example_datasets.toy_example1(num_channels=4, duration=30)
        
        params = self.SorterCLass.default_params()
        
        sorter = self.SorterCLass(recording=recording, output_folder=None,
                                        grouping_property=None, parallel=False, debug=True)
        sorter.set_params(**params)
        sorter.run()
        sorting = sorter.get_result()
        
        for unit_id in sorting.getUnitIds():
            print('unit #', unit_id, 'nb', len(sorting.getUnitSpikeTrain(unit_id)))
    
    def test_several_groups(self):
        # run sorter with several groups in paralel or not
        recording, sorting_gt = se.example_datasets.toy_example1(num_channels=8, duration=30)
        
        # make 2 artificial groups
        for ch_id in range(0, 4):
            recording.setChannelProperty(ch_id, 'group', 0)
        for ch_id in range(4, 8):
            recording.setChannelProperty(ch_id, 'group', 1)
         
        
        params = self.SorterCLass.default_params()
        
        for parallel in [False, True]:
            sorter = self.SorterCLass(recording=recording, output_folder=None,
                                            grouping_property='group', parallel=parallel, debug=True)
            sorter.set_params(**params)
            sorter.run()
            sorting = sorter.get_result()
            
        
    