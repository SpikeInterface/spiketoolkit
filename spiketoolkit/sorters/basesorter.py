"""
Here a proposal for the futur Sorter with class approach.

The main idea is to decompose all intermediate steps to get more
flexibility:
  * setup the recording (traces, output folder, and so...)
  * set parameters
  * run the sorter (with futur possibility to make it in separate env/container)
  * get the result (SortingExtractor)

One benfit shoudl to compare the "run" time between sorter without
the setup and getting result.

One new idea usefull for tridesclous and maybe other sorter would
a way to adapt params with datasets.


"""

import time
import copy
from pathlib import Path
import threading
import shutil
import os

import spikeextractors as se


class BaseSorter:

    sorter_name = ''  # convinience for reporting
    installed = False  # check at class level if isntalled or not
    SortingExtractor_Class = None  # convinience to get the extractor
    _default_params = {}
    _gui_params = [
        {'name': 'output_folder', 'type': 'str', 'value':None, 'default':None,  'title': "Sorting output folder path"},
        {'name': 'parallel', 'type': 'bool', 'value':False, 'default':False,  'title': "If True, then parallelize"},
    ]
    installation_mesg = ""  # error message when not installed

    def __init__(self, recording=None, output_folder=None, debug=False,
                 grouping_property=None, parallel=False, delete_output_folder=False):

        assert self.installed, """This sorter {} is not installed.
        Please install it with:  \n{} """.format(self.sorter_name, self.installation_mesg)

        self.debug = debug
        self.grouping_property = grouping_property
        self.parallel = parallel
        self.params = self.default_params()

        if output_folder is None:
            output_folder = 'test_' + self.sorter_name
        output_folder = Path(output_folder).absolute()

        if grouping_property is None:
            # only one groups
            self.recording_list = [recording]
            self.output_folders = [output_folder]
        else:
            # several groups
            self.recording_list = se.get_sub_extractors_by_property(recording, grouping_property)
            n_group = len(self.recording_list)
            if n_group > 1:
                self.output_folders = [output_folder / str(i) for i in range(n_group)]
            else:
                self.output_folders = [output_folder / str(0)]

        # make folders
        for output_folder in self.output_folders:
            if not output_folder.is_dir():
                os.makedirs(str(output_folder))
        self.delete_folders = delete_output_folder

    @classmethod
    def gui_params(self):
        return copy.deepcopy(self._gui_params)

    @classmethod
    def default_params(self):
        return copy.deepcopy(self._default_params)

    def set_params(self, **params):
        self.params.update(params)

    def run(self):
        for i, recording in enumerate(self.recording_list):
            self._setup_recording(recording, self.output_folders[i])

        t0 = time.perf_counter()

        if not self.parallel:
            for i, recording in enumerate(self.recording_list):
                self._run(recording, self.output_folders[i])
        else:
            # run in threads
            threads = []
            for i, recording in enumerate(self.recording_list):
                thread = threading.Thread(target=self._run, args=(recording, self.output_folders[i]))
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()

        t1 = time.perf_counter()

        if self.debug:
            print('{} run time {:0.2f}s'.format(self.sorter_name, t1-t0))

        return t1 - t0

    def _setup_recording(self, recording, output_folder):
        # need be iplemented in subclass
        # this setup ONE recording (or SubExtractor)
        raise(NotImplementedError)

    def _run(self, recording, output_folder):
        # need be iplemented in subclass
        # this run the sorter on ONE recording (or SubExtractor)
        raise(NotImplementedError)

    @staticmethod
    def get_result_from_folder(output_folder):
        raise(NotImplementedError)

    def get_result_list(self):
        sorting_list = []
        for i, recording in enumerate(self.recording_list):
            sorting = self.get_result_from_folder(self.output_folders[i])
            sorting_list.append(sorting)
        return sorting_list

    def get_result(self):
        sorting_list = self.get_result_list()
        if len(sorting_list) == 1:
            sorting = sorting_list[0]
        else:
            for i, sorting in enumerate(sorting_list):
                group = self.recording_list[i].get_channel_property(self.recording_list[i].get_channel_ids()[0], 'group')
                if sorting is not None:
                    for unit in sorting.get_unit_ids():
                        sorting.set_unit_property(unit, 'group', group)

            # reassemble the sorting outputs
            sorting_list = [sort for sort in sorting_list if sort is not None]
            multi_sorting = se.MultiSortingExtractor(sortings=sorting_list)
            sorting = multi_sorting

        if self.delete_folders:
            for out in self.output_folders:
                if self.debug:
                    print("Removing ", str(out))
                shutil.rmtree(str(out), ignore_errors=True)
        return sorting

    # new idea
    def get_params_for_particular_recording(self, rec_name):
       """
       this is speculative an nee to be discussed
       """
       return {}
