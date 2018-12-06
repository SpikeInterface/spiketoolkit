from subprocess import Popen, PIPE, CalledProcessError, call
import shlex
import tempfile
import shutil
import threading
import spikeextractors as se
from pathlib import Path
import platform
import sys
from copy import copy


class sortingThread(threading.Thread):
    def __init__(self, threadID, recording, spikesorter, output_folder=None, **kwargs):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.recording = recording
        self.sorter = spikesorter
        self.kwargs = kwargs
        self.kwargs['output_folder'] = output_folder
        print("Thread: ", self.kwargs['output_folder'])

    def run(self):
        from .mountainsort4 import mountainsort4
        from .spyking_circus import spyking_circus
        from .kilosort import kilosort
        from .ironclust import ironclust
        from .klusta import klusta
        print('Sorting ' + str(self.threadID) + ' with ' + self.sorter)
        if self.sorter == 'klusta':
            sorting = klusta(recording=self.recording, **self.kwargs)
        elif self.sorter == 'mountainsort' or self.sorter == 'mountainsort4':
            sorting = mountainsort4(recording=self.recording, **self.kwargs)
        elif self.sorter == 'kilosort':
            sorting = kilosort(recording=self.recording, **self.kwargs)
        elif self.sorter == 'spyking-circus' or self.sorter == 'spyking_circus':
            sorting = spyking_circus(recording=self.recording, **self.kwargs)
        elif self.sorter == 'ironclust':
            sorting = ironclust(recording=self.recording, **self.kwargs)
        else:
            raise ValueError("Spike sorter not supported")
        self.sorting = sorting


def _run_command_and_print_output(command):
    command_list = shlex.split(command, posix="win" not in sys.platform)
    with Popen(command_list, stdout=PIPE, stderr=PIPE) as process:
        while True:
            output_stdout = process.stdout.readline()
            output_stderr = process.stderr.readline()
            if (not output_stdout) and (not output_stderr) and (process.poll() is not None):
                break
            if output_stdout:
                print(output_stdout.decode())
            if output_stderr:
                print(output_stderr.decode())
        rc = process.poll()
        return rc

def _run_command_and_print_output_split(command_list):
    with Popen(command_list, stdout=PIPE, stderr=PIPE) as process:
        while True:
            output_stdout = process.stdout.readline()
            output_stderr = process.stderr.readline()
            if (not output_stdout) and (not output_stderr) and (process.poll() is not None):
                break
            if output_stdout:
                print(output_stdout.decode())
            if output_stderr:
                print(output_stderr.decode())
        rc = process.poll()
        return rc


def _call_command(command):
    command_list = shlex.split(command, posix="win" not in sys.platform)
    try:
        call(command_list)
    except CalledProcessError as e:
        raise Exception(e.output)

def _call_command_split(command_list):
    try:
        call(command_list)
    except CalledProcessError as e:
        raise Exception(e.output)


def _parallelSpikeSorting(recording_list, spikesorter, parallel, **kwargs):
    '''

    Parameters
    ----------
    recording_list
    spikesorter

    Returns
    -------

    '''
    output_folder = copy(kwargs['output_folder'])
    if len(recording_list) == 1:
        print("Only 1 property found!")
    if not isinstance(recording_list[0], se.RecordingExtractor):
        raise ValueError("'recording_list' must be a list of RecordingExtractor objects")
    sorting_list = []
    tmpdir_list = []
    threads = []
    for i, recording in enumerate(recording_list):
        kwargs_copy = copy(kwargs)
        tmpdir = tempfile.mkdtemp()
        tmpdir_list.append(tmpdir)
        if output_folder is None:
            tmpdir = Path(tmpdir).absolute()
            # avoid passing output_folder twice
            if 'output_folder' in kwargs_copy.keys():
                del kwargs_copy['output_folder']
            threads.append(sortingThread(threadID=i, recording=recording, spikesorter=spikesorter,
                                         output_folder=tmpdir, **kwargs_copy))
        else:
            kwargs_copy['output_folder'] = kwargs_copy['output_folder'] + '_' + str(i)
            threads.append(sortingThread(i, recording, spikesorter, **kwargs_copy))
    for t in threads:
        t.start()
        if not parallel:
            t.join()
    if parallel:
        for t in threads:
            t.join()
    for t in threads:
        sorting_list.append(t.sorting)
    for tmp in tmpdir_list:
        shutil.rmtree(tmp)
    return sorting_list


def _spikeSortByProperty(recording, spikesorter, property, parallel, **kwargs):
    '''

    Parameters
    ----------
    recording
    sorter
    kwargs

    Returns
    -------

    '''
    recording_list = se.getSubExtractorsByProperty(recording, property)
    sorting_list = _parallelSpikeSorting(recording_list, spikesorter, parallel, **kwargs)
    # add group property
    for i, sorting in enumerate(sorting_list):
        group = recording_list[i].getChannelProperty(recording_list[i].getChannelIds()[0], 'group')
        for unit in sorting.getUnitIds():
            sorting.setUnitProperty(unit, 'group', group)
    # reassemble the sorting outputs
    multi_sorting = se.MultiSortingExtractor(sortings=sorting_list)
    return multi_sorting
