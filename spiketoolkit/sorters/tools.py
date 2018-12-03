from subprocess import Popen, PIPE, CalledProcessError, call
import shlex
import tempfile
import shutil
import threading
import spikeextractors as se


class sortingThread(threading.Thread):
    def __init__(self, threadID, recording, spikesorter, **kwargs):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.recording = recording
        self.sorter = spikesorter
        self.kwargs = kwargs

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
    with Popen(shlex.split(command), stdout=PIPE, stderr=PIPE) as process:
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
    try:
        call(shlex.split(command))
    except subprocess.CalledProcessError as e:
        raise Exception(e.output)


def _parallelSpikeSorting(recording_list, spikesorter, **kwargs):
    '''

    Parameters
    ----------
    recording_list
    spikesorter

    Returns
    -------

    '''
    if len(recording_list) > 1:
        if not isinstance(recording_list[0], se.RecordingExtractor):
            raise ValueError("'recording_list' must be a list of RecordingExtractor objects")
        sorting_list = []
        tmpdir_list = []
        threads = []
        for i, recording in enumerate(recording_list):
            tmpdir = tempfile.mkdtemp()
            tmpdir_list.append(tmpdir)
            if kwargs['output_folder'] is None:
                kwargs['output_folder']=os.path.abspath('.' + tmpdir)
                threads.append(sortingThread(i, recording, spikesorter, **kwargs))
            else:
                threads.append(sortingThread(i, recording, spikesorter, **kwargs))

        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for t in threads:
            sorting_list.append(t.sorting)
        for tmp in tmpdir_list:
            shutil.rmtree(tmp)
    else:
        raise ValueError("'recording_list' must have more than 1 element")
    return sorting_list


def _spikeSortByProperty(recording, spikesorter, property, **kwargs):
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
    sorting_list = _parallelSpikeSorting(recording_list, spikesorter, **kwargs)
    # add group property
    for i, sorting in enumerate(sorting_list):
        group = recording_list[i].getChannelProperty(recording_list[i].getChannelIds()[0], 'group')
        for unit in sorting.getUnitIds():
            sorting.setUnitProperty(unit, 'group', group)
    # reassemble the sorting outputs
    multi_sorting = se.MultiSortingExtractor(sortings=sorting_list)
    return multi_sorting
