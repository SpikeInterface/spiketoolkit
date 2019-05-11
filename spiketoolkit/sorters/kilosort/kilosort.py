"""
Important note.
For facilities
kilosort_path
npy_matlab_path have been removed from args
so only the
  klp = os.getenv('KILOSORT_PATH')
  npp = os.getenv('NPY_MATLAB_PATH')
is left.

We will be able to add this with a class method


"""
from pathlib import Path
import os
import sys
import shutil
import numpy as np

from spiketoolkit.sorters.basesorter import BaseSorter
import spikeextractors as se
from ..sorter_tools import _call_command_split


def check_if_installed(kilosort_path, npy_matlab_path):
    if kilosort_path is None or npy_matlab_path is None:
        return False

    if kilosort_path is not None and kilosort_path.startswith('"'):
        kilosort_path = kilosort_path[1:-1]
        kilosort_path = Path(kilosort_path).absolut()

    if npy_matlab_path is not None and npy_matlab_path.startswith('"'):
        npy_matlab_path = npy_matlab_path[1:-1]
        npy_matlab_path = Path(npy_matlab_path).absolut()

    if (Path(kilosort_path) / 'preprocessData.m').is_file() \
            or not (Path(npy_matlab_path) / 'npy-matlab' / 'readNPY.m').is_file():
        return True
    else:
        return False


if check_if_installed(os.getenv('KILOSORT_PATH'), os.getenv('NPY_MATLAB_PATH')):
    HAVE_KILOSORT = True
else:
    HAVE_KILOSORT = False


class KilosortSorter(BaseSorter):
    """


    """

    sorter_name = 'kilosort'
    installed = HAVE_KILOSORT
    kilosort_path = os.getenv('KILOSORT_PATH')
    npy_matlab_path = os.getenv('NPY_MATLAB_PATH')

    _default_params = {
        'probe_file': None,
        'useGPU': True,
        'detect_threshold': 6,
        'car': True,
        'electrode_dimensions': None,
        'npy_matlab_path': None,
        'kilosort_path': None
    }

    installation_mesg = """\nTo use Kilosort run:\n
        >>> git clone https://github.com/cortex-lab/KiloSort
        >>> git clone https://github.com/kwikteam/npy-matlab\n
    and provide the installation path with the 'kilosort_path' and
    npy_matlab_path' arguments or by setting the KILOSORT2_PATH
    and NPY_MATLAB_PATH environment variables.\n\n

    More information on KiloSort at:
        https://github.com/cortex-lab/KiloSort
    """

    def __init__(self, **kargs):
        BaseSorter.__init__(self, **kargs)

    @staticmethod
    def set_kilosort_path(kilosort_path):
        KilosortSorter.kilosort_path = kilosort_path

    @staticmethod
    def set_npy_matlab_path(npy_matlab_path):
        KilosortSorter.npy_matlab_path = npy_matlab_path

    def _setup_recording(self, recording, output_folder):

        source_dir = Path(__file__).parent

        p = self.params

        if not check_if_installed(KilosortSorter.kilosort_path, KilosortSorter.npy_matlab_path):
            raise Exception(KilosortSorter.installation_mesg)

        # save binary file
        file_name = 'recording'
        se.write_binary_dat_format(recording, output_folder / file_name, dtype='int16')

        # set up kilosort config files and run kilosort on data
        with (source_dir / 'kilosort_master.txt').open('r') as f:
            kilosort_master = f.readlines()
        with (source_dir / 'kilosort_config.txt').open('r') as f:
            kilosort_config = f.readlines()
        with (source_dir / 'kilosort_channelmap.txt').open('r') as f:
            kilosort_channelmap = f.readlines()

        nchan = recording.get_num_channels()
        dat_file = (output_folder / (file_name + '.dat')).absolute()
        kilo_thresh = p['detect_threshold']
        Nfilt = (nchan // 32) * 32 * 8
        if Nfilt == 0:
            Nfilt = nchan * 8
        nsamples = 128 * 1024 + 64
        sample_rate = recording.get_sampling_frequency()

        if p['useGPU']:
            ug = 1
        else:
            ug = 0

        if p['car']:
            use_car = 1
        else:
            use_car = 0

        if not HAVE_KILOSORT:
            if p['kilosort_path'] is None or p['npy_matlab_path'] is None:

                raise ImportError('Kilosort is not installed\n', KilosortSorter.installation_mesg)
            else:
                KilosortSorter.set_kilosort_path(p['kilosort_path'])
                KilosortSorter.set_npy_matlab_path(p['npy_matlab_path'])

        abs_channel = (output_folder / 'kilosort_channelmap.m').absolute()
        abs_config = (output_folder / 'kilosort_config.m').absolute()
        kilosort_path = Path(KilosortSorter.kilosort_path).absolute()
        npy_matlab_path = Path(KilosortSorter.npy_matlab_path).absolute() / 'npy-matlab'

        kilosort_master = ''.join(kilosort_master).format(ug, kilosort_path, npy_matlab_path,
                                                                             output_folder, abs_channel,
                                                          abs_config)
        kilosort_config = ''.join(kilosort_config).format(nchan, nchan, sample_rate, dat_file,
                                                          Nfilt, nsamples, kilo_thresh, use_car)
        electrode_dimensions = p['electrode_dimensions']

        if 'group' in recording.get_channel_property_names():
            groups = [recording.get_channel_property(ch, 'group') for ch in recording.get_channel_ids()]
        else:
            groups = 'ones(1, Nchannels)'
        if 'location' in recording.get_channel_property_names():
            positions = np.array([recording.get_channel_property(chan, 'location') for chan in recording.get_channel_ids()])
            if electrode_dimensions is None:
                kilosort_channelmap = ''.join(kilosort_channelmap
                                              ).format(nchan,
                                                       list(positions[:, 0]),
                                                       list(positions[:, 1]),
                                                       groups,
                                                       sample_rate)
            elif len(electrode_dimensions) == 2:
                kilosort_channelmap = ''.join(kilosort_channelmap
                                              ).format(nchan,
                                                       list(positions[:, electrode_dimensions[0]]),
                                                       list(positions[:, electrode_dimensions[1]]),
                                                       groups,
                                                       recording.get_sampling_frequency())
            else:
                raise Exception("Electrode dimension should bi a list of len 2")
        else:
            raise Exception("'location' information is needed. Provide a probe information with a 'probe_file'")

        for fname, value in zip(['kilosort_master.m', 'kilosort_config.m',
                                 'kilosort_channelmap.m'],
                                [kilosort_master, kilosort_config,
                                 kilosort_channelmap]):
            with (output_folder / fname).open('w') as f:
                f.writelines(value)

    def _run(self, recording, output_folder):

        cmd = "matlab -nosplash -nodisplay -r 'run {}; quit;'".format(output_folder / 'kilosort_master.m')
        if self.debug:
            print(cmd)
        if "win" in sys.platform:
            cmd_list = ['matlab', '-nosplash', '-nodisplay', '-wait',
                        '-r','run {}; quit;'.format(output_folder / 'kilosort_master.m')]
        else:
            cmd_list = ['matlab', '-nosplash', '-nodisplay',
                        '-r', 'run {}; quit;'.format(output_folder / 'kilosort_master.m')]

        # retcode = _run_command_and_print_output_split(cmd_list)
        _call_command_split(cmd_list)

    @staticmethod
    def get_result_from_folder(output_folder):
        sorting = se.KiloSortSortingExtractor(output_folder)
        return sorting
