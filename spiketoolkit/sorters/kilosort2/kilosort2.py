"""
Important note.
For facilities
kilosort2_path
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


def check_if_installed(kilosort2_path, npy_matlab_path):
    if kilosort2_path is None or npy_matlab_path is None:
        return False

    if kilosort2_path is not None and kilosort2_path.startswith('"'):
        kilosort2_path = kilosort2_path[1:-1]
        kilosort2_path = Path(kilosort2_path).absolut()

    if npy_matlab_path is not None and npy_matlab_path.startswith('"'):
        npy_matlab_path = npy_matlab_path[1:-1]
        npy_matlab_path = Path(npy_matlab_path).absolut()

    if (Path(kilosort2_path) / 'master_kilosort.m').is_file() \
            or not (Path(npy_matlab_path) / 'npy-matlab' / 'readNPY.m').is_file():
        return True
    else:
        return False


if check_if_installed(os.getenv('KILOSORT2_PATH'), os.getenv('NPY_MATLAB_PATH')):
    HAVE_KILOSORT2 = True
else:
    HAVE_KILOSORT2 = False


class Kilosort2Sorter(BaseSorter):
    """


    """

    sorter_name = 'kilosort2'
    installed = HAVE_KILOSORT2
    kilosort2_path = os.getenv('KILOSORT2_PATH')
    npy_matlab_path = os.getenv('NPY_MATLAB_PATH')
    SortingExtractor_Class = se.KiloSortSortingExtractor

    _default_params = {
        'file_name': None,
        'probe_file': None,
        'detect_threshold': 5,
        'electrode_dimensions': None,
        'car': True,
        'npy_matlab_path': None,
        'kilosort2_path': None,
        'minFR': 0.1,
    }

    installation_mesg = """\nTo use Kilosort run:\n
        >>> git clone https://github.com/cortex-lab/KiloSort
        >>> git clone https://github.com/kwikteam/npy-matlab\n
    and provide the installation path with the 'kilosort2_path' and
    npy_matlab_path' arguments or by setting the KILOSORT_PATH
    and NPY_MATLAB_PATH environment variables.\n\n

    More information on KiloSort at:
        https://github.com/MouseLand/Kilosort2
    """

    def __init__(self, **kargs):
        BaseSorter.__init__(self, **kargs)

    @staticmethod
    def set_kilosort2_path(kilosort2_path):
        Kilosort2Sorter.kilosort2_path = kilosort2_path

    @staticmethod
    def set_npy_matlab_path(npy_matlab_path):
        Kilosort2Sorter.npy_matlab_path = npy_matlab_path

    def _setup_recording(self, recording, output_folder):

        source_dir = Path(__file__).parent

        p = self.params

        if not check_if_installed(Kilosort2Sorter.kilosort2_path, Kilosort2Sorter.npy_matlab_path):
            raise Exception(Kilosort2Sorter.installation_mesg)

        # save binary file
        if p['file_name'] is None:
            self.file_name = Path('recording')
        elif p['file_name'].suffix == '.dat':
            self.file_name = p['file_name'].stem
        p['file_name'] = self.file_name
        se.write_binary_dat_format(recording, output_folder / self.file_name, dtype='int16')

        # set up kilosort2 config files and run kilosort2 on data
        with (source_dir / 'kilosort2_master.txt').open('r') as f:
            kilosort2_master = f.readlines()
        with (source_dir / 'kilosort2_config.txt').open('r') as f:
            kilosort2_config = f.readlines()
        with (source_dir / 'kilosort2_channelmap.txt').open('r') as f:
            kilosort2_channelmap = f.readlines()

        nchan = recording.get_num_channels()
        dat_file = (output_folder / (self.file_name.name + '.dat')).absolute()
        kilo_thresh = p['detect_threshold']

        sample_rate = recording.get_sampling_frequency()

        if not HAVE_KILOSORT2:
            if p['kilosort2_path'] is None or p['npy_matlab_path'] is None:
                raise ImportError('Kilosort2 is not installed\n', Kilosort2Sorter.installation_mesg)
            else:
                Kilosort2Sorter.set_kilosort2_path(p['kilosort_path'])
                Kilosort2Sorter.set_npy_matlab_path(p['npy_matlab_path'])

        abs_channel = (output_folder / 'kilosort2_channelmap.m').absolute()
        abs_config = (output_folder / 'kilosort2_config.m').absolute()
        kilosort2_path = Path(Kilosort2Sorter.kilosort2_path).absolute()
        npy_matlab_path = Path(Kilosort2Sorter.npy_matlab_path).absolute() / 'npy-matlab'

        if p['car']:
            use_car = 1
        else:
            use_car = 0

        kilosort2_master = ''.join(kilosort2_master).format(kilosort2_path, npy_matlab_path,
                                                            output_folder, abs_channel, abs_config)
        kilosort2_config = ''.join(kilosort2_config).format(nchan, nchan, sample_rate, dat_file, p['minFR'],
                                                            kilo_thresh, use_car)
        electrode_dimensions = p['electrode_dimensions']

        if 'group' in recording.get_channel_property_names():
            groups = [recording.get_channel_property(ch, 'group') for ch in recording.get_channel_ids()]
        else:
            groups = 'ones(1, Nchannels)'
        if 'location' in recording.get_channel_property_names():
            positions = np.array([recording.get_channel_property(chan, 'location') for chan in recording.get_channel_ids()])
            if electrode_dimensions is None:
                kilosort2_channelmap = ''.join(kilosort2_channelmap
                                              ).format(nchan,
                                                       list(positions[:, 0]),
                                                       list(positions[:, 1]),
                                                       groups,
                                                       sample_rate)
            elif len(electrode_dimensions) == 2:
                kilosort2_channelmap = ''.join(kilosort2_channelmap
                                              ).format(nchan,
                                                       list(positions[:, electrode_dimensions[0]]),
                                                       list(positions[:, electrode_dimensions[1]]),
                                                       groups,
                                                       recording.get_sampling_frequency())
            else:
                raise Exception("Electrode dimension should bi a list of len 2")
        else:
            raise Exception("'location' information is needed. Provide a probe information with a 'probe_file'")

        for fname, value in zip(['kilosort2_master.m', 'kilosort2_config.m',
                                 'kilosort2_channelmap.m'],
                                [kilosort2_master, kilosort2_config,
                                 kilosort2_channelmap]):
            with (output_folder / fname).open('w') as f:
                f.writelines(value)

    def _run(self, recording, output_folder):

        cmd = "matlab -nosplash -nodisplay -r 'run {}; quit;'".format(output_folder / 'kilosort2_master.m')
        if self.debug:
            print(cmd)
        if "win" in sys.platform:
            cmd_list = ['matlab', '-nosplash', '-nodisplay', '-wait',
                        '-r','run {}; quit;'.format(output_folder / 'kilosort2_master.m')]
        else:
            cmd_list = ['matlab', '-nosplash', '-nodisplay',
                        '-r', 'run {}; quit;'.format(output_folder / 'kilosort2_master.m')]

        # retcode = _run_command_and_print_output_split(cmd_list)
        _call_command_split(cmd_list)

    @staticmethod
    def get_result_from_folder(output_folder):
        sorting = se.KiloSortSortingExtractor(output_folder)
        return sorting
