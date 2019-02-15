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
import shutil
import numpy as np

from spiketoolkit.sorters.basesorter import BaseSorter
import spikeextractors as se
from ..tools import _call_command_split



kilosort_path = os.getenv('KILOSORT_PATH')
if kilosort_path is not None and kilosort_path.startswith('"'):
    kilosort_path = kilosort_path[1:-1]
    kilosort_path = Path(kilosort_path).absolut()

npy_matlab_path = os.getenv('NPY_MATLAB_PATH')
if npy_matlab_path is not None and npy_matlab_path.startswith('"'):
    npy_matlab_path = npy_matlab_path[1:-1]
    npy_matlab_path = Path(npy_matlab_path).absolut()

if (kilosort_path is not None) and (npy_matlab_path is not None):
    HAVE_KILOSORT = True
else:
    HAVE_KILOSORT = False



class KilosortSorter(BaseSorter):
    """
    
    
    """
    
    sorter_name = 'kilosort'
    installed = HAVE_KILOSORT
    SortingExtractor_Class = se.KiloSortSortingExtractor
    
    _default_params = {
        'file_name': None,
        'probe_file': None,
    
        'useGPU': False,
        'detect_threshold': 4,
        'electrode_dimensions':None,
    }

    installation_mesg = """
        git clone https://github.com/cortex-lab/KiloSort
        git clone https://github.com/kwikteam/npy-matlab
        and provide the installation path with the 'kilosort_path' and 
        npy_matlab_path' arguments or by setting the KILOSORT_PATH 
        and NPY_MATLAB_PATH environment variables.
    
    More information on KiloSort at:
        https://github.com/cortex-lab/KiloSort

      
      
    """
    
    def __init__(self, **kargs):
        BaseSorter.__init__(self, **kargs)

    def set_params(self, **params):
        self.params = params
    
    
    def _setup_recording(self):
        
        source_dir = Path(__file__).parent
        
        p = self.params

        # save prb file:
        if p['probe_file'] is None:
            p['probe_file'] = self.output_folder / 'probe.prb'
            se.saveProbeFile(self.recording, p['probe_file'], format='klusta', radius=p['adjacency_radius'])

        # save binary file
        if p['file_name'] is None:
            self.file_name = Path('recording')
        elif file_name.suffix == '.dat':
            self.file_name = p['file_name'].stem
        p['file_name'] = self.file_name
        se.writeBinaryDatFormat(self.recording, self.output_folder / self.file_name, dtype='int16')

        # set up kilosort config files and run kilosort on data
        with (source_dir / 'kilosort_master.txt').open('r') as f:
            kilosort_master = f.readlines()
        with (source_dir / 'kilosort_config.txt').open('r') as f:
            kilosort_config = f.readlines()
        with (source_dir / 'kilosort_channelmap.txt').open('r') as f:
            kilosort_channelmap = f.readlines()

        nchan = self.recording.getNumChannels()
        dat_file = (output_folder / (self.file_name.name + '.dat')).absolute()
        kilo_thresh = p['detect_threshold']
        Nfilt = (nchan // 32) * 32 * 8
        if Nfilt == 0:
            Nfilt = nchan * 8
        nsamples = 128 * 1024 + 64
        sample_rate = self.recording.getSamplingFrequency()

        if p['useGPU']:
            ug = 1
        else:
            ug = 0

        abs_channel = (self.output_folder / 'kilosort_channelmap.m').absolute()
        abs_config = (self.output_folder / 'kilosort_config.m').absolute()
        kilosort_path = Path(self.kilosort_path).absolute()
        npy_matlab_path = Path(self.npy_matlab_path).absolute() / 'npy-matlab'

        kilosort_master = ''.join(kilosort_master).format(ug, kilosort_path, npy_matlab_path, 
                                                                                SELF.output_folder, abs_channel, abs_config)
        kilosort_config = ''.join(kilosort_config).format(nchan, nchan, sample_rate, dat_file,
                                                                                Nfilt, nsamples, kilo_thresh)
        if 'location' in self.recording.getChannelPropertyNames():
            positions = np.array([self.recording.getChannelProperty(chan, 'location') for chan in self.recording.getChannelIds()])
            if electrode_dimensions is None:
                kilosort_channelmap = ''.join(kilosort_channelmap
                                              ).format(nchan,
                                                       list(positions[:, 0]),
                                                       list(positions[:, 1]),
                                                       'ones(1, Nchannels)',
                                                       sample_rate)
            elif len(electrode_dimensions) == 2:
                kilosort_channelmap = ''.join(kilosort_channelmap
                                              ).format(nchan,
                                                       list(positions[:, electrode_dimensions[0]]),
                                                       list(positions[:, electrode_dimensions[1]]),
                                                       'ones(1, Nchannels)',
                                                       recording.getSamplingFrequency())
            else:
                raise Exception("Electrode dimension should bi a list of len 2")
        else:
            raise Exception("'location' information is needed. Provide a probe information with a 'probe_file'")

        for fname, value in zip(['kilosort_master.m', 'kilosort_config.m',
                                 'kilosort_channelmap.m'],
                                [kilosort_master, kilosort_config,
                                 kilosort_channelmap]):
            with (self.output_folder / fname).open('w') as f:
                f.writelines(value)    

    def _run(self):
        
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
    
    
def run_kilosort(
        recording,
        output_folder=None,
        by_property=None,
        parallel=False,
        debug=False,
        **params):
    sorter = KilosortSorter(recording=recording, output_folder=output_folder,
                                    by_property=by_property, parallel=parallel, debug=debug)
    sorter.set_params(**params)
    sorter.run()
    sortingextractor = sorter.get_result()
    
    return sortingextractor

