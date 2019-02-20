from pathlib import Path
import os
import sys

from ..tools import  _call_command_split
from spiketoolkit.sorters.basesorter import BaseSorter
import spikeextractors as se


if check_if_installed(os.getenv('IRONCLUST_PATH')):
    HAVE_IRONCLUST = True
else:
    HAVE_IRONCLUST = False


class IronclustSorter(BaseSorter):
    """
    """
    
    sorter_name = 'ironclust'
    installed = HAVE_IRONCLUST
    ironclust_path = os.getenv('IRONCLUST_PATH')
    # SortingExtractor_Class = se.NumpySortingExtractor
    SortingExtractor_Class = None # custum get_result
    
    _default_params = {
        'prm_template_name': None,  # Name of the template file
        'detect_sign': -1,  # Polarity of the spikes, -1, 0, or 1
        'adjacency_radius': -1,  # Channel neighborhood adjacency radius corresponding to geom file
        'detect_threshold': 5,  # Threshold for detection
        'merge_thresh': .98,  # Cluster merging threhold 0..1
        'freq_min': 300,  # Lower frequency limit for band-pass filter
        'freq_max': 6000,  # Upper frequency limit for band-pass filter
        'pc_per_chan': 3,  # Number of pc per channel
        'parallel': True,
        'ironclust_path': None
    }

    installation_mesg = """\nTo use Ironclust run:\n
        >>> pip install mountainlab_pytools kbucket\n

    and clone the repo:\n
        >>> git clone https://github.com/jamesjun/ironclust\n
    and provide the installation path with the 'ironclust_path' argument or
    by setting the IRONCLUST_PATH environment variable.\n\n
    
    More information on KiloSort at:
        https://github.com/jamesjun/ironclust
    """
    
    def __init__(self, **kargs):
        BaseSorter.__init__(self, **kargs)


    def set_params(self, **params):
        self.params = params


    @staticmethod
    def set_ironclust_path(ironclust_path):
        IronclustSorter.ironclust_path = ironclust_path

    
    def _setup_recording(self):
        p = self.params

        if not check_if_installed(IronclustSorter.ironclust_path):
            raise ModuleNotFoundError(IronclustSorter.installation_mesg)
        
        dataset_dir = self.output_folder / 'ironclust_dataset'
        if not dataset_dir.is_dir():
            dataset_dir.mkdir()

        self.output_folder = self.output_folder.absolute()
        self.dataset_dir = dataset_dir.absolute()

        # Generate three files in the dataset directory: raw.mda, geom.csv, params.json
        se.MdaRecordingExtractor.writeRecording(recording=self.recording, save_path=self.dataset_dir)
        samplerate = self.recording.getSamplingFrequency()

        if self.debug:
            print('Reading timeseries header...')
        HH = mdaio.readmda_header(str(self.dataset_dir / 'raw.mda'))
        num_channels = HH.dims[0]
        num_timepoints = HH.dims[1]
        duration_minutes = num_timepoints / samplerate / 60
        if self.debug:
            print('Num. channels = {}, Num. timepoints = {}, duration = {} minutes'.format(num_channels, num_timepoints,
                                                                                       duration_minutes))
        
        if self.debug:
            print('Creating .params file...')
        txt = ''
        txt += 'samplerate={}\n'.format(samplerate)
        txt += 'detect_sign={}\n'.format(p['detect_sign'])
        txt += 'adjacency_radius={}\n'.format(p['adjacency_radius'])
        txt += 'detect_threshold={}\n'.format(p['detect_threshold'])
        txt += 'merge_thresh={}\n'.format(p['merge_thresh'])
        txt += 'freq_min={}\n'.format(p['freq_min'])
        txt += 'freq_max={}\n'.format(p['freq_max'])
        txt += 'pc_per_chan={}\n'.format(p['pc_per_chan'])
        txt += 'prm_template_name={}\n'.format(p['prm_template_name'])
        _write_text_file(self.dataset_dir / 'argfile.txt', txt)
    
    def _run(self):
        if self.debug:
            print('Running IronClust...')
            
        cmd_path = "addpath('{}', '{}/matlab', '{}/mdaio');".format(IronclustSorter.ironclust_path, IronclustSorter.ironclust_path, IronclustSorter.ironclust_path)
        # "p_ironclust('$(tempdir)','$timeseries$','$geom$','$prm$','$firings_true$','$firings_out$','$(argfile)');"
        cmd_call = "p_ironclust('{}', '{}', '{}', '', '', '{}', '{}');" \
            .format(self.output_folder, self.dataset_dir / 'raw.mda', self.dataset_dir / 'geom.csv', self.output_folder / 'firings.mda',
                    self.dataset_dir / 'argfile.txt')
        cmd = 'matlab -nosplash -nodisplay -r "{} {} quit;"'.format(cmd_path, cmd_call)
        if self.debug:
            print(cmd)
        
        if "win" in sys.platform:
            cmd_list = ['matlab', '-nosplash', '-nodisplay', '-wait',
                        '-r', '{} {} quit;'.format(cmd_path, cmd_call)]
        else:
            cmd_list = ['matlab', '-nosplash', '-nodisplay',
                        '-r', '{} {} quit;'.format(cmd_path, cmd_call)]

        _call_command_split(cmd_list)
        
    
    def get_result(self):
        # overwrite the SorterBase.get_result

        result_fname = self.output_folder / 'firings.mda'
        
        assert result_fname.exists(),'Result file does not exist: {}'.format(str(result_fname))

        firings = mdaio.readmda(str(result_fname))
        sorting = se.NumpySortingExtractor()
        sorting.setTimesLabels(firings[1, :], firings[2, :])
        return sorting


def _write_text_file(fname, str):
    with fname.open('w') as f:
        f.write(str)

def check_if_installed(ironclust_path):
    if ironclust_path is not None and ironclust_path.startswith('"'):
        ironclust_path = ironclust_path[1:-1]
        ironclust_path = Path(ironclust_path).absolute()

    if (Path(ironclust_path) / 'matlab' / 'p_ironclust.m').is_file():
        try:
            from mountainlab_pytools import mdaio
            return True
        except ModuleNotFoundError:
            return False
    else:
        return False

def run_ironclust(
        recording,
        output_folder=None,
        by_property=None,
        parallel=False,
        debug=False,
        ironclust_path=None,
        **params):
    
    # this preserve the old signature
    if ironclust_path is not None:
        IronclustSorter.set_ironclust_path(ironclust_path)
    
    sorter = IronclustSorter(recording=recording, output_folder=output_folder,
                                    by_property=by_property, parallel=parallel, debug=debug)
    if 'ironclust_path' in  params.keys() and params['ironclust_path'] is not None:
        IronclustSorter.set_ironclust_path(params['ironclust_path'])
    else:
        IronclustSorter.set_ironclust_path(os.getenv('IRONCLUST_PATH'))
    sorter.set_params(**params)
    sorter.run()
    sortingextractor = sorter.get_result()
    
    return sortingextractor
