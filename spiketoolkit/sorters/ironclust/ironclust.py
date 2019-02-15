from pathlib import Path
import os

from ..tools import  _call_command_split
from spiketoolkit.sorters.basesorter import BaseSorter
import spikeextractors as se

ironclust_path = os.getenv('IRONCLUST_PATH')
if ironclust_path is not None and ironclust_path.startswith('"'):
    ironclust_path = ironclust_path[1:-1]
    ironclust_path = Path(ironclust_path).absolut()

if (ironclust_path is not None):
    try:
        from mountainlab_pytools import mdaio

        HAVE_IRONCLUST = True
    except ModuleNotFoundError:
        HAVE_IRONCLUST = True
else:
    HAVE_IRONCLUST = False


class IronclustSorter(BaseSorter):
    """
    """
    
    sorter_name = 'ironclust'
    installed = HAVE_IRONCLUST
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
    
    
    def _setup_recording(self):
        p = self.params
        
        self.ironclust_path = p['ironclust_path']
        if self.ironclust_path is None:
            icp = os.getenv('IRONCLUST_PATH', None)
            if icp.startswith('"'):
                icp = icp[1:-1]
            self.ironclust_path = Path(icp)
        if self.ironclust_path is None:
            raise Exception(
                'You must either set the IRONCLUST_PATH environment variable, or pass the ironclust_path parameter')
        if not (Path( self.ironclust_path) / 'matlab' / 'p_ironclust.m').is_file():
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
            
        cmd_path = "addpath('{}', '{}/matlab', '{}/mdaio');".format(self.ironclust_path, self.ironclust_path, self.ironclust_path)
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


def run_ironclust(
        recording,
        output_folder=None,
        by_property=None,
        parallel=False,
        debug=False,
        **params):
    sorter = IronclustSorter(recording=recording, output_folder=output_folder,
                                    by_property=by_property, parallel=parallel, debug=debug)
    sorter.set_params(**params)
    sorter.run()
    sortingextractor = sorter.get_result()
    
    return sortingextractor        
        
        
        
        
############################


import spikeextractors as se
import os, sys
import time


def ironclust(recording,  # Recording object
              prm_template_name=None,  # Name of the template file
              by_property=None,
              parallel=False,
              output_folder=None,  # Temporary working directory
              detect_sign=-1,  # Polarity of the spikes, -1, 0, or 1
              adjacency_radius=-1,  # Channel neighborhood adjacency radius corresponding to geom file
              detect_threshold=5,  # Threshold for detection
              merge_thresh=.98,  # Cluster merging threhold 0..1
              freq_min=300,  # Lower frequency limit for band-pass filter
              freq_max=6000,  # Upper frequency limit for band-pass filter
              pc_per_chan=3,  # Number of pc per channel
              ironclust_path=None
):
    t_start_proc = time.time()
    if by_property is None:
        sorting = _ironclust(recording, prm_template_name, output_folder, detect_sign, adjacency_radius,
                             detect_threshold, merge_thresh, freq_min, freq_max, pc_per_chan, ironclust_path)
    else:
        if by_property in recording.getChannelPropertyNames():
            sorting = _spikeSortByProperty(recording, 'ironclust', by_property, parallel,
                                           prm_template_name=prm_template_name,
                                           output_folder=output_folder, detect_sign=detect_sign,
                                           adjacency_radius=adjacency_radius, detect_threshold=detect_threshold,
                                           merge_thresh=merge_thresh, freq_min=freq_min, freq_max=freq_max,
                                           pc_per_chan=pc_per_chan, ironclust_path=ironclust_path)
        else:
            print("Property not available! Running normal spike sorting")
            sorting = _ironclust(recording, prm_template_name, output_folder, detect_sign, adjacency_radius,
                                 detect_threshold, merge_thresh, freq_min, freq_max, pc_per_chan, ironclust_path)
    print('Elapsed time: ', time.time() - t_start_proc)

    return sorting
