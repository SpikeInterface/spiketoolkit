import numpy as np

from spiketoolkit.sorters.basesorter import BaseSorter
from ..tools import _run_command_and_print_output
import spikeextractors as se

try:
    HAVE_YASS = True
except ImportError:
    HAVE_YASS = False


class YassSorter(BaseSorter):
    """
    """
    
    sorter_name = 'yass'
    installed = HAVE_YASS
    
    _default_params = {
        'detect_sign' : -1,  # -1 - 1 - 0
        'template_width_ms' : 1,  # yass parameter
        'filter' : True,
        'adjacency_radius' : 100,
    }
    
    installation_mesg = """
    pip install tensorflow
    pip install yass-algorithm[tf]
    """
    
    def __init__(self, **kargs):
        BaseSorter.__init__(self, **kargs)

    def _setup_recording(self, recording, output_folder):
        p = self.params
        
        # save probe file
        probe_file = output_folder / 'probe.npy'
        se.saveProbeFile(recording, probe_file, format='yass')
        
        # write binary signal files
        traces = recording.getTraces().astype('float32')
        fReversePolarity=(p['detect_sign'] > 0)
        if fReversePolarity:
            traces = traces * -1
        bin_file = output_folder / 'signal_raw.bin'
        with open(bin_file, 'wb') as f:
            np.ravel(traces, order='F').tofile(f)
        
        # write config.yml
        source_dir = Path(__file__).parent
        with (source_dir / 'config_default.yaml').open('r') as f:
            yass_config = f.read()

        n_channels = recording.getNumChannels()
        sampling_rate = recording.getSamplingFrequency()

        yass_config = yass_config.format(output_folder, bin_file, probe_file, 'single',
                                    int(sampling_rate), n_channels, 
                                    p['adjacency_radius'],
                                    p['template_width_ms'], p['filter'])
        
        with open(output_folder /'config.yaml', 'w') as f:
            f.write(yass_config)

    
    def _run(self, recording, output_folder):
        cmd = 'yass {}'.format(output_folder/ 'config.yaml')
        _run_command_and_print_output_split(cmd)
    
    @staticmethod
    def get_result_from_folder(output_folder):
        sorting = se.YassSortingExtractor(output_folder)
        return sorting


