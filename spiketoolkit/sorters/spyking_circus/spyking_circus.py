from pathlib import Path
import os
import shutil
import numpy as np


from spiketoolkit.sorters.basesorter import BaseSorter
import spikeextractors as se
from ..tools import _run_command_and_print_output

try:
    import circus
    HAVE_SC = True
except ModuleNotFoundError:
    HAVE_SC = False


class SpykingcircusSorter(BaseSorter):
    """
    """
    
    sorter_name = 'spiykingcircus'
    installed = HAVE_SC
    SortingExtractor_Class = se.SpykingCircusSortingExtractor
    
    _default_params = {
        'probe_file' : None,
        'file_name' : None,
    
        'detect_sign': -1,  # -1 - 1 - 0
        'adjacency_radius': 100,  # Channel neighborhood adjacency radius corresponding to geom file
        'detect_threshold': 6,  # Threshold for detection
        'template_width_ms': 3,  # Spyking circus parameter
        'filter': True,
        'merge_spikes': True,
        'n_cores': None,
        'electrode_dimensions': None,
        'whitening_max_elts': 1000,  # I believe it relates to subsampling and affects compute time
        'clustering_max_elts': 10000,  # I believe it relates to subsampling and affects compute time
        }
    
    installation_mesg = """
        >>> pip install spyking-circus
        
        Need OpenMPI working, for ubuntu do: 
            sudo apt install libopenmpi-dev"
        
        More information on Spyking-Circus at: "
            https://spyking-circus.readthedocs.io/en/latest/
    """
    
    def __init__(self, **kargs):
        BaseSorter.__init__(self, **kargs)

    def set_params(self, **params):
        self.params = params
    
    def _setup_recording(self):
        p = self.params
        source_dir = Path(__file__).parent
        
        # save prb file:
        if p['probe_file'] is None:
            p['probe_file'] = self.output_folder / 'probe.prb'
            se.saveProbeFile(self.recording, p['probe_file'], format='spyking_circus',
                                radius=p['adjacency_radius'], dimensions=p['electrode_dimensions'])

        # save binary file
        if p['file_name'] is None:
            self.file_name = Path('recording')
        elif file_name.suffix == '.dat':
            self.file_name = p['file_name'].stem
        p['file_name'] = self.file_name
        np.save(str(self.output_folder / self.file_name), self.recording.getTraces().astype('float32'))

        if p['detect_sign'] < 0:
            detect_sign = 'negative'
        elif p['detect_sign'] > 0:
            detect_sign = 'positive'
        else:
            detect_sign = 'both'
        
        sample_rate = float(self.recording.getSamplingFrequency())
        
        # set up spykingcircus config file
        with (source_dir / 'config_default.params').open('r') as f:
            circus_config = f.readlines()
        if p['merge_spikes']:
            auto = 1e-5
        else:
            auto = 0
        circus_config = ''.join(circus_config).format(sample_rate, p['probe_file'], p['template_width_ms'],
                    p['detect_threshold'], detect_sign, p['filter'], p['whitening_max_elts'], 
                    p['clustering_max_elts'], auto)
        with (self.output_folder / (self.file_name.name + '.params')).open('w') as f:
            f.writelines(circus_config)

        if p['n_cores'] is None:
            p['n_cores'] = np.maximum(1, int(os.cpu_count() / 2))


    def _run(self):
        n_cores = self.params['n_cores']
        cmd = 'spyking-circus {} -c {} '.format(self.output_folder / (self.file_name.name + '.npy'), n_cores)
        cmd_merge = 'spyking-circus {} -m merging -c {} '.format(self.output_folder / (self.file_name.name + '.npy'), n_cores)
        # cmd_convert = 'spyking-circus {} -m converting'.format(join(output_folder, file_name+'.npy'))
        if self.debug:
            print(cmd)
        retcode = _run_command_and_print_output(cmd)
        if retcode != 0:
            raise Exception('Spyking circus returned a non-zero exit code')
        if self.params['merge_spikes']:
            if self.debug:
                print(cmd_merge)
            retcode = _run_command_and_print_output(cmd_merge)
            if retcode != 0:
                raise Exception('Spyking circus merging returned a non-zero exit code')

    def get_result(self):
        # overwrite the SorterBase.get_result
        sorting = se.SpykingCircusSortingExtractor(self.output_folder / self.file_name)
        return sorting


