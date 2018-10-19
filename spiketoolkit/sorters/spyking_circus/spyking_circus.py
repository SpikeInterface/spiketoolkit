import spikeinterface as si

import os
import time
import numpy as np
from os.path import join
from spiketoolkit.sorters.tools import run_command_and_print_output

def spyking_circus(
        recording,
        output_folder=None,  # Temporary working directory
        probe_file=None,
        file_name=None,
        detect_sign='negative',  # 'negative' - 'positive' - 'both'
        adjacency_radius=100,  # Channel neighborhood adjacency radius corresponding to geom file
        spike_thresh=6,  # Threshold for detection
        template_width_ms=3,  # Spyking circus parameter
        filter=True,
        merge_spikes=True,
        n_cores=None,
        electrode_dimensions=None,
        whitening_max_elts=1000,  # I believe it relates to subsampling and affects compute time
        clustering_max_elts=10000,  # I believe it relates to subsampling and affects compute time
    ):
    try:
        import circus
    except ModuleNotFoundError:
        raise ModuleNotFoundError("\nTo use Spyking-Circus, install spyking-circus: \n\n"
                                  "\npip install spyking-circus"
                                  "\nfor ubuntu install openmpi: "
                                  "\nsudo apt install libopenmpi-dev"
                                  "\nMore information on Spyking-Circus at: "
                                  "\nhttps://spyking-circus.readthedocs.io/en/latest/")
    source_dir = os.path.dirname(os.path.realpath(__file__))

    if output_folder is None:
        output_folder = 'spyking_circus'
    else:
        output_folder = join(output_folder, 'spyking_circus')

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # save prb file:
    if probe_file is None:
        si.saveProbeFile(recording, join(output_folder, 'probe.prb'), format='spyking_circus', radius=adjacency_radius,
                         dimensions=electrode_dimensions)
        probe_file = join(output_folder, 'probe.prb')
    # save binary file
    if file_name is None:
        file_name = 'recording'
    elif file_name.endswith('.npy'):
        file_name = file_name[file_name.find('.npy')]
    np.save(join(output_folder, file_name), recording.getTraces().astype('float32'))

    # set up spykingcircus config file
    with open(join(source_dir, 'config_default.params'), 'r') as f:
        circus_config = f.readlines()
    if merge_spikes:
        auto = 1e-5
    else:
        auto = 0
    circus_config = ''.join(circus_config).format(
        float(recording.getSamplingFrequency()), probe_file, template_width_ms, spike_thresh, detect_sign, filter,
        whitening_max_elts, clustering_max_elts, auto
    )
    with open(join(output_folder, file_name + '.params'), 'w') as f:
        f.writelines(circus_config)

    print('Running spyking circus...')
    t_start_proc = time.time()
    if n_cores is None:
        n_cores = np.maximum(1, int(os.cpu_count()/2))

    cmd = 'spyking-circus {} -c {} '.format(join(output_folder, file_name+'.npy'), n_cores)
    cmd_merge = 'spyking-circus {} -m merging -c {} '.format(join(output_folder, file_name+'.npy'), n_cores)
    print(cmd)
    retcode = run_command_and_print_output(cmd)
    if retcode != 0:
        raise Exception('Spyking circus returned a non-zero exit code')
    print(cmd_merge)
    retcode = run_command_and_print_output(cmd_merge)
    if retcode != 0:
        raise Exception('Spyking circus merging returned a non-zero exit code')
    processing_time = time.time() - t_start_proc
    print('Elapsed time: ', processing_time)
    sorting = si.SpykingCircusSortingExtractor(join(output_folder, file_name))

    return sorting
