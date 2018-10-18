import spikeinterface as si

import os
import time
import numpy as np
import tempfile
from os.path import join
import subprocess


# Installation notes for ubuntu:
# sudo apt install libopenmpi-dev
# pip install spyking-circus

def spyking_circus_a(
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
        whitening_max_elts=1000,  # I believe it relates to subsampling and affects compute time
        clustering_max_elts=10000,  # I believe it relates to subsampling and affects compute time
    ):
    try:
        import circus
    except ModuleNotFoundError:
        raise ModuleNotFoundError("\nTo use Spyking-Circus, install spyking-circus: \n\n"
                                  "\npip install spyking-circus\n"
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
                         dimensions=[1,2])
        probe_file = join(output_folder, 'probe.prb')
    # save binary file
    if file_name is None:
        file_name = 'recording'
    elif file_name.endswith('.npy'):
        file_name = file_name[file_name.find('.npy')]
    np.save(join(output_folder, file_name), recording.getTraces())
    # si.writeBinaryDatFormat(recording, file_name, transpose=True)

    # set up spykingcircus config file
    with open(join(source_dir, 'config_default.params'), 'r') as f:
        circus_config = f.readlines()
    if merge_spikes:
        auto = 1e-5
    else:
        auto = 0
    circus_config = ''.join(circus_config).format(
        float(recording.getSamplingFrequency()), probe_file, template_width_ms, spike_thresh, detect_sign, filter, auto
    )
    with open(join(output_folder, file_name + '.params'), 'w') as f:
        f.writelines(circus_config)

    try:
        if n_cores is None:
            n_cores = np.maximum(1, int(os.cpu_count()/2))

        t_start_proc = time.time()
        subprocess.check_output(['spyking-circus', join(output_folder, file_name+'.npy'), '-c', str(n_cores)])
        if merge_spikes:
            subprocess.call(['spyking-circus', join(output_folder, file_name+'.npy'), '-m', 'merging', '-c', str(n_cores)])
        processing_time = time.time() - t_start_proc
        print('Elapsed time: ', processing_time)
    except subprocess.CalledProcessError as e:
        raise Exception(e.output)

    sorting = si.SpykingCircusSortingExtractor(join(output_folder, file_name))
    return sorting
