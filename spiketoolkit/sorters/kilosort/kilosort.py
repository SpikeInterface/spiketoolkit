import spikeinterface as si

import os
import time
import numpy as np
from os.path import join
from spiketoolkit.sorters.tools import run_command_and_print_output

def kilosort(
        recording,
        kilosort_path=None,
        npy_matlab_path=None,
        output_folder=None,
        useGPU=False,
        probe_file=None,
        file_name=None,
        spike_thresh=5,
        electrode_dimensions=None
    ):
    if kilosort_path is None:
        kilosort_path = os.getenv('KILOSORT_PATH', None)
    if npy_matlab_path is None:
        npy_matlab_path = os.getenv('NPY_MATLAB_PATH', None)
    if not os.path.isfile(join(kilosort_path, 'preprocessData.m')) \
            or not os.path.isfile(join(npy_matlab_path, 'readNPY.m')):
        raise ModuleNotFoundError("\nTo use KiloSort, install KiloSort and npy-matlab from sources: \n\n"
                                  "\ngit clone https://github.com/cortex-lab/KiloSort\n"
                                  "\ngit clone https://github.com/kwikteam/npy-matlab\n"
                                  "and provide the installation path with the 'kilosort_path' and "
                                  "'npy_matlab_path' arguments or by setting the KILOSORT_PATH and NPY_MATLAB_PATH"
                                  "environment variables.\n+n"
                                  "\nMore information on KiloSort at: "
                                  "\nhttps://github.com/cortex-lab/KiloSort")

    source_dir = os.path.dirname(os.path.realpath(__file__))

    if output_folder is None:
        output_folder = os.path.abspath('kilosort')
    else:
        output_folder = os.path.abspath(join(output_folder, 'kilosort'))

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    kilosort_path = os.path.abspath(kilosort_path)
    npy_matlab_path = os.path.abspath(npy_matlab_path)

    if probe_file is not None:
        si.loadProbeFile(recording, probe_file)

    # save binary file
    if file_name is None:
        file_name = 'recording'
    elif file_name.endswith('.dat'):
        file_name = file_name[file_name.find('.dat')]
    si.writeBinaryDatFormat(recording, join(output_folder, file_name))

    # set up kilosort config files and run kilosort on data
    with open(join(source_dir, 'kilosort_master.txt'), 'r') as f:
        kilosort_master = f.readlines()
    with open(join(source_dir, 'kilosort_config.txt'), 'r') as f:
        kilosort_config = f.readlines()
    with open(join(source_dir, 'kilosort_channelmap.txt'), 'r') as f:
        kilosort_channelmap = f.readlines()

    nchan = recording.getNumChannels()
    dat_file = file_name +'.dat'
    kilo_thresh = spike_thresh
    Nfilt = (nchan // 32) * 32 * 4
    if Nfilt == 0:
        Nfilt = 64
    nsamples = 128 * 1024 + 32

    if useGPU:
        ug = 1
    else:
        ug = 0

    kilosort_master = ''.join(kilosort_master).format(
        ug, kilosort_path, npy_matlab_path, output_folder, join(output_folder, 'results')
    )
    kilosort_config = ''.join(kilosort_config).format(
        nchan, nchan, recording.getSamplingFrequency(), dat_file , Nfilt, nsamples, kilo_thresh
    )
    if 'location' in recording.getChannelPropertyNames():
        positions = np.array([recording.getChannelProperty(chan, 'location') for chan in range(nchan)])
        if electrode_dimensions is None:
            kilosort_channelmap = ''.join(kilosort_channelmap
                                          ).format(nchan,
                                                   list(positions[:, 0]),
                                                   list(positions[:, 1]),
                                                   'ones(1, Nchannels)',
                                                   recording.getSamplingFrequency())
        elif len(electrode_dimensions)==2:
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
        with open(join(output_folder, fname), 'w') as f:
            f.writelines(value)

    # start sorting with kilosort
    print('Running KiloSort')
    cwd = os.getcwd()
    t_start_proc = time.time()
    os.chdir(output_folder)
    cmd = 'matlab -nosplash -nodisplay -r "run kilosort_master.m; quit;"'
    print(cmd)
    retcode = run_command_and_print_output(cmd)

    if retcode != 0:
        raise Exception('KiloSort returned a non-zero exit code')
    print('Elapsed time: ', time.time() - t_start_proc)

    sorting = si.KiloSortSortingExtractor(join(output_folder, 'results'))
    os.chdir(cwd)
    return sorting
