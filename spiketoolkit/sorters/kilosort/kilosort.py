import spikeextractors as se
import os
import time
import numpy as np
from os.path import join
from ..tools import _run_command_and_print_output, _call_command, _spikeSortByProperty


def kilosort(
        recording,
        output_folder=None,
        by_property=None,
        kilosort_path=None,
        npy_matlab_path=None,
        useGPU=False,
        probe_file=None,
        file_name=None,
        detect_threshold=4,
        electrode_dimensions=None
):
    '''

    Parameters
    ----------
    recording
    output_folder
    by_property
    kilosort_path
    npy_matlab_path
    useGPU
    probe_file
    file_name
    detect_threshold
    electrode_dimensions

    Returns
    -------

    '''
    print(kilosort_path)
    t_start_proc = time.time()
    if by_property is None:
        sorting = _kilosort(recording, output_folder, kilosort_path, npy_matlab_path, useGPU, probe_file, file_name,
                            detect_threshold, electrode_dimensions)
    else:
        if by_property in recording.getChannelPropertyNames():
            sorting = _spikeSortByProperty(recording, 'kilosort', by_property, output_folder=output_folder,
                                           kilosort_path=kilosort_path, npy_matlab_path=npy_matlab_path,
                                           useGPU=useGPU, probe_file=probe_file, file_name=file_name,
                                           detect_threshold=detect_threshold,
                                           electrode_dimensions=electrode_dimensions)
        else:
            print("Property not available! Running normal spike sorting")
            sorting = _kilosort(recording, output_folder, kilosort_path, npy_matlab_path, useGPU, probe_file, file_name,
                                detect_threshold, electrode_dimensions)

    print('Elapsed time: ', time.time() - t_start_proc)

    return sorting


def _kilosort(
        recording,
        output_folder=None,
        kilosort_path=None,
        npy_matlab_path=None,
        useGPU=False,
        probe_file=None,
        file_name=None,
        detect_threshold=4,
        electrode_dimensions=None
):
    if kilosort_path is None or kilosort_path=='None':
        kilosort_path = os.getenv('KILOSORT_PATH')
    if npy_matlab_path is None or npy_matlab_path=='None':
        npy_matlab_path = os.getenv('NPY_MATLAB_PATH')
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
    print(output_folder)
    if output_folder is None:
        output_folder = os.path.abspath('./kilosort')
    else:
        output_folder = os.path.abspath(output_folder)

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    kilosort_path = os.path.abspath(kilosort_path)
    npy_matlab_path = os.path.abspath(npy_matlab_path)

    if probe_file is not None:
        se.loadProbeFile(recording, probe_file)

    # save binary file
    if file_name is None:
        file_name = 'recording'
    elif file_name.endswith('.dat'):
        file_name = file_name[file_name.find('.dat')]
    se.writeBinaryDatFormat(recording, join(output_folder, file_name), dtype='int16')

    # set up kilosort config files and run kilosort on data
    with open(join(source_dir, 'kilosort_master.txt'), 'r') as f:
        kilosort_master = f.readlines()
    with open(join(source_dir, 'kilosort_config.txt'), 'r') as f:
        kilosort_config = f.readlines()
    with open(join(source_dir, 'kilosort_channelmap.txt'), 'r') as f:
        kilosort_channelmap = f.readlines()

    nchan = recording.getNumChannels()
    dat_file = os.path.abspath(join(output_folder, file_name + '.dat'))
    kilo_thresh = detect_threshold
    Nfilt = (nchan // 32) * 32 * 8
    if Nfilt == 0:
        Nfilt = nchan * 8
    nsamples = 128 * 1024 + 64

    if useGPU:
        ug = 1
    else:
        ug = 0

    abs_channel = os.path.abspath(join(output_folder, 'kilosort_channelmap.m'))
    abs_config = os.path.abspath(join(output_folder, 'kilosort_config.m'))

    kilosort_master = ''.join(kilosort_master).format(
        ug, kilosort_path, npy_matlab_path, output_folder, abs_channel, abs_config
    )
    kilosort_config = ''.join(kilosort_config).format(
        nchan, nchan, recording.getSamplingFrequency(), dat_file, Nfilt, nsamples, kilo_thresh
    )
    if 'location' in recording.getChannelPropertyNames():
        positions = np.array([recording.getChannelProperty(chan, 'location') for chan in recording.getChannelIds()])
        if electrode_dimensions is None:
            kilosort_channelmap = ''.join(kilosort_channelmap
                                          ).format(nchan,
                                                   list(positions[:, 0]),
                                                   list(positions[:, 1]),
                                                   'ones(1, Nchannels)',
                                                   recording.getSamplingFrequency())
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
        with open(join(output_folder, fname), 'w') as f:
            f.writelines(value)

    # start sorting with kilosort
    print('Running KiloSort')
    cmd = 'matlab -nosplash -nodisplay -r "run {}; quit;"'.format(join(output_folder, 'kilosort_master.m'))
    print(cmd)
    _call_command(cmd)
    # retcode = run_command_and_print_output(cmd)
    # if retcode != 0:
    #     raise Exception('KiloSort returned a non-zero exit code')

    sorting = se.KiloSortSortingExtractor(join(output_folder))
    return sorting
