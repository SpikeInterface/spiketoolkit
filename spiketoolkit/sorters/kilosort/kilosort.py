import spikeextractors as se
import os, sys
import time
import numpy as np
from pathlib import Path
from os.path import join
from ..tools import _run_command_and_print_output, _run_command_and_print_output_split,\
    _call_command, _call_command_split, _spikeSortByProperty


def kilosort(
        recording,
        output_folder=None,
        by_property=None,
        parallel=False,
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
    t_start_proc = time.time()
    if by_property is None:
        sorting = _kilosort(recording, output_folder, kilosort_path, npy_matlab_path, useGPU, probe_file, file_name,
                            detect_threshold, electrode_dimensions)
    else:
        if by_property in recording.getChannelPropertyNames():
            sorting = _spikeSortByProperty(recording, 'kilosort', by_property, parallel, output_folder=output_folder,
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
        klp = os.getenv('KILOSORT_PATH')
        if klp.startswith('"'):
            klp = klp[1:-1]
        kilosort_path = Path(klp)
    if npy_matlab_path is None or npy_matlab_path=='None':
        npp = os.getenv('NPY_MATLAB_PATH')
        if npp.startswith('"'):
            npp = npp[1:-1]
        npy_matlab_path = Path(npp)
    if not (Path(kilosort_path) / 'preprocessData.m').is_file() \
            or not (Path(npy_matlab_path) / 'npy-matlab' / 'readNPY.m').is_file():
        raise ModuleNotFoundError("\nTo use KiloSort, install KiloSort and npy-matlab from sources: \n\n"
                                  "\ngit clone https://github.com/cortex-lab/KiloSort\n"
                                  "\ngit clone https://github.com/kwikteam/npy-matlab\n"
                                  "and provide the installation path with the 'kilosort_path' and "
                                  "'npy_matlab_path' arguments or by setting the KILOSORT_PATH and NPY_MATLAB_PATH"
                                  "environment variables.\n+n"
                                  "\nMore information on KiloSort at: "
                                  "\nhttps://github.com/cortex-lab/KiloSort")
    source_dir = Path(__file__).parent
    if output_folder is None:
        output_folder = Path('kilosort')
    else:
        output_folder = Path(output_folder)
    if not output_folder.is_dir():
        output_folder.mkdir()
    output_folder = output_folder.absolute()


    if probe_file is not None:
        recording = se.loadProbeFile(recording, probe_file)

    # save binary file
    if file_name is None:
        file_name = Path('recording')
    elif file_name.suffix == '.dat':
        file_name = file_name.stem
    se.writeBinaryDatFormat(recording, output_folder / file_name, dtype='int16')

    # set up kilosort config files and run kilosort on data
    with (source_dir / 'kilosort_master.txt').open('r') as f:
        kilosort_master = f.readlines()
    with (source_dir / 'kilosort_config.txt').open('r') as f:
        kilosort_config = f.readlines()
    with (source_dir / 'kilosort_channelmap.txt').open('r') as f:
        kilosort_channelmap = f.readlines()

    nchan = recording.getNumChannels()
    dat_file = (output_folder / (file_name.name + '.dat')).absolute()
    kilo_thresh = detect_threshold
    Nfilt = (nchan // 32) * 32 * 8
    if Nfilt == 0:
        Nfilt = nchan * 8
    nsamples = 128 * 1024 + 64

    if useGPU:
        ug = 1
    else:
        ug = 0

    abs_channel = (output_folder / 'kilosort_channelmap.m').absolute()
    abs_config = (output_folder / 'kilosort_config.m').absolute()
    kilosort_path = kilosort_path.absolute()
    npy_matlab_path = npy_matlab_path.absolute() / 'npy-matlab'

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
        with (output_folder / fname).open('w') as f:
            f.writelines(value)

    # start sorting with kilosort
    print('Running KiloSort')
    cmd = "matlab -nosplash -nodisplay -r 'run {}; quit;'".format(output_folder / 'kilosort_master.m')
    print(cmd)
    if sys.platform == "win":
        cmd_list = ['matlab', '-nosplash', '-nodisplay', '-wait',
                    '-r','run {}; quit;'.format(output_folder / 'kilosort_master.m')]
    else:
        cmd_list = ['matlab', '-nosplash', '-nodisplay',
                    '-r', 'run {}; quit;'.format(output_folder / 'kilosort_master.m')]
    retcode = _run_command_and_print_output_split(cmd_list)
    if not (output_folder / 'spike_times.npy').is_file():
        raise Exception('KiloSort did not run successfully')
    sorting = se.KiloSortSortingExtractor(output_folder)
    return sorting
