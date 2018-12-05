import spikeextractors as se
import os
import time
import numpy as np
from pathlib import Path
from ..tools import _run_command_and_print_output, _spikeSortByProperty


def spyking_circus(
        recording,
        output_folder=None,  # Temporary working directory
        by_property=None,
        parallel=False,
        probe_file=None,
        file_name=None,
        detect_sign=-1,  # -1 - 1 - 0
        adjacency_radius=100,  # Channel neighborhood adjacency radius corresponding to geom file
        detect_threshold=6,  # Threshold for detection
        template_width_ms=3,  # Spyking circus parameter
        filter=True,
        merge_spikes=True,
        n_cores=None,
        electrode_dimensions=None,
        whitening_max_elts=1000,  # I believe it relates to subsampling and affects compute time
        clustering_max_elts=10000,  # I believe it relates to subsampling and affects compute time
):
    '''

    Parameters
    ----------
    recording
    output_folder
    by_property
    probe_file
    file_name
    detect_sign
    adjacency_radius
    detect_threshold
    template_width_ms
    filter
    merge_spikes
    n_cores
    electrode_dimensions
    whitening_max_elts
    clustering_max_elts

    Returns
    -------

    '''
    t_start_proc = time.time()
    if by_property is None:
        sorting = _spyking_circus(recording, output_folder, probe_file, file_name, detect_sign, adjacency_radius,
                                  detect_threshold, template_width_ms, filter, merge_spikes, n_cores,
                                  electrode_dimensions, whitening_max_elts, clustering_max_elts)
    else:
        if by_property in recording.getChannelPropertyNames():
            sorting = _spikeSortByProperty(recording, 'spyking-circus', by_property, parallel,
                                           output_folder=output_folder,
                                           probe_file=probe_file, file_name=file_name, detect_sign=detect_sign,
                                           adjacency_radius=adjacency_radius, detect_threshold=detect_threshold,
                                           template_width_ms=template_width_ms, filter=filter,
                                           merge_spikes=merge_spikes, n_cores=n_cores,
                                           electrode_dimensions=electrode_dimensions,
                                           whitening_max_elts=whitening_max_elts,
                                           clustering_max_elts=clustering_max_elts)
        else:
            print("Property not available! Running normal spike sorting")
            sorting = _spyking_circus(recording, output_folder, probe_file, file_name, detect_sign, adjacency_radius,
                                      detect_threshold, template_width_ms, filter, merge_spikes, n_cores,
                                      electrode_dimensions, whitening_max_elts, clustering_max_elts)

    print('Elapsed time: ', time.time() - t_start_proc)

    return sorting


def _spyking_circus(
        recording,
        output_folder=None,  # Temporary working directory
        probe_file=None,
        file_name=None,
        detect_sign=-1,  # -1 - 1 - 0
        adjacency_radius=100,  # Channel neighborhood adjacency radius corresponding to geom file
        detect_threshold=6,  # Threshold for detection
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
    source_dir = Path(__file__).parent
    if output_folder is None:
        output_folder = Path('spyking_circus')
    else:
        output_folder = Path(output_folder)
    if not output_folder.is_dir():
        output_folder.mkdir()

    # save prb file:
    if probe_file is None:
        probe_file = output_folder / 'probe.prb'
        se.saveProbeFile(recording, probe_file, format='spyking_circus', radius=adjacency_radius,
                         dimensions=electrode_dimensions)
    # save binary file
    if file_name is None:
        file_name = Path('recording')
    elif file_name.suffix == '.npy':
        file_name = file_name.stem
    np.save(output_folder / file_name, recording.getTraces().astype('float32'))

    if detect_sign < 0:
        detect_sign = 'negative'
    elif detect_sign > 0:
        detect_sign = 'positive'
    else:
        detect_sign = 'both'

    # set up spykingcircus config file
    with (source_dir / 'config_default.params').open('r') as f:
        circus_config = f.readlines()
    if merge_spikes:
        auto = 1e-5
    else:
        auto = 0
    circus_config = ''.join(circus_config).format(
        float(recording.getSamplingFrequency()), probe_file, template_width_ms, detect_threshold, detect_sign, filter,
        whitening_max_elts, clustering_max_elts, auto
    )
    with (output_folder / (file_name.name + '.params')).open('w') as f:
        f.writelines(circus_config)

    print('Running spyking circus...')
    if n_cores is None:
        n_cores = np.maximum(1, int(os.cpu_count() / 2))

    cmd = 'spyking-circus {} -c {} '.format(output_folder / (file_name.name + '.npy'), n_cores)
    cmd_merge = 'spyking-circus {} -m merging -c {} '.format(output_folder / (file_name.name + '.npy'), n_cores)
    # cmd_convert = 'spyking-circus {} -m converting'.format(join(output_folder, file_name+'.npy'))
    print(cmd)
    retcode = _run_command_and_print_output(cmd)
    if retcode != 0:
        raise Exception('Spyking circus returned a non-zero exit code')
    if merge_spikes:
        print(cmd_merge)
        retcode = _run_command_and_print_output(cmd_merge)
        if retcode != 0:
            raise Exception('Spyking circus merging returned a non-zero exit code')
    sorting = se.SpykingCircusSortingExtractor(output_folder / file_name)

    return sorting
