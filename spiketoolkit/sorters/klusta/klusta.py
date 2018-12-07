import spikeextractors as se
from pathlib import Path
import time
from spiketoolkit.sorters.tools import _run_command_and_print_output, _spikeSortByProperty, _call_command


def klusta(
        recording,  # The recording extractor
        output_folder=None,
        by_property=None,
        parallel=False,
        probe_file=None,
        file_name=None,
        adjacency_radius=None,
        threshold_strong_std_factor=5,
        threshold_weak_std_factor=2,
        detect_sign=-1,
        extract_s_before=16,
        extract_s_after=32,
        n_features_per_channel=3,
        pca_n_waveforms_max=10000,
        num_starting_clusters=50
):
    '''

    Parameters
    ----------
    recording
    output_folder
    by_property
    probe_file
    file_name
    threshold_strong_std_factor
    threshold_weak_std_factor
    detect_sign
    extract_s_before
    extract_s_after
    n_features_per_channel
    pca_n_waveforms_max
    num_starting_clusters

    Returns
    -------

    '''
    t_start_proc = time.time()
    if by_property is None:
        sorting = _klusta(recording, output_folder, probe_file, file_name, adjacency_radius, threshold_strong_std_factor,
                          threshold_weak_std_factor, detect_sign, extract_s_before, extract_s_after,
                          n_features_per_channel, pca_n_waveforms_max, num_starting_clusters)
    else:
        if by_property in recording.getChannelPropertyNames():
            sorting = _spikeSortByProperty(recording, 'klusta', by_property, parallel,
                                           output_folder=output_folder,
                                           probe_file=probe_file, file_name=file_name,
                                           adjacency_radius=adjacency_radius,
                                           threshold_strong_std_factor=threshold_strong_std_factor,
                                           threshold_weak_std_factor=threshold_weak_std_factor,
                                           detect_sign=detect_sign, extract_s_before=extract_s_before,
                                           extract_s_after=extract_s_after,
                                           n_features_per_channel=n_features_per_channel,
                                           pca_n_waveforms_max=pca_n_waveforms_max,
                                           num_starting_clusters=num_starting_clusters)
        else:
            print("Property not available! Running normal spike sorting")
            sorting = _klusta(recording, output_folder, probe_file, file_name, adjacency_radius,
                              threshold_strong_std_factor,
                              threshold_weak_std_factor, detect_sign, extract_s_before, extract_s_after,
                              n_features_per_channel, pca_n_waveforms_max, num_starting_clusters)

    print('Elapsed time: ', time.time() - t_start_proc)

    return sorting


def _klusta(
        recording,  # The recording extractor
        output_folder=None,
        probe_file=None,
        file_name=None,
        adjacency_radius=None,
        threshold_strong_std_factor=5,
        threshold_weak_std_factor=2,
        detect_sign=-1,
        extract_s_before=16,
        extract_s_after=32,
        n_features_per_channel=3,
        pca_n_waveforms_max=10000,
        num_starting_clusters=50
):
    try:
        import klusta
        import klustakwik2
    except ModuleNotFoundError:
        raise ModuleNotFoundError("\nTo use Klusta, install klusta and klustakwik2: \n\n"
                                  "\npip install klusta klustakwik\n"
                                  "\nMore information on klusta at: "
                                  "\nhttps://github.com/kwikteam/phy"
                                  "\nhttps://github.com/kwikteam/klusta")
    source_dir = Path(__file__).parent
    if output_folder is None:
        output_folder = Path('klusta')
    else:
        output_folder = Path(output_folder)
    if not output_folder.is_dir():
        output_folder.mkdir()

    # save prb file:
    if probe_file is None:
        probe_file = output_folder / 'probe.prb'
        se.saveProbeFile(recording, probe_file, format='klusta', radius=adjacency_radius)

    # save binary file
    if file_name is None:
        file_name = Path('recording')
    elif file_name.suffix == '.dat':
        file_name = file_name.stem
    se.writeBinaryDatFormat(recording, output_folder / file_name)

    if detect_sign < 0:
        detect_sign = 'negative'
    elif detect_sign > 0:
        detect_sign = 'positive'
    else:
        detect_sign = 'both'

    # set up klusta config file
    with (source_dir / 'config_default.prm').open('r') as f:
        klusta_config = f.readlines()

    klusta_config = ''.join(klusta_config).format(
        output_folder / file_name, probe_file, float(recording.getSamplingFrequency()), recording.getNumChannels(),
        "'float32'",
        threshold_strong_std_factor, threshold_weak_std_factor, "'" + detect_sign + "'", extract_s_before,
        extract_s_after,
        n_features_per_channel, pca_n_waveforms_max, num_starting_clusters
    )

    with (output_folder /'config.prm').open('w') as f:
        f.writelines(klusta_config)

    print('Running Klusta')
    cmd = 'klusta {} --overwrite'.format(output_folder /'config.prm')
    print(cmd)
    _call_command(cmd)
    if not (output_folder / (file_name.name + '.kwik')).is_file():
        raise Exception('Klusta did not run successfully')

    sorting = se.KlustaSortingExtractor(output_folder / (file_name.name + '.kwik'))

    return sorting
