import spikeinterface as si
import os
from os.path import join
import time
from spiketoolkit.sorters.tools import run_command_and_print_output

def klusta(
        recording, # The recording extractor
        output_folder=None,
        probe_file=None,
        file_name=None,
        threshold_strong_std_factor=5,
        threshold_weak_std_factor=2,
        detect_spikes='negative',
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
    source_dir = os.path.dirname(os.path.realpath(__file__))

    if output_folder is None:
        output_folder = 'klusta'
    else:
        output_folder = join(output_folder, 'klusta')

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # save prb file:
    if probe_file is None:
        si.saveProbeFile(recording, join(output_folder, 'probe.prb'), format='klusta')
        probe_file = join(output_folder, 'probe.prb')
    # save binary file
    if file_name is None:
        file_name = 'recording'
    elif file_name.endswith('.dat'):
        file_name = file_name[file_name.find('.dat')]
    si.writeBinaryDatFormat(recording, join(output_folder, file_name))

    # set up klusta config file
    with open(join(source_dir, 'config_default.prm'), 'r') as f:
        klusta_config = f.readlines()

    klusta_config = ''.join(klusta_config).format(
        join(output_folder, file_name), probe_file, float(recording.getSamplingFrequency()), recording.getNumChannels(), "'float32'",
        threshold_strong_std_factor, threshold_weak_std_factor, "'"+detect_spikes+"'", extract_s_before, extract_s_after,
        n_features_per_channel, pca_n_waveforms_max, num_starting_clusters
    )

    with open(join(output_folder, 'config.prm'), 'w') as f:
        f.writelines(klusta_config)

    print('Running Klusta')
    t_start_proc = time.time()
    cmd = 'klusta {} --overwrite'.format(join(output_folder, 'config.prm'))
    print(cmd)
    retcode = run_command_and_print_output(cmd)
    if retcode != 0:
        raise Exception('Klusta returned a non-zero exit code')
    print('Elapsed time: ', time.time() - t_start_proc)

    sorting = si.KlustaSortingExtractor(join(output_folder, file_name +'.kwik'))

    return sorting
