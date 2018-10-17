import spikeinterface as si
import os
from os.path import join
import time

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
    source_dir = os.path.dirname(os.path.realpath(__file__))

    if output_folder is None:
        output_folder = os.getcwd()
    elif not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # save prb file:
    if probe_file is None:
        si.saveProbeFile(recording, join(output_folder, 'probe.prb'), format='klusta')
        probe_file = join(output_folder, 'probe.prb')
    # save binary file
    if file_name is None:
        file_name = join(output_folder, 'recording')
    elif file_name.endswith('.dat'):
        file_name = file_name[file_name.find('.dat')]
    si.writeBinaryDatFormat(recording, file_name)



    # set up klusta config file
    with open(join(source_dir, 'config_default.prm'), 'r') as f:
        klusta_config = f.readlines()

    klusta_config = ''.join(klusta_config).format(
        file_name, probe_file, float(recording.getSamplingFrequency()), recording.getNumChannels(), "'float32'",
        threshold_strong_std_factor, threshold_weak_std_factor, "'"+detect_spikes+"'", extract_s_before, extract_s_after,
        n_features_per_channel, pca_n_waveforms_max, num_starting_clusters
    )

    with open(join(output_folder, 'config.prm'), 'w') as f:
        f.writelines(klusta_config)

    print('Running klusta')
    try:
        import klusta
        # import klustakwik2
    except ImportError:
        raise ImportError('Install klusta and klustakwik2 packages to run klusta option')

    import subprocess
    try:
        t_start_proc = time.time()
        subprocess.check_output(['klusta', join(output_folder, 'config.prm'), '--overwrite'])
        processing_time = time.time() - t_start_proc
        print('Elapsed time: ', processing_time)
    except subprocess.CalledProcessError as e:
        raise Exception(e.output)

    sorting = si.KlustaSortingExtractor(join(output_folder, file_name +'.kwik'))

    return sorting
