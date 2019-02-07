
import os, shutil
from pathlib import Path
import spikeextractors as se

def tridesclous(
        recording,  # The recording extractor
        output_folder=None,
        debug=True,
        **params
        ):

    try:
        import tridesclous as tdc
    except ModuleNotFoundError:
        raise ModuleNotFoundError("tridesclous is not installed")

    output_folder = Path(output_folder)
    
    # reset the output folder
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    
    os.makedirs(output_folder)
    

    # save prb file:
    probe_file = output_folder / 'probe.prb'
    se.saveProbeFile(recording, probe_file, format='spyking_circus')
    
    # save binary file
    raw_filename = output_folder / 'raw_signals.raw'
    with open(raw_filename, mode='wb') as f:
        f.write(recording.getTraces().astype('float32').tobytes())
    
    # tdc job is done in a subfolder
    tdc_dirname = output_folder / 'tdc'
    # initialize source and probe file
    dataio = tdc.DataIO(dirname=tdc_dirname)
    print(recording.getSamplingFrequency())
    print(recording.getChannelIds())
    
    dataio.set_data_source(type='RawData', filenames=[raw_filename], dtype='float32',
                                    sample_rate=recording.getSamplingFrequency(),
                                    total_channel=len(recording.getChannelIds()))
    dataio.set_probe_file(probe_file)
    if debug:
        print(dataio)
    
    # make catalogue
    # TODO check which channel_group
    cc = tdc.CatalogueConstructor(dataio=dataio, chan_grp=0)
    tdc.apply_all_catalogue_steps(cc, verbose=debug, **params)
    if debug:
        print(cc)
    cc.make_catalogue_for_peeler()
    
    # apply Peeler (template matching)
    initial_catalogue = dataio.load_catalogue(chan_grp=0)
    peeler = tdc.Peeler(dataio)
    peeler.change_params(catalogue=initial_catalogue,
                        )
                        #~ use_sparse_template=True,
                        #~ sparse_threshold_mad=1.5,
                        #~ use_opencl_with_sparse=True,)
    peeler.run(duration=None, progressbar=True)
    
    sorting = se.TridesclousSortingExtractor(tdc_dirname)
    return sorting
    
    




tridesclous_default_params = {
    'fullchain_kargs':{
        'duration' : 300.,
        'preprocessor' : {
            'highpass_freq' : None,
            'lowpass_freq' : None,
            'smooth_size' : 0,
            'chunksize' : 1024,
            'lostfront_chunksize' : 128,
            'signalpreprocessor_engine' : 'numpy',
            'common_ref_removal':False,
        },
        'peak_detector' : {
            'peakdetector_engine' : 'numpy',
            'peak_sign' : '-',
            'relative_threshold' : 5.5,
            'peak_span' : 0.0002,
        },
        'noise_snippet' : {
            'nb_snippet' : 300,
        },
        'extract_waveforms' : {
            'n_left' : -45,
            'n_right' : 60,
            'mode' : 'rand',
            'nb_max' : 20000,
            'align_waveform' : False,
        },
        'clean_waveforms' : {
            'alien_value_threshold' : 100.,
        },
    },
    'feat_method': 'peak_max',
    'feat_kargs': {},
    'clust_method': 'sawchaincut',
    'clust_kargs' :{'kde_bandwith': 1.},
}
