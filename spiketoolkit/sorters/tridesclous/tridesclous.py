from pathlib import Path
import os
import shutil

from spiketoolkit.sorters.basesorter import BaseSorter
import spikeextractors as se

try:
    import tridesclous as tdc
    HAVE_TDC = True
except ImportError:
    HAVE_TDC = False


class TridesclousSorter(BaseSorter):
    """
    tridesclous is one of the more convinient, fast and elegant
    spike sorter.
    Everyone should test it.
    """
    
    sorter_name = 'tridesclous'
    installed = HAVE_TDC
    SortingExtractor_Class = se.TridesclousSortingExtractor
    
    _default_params = None  # later
    
    installation_mesg = """
       >>> pip install https://github.com/tridesclous/tridesclous/archive/master.zip
    
    More information on klusta at:
      * https://github.com/tridesclous/tridesclous
      * https://tridesclous.readthedocs.io
    """
    
    def __init__(self, **kargs):
        BaseSorter.__init__(self, **kargs)

    def _setup_recording(self, recording, output_folder):
        # reset the output folder
        if output_folder.is_dir():
            shutil.rmtree(str(output_folder))
        os.makedirs(str(output_folder))

        # save prb file:
        probe_file = output_folder / 'probe.prb'
        se.saveProbeFile(recording, probe_file, format='spyking_circus')
        
        # save binary file
        raw_filename = output_folder / 'raw_signals.raw'
        traces = recording.getTraces()
        dtype = traces.dtype
        with raw_filename.open('wb') as f:
            f.write(traces.T.tobytes())
        
        # initialize source and probe file
        tdc_dataio = tdc.DataIO(dirname=str(output_folder))
        nb_chan = recording.getNumChannels()
        
        tdc_dataio.set_data_source(type='RawData', filenames=[str(raw_filename)],
                                   dtype=dtype.str, sample_rate=recording.getSamplingFrequency(),
                                   total_channel=nb_chan)
        tdc_dataio.set_probe_file(str(probe_file))
        if self.debug:
            print(tdc_dataio)
    
    def _run(self, recording, output_folder):
        nb_chan = recording.getNumChannels()
    
        # check params and OpenCL when many channels
        use_sparse_template = False
        use_opencl_with_sparse = False
        if nb_chan >64: # this limit depend on the platform of course
            if tdc.cltools.HAVE_PYOPENCL:
                # force opencl
                self.params['fullchain_kargs']['preprocessor']['signalpreprocessor_engine'] = 'opencl'
                use_sparse_template = True
                use_opencl_with_sparse = True
            else:
                print('OpenCL is not available processing will be slow, try install it')
        
        tdc_dataio = tdc.DataIO(dirname=str(output_folder))
        # make catalogue
        chan_grps = list(tdc_dataio.channel_groups.keys())
        for chan_grp in chan_grps:
            cc = tdc.CatalogueConstructor(dataio=tdc_dataio, chan_grp=chan_grp)
            tdc.apply_all_catalogue_steps(cc, verbose=self.debug, **self.params)
            if self.debug:
                print(cc)
            cc.make_catalogue_for_peeler()
            
            # apply Peeler (template matching)
            initial_catalogue = tdc_dataio.load_catalogue(chan_grp=chan_grp)
            peeler = tdc.Peeler(tdc_dataio)
            peeler.change_params(catalogue=initial_catalogue,
                                 use_sparse_template=use_sparse_template,
                                 sparse_threshold_mad=1.5,
                                 use_opencl_with_sparse=use_opencl_with_sparse,)
            peeler.run(duration=None, progressbar=self.debug)


TridesclousSorter._default_params = {
    'fullchain_kargs': {
        'duration': 300.,
        'preprocessor': {
            'highpass_freq': None,
            'lowpass_freq': None,
            'smooth_size': 0,
            'chunksize': 1024,
            'lostfront_chunksize': 128,
            'signalpreprocessor_engine': 'numpy',
            'common_ref_removal':False,
        },
        'peak_detector': {
            'peakdetector_engine': 'numpy',
            'peak_sign': '-',
            'relative_threshold': 5.5,
            'peak_span': 0.0002,
        },
        'noise_snippet': {
            'nb_snippet': 300,
        },
        'extract_waveforms': {
            'n_left': -45,
            'n_right': 60,
            'mode': 'rand',
            'nb_max': 20000,
            'align_waveform': False,
        },
        'clean_waveforms': {
            'alien_value_threshold': 100.,
        },
    },
    'feat_method': 'peak_max',
    'feat_kargs': {},
    'clust_method': 'sawchaincut',
    'clust_kargs': {'kde_bandwith': 1.},
}
