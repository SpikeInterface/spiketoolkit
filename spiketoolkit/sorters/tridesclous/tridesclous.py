from pathlib import Path
import os
import shutil

from spiketoolkit.sorters.basesorter import BaseSorter
from spiketoolkit.sorters.tools import _run_command_and_print_output, _spikeSortByProperty, _call_command
import spikeextractors as se

try:
    import tridesclous as tdc
    HAVE_TDC = True
except ModuleNotFoundError:
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
    
    _default_params = None # later
    
    installation_mesg = """
       >>> pip install tridesclous
    
    More information on klusta at:
      * https://github.com/tridesclous/tridesclous
      * https://tridesclous.readthedocs.io
    """
    
    def __init__(self, **kargs):
        BaseSorter.__init__(self, **kargs)

    def set_params(self, **params):
        self.params = params
    
    
    def _setup_recording(self):
        # reset the output folder
        if os.path.exists(self.output_folder):
            shutil.rmtree(self.output_folder)
        
        os.makedirs(self.output_folder)

        # save prb file:
        probe_file = self.output_folder / 'probe.prb'
        print('probe_file', probe_file)
        se.saveProbeFile(self.recording, probe_file, format='spyking_circus')
        
        # save binary file
        raw_filename = self.output_folder / 'raw_signals.raw'
        traces = self.recording.getTraces()
        dtype = traces.dtype
        with open(raw_filename, mode='wb') as f:
            f.write(traces.T.tobytes())
        
        # initialize source and probe file
        self.tdc_dataio = tdc.DataIO(dirname=self.output_folder)
        nb_chan = self.recording.getNumChannels()
        
        self.tdc_dataio.set_data_source(type='RawData', filenames=[raw_filename],
                        dtype=dtype.str, sample_rate=self.recording.getSamplingFrequency(),
                                        total_channel=nb_chan)
        self.tdc_dataio.set_probe_file(probe_file)
        if self.debug:
            print(self.tdc_dataio)        
        
    def _run(self):
        nb_chan = self.recording.getNumChannels()
    
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
        
        # make catalogue
        # TODO check which channel_group
        cc = tdc.CatalogueConstructor(dataio=self.tdc_dataio, chan_grp=0)
        tdc.apply_all_catalogue_steps(cc, verbose=self.debug, **self.params)
        if self.debug:
            print(cc)
        cc.make_catalogue_for_peeler()
        
        # apply Peeler (template matching)
        initial_catalogue = self.tdc_dataio.load_catalogue(chan_grp=0)
        peeler = tdc.Peeler(self.tdc_dataio)
        peeler.change_params(catalogue=initial_catalogue,
                            use_sparse_template=use_sparse_template,
                            sparse_threshold_mad=1.5,
                            use_opencl_with_sparse=use_opencl_with_sparse,)
        peeler.run(duration=None, progressbar=True)


def run_tridesclous(
        recording,
        output_folder=None,
        by_property=None,
        parallel=False,
        debug=False,
        **params):
    print('rec', recording)
    sorter = TridesclousSorter(recording=recording, output_folder=output_folder,
                                    by_property=by_property, parallel=parallel, debug=debug)
    sorter.set_params(**params)
    sorter.run()
    sortingextractor = sorter.get_result()
    
    return sortingextractor



TridesclousSorter._default_params = {
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
