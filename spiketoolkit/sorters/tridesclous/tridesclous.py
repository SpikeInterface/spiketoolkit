from pathlib import Path
import os
import shutil
import numpy as np

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

    _default_params = None  # later

    installation_mesg = """
       >>> pip install https://github.com/tridesclous/tridesclous/archive/master.zip

    More information on tridesclous at:
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
        se.save_probe_file(recording, probe_file, format='spyking_circus')

        # source file
        if isinstance(recording, se.BinDatRecordingExtractor) and recording._frame_first:
            # no need to copy
            raw_filename = recording._datfile
            dtype = recording._timeseries.dtype.str
            nb_chan = len(recording._channels)
            offset = recording._timeseries.offset
        else:
            if self.debug:
                print('Local copy of recording')
            # save binary file (chunk by hcunk) into a new file
            raw_filename = output_folder / 'raw_signals.raw'
            n_chan = recording.get_num_channels()
            chunksize = 2**24// n_chan
            se.write_binary_dat_format(recording, raw_filename, time_axis=0, dtype='float32', chunksize=chunksize)
            dtype='float32'
            offset = 0

        # initialize source and probe file
        tdc_dataio = tdc.DataIO(dirname=str(output_folder))
        nb_chan = recording.get_num_channels()

        tdc_dataio.set_data_source(type='RawData', filenames=[str(raw_filename)],
                                   dtype=dtype, sample_rate=recording.get_sampling_frequency(),
                                   total_channel=nb_chan, offset=offset)
        tdc_dataio.set_probe_file(str(probe_file))
        if self.debug:
            print(tdc_dataio)

    def _run(self, recording, output_folder):
        nb_chan = recording.get_num_channels()

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

    @staticmethod
    def get_result_from_folder(output_folder):
        sorting = se.TridesclousSortingExtractor(output_folder)
        return sorting


TridesclousSorter._default_params = {
    'fullchain_kargs': {
        'duration': 300.,
        'preprocessor': {
            'highpass_freq': 400.,
            'lowpass_freq': 5000.,
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
