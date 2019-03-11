import os
import shutil

from spiketoolkit.sorters.basesorter import BaseSorter
import spikeextractors as se

try:
    import herdingspikes as hs
    HAVE_HS = True
except ImportError:
    HAVE_HS = False


class HerdingspikesSorter(BaseSorter):
    """
    HerdingSpikes is a sorter based on estimated spike location, developed by
    researchers at the University of Edinburgh. It's a fast and scalable choice.

    See: HILGEN, Gerrit, et al. Unsupervised spike sorting for large-scale,
    high-density multielectrode arrays. Cell reports, 2017, 18.10: 2521-2532.
    """

    sorter_name = 'herdingspikes'
    installed = HAVE_HS
    SortingExtractor_Class = se.HS2SortingExtractor

    _default_params = None  # later

    installation_mesg = """
    More information on HerdingSpikes at:
      * https://github.com/mhhennig/hs2
    """

    def __init__(self, **kargs):
        BaseSorter.__init__(self, **kargs)

    def _setup_recording(self, recording, output_folder):
        # reset the output folder
        if output_folder.is_dir():
            shutil.rmtree(str(output_folder))
        os.makedirs(str(output_folder))

        # # save prb file:
        # probe_file = output_folder / 'probe.prb'
        # se.saveProbeFile(recording, probe_file, format='spyking_circus')
        #
        # # save binary file
        # raw_filename = output_folder / 'raw_signals.raw'
        # traces = recording.getTraces()
        # dtype = traces.dtype
        # with raw_filename.open('wb') as f:
        #     f.write(traces.T.tobytes())
        #
        # # initialize source and probe file
        # hs_dataio = hs.DataIO(dirname=str(output_folder))
        # nb_chan = recording.getNumChannels()
        #
        # hs_dataio.set_data_source(type='RawData', filenames=[str(raw_filename)],
        #                           dtype=dtype.str,
        #                           sample_rate=recording.getSamplingFrequency(),
        #                           total_channel=nb_chan)
        # hs_dataio.set_probe_file(str(probe_file))
        # if self.debug:
        #     print(hs_dataio)

        inner_radius = 50
        neighbor_radius = 50
        noise_duration = 4
        spike_peak_duration = 4

        # this should have its name changed
        self.Probe = hs.probe.RecordingExtractor(
            recording, inner_radius=inner_radius, neighbor_radius=neighbor_radius,
            noise_duration=noise_duration, spike_peak_duration=spike_peak_duration)

    def _run(self, recording, output_folder):
        # detection parameters
        to_localize = True
        cutout_start = 10
        cutout_end = 34
        threshold = 20
        num_com_centers = 3

        H = hs.HSDetection(self.Probe, to_localize, num_com_centers, cutout_start,
                           cutout_end, threshold, maa=0, maxsl=13, minsl=2, ahpthr=5,
                           out_file_name="HS2_detected", file_directory_name=output_folder,
                           decay_filtering=False, save_all=False)

        H.DetectFromRaw(load=True)

        C = hs.HSClustering(H)
        C.ShapePCA(pca_ncomponents=2, pca_whiten=True)
        C.CombinedClustering(alpha=6, bandwidth=6, bin_seeding=False, n_jobs=-1)

        sorted_file = str(output_folder / 'HS2_sorted.hdf5')
        C.SaveHDF5(sorted_file)


HerdingspikesSorter._default_params = {
    'fullchain_kargs': {
        'duration': 300.,
        'preprocessor': {
            'highpass_freq': None,
            'lowpass_freq': None,
            'smooth_size': 0,
            'chunksize': 1024,
            'lostfront_chunksize': 128,
            'signalpreprocessor_engine': 'numpy',
            'common_ref_removal': False,
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
