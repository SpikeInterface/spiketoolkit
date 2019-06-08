from spiketoolkit.sorters.basesorter import BaseSorter
import spiketoolkit as st
import spikeextractors as se
import copy

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

    _extra_gui_params = [
        {'name': 'clustering_bandwidth', 'type': 'float', 'value': 5.0, 'default': 5.0,
            'title': "Meanshift bandwidth"},
        {'name': 'clustering_alpha', 'type': 'float', 'value': 8.0, 'default': 8.0,
            'title': "Scalar for the PC components when clustering"},
        {'name': 'clustering_n_jobs', 'type': 'int', 'value': -1, 'default': -1,
            'title': "Number of cores. Default uses all cores."},
        {'name': 'clustering_bin_seeding', 'type': 'bool', 'value': True,
            'default': True, 'title': "Clustering bin seeding"},
        {'name': 'clustering_min_bin_freq', 'type': 'int', 'value': 8, 'default': 8,
            'title': "Minimum spikes per bin for bin seeding"},
        {'name': 'clustering_subset', 'type': 'int', 'value': None, 'default': None,
            'title': "Number of spikes used to build clusters. All by default."},
        {'name': 'left_cutout_time', 'type': 'float', 'value': 0.2, 'default': 0.2,
            'title': "Cutout size before peak (ms)"},
        {'name': 'right_cutout_time', 'type': 'float', 'value': 1.0, 'default': 1.0,
            'title': "Cutout size after peak (ms)"},
        {'name': 'detection_threshold', 'type': 'int', 'value': 20, 'default': 20,
            'title': "Detection threshold"},
        {'name': 'probe_masked_channels', 'type': 'list', 'value': [], 'default': [],
            'title': "Masked channels"},
        {'name': 'freq_min', 'type': 'float', 'value': 300.0, 'default': 300.0,
            'title': "Low-pass frequency"},
        {'name': 'freq_max', 'type': 'float', 'value': 6000.0, 'default': 6000.0,
            'title': "High-pass frequency"},
        {'name': 'filter', 'type': 'bool', 'value': False, 'default': False,
            'title': "Bandpass filters the recording if True"},
        {'name': 'pre_scale', 'type': 'bool', 'value': False, 'default': False,
            'title': "Scales recording traces to optimize HerdingSpikes performance"},
        {'name': 'pre_scale_value', 'type': 'float', 'value': 200.0, 'default': 200.0,
            'title': "Scale to apply in case of pre-scaling of traces"},
    ]

    _gui_params = copy.deepcopy(BaseSorter._gui_params)
    for param in _extra_gui_params:
        _gui_params.append(param)

    _default_params = None  # later

    installation_mesg = """
    More information on HerdingSpikes at:
      * https://github.com/mhhennig/hs2
    """

    def __init__(self, **kargs):
        BaseSorter.__init__(self, **kargs)

    def _setup_recording(self, recording, output_folder):
        p = self.params

        # Bandpass filter
        if p['filter'] and p['freq_min'] is not None and p['freq_max'] is not None:
            recording = st.preprocessing.bandpass_filter(
                recording=recording, freq_min=p['freq_min'], freq_max=p['freq_max'])

        if p['pre_scale']:
            recording = st.preprocessing.normalize_by_quantile(
                recording=recording, scale=p['pre_scale_value'],
                median=0.0, q1=0.05, q2=0.95
            )

        # this should have its name changed
        self.Probe = hs.probe.RecordingExtractor(
            recording,
            masked_channels=p['probe_masked_channels'],
            inner_radius=p['probe_inner_radius'],
            neighbor_radius=p['probe_neighbor_radius'],
            event_length=p['probe_event_length'],
            peak_jitter=p['probe_peak_jitter'])

    def _run(self, recording, output_folder):
        p = self.params

        self.H = hs.HSDetection(
            self.Probe, file_directory_name=str(output_folder),
            left_cutout_time=p['left_cutout_time'],
            right_cutout_time=p['right_cutout_time'],
            threshold=p['detection_threshold'],
            to_localize=True,
            num_com_centers=p['num_com_centers'],
            maa=p['maa'],
            ahpthr=p['ahpthr'],
            out_file_name=p['out_file_name'],
            decay_filtering=p['decay_filtering'],
            save_all=p['save_all'],
            amp_evaluation_time=p['amp_evaluation_time'],
            spk_evaluation_time=p['spk_evaluation_time']
        )

        self.H.DetectFromRaw(load=True, tInc=1000000)

        sorted_file = str(output_folder / 'HS2_sorted.hdf5')
        if(not self.H.spikes.empty):
            self.C = hs.HSClustering(self.H)
            self.C.ShapePCA(pca_ncomponents=p['pca_ncomponents'],
                            pca_whiten=p['pca_whiten'])
            self.C.CombinedClustering(
                alpha=p['clustering_alpha'],
                cluster_subset=p['clustering_subset'],
                bandwidth=p['clustering_bandwidth'],
                bin_seeding=p['clustering_bin_seeding'],
                n_jobs=p['clustering_n_jobs'],
                min_bin_freq=p['clustering_min_bin_freq']
            )
        else:
            self.C = hs.HSClustering(self.H)

        print('Saving to', sorted_file)
        self.C.SaveHDF5(sorted_file, sampling=self.Probe.fps)

    @staticmethod
    def get_result_from_folder(output_folder):
        return se.HS2SortingExtractor(output_folder / 'HS2_sorted.hdf5')


HerdingspikesSorter._default_params = {
    # core params
    'clustering_bandwidth': 5.0,
    'clustering_alpha': 8.0,
    'clustering_n_jobs': -1,
    'clustering_bin_seeding': True,
    'clustering_min_bin_freq': 8,
    'clustering_subset': None,
    'left_cutout_time': 0.2,
    'right_cutout_time': 1.0,
    'detection_threshold': 20,

    # extra probe params
    'probe_masked_channels': [],
    'probe_inner_radius': 75,
    'probe_neighbor_radius': 90,
    'probe_event_length': 0.25,
    'probe_peak_jitter': 0.1,

    # extra detection params
    'num_com_centers': 1,
    'maa': 5,
    'ahpthr': 10,
    'out_file_name': "HS2_detected",
    'decay_filtering': False,
    'save_all': False,
    'amp_evaluation_time': 0.2,
    'spk_evaluation_time': 1.4,

    # extra pca params
    'pca_ncomponents': 2,
    'pca_whiten': True,

    # bandpass filter
    'freq_min': 300.0,
    'freq_max': 6000.0,
    'filter': False,

    # rescale traces
    'pre_scale': False,  # TODO consider setting default to True
    'pre_scale_value': 200.0

}
