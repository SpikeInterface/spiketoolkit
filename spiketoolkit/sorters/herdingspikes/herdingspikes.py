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

        # this should have its name changed
        self.Probe = hs.probe.RecordingExtractor(recording,
                                                 **self.params['probe_params'])

    def _run(self, recording, output_folder):

        H = hs.HSDetection(self.Probe, file_directory_name=output_folder,
                           left_cutout_time=self.params['left_cutout_time'],
                           right_cutout_time=self.params['right_cutout_time'],
                           threshold=self.params['detection_threshold'],
                           **self.params['detection_params'])

        H.DetectFromRaw(load=True)

        C = hs.HSClustering(H)
        C.ShapePCA(**self.params['pca_params'])
        C.CombinedClustering(bandwidth=self.params['clustering_bandwidth'],
                             alpha=self.params['clustering_alpha'],
                             n_jobs=self.params['clustering_n_jobs'],
                             bin_seeding=self.params['clustering_bin_seeding'])

        sorted_file = str(output_folder / 'HS2_sorted.hdf5')
        C.SaveHDF5(sorted_file)

    @staticmethod
    def get_result_from_folder(output_folder):
        return se.HS2SortingExtractor(output_folder / 'HS2_sorted.hdf5')


HerdingspikesSorter._default_params = {
    'clustering_bandwidth': 6.0,
    'clustering_alpha': 6.0,
    'clustering_n_jobs': -1,
    'clustering_bin_seeding': False,
    'left_cutout_time': 1.0,
    'right_cutout_time': 2.2,
    'detection_threshold': 20,

    'probe_params': {
        'inner_radius': 50,
        'neighbor_radius': 50,
        'noise_duration': None,
        'spike_peak_duration': None,
        'event_length': 0.5,
        'peak_jitter': 0.2
    },

    'extra_detection_params': {
        'to_localize': True,
        'cutout_start': None,
        'cutout_end': None,
        'num_com_centers': 1,
        'maa': 0,
        'maxsl': None,
        'minsl': None,
        'ahpthr': 0,
        'out_file_name': "HS2_detected",
        'decay_filtering': False,
        'save_all': False,
        'amp_evaluation_time': 0.4,
        'spk_evaluation_time': 1.7
    },

    'extra_pca_params': {
        'pca_ncomponents': 2,
        'pca_whiten': True
    },

}
