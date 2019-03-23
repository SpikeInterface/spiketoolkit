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
        # if output_folder.is_dir():
        #     pass
            # shutil.rmtree(str(output_folder))


        # this should have its name changed
        self.Probe = hs.probe.RecordingExtractor(recording,
                                                 **self.params['probe_params'])

    def _run(self, recording, output_folder):

        H = hs.HSDetection(self.Probe, file_directory_name=str(output_folder),
                           **self.params['detection_params'])  # risky

        H.DetectFromRaw(load=True)


        sorted_file = str(output_folder / 'HS2_sorted.hdf5')
        if(not H.spikes.empty):
            C = hs.HSClustering(H)
            C.ShapePCA(**self.params['pca_params'])
            C.CombinedClustering(**self.params['clustering_params'])
            C.SaveHDF5(sorted_file)
        else:
            C = hs.HSClustering(H)
            C.SaveHDF5(sorted_file)

    @staticmethod
    def get_result_from_folder(output_folder):
        sorting = se.HS2SortingExtractor(output_folder / 'HS2_sorted.hdf5')
        return sorting


HerdingspikesSorter._default_params = {
    'probe_params': {
        'inner_radius': 50,
        'neighbor_radius': 50,
        'noise_duration': 4,
        'spike_peak_duration': 4
    },

    'detection_params': {
        'to_localize': True,
        'cutout_start': 10,
        'cutout_end': 34,
        'threshold': 20,
        'num_com_centers': 3,
        'maa': 0,
        'maxsl': 13,
        'minsl': 2,
        'ahpthr': 5,
        'out_file_name': "HS2_detected",
        'decay_filtering': False,
        'save_all': False
    },

    'pca_params': {
        'pca_ncomponents': 2,
        'pca_whiten': True
    },

    'clustering_params': {
        'alpha': 6.0,
        'bandwidth': 6.0,
        'bin_seeding': False,
        'n_jobs': -1
    }
}
