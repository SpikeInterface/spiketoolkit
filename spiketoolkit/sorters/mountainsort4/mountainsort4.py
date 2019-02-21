"""
I need help here because:
  * there is no spikeextractor in spikeextractor module
  * there is no output_folder

Reading the code do not make evident if there is a persistency on disk.

"""
import spiketoolkit as st
from spiketoolkit.sorters.basesorter import BaseSorter
import spikeextractors as se

try:
    import ml_ms4alg
    HAVE_MS4 = True
except ModuleNotFoundError:
    HAVE_MS4 = False


class Mountainsort4Sorter(BaseSorter):
    """
    Mountainsort
    """
    
    sorter_name = 'mountainsort4'
    installed = HAVE_MS4
    
    SortingExtractor_Class = None # there is not extractor !!!!!!!!!!!!!!!!!!!!!!!!
    
    _default_params = {
        'detect_sign': -1,  # Use -1, 0, or 1, depending on the sign of the spikes in the recording
        'adjacency_radius': -1,  # Use -1 to include all channels in every neighborhood
        'freq_min': 300,  # Use None for no bandpass filtering
        'freq_max': 6000,
        'whiten': True,  # Whether to do channel whitening as part of preprocessing
        'clip_size': 50,
        'detect_threshold': 3,
        'detect_interval': 10,  # Minimum number of timepoints between events detected on the same channel
        'noise_overlap_threshold': 0.15,  # Use None for no automated curation'
        'parallel': True
    }
    
    installation_mesg = """
       >>> pip install ml_ms4alg
    
    More information on mountainsort at:
      * https://github.com/flatironinstitute/mountainsort
    """
    
    def __init__(self, **kargs):
        BaseSorter.__init__(self, **kargs)

    def _setup_recording(self, recording, output_folder):
        # alias to params
        p = self.params
        # Bandpass filter
        if p['freq_min'] is not None:
            recording = st.preprocessing.bandpass_filter(recording=recording, freq_min=p['freq_min'],
                                                         freq_max=p['freq_max'])

        # Whiten
        if p['whiten']:
            recording = st.preprocessing.whiten(recording=recording)

        # Check location
        if 'location' not in recording.getChannelPropertyNames():
            for i, chan in enumerate(recording.getChannelIds()):
                recording.setChannelProperty(chan, 'location', [0, i])

        self.recording = recording
    
    def _run(self, recording, output_folder):
        # Sort
        p = self.params
        sorting = ml_ms4alg.mountainsort4(
            recording=self.recording,
            detect_sign=p['detect_sign'],
            adjacency_radius=p['adjacency_radius'],
            clip_size=p['clip_size'],
            detect_threshold=p['detect_threshold'],
            detect_interval=p['detect_interval']
        )

        # Curate
        if p['noise_overlap_threshold'] is not None:
            sorting = ml_ms4alg.mountainsort4_curation(
                recording=self.recording,
                sorting=sorting,
                noise_overlap_threshold=p['noise_overlap_threshold']
            )
        self.sorting = sorting


    def _get_one_result(self, recording, output_folder):
        # overwrite the SorterBase.get_result
        return self.sorting
