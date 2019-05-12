"""
I need help here because:
  * there is no spikeextractor in spikeextractor module
  * there is no output_folder

Reading the code do not make evident if there is a persistency on disk.

"""
import spiketoolkit as st
from spiketoolkit.sorters.basesorter import BaseSorter
import spikeextractors as se
import copy

try:
    import ml_ms4alg
    HAVE_MS4 = True
except ImportError:
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
        'filter': False,
        'curation': True,
        'whiten': True,  # Whether to do channel whitening as part of preprocessing
        'clip_size': 50,
        'detect_threshold': 3,
        'detect_interval': 10,  # Minimum number of timepoints between events detected on the same channel
        'noise_overlap_threshold': 0.15,  # Use None for no automated curation'
    }

    _extra_params = [
        {'name': 'detect_sign', 'type': 'int', 'value':-1, 'default':-1,  'title': "Use -1, 0, or 1, depending on the sign of the spikes in the recording"},
        {'name': 'adjacency_radius', 'type': 'int', 'value':-1, 'default':-1,  'title': "Use -1 to include all channels in every neighborhood"},
        {'name': 'detect_sign', 'type': 'int', 'value':-1, 'default':-1,  'title': "Use -1, 0, or 1, depending on the sign of the spikes in the recording"},
        {'name': 'freq_min', 'type': 'float', 'value':300.0, 'default':300.0, 'title': "Low-pass frequency"},
        {'name': 'freq_max', 'type': 'float', 'value':6000.0, 'default':6000.0, 'title': "High-pass frequency"},
        {'name': 'filter', 'type': 'bool', 'value':False, 'default':False,  'title': "Bandpass filters the recording if True"},
        {'name': 'whiten', 'type': 'bool', 'value':True, 'default':True,  'title': "Whitens the recording if True"},
        {'name': 'clip_size', 'type': 'int', 'value':50, 'default':50,  'title': "Clip size"},
        {'name': 'detect_threshold', 'type': 'int', 'value':3, 'default':3,  'title': "Threshold for detection"},
        {'name': 'clip_size', 'type': 'int', 'value':50, 'default':50,  'title': "Clip size"},
        {'name': 'detect_interval', 'type': 'int', 'value':10, 'default':10,  'title': "Minimum number of timepoints between events detected on the same channel"},
        {'name': 'noise_overlap_threshold', 'type': 'float', 'value':0.15, 'default':0.15,  'title': "Use None for no automated curation"},
    ]

    _gui_params = copy.deepcopy(BaseSorter._gui_params)
    for param in _extra_params:
        _gui_params.append(param)
    installation_mesg = """
       >>> pip install ml_ms4alg

    More information on mountainsort at:
      * https://github.com/flatironinstitute/mountainsort
    """

    def __init__(self, **kargs):
        BaseSorter.__init__(self, **kargs)

    def _setup_recording(self, recording, output_folder):
        pass

    def _run(self, recording, output_folder):
        # Sort
        # alias to params
        p = self.params
        print(p)

        ind = self.recording_list.index(recording)

        # Bandpass filter
        if p['filter'] and p['freq_min'] is not None and p['freq_max'] is not None:
            recording = st.preprocessing.bandpass_filter(recording=recording, freq_min=p['freq_min'],
                                                         freq_max=p['freq_max'])

        # Whiten
        if p['whiten']:
            recording = st.preprocessing.whiten(recording=recording)

        # Check location
        if 'location' not in recording.get_channel_property_names():
            for i, chan in enumerate(recording.get_channel_ids()):
                recording.set_channel_property(chan, 'location', [0, i])

        sorting = ml_ms4alg.mountainsort4(
            recording=recording,
            detect_sign=p['detect_sign'],
            adjacency_radius=p['adjacency_radius'],
            clip_size=p['clip_size'],
            detect_threshold=p['detect_threshold'],
            detect_interval=p['detect_interval']
        )

        # Curate
        if p['noise_overlap_threshold'] is not None and p['curation'] is True:
            print('Curating')
            sorting = ml_ms4alg.mountainsort4_curation(
                recording=recording,
                sorting=sorting,
                noise_overlap_threshold=p['noise_overlap_threshold']
            )

        se.MdaSortingExtractor.write_sorting(sorting, str(output_folder / 'firings.mda'))

    @staticmethod
    def get_result_from_folder(output_folder):
        sorting = se.MdaSortingExtractor(str(output_folder / 'firings.mda'))
        return sorting
