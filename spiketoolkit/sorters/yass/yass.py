from pathlib import Path
import os
import shutil

from spiketoolkit.sorters.basesorter import BaseSorter
import spikeextractors as se

try:
    HAVE_YASS = True
except ImportError:
    HAVE_YASS = False


class YassSorter(BaseSorter):
    """
    """
    
    sorter_name = 'yass'
    installed = HAVE_YASS
    
    _default_params = None  # later
    
    installation_mesg = """
    pip install tensorflow
    pip install yass-algorithm[tf]
    """
    
    def __init__(self, **kargs):
        BaseSorter.__init__(self, **kargs)

    def _setup_recording(self, recording, output_folder):
        pass
    
    def _run(self, recording, output_folder):
        pass
    
    @staticmethod
    def get_result_from_folder(output_folder):
        sorting = se.YassSortingExtractor(output_folder)
        return sorting


YassSorter._default_params = {

}
