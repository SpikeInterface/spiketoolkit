
from .klusta import KlustaSorter
from .tridesclous import TridesclousSorter
from .mountainsort4 import Mountainsort4Sorter
from .ironclust import IronclustSorter
from .kilosort import KilosortSorter
from .kilosort2 import Kilosort2Sorter
from .spyking_circus import SpykingcircusSorter
from .herdingspikes import HerdingspikesSorter


sorter_full_list = [
    KlustaSorter,
    TridesclousSorter,
    Mountainsort4Sorter,
    IronclustSorter,
    KilosortSorter,
    Kilosort2Sorter,
    SpykingcircusSorter,
    HerdingspikesSorter
]

sorter_dict = {s.sorter_name: s for s in sorter_full_list}

installed_sorter_list = [s for s in sorter_full_list if s.installed]


# generic laucnher via function approach
def run_sorter(sorter_name_or_class, recording, output_folder=None, delete_output_folder=False,
               grouping_property=None, parallel=False, debug=False, **params):
    """
    Generic function to run a sorter via function approach.

    2 Usage with name or class:

    by name:
       >>> sorting = run_sorter('tridesclous', recording)

    by class:
       >>> sorting = run_sorter(TridesclousSorter, recording)

    """

    if isinstance(sorter_name_or_class, str):
        SorterClass = sorter_dict[sorter_name_or_class]
    elif sorter_name_or_class in sorter_full_list:
        SorterClass = sorter_name_or_class
    else:
        raise(ValueError('Unknown sorter'))

    sorter = SorterClass(recording=recording, output_folder=output_folder, grouping_property=grouping_property,
                         parallel=parallel, debug=debug, delete_output_folder=delete_output_folder)
    sorter.set_params(**params)
    sorter.run()
    sortingextractor = sorter.get_result()

    return sortingextractor


def available_sorters():
    return sorted(list(sorter_dict.keys()))


# make aliases
# this make compatible with the old by input/output functional approach

def run_klusta(*args, **kargs):
    return run_sorter('klusta', *args, **kargs)

def run_tridesclous(*args, **kargs):
    return run_sorter('tridesclous', *args, **kargs)

def run_mountainsort4(*args, **kargs):
    return run_sorter('mountainsort4', *args, **kargs)

def run_ironclust(*args, **kargs):
    return run_sorter('ironclust', *args, **kargs)

def run_kilosort(*args, **kargs):
    return run_sorter('kilosort', *args, **kargs)

def run_kilosort2(*args, **kargs):
    return run_sorter('kilosort2', *args, **kargs)

def run_spykingcircus(*args, **kargs):
    return run_sorter('spykingcircus', *args, **kargs)

def run_herdingspikes(*args, **kargs):
    return run_sorter('herdingspikes', *args, **kargs)
