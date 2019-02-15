
from .klusta import KlustaSorter, run_klusta
from .tridesclous import TridesclousSorter, run_tridesclous
# from .mountainsort4 import Mountainsort4Sorter, run_mountainsort4
from .ironclust import IronclustSorter, run_ironclust
from .kilosort import KilosortSorter, run_kilosort
from .spyking_circus import SpykingcircusSorter, run_spykingcircus


sorter_full_list = [
    KlustaSorter,
    TridesclousSorter,
    # Mountainsort4Sorter,
    IronclustSorter,
    KilosortSorter,
    SpykingcircusSorter,
]

installed_sorter_list = [ s for s in sorter_full_list if s.installed]

