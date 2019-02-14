
from .klusta import KlustaSorter, run_klusta
from .tridesclous import TridesclousSorter, run_tridesclous

sorter_full_list = [
    KlustaSorter,
    TridesclousSorter,
]

installed_sorter_list = [ s for s in sorter_full_list if s.installed]

