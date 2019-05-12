[![Build Status](https://travis-ci.org/SpikeInterface/spiketoolkit.svg?branch=master)](https://travis-ci.org/SpikeInterface/spiketoolkit)

Alpha Development
Version 0.2.3


# SpikeToolkit

SpikeToolkit is a module that was designed to make running, curating, evaluating, and comparing popular spike sorting algorithms as simple as possible.

Its tools and functions are built using [spikeextractors](https://github.com/SpikeInterface/spikeextractors) objects, allowing for straightforward, standardized analysis and spike sorting.

## Getting Started 

To get started with SpikeToolkit, you can install it with pip:

```shell
pip install spiketoolkit
```
You can also install SpikeToolkit locally by cloning the repo into your code base. If you install SpikeToolkit locally, you need to run the setup.py file.

```shell
git clone https://github.com/SpikeInterface/spiketoolkit.git

cd spiketoolkit
python setup.py install
```

SpikeInterface allows the user to extract data from either raw or spike sorted datasets with a RecordingExtractor or SortingExtractor, respectively.

**Run spike sorting algorithms**

To run spike sorting algorithm, a `RecordingExtractor` object needs to be instantiated using the `spikeextractors` package

In this [example](https://github.com/SpikeInterface/spiketoolkit/tree/master/examples) from the examples repo, we show how to run several spike sorters on a toy dataset.

Each spike sorter must be installed separately. If one of the spike sorters is not installed, an error message detailing the installation procedure is given. Below is a list spike sorters we have made compatible with SpikeInterface:

- [Mountainsort](https://github.com/flatironinstitute/mountainsort)
- [SpyKING circus](https://github.com/spyking-circus/spyking-circus)
- [KiloSort](https://github.com/cortex-lab/KiloSort)
- [Klusta](https://github.com/kwikteam/klusta)
- [Ironclust](https://github.com/jamesjun/ironclust)
- [tridesclous](https://github.com/tridesclous/tridesclous)

SpikeToolkit is designed to make the spike sorting procedure _painless_ and easy. In the following example, 4 spike sorters (Mountainsrt, Spyking Circus, Kilosort and Tridesclous) are run on the same recordings.

```python
import spikeextractors as se
import spiketoolkit as st

# load recording using spikeextractors (e.g. Open Ephys recording)
recording = se.OpenEphysRecordingExtractor('path-to-open-ephys-folder')

# run spike sorters (with default parameters)
sorting_MS = st.sorters.run_mountainsort4(recording)
sorting_SC = st.sorters.run_spykingcircus(recording)
sorting_KS = st.sorters.run_kilosort(recording, kilosort_path='path-to-kilosort-matlab-installation')
sorting_TDC = st.sorters.run_tridesclous(recording)
```


**Curating spike sorting outputs**

Manual curation of spike sorting outputs is recommended for all algorithms. This includes visually inspecting the spike waveforms, correlograms, and clusters of each unit found in the recording.

With SpikeToolit you can export any sorting output to the  [phy](https://github.com/kwikteam/phy) template-gui, manually curate the data, and re-import the curated sorting output:

```python
# esport Mountainsort output to phy
st.export_to_phy(sorting_MS)
# curate the data running: phy template-gui path-to-exported-params.py
# reimport curated sorting output
sorting_MS_curated = se.PhysortingExtractors('path-to-created-phy-folder')
```

**Compare sorting outputs**

SpikeToolkit is designed to make spike sorting comparison and evaluation easy and straightforward. Using the `sorting_MS`, `sorting_SC`, and `sorting_SC` output from the previous section, one can run pairwise comparisons:

```python
comparison_MS_SC = st.comparison.SortingComparison(sorting_MS, sorting_SC)
```

The `SortingComparison` class finds best matching units based on the fraction of matched spikes. Units that are not matched to any other unit are assigned to -1.

Alternatively, one can run a multi-sorting comparison that finds units in agreement amongst multiple spike sorters:

```python
multi_comparison = st.comparison.MultiSortingComparison([sorting_MS, sorting_SC, sorting_KS])
# extract units shared among all 3 spike sorting outputs
agreement_sorting = multi_comparison.get_agreement_sorting(minimum_match=3)
```

## Interactive Example

To experiment with RecordingExtractors, SortingExtractors, and their associated tools, in a pre-installed environment, we have provided a [Collaborative environment](https://gist.github.com/magland/e43542fe2dfe856fd04903b9ff1f8e4e). If you click on the link and then click on "Open in Collab", you can run the notebook and try out the features of and tools for SpikeInterface.
<br/>


## Run test

pytest

### Authors

[Alessio Paolo Buccino](https://www.mn.uio.no/ifi/english/people/aca/alessiob/) - Center for Inegrative Neurolasticity (CINPLA), Department of Biosciences, Physics, and Informatics, University of Oslo, Oslo, Norway

[Cole Hurwitz](https://www.inf.ed.ac.uk/people/students/Cole_Hurwitz.html) - The Institute for Adaptive and Neural Computation (ANC), University of Edinburgh, Edinburgh, Scotland 

[Jeremy Magland](https://www.simonsfoundation.org/team/jeremy-magland/) - Center for Computational Biology (CCB), Flatiron Institute, New York, United States

[Matthias Hennig](http://homepages.inf.ed.ac.uk/mhennig/) - The Institute for Adaptive and Neural Computation (ANC), University of Edinburgh, Edinburgh, Scotland
<br/>
<br/>
For any correspondence, contact Alessio Buccino at alessiop.buccino@gmail.com

### Contributors

[Samuel Garcia](https://github.com/samuelgarcia) - Centre de Recherche en Neuroscience de Lyon (CRNL), Lyon, France
