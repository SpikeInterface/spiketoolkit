[![Build Status](https://travis-ci.org/SpikeInterface/spiketoolkit.svg?branch=master)](https://travis-ci.org/SpikeInterface/spiketoolkit)

Alpha Development
Version 0.1.7


# SpikeToolkit

SpikeToolkit is a module that enables to run several spike sorting algoithms, curate, evaluate, and compare their outputs.

It is interfaced with SpikeExtractors objects, and the spike sorting pipeline easy and standardized among different software packages. 
<br/>
<br/>
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

To run spike sorting algorithm, a `RecordingExtractor` object needs to be inetantiated using the `spikeextracor` package

In this [example](https://github.com/colehurwitz31/spikeinterface/blob/master/examples/run_all_sorters.ipynb) from the examples repo, we show how to use run several of the implemented spike sorters.

Each spike sorter must be installed separately and if not installed, an Error message with the installation procedure is shown. This is a list of currently available spike sorters:

- [Mountainsort](https://github.com/flatironinstitute/mountainsort)
- [SpyKING circus](https://github.com/spyking-circus/spyking-circus)
- [KiloSort](https://github.com/cortex-lab/KiloSort)
- [Klusta](https://github.com/kwikteam/klusta)
- [Ironclust](https://github.com/jamesjun/ironclust)

SpikeToolkit is designed to make the spike sorting procedure _painless_ and easy. In the following example, 3 spike sorters (Mountainsrt, Spyking Circus, and Kilosort) are run on the same recordings.

```python
import spikeextractor as se
import spiketoolkit as st

# load recording using spikeextractors (e.g. Open Ephys recording)
recording = se.OpenEphysRecordingExtractor('path-to-open-ephys-folder')
# run spike sorters (with default parameters)
sorting_MS = st.sorters.mountainsort4(recording)
sorting_SC = st.sorters.spyking_circus(recording)
sorting_KS = st.sorters.kilosort(recording, kilosort_path='pat-to-kilosort-matlab-installation')
```

Other parameters are exposed using arguments. In order to find out which parameters are available, you can run in ipython:
```python
st.sorters.mountainsort?
st.sorters.spyking_circus?
st.sorters.kilosort?
```

**Curate spike sorting output**

Manual curation of spike sorting output is reccommended to at least visually checking spike waveforms, correlograms, and clustering. 
With SpikeToolit you can export any sorting output using the  [phy](https://github.com/kwikteam/phy) template-gui, manually curate the data, and reimport the curated sorting output using SpikeExtractors.
```python
# esport Mountainsort output to phy
st.exportToPhy(sorting_MS)
# curate the data running: phy template-gui path-to-exported-params.py
# reimport curated sorting output
sorting_MS_curated = se.PhysortingExtractors('path-to-created-phy-folder')
```

**Compare sorting outputs**

SpikeToolkit is designed to make spike sorting comparison and evaluation easy and straightforward. Using the `sorting_MS`, `sorting_SC`, and `sorting_SC` output from the previous section one can run pairwise comparisons by:
```python
comparison_MS_SC = st.comparison.SortingComparison(sorting_MS, sorting_SC)
```
The `SortingComparison` class finds best matching unit based on the fraction of matching spikes. Units that are not matched are assigned to -1.
Alternatively, one can find units in agreement with multiple psike sorters:
```python
multi_comparison = st.comparison.MultiSortingComparison([sorting_MS, sorting_SC, sorting_KS])
# extract units shared among all 3 spike sorting outputs
agreement_sorting = multi_comparison.getAgreementSorting(minimum_match=3)
```

## Interactive Example

To experiment with RecordingExtractors and SortingExtractors in a pre-installed environment, we have provided a [Collaborative environment](https://gist.github.com/magland/e43542fe2dfe856fd04903b9ff1f8e4e). If you click on the link and then click on "Open in Collab", you can run the notebook and try out the features of and tools for SpikeInterface.
<br/>


### Authors

[Alessio Paolo Buccino](https://www.mn.uio.no/ifi/english/people/aca/alessiob/) - Center for Inegrative Neurolasticity (CINPLA), Department of Biosciences, Physics, and Informatics, University of Oslo, Oslo, Norway

[Cole Hurwitz](https://www.inf.ed.ac.uk/people/students/Cole_Hurwitz.html) - The Institute for Adaptive and Neural Computation (ANC), University of Edinburgh, Edinburgh, Scotland 

[Jeremy Magland](https://www.simonsfoundation.org/team/jeremy-magland/) - Center for Computational Biology (CCB), Flatiron Institute, New York, United States

[Matthias Hennig](http://homepages.inf.ed.ac.uk/mhennig/) - The Institute for Adaptive and Neural Computation (ANC), University of Edinburgh, Edinburgh, Scotland
<br/>
<br/>
For any correspondence, contact Alessio Buccino at alessiop.buccino@gmail.com
