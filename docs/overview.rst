Overview
========

Installation
------------

To get started with SpikeToolkit, you can install it with pip:

.. parsed-literal::
  pip install spiketoolkit

You can also install SpikeToolkit locally by cloning the repo into your code base. If you install SpikeToolkit locally, you need to run the setup.py file.

.. parsed-literal::
  git clone https://github.com/SpikeInterface/spiketoolkit.git
  cd spiketoolkit
  python setup.py install
  
Supported Spike Sorters
-----------------------

SpikeToolkit supports a variety of popular spike sorting algorithms. Adding new spike sorting algorithms is straightforward as well so we expect this list to grow in future versions.

- `Mountainsort 
  <https://github.com/flatironinstitute/mountainsort>`_

- `SpyKING circus 
  <https://github.com/spyking-circus/spyking-circus>`_

- `KiloSort 
  <https://github.com/cortex-lab/KiloSort>`_

- `Kilosort2 
  <https://github.com/MouseLand/Kilosort2>`_

- `HerdingSpikes2 
  <https://github.com/mhhennig/HS2>`_

- `Klusta 
  <https://github.com/kwikteam/klusta>`_

- `Ironclust 
  <https://github.com/jamesjun/ironclust)>`_

- `Tridesclous 
  <https://github.com/tridesclous/tridesclous>`_
  
To use these sorters, one must install each one as instructed by the developers. Installed sorting algorithms will automatically be detected by SpikeToolkit and be ready for use.

Contact
-------

If you have questions or comments, contact Alessio Buccino: alessiop.buccino@gmail.com
