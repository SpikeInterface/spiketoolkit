
Sorters module
==============

This notebook shows how to use the spiketoolkit.sorters module to: 1.
check available sorters 2. check and set sorters parameters 3. run
sorters 4. use the spike sorter launcher 5. spike sort by property

.. code:: python

    # For development purposes, reload imported modules when source changes
    %load_ext autoreload
    %autoreload 2
    
    import spikeextractors as se
    import spiketoolkit as st
    import spikewidgets as sw
    import os
    import time
    from pprint import pprint


.. parsed-literal::

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload


Create a toy example dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    recording, sorting_true = se.example_datasets.toy_example(duration=60)

1) Check available sorters
--------------------------

.. code:: python

    print(st.sorters.available_sorters())


.. parsed-literal::

    ['herdingspikes', 'ironclust', 'kilosort', 'kilosort2', 'klusta', 'mountainsort4', 'spykingcircus', 'tridesclous']


This will list the sorters installed in the machine. Each spike sorter
is implemented in a class. To access the class names you can run:

.. code:: python

    st.sorters.installed_sorter_list




.. parsed-literal::

    [spiketoolkit.sorters.klusta.klusta.KlustaSorter,
     spiketoolkit.sorters.tridesclous.tridesclous.TridesclousSorter,
     spiketoolkit.sorters.mountainsort4.mountainsort4.Mountainsort4Sorter,
     spiketoolkit.sorters.ironclust.ironclust.IronclustSorter,
     spiketoolkit.sorters.kilosort.kilosort.KilosortSorter,
     spiketoolkit.sorters.kilosort2.kilosort2.Kilosort2Sorter,
     spiketoolkit.sorters.spyking_circus.spyking_circus.SpykingcircusSorter,
     spiketoolkit.sorters.herdingspikes.herdingspikes.HerdingspikesSorter]



2) Check and set sorters parameters
-----------------------------------

To check which parameters are available for each spike sorter you can
run:

.. code:: python

    default_ms4_params = st.sorters.Mountainsort4Sorter.default_params()
    pprint(default_ms4_params)


.. parsed-literal::

    {'adjacency_radius': -1,
     'clip_size': 50,
     'curation': True,
     'detect_interval': 10,
     'detect_sign': -1,
     'detect_threshold': 3,
     'filter': False,
     'freq_max': 6000,
     'freq_min': 300,
     'noise_overlap_threshold': 0.15,
     'whiten': True}


Parameters can be changed either by passing a full dictionary, or by
passing single arguments.

.. code:: python

    # Mountainsort4 spike sorting
    default_ms4_params['detect_threshold'] = 4
    default_ms4_params['curation'] = False
    
    # parameters set by params dictionary
    sorting_MS4 = st.sorters.run_mountainsort4(recording=recording, **default_ms4_params, 
                                               output_folder='tmp_MS4')


.. parsed-literal::

    {'detect_sign': -1, 'adjacency_radius': -1, 'freq_min': 300, 'freq_max': 6000, 'filter': False, 'curation': False, 'whiten': True, 'clip_size': 50, 'detect_threshold': 4, 'detect_interval': 10, 'noise_overlap_threshold': 0.15}
    Using 6 workers.
    Using tmpdir: /tmp/tmpjxyla5jm
    Num. workers = 6
    Preparing /tmp/tmpjxyla5jm/timeseries.hdf5...
    Preparing neighborhood sorters (M=4, N=1800000)...
    Neighboorhood of channel 1 has 4 channels.
    Neighboorhood of channel 0 has 4 channels.
    Neighboorhood of channel 2 has 4 channels.
    Neighboorhood of channel 3 has 4 channels.
    Detecting events on channel 4 (phase1)...
    Detecting events on channel 2 (phase1)...
    Detecting events on channel 3 (phase1)...
    Detecting events on channel 1 (phase1)...
    Elapsed time for detect on neighborhood: 0:00:00.199391
    Num events detected on channel 4 (phase1): 457
    Computing PCA features for channel 4 (phase1)...
    Elapsed time for detect on neighborhood: 0:00:00.210565
    Num events detected on channel 1 (phase1): 697
    Computing PCA features for channel 1 (phase1)...
    Elapsed time for detect on neighborhood: 0:00:00.218763
    Num events detected on channel 2 (phase1): 862
    Computing PCA features for channel 2 (phase1)...
    Elapsed time for detect on neighborhood: 0:00:00.235339
    Num events detected on channel 3 (phase1): 760
    Computing PCA features for channel 3 (phase1)...
    Clustering for channel 4 (phase1)...
    Clustering for channel 1 (phase1)...
    Found 5 clusters for channel 4 (phase1)...
    Computing templates for channel 4 (phase1)...
    Clustering for channel 3 (phase1)...
    Found 7 clusters for channel 1 (phase1)...
    Clustering for channel 2 (phase1)...
    Re-assigning events for channel 4 (phase1)...
    Computing templates for channel 1 (phase1)...
    Found 7 clusters for channel 3 (phase1)...
    Computing templates for channel 3 (phase1)...
    Re-assigning events for channel 1 (phase1)...
    Found 7 clusters for channel 2 (phase1)...
    Computing templates for channel 2 (phase1)...
    Re-assigning events for channel 3 (phase1)...
    Re-assigning events for channel 2 (phase1)...
    Re-assigning 1 events from 2 to 1 with dt=-1 (k=4)
    Re-assigning 20 events from 2 to 3 with dt=-3 (k=6)
    Neighboorhood of channel 0 has 4 channels.
    Neighboorhood of channel 2 has 4 channels.
    Neighboorhood of channel 1 has 4 channels.
    Neighboorhood of channel 3 has 4 channels.
    Computing PCA features for channel 4 (phase2)...
    Computing PCA features for channel 2 (phase2)...
    Computing PCA features for channel 1 (phase2)...
    Computing PCA features for channel 3 (phase2)...
    No duplicate events found for channel 3 in phase2
    No duplicate events found for channel 1 in phase2
    No duplicate events found for channel 2 in phase2
    No duplicate events found for channel 0 in phase2
    Clustering for channel 4 (phase2)...
    Clustering for channel 1 (phase2)...
    Found 2 clusters for channel 4 (phase2)...
    Found 3 clusters for channel 1 (phase2)...
    Clustering for channel 2 (phase2)...
    Clustering for channel 3 (phase2)...
    Found 5 clusters for channel 2 (phase2)...
    Found 5 clusters for channel 3 (phase2)...
    Preparing output...
    Done with ms4alg.
    Cleaning tmpdir::::: /tmp/tmpjxyla5jm


.. code:: python

    # parameters set by params dictionary
    sorting_MS4_10 = st.sorters.run_mountainsort4(recording=recording, detect_threshold=10, 
                                               output_folder='tmp_MS4')


.. parsed-literal::

    {'detect_sign': -1, 'adjacency_radius': -1, 'freq_min': 300, 'freq_max': 6000, 'filter': False, 'curation': True, 'whiten': True, 'clip_size': 50, 'detect_threshold': 10, 'detect_interval': 10, 'noise_overlap_threshold': 0.15}
    Using 6 workers.
    Using tmpdir: /tmp/tmpmpeuapff
    Num. workers = 6
    Preparing /tmp/tmpmpeuapff/timeseries.hdf5...
    Preparing neighborhood sorters (M=4, N=1800000)...
    Neighboorhood of channel 0 has 4 channels.
    Neighboorhood of channel 3 has 4 channels.
    Neighboorhood of channel 1 has 4 channels.
    Neighboorhood of channel 2 has 4 channels.
    Detecting events on channel 4 (phase1)...
    Detecting events on channel 1 (phase1)...
    Detecting events on channel 3 (phase1)...
    Detecting events on channel 2 (phase1)...
    Elapsed time for detect on neighborhood: 0:00:00.208714
    Elapsed time for detect on neighborhood: 0:00:00.208733
    Elapsed time for detect on neighborhood: 0:00:00.208675
    Num events detected on channel 2 (phase1): 152
    Num events detected on channel 1 (phase1): 1
    Computing PCA features for channel 2 (phase1)...
    Computing PCA features for channel 1 (phase1)...
    Num events detected on channel 4 (phase1): 148
    Computing PCA features for channel 4 (phase1)...
    Elapsed time for detect on neighborhood: 0:00:00.220430
    Num events detected on channel 3 (phase1): 142
    Computing PCA features for channel 3 (phase1)...
    Clustering for channel 1 (phase1)...
    Found 1 clusters for channel 1 (phase1)...
    Computing templates for channel 1 (phase1)...
    Clustering for channel 4 (phase1)...
    Found 1 clusters for channel 4 (phase1)...
    Re-assigning events for channel 1 (phase1)...
    Computing templates for channel 4 (phase1)...
    Re-assigning events for channel 4 (phase1)...
    Clustering for channel 2 (phase1)...
    Clustering for channel 3 (phase1)...
    Found 1 clusters for channel 3 (phase1)...
    Found 1 clusters for channel 2 (phase1)...
    Computing templates for channel 2 (phase1)...
    Computing templates for channel 3 (phase1)...
    Re-assigning events for channel 2 (phase1)...
    Re-assigning events for channel 3 (phase1)...
    Neighboorhood of channel 1 has 4 channels.
    Neighboorhood of channel 3 has 4 channels.
    Neighboorhood of channel 0 has 4 channels.
    Neighboorhood of channel 2 has 4 channels.
    Computing PCA features for channel 1 (phase2)...
    No duplicate events found for channel 0 in phase2
    Computing PCA features for channel 2 (phase2)...
    Computing PCA features for channel 3 (phase2)...
    Computing PCA features for channel 4 (phase2)...
    No duplicate events found for channel 3 in phase2
    Clustering for channel 1 (phase2)...
    No duplicate events found for channel 1 in phase2
    No duplicate events found for channel 2 in phase2
    Found 0 clusters for channel 1 (phase2)...
    Clustering for channel 4 (phase2)...
    Clustering for channel 3 (phase2)...
    Clustering for channel 2 (phase2)...
    Found 1 clusters for channel 4 (phase2)...
    Found 1 clusters for channel 2 (phase2)...
    Found 1 clusters for channel 3 (phase2)...
    Preparing output...
    Done with ms4alg.
    Cleaning tmpdir::::: /tmp/tmpmpeuapff
    Curating


.. code:: python

    print('Units found with threshold = 4:', sorting_MS4.get_unit_ids())
    print('Units found with threshold = 10:', sorting_MS4_10.get_unit_ids())


.. parsed-literal::

    Units found with threshold = 4: [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]
    Units found with threshold = 10: [1 2 3]


3) Run sorters
--------------

.. code:: python

    # SpyKING Circus spike sorting
    sorting_SC = st.sorters.run_spykingcircus(recording, output_folder='tmp_SC')
    print('Units found with Spyking Circus:', sorting_SC.get_unit_ids())

.. code:: python

    # KiloSort spike sorting (KILOSORT_PATH and NPY_MATLAB_PATH can be set as environment variables)
    sorting_KS = st.sorters.run_kilosort(recording, output_folder='tmp_KS')
    print('Units found with Kilosort:', sorting_KS.get_unit_ids())

.. code:: python

    # Kilosort2 spike sorting (KILOSORT2_PATH and NPY_MATLAB_PATH can be set as environment variables)
    sorting_KS2 = st.sorters.run_kilosort2(recording, output_folder='tmp_KS2')
    print('Units found with Kilosort2', sorting_KS2.get_unit_ids())

.. code:: python

    # Klusta spike sorting
    sorting_KL = st.sorters.run_klusta(recording, output_folder='tmp_KL')
    print('Units found with Klusta:', sorting_KL.get_unit_ids())

.. code:: python

    # IronClust spike sorting (IRONCLUST_PATH can be set as environment variables)
    sorting_IC = st.sorters.run_ironclust(recording, output_folder='tmp_IC')
    print('Units found with Ironclust:', sorting_IC.get_unit_ids())

.. code:: python

    # Tridesclous spike sorting
    sorting_TDC = st.sorters.run_tridesclous(recording, output_folder='tmp_TDC')
    print('Units found with Tridesclous:', sorting_TDC.get_unit_ids())

4) Use the spike sorter launcher
--------------------------------

The launcher enables to call any spike sorter with the same functions:
``run_sorter`` and ``run_sorters``. For running multiple sorters on the
same recording extractor or a collection of them, the ``run_sorters``
function can be used.

.. code:: python

    st.sorters.run_sorters?

.. code:: python

    recording_list = [recording]
    sorter_list = ['klusta', 'mountainsort4', 'tridesclous']

.. code:: python

    sorting_output = st.sorters.run_sorters(sorter_list, recording_list, working_folder='working')


.. parsed-literal::

    'group' property is not available and it will not be saved.
    {'detect_sign': -1, 'adjacency_radius': -1, 'freq_min': 300, 'freq_max': 6000, 'filter': False, 'curation': True, 'whiten': True, 'clip_size': 50, 'detect_threshold': 3, 'detect_interval': 10, 'noise_overlap_threshold': 0.15}
    Using 6 workers.
    Using tmpdir: /tmp/tmp2grdc4zr
    Num. workers = 6
    Preparing /tmp/tmp2grdc4zr/timeseries.hdf5...
    Preparing neighborhood sorters (M=4, N=1800000)...
    Neighboorhood of channel 0 has 4 channels.
    Neighboorhood of channel 3 has 4 channels.
    Neighboorhood of channel 1 has 4 channels.
    Neighboorhood of channel 2 has 4 channels.
    Detecting events on channel 1 (phase1)...
    Detecting events on channel 4 (phase1)...
    Detecting events on channel 2 (phase1)...
    Detecting events on channel 3 (phase1)...
    Elapsed time for detect on neighborhood: 0:00:00.191891
    Elapsed time for detect on neighborhood: 0:00:00.191369
    Num events detected on channel 3 (phase1): 2122
    Num events detected on channel 1 (phase1): 2913
    Computing PCA features for channel 3 (phase1)...
    Computing PCA features for channel 1 (phase1)...
    Elapsed time for detect on neighborhood: 0:00:00.230110
    Num events detected on channel 4 (phase1): 1919
    Computing PCA features for channel 4 (phase1)...
    Elapsed time for detect on neighborhood: 0:00:00.236283
    Num events detected on channel 2 (phase1): 1904
    Computing PCA features for channel 2 (phase1)...
    Clustering for channel 1 (phase1)...
    Clustering for channel 2 (phase1)...
    Clustering for channel 3 (phase1)...
    Clustering for channel 4 (phase1)...
    Found 10 clusters for channel 2 (phase1)...
    Computing templates for channel 2 (phase1)...
    Found 9 clusters for channel 1 (phase1)...
    Computing templates for channel 1 (phase1)...
    Re-assigning events for channel 2 (phase1)...
    Re-assigning 1 events from 2 to 1 with dt=-3 (k=5)
    Re-assigning 2 events from 2 to 4 with dt=-8 (k=10)
    Found 12 clusters for channel 3 (phase1)...
    Computing templates for channel 3 (phase1)...
    Re-assigning events for channel 3 (phase1)...
    Re-assigning 2 events from 3 to 2 with dt=-1 (k=6)
    Re-assigning events for channel 1 (phase1)...
    Re-assigning 11 events from 3 to 1 with dt=-5 (k=7)
    Found 6 clusters for channel 4 (phase1)...
    Computing templates for channel 4 (phase1)...
    Re-assigning events for channel 4 (phase1)...
    Neighboorhood of channel 0 has 4 channels.
    Neighboorhood of channel 2 has 4 channels.
    Neighboorhood of channel 3 has 4 channels.
    Neighboorhood of channel 1 has 4 channels.
    Computing PCA features for channel 3 (phase2)...
    Computing PCA features for channel 4 (phase2)...
    Computing PCA features for channel 1 (phase2)...
    No duplicate events found for channel 3 in phase2
    Computing PCA features for channel 2 (phase2)...
    No duplicate events found for channel 0 in phase2
    No duplicate events found for channel 2 in phase2
    No duplicate events found for channel 1 in phase2
    Clustering for channel 3 (phase2)...
    Clustering for channel 1 (phase2)...
    Clustering for channel 2 (phase2)...
    Clustering for channel 4 (phase2)...
    Found 5 clusters for channel 3 (phase2)...
    Found 5 clusters for channel 2 (phase2)...
    Found 3 clusters for channel 4 (phase2)...
    Found 3 clusters for channel 1 (phase2)...
    Preparing output...
    Done with ms4alg.
    Cleaning tmpdir::::: /tmp/tmp2grdc4zr
    Curating
    'group' property is not available and it will not be saved.
    probe allready in dir


.. parsed-literal::

    /home/alessiob/.virtualenvs/sorting/lib/python3.6/site-packages/tridesclous/dataio.py:215: ResourceWarning: unclosed file <_io.TextIOWrapper name='/home/alessiob/Documents/Codes/spike_sorting/spikeinterface/spiketoolkit/examples/working/output_folders/recording_0/tridesclous/default.prb' mode='r' encoding='UTF-8'>
      exec(open(probe_filename).read(), None, d)
    /home/alessiob/.virtualenvs/sorting/lib/python3.6/site-packages/tridesclous/dataio.py:215: ResourceWarning: unclosed file <_io.TextIOWrapper name='/home/alessiob/Documents/Codes/spike_sorting/spikeinterface/spiketoolkit/examples/working/output_folders/recording_0/tridesclous/probe.prb' mode='r' encoding='UTF-8'>
      exec(open(probe_filename).read(), None, d)


.. parsed-literal::

    order_clusters waveforms_rms
    make_catalogue 0.0313661671243608


.. parsed-literal::

    /home/alessiob/.virtualenvs/sorting/lib/python3.6/site-packages/tridesclous/dataio.py:215: ResourceWarning: unclosed file <_io.TextIOWrapper name='working/output_folders/recording_0/tridesclous/probe.prb' mode='r' encoding='UTF-8'>
      exec(open(probe_filename).read(), None, d)


.. code:: python

    for sorter, extractor in sorting_output['recording_0'].items():
        print(sorter, extractor.get_unit_ids())


.. parsed-literal::

    klusta [0, 2, 3, 4, 5, 6, 7]
    mountainsort4 [ 2  3  5  6  7 10 11 16]
    tridesclous [0, 1, 2, 3, 4]


5) Spike sort by property
-------------------------

Sometimes, you might want to sort your data depending on a specific
property of your recording channels.

For example, when using multiple tetrodes, a good idea is to sort each
tetrode separately. In this case, channels belonging to the same tetrode
will be in the same 'group'. Alternatively, for long silicon probes,
such as Neuropixels, you could sort different areas separately, for
example hippocampus and thalamus.

All this can be done by sorting by 'property'. Properties can be loaded
to the recording channels either manually (using the
``set_channel_property`` method, or by using a probe file. In this
example we will create a 16 channel recording and split it in four
tetrodes.

.. code:: python

    recording_tetrodes, sorting_true = se.example_datasets.toy_example(duration=60, num_channels=16)
    
    # initially there is no group information
    print(recording_tetrodes.get_channel_property_names())


.. parsed-literal::

    ['location']


.. code:: python

    # working in linux only
    !cat tetrode_16.prb


.. parsed-literal::

    channel_groups = {
        0: {
            'channels': [0,1,2,3],
        },
        1: {
            'channels': [4,5,6,7],
        },
        2: {
            'channels': [8,9,10,11],
        },
        3: {
            'channels': [12,13,14,15],
        }
    }


.. code:: python

    # load probe file to add group information
    recording_tetrodes = se.load_probe_file(recording_tetrodes, 'tetrode_16.prb')
    print(recording_tetrodes.get_channel_property_names())


.. parsed-literal::

    ['group', 'location']


We can now use the launcher to spike sort by the property 'group'. The
different groups can also be sorted in parallel, and the output sorting
extractor will have the same property used for sorting. Running in
parallel can speed up the computations.

.. code:: python

    t_start = time.time()
    sorting_tetrodes = st.sorters.run_sorter('klusta', recording_tetrodes, output_folder='tmp_tetrodes', 
                                             grouping_property='group', parallel=False)
    print('Elapsed time: ', time.time() - t_start)


.. parsed-literal::

    Elapsed time:  11.47568941116333


.. code:: python

    t_start = time.time()
    sorting_tetrodes_p = st.sorters.run_sorter('klusta', recording_tetrodes, output_folder='tmp_tetrodes', 
                                               grouping_property='group', parallel=True)
    print('Elapsed time parallel: ', time.time() - t_start)

.. code:: python

    print('Units non parallel: ', sorting_tetrodes.get_unit_ids())
    print('Units parallel: ', sorting_tetrodes_p.get_unit_ids())

Now that spike sorting is done, it's time to do some postprocessing,
comparison, and validation of the results!
