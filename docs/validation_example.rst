
Validation Tutorial
===================

This notebook shows how to use the spiketoolkit.validation module to:

1. compute biophysical metrics
2. compute quality metrics

.. code:: python

    import spikeextractors as se
    import spiketoolkit as st

First, let's create a toy example and spike sort it:

.. code:: python

    recording, sorting = se.example_datasets.toy_example(num_channels=4, duration=30)

.. code:: python

    sorting_KL = st.sorters.run_klusta(recording)


.. parsed-literal::

    'group' property is not available and it will not be saved.


1) Compute ISI ratio violations (biophysical metric)
----------------------------------------------------

.. code:: python

    ISI_ratios = st.validation.compute_ISI_violation_ratio(sorting_KL, recording.get_sampling_frequency())

.. code:: python

    for u_i, u in enumerate(sorting_KL.get_unit_ids()):
        print('Unit', u, 'ISI violation ratio', ISI_ratios[u_i])


.. parsed-literal::

    Unit 0 ISI violation ratio 0.1
    Unit 2 ISI violation ratio 0.0
    Unit 3 ISI violation ratio 0.0
    Unit 4 ISI violation ratio 0.0
    Unit 5 ISI violation ratio 0
    Unit 6 ISI violation ratio 0.0
    Unit 7 ISI violation ratio 0.0
    Unit 8 ISI violation ratio 0.0


2) Compute signal-to-noise ratio (quality metric)
-------------------------------------------------

.. code:: python

    snrs = st.validation.compute_unit_SNR(recording, sorting_KL)

.. code:: python

    for u_i, u in enumerate(sorting_KL.get_unit_ids()):
        print('Unit', u, 'SNR', snrs[u_i])


.. parsed-literal::

    Unit 0 SNR 21.800821413014827
    Unit 2 SNR 7.3074225607330625
    Unit 3 SNR 15.407656249800857
    Unit 4 SNR 8.529151798949432
    Unit 5 SNR 6.11031220236216
    Unit 6 SNR 6.110241154025549
    Unit 7 SNR 18.7016296447887
    Unit 8 SNR 4.931793645771355


Validation metrics are saved as unit property by default. If you don’t
want to save them as properties, you can add ``save_as_property=False``
in the function call.

.. code:: python

    for u in sorting_KL.get_unit_ids():
        print('Unit', u, 'SNR', sorting_KL.get_unit_property(u, 'snr'), 
              'ISI violation ratio', sorting_KL.get_unit_property(u, 'ISI_violation_ratio'))


.. parsed-literal::

    Unit 0 SNR 21.800821413014827 ISI violation ratio 0.1
    Unit 2 SNR 7.3074225607330625 ISI violation ratio 0.0
    Unit 3 SNR 15.407656249800857 ISI violation ratio 0.0
    Unit 4 SNR 8.529151798949432 ISI violation ratio 0.0
    Unit 5 SNR 6.11031220236216 ISI violation ratio 0
    Unit 6 SNR 6.110241154025549 ISI violation ratio 0.0
    Unit 7 SNR 18.7016296447887 ISI violation ratio 0.0
    Unit 8 SNR 4.931793645771355 ISI violation ratio 0.0

