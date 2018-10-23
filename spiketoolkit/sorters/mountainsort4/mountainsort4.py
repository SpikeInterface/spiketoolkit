import spiketoolkit as st
import time

def mountainsort4(
        recording, # The recording extractor
        detect_sign, # Use -1, 0, or 1, depending on the sign of the spikes in the recording
        adjacency_radius, # Use -1 to include all channels in every neighborhood
        freq_min=300, # Use None for no bandpass filtering
        freq_max=6000,
        whiten=True, # Whether to do channel whitening as part of preprocessing
        clip_size=50,
        detect_threshold=3,
        detect_interval=10, # Minimum number of timepoints between events detected on the same channel
        noise_overlap_threshold=0.15 # Use None for no automated curation
        ):
    try:
        import ml_ms4alg
    except ModuleNotFoundError:
        raise ModuleNotFoundError("\nTo use Mountainsort, install ml_ms4alg: \n\n"
                                  "\npip install ml_ms4alg\n"
                                  "\nMore information on Mountainsort at: "
                                  "\nhttps://github.com/flatironinstitute/mountainsort")

    t_start_proc = time.time()

    # Bandpass filter
    if freq_min is not None:
        recording=st.filters.bandpass_filter(recording=recording, freq_min=freq_min, freq_max=freq_max)

    # Whiten
    if whiten:
        recording=st.filters.whiten(recording=recording)

    # Sort
    sorting=ml_ms4alg.mountainsort4(
    recording=recording,
    detect_sign=detect_sign,
    adjacency_radius=adjacency_radius,
    clip_size=clip_size,
    detect_threshold=detect_threshold,
    detect_interval=detect_interval
    )

    # Curate
    if noise_overlap_threshold is not None:
        sorting=ml_ms4alg.mountainsort4_curation(
          recording=recording,
          sorting=sorting,
          noise_overlap_threshold=noise_overlap_threshold
        )
    print('Elapsed time: ', time.time() - t_start_proc)

    return sorting
