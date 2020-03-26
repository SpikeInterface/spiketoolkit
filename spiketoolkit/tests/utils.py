import numpy as np
import scipy.signal as ss
import spikeextractors as se
import os, shutil


def check_signal_power_signal1_below_signal2(signals1, signals2, freq_range, fs):
    '''
    Check that spectrum power of signal1 is below the one of signal 2 in the range freq_range
    '''
    f1, pow1 = ss.welch(signals1, fs, nfft=1024)
    f2, pow2 = ss.welch(signals2, fs, nfft=1024)

    below = True

    for (p1, p2) in zip(pow1, pow2):

        r1_idxs = np.where((f1 > freq_range[0]) & (f1 <= freq_range[1]))
        r2_idxs = np.where((f2 > freq_range[0]) & (f2 <= freq_range[1]))

        sump1 = np.sum(p1[r1_idxs])
        sump2 = np.sum(p2[r2_idxs])

        if sump1 >= sump2:
            below = False
            break

    return below


def create_wf(min_val=-100, max_val=50, n_samples=100):
    '''
    Creates stereotyped waveform
    '''
    wf = np.zeros(n_samples)
    inter = n_samples // 4
    wf[:inter] = np.linspace(0, min_val, inter)
    wf[inter:3 * inter] = np.linspace(min_val, max_val, 2 * inter)
    wf[3 * inter:] = np.linspace(max_val, 0, n_samples - 3 * inter)

    return wf


def generate_template_with_random_amps(n_ch, wf):
    '''
    Creates stereotyped templates from waveform
    '''
    amps = []
    i = 1
    found = False
    while len(amps) < n_ch - 1 and i < 1000:
        a = np.random.rand()
        i = i + 1
        if a < 0.2 or a > 0.5:
            continue
        if sum(amps) + a < 0.9:
            amps.append(a)
    if len(amps) == n_ch - 1:
        amps.append(1 - sum(amps))
        found = True
        template = np.zeros((n_ch, len(wf)))
        for i, a in enumerate(amps):
            template[i] = a * wf
    else:
        template = []

    return template, amps, found


def create_signal_with_known_waveforms(n_channels=4, n_waveforms=2, n_wf_samples=100, duration=5, fs=30000):
    '''
    Creates stereotyped recording, sorting, with waveforms, templates, and max_chans
    '''
    a_min = [-200, -50]
    a_max = [10, 50]
    wfs = []

    # gen waveforms
    for w in range(n_waveforms):
        amp_min = np.random.randint(a_min[0], a_min[1])
        amp_max = np.random.randint(a_max[0], a_max[1])

        wf = create_wf(amp_min, amp_max, n_wf_samples)
        wfs.append(wf)

    # gen templates
    templates = []
    max_chans = []
    for wf in wfs:
        found = False
        while not found:
            template, amps, found = generate_template_with_random_amps(n_channels, wf)
        templates.append(template)
        max_chans.append(np.argmax(amps))

    templates = np.array(templates)
    n_samples = int(fs * duration)

    # gen spiketrains
    interval = 10 * n_wf_samples
    times = np.arange(interval, duration * fs - interval, interval).astype(int)
    labels = np.zeros(len(times)).astype(int)
    for i, wf in enumerate(wfs):
        labels[i::len(wfs)] = i

    timeseries = np.zeros((n_channels, n_samples))
    waveforms = []
    amplitudes = []
    for i, tem in enumerate(templates):
        idxs = np.where(labels == i)
        wav = []
        amps = []
        for t in times[idxs]:
            rand_val = np.random.randn() * 0.01 + 1
            timeseries[:, t - n_wf_samples // 2:t + n_wf_samples // 2] = rand_val * tem
            wav.append(rand_val * tem)
            amps.append(np.min(rand_val * tem))
        wav = np.array(wav)
        amps = np.array(amps)
        waveforms.append(wav)
        amplitudes.append(amps)

    rec = se.NumpyRecordingExtractor(timeseries=timeseries, sampling_frequency=fs)
    sort = se.NumpySortingExtractor()
    sort.set_times_labels(times=times, labels=labels)
    sort.set_sampling_frequency(fs)

    return rec, sort, waveforms, templates, max_chans, amplitudes


def create_fake_waveforms_with_known_pc():
    # HINT: start from Guassians in PC space and stereotyped waveforms and build dataset.
    pass


def check_recordings_equal(RX1, RX2):
    M = RX1.get_num_channels()
    N = RX1.get_num_frames()
    # get_channel_ids
    assert np.allclose(RX1.get_channel_ids(), RX2.get_channel_ids())
    # get_num_channels
    assert np.allclose(RX1.get_num_channels(), RX2.get_num_channels())
    # get_num_frames
    assert np.allclose(RX1.get_num_frames(), RX2.get_num_frames())
    # get_sampling_frequency
    assert np.allclose(RX1.get_sampling_frequency(), RX2.get_sampling_frequency())
    # get_traces
    assert np.allclose(RX1.get_traces(), RX2.get_traces())
    sf = 0
    ef = N
    ch = [RX1.get_channel_ids()[0], RX1.get_channel_ids()[-1]]
    assert np.allclose(RX1.get_traces(channel_ids=ch, start_frame=sf, end_frame=ef),
                       RX2.get_traces(channel_ids=ch, start_frame=sf, end_frame=ef))
    for f in range(0, RX1.get_num_frames(), 10):
        assert np.isclose(RX1.frame_to_time(f), RX2.frame_to_time(f))
        assert np.isclose(RX1.time_to_frame(RX1.frame_to_time(f)), RX2.time_to_frame(RX2.frame_to_time(f)))
    # get_snippets
    frames = [30, 50, 80]
    snippets1 = RX1.get_snippets(reference_frames=frames, snippet_len=20)
    snippets2 = RX2.get_snippets(reference_frames=frames, snippet_len=(10, 10))
    for ii in range(len(frames)):
        assert np.allclose(snippets1[ii], snippets2[ii])
        

def check_sorting_return_types(SX):
    unit_ids = SX.get_unit_ids()
    assert (all(isinstance(id, int) or isinstance(id, np.integer) for id in unit_ids))
    for id in unit_ids:
        train = SX.get_unit_spike_train(id)
        # print(train)
        assert (all(isinstance(x, int) or isinstance(x, np.integer) for x in train))


def check_sortings_equal(self, SX1, SX2):
    # get_unit_ids
    ids1 = np.sort(np.array(SX1.get_unit_ids()))
    ids2 = np.sort(np.array(SX2.get_unit_ids()))
    assert (np.allclose(ids1, ids2))
    for id in ids1:
        train1 = np.sort(SX1.get_unit_spike_train(id))
        train2 = np.sort(SX2.get_unit_spike_train(id))
        assert np.array_equal(train1, train2)


def check_dumping(extractor):
    extractor.dump(file_name='test.json')
    extractor_loaded = se.load_extractor_from_json('test.json')

    if 'Recording' in str(type(extractor)):
        check_recordings_equal(extractor, extractor_loaded)
    elif 'Sorting' in str(type(extractor)):
        check_sortings_equal(extractor, extractor_loaded)


def create_dumpable_recording(duration=10, num_channels=4, K=10, seed=0, folder='test', recording=None):
    if recording is not None:
        rec = recording
    else:
        rec, sort = se.example_datasets.toy_example(duration=duration, num_channels=num_channels, K=K, seed=seed)

    if 'location' not in rec.get_shared_channel_property_names():
        rec.set_channel_locations(channel_ids=rec.get_channel_ids(),
                                  locations=np.random.randn(rec.get_num_channels(), 2))

    se.MdaRecordingExtractor.write_recording(rec, folder)
    rec_mda = se.MdaRecordingExtractor(folder)

    return rec_mda


def create_dumpable_sorting(duration=10, num_channels=4, K=10, seed=0, folder='test', fs=30000, sorting=None):
    if sorting is not None:
        sort = sorting
    else:
        rec, sort = se.example_datasets.toy_example(duration=duration, num_channels=num_channels, K=K, seed=seed)
    if not os.path.isdir(folder):
        os.mkdir(folder)
    se.NpzSortingExtractor.write_sorting(sort, folder + '/sorting.npz')
    sort_npz = se.NpzSortingExtractor(folder + '/sorting.npz')

    return sort_npz


def create_dumpable_extractors(duration=10, num_channels=4, K=10, seed=0, folder='test'):
    rec, sort = se.example_datasets.toy_example(duration=duration, num_channels=num_channels, K=K, seed=seed)

    se.MdaRecordingExtractor.write_recording(rec, folder)
    rec_mda = se.MdaRecordingExtractor(folder)
    se.NpzSortingExtractor.write_sorting(sort, folder + '/sorting.npz')
    sort_npz = se.NpzSortingExtractor(folder + '/sorting.npz')

    return rec_mda, sort_npz
