import numpy as np
import scipy.signal as ss
import spikeextractors as se


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