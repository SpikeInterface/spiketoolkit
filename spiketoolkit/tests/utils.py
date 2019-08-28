import numpy as np
import scipy.signal as ss


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

