import numpy as np
import quantities as pq
from quantities import Quantity
import neo
import elephant
import matplotlib.pylab as plt
import spikeinterface as si

def cc_max_spiketrains(st, st_id, other_st, max_lag=20*pq.ms):
    '''

    Parameters
    ----------
    st
    st_id
    other_st

    Returns
    -------

    '''
    from elephant.spike_train_correlation import cch, corrcoef

    cc_vec = np.zeros(len(other_st))
    for p, p_st in enumerate(other_st):
        # cc, bin_ids = cch(st, p_st, kernel=np.hamming(100))
        cc, bin_ids = cch(st, p_st, kernel=np.hamming(10), window=[-max_lag, max_lag])
        # central_bin = len(cc) // 2
        # normalize by number of spikes
        # cc_vec[p] = np.max(cc[central_bin-10:central_bin+10]) #/ (len(st) + len(p_st))
        cc_vec[p] = np.max(cc) #/ (len(st) + len(p_st))

    return st_id, cc_vec

def bin_spiketimes(spike_times, fs=None, T=None, t_stop=None):
    '''

    Parameters
    ----------
    spike_times
    fs
    T

    Returns
    -------

    '''
    import elephant.conversion as conv
    import neo
    resampled_mat = []
    binned_spikes = []
    spiketrains = []

    if isinstance(spike_times[0], neo.SpikeTrain):
        unit = spike_times[0].units
        spike_times = [st.times.magnitude for st in spike_times]*unit
    for st in spike_times:
        if t_stop:
            t_st = t_stop.rescale(pq.ms)
        else:
            t_st = st[-1].rescale(pq.ms)
        st_pq = [s.rescale(pq.ms).magnitude for s in st]*pq.ms
        spiketrains.append(neo.SpikeTrain(st_pq, t_st))
    if not fs and not T:
        print('Provide either sampling frequency fs or time period T')
    elif fs:
        if not isinstance(fs, Quantity):
            raise ValueError("fs must be of type pq.Quantity")
        binsize = 1./fs
        binsize.rescale('ms')
        resampled_mat = []
        spikes = conv.BinnedSpikeTrain(spiketrains, binsize=binsize)
        for sts in spiketrains:
            spikes = conv.BinnedSpikeTrain(sts, binsize=binsize)
            resampled_mat.append(np.squeeze(spikes.to_array()))
            binned_spikes.append(spikes)
    elif T:
        binsize = T
        if not isinstance(T, Quantity):
            raise ValueError("T must be of type pq.Quantity")
        binsize.rescale('ms')
        resampled_mat = []
        for sts in spiketrains:
            spikes = conv.BinnedSpikeTrain(sts, binsize=binsize)
            resampled_mat.append(np.squeeze(spikes.to_array()))
            binned_spikes.append(spikes)


    return np.array(resampled_mat), binned_spikes



def evaluate_spiketrains(gtst, sst, t_jitt = 1, parallel=True, nprocesses=None,
                         pairs=[]):
    '''

    Parameters
    ----------
    gtst
    sst
    t_jitt
    overlapping
    parallel
    nprocesses

    Returns
    -------

    '''
    import neo
    import multiprocessing
    from elephant.spike_train_correlation import cch, corrcoef
    from scipy.optimize import linear_sum_assignment
    import time

    if nprocesses is None:
        num_cores = len(gtst)
    else:
        num_cores = nprocesses

    t_stop = gtst[0].t_stop

    sst_clip = sst
    gtst_clip = gtst

    if len(pairs) == 0:
        print('Computing correlations between spiketrains')

        or_mat, original_st = bin_spiketimes(gtst_clip, T=1*pq.ms, t_stop=t_stop)
        pr_mat, predicted_st = bin_spiketimes(sst_clip, T=1*pq.ms, t_stop=t_stop)
        cc_matr = np.zeros((or_mat.shape[0], pr_mat.shape[0]))

        if parallel:
            pool = multiprocessing.Pool(nprocesses)
            results = [pool.apply_async(cc_max_spiketrains, (st, st_id, predicted_st,))
                       for st_id, st in enumerate(original_st)]

            idxs = []
            cc_vecs = []
            for result in results:
                idxs.append(result.get()[0])
                cc_vecs.append(result.get()[1])

            for (id, cc_vec) in zip(idxs, cc_vecs):
                cc_matr[id] = [c / (len(gtst_clip[id]) + len(sst_clip[i])) for i, c in enumerate(cc_vec)]
            pool.close()
        else:
            max_lag = 20*pq.ms
            for o, o_st in enumerate(original_st):
                for p, p_st in enumerate(predicted_st):
                    # cc, bin_ids = cch(o_st, p_st, kernel=np.hamming(100))
                    cc, bin_ids = cch(o_st, p_st, kernel=np.hamming(10), window=[-max_lag, max_lag])
                    # normalize by number of spikes
                    cc_matr[o, p] = np.max(cc) / (len(gtst_clip[o]) + len(sst_clip[p])) # (abs(len(gtst[o]) - len(sst[p])) + 1)
        cc_matr /= np.max(cc_matr)

        print('Pairing spike trains')
        t_hung_st = time.time()
        cc2 = cc_matr ** 2
        col_ind, row_ind = linear_sum_assignment(-cc2)
        put_pairs = -1 * np.ones((len(gtst), 2)).astype('int')

        for i in range(len(gtst)):
            # if i in row_ind:
            idx = np.where(i == col_ind)
            if len(idx[0]) != 0:
                if cc2[col_ind[idx], row_ind[idx]] > 0.1:
                    put_pairs[i] = [int(col_ind[idx]), int(row_ind[idx])]
        t_hung_end = time.time()
    else:
        put_pairs = pairs

    [gt.annotate(paired=False) for gt in gtst_clip]
    [st.annotate(paired=False) for st in sst_clip]
    for pp in put_pairs:
        if pp[0] != -1:
            gtst_clip[pp[0]].annotate(paired=True)
        if pp[1] != -1:
            sst_clip[pp[1]].annotate(paired=True)

    # Evaluate
    for i, gt in enumerate(gtst_clip):
        lab_gt = np.array(['UNPAIRED'] * len(gt))
        gt.annotate(labels=lab_gt)
    for i, st in enumerate(sst_clip):
        lab_st = np.array(['UNPAIRED'] * len(st))
        st.annotate(labels=lab_st)

    print('Finding TP')
    for gt_i, gt in enumerate(gtst_clip):
        if put_pairs[gt_i, 0] != -1:
            lab_gt = gt.annotations['labels']
            st_sel = sst_clip[put_pairs[gt_i, 1]]
            lab_st = sst_clip[put_pairs[gt_i, 1]].annotations['labels']
            # from gtst: TP, TPO, TPSO, FN, FNO, FNSO
            for sp_i, t_sp in enumerate(gt):
                id_sp = np.where((st_sel > t_sp - t_jitt) & (st_sel < t_sp + t_jitt))[0]
                if len(id_sp) == 1:
                    if 'overlap' in gt.annotations.keys():
                        if gt.annotations['overlap'][sp_i] == 'NO':
                            lab_gt[sp_i] = 'TP'
                            lab_st[id_sp] = 'TP'
                        elif gt.annotations['overlap'][sp_i] == 'O':
                            lab_gt[sp_i] = 'TPO'
                            lab_st[id_sp] = 'TPO'
                        elif gt.annotations['overlap'][sp_i] == 'SO':
                            lab_gt[sp_i] = 'TPSO'
                            lab_st[id_sp] = 'TPSO'
                    else:
                        lab_gt[sp_i] = 'TP'
                        lab_st[id_sp] = 'TP'
            sst_clip[put_pairs[gt_i, 1]].annotate(labels=lab_st)
        else:
            lab_gt = np.array(['FN'] * len(gt))
        gt.annotate(labels=lab_gt)


    # find CL-CLO-CLSO
    print('Finding CL')
    for gt_i, gt in enumerate(gtst_clip):
        lab_gt = gt.annotations['labels']
        for l_gt, lab in enumerate(lab_gt):
            if lab == 'UNPAIRED':
                for st_i, st in enumerate(sst_clip):
                    if st.annotations['paired']:
                        t_up = gt[l_gt]
                        id_sp = np.where((st > t_up - t_jitt) & (st < t_up + t_jitt))[0]
                        lab_st = st.annotations['labels']
                        if len(id_sp) == 1 and lab_st[id_sp] == 'UNPAIRED':
                            if 'overlap' in gt.annotations.keys():
                                if gt.annotations['overlap'][l_gt] == 'NO':
                                    lab_gt[l_gt] = 'CL_' + str(gt_i) + '_' + str(st_i)
                                    if lab_st[id_sp] == 'UNPAIRED':
                                        lab_st[id_sp] = 'CL_NP'
                                elif gt.annotations['overlap'][l_gt] == 'O':
                                    lab_gt[l_gt] = 'CLO_' + str(gt_i) + '_' + str(st_i)
                                    if lab_st[id_sp] == 'UNPAIRED':
                                        lab_st[id_sp] = 'CLO_NP'
                                elif gt.annotations['overlap'][l_gt] == 'SO':
                                    lab_gt[l_gt] = 'CLSO_' + str(gt_i) + '_' + str(st_i)
                                    if lab_st[id_sp] == 'UNPAIRED':
                                        lab_st[id_sp] = 'CLSO_NP'
                            else:
                                lab_gt[l_gt] = 'CL_' + str(gt_i) + '_' + str(st_i)
                                # print( 'here'
                                if lab_st[id_sp] == 'UNPAIRED':
                                    lab_st[id_sp] = 'CL_NP'
                        st.annotate(labels=lab_st)
        gt.annotate(labels=lab_gt)

    print( 'Finding FP and FN')
    for gt_i, gt in enumerate(gtst_clip):
        lab_gt = gt.annotations['labels']
        for l_gt, lab in enumerate(lab_gt):
            if lab == 'UNPAIRED':
                if 'overlap' in gt.annotations.keys():
                    if gt.annotations['overlap'][l_gt] == 'NO':
                        lab_gt[l_gt] = 'FN'
                    elif gt.annotations['overlap'][l_gt] == 'O':
                        lab_gt[l_gt] = 'FNO'
                    elif gt.annotations['overlap'][l_gt] == 'SO':
                        lab_gt[l_gt] = 'FNSO'
                else:
                    lab_gt[l_gt] = 'FN'
        gt.annotate(labels=lab_gt)

    for st_i, st in enumerate(sst_clip):
        lab_st = st.annotations['labels']
        for st_i, lab in enumerate(lab_st):
            if lab == 'UNPAIRED':
                    lab_st[st_i] = 'FP'
        st.annotate(labels=lab_st)

    TOT_GT = sum([len(gt) for gt in gtst_clip])
    TOT_ST = sum([len(st) for st in sst_clip])
    total_spikes = TOT_GT + TOT_ST

    TP = sum([len(np.where('TP' == gt.annotations['labels'])[0]) for gt in gtst_clip])
    TPO = sum([len(np.where('TPO' == gt.annotations['labels'])[0]) for gt in gtst_clip])
    TPSO = sum([len(np.where('TPSO' == gt.annotations['labels'])[0]) for gt in gtst_clip])

    print( 'TP :', TP, TPO, TPSO, TP+TPO+TPSO)

    CL = sum([len([i for i, v in enumerate(gt.annotations['labels']) if 'CL' in v]) for gt in gtst_clip])
         # + sum([len(np.where('CL' == st.annotations['labels'])[0]) for st in sst])
    CLO = sum([len([i for i, v in enumerate(gt.annotations['labels']) if 'CLO' in v]) for gt in gtst_clip])
          # + sum([len(np.where('CLO' == st.annotations['labels'])[0]) for st in sst])
    CLSO = sum([len([i for i, v in enumerate(gt.annotations['labels']) if 'CLSO' in v]) for gt in gtst_clip])
           # + sum([len(np.where('CLSO' == st.annotations['labels'])[0]) for st in sst_clip])

    print( 'CL :', CL, CLO, CLSO, CL+CLO+CLSO)

    FN = sum([len(np.where('FN' == gt.annotations['labels'])[0]) for gt in gtst_clip])
    FNO = sum([len(np.where('FNO' == gt.annotations['labels'])[0]) for gt in gtst_clip])
    FNSO = sum([len(np.where('FNSO' == gt.annotations['labels'])[0]) for gt in gtst_clip])

    print( 'FN :', FN, FNO, FNSO, FN+FNO+FNSO)


    FP = sum([len(np.where('FP' == st.annotations['labels'])[0]) for st in sst_clip])

    print( 'FP :', FP)

    print( 'TOTAL: ', TOT_GT, TOT_ST, TP+TPO+TPSO+CL+CLO+CLSO+FN+FNO+FNSO+FP)

    counts = {'TP': TP, 'TPO': TPO, 'TPSO': TPSO,
              'CL': CL, 'CLO': CLO, 'CLSO': CLSO,
              'FN': FN, 'FNO': FNO, 'FNSO': FNSO,
              'FP': FP, 'TOT': total_spikes, 'TOT_GT': TOT_GT, 'TOT_ST': TOT_ST}


    return counts, put_pairs #, cc_matr

def confusion_matrix(gtst, sst, pairs, plot_fig=True, xlabel=None, ylabel=None):
    '''

    Parameters
    ----------
    gtst
    sst
    pairs 1D array with paired sst to gtst

    Returns
    -------

    '''
    conf_matrix = np.zeros((len(gtst)+1, len(sst)+1), dtype=int)
    idxs_pairs_clean = np.where(pairs != -1)
    idxs_pairs_dirty = np.where(pairs == -1)
    pairs_clean = pairs[idxs_pairs_clean]
    gtst_clean = np.array(gtst)[idxs_pairs_clean]
    gtst_extra = np.array(gtst)[idxs_pairs_dirty]

    gtst_idxs = np.append(idxs_pairs_clean, idxs_pairs_dirty)
    sst_idxs = pairs_clean
    sst_extra = []

    for gt_i, gt in enumerate(gtst_clean):
        if gt.annotations['paired']:
            tp = len(np.where('TP' == gt.annotations['labels'])[0])
            conf_matrix[gt_i, gt_i] =  int(tp)
            for st_i, st in enumerate(sst):
                cl_str = str(gt_i) + '_' + str(st_i)
                cl = len([i for i, v in enumerate(gt.annotations['labels']) if 'CL' in v and cl_str in v])
                if cl != 0:
                    st_p = np.where(st_i == pairs_clean)
                    conf_matrix[gt_i, st_p] = int(cl)
        fn = len(np.where('FN' == gt.annotations['labels'])[0])
        conf_matrix[gt_i, -1] = int(fn)
    for gt_i, gt in enumerate(gtst_extra):
        fn = len(np.where('FN' == gt.annotations['labels'])[0])
        conf_matrix[gt_i+len(gtst_clean), -1] = int(fn)
    for st_i, st in enumerate(sst):
        fp = len(np.where('FP' == st.annotations['labels'])[0])
        st_p = np.where(st_i == pairs_clean)[0]
        if len(st_p) != 0:
            conf_matrix[-1, st_p] = fp
        else:
            sst_extra.append(int(st_i))
            conf_matrix[-1, len(pairs_clean) + len(sst_extra) - 1] = fp

    if plot_fig:
        fig, ax = plt.subplots()
        # Using matshow here just because it sets the ticks up nicely. imshow is faster.
        ax.matshow(conf_matrix, cmap='Greens')

        for (i, j), z in np.ndenumerate(conf_matrix):
            if z != 0:
                if z > np.max(conf_matrix)/2.:
                    ax.text(j, i, '{:d}'.format(z), ha='center', va='center', color='white')
                else:
                    ax.text(j, i, '{:d}'.format(z), ha='center', va='center', color='black')
                    # ,   bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

        ax.axhline(int(len(gtst)-1)+0.5, color='black')
        ax.axvline(int(len(sst)-1)+0.5, color='black')

        # Major ticks
        ax.set_xticks(np.arange(0, len(sst) + 1))
        ax.set_yticks(np.arange(0, len(gtst) + 1))
        ax.xaxis.tick_bottom()
        # Labels for major ticks
        ax.set_xticklabels(np.append(np.append(sst_idxs, sst_extra).astype(int), 'FN'), fontsize=12)
        ax.set_yticklabels(np.append(gtst_idxs, 'FP'), fontsize=12)

        if xlabel==None:
            ax.set_xlabel('Sorted spike trains', fontsize=15)
        else:
            ax.set_xlabel(xlabel, fontsize=20)
        if ylabel==None:
            ax.set_ylabel('Ground truth spike trains', fontsize=15)
        else:
            ax.set_ylabel(ylabel, fontsize=20)

    return conf_matrix, ax



def compute_performance(counts):

    tp_rate = float(counts['TP']) / counts['TOT_GT'] * 100
    tpo_rate = float(counts['TPO']) / counts['TOT_GT'] * 100
    tpso_rate = float(counts['TPSO']) / counts['TOT_GT'] * 100
    tot_tp_rate = float(counts['TP'] + counts['TPO'] + counts['TPSO']) / counts['TOT_GT'] * 100

    cl_rate = float(counts['CL']) / counts['TOT_GT'] * 100
    clo_rate = float(counts['CLO']) / counts['TOT_GT'] * 100
    clso_rate = float(counts['CLSO']) / counts['TOT_GT'] * 100
    tot_cl_rate = float(counts['CL'] + counts['CLO'] + counts['CLSO']) / counts['TOT_GT'] * 100

    fn_rate = float(counts['FN']) / counts['TOT_GT'] * 100
    fno_rate = float(counts['FNO']) / counts['TOT_GT'] * 100
    fnso_rate = float(counts['FNSO']) / counts['TOT_GT'] * 100
    tot_fn_rate = float(counts['FN'] + counts['FNO'] + counts['FNSO']) / counts['TOT_GT'] * 100

    fp_gt = float(counts['FP']) / counts['TOT_GT'] * 100
    fp_st = float(counts['FP']) / counts['TOT_ST'] * 100

    accuracy = tot_tp_rate / (tot_tp_rate + tot_fn_rate + fp_gt) * 100
    sensitivity = tot_tp_rate / (tot_tp_rate + tot_fn_rate) * 100
    miss_rate = tot_fn_rate / (tot_tp_rate + tot_fn_rate) * 100
    precision = tot_tp_rate / (tot_tp_rate + fp_gt) * 100
    false_discovery_rate = fp_gt / (tot_tp_rate + fp_gt) * 100

    # print 'PERFORMANCE: \n'
    # print '\nTP: ', tp_rate, ' %'
    # print 'TPO: ', tpo_rate, ' %'
    # print 'TPSO: ', tpso_rate, ' %'
    # print 'TOT TP: ', tot_tp_rate, ' %'
    #
    # print '\nCL: ', cl_rate, ' %'
    # print 'CLO: ', clo_rate, ' %'
    # print 'CLSO: ', clso_rate, ' %'
    # print 'TOT CL: ', tot_cl_rate, ' %'
    #
    # print '\nFN: ', fn_rate, ' %'
    # print 'FNO: ', fno_rate, ' %'
    # print 'FNSO: ', fnso_rate, ' %'
    # print 'TOT FN: ', tot_fn_rate, ' %'
    #
    # print '\nFP (%GT): ', fp_gt, ' %'
    # print '\nFP (%ST): ', fp_st, ' %'
    #
    # print '\nACCURACY: ', accuracy, ' %'
    # print 'SENSITIVITY: ', sensitivity, ' %'
    # print 'MISS RATE: ', miss_rate, ' %'
    # print 'PRECISION: ', precision, ' %'
    # print 'FALSE DISCOVERY RATE: ', false_discovery_rate, ' %'

    performance = {'tot_tp': tot_tp_rate, 'tot_cl': tot_cl_rate, 'tot_fn': tot_fn_rate, 'tot_fp': fp_gt,
                   'accuracy': accuracy, 'sensitivity': sensitivity, 'precision': precision, 'miss_rate': miss_rate,
                   'false_disc_rate': false_discovery_rate}

    return performance



sorting_sc = si.SpykingCircusSortingExtractor('/home/alessiob/Documents/Codes/spike_sorting/spiketoolkit/examples/spyking_circus/recording')
sorting_klusta = si.KlustaSortingExtractor('/home/alessiob/Documents/Codes/spike_sorting/spiketoolkit/examples/klusta/recording.kwik')
mearecording = si.MEArecRecordingExtractor('/home/alessiob/Documents/Codes/MEArec/data/recordings/recordings_20cells_Neuronexus-32_30.0_10.0uV_12-10-2018:15:18')
fs = mearecording.getSamplingFrequency() * pq.Hz
duration = mearecording.getNumFrames() / mearecording.getSamplingFrequency() * pq.s

# make neo objects
sc_sst = [neo.core.SpikeTrain(times=sorting_sc.getUnitSpikeTrain(unit)/fs,
                              t_start=0 * pq.s,
                              t_stop=duration) for unit in sorting_sc.getUnitIds()]

sc_kl = [neo.core.SpikeTrain(times=sorting_klusta.getUnitSpikeTrain(unit) / fs,
                              t_start=0 * pq.s,
                              t_stop=duration) for unit in
         sorting_klusta.getUnitIds()]

print('Matching spike trains')
counts, pairs = evaluate_spiketrains(sc_sst, sc_kl, t_jitt = 3*pq.ms)

print('PAIRS: ')
print(pairs)

print( 'Computing performance')
performance = compute_performance(counts)

print( 'Calculating confusion matrix')
conf = confusion_matrix(sc_sst, sc_kl, pairs[:,1])

    # print( 'Matching only identified spike trains')
    # pairs_gt_red = pairs[:, 0][np.where(pairs[:, 0]!=-1)]
    # gtst_id = np.array(gtst_red)[pairs_gt_red]
    # counts_id, pairs_id = evaluate_spiketrains(gtst_id, sst, t_jitt = 3*pq.ms)
    #
    # print( 'Computing performance on only identified spike trains')
    # performance_id = compute_performance(counts_id)
    #
    # print( 'Calculating confusion matrix')
    # conf_id = confusion_matrix(gtst_id, sst, pairs_id[:, 1])