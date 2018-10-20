import numpy as np
import spikeinterface as si
from scipy.optimize import linear_sum_assignment

class SortingComparison():
    def __init__(self, sorting1, sorting2, delta_tp=10, minimum_accuracy=0.5):
        self._sorting1 = sorting1
        self._sorting2 = sorting2
        self._delta_tp = delta_tp
        self._min_accuracy = minimum_accuracy
        self._do_matching()
    
    def getSorting1(self):
        return self._sorting1
    
    def getSorting2(self):
        return self._sorting2
    
    def getMappedSorting1(self):
        return MappedSortingExtractor(self._sorting2, self._unit_map12)
        
    def getMappedSorting2(self):
        return MappedSortingExtractor(self._sorting1, self._unit_map21)
    
    def getMatchingEventCount(self,  unit1,  unit2):
        if (unit1 is not None) and (unit2 is not None):
            if unit1 != -1:
                a = self._matching_event_counts_12[unit1]
                if unit2 in a:
                    return a[unit2]
                else:
                    return 0
            else:
                return 0
        else:
            raise Exception('getMatchingEventCount: unit1 and unit2 must not be None.')
            
    def _compute_agreement_score(self, num_matches,  num1,  num2):
        denom = num1 + num2 - num_matches
        if denom == 0:
            return 0
        return num_matches / denom

    def _compute_safe_frac(self, numer, denom):
        if denom == 0:
            return 0
        return float(numer) / denom
        
    def getBestUnitMatch1(self, unit1):
        if unit1 in self._best_match_units_12:
            return self._best_match_units_12[unit1]
        else:
            return None
        
    def getBestUnitMatch2(self, unit2):
        if unit2 in self._best_match_units_21:
            return self._best_match_units_21[unit2]
        else:
            return None
        
    def getMatchingUnitList1(self, unit1):
        a = self._matching_event_counts_12[unit1]
        return list(a.keys())
        
    def getMatchingUnitList2(self, unit2):
        a = self._matching_event_counts_21[unit2]
        return list(a.keys())
        
    def getAgreementFraction(self, unit1=None, unit2=None):
        if (unit1 is not None) and (unit2 is None):
            if unit1 != -1:
                unit2 = self.getBestUnitMatch1(unit1)
                if unit2 is None or unit2 == -1:
                    return 0
            else:
                return 0
        if (unit1 is None) and (unit2 is not None):
            if unit1 != -1 and unit2 != -1:
                unit1 = self.getBestUnitMatch2(unit2)
                if unit1 is None or unit1 == -1:
                    return 0
            else:
                return 0
        if (unit1 is None) and (unit2 is None):
            raise Exception('getAgreementFraction: at least one of unit1 and unit2 must not be None.')

        if unit1 != -1 and unit2 != -1:
            a = self._matching_event_counts_12[unit1]
            if unit2 not in a:
                return 0
        else:
            return 0
        return self._compute_agreement_score(a[unit2], self._event_counts_1[unit1], self._event_counts_2[unit2])
        
    def getFalsePositiveFraction(self, unit1, unit2=None):
        if unit1 is None:
            raise Exception('getFalsePositiveFraction: unit1 must not be None')
        if unit2 is None:
            unit2 = self.getBestUnitMatch1(unit1)
            if unit2 is None or unit2 == -1:
                return 0
            
        if unit1 != -1 and unit2 != -1:
            a = self._matching_event_counts_12[unit1]
            if unit2 not in a:
                return 0
        else:
            return 0
        return 1 - self._compute_safe_frac(a[unit2], self._event_counts_1[unit1])
    
    def getFalseNegativeFraction(self, unit1, unit2=None):
        if unit1 is None:
            raise Exception('getFalsePositiveFraction: unit1 must not be None')
        if unit2 is None:
            unit2 = self.getBestUnitMatch1(unit1)
            if unit2 is None:
                return 0
            
        if unit1 != -1 and unit2 != -1:
            a = self._matching_event_counts_12[unit1]
            if unit2 not in a:
                return 0
        else:
            return 0
        return 1 - self._compute_safe_frac(a[unit2], self._event_counts_2[unit2])
    
    def _do_matching(self):
        self._event_counts_1 = dict()
        self._event_counts_2 = dict()
        self._matching_event_counts_12 = dict()
        self._best_match_units_12 = dict()
        self._matching_event_counts_21 = dict()
        self._best_match_units_21 = dict()
        self._unit_map12 = dict()
        self._unit_map21 = dict()

        sorting1 = self._sorting1
        sorting2 = self._sorting2
        unit1_ids = sorting1.getUnitIds()
        unit2_ids = sorting2.getUnitIds()
        N1 = len(unit1_ids)
        N2 = len(unit2_ids)
        
        # Compute events counts
        event_counts1 = np.zeros((N1)).astype(np.int64)
        for i1, u1 in enumerate(unit1_ids):
            times1 = sorting1.getUnitSpikeTrain(u1)
            event_counts1[i1] = len(times1)
            self._event_counts_1[u1] = len(times1)
        event_counts2 = np.zeros((N2)).astype(np.int64)
        for i2, u2 in enumerate(unit2_ids):
            times2 = sorting2.getUnitSpikeTrain(u2)
            event_counts2[i2] = len(times2)
            self._event_counts_2[u2] = len(times2)
            
        # Compute matching events
        matching_event_counts = np.zeros((N1, N2)).astype(np.int64)
        scores = np.zeros((N1, N2))
        for i1, u1 in enumerate(unit1_ids):
            times1 = sorting1.getUnitSpikeTrain(u1)
            for i2, u2 in enumerate(unit2_ids):
                times2 = sorting2.getUnitSpikeTrain(u2)
                num_matches = count_matching_events(times1,  times2,  delta=self._delta_tp)
                matching_event_counts[i1, i2] = num_matches
                scores[i1, i2] = self._compute_agreement_score(num_matches, event_counts1[i1], event_counts2[i2])
        
        # Find best matches for spiketrains 1
        for i1, u1 in enumerate(unit1_ids):
            scores0 = scores[i1, :]
            self._matching_event_counts_12[u1] = dict()
            if np.max(scores0) > 0:
                inds0 = np.where(scores0 > 0)[0]
                for i2 in inds0:
                    self._matching_event_counts_12[u1][unit2_ids[i2]] = matching_event_counts[i1, i2]
                i2_best = np.argmax(scores0)
                self._best_match_units_12[u1] = unit2_ids[i2_best]
            else:
                self._best_match_units_12[u1] = -1

        # Find best matches for spiketrains 2
        for i2, u2 in enumerate(unit2_ids):
            scores0 = scores[:, i2]
            self._matching_event_counts_21[u2] = dict()
            if np.max(scores0) > 0:
                inds0 = np.where(scores0 > 0)[0]
                for i1 in inds0:
                    self._matching_event_counts_21[u2][unit1_ids[i1]] = matching_event_counts[i1, i2]
                i1_best = np.argmax(scores0)
                self._best_match_units_21[u2] = unit1_ids[i1_best]
            else:
                self._best_match_units_21[u2] = -1
        
        # Assign best matches
        [inds1, inds2] = linear_sum_assignment(-scores)
        inds1 = list(inds1)
        inds2 = list(inds2)
        k2 = np.max(unit2_ids) + 1
        for i1,  u1 in enumerate(unit1_ids):
            if i1 in inds1:
                aa = inds1.index(i1)
                i2 = inds2[aa]
                u2 = unit2_ids[i2]
                if self.getAgreementFraction(u1, u2) > self._min_accuracy:
                    self._unit_map12[u1] = u2
                else:
                    self._unit_map12[u1] = -1
            else:
                # self._unit_map12[u1] = k2
                # k2 = k2+1
                self._unit_map12[u1] = -1
        k1 = np.max(unit1_ids)+1
        for i2,  u2 in enumerate(unit2_ids):
            if i2 in inds2:
                aa = inds2.index(i2)
                i1 = inds1[aa]
                u1 = unit1_ids[i1]
                if self.getAgreementFraction(u1, u2) > self._min_accuracy:
                    self._unit_map21[u2] = u1
                else:
                    self._unit_map21[u2] = -1
            else:
                # self._unit_map21[u2] = k1
                # k1 = k1+1
                self._unit_map21[u2] = -1
                
    def _do_counting(self):
        sorting1 = self._sorting1
        sorting2 = self._sorting2
        unit1_ids = sorting1.getUnitIds()
        unit2_ids = sorting2.getUnitIds()
        labels_st1 = []
        labels_st2 = []
        N1 = len(unit1_ids)
        N2 = len(unit2_ids)
        # Evaluate
        for u1 in unit1_ids:
            st1 = sorting1.getUnitSpikeTrain(u1)
            lab_st1 = np.array(['UNPAIRED'] * len(st1))
            labels_st1.append(lab_st1)
        for u2 in unit2_ids:
            st2 = sorting1.getUnitSpikeTrain(u2)
            lab_st2 = np.array(['UNPAIRED'] * len(st2))
            labels_st2.append(lab_st2)

        print('Finding TP')
        for gt_i, gt in enumerate(unit1_ids):
            if put_pairs[gt_i, 0] != -1:
                lab_gt = gt.annotations['labels']
                st_sel = unit2_ids[put_pairs[gt_i, 1]]
                lab_st = unit2_ids[put_pairs[gt_i, 1]].annotations['labels']
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
                unit2_ids[put_pairs[gt_i, 1]].annotate(labels=lab_st)
            else:
                lab_gt = np.array(['FN'] * len(gt))
            gt.annotate(labels=lab_gt)

        # find CL-CLO-CLSO
        print('Finding CL')
        for gt_i, gt in enumerate(unit1_ids):
            lab_gt = gt.annotations['labels']
            for l_gt, lab in enumerate(lab_gt):
                if lab == 'UNPAIRED':
                    for st_i, st in enumerate(unit2_ids):
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

        print('Finding FP and FN')
        for gt_i, gt in enumerate(unit1_ids):
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

        for st_i, st in enumerate(unit2_ids):
            lab_st = st.annotations['labels']
            for st_i, lab in enumerate(lab_st):
                if lab == 'UNPAIRED':
                    lab_st[st_i] = 'FP'
            st.annotate(labels=lab_st)

        TOT_GT = sum([len(gt) for gt in unit1_ids])
        TOT_ST = sum([len(st) for st in unit2_ids])
        total_spikes = TOT_GT + TOT_ST

        TP = sum([len(np.where('TP' == gt.annotations['labels'])[0]) for gt in unit1_ids])
        TPO = sum([len(np.where('TPO' == gt.annotations['labels'])[0]) for gt in unit1_ids])
        TPSO = sum([len(np.where('TPSO' == gt.annotations['labels'])[0]) for gt in unit1_ids])

        print('TP :', TP, TPO, TPSO, TP + TPO + TPSO)

        CL = sum([len([i for i, v in enumerate(gt.annotations['labels']) if 'CL' in v]) for gt in unit1_ids])
        # + sum([len(np.where('CL' == st.annotations['labels'])[0]) for st in sst])
        CLO = sum([len([i for i, v in enumerate(gt.annotations['labels']) if 'CLO' in v]) for gt in unit1_ids])
        # + sum([len(np.where('CLO' == st.annotations['labels'])[0]) for st in sst])
        CLSO = sum([len([i for i, v in enumerate(gt.annotations['labels']) if 'CLSO' in v]) for gt in unit1_ids])
        # + sum([len(np.where('CLSO' == st.annotations['labels'])[0]) for st in unit2_ids])

        print('CL :', CL, CLO, CLSO, CL + CLO + CLSO)

        FN = sum([len(np.where('FN' == gt.annotations['labels'])[0]) for gt in unit1_ids])
        FNO = sum([len(np.where('FNO' == gt.annotations['labels'])[0]) for gt in unit1_ids])
        FNSO = sum([len(np.where('FNSO' == gt.annotations['labels'])[0]) for gt in unit1_ids])

        print('FN :', FN, FNO, FNSO, FN + FNO + FNSO)

        FP = sum([len(np.where('FP' == st.annotations['labels'])[0]) for st in unit2_ids])

        print('FP :', FP)

        print('TOTAL: ', TOT_GT, TOT_ST, TP + TPO + TPSO + CL + CLO + CLSO + FN + FNO + FNSO + FP)

        counts = {'TP': TP, 'TPO': TPO, 'TPSO': TPSO,
                  'CL': CL, 'CLO': CLO, 'CLSO': CLSO,
                  'FN': FN, 'FNO': FNO, 'FNSO': FNSO,
                  'FP': FP, 'TOT': total_spikes, 'TOT_GT': TOT_GT, 'TOT_ST': TOT_ST}

        return counts,

class MappedSortingExtractor(si.SortingExtractor):
    def __init__(self, sorting, unit_map):
        si.SortingExtractor.__init__(self)
        self._sorting = sorting
        self._unit_map = unit_map
        self._unit_ids = list(self._unit_map.keys())
        
    def getUnitIds(self, unit_ids=None):
        if unit_ids is None:
            return self._unit_ids
        else:
            return self._unit_ids[unit_ids]

    def getMappedUnitIds(self, unit_ids=None):
        if unit_ids is None:
            return list(self._unit_map.values())
        elif isinstance(unit_ids, (int, np.integer)):
            return self._unit_map[unit_ids]
        else:
            return list([self._unit_map[u] for u in self._unit_ids if u in unit_ids])
    
    def getUnitSpikeTrain(self, unit_id, start_frame=None, end_frame=None):
        unit2 = self._unit_map[unit_id]
        if unit2 != -1:
            return self._sorting.getUnitSpikeTrain(unit2, start_frame=start_frame, end_frame=end_frame)
        else:
            print(unit_id, " is not matched!")
            return None

    
def count_matching_events(times1, times2, delta=10):
    times_concat = np.concatenate((times1, times2))
    membership = np.concatenate((np.ones(times1.shape)*1, np.ones(times2.shape)*2))
    indices = times_concat.argsort()
    times_concat_sorted = times_concat[indices]
    membership_sorted = membership[indices]
    diffs = times_concat_sorted[1:] - times_concat_sorted[:-1]
    inds = np.where((diffs <= delta) & (membership_sorted[0:-1] != membership_sorted[1:]))[0]
    if (len(inds) == 0):
        return 0
    inds2 = np.where(inds[:-1] + 1 != inds[1:])[0]
    return len(inds2) + 1


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