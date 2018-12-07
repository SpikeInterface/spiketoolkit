import numpy as np
import spikeextractors as se
from scipy.optimize import linear_sum_assignment


class SortingComparison():
    def __init__(self, sorting1, sorting2, sorting1_name=None, sorting2_name=None, delta_tp=10, minimum_accuracy=0.5,
                 count=False, verbose=False):
        self._sorting1 = sorting1
        self._sorting2 = sorting2
        self.sorting1_name = sorting1_name
        self.sorting2_name = sorting2_name
        self._delta_tp = delta_tp
        self._min_accuracy = minimum_accuracy
        if verbose:
            print("Matching...")
        self._do_matching()

        self._counts = None
        if count:
            if verbose:
                print("Counting...")
            self._do_counting(verbose=False)

    def getSorting1(self):
        return self._sorting1

    def getSorting2(self):
        return self._sorting2

    def getLabels1(self, unit_id):
        if unit_id in self._sorting1.getUnitIds():
            return self._labels_st1[unit_id]
        else:
            raise Exception("Unit_id is not a valid unit")

    def getLabels2(self, unit_id):
        if unit_id in self._sorting1.getUnitIds():
            return self._labels_st1[unit_id]
        else:
            raise Exception("Unit_id is not a valid unit")

    def getMappedSorting1(self):
        return MappedSortingExtractor(self._sorting2, self._unit_map12)

    def getMappedSorting2(self):
        return MappedSortingExtractor(self._sorting1, self._unit_map21)

    def getMatchingEventCount(self, unit1, unit2):
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

    def _compute_agreement_score(self, num_matches, num1, num2):
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

    def computeCounts(self):
        if self._counts is None:
            self._do_counting(verbose=False)

    def plotConfusionMatrix(self, xlabel=None, ylabel=None):
        import matplotlib.pylab as plt

        if self._counts is None:
            self._do_counting(verbose=False)

        sorting1 = self._sorting1
        sorting2 = self._sorting2
        unit1_ids = sorting1.getUnitIds()
        unit2_ids = sorting2.getUnitIds()
        N1 = len(unit1_ids)
        N2 = len(unit2_ids)
        st1_idxs, st2_idxs = self._do_confusion()
        fig, ax = plt.subplots()
        # Using matshow here just because it sets the ticks up nicely. imshow is faster.
        ax.matshow(self._confusion_matrix, cmap='Greens')

        for (i, j), z in np.ndenumerate(self._confusion_matrix):
            if z != 0:
                if z > np.max(self._confusion_matrix) / 2.:
                    ax.text(j, i, '{:d}'.format(z), ha='center', va='center', color='white')
                else:
                    ax.text(j, i, '{:d}'.format(z), ha='center', va='center', color='black')

        ax.axhline(int(N1 - 1) + 0.5, color='black')
        ax.axvline(int(N2 - 1) + 0.5, color='black')

        # Major ticks
        ax.set_xticks(np.arange(0, N2 + 1))
        ax.set_yticks(np.arange(0, N1 + 1))
        ax.xaxis.tick_bottom()
        # Labels for major ticks
        ax.set_xticklabels(np.append(st2_idxs, 'FN'), fontsize=12)
        ax.set_yticklabels(np.append(st1_idxs, 'FP'), fontsize=12)

        if xlabel == None:
            if self.sorting2_name is None:
                ax.set_xlabel('Sorting 2', fontsize=15)
            else:
                ax.set_xlabel(self.sorting2_name, fontsize=15)
        else:
            ax.set_xlabel(xlabel, fontsize=20)
        if ylabel == None:
            if self.sorting1_name is None:
                ax.set_ylabel('Sorting 1', fontsize=15)
            else:
                ax.set_ylabel(self.sorting1_name, fontsize=15)
        else:
            ax.set_ylabel(ylabel, fontsize=20)

        return ax

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
                num_matches = count_matching_events(times1, times2, delta=self._delta_tp)
                matching_event_counts[i1, i2] = num_matches
                scores[i1, i2] = self._compute_agreement_score(num_matches, event_counts1[i1], event_counts2[i2])

        # Find best matches for spiketrains 1
        for i1, u1 in enumerate(unit1_ids):
            scores0 = scores[i1, :]
            self._matching_event_counts_12[u1] = dict()
            if scores0:
                if np.max(scores0) > 0:
                    inds0 = np.where(scores0 > 0)[0]
                    for i2 in inds0:
                        self._matching_event_counts_12[u1][unit2_ids[i2]] = matching_event_counts[i1, i2]
                    i2_best = np.argmax(scores0)
                    self._best_match_units_12[u1] = unit2_ids[i2_best]
                else:
                    self._best_match_units_12[u1] = -1
            else:
                self._best_match_units_12[u1] = -1

        # Find best matches for spiketrains 2
        for i2, u2 in enumerate(unit2_ids):
            scores0 = scores[:, i2]
            self._matching_event_counts_21[u2] = dict()
            if scores0:
                if np.max(scores0) > 0:
                    inds0 = np.where(scores0 > 0)[0]
                    for i1 in inds0:
                        self._matching_event_counts_21[u2][unit1_ids[i1]] = matching_event_counts[i1, i2]
                    i1_best = np.argmax(scores0)
                    self._best_match_units_21[u2] = unit1_ids[i1_best]
                else:
                    self._best_match_units_21[u2] = -1
            else:
                self._best_match_units_21[u2] = -1

        # Assign best matches
        [inds1, inds2] = linear_sum_assignment(-scores)
        inds1 = list(inds1)
        inds2 = list(inds2)
        k2 = np.max(unit2_ids) + 1
        for i1, u1 in enumerate(unit1_ids):
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
        k1 = np.max(unit1_ids) + 1
        for i2, u2 in enumerate(unit2_ids):
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

    def _do_counting(self, verbose=False):
        sorting1 = self._sorting1
        sorting2 = self._sorting2
        unit1_ids = sorting1.getUnitIds()
        unit2_ids = sorting2.getUnitIds()
        self._labels_st1 = dict()
        self._labels_st2 = dict()
        N1 = len(unit1_ids)
        N2 = len(unit2_ids)
        # Evaluate
        for u1 in unit1_ids:
            st1 = sorting1.getUnitSpikeTrain(u1)
            lab_st1 = np.array(['UNPAIRED'] * len(st1))
            self._labels_st1[u1] = lab_st1
        for u2 in unit2_ids:
            st2 = sorting2.getUnitSpikeTrain(u2)
            lab_st2 = np.array(['UNPAIRED'] * len(st2))
            self._labels_st2[u2] = lab_st2

        if verbose:
            print('Finding TP')
        for u_i, u1 in enumerate(sorting1.getUnitIds()):
            if self.getMappedSorting1().getMappedUnitIds(u1) != -1:
                lab_st1 = self._labels_st1[u1]
                lab_st2 = self._labels_st2[self.getMappedSorting1().getMappedUnitIds(u1)]
                mapped_st = self.getMappedSorting1().getUnitSpikeTrain(u1)
                # from gtst: TP, TPO, TPSO, FN, FNO, FNSO
                for sp_i, n_sp in enumerate(sorting1.getUnitSpikeTrain(u1)):
                    id_sp = np.where((mapped_st > n_sp - self._delta_tp) & (mapped_st < n_sp + self._delta_tp))[0]
                    if len(id_sp) == 1:
                        lab_st1[sp_i] = 'TP'
                        lab_st2[id_sp] = 'TP'
            else:
                lab_st1 = np.array(['FN'] * len(sorting1.getUnitSpikeTrain(u1)))

        # find CL-CLO-CLSO
        if verbose:
            print('Finding CL')
        for u_i, u1 in enumerate(sorting1.getUnitIds()):
            lab_st1 = self._labels_st1[u1]
            st1 = sorting1.getUnitSpikeTrain(u1)
            for l_gt, lab in enumerate(lab_st1):
                if lab == 'UNPAIRED':
                    for u_j, u2 in enumerate(sorting2.getUnitIds()):
                        if u2 in self.getMappedSorting1().getMappedUnitIds() \
                                and self.getMappedSorting1().getMappedUnitIds(u1) != -1:
                            lab_st2 = self._labels_st2[u2]
                            st2 = sorting2.getUnitSpikeTrain(u2)

                            n_up = st1[l_gt]
                            id_sp = np.where((st2 > n_up - self._delta_tp) & (st2 < n_up + self._delta_tp))[0]
                            if len(id_sp) == 1 and lab_st2[id_sp] == 'UNPAIRED':
                                lab_st1[l_gt] = 'CL_' + str(u1) + '_' + str(u2)
                                lab_st2[id_sp] = 'CL_' + str(u2) + '_' + str(u1)
                                # if lab_st2[id_sp] == 'UNPAIRED':
                                #     lab_st2[id_sp] = 'CL_NP'

        if verbose:
            print('Finding FP and FN')
        for u1 in sorting1.getUnitIds():
            lab_st1 = self._labels_st1[u1]
            for l_gt, lab in enumerate(lab_st1):
                if lab == 'UNPAIRED':
                    lab_st1[l_gt] = 'FN'

        for u2 in sorting2.getUnitIds():
            lab_st2 = self._labels_st2[u2]
            for l_gt, lab in enumerate(lab_st2):
                if lab == 'UNPAIRED':
                    lab_st2[l_gt] = 'FP'

        TOT_ST1 = sum([len(sorting1.getUnitSpikeTrain(unit)) for unit in sorting1.getUnitIds()])
        TOT_ST2 = sum([len(sorting2.getUnitSpikeTrain(unit)) for unit in sorting2.getUnitIds()])
        total_spikes = TOT_ST1 + TOT_ST2
        TP = sum([len(np.where('TP' == self._labels_st1[unit])[0]) for unit in sorting1.getUnitIds()])
        CL = sum(
            [len([i for i, v in enumerate(self._labels_st1[unit]) if 'CL' in v]) for unit in sorting1.getUnitIds()])
        FN = sum([len(np.where('FN' == self._labels_st1[unit])[0]) for unit in sorting1.getUnitIds()])
        FP = sum([len(np.where('FP' == self._labels_st2[unit])[0]) for unit in sorting2.getUnitIds()])
        self.counts = {'TP': TP, 'CL': CL, 'FN': FN, 'FP': FP, 'TOT': total_spikes, 'TOT_ST1': TOT_ST1,
                       'TOT_ST2': TOT_ST2}

        if verbose:
            print('TP :', TP)
            print('CL :', CL)
            print('FN :', FN)
            print('FP :', FP)
            print('TOTAL: ', TOT_ST1, TOT_ST2, TP + CL + FN + FP)

    def _do_confusion(self):
        # def confusion_matrix(gtst, sst, pairs, plot_fig=True, xlabel=None, ylabel=None):
        '''

        Parameters
        ----------
        gtst
        sst
        pairs 1D array with paired sst to gtst

        Returns
        -------

        '''
        sorting1 = self._sorting1
        sorting2 = self._sorting2
        unit1_ids = sorting1.getUnitIds()
        unit2_ids = sorting2.getUnitIds()
        N1 = len(unit1_ids)
        N2 = len(unit2_ids)

        conf_matrix = np.zeros((N1 + 1, N2 + 1), dtype=int)
        idxs_matched = np.where(np.array(self.getMappedSorting1().getMappedUnitIds()) != -1)
        if len(idxs_matched) > 0:
            idxs_matched = idxs_matched[0]
        idxs_unmatched = np.where(np.array(self.getMappedSorting1().getMappedUnitIds()) == -1)
        if len(idxs_unmatched) > 0:
            idxs_unmatched = idxs_unmatched[0]
        unit_map_matched = np.array(self.getMappedSorting1().getMappedUnitIds())[idxs_matched]

        st1_idxs = np.append(np.array(sorting1.getUnitIds())[idxs_matched],
                             np.array(sorting1.getUnitIds())[idxs_unmatched])
        st2_matched = unit_map_matched
        st2_unmatched = []

        for u_i, u1 in enumerate(np.array(sorting1.getUnitIds())[idxs_matched]):
            lab_st1 = self._labels_st1[u1]
            tp = len(np.where('TP' == lab_st1)[0])
            conf_matrix[u_i, u_i] = int(tp)
            for u_j, u2 in enumerate(sorting2.getUnitIds()):
                lab_st2 = self._labels_st2[u2]
                cl_str = str(u1) + '_' + str(u2)
                cl = len([i for i, v in enumerate(lab_st1) if 'CL' in v and cl_str in v])
                if cl != 0:
                    st_p = np.where(u2 == unit_map_matched)
                    conf_matrix[u_i, st_p] = int(cl)
            fn = len(np.where('FN' == lab_st1)[0])
            conf_matrix[u_i, -1] = int(fn)

        for u_i, u1 in enumerate(np.array(sorting1.getUnitIds())[idxs_unmatched]):
            lab_st1 = self._labels_st1[u1]
            fn = len(np.where('FN' == lab_st1)[0])
            conf_matrix[u_i + len(idxs_matched), -1] = int(fn)

        for u_j, u2 in enumerate(sorting2.getUnitIds()):
            lab_st2 = self._labels_st2[u2]
            fp = len(np.where('FP' == lab_st2)[0])
            st_p = np.where(u2 == unit_map_matched)[0]
            if len(st_p) != 0:
                conf_matrix[-1, st_p] = int(fp)
            else:
                st2_unmatched.append(int(u2))
                conf_matrix[-1, len(idxs_matched) + len(st2_unmatched) - 1] = int(fp)

        self._confusion_matrix = conf_matrix
        st2_idxs = np.append(st2_matched, st2_unmatched)

        return st1_idxs, st2_idxs

    @staticmethod
    def compareSpikeTrains(spiketrain1, spiketrain2, delta_tp=10, verbose=False):
        lab_st1 = np.array(['UNPAIRED'] * len(spiketrain1))
        lab_st2 = np.array(['UNPAIRED'] * len(spiketrain2))

        if verbose:
            print('Finding TP')
        # from gtst: TP, TPO, TPSO, FN, FNO, FNSO
        for sp_i, n_sp in enumerate(spiketrain1):
            id_sp = np.where((spiketrain2 > n_sp - delta_tp) & (spiketrain2 < n_sp + delta_tp))[0]
            if len(id_sp) == 1:
                lab_st1[sp_i] = 'TP'
                lab_st2[id_sp] = 'TP'

        if verbose:
            print('Finding FP and FN')
        for l_gt, lab in enumerate(lab_st1):
            if lab == 'UNPAIRED':
                lab_st1[l_gt] = 'FN'

        for l_gt, lab in enumerate(lab_st2):
            if lab == 'UNPAIRED':
                lab_st2[l_gt] = 'FP'

        return lab_st1, lab_st2


class MappedSortingExtractor(se.SortingExtractor):
    def __init__(self, sorting, unit_map):
        se.SortingExtractor.__init__(self)
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
    membership = np.concatenate((np.ones(times1.shape) * 1, np.ones(times2.shape) * 2))
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
    conf_matrix = np.zeros((len(gtst) + 1, len(sst) + 1), dtype=int)
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
            conf_matrix[gt_i, gt_i] = int(tp)
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
        conf_matrix[gt_i + len(gtst_clean), -1] = int(fn)
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
                if z > np.max(conf_matrix) / 2.:
                    ax.text(j, i, '{:d}'.format(z), ha='center', va='center', color='white')
                else:
                    ax.text(j, i, '{:d}'.format(z), ha='center', va='center', color='black')
                    # ,   bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

        ax.axhline(int(len(gtst) - 1) + 0.5, color='black')
        ax.axvline(int(len(sst) - 1) + 0.5, color='black')

        # Major ticks
        ax.set_xticks(np.arange(0, len(sst) + 1))
        ax.set_yticks(np.arange(0, len(gtst) + 1))
        ax.xaxis.tick_bottom()
        # Labels for major ticks
        ax.set_xticklabels(np.append(np.append(sst_idxs, sst_extra).astype(int), 'FN'), fontsize=12)
        ax.set_yticklabels(np.append(gtst_idxs, 'FP'), fontsize=12)

        if xlabel == None:
            ax.set_xlabel('Sorted spike trains', fontsize=15)
        else:
            ax.set_xlabel(xlabel, fontsize=20)
        if ylabel == None:
            ax.set_ylabel('Ground truth spike trains', fontsize=15)
        else:
            ax.set_ylabel(ylabel, fontsize=20)

    return conf_matrix, ax


def compute_performance(SC):
    counts = SC.counts

    tp_rate = float(counts['TP']) / counts['TOT_ST1'] * 100
    cl_rate = float(counts['CL']) / counts['TOT_ST1'] * 100
    fn_rate = float(counts['FN']) / counts['TOT_ST1'] * 100
    fp_st1 = float(counts['FP']) / counts['TOT_ST1'] * 100
    fp_st2 = float(counts['FP']) / counts['TOT_ST2'] * 100

    accuracy = tp_rate / (tp_rate + fn_rate + fp_st1) * 100
    sensitivity = tp_rate / (tp_rate + fn_rate) * 100
    miss_rate = fn_rate / (tp_rate + fn_rate) * 100
    precision = tp_rate / (tp_rate + fp_st1) * 100
    false_discovery_rate = fp_st1 / (tp_rate + fp_st1) * 100

    print('PERFORMANCE: \n')
    print('TP: ', tp_rate, ' %')
    print('CL: ', cl_rate, ' %')
    print('FN: ', fn_rate, ' %')
    print('FP (%ST1): ', fp_st1, ' %')
    print('FP (%ST2): ', fp_st2, ' %')

    print('\nACCURACY: ', accuracy, ' %')
    print('SENSITIVITY: ', sensitivity, ' %')
    print('MISS RATE: ', miss_rate, ' %')
    print('PRECISION: ', precision, ' %')
    print('FALSE DISCOVERY RATE: ', false_discovery_rate, ' %')

    performance = {'tp': tp_rate, 'cl': cl_rate, 'fn': fn_rate, 'fp_st1': fp_st1, 'fp_st2': fp_st2,
                   'accuracy': accuracy, 'sensitivity': sensitivity, 'precision': precision, 'miss_rate': miss_rate,
                   'false_disc_rate': false_discovery_rate}

    return performance
