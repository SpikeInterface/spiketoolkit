import numpy as np
import spikeinterface as si
from scipy.optimize import linear_sum_assignment
from .sortingcomparison import SortingComparison

class MultiSortingComparison():
    def __init__(self, sorting_list, name_list=None, delta_tp=10, minimum_accuracy=0.5):
        if len(sorting_list) > 1 and np.all(isinstance(s, si.SortingExtractor) for s in sorting_list):
            self._sorting_list = sorting_list
        if name_list is not None and len(name_list) == len(sorting_list):
            self._name_list = name_list
        else:
            self._name_list = range(len(sorting_list))
        self._delta_tp = delta_tp
        self._min_accuracy = minimum_accuracy
        self._do_matching()

    def getSortingList(self):
        return self._sorting_list

    def getAgreementSorting(self, minimum_matching=0):
        return AgreementSortingExtractor(self, min_agreement=minimum_matching)

    def _do_matching(self):
        # do pairwise matching
        self.sorting_comparisons = {}
        for i in range(len(self._sorting_list)):
            comparison_ = []
            for j in range(len(self._sorting_list)):
                if i != j:
                    comparison_.append(SortingComparison(self._sorting_list[i], self._sorting_list[j],
                                                         sorting1_name=self._name_list[i],
                                                         sorting2_name=self._name_list[j],
                                                         delta_tp=self._delta_tp,
                                                         minimum_accuracy=self._min_accuracy))
            self.sorting_comparisons[self._name_list[i]] = comparison_

        agreement = {}
        for s_i, sort_comp in self.sorting_comparisons.items():
            unit_agreement = {}
            units = sort_comp[0].getSorting1().getUnitIds()
            for unit in units:
                matched_list = {}
                matched_agreement = []
                for sc in sort_comp:
                    matched_list[sc.sorting2_name] = sc.getMappedSorting1().getMappedUnitIds(unit)
                    matched_agreement.append(sc.getAgreementFraction(unit,
                                                                     sc.getMappedSorting1().getMappedUnitIds(unit)))
                unit_agreement[unit] = {'units': matched_list, 'score': matched_agreement}
            agreement[s_i] = unit_agreement

        self.agreement = agreement

        # find units in agreement
        new_unit_ids = []
        unit_avg_agreement = []
        mathced_in = []
        unit_id = 0
        self._new_units = {}
        self._spiketrains = []
        for s_i, agr in agreement.items():
            for unit in agr.keys():
                unit_assignments = list(agr[unit]['units'].values())
                idxs = np.where(np.array(unit_assignments) != -1)
                if len(idxs[0]) != 0:
                    avg_agr = np.mean(np.array(agr[unit]['score'])[idxs])
                else:
                    avg_agr = 0
                matched_num = len(np.where(np.array(unit_assignments) != -1)[0]) + 1
                sorting_idxs = agr[unit]['units']
                sorting_idxs[s_i] = unit
                if len(self._new_units.keys()) == 0:
                    self._new_units[unit_id] = {'matched_number': matched_num,
                                                'avg_agreement': avg_agr,
                                                'sorting_idx': sorting_idxs}
                    self._spiketrains.append(self.sorting_comparisons[s_i][0].getSorting1().getUnitSpikeTrain(unit))
                    unit_id += 1
                else:
                    matched_already = False
                    for n_u, n_val in self._new_units.items():
                        if n_val['sorting_idx'][s_i] == unit:
                            matched_already = True
                    if not matched_already:
                        self._new_units[unit_id] = {'matched_number': matched_num,
                                                    'avg_agreement': avg_agr,
                                                    'sorting_idx': sorting_idxs}
                        self._spiketrains.append(self.sorting_comparisons[s_i][0].getSorting1().getUnitSpikeTrain(unit))
                        unit_id += 1

class AgreementSortingExtractor(si.SortingExtractor):
    def __init__(self, multisortingcomparison, min_agreement=0):
        si.SortingExtractor.__init__(self)
        self._msc = multisortingcomparison
        if min_agreement == 0:
            self._unit_ids = list(self._msc._new_units.keys())
        else:
            self._unit_ids = list(u for u in self._msc._new_units.keys()
                                  if self._msc._new_units[u]['matched_number'] >= min_agreement)

        for unit in self._unit_ids:
            self.setUnitProperty(unit_id=unit, property_name='matched_number',
                                 value=self._msc._new_units[unit]['matched_number'])
            self.setUnitProperty(unit_id=unit, property_name='avg_agreement',
                                 value=self._msc._new_units[unit]['avg_agreement'])
            self.setUnitProperty(unit_id=unit, property_name='sorter_unit_ids',
                                 value=self._msc._new_units[unit]['sorting_idx'])

    def getUnitIds(self, unit_ids=None):
        if unit_ids is None:
            return self._unit_ids
        else:
            return self._unit_ids[unit_ids]

    def getUnitSpikeTrain(self, unit_id, start_frame=None, end_frame=None):
        if unit_id not in self.getUnitIds():
            raise Exception("Unit id is invalid")
        return self._msc._spiketrains[self.getUnitIds().item(unit_id)]