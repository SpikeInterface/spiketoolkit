import numpy as np
import spikeextractors as se
from scipy.optimize import linear_sum_assignment
from .sortingcomparison import SortingComparison
import networkx as nx


class MultiSortingComparison():
    def __init__(self, sorting_list, name_list=None, delta_tp=10, minimum_accuracy=0.5):
        if len(sorting_list) > 1 and np.all(isinstance(s, se.SortingExtractor) for s in sorting_list):
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
                    print("Comparing: ", self._name_list[i], " and ", self._name_list[j])
                    comparison_.append(SortingComparison(self._sorting_list[i], self._sorting_list[j],
                                                         sorting1_name=self._name_list[i],
                                                         sorting2_name=self._name_list[j],
                                                         delta_tp=self._delta_tp,
                                                         minimum_accuracy=self._min_accuracy,
                                                         verbose=True))
            self.sorting_comparisons[self._name_list[i]] = comparison_

        # create graph
        agreement = {}
        graph = nx.Graph()
        for sort_name, sort_comp in self.sorting_comparisons.items():
            unit_agreement = {}
            units = sort_comp[0].getSorting1().getUnitIds()
            for unit in units:
                matched_list = {}
                matched_agreement = {}
                for sc in sort_comp:
                    mapped_unit = sc.getMappedSorting1().getMappedUnitIds(unit)
                    mapped_agr = sc.getAgreementFraction(unit, sc.getMappedSorting1().getMappedUnitIds(unit))
                    matched_list[sc.sorting2_name] = mapped_unit
                    matched_agreement[sc.sorting2_name] = mapped_agr

                    node1_name = sort_name + '_' + str(unit)
                    graph.add_node(node1_name)
                    if mapped_unit != -1:
                        node2_name = sc.sorting2_name + '_' + str(mapped_unit)
                        if node2_name not in graph:
                            graph.add_node(node2_name)
                            print('Adding edge: ', node1_name, node2_name)
                            graph.add_edge(node1_name, node2_name, weight=mapped_agr)
                unit_agreement[unit] = {'units': matched_list, 'score': matched_agreement}
            agreement[sort_name] = unit_agreement
        self.agreement = agreement
        self.graph = graph.to_undirected()

        self._new_units = {}
        self._spiketrains = []
        added_nodes = []
        unit_id = 0
        for n in self.graph.nodes():
            edges = graph.edges(n, data=True)
            sorter, unit = (str(n)).split('_')
            unit = int(unit)
            if len(edges) == 0:
                matched_num = 1
                avg_agr = 0
                sorting_idxs = {sorter: unit}
                self._new_units[unit_id] = {'matched_number': matched_num,
                                            'avg_agreement': avg_agr,
                                            'sorter_unit_ids': sorting_idxs}
                self._spiketrains.append(self._sorting_list[self._name_list.index(sorter)].getUnitSpikeTrain(unit))
                unit_id += 1
                # print("ADDING NODE (no match): ", n)
                added_nodes.append(str(n))
            else:
                # check if other nodes have edges (we should also check edges of
                all_edges = list(edges)
                for e in edges:
                    n1, n2, d = e
                    new_edge = self.graph.edges(n2, data=True)
                    if len(new_edge) > 0:
                        for e_n in new_edge:
                            e_n1, e_n2, d = e_n
                            if sorted([e_n1, e_n2]) not in [sorted([u, v]) for u, v, _ in all_edges]:
                                all_edges.append(e_n)
                matched_num = len(all_edges) + 1
                avg_agr = np.mean([d['weight'] for u, v, d in all_edges])
                max_edge = list(all_edges)[np.argmax([d['weight'] for u, v, d in all_edges])]
                n1, n2, d = max_edge

                if n1 not in added_nodes and n2 not in added_nodes:
                    sorter1, unit1 = n1.split('_')
                    sorter2, unit2 = n2.split('_')
                    unit1 = int(unit1)
                    unit2 = int(unit2)
                    sp1 = self._sorting_list[self._name_list.index(sorter1)].getUnitSpikeTrain(unit1)
                    sp2 = self._sorting_list[self._name_list.index(sorter1)].getUnitSpikeTrain(unit1)
                    lab1, lab2 = SortingComparison.compareSpikeTrains(sp1, sp2)
                    tp_idx1 = np.where(np.array(lab1) == 'TP')
                    tp_idx2 = np.where(np.array(lab2) == 'TP')
                    assert len(tp_idx1) == len(tp_idx2)
                    sp_tp1 = list(np.array(sp1)[tp_idx1])
                    sp_tp2 = list(np.array(sp2)[tp_idx2])
                    assert np.all(sp_tp1 == sp_tp2)
                    sorting_idxs = {sorter1: unit1, sorter2: unit2}
                    self._new_units[unit_id] = {'matched_number': matched_num,
                                                'avg_agreement': avg_agr,
                                                'sorter_unit_ids': sorting_idxs}
                    self._spiketrains.append(sp_tp1)
                    unit_id += 1
                    # print("ADDING NODES: ", n, n1, n2, d['weight'])
                    added_nodes.append(str(n))
                    added_nodes.append(str(n1))
                    added_nodes.append(str(n2))
        self.added_nodes = added_nodes

    def plotAgreement(self, minimum_matching=0):
        import matplotlib.pylab as plt
        sorted_name_list = sorted(self._name_list)
        sorting_agr = AgreementSortingExtractor(self, minimum_matching)
        unit_ids = sorting_agr.getUnitIds()
        agreement_matrix = np.zeros((len(unit_ids), len(sorted_name_list)))

        for u_i, unit in enumerate(unit_ids):
            for sort_name, sorter in enumerate(sorted_name_list):
                assigned_unit = sorting_agr.getUnitProperty(unit, 'sorter_unit_ids')[sorter]
                if assigned_unit == -1:
                    agreement_matrix[u_i, sort_name] = np.nan
                else:
                    agreement_matrix[u_i, sort_name] = sorting_agr.getUnitProperty(unit, 'avg_agreement')

        fig, ax = plt.subplots()
        # Using matshow here just because it sets the ticks up nicely. imshow is faster.
        ax.matshow(agreement_matrix, cmap='Greens')

        # Major ticks
        ax.set_xticks(np.arange(0, len(sorted_name_list)))
        ax.set_yticks(np.arange(0, len(unit_ids)))
        ax.xaxis.tick_bottom()
        # Labels for major ticks
        ax.set_xticklabels(sorted_name_list, fontsize=12)
        ax.set_yticklabels(unit_ids, fontsize=12)

        ax.set_xlabel('Sorters', fontsize=15)
        ax.set_ylabel('Units', fontsize=20)

        return ax


class AgreementSortingExtractor(se.SortingExtractor):
    def __init__(self, multisortingcomparison, min_agreement=0):
        se.SortingExtractor.__init__(self)
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
                                 value=self._msc._new_units[unit]['sorter_unit_ids'])

    def getUnitIds(self, unit_ids=None):
        if unit_ids is None:
            return self._unit_ids
        else:
            return self._unit_ids[unit_ids]

    def getUnitSpikeTrain(self, unit_id, start_frame=None, end_frame=None):
        if unit_id not in self.getUnitIds():
            raise Exception("Unit id is invalid")
        return np.array(self._msc._spiketrains[self.getUnitIds().index(unit_id)])
