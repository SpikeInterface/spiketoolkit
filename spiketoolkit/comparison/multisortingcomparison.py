import numpy as np
import spikeextractors as se
from scipy.optimize import linear_sum_assignment
from .sortingcomparison import SortingComparison
from .comparisontools import compare_spike_trains

import networkx as nx


class MultiSortingComparison():
    def __init__(self, sorting_list, name_list=None, delta_time=0.3, min_accuracy=0.5, n_jobs=-1, verbose=False):
        if len(sorting_list) > 1 and np.all(isinstance(s, se.SortingExtractor) for s in sorting_list):
            self._sorting_list = sorting_list
        if name_list is not None and len(name_list) == len(sorting_list):
            self._name_list = name_list
        else:
            self._name_list = ['sorter' + str(i) for i in range(len(sorting_list))]
        self._delta_time = delta_time
        self._min_accuracy = min_accuracy
        self._n_jobs = n_jobs
        self._do_matching(verbose)

    def get_sorting_list(self):
        return self._sorting_list

    def get_agreement_sorting(self, minimum_matching=0):
        return AgreementSortingExtractor(self, min_agreement=minimum_matching)

    def _do_matching(self, verbose):
        # do pairwise matching
        self.sorting_comparisons = {}
        for i in range(len(self._sorting_list)):
            comparison_ = {}
            for j in range(len(self._sorting_list)):
                if i != j:
                    if verbose:
                        print("Comparing: ", self._name_list[i], " and ", self._name_list[j])
                    comparison_[self._name_list[j]] = SortingComparison(self._sorting_list[i], self._sorting_list[j],
                                                                        sorting1_name=self._name_list[i],
                                                                        sorting2_name=self._name_list[j],
                                                                        delta_time=self._delta_time,
                                                                        min_accuracy=self._min_accuracy,
                                                                        n_jobs=self._n_jobs,
                                                                        verbose=verbose)
            self.sorting_comparisons[self._name_list[i]] = comparison_

        assert np.all([s.get_sampling_frequency() == self._sorting_list[0].get_sampling_frequency()
                       for s in self._sorting_list])

        tolerance = int(self._delta_time / 1000 * self._sorting_list[0].get_sampling_frequency())

        # create graph
        agreement = {}
        graph = nx.Graph()
        for sort_name, sort_comp in self.sorting_comparisons.items():
            unit_agreement = {}
            sort_comp2 = list(sort_comp.keys())
            units = sort_comp[sort_comp2[0]].sorting1.get_unit_ids()
            for unit in units:
                matched_list = {}
                matched_agreement = {}
                for k, sc in sort_comp.items():
                    mapped_unit = sc.get_mapped_sorting1().get_mapped_unit_ids(unit)
                    mapped_agr = sc.get_agreement_fraction(unit, sc.get_mapped_sorting1().get_mapped_unit_ids(unit))
                    matched_list[sc.sorting2_name] = mapped_unit
                    matched_agreement[sc.sorting2_name] = mapped_agr
                    node1_name = str(sort_name) + '_' + str(unit)
                    graph.add_node(node1_name)
                    if mapped_unit != -1:
                        node2_name = str(sc.sorting2_name) + '_' + str(mapped_unit)
                        if node2_name not in graph:
                            graph.add_node(node2_name)
                        if (node1_name, node2_name) not in graph.edges:
                            if verbose:
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
                unit_id += 1
                added_nodes.append(str(n))
            else:
                # check if other nodes have edges (we should also check edges of
                all_edges = list(edges)
                for e in edges:
                    n1, n2, d = e
                    n2_edges = self.graph.edges(n2, data=True)
                    if len(n2_edges) > 0:
                        for e_n in n2_edges:
                            n_n1, n_n2, d = e_n
                            if sorted([n_n1, n_n2]) not in [sorted([u, v]) for u, v, _ in all_edges]:
                                all_edges.append(e_n)
                matched_num = len(all_edges) + 1
                avg_agr = np.mean([d['weight'] for u, v, d in all_edges])
                max_edge = list(all_edges)[np.argmax([d['weight'] for u, v, d in all_edges])]

                for edge in all_edges:
                    n1, n2, d = edge
                    if n1 not in added_nodes or n2 not in added_nodes:
                        sorter1, unit1 = n1.split('_')
                        sorter2, unit2 = n2.split('_')
                        unit1 = int(unit1)
                        unit2 = int(unit2)
                        sorting_idxs = {sorter1: unit1, sorter2: unit2}
                        if unit_id not in self._new_units.keys():
                            self._new_units[unit_id] = {'matched_number': matched_num,
                                                        'avg_agreement': avg_agr,
                                                        'sorter_unit_ids': sorting_idxs}
                        else:
                            full_sorting_idxs = self._new_units[unit_id]['sorter_unit_ids']
                            for s, u in sorting_idxs.items():
                                if s not in full_sorting_idxs:
                                    full_sorting_idxs[s] = u
                            self._new_units[unit_id] = {'matched_number': matched_num,
                                                        'avg_agreement': avg_agr,
                                                        'sorter_unit_ids': full_sorting_idxs}
                        added_nodes.append(str(n))
                        if n1 not in added_nodes:
                            added_nodes.append(str(n1))
                        if n2 not in added_nodes:
                            added_nodes.append(str(n2))
                unit_id += 1

        # extract best matches true positive spike trains
        for u, v in self._new_units.items():
            if len(v['sorter_unit_ids'].keys()) == 1:
                self._spiketrains.append(self._sorting_list[self._name_list.index(
                    list(v['sorter_unit_ids'].keys())[0])].get_unit_spike_train(list(v['sorter_unit_ids'].values())[0]))
            else:
                nodes = []
                edges = []
                for sorter, unit in v['sorter_unit_ids'].items():
                    nodes.append((sorter + '_' + str(unit)))
                for n1 in nodes:
                    for n2 in nodes:
                        if n1 != n2:
                            if (n1, n2) not in edges and (n2, n1) not in edges:
                                if (n1, n2) in self.graph.edges:
                                    edges.append((n1, n2))
                                elif (n2, n1) in self.graph.edges:
                                    edges.append((n2, n1))
                max_weight = 0
                for e in edges:
                    w = self.graph.edges.get(e)['weight']
                    if w > max_weight:
                        max_weight = w
                        max_edge = (e[0], e[1], w)
                n1, n2, d = max_edge
                sorter1, unit1 = n1.split('_')
                sorter2, unit2 = n2.split('_')
                unit1 = int(unit1)
                unit2 = int(unit2)
                sp1 = self._sorting_list[self._name_list.index(sorter1)].get_unit_spike_train(unit1)
                sp2 = self._sorting_list[self._name_list.index(sorter2)].get_unit_spike_train(unit2)
                lab1, lab2 = compare_spike_trains(sp1, sp2)
                tp_idx1 = np.where(np.array(lab1) == 'TP')[0]
                tp_idx2 = np.where(np.array(lab2) == 'TP')[0]
                assert len(tp_idx1) == len(tp_idx2)
                sp_tp1 = list(np.array(sp1)[tp_idx1])
                sp_tp2 = list(np.array(sp2)[tp_idx2])
                assert np.allclose(sp_tp1, sp_tp2, atol=tolerance)
                self._spiketrains.append(sp_tp1)
        self.added_nodes = added_nodes

    def _do_agreement_matrix(self, minimum_matching=0):
        sorted_name_list = sorted(self._name_list)
        sorting_agr = AgreementSortingExtractor(self, minimum_matching)
        unit_ids = sorting_agr.get_unit_ids()
        agreement_matrix = np.zeros((len(unit_ids), len(sorted_name_list)))

        for u_i, unit in enumerate(unit_ids):
            for sort_name, sorter in enumerate(sorted_name_list):
                if sorter in sorting_agr.get_unit_property(unit, 'sorter_unit_ids').keys():
                    assigned_unit = sorting_agr.get_unit_property(unit, 'sorter_unit_ids')[sorter]
                else:
                    assigned_unit = -1
                if assigned_unit == -1:
                    agreement_matrix[u_i, sort_name] = np.nan
                else:
                    agreement_matrix[u_i, sort_name] = sorting_agr.get_unit_property(unit, 'avg_agreement')
        return agreement_matrix

    def plot_agreement(self, minimum_matching=0):
        import matplotlib.pylab as plt
        sorted_name_list = sorted(self._name_list)
        sorting_agr = AgreementSortingExtractor(self, minimum_matching)
        unit_ids = sorting_agr.get_unit_ids()
        agreement_matrix = self._do_agreement_matrix(minimum_matching)

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
            self.set_unit_property(unit_id=unit, property_name='matched_number',
                                   value=self._msc._new_units[unit]['matched_number'])
            self.set_unit_property(unit_id=unit, property_name='avg_agreement',
                                   value=self._msc._new_units[unit]['avg_agreement'])
            self.set_unit_property(unit_id=unit, property_name='sorter_unit_ids',
                                   value=self._msc._new_units[unit]['sorter_unit_ids'])

    def get_unit_ids(self, unit_ids=None):
        if unit_ids is None:
            return self._unit_ids
        else:
            return self._unit_ids[unit_ids]

    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        if unit_id not in self.get_unit_ids():
            raise Exception("Unit id is invalid")
        return np.array(self._msc._spiketrains[list(self._msc._new_units.keys()).index(unit_id)])


def compare_multiple_sorters(sorting_list, name_list=None, delta_time=0.3, min_accuracy=0.5,
                             n_jobs=-1, verbose=False):
    '''
    Compares multiple spike sorter outputs.

    - Pair-wise comparisons are made
    - An agreement graph is built based on the agreement score

    It allows to return a consensus-based sorting extractor with the `get_agreement_sorting()` method.

    Parameters
    ----------
    sorting_list: list
        List of sorting extractor objects to be compared
    name_list: list
        List of spike sorter names. If not given, sorters are named as 'sorter0', 'sorter1', 'sorter2', etc.
    delta_time: float
        Number of ms to consider coincident spikes (default 0.3 ms)
    min_accuracy: float
        Minimum agreement score to match units (default 0.5)
    n_jobs: int
       Number of cores to use in parallel. Uses all availible if -1
    verbose: bool
        if True, output is verbose

    Returns
    -------
    multi_sorting_comparison: MultiSortingComparison
        MultiSortingComparison object with the multiple sorter comparison
    '''
    return MultiSortingComparison(sorting_list, name_list, delta_time, min_accuracy, n_jobs, verbose)
