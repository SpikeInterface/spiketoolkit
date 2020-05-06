from spikeextractors import SortingExtractor
import numpy as np


# A Sorting Extractor that allows for manual curation of a sorting result (Represents curation as a tree of units)

class CurationSortingExtractor(SortingExtractor):

    def __init__(self, parent_sorting, curation_steps=None):
        SortingExtractor.__init__(self)
        self._parent_sorting = parent_sorting
        self._original_unit_ids = list(np.copy(parent_sorting.get_unit_ids()))
        self._all_ids = list(np.copy(parent_sorting.get_unit_ids()))

        # Create and store roots with original unit ids and cached spiketrains
        self._roots = []
        for unit_id in self._original_unit_ids:
            root = Unit(unit_id)
            root.set_spike_train(parent_sorting.get_unit_spike_train(unit_id))
            self._roots.append(root)
        '''
        Copies over properties and spike features from parent_sorting.
        Only spike features will be preserved with merges and splits, properties
        cannot be resolved in these cases.
        '''
        self.copy_unit_properties(parent_sorting)
        self.copy_unit_spike_features(parent_sorting)

        self.curation_steps = curation_steps
        self._kwargs = {'parent_sorting': parent_sorting.make_serialized_dict(), 'curation_steps': self.curation_steps}

        self.curation_steps = []
        if curation_steps is not None:
            assert isinstance(curation_steps,
                              list), "previous_curation_steps must be a list of previous curation commands"
            for i, curation_step in enumerate(curation_steps):
                command, arguments = curation_step
                if command == 'exclude_units':
                    assert len(arguments) == 1, "Length of arguments must be 1 for exclude_units"
                    unit_ids = arguments[0]
                    self.exclude_units(unit_ids=unit_ids)
                elif command == 'merge_units':
                    assert len(arguments) == 1, "Length of arguments must be 1 for merge_units"
                    unit_ids = arguments[0]
                    self.merge_units(unit_ids=unit_ids)
                elif command == 'split_unit':
                    assert len(arguments) == 2, "Length of arguments must be 2 for split_unit"
                    unit_id = arguments[0]
                    indices = arguments[1]

                    self.split_unit(unit_id=unit_id, indices=indices)
                else:
                    raise ValueError("{} is not a valid curation command".format(command))

    def get_unit_ids(self):
        unit_ids = []
        for root in self._roots:
            unit_ids.append(root.unit_id)
        return unit_ids

    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = np.Inf

        valid_unit_id = False
        spike_train = np.asarray([])
        for root in self._roots:
            if root.unit_id == unit_id:
                valid_unit_id = True
                full_spike_train = root.get_spike_train()
                inds = np.where((start_frame <= full_spike_train) & (full_spike_train < end_frame))
                spike_train = full_spike_train[inds]
        if valid_unit_id:
            return spike_train
        else:
            raise ValueError(str(unit_id) + " is an invalid unit id")

    def get_sampling_frequency(self):
        return self._parent_sorting.get_sampling_frequency()

    def print_curation_tree(self, unit_id):
        '''This function prints the current curation tree for the unit_id (roots are current unit ids).

        Parameters
        ----------
        unit_id: in
            The unit id whose curation history will be printed.
        '''
        root_ids = []
        for i in range(len(self._roots)):
            root_id = self._roots[i].unit_id
            root_ids.append(root_id)
        if unit_id in root_ids:
            root_index = root_ids.index(unit_id)
            print(self._roots[root_index])
        else:
            raise ValueError("invalid unit id")

    def exclude_units(self, unit_ids):
        '''This function deletes roots from the curation tree according to the given unit_ids

        Parameters
        ----------
        unit_ids: list or int
            The unit ids to be excluded
        append_curation_step: bool
            Appends the curation step to the object keyword arguments
        '''
        if isinstance(unit_ids, (int, np.integer)):
            unit_ids = [unit_ids]
        if len(unit_ids) == 0:
            return None

        root_ids = []
        for i in range(len(self._roots)):
            root_id = self._roots[i].unit_id
            root_ids.append(root_id)

        if set(unit_ids).issubset(set(root_ids)) and len(unit_ids) > 0:
            indices_to_be_deleted = []
            for unit_id in unit_ids:
                root_index = root_ids.index(unit_id)
                indices_to_be_deleted.append(root_index)
                if unit_id in self._features:
                    del self._features[unit_id]
            self._roots = [self._roots[i] for i, _ in enumerate(root_ids) if i not in indices_to_be_deleted]
            self.curation_steps.append(('exclude_units', (list(unit_ids),)))
            self._kwargs['curation_steps'] = self.curation_steps
        else:
            raise ValueError(str(unit_ids) + " has one or more invalid unit ids")

    def merge_units(self, unit_ids):
        '''This function merges two roots from the curation tree according to the given unit_ids. It creates a new
        unit_id and root that has the merged roots as children.

        Parameters
        ----------
        unit_ids: list
            The unit ids to be merged

        Returns
        -------
        new_root_id: int
            The unit id of the new merged unit.
        '''

        if len(unit_ids) <= 1:
            return None

        root_ids = []
        for i in range(len(self._roots)):
            root_id = self._roots[i].unit_id
            root_ids.append(root_id)

        indices_to_be_deleted = []
        if set(unit_ids).issubset(set(root_ids)):
            # Find all unique feature names and create all feature lists
            all_feature_names = []
            for unit_id in unit_ids:
                feature_names = self.get_unit_spike_feature_names(unit_id)
                all_feature_names.append(feature_names)

            shared_feature_names = set(all_feature_names[0])
            for feature_names in all_feature_names[1:]:
                shared_feature_names.intersection_update(feature_names)
            shared_feature_names = list(shared_feature_names)

            shared_features = []
            shared_features_idxs = []
            for i in range(len(shared_feature_names)):
                shared_features.append([])
                shared_features_idxs.append([])

            new_root_id = max(self._all_ids) + 1
            self._all_ids.append(new_root_id)
            new_root = Unit(new_root_id)
            all_spike_trains = []
            for unit_id in unit_ids:
                root_index = root_ids.index(unit_id)
                new_root.add_child(self._roots[root_index])
                all_spike_trains.append(self._roots[root_index].get_spike_train())
                for i, feature_name in enumerate(shared_feature_names):
                    if not feature_name.endswith('_idxs'):
                        features = self.get_unit_spike_features(unit_id, feature_name)
                        shared_features[i].append(features)
                        if feature_name + "_idxs" in shared_feature_names:
                            features_idxs = self.get_unit_spike_features(unit_id, feature_name + "_idxs")
                        else:
                            features_idxs = []
                        shared_features_idxs[i].append(features_idxs)

                del self._features[unit_id]
                self._roots[root_index].set_spike_train(np.asarray([]))  # clear spiketrain
                indices_to_be_deleted.append(root_index)

            spike_train = np.concatenate(all_spike_trains)
            sort_indices = np.argsort(spike_train)
            new_root.set_spike_train(np.asarray(spike_train)[sort_indices])
            # del all_spike_trains
            self._roots = [self._roots[i] for i, _ in enumerate(root_ids) if i not in indices_to_be_deleted]
            self._roots.append(new_root)

            # copy features
            for i, feature_name in enumerate(shared_feature_names):
                if not feature_name.endswith('_idxs'):
                    # Calc new idxs list, empty if their is no idxs for the feature 
                    shared_features_idxs_unsorted = []
                    for n, feature_idxs in enumerate(shared_features_idxs[i]):
                        new_idxs = []
                        for idxs in feature_idxs:
                            new_idxs.append(np.argwhere(spike_train[sort_indices] == all_spike_trains[n][idxs])[0][0])
                        shared_features_idxs_unsorted += new_idxs
                    if not len(shared_features_idxs_unsorted) == 0:
                        arg_sort_idxs = np.argsort(shared_features_idxs_unsorted)
                        shared_features_idxs_sorted = np.array(shared_features_idxs_unsorted)[arg_sort_idxs]
                        shared_features_sorted = np.concatenate(shared_features[i])[arg_sort_idxs]
                    else:  # if empty, don't use idxs
                        shared_features_idxs_sorted = None
                        shared_features_sorted = np.concatenate(shared_features[i])[sort_indices]
                    self.set_unit_spike_features(new_root_id, feature_name,
                                                 shared_features_sorted,
                                                 indexes=shared_features_idxs_sorted)
            # properties are not copied
            del spike_train
            del all_spike_trains
            self.curation_steps.append(('merge_units', (list(unit_ids),)))
            self._kwargs['curation_steps'] = self.curation_steps
            return new_root_id
        else:
            raise ValueError(str(unit_ids) + " has one or more invalid unit ids")

    def split_unit(self, unit_id, indices):
        '''This function splits a root from the curation tree according to the given unit_id and indices. It creates
        two new unit_ids and roots that have the split root as a child. This function splits the spike train of the
        root by the given indices.

        Parameters
        ----------
        unit_id: int
            The unit id to be split
        indices: list
            The indices of the unit spike train at which the spike train will be split.

        Returns
        -------
        new_root_ids: tuple
            A tuple of new unit ids after the split (integers).
        '''
        root_ids = []
        for i in range(len(self._roots)):
            root_id = self._roots[i].unit_id
            root_ids.append(root_id)

        if unit_id in root_ids:
            indices_1 = np.sort(np.asarray(list(set(indices)), dtype=int))

            root_index = root_ids.index(unit_id)
            new_child = self._roots[root_index]
            original_spike_train = self._roots[root_index].get_spike_train()

            try:
                spike_train_1 = original_spike_train[indices_1]
            except IndexError:
                print(str(indices) + " out of bounds for the spike train of " + str(unit_id))

            indices_2 = np.array(list(set(range(len(original_spike_train))) - set(indices_1)), dtype=int)
            spike_train_2 = original_spike_train[indices_2]
            del original_spike_train

            new_root_1_id = max(self._all_ids) + 1
            self._all_ids.append(new_root_1_id)
            new_root_1 = Unit(new_root_1_id)
            new_root_1.add_child(new_child)
            new_root_1.set_spike_train(spike_train_1)

            new_root_2_id = max(self._all_ids) + 1
            self._all_ids.append(new_root_2_id)
            new_root_2 = Unit(new_root_2_id)
            new_root_2.add_child(new_child)
            new_root_2.set_spike_train(spike_train_2)

            self._roots.append(new_root_1)
            self._roots.append(new_root_2)

            # copy features
            for feature_name in self.get_unit_spike_feature_names(unit_id):
                if feature_name.endswith('_idxs'):
                    continue
                full_features = self.get_unit_spike_features(unit_id, feature_name)
                if isinstance(full_features, (list, range)):
                    full_features = np.array(full_features)
                if not feature_name + '_idxs' in self.get_unit_spike_feature_names(unit_id):
                    self.set_unit_spike_features(new_root_1_id, feature_name, full_features[indices_1])
                    self.set_unit_spike_features(new_root_2_id, feature_name, full_features[indices_2])
                else:
                    full_features_idxs = np.array(self.get_unit_spike_features(unit_id, feature_name + '_idxs'))
                    indices_1_idxs = np.array([n for n, i in enumerate(full_features_idxs) if i in indices], dtype=int)
                    indices_2_idxs = np.array([n for n, i in enumerate(full_features_idxs) if not i in indices],
                                              dtype=int)
                    # Calc new idxs after split
                    indexes_1 = []
                    indexes_2 = []
                    for i in full_features_idxs:
                        if i in indices:
                            indexe_1 = np.count_nonzero(np.array(indices) < i)
                            indexes_1.append(indexe_1)
                        else:
                            indexe_2 = i - np.count_nonzero(np.array(indices) < i)
                            indexes_2.append(indexe_2)
                    indexes_1 = np.array(indexes_1, dtype=int)
                    indexes_2 = np.array(indexes_2, dtype=int)
                    self.set_unit_spike_features(new_root_1_id, feature_name,
                                                 full_features[indices_1_idxs],
                                                 indexes=indexes_1)
                    self.set_unit_spike_features(new_root_2_id, feature_name,
                                                 full_features[indices_2_idxs],
                                                 indexes=indexes_2)

            # properties are not copied

            del self._features[unit_id]
            del self._roots[root_index]
            self.curation_steps.append(('split_unit', (unit_id, indices)))
            self._kwargs['curation_steps'] = self.curation_steps
            return (new_root_1_id, new_root_2_id)
        else:
            raise ValueError(str(unit_id) + " non-valid unit id")


# The Unit class is a node in the curation tree. Each Unit contains its unit_id, children, and spike_train.
class Unit(object):
    def __init__(self, unit_id):
        self.unit_id = unit_id
        self.children = []
        self.spike_train = np.asarray([])

    def set_spike_train(self, spike_train):
        self.spike_train = spike_train

    def get_spike_train(self):
        return self.spike_train

    def add_child(self, child):
        self.children.append(child)

    def get_children(self):
        return self.children

    def __str__(self, level=0):
        if level == 0:
            ret = "\t" * (max(level - 1, 0)) + repr(self.unit_id) + "\n"
        else:
            ret = "\t" * (max(level - 1, 0)) + "^-------" + repr(self.unit_id) + "\n"
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret
