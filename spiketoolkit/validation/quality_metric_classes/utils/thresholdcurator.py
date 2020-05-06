from .curationsortingextractor import CurationSortingExtractor


class ThresholdCurator(CurationSortingExtractor):
    def __init__(self, sorting, metric, threshold=None, threshold_sign=None):
        '''
        Parent class for all threshold-based curators.
        
        Parameters
        ----------
        sorting: SortingExtractor
            The sorting result to be evaluated.
        metrics: np.array
            The metric to be thresholded.
        '''
        CurationSortingExtractor.__init__(self, parent_sorting=sorting)
        self._metric = metric
        self._threshold = threshold
        self._threshold_sign = threshold_sign
        # bypass dumping mechanism of CurationSortingExtractor
        del self._kwargs
        self._kwargs = {'sorting': sorting.make_serialized_dict(), 'metric': metric,
                        'threshold': threshold, 'threshold_sign': threshold_sign}
        if threshold is not None and threshold_sign is not None:
            self.threshold_sorting(threshold, threshold_sign)

    def threshold_sorting(self, threshold, threshold_sign):
        '''
        Parameters
        ----------
        threshold:
            The threshold for the given metric.
        threshold_sign: str
            If 'less', will threshold any metric less than the given threshold.
            If 'less_or_equal', will threshold any metric less than or equal to the given threshold.
            If 'greater', will threshold any metric greater than the given threshold.
            If 'greater_or_equal', will threshold any metric greater than or equal to the given threshold.
        '''
        units_to_be_excluded = []
        for i, unit_id in enumerate(self._parent_sorting.get_unit_ids()):
            if threshold_sign == 'less':
                if self._metric[i] < threshold:
                    units_to_be_excluded.append(unit_id)
            elif threshold_sign == 'less_or_equal':
                if self._metric[i] <= threshold:
                    units_to_be_excluded.append(unit_id)
            elif threshold_sign == 'greater':
                if self._metric[i] > threshold:
                    units_to_be_excluded.append(unit_id)
            elif threshold_sign == 'greater_or_equal':
                if self._metric[i] >= threshold:
                    units_to_be_excluded.append(unit_id)
            else:
                raise ValueError('Not a correct threshold sign.')
        self.exclude_units(units_to_be_excluded)
        # bypass dumping mechanism of CurationSortingExtractor
        if 'curation_steps' in self._kwargs.keys():
            del self._kwargs['curation_steps']
        self._kwargs['threshold'] = threshold
        self._kwargs['threshold_sign'] = threshold_sign
