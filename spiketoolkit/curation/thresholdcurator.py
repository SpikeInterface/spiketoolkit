from .curationsortingextractor import CurationSortingExtractor
import spiketoolkit as st


class ThresholdCurator(CurationSortingExtractor):
    def __init__(self, sorting, metrics_epoch):
        '''
        Parent class for all threshold-based curators.
        
        Parameters
        ----------
        sorting: SortingExtractor
            The sorting result to be evaluated.
        metrics_epoch: np.array
            The metrics for the given epoch to be thresholded.
        '''
        CurationSortingExtractor.__init__(self, parent_sorting=sorting)
        self._metrics_epoch = metrics_epoch

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
                if self._metrics_epoch[i] < threshold:
                    units_to_be_excluded.append(unit_id)
            elif threshold_sign == 'less_or_equal':
                if self._metrics_epoch[i] <= threshold:
                    units_to_be_excluded.append(unit_id)
            elif threshold_sign == 'greater':
                if self._metrics_epoch[i] > threshold:
                    units_to_be_excluded.append(unit_id)
            elif threshold_sign == 'greater_or_equal':
                if self._metrics_epoch[i] >= threshold:
                    units_to_be_excluded.append(unit_id)
            else:
                raise ValueError('Not a correct threshold sign.')
        self.exclude_units(units_to_be_excluded)
