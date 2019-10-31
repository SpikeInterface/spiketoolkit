from .curationlist import *
from .curationsortingextractor import CurationSortingExtractor
import spiketoolkit as st


"""
All supported metrics are:
    'snr',
    'num_spikes',
    'firing_rate',
    'isi_viol',
    'presence_ratio',
"""


class ThresholdMetrics(CurationSortingExtractor):
    curator_name = 'ThresholdMetrics'
    installed = True  # check at class level if installed or not
    curator_gui_params = [
        # TODO: this part needs to be updated in the future.
    ]
    installation_mesg = ""  # err

    def __init__(
        self, sorting, recording, metrics, thresholds, threshold_signs,
        mode=None,
        metric_calculator=None
    ):
        assert len(metrics) == len(thresholds) and len(metrics) == len(threshold_signs), """
            Please make sure that the numbers of metrics, thresholds and signs are equal.
        """

        CurationSortingExtractor.__init__(self, parent_sorting=sorting)
        metric_name = 'metrics'

        self._curation_function_dict = self.build_curation_function_full_dict()
        self._curation_with_recording = self.build_curation_recording_need_list()

        self._units_to_be_excluded = None

        for idx, metric in enumerate(metrics):
            if metric not in self._curation_with_recording:
                _metric_units_to_be_excluded = self._curation_function_dict[metric](
                    sorting,
                    threshold=thresholds[idx],
                    threshold_sign=threshold_signs[idx],
                    sampling_frequency=recording.get_sampling_frequency(),
                    metric_calculator=metric_calculator
                ).get_units_to_be_excluded()
            else:
                _metric_units_to_be_excluded = self._curation_function_dict[metric](
                    sorting=sorting,
                    recording=recording,
                    threshold=thresholds[idx],
                    threshold_sign=threshold_signs[idx],
                    metric_calculator=metric_calculator
                ).get_units_to_be_excluded()
            
            if self._units_to_be_excluded is None:
                self._units_to_be_excluded = _metric_units_to_be_excluded
            else:
                if mode == 'and':
                    self._units_to_be_excluded = self.intersecton(self._units_to_be_excluded, _metric_units_to_be_excluded)
                elif mode == 'or':
                    self._units_to_be_excluded = \
                        self.union(self._units_to_be_excluded, _metric_units_to_be_excluded)
                else:
                    raise NotImplementedError('Mode %s is unknown' % mode)

        self.exclude_units(self._units_to_be_excluded)
        
    @staticmethod
    def build_curation_function_full_dict():
        return {
            'snr': threshold_snr,
            'num_spikes': threshold_num_spikes,
            'firing_rate': threshold_firing_rate,
            'isi_viol': threshold_isi_violations,
            'presence_ratio': threshold_presence_ratio
        }

    @staticmethod
    def build_curation_recording_need_list():
        return ['snr']

    @staticmethod
    def union(l1:list, l2:list) -> list:
        return list(set(l1) | set(l2))
    
    @staticmethod
    def intersecton(l1:list, l2:list) -> list:
        return list(set(l1) & set(l2))


def threshold_metrics(sorting, recording, metrics, thresholds, threshold_signs, mode='or',
                      metric_calculator=None):
    """
    Excludes units based on specified metrics and corresponding rules.

    Parameters
    ----------
    sorting: SortingExtractor
        The sorting extractor to be thresholded.
    recording: RecordingExtractor
        The recording extractor for sortings.
    metrics:
        The metrics for thresholding unit.
    threshold:
        The thresholds for the given metrics.
    threshold_sign: str
        If 'less', will threshold any metric less than the given threshold.
        If 'less_or_equal', will threshold any metric less than or equal to the given threshold.
        If 'greater', will threshold any metric greater than the given threshold.
        If 'greater_or_equal', will threshold any metric greater than or equal to the given threshold.
    metric_calculator: MetricCalculator
        A metric calculator can be passed in with cached metrics.
    Returns
    -------
    thresholded_sorting: ThresholdNumSpikes
        The thresholded sorting extractor
    """
    return ThresholdMetrics(
        sorting=sorting,
        recording=recording,
        metrics=metrics,
        thresholds=thresholds,
        threshold_signs=threshold_signs,
        mode=mode,
        metric_calculator=metric_calculator
    )



