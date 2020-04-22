from abc import ABC, abstractmethod


# Baseclass for each quality metric

class QualityMetric(ABC):
    def __init__(
            self,
            metric_data,
            metric_name
    ):
        '''
        Parameters
        ----------
        metric_data: MetricData
            An object for storing and computing preprocessed data 
        '''
        self._metric_data = metric_data
        self._metric_name = metric_name

    # implemented by quality metric subclasses
    @abstractmethod
    def compute_metric(self, **kwargs):
        pass

    @abstractmethod
    def threshold_metric(self, threshold, threshold_sign, **kwargs):
        '''
        Parameters
        ----------
        threshold: int or float
            The threshold for the given metric.
        threshold_sign: str
            If 'less', will threshold any metric less than the given threshold.
            If 'less_or_equal', will threshold any metric less than or equal to the given threshold.
            If 'greater', will threshold any metric greater than the given threshold.
            If 'greater_or_equal', will threshold any metric greater than or equal to the given threshold.
        Returns
        -------
        tc: ThresholdCurator
            The thresholded sorting extractor.
        '''
        pass

    def save_property_or_features(self, sorting, metric, metric_name):
        for i_u, u in enumerate(self._metric_data._unit_ids):
            sorting.set_unit_property(u, metric_name, metric[i_u])
