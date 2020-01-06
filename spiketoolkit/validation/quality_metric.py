from abc import ABC, abstractmethod

# Baseclass for each quality metric

class QualityMetric(ABC):
    def __init__(
        self,
        metric_data,
    ):
        '''
        Parameters
        ----------
        metric_data: MetricData
            An object for storing and computing preprocessed data 
        '''
        self._metric_data = metric_data

    #implemented by quality metric subclasses
    @abstractmethod
    def compute_metric(self):
        pass

    @abstractmethod
    def threshold_metric(self, threshold, threshold_sign, epoch=0):
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
        epoch:
            The threshold will be applied to the specified epoch. 
            If epoch is None, then it will default to the first epoch. 
        Returns
        -------
        tc: ThresholdCurator
            The thresholded sorting extractor.
        '''
        pass