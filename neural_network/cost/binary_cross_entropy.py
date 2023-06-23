import numpy as np
from typing import Union
from numbers import Number

from ..base.cost_mixin import CostMixin
from ..base.metadata_mixin import MetadataMixin
from ..base.save_mixin import SaveMixin


from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export


@export
class BinaryCrossEntropy(CostMixin, MetadataMixin, SaveMixin):
    name = 'binarycrossentropy'

    '''
    Provides static methods to compute the binary cost entropy loss
    and it's derivative corresponding to the sigmoid function
    '''
    @staticmethod
    @type_safe
    @not_none
    def apply(y_true: np.ndarray, y_pred: np.ndarray) -> Union[Number, np.number]:
        '''
        Applies the cross entropy loss on the given labels and predictions

        Parameters:
            y_true: numpy.ndarray of shape (n_samples, n_labels)
                The known true labels
            y_pred: numpy.ndarray of shape (n_samples, n_labels)
                The predicted values

        Returns:
            float: average loss over all samples
        '''
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        return np.mean(-(
            (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        ))

    @staticmethod
    @type_safe
    @not_none
    def derivative(
            y_true: np.ndarray,
            y_pred: np.ndarray) -> np.ndarray:
        '''
        Applies the derivative of binary cross entropy loss on the given
        labels and predictions corresponding to the sigmoid function

        Parameters:
            y_true: numpy.ndarray of shape (n_samples, n_labels)
                The known true labels
            y_pred: numpy.ndarray of shape (n_samples, n_labels)
                The predicted values

        Returns:
            numpy.ndarray: The gradient with respect to each sample's known and predicted labels
        '''
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        return (y_pred - y_true) / (y_pred * (1 - y_pred))
