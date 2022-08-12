import numpy as np

from ..exceptions import ExceptionFactory
from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export


errors = {
    'R2ScoreError': ExceptionFactory.register('R2ScoreError'),
}


@type_safe
@not_none
@export
def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    '''
    Calculates accuracy of the model using the R2 Score

    Parameters:
        y_true: np.ndarray of shape (n_samples,)
            known true values
        y_pred: np.ndarray of shape (n_samples,)
            predicted values

    Returns:
        float: The R2 score
    '''
    if y_true.ndim > 1 or y_pred.ndim > 1:
        raise errors['R2ScoreError'](
            f'y_true and y_pred be one dimensional, found '
            f'y_true.shape={y_true.shape}, y_pred.shape={y_pred.shape}'
        )

    if y_true.shape != y_pred.shape:
        raise errors['R2ScoreError'](
            f'y_true and y_pred must have the same shapes, '
            f'{y_true.shape} != {y_pred.shape}'
        )

    ssr = np.sum((y_pred - y_true) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)

    if np.isclose(sst, 0):
        return 0

    return float(1 - (ssr / sst))
