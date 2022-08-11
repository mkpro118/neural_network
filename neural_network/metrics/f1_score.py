import numpy as np

from ..exceptions import ExceptionFactory
from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export
from .precision_score import precision_score
from .recall_score import recall_score

errors = {
    'F1ScoreError': ExceptionFactory.register('F1ScoreError'),
}


_f1 = lambda p, r: 2 * p * r / (p + r)


@type_safe
@not_none
@export
def f1_score(y_true: np.ndarray, y_pred: np.ndarray, *,
             average: str = 'binary', use_multiprocessing: bool = False) -> float:
    '''
    Calculates the f1 score of the model

    It is computed using the formula
           2 * Precision * Recall
    f1 = --------------------------
             Precision + Recall


    Parameters:
        y_true: np.ndarray of shape (n_samples, n_classes)
            known one hot encoded labels
        y_pred: np.ndarray of shape (n_samples, n_classes)
            predicted one hot encoded labels
        average: str, deault = 'binary'
            The averaging mode of the f1 score, the precision and recall
            are calculated the same averaging mode
        use_multiprocessing: bool, default = False
            Set to true to use multiprocessing while computing the confusion
            matrix. Useful for large amounts of data

    Returns:
        float: The f1 score
    '''
    if y_true.shape != y_pred.shape:
        raise errors['F1ScoreError'](
            f'y_true and y_pred must have the same shapes, '
            f'{y_true.shape} != {y_pred.shape}'
        )

    if average not in ['binary', 'micro', 'macro']:
        raise errors['F1ScoreError'](
            'average must be one of [\'binary\', \'micro\', \'macro\']'
        )

    if average == 'binary':
        if len(np.unique(y_true)) != 2:
            raise errors['F1ScoreError'](
                f'y_true has been detected to contain multiple labels, but average=\'binary\'. '
                f'Set the scoring setting from one of [\'micro\', \'macro\']'
            )
        if len(np.unique(y_pred)) != 2:
            raise errors['F1ScoreError'](
                f'y_pred has been detected to contain multiple labels, but average=\'binary\'. '
                f'Set the scoring setting from one of [\'micro\', \'macro\']'
            )

    precision = precision_score(y_true, y_pred, average=average, use_multiprocessing=use_multiprocessing)
    recall = recall_score(y_true, y_pred, average=average, use_multiprocessing=use_multiprocessing)

    return _f1(precision, recall)
