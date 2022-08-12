'''
Provides an API for evaluating performance metrics

Supported metrics are
    accuracy_by_label(y_true: np.ndarray,
                      y_pred: np.ndarray) -> np.ndarray

    accuracy_score(y_true: np.ndarray,
                   y_pred: np.ndarray) -> float

    confusion_matrix(y_true: np.ndarray,
                     y_pred: np.ndarray) -> np.ndarray

    multilabel_confusion_matrix(y_true: np.ndarray,
                                y_pred: np.ndarray) -> np.ndarray

    precision_score(y_true: np.ndarray,
                    y_pred: np.ndarray) -> np.float

    r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float

    recall(y_true: np.ndarray, y_pred: np.ndarray) -> np.float
'''

# Imports to have functions in the module namespace
from .accuracy_by_label import (
    accuracy_by_label,
    __name_to_symbol_map__ as abl_sym_map,
)

from .accuracy_score import (
    accuracy_score,
    __name_to_symbol_map__ as as_sym_map,
)

from .confusion_matrix import (
    confusion_matrix,
    multilabel_confusion_matrix,
    __name_to_symbol_map__ as cm_sym_map,
)

from .f1_score import (
    f1_score,
    __name_to_symbol_map__ as f1_sym_map,
)

from .precision_score import (
    precision_score,
    __name_to_symbol_map__ as ps_sym_map,
)

from .r2_score import (
    r2_score,
    __name_to_symbol_map__ as r2s_sym_map,
)

from .recall_score import (
    recall_score,
    __name_to_symbol_map__ as rs_sym_map,
)


__name_to_symbol_map__ = {
    **abl_sym_map,
    **as_sym_map,
    **cm_sym_map,
    **f1_sym_map,
    **ps_sym_map,
    **r2s_sym_map,
    **rs_sym_map,
}

del abl_sym_map
del as_sym_map
del cm_sym_map
del f1_sym_map
del ps_sym_map
del r2s_sym_map
del rs_sym_map
