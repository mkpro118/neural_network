'''
Provides an API for selecting the best models using cross validators

Supported Cross Validators are
    KFold
    RepeatedKFold
    StratifiedKFold
    StratifiedRepeatedKFold

    train_test_split(X: Union[list, tuple, np.ndarray],
                     y: Union[list, tuple, np.ndarray]) -> np.ndarray

If needed, a name to symbol map is provided, accessible by
`model_selection.__name_to_symbol_map__`
'''

from .kfold import (
    KFold,
    __name_to_symbol_map__ as kf_sym_map,
)

from .repeated_kfold import (
    RepeatedKFold,
    __name_to_symbol_map__ as rkf_sym_map,
)

from .stratified_kfold import (
    StratifiedKFold,
    __name_to_symbol_map__ as skf_sym_map,
)

from .stratified_repeated_kfold import (
    StratifiedRepeatedKFold,
    __name_to_symbol_map__ as srkf_sym_map,
)

from .train_test_split import (
    train_test_split,
    __name_to_symbol_map__ as tts_sym_map,
)

__name_to_symbol_map__ = {
    **kf_sym_map,
    **rkf_sym_map,
    **skf_sym_map,
    **srkf_sym_map,
    **tts_sym_map,
}

del kf_sym_map
del rkf_sym_map
del skf_sym_map
del srkf_sym_map
del tts_sym_map
