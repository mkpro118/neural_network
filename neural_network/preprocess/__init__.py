'''
Provides an API to preprocess data before training a model

Supported preprocessing functions are
    OneHotEncoder
    Scaler
    Standardizer

If needed, a name to symbol map is provided, accessible by
`preprocess.__name_to_symbol_map__`
'''

from .one_hot_encoder import (
    OneHotEncoder,
    __name_to_symbol_map__ as ohe_sym_map,
)

from .scaler import (
    Scaler,
    __name_to_symbol_map__ as sc_sym_map,
)

from .standardizer import (
    Standardizer,
    __name_to_symbol_map__ as st_sym_map,
)


__name_to_symbol_map__ = {
    **ohe_sym_map,
    **sc_sym_map,
    **st_sym_map,
}
