'''
Provides and API for building layers in a Artifical Neural Network

Support layers are
    BatchNormalization
    Convolutional
    Dense (FC)
    Flatten

Note: currently, we do not support stride and padding in convolutions

If needed, a name to symbol map is provided, accessible by
`layers.__name_to_symbol_map__`
'''

# Imports to have classes in the module namespace
from .batch_normalization import (
    BatchNormalization,
    __name_to_symbol_map__ as bn_sym_map,
)

from .convolutional import (
    Convolutional,
    __name_to_symbol_map__ as conv_sym_map,
)

from .dense import (
    Dense,
    __name_to_symbol_map__ as dense_sym_map,
)

from .flatten import (
    Flatten,
    __name_to_symbol_map__ as flatten_sym_map,
)

__name_to_symbol_map__ = {
    **bn_sym_map,
    **conv_sym_map,
    **dense_sym_map,
    **flatten_sym_map,
}

del bn_sym_map
del conv_sym_map
del dense_sym_map
del flatten_sym_map
