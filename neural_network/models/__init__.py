'''
Provides an API for building Neural Network Models

Supported models are
    Sequential

If needed, a name to symbol map is provided, accessible by
`model_selection.__name_to_symbol_map__`
'''

from .sequential import (
    Sequential,
    __name_to_symbol_map__ as seq_sym_map,
)

__name_to_symbol_map__ = {
    **seq_sym_map,
}
