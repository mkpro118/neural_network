'''
Provides classes for computing cost functions

In order to use a custom cost functions, create a subclass
inheriting from neural_network.base.cost_mixin.CostMixin

Inheritance allows the API to determine if an instance is an
Cost Function

Supported costs are
    CrossEntropy
    MeanSquaredError

We allow users to specify the cost functions by name in our layers
API, however, if needed, a name to symbol map is provided, accessible by
`cost.__name_to_symbol_map__`
'''

# Imports to have classes in the module namespace
from .cross_entropy import (
    CrossEntropy,
    __name_to_symbol_map__ as cross_entropy_symbol_map,
)

from .mean_squared_error import (
    MeanSquaredError,
    __name_to_symbol_map__ as mse_symbol_map,
)

__name_to_symbol_map__ = {
    **cross_entropy_symbol_map,
    **mse_symbol_map,
}

del cross_entropy_symbol_map
del mse_symbol_map
