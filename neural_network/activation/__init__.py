'''
Provides classes for computing activation functions

In order to use a custom activation functions, create a subclass
inheriting from neural_network.base.activation_mixin.ActivationMixin

Inheritance allows the API to determine if an instance is an
Activation Function

Supported activations are
    LeakyRelu     activation='leaky_relu'
    ReLU          activation='relu'
    Sigmoid       activation='sigmoid'
    Softmax       activation='softmax'
    Tanh          activation='tanh'
    NoActivation  activation=None, or no keyword argument

Users can specify the activation functions by name in the layers API,
however, if needed, a name to symbol map is provided, accessible by
`activation.__name_to_symbol_map__`
'''

# Imports to have classes in the module namespace
from .leaky_relu import (
    LeakyReLU,
    __name_to_symbol_map__ as leaky_relu_symbol_map,
)

from .relu import (
    ReLU,
    __name_to_symbol_map__ as relu_symbol_map,
)

from .sigmoid import (
    Sigmoid,
    __name_to_symbol_map__ as sigmoid_symbol_map,
)

from .softmax import (
    Softmax,
    __name_to_symbol_map__ as softmax_symbol_map,
)

from .tanh import (
    Tanh,
    __name_to_symbol_map__ as tanh_symbol_map,
)

from ..base.activation_mixin import ActivationMixin
from ..utils.exports import export

import numpy as np


# No activation function
@export
class NoActivation(ActivationMixin):
    @staticmethod
    def apply(X):
        return X

    @staticmethod
    def derivative(X):
        return np.ones_like(X)


__name_to_symbol_map__ = {
    **leaky_relu_symbol_map,
    **relu_symbol_map,
    **sigmoid_symbol_map,
    **softmax_symbol_map,
    **tanh_symbol_map,
    'NoActivation': NoActivation,
}

# don't need these anymore
del ActivationMixin
del leaky_relu_symbol_map
del relu_symbol_map
del sigmoid_symbol_map
del softmax_symbol_map
del tanh_symbol_map
