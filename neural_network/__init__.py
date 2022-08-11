'''
Provides an API for building Artificial Neural Networks

Modules:
    activation
        API for activation functions and their derivatives

    base
        Contains base classes and mixins used in the API

    cost
        API for loss functions and their derivatives

    decomposition
        API for decomposition/feature reduction

    layers
        API for layers in a neural network model

    metrics
        API for evaluating model performance

    model_selection
        API for model selection using cross validators

    models
        API for building models

    preprocess
        API for preprocessing data

    utils
        API for utility functions

Note for those who would like to inspect the source code
    All public classes and function are marked with an `@export`
    decorator to signify they are the ones that users are supposed
    to use in normal circumstances

This API strictly enforces type hints and safety.
Type hints are provided for all public functions.
Avoid using functions with no type hints,
they are supposed to be internal functions

Note: Python version 3.6.0 or greater is required for this API
Note: Python version 3.8.0 or greater is required for type safety on
      typing.Union subscripts
'''

import sys

if sys.version_info < (3, 6, 0):
    print(
        "neural_network requires python3 version >= 3.6.0",
        file=sys.stderr
    )
    sys.exit(1)
elif sys.version_info < (3, 8, 0):
    import warnings
    x, y, z = map(str, sys.version_info[:3])
    warnings.warn(
        f'typing support is not fully compatible with python version '
        f"{'.'.join((x, y, z))}"
    )
