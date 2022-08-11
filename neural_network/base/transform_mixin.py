import numpy as np
from typing import Union

from .mixin import mixin
from ..utils.typesafety import type_safe, not_none


@mixin  # Prevents instantiation
class TransformMixin:
    '''
    Mixin for all Transformer Classes.

    Provides a single method to both fit the transformer and apply the
    transformation

    Methods:
        `fit_transform(X: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray`:
            Fits the transformer and applies the transformation
        `_check_is_fitted(self) -> None`:
            Ensures the transformer is fitted
    '''

    @type_safe
    @not_none(nullable=('y', ))
    def fit_transform(self, X: np.ndarray, y: np.ndarray = None, *,
                      return_self: bool = False, **kwargs) -> Union[np.ndarray, tuple]:
        '''
        Fits the transformer with inputs, then applies the transformation

        Parameters:
            X: numpy.ndarray
                The data to fit and transform
            y: numpy.ndarray, default = None
                The data to fit. Optional, is not be used by all transformers,
                exists for compatibility with those that do.

        Returns:
            self: The instance of the transformer, only if `return_self` is True
            numpy.ndarray: The transformed data
        '''
        self.fit(X, y, **kwargs)
        if return_self:
            return self, self.transform(X, y, **kwargs)
        else:
            return self.transform(X, y, **kwargs)

    @type_safe
    @not_none
    def _check_is_fitted(self):
        '''
        Ensure the instance is fitted before transformation
        '''

        if any(((getattr(self, attr, None) is None) for attr in self._attrs)):
            raise ValueError(f'{self.__class__} object is not yet fitted!')
