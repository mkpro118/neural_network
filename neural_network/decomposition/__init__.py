'''
Provides classes for perform decomposition and feature reduction

Supported decomposition classes are
    LinearDiscriminantAnalysis
    PrincipalComponentAnalysis

If needed, a name to symbol map is provided, accessible by
`decomposition.__name_to_symbol_map__`
'''

# Imports to have classes in the module namespace
from .linear_discriminant_analysis import (
    LinearDiscriminantAnalysis,
    __name_to_symbol_map__ as lda_symbol_map,
)

from .principal_component_analysis import (
    PrincipalComponentAnalysis,
    __name_to_symbol_map__ as pca_symbol_map,
)

__name_to_symbol_map__ = {
    **lda_symbol_map,
    **pca_symbol_map,
}

del lda_symbol_map
del pca_symbol_map
