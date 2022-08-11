import numpy as np
from ..base.metadata_mixin import MetadataMixin
from ..base.save_mixin import SaveMixin
from ..base.transform_mixin import TransformMixin
from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export


@export
class LinearDiscriminantAnalysis(TransformMixin, MetadataMixin, SaveMixin):
    '''
    Used to reduce feature dimension by performing Linear Discriminant Analysis
    '''

    @type_safe
    def __init__(self, n_components: int = None, *, solver: str = 'svd'):
        '''
        Initalize the LDA instance

        Parameters:
            n_components: int, default = None
                Number of components in the reduced dimension matrix
            solver: str, default = 'svd'
                Solver method to use, can be 'svd' or 'eigen'
        '''
        self.n_components = n_components
        self.solver = solver
        self._solver_fn = self._eigen_solver if self.solver == 'eigen' else self._svd_solver

    @type_safe(skip=('return', ))
    @not_none
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'LinearDiscriminantAnalysis':
        '''
        Fits the LDA instance with the given features and labels

        Parameters:
            X: np.ndarray of shape (n_samples, n_features)
                The feature matrix to use for LDA
            y: np.ndarray of shape (n_samples, 1)
                The label vector to perform LDA with

        Returns:
            self: a fitted LDA instance
        '''
        self.X = X
        self.y = y

        if X.ndim != 2:
            raise ValueError('Parameter X must be 2 dimensional')

        self.n_samples, self.n_features = X.shape
        self.labels = np.unique(y)
        if self.n_components is None:
            self.n_components = min(len(self.labels) - 1, self.n_features)

        # Cannot project into a bigger space
        # n_components must be smaller than the number of features
        if self.n_components >= self.n_features:
            raise ValueError(
                f'Number of components is {self.n_components}, '
                f'but feature dimension is only {self.n_features}'
            )
        self.n_labels = len(self.labels)

        # Calculate overall mean for all features
        self.feature_means = np.mean(self.X, axis=0)

        # Computing Sw and Sb
        # Sw - Within class scatter
        self.Sw = np.zeros((self.n_features,) * 2)
        # Sb - Between class scatter
        self.Sb = np.zeros((self.n_features,) * 2)
        for label in self.labels:
            # filter indexes by labels
            label_filter = self.y == label

            # Find the difference between mean for the label, and overall mean
            mean_diff = np.mean(self.X[label_filter], axis=0) - self.feature_means
            mean_diff = mean_diff.reshape((1, self.n_features))

            self.Sw += np.cov(self.X[label_filter], rowvar=False)
            self.Sb += np.sum(label_filter) * (mean_diff.T @ mean_diff)

        # Compute Sw inverse @ Sb
        self.Sw_inv_Sb = np.linalg.inv(self.Sw) @ self.Sb

        # Solve for linear discriminant
        self._solver_fn()

        # Ignore complex parts if they show up.
        self.linear_discriminants = self.linear_discriminants.real

        self._attrs = (
            'X', 'y', 'n_samples', 'n_features', 'labels',
            'n_labels', 'feature_means', 'Sw', 'Sb', 'Sw_inv_Sb',
            'linear_discriminants',
        )
        if self.solver == 'svd':
            self._attrs += ('U', 'S', 'Vh')
        elif self.solver == 'eigen':
            self._attrs += ('eig_vals', 'eig_vecs')
        return self

    def _svd_solver(self) -> None:
        # Perform singular value decomposition
        self.U, self.S, self.Vh = np.linalg.svd(self.Sw_inv_Sb, full_matrices=False)

        # Linear Discriminants are Vh[0 to n_components - 1]
        self.linear_discriminants = self.Vh[:self.n_components].T

    def _eigen_solver(self) -> None:
        eig_vals, eig_vecs = np.linalg.eig(self.Sw_inv_Sb.T @ self.Sw_inv_Sb)

        sorted_indices = np.argsort(abs(eig_vals))[::-1]
        self.eig_vals = eig_vals[sorted_indices]
        self.eig_vecs = eig_vecs[:, sorted_indices]
        self.linear_discriminants = -self.eig_vecs[:, : self.n_components]

    @type_safe
    @not_none
    def transform(self, X: np.ndarray, y: np.ndarray = None, **kwargs):
        '''
        Reduces the feature dimensions of a given matrix to the
        number of components specified in the LDA instance

        Parameters:
            X: np.ndarray of shape (m, n)
                The matrix to transform
        Returns:
            numpy.ndarray: The transformed matrix
        '''
        self._check_is_fitted()
        _X = np.zeros(shape=(X.shape[0], self.n_components))
        for i in range(_X.shape[0]):
            yi = self.linear_discriminants.T @ X[i: i + 1, :].T
            _X[i] = np.reshape(yi, (self.n_components,))
        return _X
