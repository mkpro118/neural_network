import math
import dataclasses
import numpy as np
from typing import Optional, Any

from ..base.metadata_mixin import MetadataMixin
from ..base.save_mixin import SaveMixin
from ..base.transform_mixin import TransformMixin
from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export


@export
class DecisionTreeClassifier(MetadataMixin, SaveMixin, TransformMixin):
    entropy_modes = {
        'shannon': 2,
        'natural': math.e,
        'hartley': 10,
    }

    @dataclasses.dataclass
    class Node:
        feature: Any = None
        threshold: Any = None
        left: Optional['DecisionTreeClassifier.Node'] = None
        right: Optional['DecisionTreeClassifier.Node'] = None
        _: dataclasses.KW_ONLY
        label: Any = None
        values_and_counts: Optional[tuple] = None
        reason: Optional[str] = None

        def _is_leaf(self):
            return (self.label is not None)

        def _max_depth(self):
            if self._is_leaf():
                return 0

            left_depth = self.left._max_depth() if self.left else 0
            right_depth = self.right._max_depth() if self.right else 0

            return max(left_depth, right_depth) + 1

        def __str__(self):
            if self._is_leaf():
                return (
                    f'Node('
                    f'feature={self.feature}, '
                    f'threshold={self.threshold}, '
                    f'left={self.left}, '
                    f'right={self.right}, '
                )
            return (
                f'Node('
                f'label={self.label}, '
                f'values_and_counts={self.values_and_counts}, '
                f'reason={self.reason})'
            )

    @type_safe
    @not_none(nullable=('n_features',))
    def __init__(self, min_split: int = 2,
                 max_depth: int = 64, n_features: Optional[int] = None):
        self.min_split = min_split
        self.max_depth = max_depth
        self.n_features = n_features

    @type_safe
    @not_none(nullable=('feature_idxs',))
    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_idxs: Optional[np.ndarray] = None) -> 'DecisionTreeClassifier':
        assert X.ndim == 2, 'X must be of shape (n_samples, n_features)'

        y = np.ravel(y)
        assert y.shape[0] == X.shape[0], 'y must be of shape (n_samples,)'

        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)

        self.labels = np.unique(y)
        self.n_features = min(
            X.shape[-1], self.n_features) if self.n_features else X.shape[-1]

        self.rng = np.random.default_rng()

        self._info_gain_per_split: list = []
        self._build_tree(X, y)
        return self

    @type_safe
    @not_none
    def _build_tree(self, X: np.ndarray, y: np.ndarray):
        self.root = self.__build_tree(X, y)

    @type_safe
    @not_none
    def __build_tree(self, X: np.ndarray, y: np.ndarray, *, depth: int = 0):
        n_samples, n_features = X.shape
        n_labels = np.unique(y).shape[0]

        if n_labels == 1:
            return DecisionTreeClassifier.Node(
                label=y[0],
                values_and_counts=np.unique(y, return_counts=True),
                reason='n_labels = 1'
            )

        if n_features == 0:
            values, counts = np.unique(y, return_counts=True)
            return DecisionTreeClassifier.Node(
                label=values[np.argmax(counts)],
                values_and_counts=(values, counts),
                reason='n_features = 0'
            )

        if depth >= self.max_depth:
            values, counts = np.unique(y, return_counts=True)
            return DecisionTreeClassifier.Node(
                label=values[np.argmax(counts)],
                values_and_counts=(values, counts),
                reason='max_depth'
            )

        if n_samples < self.min_split:
            values, counts = np.unique(y, return_counts=True)
            return DecisionTreeClassifier.Node(
                label=values[np.argmax(counts)],
                values_and_counts=(values, counts),
                reason=f'n_samples < self.min_split ({n_samples} < {self.min_split})'
            )

        feature_idxs = np.asarray(self.rng.choice(
            n_features, self.n_features, replace=False)).astype(int)

        best_feature, best_threshold, info_gain = self._best_split(X, y,
                                                                   feature_idxs)

        self._info_gain_per_split.append(info_gain)

        if best_feature is None or best_threshold is None:
            values, counts = np.unique(y, return_counts=True)
            return DecisionTreeClassifier.Node(
                label=values[np.argmax(counts)],
                values_and_counts=(values, counts),
                reason='no optimal split'
            )

        left_idxs, right_idxs = DecisionTreeClassifier._split(
            X[:, best_feature], best_threshold)

        if np.all(left_idxs) or np.all(right_idxs):
            values, counts = np.unique(y, return_counts=True)
            return DecisionTreeClassifier.Node(
                label=values[np.argmax(counts)],
                values_and_counts=(values, counts),
                reason='further splits are meaningless'
            )

        left = self.__build_tree(
            X[left_idxs, :], y[left_idxs], depth=depth + 1)
        right = self.__build_tree(
            X[right_idxs, :], y[right_idxs], depth=depth + 1)

        return DecisionTreeClassifier.Node(best_feature, best_threshold, left, right)

    @type_safe
    @not_none
    def _best_split(self, X: np.ndarray, y: np.ndarray,
                    feature_idxs: np.ndarray) -> tuple[Optional[int], Optional[Any], float]:
        best_info_gain = -1.
        split_idx, split_threshold = None, None

        for feature_idx in feature_idxs:
            X_col = X[:, feature_idx]
            thresholds = np.unique(X_col)

            for threshold in thresholds:
                info_gain = DecisionTreeClassifier.info_gain(
                    X_col, y, threshold)

                if info_gain <= best_info_gain:
                    continue

                best_info_gain = info_gain
                split_idx = feature_idx
                split_threshold = threshold

        return split_idx, split_threshold, best_info_gain

    @type_safe
    @not_none
    def predict(self, X: np.ndarray) -> Any:
        return np.asarray(
            [DecisionTreeClassifier._predict_traverse(self.root, x) for x in X]
        )

    @type_safe
    @not_none
    def summary(self, show_counts: bool = False,
                show_reason: bool = False, indent: int = 2) -> str:
        return DecisionTreeClassifier._summary_traverse(
            self.root,
            show_counts=show_counts,
            show_reason=show_reason,
            indent=indent
        )

    def get_max_depth(self):
        return self.root._max_depth()

    @type_safe
    @not_none
    @staticmethod
    def _summary_traverse(node: Node, depth: int = 0,
                          show_counts: bool = False,
                          show_reason: bool = False, indent: int = 2) -> str:
        if node._is_leaf():
            # type: ignore[misc]
            counts = ', '.join(map(str, zip(*node.values_and_counts)))
            return (
                f'{(" " * indent) * depth}return {node.label}' + (
                    f' | reason: {node.reason}' if show_reason else '') + (
                    f' | counts: {counts}\n' if show_counts else '\n')
            )

        summary = f'{(" " * indent) * depth}if x{node.feature} <= {node.threshold}\n'

        assert node.left, str(node)  # For mypy
        summary += DecisionTreeClassifier._summary_traverse(
            node.left, depth + 1, show_counts=show_counts,
            show_reason=show_reason, indent=indent
        )

        summary += f'{(" " * indent) * depth}else\n'

        assert node.right, str(node)  # For mypy
        summary += DecisionTreeClassifier._summary_traverse(
            node.right, depth + 1, show_counts=show_counts,
            show_reason=show_reason, indent=indent
        )

        return summary

    @type_safe
    @not_none
    @staticmethod
    def _predict_traverse(node: Node, x: np.ndarray) -> Any:
        if node._is_leaf():
            return node.label

        if x[node.feature] <= node.threshold:
            assert node.left, 'Invalid DecisionTreeClassifier!'
            return DecisionTreeClassifier._predict_traverse(node.left, x)

        assert node.right, 'Invalid DecisionTreeClassifier!'
        return DecisionTreeClassifier._predict_traverse(node.right, x)

    @type_safe
    @not_none
    @staticmethod
    def _split(X: np.ndarray, threshold: Any) -> tuple[np.ndarray, np.ndarray]:
        return X <= threshold, X > threshold

    @type_safe
    @not_none
    @staticmethod
    def info_gain(feature_col: np.ndarray,
                  labels: np.ndarray, threshold: Any) -> float:
        current_entropy = DecisionTreeClassifier.entropy(labels)

        left_idxs, right_idxs = DecisionTreeClassifier._split(
            feature_col, threshold)

        if any((not len(left_idxs), not len(right_idxs))):
            return 0.

        n = len(labels)

        left = labels[left_idxs]
        right = labels[right_idxs]

        left_entropy = DecisionTreeClassifier.entropy(left)
        right_entropy = DecisionTreeClassifier.entropy(right)

        left_weight = len(left) / n
        right_weight = len(right) / n

        split_entropy = np.sum((left_weight * left_entropy,
                                right_weight * right_entropy))  # type: ignore

        info_gain = current_entropy - split_entropy

        return info_gain

    @type_safe
    @not_none
    @staticmethod
    def entropy(labels: np.ndarray, *,
                base: Optional[float] = None,
                mode: Optional[str] = None) -> float:
        assert labels.ndim == 1, 'labels must be of shape (n_samples,)'

        if base:
            assert not mode, 'only one of base or mode should be provided'
            assert base > 0, 'base should be a positive number'
        elif mode:
            base = DecisionTreeClassifier.entropy_modes.get(mode)
            assert base, (
                f'invalid {mode=}, valid modes are '
                f'{", ".join(DecisionTreeClassifier.entropy_modes.keys())}'
            )
        else:
            base = 2

        probs = np.unique(labels, return_counts=True)[1] / labels.shape[0]
        return -sum((prob * math.log(prob, base) for prob in probs))
