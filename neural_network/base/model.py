from typing import Union
import numpy as np
from collections import defaultdict

from .activation_mixin import ActivationMixin
from .mixin import mixin
from .metadata_mixin import MetadataMixin
from .save_mixin import SaveMixin

from ..exceptions import ExceptionFactory
from ..utils.typesafety import type_safe, not_none

from ..metrics import (
    accuracy_score,
    accuracy_by_label,
    f1_score,
    precision_score,
    r2_score,
    recall_score,
)

from ..cost import CrossEntropy, MeanSquaredError

errors = {
    'UncompiledModelError': ExceptionFactory.register('UncompiledModelError'),
    'UnknownCostError': ExceptionFactory.register('UnknownCostError'),
    'UnknownMetricError': ExceptionFactory.register('UnknownMetricError'),
    'NotImplementedError': ExceptionFactory.register('NotImplementedError'),
}

known_metrics = {
    'accuracy': (accuracy_score, r2_score),
    'accuracy_score': accuracy_score,
    'accuracy_by_label': accuracy_by_label,
    'f1_score': f1_score,
    'precision_score': precision_score,
    'recall_score': recall_score,
    'r2_score': r2_score,
}

known_metrics_inv = {v: k for k, v in known_metrics.items()}

known_costs = {
    'crossentropy': CrossEntropy,
    'mse': MeanSquaredError,
}


@mixin  # Prevents instantiation
class Model(MetadataMixin, SaveMixin):
    '''
    Mixin for easier definition of Model classes

    Inherited from MetadataMixin
        `get_metadata`

    Inherited from SaveMixin
        `save`
    '''

    def __init__(self, *, name: str = None, num_checkpoints: int = 5):
        self.name = name or f'{self.__class__.__name__} Model'
        self.history = {
            'overall': defaultdict(list),
            'validation': defaultdict(list),
        }
        self.num_checkpoints = num_checkpoints
        self.checkpoints = []
        self.best_accuracy = 0.
        self.best_loss = 0.
        self._trainable = True
        self._attrs = ('cost', 'metrics')

    @type_safe
    @not_none(nullable=('metrics',))
    def compile(self, cost: Union[str, ActivationMixin], metrics: Union[list, tuple]):
        if isinstance(cost, str):
            cost = known_costs.get(cost, None)
            if cost is None:
                raise errors['UnknownCostError'](
                    f'cost={cost} is not a recognized cost function. Known cost functions are '
                    f'{", ".join(known_costs.keys())}. Alternatively custom cost functions can '
                    f'be defined by subclassing neural_network.base.cost_mixin.CostMixin'
                )

        self.cost = cost
        self.cost_name = getattr(cost, 'name', 'unrecognized cost')

        self.metrics = set()
        self.metrics_names = []

        if not metrics:
            if cost is CrossEntropy or isinstance(cost, CrossEntropy):
                metrics = ['accuracy_score']
            else:
                metrics = ['r2_score']
        elif 'accuracy' not in metrics and accuracy_score not in metrics and r2_score not in metrics:
            if cost is CrossEntropy or isinstance(cost, CrossEntropy):
                metrics.append('accuracy_score')
            else:
                metrics.append('r2_score')

        for metric in metrics:
            if isinstance(metric, str):
                self.metrics_names.append(metric)
                if metric == 'accuracy':
                    if cost is CrossEntropy or isinstance(cost, CrossEntropy):
                        metric = known_metrics.get('accuracy_score', None)
                    else:
                        metric = known_metrics.get('r2_score', None)
                else:
                    metric = known_metrics.get(metric, None)

                if metric is None:
                    self.metrics_names.pop()
                    raise errors['UnknownMetricError'](
                        f'metric={metric} is not a recognized metric function. Known metrics are '
                        f'{", ".join(known_metrics.keys())}'
                    )
            self.metrics.add(metric)

    @type_safe
    @not_none
    def _check_compiled(self):
        if not all((hasattr(self, attr) for attr in self._attrs)):
            raise errors['UncompiledModelError'](
                f'Cannot fit a model before compiling it. Use model.compile(cost=cost, metrics=metrics) '
                f'to compile the model'
            )

    @type_safe
    @not_none
    def fit(self, *, verbose: bool = True):
        self._check_compiled()
        self.verbose = verbose

    @type_safe
    @not_none
    def _get_batches(self, X: np.ndarray, y: np.ndarray,
                     batch_size: int, shuffle: bool):
        n_samples = len(X)
        indices = np.arange(n_samples, dtype=int)
        if shuffle:
            np.random.default_rng().shuffle(indices)
        for i in range(0, n_samples, batch_size):
            idxs = indices[i: min(i + batch_size, n_samples)]
            yield X[idxs], y[idxs]

    def _train(self, X: np.ndarray, y: np.ndarray):
        raise errors['NotImplementedError'](
            f'Descendant classes must define their implementation of the train method'
        )

    def predict(self, *args, **kwargs):
        raise errors['NotImplementedError'](
            f'Descendant classes must define their implementation of the predict method'
        )

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        self._trainable = value
        for layer in self.layers:
            layer.trainable = self._trainable
