from typing import Union
from numbers import Integral
import numpy as np
import sys

from ..base.cost_mixin import CostMixin
from ..base.classifier_mixin import ClassifierMixin
from ..base.layer import Layer
from ..base.model import Model
from ..exceptions import ExceptionFactory
from ..layers import __name_to_symbol_map__
from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export
from ..utils.timeit import timeit
from ..utils.print_utils import ProgressBar

errors = {
    'SequentialModelError': ExceptionFactory.register('SequentialModelError'),
}

known_layers = {
    **__name_to_symbol_map__,
}


@export
class Sequential(Model, ClassifierMixin):

    STEP_PREFIX = '  Step'

    @type_safe
    def __init__(self, *, layers: Union[list, tuple] = None, from_model: 'Sequential' = None,
                 num_checkpoints: int = 5, name: str = None):
        super().__init__(name=name, num_checkpoints=num_checkpoints)

        self.layers = []

        if from_model is not None:
            if not isinstance(from_model, Sequential):
                raise errors['SequentialModelError'](
                    f'from_model argument must be a sequential model, found {from_model} '
                    f'of type {type(from_model)}'
                )

            self.layers.extend(from_model.layers)

        if layers is not None:
            self.layers.extend(layers)

    @type_safe
    @not_none
    def add(self, *layers: tuple):
        if not layers:
            raise errors['SequentialModelError'](
                f'Found no layers to add'
            )

        for layer in layers:
            if not isinstance(layer, Layer):
                raise errors['SequentialModelError'](
                    f'{layer} of type {type(layer)} is not a descendant of neural_network.base.Layer. '
                    f'If {layer} is a custom layer, esnure that it is a subclass of neural_network.base.Layer.'
                )

            self.layers.append(layer)

    @type_safe
    @not_none
    def compile(self, cost: Union[str, CostMixin],
                metrics: Union[list, tuple]) -> 'Sequential':
        if not self.layers:
            raise errors['SequentialModelError'](
                'There must be at least one layer before the model can compile'
            )

        super().compile(cost=cost, metrics=metrics)

        for layer in self.layers:
            if hasattr(layer, 'input_shape'):
                next_dim = layer.input_shape

                break
        else:
            raise errors['SequentialModelError'](
                f'The first layer must have a defined input shape. Use the '
                f'keyword argument input_shape on the first layer to define an input shape'
            )

        for _id, layer in enumerate(self.layers, start=1):
            next_dim = layer.build(_id, next_dim)

        for layer in reversed(self.layers):
            if layer.activation is None:
                continue

            self.final_activation = layer.activation
            break

        self._check_final_layer_compatibility()

        self.num_layers = len(self.layers)

        self.built = True

    def _check_final_layer_compatibility(self):
        predicate = any((
            all((
                self.final_activation.name == 'softmax',
                self.cost.name != 'crossentropy'
            )),
            all((
                self.final_activation.name != 'softmax',
                self.cost.name == 'crossentropy'
            )),
        ))
        if predicate:
            raise errors['SequentialModelError'](
                f'Final layer activation and cost function are not supported. '
                f'If the final layer uses \'softmax\', use the \'crossentropy\' cost. '
                f'For any other final layer activation use \'mse\''
            )

    @type_safe
    @not_none
    def summary(self, return_: bool = False):
        trainable_params = sum((layer.trainable_params for layer in self.layers))
        non_trainable_params = sum((layer.non_trainable_params for layer in self.layers))

        s = (
            f'Sequential Model: \'{self.name}\' with {self.num_layers} layers\n'
            f'{self}\n'
            f'Total Trainable Params = {trainable_params:,}\n'
            f'Total Non-Trainable Params = {non_trainable_params:,}'
        )

        if return_:
            return s

        print(s)

    @type_safe
    @not_none
    def fit(self,
            X: np.ndarray = None,
            y: np.ndarray = None, *,
            epochs: Union[int, Integral, np.integer] = None,
            batch_size: Union[int, Integral, np.integer] = None,
            steps_per_epoch: Union[int, Integral, np.integer] = None,
            shuffle: bool = True,
            validation_data: Union[np.ndarray, list, tuple] = None,
            verbose: bool = True,
            get_trainer: bool = False):

        super().fit(verbose=verbose)

        if y.ndim not in (1, 2,):
            raise errors['SequentialModelError'](
                f'Training labels must be a 1 or 2 dimensional array'
            )

        if y.ndim == 1:
            self.model_type = 'regressor'
        else:
            self.model_type = 'classifier'

        if verbose:
            self._dynamic_display = any((
                all((
                    hasattr(sys.stdout, 'isatty'),
                    sys.stdout.isatty()
                )),
                'ipykernel' in sys.modules,
            ))

        trainer = self._train(X, y, validation_data, epochs, batch_size, steps_per_epoch, shuffle)

        if get_trainer:
            return trainer

        for _ in trainer:
            pass

    @timeit.register('train', 'trainer')
    def _train(self, X, y, validation_data, epochs, batch_size, steps_per_epoch, shuffle):
        self.is_training = True

        if not epochs:
            epochs = 10

        if batch_size and steps_per_epoch:
            batch_size = len(X) // steps_per_epoch + 1
        elif batch_size and not steps_per_epoch:
            steps_per_epoch = len(X) // batch_size
        elif steps_per_epoch and not batch_size:
            batch_size = len(X) // steps_per_epoch + 1
        else:
            steps_per_epoch = 5
            batch_size = len(X) // steps_per_epoch + 1

        for epoch in range(epochs):
            if self.verbose:
                print(f'\nEpoch {epoch + 1: >{len(str(epochs))}}/{epochs}')

            self._run_epoch(X, y, batch_size, steps_per_epoch, shuffle)

            if self.verbose:
                time = timeit.get_recent_execution_times('run_epoch')

                if time > 60:
                    mins, secs = divmod(time, 60)
                    print(f'Time taken: {int(mins)}m {secs:05.2f}s', end=' ')
                elif time > 1:
                    print(f'Time taken: {time:5.3f}s', end=' ')
                else:
                    ms = int(time * 1000)
                    print(f'Time taken: {ms:3d}ms', end=' ')

                time = timeit.get_recent_execution_times('run_batch', steps_per_epoch)

                time = sum(time) / steps_per_epoch

                if time > 60:
                    mins, secs = divmod(time, 60)
                    print(f'[avg {int(mins)}m:{secs:05.2f}s/step]')
                elif time > 1:
                    print(f'[avg {time:5.3f}s/step]')
                else:
                    ms = time * 1000
                    print(f'[avg {ms:3.2f}ms/step]')

            if validation_data:
                targets = validation_data[1]

                predictions = self.predict(validation_data[0])

                error = self.cost.apply(targets, predictions)

                self.history['validation']['loss'].append(error)

                error = np.around(error, 4)

                if self.model_type == 'classifier':
                    predictions = (predictions == predictions.max(axis=1)[:, None]).astype(int)
                elif self.model_type == 'regressor':
                    predictions = predictions.flatten()

                metrics = {}

                for metric in self.metrics:
                    acc = metric(targets, predictions)
                    self.history['validation'][f'{metric.__name__ if metric.__name__ not in ("accuracy_score", "r2_score") else "accuracy"}'].append(acc)
                    metrics[f'{metric.__name__}'] = np.around(acc, 4)

                if self.verbose:
                    print(f'  Validation loss: {error}', end=' ')
                    print(
                        ' | '.join(
                            map(
                                lambda x: f'{"accuracy" if x[0] in ["accuracy_score", "r2_score"] else x[0]}: {x[1]}',
                                metrics.items()
                            )
                        )
                    )

            targets = y

            predictions = self.predict(X)

            error = self.cost.apply(targets, predictions)

            self.history['overall']['loss'].append(error)

            error = np.around(error, 4)

            if self.model_type == 'classifier':
                predictions = (predictions == predictions.max(axis=1)[:, None]).astype(int)
            elif self.model_type == 'regressor':
                predictions = predictions.flatten()

            metrics = {}

            for metric in self.metrics:
                acc = metric(targets, predictions)
                self.history['overall'][f'{metric.__name__ if metric.__name__ not in ("accuracy_score", "r2_score") else "accuracy"}'].append(acc)
                metrics[f'{metric.__name__}'] = np.around(acc, 4)

            if self.verbose:
                print(f'  Overall loss: {error}', end=' ')
                print(
                    ' | '.join(
                        map(
                            lambda x: f'{"accuracy" if x[0] in ["accuracy_score", "r2_score"] else x[0]}: {x[1]}',
                            metrics.items()
                        )
                    )
                )

            acc = self.history['overall']['accuracy'][-1]

            if not self.checkpoints:
                self.checkpoints.append((
                    Sequential.build_from_config(self.get_metadata()),
                    acc,
                    error,
                ))
                self.best_accuracy = acc
                self.best_loss = error
            elif any((acc > b_acc for _, b_acc, _ in self.checkpoints)):
                sort_spec = ((1, True,), (2, False))
                self.checkpoints.append((
                    Sequential.build_from_config(self.get_metadata()),
                    acc,
                    error,
                ))
                for idx, rev in sort_spec[::-1]:
                    self.checkpoints.sort(key=lambda x: x[idx], reverse=rev)
                self.checkpoints = self.checkpoints[:self.num_checkpoints]
                self.best_accuracy = self.checkpoints[0][1]
                self.best_loss = sorted(self.checkpoints, key=lambda x: x[2])[0][2]

            _data = {
                'overall': {
                    'loss': self.history['overall']['loss'][-1],
                    'accuracy': self.history['overall']['accuracy'][-1],
                },
            }

            if validation_data:
                _data.update({
                    'validation': {
                        'loss': self.history['validation']['loss'][-1],
                        'accuracy': self.history['validation']['accuracy'][-1],
                    },
                })

            yield _data
        self.is_training = False

    @timeit.register('run_epoch', 'epoch runner')
    def _run_epoch(self, X, y, batch_size, steps_per_epoch, shuffle):
        batches = self._get_batches(X, y, batch_size, shuffle)

        with ProgressBar(steps_per_epoch, prefix=Sequential.STEP_PREFIX) as pb:
            for step, (X, y) in zip(range(steps_per_epoch), batches):
                if self.verbose:
                    pb.update()
                self._run_batch(X, y)

    @timeit.register('run_batch', 'batch runner')
    def _run_batch(self, X, y):
        predictions = self.predict(X, training=True)

        _predictions = predictions

        if self.model_type == 'regressor':
            _predictions = predictions.reshape(y.shape)

        error_gradient = np.clip(self.cost.derivative(y, _predictions) * self.final_activation.derivative(_predictions), -1e6, 1e6)

        error_gradient = error_gradient.reshape(predictions.shape)

        for idx, layer in enumerate(reversed(self.layers)):
            error_gradient = np.clip(layer.backward(error_gradient), -1e6, 1e6)

    def predict(self, X: np.ndarray, *, classify: bool = False, training: bool = False):
        for idx, layer in enumerate(self.layers):
            X = layer.forward(X, is_training=self.is_training)

        if classify:
            return (X == X.max(axis=1)[:, None]).astype(int)

        return X

    def get_metadata(self):
        allowed_keys = {
            'cost',
            'metrics',
            'trainable',
            'verbose',
        }

        data = super().get_metadata()

        data.update({
            'metrics': data['metrics_names'],
            'cost': data['cost_name'],
        })

        data = {k: v for k, v in data.items() if k in allowed_keys}

        data['layers'] = {}
        for layer in self.layers:
            data['layers'].update({f'layer{layer._id}': layer.get_metadata()})
            data['layers'][f'layer{layer._id}'].update({'type': layer.__class__.__name__})

        return data

    @classmethod
    def build_from_config(cls, config):
        allowed_keys = {
            'cost',
            'metrics',
            'layers',
            'trainable',
            'verbose',
        }

        required_keys = {
            'cost',
            'layers',
            'metrics',
            'trainable',
        }

        for rkey in required_keys:
            if rkey not in config:
                raise errors['SequentialModelError'](
                    f'config does not have required key \'{rkey}\''
                )

        data = {k: v for k, v in config.items() if k in allowed_keys}

        model = cls()

        layers = {
            k: data['layers'][k] for k in sorted(
                data['layers'].keys(),
                key=lambda x: int(x.replace('layer', ''))
            )
        }

        for idx, layer in enumerate(layers.values(), start=1):
            if layer is not layers[f'layer{idx}']:
                raise errors['SequentialModelError'](
                    f'config is invalid, layer id\'s do not match'
                )

            if 'type' not in layer:
                raise errors['SequentialModelError'](
                    f'config is invalid, layer types are missing'
                )

            layer_type = known_layers[layer['type']]
            required_layer_keys = layer_type._attrs
            for rkey in required_layer_keys:
                if rkey not in layer:
                    raise errors['SequentialModelError'](
                        f'config\'s  does not have required key \'{rkey}\''
                    )
            pos_params = tuple((layer[param] for param in layer_type.pos_params))
            kw_params = {param: layer[param] for param in layer_type.kw_params}

            model.add(layer_type(*pos_params, **kw_params))

        model.compile(data['cost'], data['metrics'])
        for layer in model.layers:
            if layer.trainable:
                layer.weights = np.array(layers[f'layer{layer._id}']['weights'])
                layer.bias = np.array(layers[f'layer{layer._id}']['bias'])

        setattr(model, 'trainable', data['trainable'])
        setattr(model, 'verbose', data['verbose'])

        return model

    def __str__(self):
        if not self.layers:
            return f'Uncompiled model \'{self.name}\''
        s = f'Input Shape: {tuple(self.layers[0].input_shape)}\n'

        f = lambda x: (
            max(map(lambda y: len(y[0]), x)),
            max(map(lambda y: len(y[1]), x)),
            max(map(lambda y: len(y[2]), x)),
        )
        x = tuple(map(lambda x: (f'{x._id}', f'{x.__class__.__name__}', f'{x}'), self.layers))
        l0, l1, l2 = f(x)

        def _l1(x):
            _ = (l1 - x)
            __ = _ // 2
            return __, __ + (_ & 1)

        def _l2(x):
            _ = (l2 - x)
            __ = _ // 2
            return __, __ + (_ & 1)

        f_ = lambda x: (
            f'| {" " * (l0 - len(x[0]))}{x[0]} '
            f'| {" " * _l1(len(x[1]))[0]}{x[1]}{" " * _l1(len(x[1]))[1]} '
            f'| {" " * _l2(len(x[2]))[0]}{x[2]}{" " * _l2(len(x[2]))[1]} |'
        )
        l3 = max(map(len, map(f_, x)))
        s += f"{'-' * l3}\n"
        s += (
            f'| {" " * (l0 - 1)}# '
            f'| {" " * _l1(5)[0]}Layer{" " * _l1(5)[1]} '
            f'| {" " * (_l2(4)[0])}Info{" " * (_l2(4)[1])} |\n'
        )
        s += f"{'-' * l3}\n"
        s += '\n'.join(map(f_, x))
        s += f"\n{'-' * l3}"
        return s
