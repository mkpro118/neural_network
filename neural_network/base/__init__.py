'''
Provides base functionality for the API and allows flexibility for
customization

These classes should be used only if customization is required.
The classes don't belong to the package namespace, users must explicitly
import them using the complete path. All classes and functions below
are listed with their qualified name after `neural_network.base`

If a custom mixin is defined, use the mixin.mixin decorator to designate
it as a mixin. See mixin.mixin docs for it's functionality

Base classes are
    activation_mixin.ActivationMixin
        Desginates a class as an activation function

    classifier_mixin.ClassifierMixin
        Provides a shortcut for evaluting model performance

    cost_mixin.CostMixin
        Desginates a class as an cost function

    layer.Layer
        Base class for all Layers in this API

    metadata_mixin.MetadataMixin
        Provides method `get_metadata` which returns the public
        metadata of any object as a dictionary

        Note: See save_mixin.SaveMixin

    model.Model
        Base class for all Models in this API

    save_mixin.SaveMixin
        Provides method `save` which saves the public metadata of any
        object to a given file as JSON, or optionally, any JSON
        serializable data to any file. It also supports converting
        NumPy values to JSON serializable objects

        The `save` method tries the `get_metadata` to automatically
        compute the metadata to save. It is recommended to inherit from
        both metadata_mixin.MetadataMixin and save_mixin.SaveMixin
        simultaneously to easily save object states

    transform_mixin.TransformMixin
        Provides method `fit_transform` on transformers to fit and
        transform with a single method
'''
