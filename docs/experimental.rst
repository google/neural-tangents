:github_url: https://github.com/google/neural-tangents/tree/main/docs/experimental.rst



`nt.experimental` -- prototypes
======================================

.. warning::
    This module contains new highly-experimental prototypes. Please beware that they are not properly tested, not supported, and may suffer from sub-optimal performance. Use at your own risk!

.. automodule:: neural_tangents.experimental
.. currentmodule:: neural_tangents.experimental

Kernel functions
--------------------------------------
Finite-width NTK kernel function *in Tensorflow*. See the `Python <https://github.com/google/neural-tangents/blob/main/examples/experimental/empirical_ntk_tf.py>`_ and `Colab <https://colab.research.google.com/github/google/neural-tangents/blob/main/notebooks/experimental/empirical_ntk_resnet_tf.ipynb>`_ usage examples.

.. autofunction:: empirical_ntk_fn_tf

Helper functions
--------------------------------------
A helper function to convert Tensorflow stateful models into functional-style, stateless `apply_fn(params, x)` forward pass function and extract the respective `params`.

.. autofunction:: get_apply_fn_and_params
