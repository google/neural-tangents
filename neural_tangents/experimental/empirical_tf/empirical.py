# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Experimental prototype of empirical NTK computation in Tensorflow.

This module is applicable to :class:`tf.Module`, :class:`tf.keras.Model`, or
:obj:`tf.function` functions, subject to some conditions (see docstring of
:obj:`empirical_ntk_fn_tf`).

The kernel function follows the API of :obj:`neural_tangents.empirical_ntk_fn`.
Please read the respective docstring for more details.

.. warning::
  This module currently appears to have long compile times (but OK runtime),
  is prone to triggering XLA errors, and does not distinguish between trainable
  and non-trainable parameters of the model.

For details about the empirical (finite width) NTK computation, please see
"`Fast Finite Width Neural Tangent Kernel <https://arxiv.org/abs/2206.08720>`_".

Example:
  >>> import tensorflow as tf
  >>> from tensorflow.keras import layers
  >>> import neural_tangents as nt
  >>> #
  >>> x_train = tf.random.normal((20, 32, 32, 3))
  >>> x_test = tf.random.normal((5, 32, 32, 3))
  >>> #
  >>> # A CNN.
  >>> f = tf.keras.Sequential()
  >>> f.add(layers.Conv2D(32, (3, 3), activation='relu',
  >>>                     input_shape=x_train.shape[1:]))
  >>> f.add(layers.Conv2D(32, (3, 3), activation='relu'))
  >>> f.add(layers.Conv2D(32, (3, 3)))
  >>> f.add(layers.Flatten())
  >>> f.add(layers.Dense(10))
  >>> #
  >>> f.build((None, *x_train.shape[1:]))
  >>> _, params = nt.experimental.get_apply_fn_and_params(f)
  >>> #
  >>> # Default setting: reducing over logits (default `trace_axes=(-1,)`;
  >>> # pass `vmap_axes=0` because the network is iid along the batch axis, no
  >>> # BatchNorm.
  >>> kernel_fn = nt.experimental.empirical_ntk_fn_tf(f, vmap_axes=0)
  >>> #
  >>> # (5, 20) tf.Tensor test-train NTK
  >>> nngp_test_train = kernel_fn(x_test, x_train, params)
  >>> ntk_test_train = kernel_fn(x_test, x_train, params)
  >>> #
  >>> # Full kernel: not reducing over logits.
  >>> kernel_fn = nt.experimental.empirical_ntk_fn_tf(f, trace_axes=(),
  >>>                                                 vmap_axes=0)
  >>> #
  >>> # (5, 20, 10, 10) tf.Tensor test-train NTK.
  >>> k_test_train = kernel_fn(x_test, x_train, params)
  >>> #
  >>> # An FCN
  >>> f = tf.keras.Sequential()
  >>> f.add(layers.Flatten())
  >>> f.add(layers.Dense(1024, activation='relu'))
  >>> f.add(layers.Dense(1024, activation='relu'))
  >>> f.add(layers.Dense(10))
  >>> #
  >>> f.build((None, *x_train.shape[1:]))
  >>> _, params = nt.experimental.get_apply_fn_and_params(f)
  >>> #
  >>> # Use ntk-vector products since the network has many parameters
  >>> # relative to the cost of forward pass.
  >>> ntk_fn = nt.experimental.empirical_ntk_fn_tf(f, vmap_axes=0,
  >>>                                              implementation=2)
  >>> #
  >>> # (5, 5) tf.Tensor test-test NTK
  >>> ntk_test_test = ntk_fn(x_test, None, params)
  >>> #
  >>> # Compute only NTK diagonal variances:
  >>> ntk_fn = nt.experimental.empirical_ntk_fn_tf(f, diagonal_axes=(0,))
  >>> #
  >>> # (20,) tf.Tensor train-train NTK diagonal
  >>> ntk_train_train_diag = ntk_fn(x_train, None, params)
"""

from typing import Callable, Optional, Union
import warnings

from jax.experimental import jax2tf
from neural_tangents._src.empirical import NtkImplementation, empirical_ntk_fn, DEFAULT_NTK_IMPLEMENTATION, _DEFAULT_NTK_FWD, _DEFAULT_NTK_J_RULES, _DEFAULT_NTK_S_RULES
from neural_tangents._src.utils.typing import Axes, PyTree, VMapAxes
import tensorflow as tf
import tf2jax


def empirical_ntk_fn_tf(
    f: Union[tf.Module, tf.types.experimental.GenericFunction],
    trace_axes: Axes = (-1,),
    diagonal_axes: Axes = (),
    vmap_axes: VMapAxes = None,
    implementation: Union[NtkImplementation, int] = DEFAULT_NTK_IMPLEMENTATION,
    _j_rules: bool = _DEFAULT_NTK_J_RULES,
    _s_rules: bool = _DEFAULT_NTK_S_RULES,
    _fwd: Optional[bool] = _DEFAULT_NTK_FWD,
) -> Callable[..., PyTree]:
  r"""Returns a function to draw a single sample the NTK of a given network `f`.

  This function follows the API of :obj:`neural_tangents.empirical_ntk_fn`, but
  is applicable to Tensorflow :class:`tf.Module`, :class:`tf.keras.Model`, or
  :obj:`tf.function`, via a TF->JAX->TF roundtrip using `tf2jax` and `jax2tf`.
  Docstring below adapted from :obj:`neural_tangents.empirical_ntk_fn`.

  .. warning::
    This function is experimental and risks returning wrong results or
    performing slowly. It is intended to demonstrate the usage of
    :obj:`neural_tangents.empirical_ntk_fn` in Tensorflow, but has not been
    extensively tested. Specifically, it appears to have very long
    compile times (but OK runtime), is prone to triggering XLA errors, and does
    not distinguish between trainable and non-trainable parameters of the model.

  TODO(romann): support division between trainable and non-trainable variables.

  TODO(romann): investigate slow compile times.

  Args:
    f:
      :class:`tf.Module` or :obj:`tf.function` whose NTK we are computing. Must
      satisfy the following:

        - if a :obj:`tf.function`, must have the signature of `f(params, x)`.

        - if a :class:`tf.Module`, must be either a :class:`tf.keras.Model`, or
          be callable.

        - input signature (`f.input_shape` for :class:`tf.Module` or
          :class:`tf.keras.Model`, or `f.input_signature` for `tf.function`)
          must be known.

    trace_axes:
      output axes to trace the output kernel over, i.e. compute only the trace
      of the covariance along the respective pair of axes (one pair for each
      axis in `trace_axes`). This allows to save space and compute if you are
      only interested in the respective trace, but also improve approximation
      accuracy if you know that covariance along these pairs of axes converges
      to a `constant * identity matrix` in the limit of interest (e.g.
      infinite width or infinite `n_samples`). A common use case is the channel
      / feature / logit axis, since activation slices along such axis are i.i.d.
      and the respective covariance along the respective pair of axes indeed
      converges to a constant-diagonal matrix in the infinite width or infinite
      `n_samples` limit.
      Also related to "contracting dimensions" in XLA terms.
      (https://www.tensorflow.org/xla/operation_semantics#dotgeneral)

    diagonal_axes:
      output axes to diagonalize the output kernel over, i.e. compute only the
      diagonal of the covariance along the respective pair of axes (one pair for
      each axis in `diagonal_axes`). This allows to save space and compute, if
      off-diagonal values along these axes are not needed, but also improve
      approximation accuracy if their limiting value is known theoretically,
      e.g. if they vanish in the limit of interest (e.g. infinite
      width or infinite `n_samples`). If you further know that on-diagonal
      values converge to the same constant in your limit of interest, you should
      specify these axes in `trace_axes` instead, to save even more compute and
      gain even more accuracy. A common use case is computing the variance
      (instead of covariance) along certain axes.
      Also related to "batch dimensions" in XLA terms.
      (https://www.tensorflow.org/xla/operation_semantics#dotgeneral)

    vmap_axes:
      A triple of `(in_axes, out_axes, kwargs_axes)`
      passed to `vmap` to evaluate the empirical NTK in parallel ove these axes.
      Precisely, providing this argument implies that `f.call(x, **kwargs)`
      equals to a concatenation along `out_axes` of `f` applied to slices of
      `x` and `**kwargs` along `in_axes` and `kwargs_axes`. In other words, it
      certifies that `f` can be evaluated as a `vmap` with `out_axes=out_axes`
      over `x` (along `in_axes`) and those arguments in `**kwargs` that are
      present in `kwargs_axes.keys()` (along `kwargs_axes.values()`).

      This allows us to evaluate Jacobians much more
      efficiently. If `vmap_axes` is not a triple, it is interpreted as
      `in_axes = out_axes = vmap_axes, kwargs_axes = {}`. For example a very
      common use case is `vmap_axes=0` for a neural network with leading (`0`)
      batch dimension, both for inputs and outputs, and no interactions between
      different elements of the batch (e.g. no BatchNorm, and, in the case of
      `nt.stax`, also no Dropout). However, if there is interaction between
      batch elements or no concept of a batch axis at all, `vmap_axes` must be
      set to `None`, to avoid wrong (and potentially silent) results.

    implementation:
      An :class:`NtkImplementation` value (or an :class:`int` `0`, `1`, `2`, or
      `3`). See the :class:`NtkImplementation` docstring for details.

    _j_rules:
      Internal debugging parameter, applicable only when `implementation` is
      :attr:`~neural_tangents.NtkImplementation.STRUCTURED_DERIVATIVES` (`3`)
      or :attr:`~neural_tangents.NtkImplementation.AUTO` (`0`). Set to `True`
      to allow custom Jacobian rules for intermediary primitive `dy/dw`
      computations for MJJMPs (matrix-Jacobian-Jacobian-matrix products). Set
      to `False` to use JVPs or VJPs, via JAX's :obj:`jax.jacfwd` or
      :obj:`jax.jacrev`. Custom Jacobian rules (`True`) are expected to be not
      worse, and sometimes better than automated alternatives, but in case of a
      suboptimal implementation setting it to `False` could improve performance.

    _s_rules:
      Internal debugging parameter, applicable only when `implementation` is
      :attr:`~neural_tangents.NtkImplementation.STRUCTURED_DERIVATIVES` (`3`) or
      :attr:`~neural_tangents.NtkImplementation.AUTO` (`0`). Set to `True` to
      allow efficient MJJMp rules for structured `dy/dw` primitive Jacobians.
      In practice should be set to `True`, and setting it to `False` can lead
      to dramatic deterioration of performance.

    _fwd:
      Internal debugging parameter, applicable only when `implementation` is
      :attr:`~neural_tangents.NtkImplementation.STRUCTURED_DERIVATIVES` (`3`) or
      :attr:`~neural_tangents.NtkImplementation.AUTO` (`0`). Set to `True` to
      allow :obj:`jax.jvp` in intermediary primitive Jacobian `dy/dw`
      computations, `False` to always use :obj:`jax.vjp`. `None` to decide
      automatically based on input/output sizes. Applicable when
      `_j_rules=False`, or when a primitive does not have a Jacobian rule.
      Should be set to `None` for best performance.

  Returns:
    A function `ntk_fn` that computes the empirical ntk.
  """
  warnings.warn('This function is an early proof-of-concept.')

  kwargs = dict(
      trace_axes=trace_axes,
      diagonal_axes=diagonal_axes,
      vmap_axes=vmap_axes,
      implementation=implementation,
      _j_rules=_j_rules,
      _s_rules=_s_rules,
      _fwd=_fwd,
  )
  if isinstance(f, tf.Module):
    apply_fn, _ = get_apply_fn_and_params(f)

  elif isinstance(f, tf.types.experimental.GenericFunction):
    apply_fn = tf2jax.convert_functional(f, *f.input_signature)

  else:
    raise NotImplementedError(f'Got `f={f}` of unsupported type {type(f)}, '
                              f'please file a bug at '
                              f'https://github.com/google/neural-tangents.')

  ntk_fn = empirical_ntk_fn(apply_fn, **kwargs)
  ntk_fn = jax2tf.convert(ntk_fn)
  ntk_fn = tf.function(ntk_fn, jit_compile=True, autograph=False)
  return ntk_fn


def get_apply_fn_and_params(f: tf.Module):
  """Converts a :class:`tf.Module` into a forward-pass `apply_fn` and `params`.

  Use this function to extract `params` to pass to the Tensorflow empirical NTK
  kernel function.

  .. warning::
    This function does not distinguish between trainable and non-trainable
    parameters of the model.

  Args:
    f:
      a :class:`tf.Module` to convert to a `apply_fn(params, x)` function. Must
      have an `input_shape` attribute set (specifying shape of `x`), and be
      callable or be a :class:`tf.keras.Model`.

  Returns:
    A tuple fo `(apply_fn, params)`, where `params` is a `PyTree[tf.Tensor]`.
  """
  @tf.function
  def forward_tf(x: PyTree) -> PyTree:
    if isinstance(f, tf.keras.Model):
      return f.call(x, training=False)

    if not hasattr(f, '__call__'):
      raise NotImplementedError(f'Got `f={f}` of type {type(f)}, '
                                f'that is not callable. Please file a bug at '
                                f'https://github.com/google/neural-tangents.')

    return f(x)

  if not hasattr(f, 'input_shape'):
    raise NotImplementedError(f'`f={f}` must have `input_shape` set. '
                              f'Please file a bug at '
                              f'https://github.com/google/neural-tangents.')

  apply_fn_, params = tf2jax.convert(forward_tf, tf.TensorSpec(f.input_shape))

  def apply_fn(params: PyTree, x: PyTree) -> PyTree:
    outputs, _ = apply_fn_(params, x)  # Dropping parameters (not updated).
    return outputs

  return apply_fn, params
