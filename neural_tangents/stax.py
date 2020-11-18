# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License');
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

"""Closed-form NNGP and NTK library.

This library contains layer constructors mimicking those in
`jax.experimental.stax` with similar API apart apart from:

1) Instead of `(init_fn, apply_fn)` tuple, layer constructors return a triple
`(init_fn, apply_fn, kernel_fn)`, where the added `kernel_fn` maps a
`Kernel` to a new `Kernel`, and represents the change in the
analytic NTK and NNGP kernels (`Kernel.nngp`, `Kernel.ntk`). These functions
are chained / stacked together within the `serial` or `parallel`
combinators, similarly to `init_fn` and `apply_fn`.

2) In layers with random weights, NTK parameterization is used by default
(https://arxiv.org/abs/1806.07572, page 3). Standard parameterization
(https://arxiv.org/abs/2001.07301) can be specified for `Conv` and `Dense`
layers by a keyword argument `parameterization`.

3) Some functionality may be missing (e.g. `BatchNorm`), and some may be
present only in our library (e.g. `CIRCULAR` padding, `LayerNorm`,
`GlobalAvgPool`, `GlobalSelfAttention`, flexible batch and channel axes etc.).

Example:
  >>>  from jax import random
  >>>  import neural_tangents as nt
  >>>  from neural_tangents import stax
  >>>
  >>>  key1, key2 = random.split(random.PRNGKey(1), 2)
  >>>  x_train = random.normal(key1, (20, 32, 32, 3))
  >>>  y_train = random.uniform(key1, (20, 10))
  >>>  x_test = random.normal(key2, (5, 32, 32, 3))
  >>>
  >>>  init_fn, apply_fn, kernel_fn = stax.serial(
  >>>      stax.Conv(128, (3, 3)),
  >>>      stax.Relu(),
  >>>      stax.Conv(256, (3, 3)),
  >>>      stax.Relu(),
  >>>      stax.Conv(512, (3, 3)),
  >>>      stax.Flatten(),
  >>>      stax.Dense(10)
  >>>  )
  >>>
  >>>  predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, x_train,
  >>>                                                        y_train)
  >>>
  >>>  # (5, 10) np.ndarray NNGP test prediction
  >>>  y_test_nngp = predict_fn(x_test=x_test, get='nngp')
  >>>
  >>>  # (5, 10) np.ndarray NTK prediction
  >>>  y_test_ntk = predict_fn(x_test=x_test, get='ntk')
"""


import enum
import functools
import operator as op
import string
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union, Sequence, TypeVar, Any
import warnings

import frozendict
import jax
from jax import lax
from jax import numpy as np
from jax import ops
from jax import random
from jax.api import ShapeDtypeStruct, eval_shape, grad, ShapedArray, vmap
import jax.experimental.stax as ostax
from jax.lib import xla_bridge
from jax.scipy.special import erf
from jax.tree_util import tree_map
from neural_tangents.utils import utils, dataclasses
from neural_tangents.utils.kernel import Kernel
from neural_tangents.utils.typing import AnalyticKernelFn, Axes, Get, InitFn, ApplyFn, InternalLayer, Layer, LayerKernelFn, PyTree, NTTree, Kernels
import numpy as onp
import scipy as osp


# Enums


class Padding(enum.Enum):
  """Type of padding in pooling and convolutional layers."""
  CIRCULAR = 'CIRCULAR'
  SAME = 'SAME'
  VALID = 'VALID'


class Pooling(enum.Enum):
  """Type of pooling in pooling layers."""
  AVG = 'AVG'
  SUM = 'SUM'


# Decorators


def layer(layer_fn: Callable[..., InternalLayer]) -> Callable[..., Layer]:
  """A convenience decorator to be added to all public layers like `Relu` etc.

  Makes the `kernel_fn` of the layer work with both input `np.ndarray`
  (when the layer is the first one applied to inputs), and with `Kernel` for
  intermediary layers. Also adds optional arguments to the `kernel_fn` to
  allow specifying the computation and returned results with more flexibility.

  Args:
    layer_fn: Layer function returning triple `(init_fn, apply_fn, kernel_fn)`.

  Returns:
    A function with the same signature as `layer` with `kernel_fn` now
    accepting `np.ndarray` as inputs if needed, and accepts optional `get`,
    `diagonal_batch`, `diagonal_spatial` arguments.
  """
  name = layer_fn.__name__

  @utils.wraps(layer_fn)
  def new_layer_fns(*args, **kwargs):
    init_fn, apply_fn, kernel_fn = layer_fn(*args, **kwargs)
    kernel_fn = _preprocess_kernel_fn(init_fn, apply_fn, kernel_fn)
    init_fn.__name__ = apply_fn.__name__ = kernel_fn.__name__ = name
    return init_fn, apply_fn, kernel_fn

  return new_layer_fns


def _requires(**static_reqs):
  """Returns a decorator that augments `kernel_fn` with consistency checks.

  Use this to specify your `kernel_fn` input kernel requirements.

  See Also:
    `stax.Diagonal`, `stax.Input`, `stax.Output`.

  """

  def req(kernel_fn: LayerKernelFn):
    """Returns `kernel_fn` with additional consistency checks."""

    @utils.wraps(kernel_fn)
    def new_kernel_fn(k: NTTree[Kernel], **kwargs) -> NTTree[Kernel]:
      """Executes `kernel_fn` on `kernels` after checking consistency."""
      fused_reqs = _fuse_requirements(static_reqs, {}, **kwargs)

      # `FanInConcat / FanInSum` have no requirements and
      # execute custom consistency checks.
      if isinstance(k, Kernel):
        for key, v in fused_reqs.items():
          if v is not None:  # `None` is treated as explicitly not having a req.
            if key in ('diagonal_batch', 'diagonal_spatial'):
              if (getattr(k, key) is True and
                  (v is False or
                   (isinstance(v, _Diagonal) and v.input == _Bool.NO))):
                raise ValueError(f'{kernel_fn} requires `{key} == {v}`, but '
                                 f'input kernel has `{key} == True`, hence '
                                 f'does not contain sufficient information. '
                                 f'Please recompute the input kernel with '
                                 f'`{key} == {v}`.')

            elif key in ('batch_axis', 'channel_axis'):
              ndim = len(k.shape1)
              v_kernel = getattr(k, key)
              v_pos = v % ndim
              if v_kernel != v_pos:
                raise ValueError(f'{kernel_fn} requires `{key} == {v_pos}`, '
                                 f'but input kernel has `{key} == {v_kernel}`, '
                                 f'making the infinite limit ill-defined.')

            else:
              # Any other name is recognized as a keyword-argument threaded
              # through all `kernel_fn` down to `_inputs_to_kernel` rather than
              # a requirement for this layer.
              pass

      return kernel_fn(k, **kwargs)

    setattr(new_kernel_fn, _INPUT_REQ, frozendict.frozendict(static_reqs))
    return new_kernel_fn

  return req


def supports_masking(remask_kernel: bool):
  """Returns a decorator that turns layers into layers supporting masking.

  Specifically:
  1) `init_fn` is left unchanged.
  2) `apply_fn` is turned from
    a function that accepts a `mask=None` keyword argument (which indicates
      `inputs[mask]` must be masked), into
    a function that accepts a `mask_constant=None` keyword argument (which
      indicates `inputs[inputs == mask_constant]` must be masked).
  3) `kernel_fn` is modified to
    3.a) propagate the `kernel.mask1` and `kernel.mask2` through intermediary
      layers, and,
    3.b) if `remask_kernel == True`, zeroes-out covariances between entries of
      which at least one is masked.
  4) If the decorated layers has a `mask_fn`, it is used to propagate masks
    forward through the layer, in both `apply_fn` and `kernel_fn`. If not, it is
     assumed the mask remains unchanged.

  Must be applied before the `layer` decorator.

  Args:
    remask_kernel: `True` to zero-out kernel covariance entries between masked
      inputs after applying `kernel_fn`. Some layers don't need this and setting
      `remask_kernel=False` can save compute.

  Returns:
    A decorator that turns functions returning
    `(init_fn, apply_fn, kernel_fn[, mask_fn])`
    into functions returning
    `(init_fn, apply_fn_with_masking, kernel_fn_with_masking)`.
  """
  def supports_masking(layer):

    @utils.wraps(layer)
    def layer_with_masking(*args, **kwargs):
      layer_fns = layer(*args, **kwargs)
      init_fn, apply_fn, kernel_fn = layer_fns[:3]

      if len(layer_fns) == 3:
        # No mask propagation function supplied - use identity.
        _mask_fn = lambda mask, input_shape: mask
      elif len(layer_fns) == 4:
        # Custom mask propagation function supplied.
        _mask_fn = layer_fns[3]
      else:
        raise ValueError(f'Expected 3 (`init_fn`, `apply_fn`, `kernel_fn`) or 4'
                         f' (..., `mask_fn`) layer functions, '
                         f'got {len(layer_fns)}.')

      @utils.wraps(_mask_fn)
      def mask_fn(mask, input_shape):
        if mask is None:
          return None
        return _mask_fn(mask, input_shape)

      def apply_fn_with_masking(params, inputs, *,
                                mask_constant=None, **kwargs):
        masked_inputs = utils.get_masked_array(inputs, mask_constant)
        inputs = utils.nt_tree_fn()(lambda x: x.masked_value)(masked_inputs)
        mask = utils.nt_tree_fn()(lambda x: x.mask)(masked_inputs)
        # inputs, mask = inputs.masked_value, inputs.mask
        outputs = apply_fn(params, inputs, mask=mask, **kwargs)
        outputs_mask = mask_fn(mask,
                               inputs.shape if isinstance(inputs, np.ndarray)
                               else [i.shape for i in inputs])
        if outputs_mask is None:
          return outputs
        return utils.MaskedArray(outputs, outputs_mask)

      def kernel_fn_with_masking(k: NTTree[Kernel], **user_reqs):
        mask1 = utils.nt_tree_fn()(lambda k: k.mask1)(k)
        shape1 = utils.nt_tree_fn()(lambda k: k.shape1)(k)
        mask2 = utils.nt_tree_fn()(lambda k: k.mask2)(k)
        shape2 = utils.nt_tree_fn()(lambda k: k.shape2)(k)

        mask1, mask2 = mask_fn(mask1, shape1), mask_fn(mask2, shape2)

        k = kernel_fn(k, **user_reqs)  # type: Kernel

        if remask_kernel:
          remask_fn = utils.nt_tree_fn()(lambda k, m1, m2: k.mask(m1, m2))
          k = remask_fn(k, mask1, mask2)
        else:
          replace_fn = utils.nt_tree_fn()(
              lambda k, m1, m2: k.replace(mask1=m1, mask2=m2))
          k = replace_fn(k, mask1, mask2)
        return k

      if hasattr(kernel_fn, _INPUT_REQ):
        setattr(kernel_fn_with_masking,
                _INPUT_REQ,
                getattr(kernel_fn, _INPUT_REQ))

      return init_fn, apply_fn_with_masking, kernel_fn_with_masking

    return layer_with_masking

  return supports_masking


# LAYERS


@layer
def serial(*layers: Layer) -> InternalLayer:
  """Combinator for composing layers in serial.

  Based on `jax.experimental.stax.serial`.

  Args:
    *layers:
      a sequence of layers, each an `(init_fn, apply_fn, kernel_fn)` triple.

  Returns:
    A new layer, meaning an `(init_fn, apply_fn, kernel_fn)` triple,
    representing the serial composition of the given sequence of layers.
  """
  init_fns, apply_fns, kernel_fns = zip(*layers)
  init_fn, apply_fn = ostax.serial(*zip(init_fns, apply_fns))

  @_requires(**_get_input_req_attr(kernel_fns, fold=op.rshift))
  def kernel_fn(k: NTTree[Kernel], **kwargs) -> NTTree[Kernel]:
    # TODO(xlc): if we drop `x1_is_x2` and use `rng` instead, need split key
    # inside kernel functions here and parallel below.
    for f in kernel_fns:
      k = f(k, **kwargs)
    return k

  return init_fn, apply_fn, kernel_fn


@layer
def parallel(*layers: Layer) -> InternalLayer:
  """Combinator for composing layers in parallel.

  The layer resulting from this combinator is often used with the `FanOut` and
  `FanInSum`/`FanInConcat` layers. Based on `jax.experimental.stax.parallel`.

  Args:
    *layers:
      a sequence of layers, each with a `(init_fn, apply_fn, kernel_fn)` triple.

  Returns:
    A new layer, meaning an `(init_fn, apply_fn, kernel_fn)` triples,
    representing the parallel composition of the given sequence of layers. In
    particular, the returned layer takes a sequence of inputs and returns a
    sequence of outputs with the same length as the argument `layers`.
  """
  init_fns, apply_fns, kernel_fns = zip(*layers)
  init_fn_stax, apply_fn_stax = ostax.parallel(*zip(init_fns, apply_fns))

  def init_fn(rng, input_shape):
    return type(input_shape)(init_fn_stax(rng, input_shape))

  def apply_fn(params, inputs, **kwargs):
    return type(inputs)(apply_fn_stax(params, inputs, **kwargs))

  @_requires(**_get_input_req_attr(kernel_fns, fold=op.and_))
  def kernel_fn(ks: NTTree[Kernel], **kwargs) -> NTTree[Kernel]:
    return type(ks)(f(k, **kwargs) for k, f in zip(ks, kernel_fns))

  return init_fn, apply_fn, kernel_fn


@layer
@supports_masking(remask_kernel=False)
def DotGeneral(
    *,
    lhs: Union[np.ndarray, float] = None,
    rhs: Union[np.ndarray, float] = None,
    dimension_numbers: lax.DotDimensionNumbers = (((), ()), ((), ())),
    precision: Optional[lax.Precision] = None,
    batch_axis: int = 0,
    channel_axis: int = -1
) -> InternalLayer:
  r"""Layer constructor for a constant (non-trainable) rhs/lhs Dot General.

  Dot General allows to express any linear transformation on the inputs,
  including but not limited to matrix multiplication, pooling, convolutions,
  permutations, striding, masking etc (but specialized implementations are
  typically much more efficient).

  Returned `apply_fn` is calling
  `jax.lax.dot_general(inputs, rhs, dimension_numbers, precision)` or
  `jax.lax.dot_general(lhs, inputs, dimension_numbers, precision)`, depending
  on whether `lhs` or `rhs` is specified (not `None`).

  Example:
    >>>  from jax import random
    >>>  import jax.numpy as np
    >>>  from neural_tangents import stax
    >>>
    >>>  # Two time series stacked along the second (H) dimension.
    >>>  x = random.normal(random.PRNGKey(1), (5, 2, 32, 3))  # NHWC
    >>>
    >>>  # Multiply all outputs by a scalar:
    >>>  nn = stax.serial(
    >>>      stax.Conv(128, (1, 3)),
    >>>      stax.Relu(),
    >>>      stax.DotGeneral(rhs=2.),  # output shape is (5, 2, 30, 128)
    >>>      stax.GlobalAvgPool()      # (5, 128)
    >>>  )
    >>>
    >>>  # Subtract second time series from the first one:
    >>>  nn = stax.serial(
    >>>      stax.Conv(128, (1, 3)),
    >>>      stax.Relu(),
    >>>      stax.DotGeneral(
    >>>          rhs=np.array([1., -1.]),
    >>>          dimension_numbers=(((1,), (0,)), ((), ()))),  # (5, 30, 128)
    >>>      stax.GlobalAvgPool()                              # (5, 128)
    >>>  )
    >>>
    >>>  # Flip outputs with each other
    >>>  nn = stax.serial(
    >>>      stax.Conv(128, (1, 3)),
    >>>      stax.Relu(),
    >>>      stax.DotGeneral(
    >>>          lhs=np.array([[0., 1.], [1., 0.]]),
    >>>          dimension_numbers=(((1,), (1,)), ((), ()))),  # (5, 2, 30, 128)
    >>>      stax.GlobalAvgPool()                              # (5, 128)
    >>>  )

  See Also:
    https://www.tensorflow.org/xla/operation_semantics#dotgeneral

  Args:
    lhs:
      a constant array to dot with. `None` means layer `inputs` are the
      left-hand side.
    rhs:
      a constant array to dot with. `None` means layer `inputs` are the
      right-hand side. If both `lhs` and `rhs` are `None` the layer is the same
      as `Identity`.
    dimension_numbers:
      a tuple of tuples of the form
      `((lhs_contracting_dims, rhs_contracting_dims),
        (lhs_batch_dims, rhs_batch_dims))`.
    precision:
      Optional. Either `None`, which means the default precision for the
      backend, or a `lax.Precision` enum value (`Precision.DEFAULT`,
      `Precision.HIGH` or `Precision.HIGHEST`).
    batch_axis:
      batch axis for `inputs`. Defaults to `0`, the leading axis. Can be present
      in `dimension_numbers`, but contraction along `batch_axis` will not allow
      for further layers to be applied afterwards.
    channel_axis:
      channel axis for `inputs`. Defaults to `-1`, the trailing axis. For
      `kernel_fn`, channel size is considered to be infinite. Cannot be present
      in `dimension_numbers`.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  if rhs is not None and lhs is not None:
    raise ValueError('At most one of constant `rhs` and `lhs` can be non-`None`'
                     ', since the other factor is considered to be the layer '
                     '`inputs`.')
  is_lhs = rhs is None
  other = np.array(lhs if is_lhs else rhs)

  def dot_fn(x):
    args = (x, other.astype(x.dtype))[::(-1 if is_lhs else 1)]
    return lax.dot_general(*args, dimension_numbers, precision)

  def init_fn(rng, input_shape):
    out = eval_shape(dot_fn, ShapeDtypeStruct(input_shape, other.dtype))
    return out.shape, ()

  def apply_fn(params, inputs, **kwargs):
    return dot_fn(inputs)

  # If a dimension is contracted, respective pairwise covariances are needed to
  # compute the covariance of contractions.
  input_cs = dimension_numbers[0][1 if is_lhs else 0]
  diagonal_batch = (batch_axis not in input_cs) or (rhs is None and lhs is None)
  diagonal_spatial = _Diagonal(
      input=_Bool.YES
      if (input_cs in ((), (batch_axis,)) or (rhs is None and lhs is None))
      else _Bool.NO)  # pytype:disable=wrong-keyword-args

  @_requires(diagonal_batch=diagonal_batch,
             diagonal_spatial=diagonal_spatial,
             batch_axis=batch_axis,
             channel_axis=channel_axis)
  def kernel_fn(k: Kernel, **kwargs) -> Kernel:
    return k.dot_general(other, other, is_lhs, dimension_numbers)

  def mask_fn(mask, input_shape):
    mask_shape = list(input_shape)
    mask_shape[channel_axis] = mask.shape[channel_axis]
    return ~dot_fn(~np.broadcast_to(mask, mask_shape))

  return init_fn, apply_fn, kernel_fn, mask_fn


@layer
@supports_masking(remask_kernel=True)
def Aggregate(
    aggregate_axis: Axes = None,
    batch_axis: int = 0,
    channel_axis: int = -1) -> InternalLayer:
  r"""Layer constructor for aggregation operator (graphical neural network).

  See e.g. arXiv: 1905.13192.

  Specifically, each `N+2`-D `input` of shape `(batch, X_1, ..., X_N, channels)`
  (subject to `batch_axis` and `channel_axis`) is accompanied by a [weighted]
  2-adjacency `2K+1`-D tensor `pattern` of shape
  `(batch, X_i1, ..., X_iK, X_i1, ..., X_iK)` (i.e. leading batch dimensions,
  repeated spatial dimensions, no channel dimension) and the output tensor is
  `lax.dot_general(inputs, pattern, ((aggregate_axes, range(1, K + 1)),
                                     (batch_axis,), (0,)))`
  with the `batch_axis` and `channel_axis` preserved. `K = len(aggregate_axes)`.

  Qualitatively, having `pattern[n, i1, ..., iK, j1, ..., jK] == w` represents
  a directed edge from pixel / token `(i1, ..., iK)` to `(j1, ..., jK)` with
  weight `w` in an individual input sample `n`. The `apply_fn` of this
  layer replaces all nodes with the (weighted) sum of (incoming) adjacent nodes
  to the given node.

  Note that individual inputs can have more than `K` dimensions (e.g. channels,
  other coordinates), in which case slices along these coordinates are
  processed in the same way independently.

  Example:
    >>>  # 1D inputs
    >>>  x = random.normal(random.PRNGKey(1), (5, 3, 32))  # NCH
    >>>
    >>>  # 1) NHH binary adjacency matrix
    >>>  A = random.bernoulli(random.PRNGKey(2), 0.5, (5, 32, 32))
    >>>  # `A[n, h1, h2] == True`
    >>>  # means an edge between tokens `h1` and `h2` in sample `n`.
    >>>
    >>>  init_fn, apply_fn, kernel_fn = stax.Aggregate(aggregate_axis=2,
    >>>                                                batch_axis=0,
    >>>                                                channel_axis=1)
    >>>
    >>>  out = apply_fn((), x, pattern=A)
    >>>  # output is the same as `x @ A` of shape (5, 3, 32)
    >>>
    >>>
    >>>  # 2D inputs
    >>>  x = random.normal(random.PRNGKey(1), (5, 3, 32, 16))  # NCHW
    >>>
    >>>  # 2) NHWHW binary adjacency matrix
    >>>  A = random.bernoulli(random.PRNGKey(2), 0.5, (5, 32, 16, 32, 16))
    >>>  # `A[n, h1, w1, h2, w2] == True`
    >>>  # means an edge between pixels `(h1, w1)` and `(h2, w2)` in image `n`.
    >>>
    >>>  init_fn, apply_fn, kernel_fn = stax.Aggregate(aggregate_axis=(2, 3),
    >>>                                                batch_axis=0,
    >>>                                                channel_axis=1)
    >>>
    >>>  out = apply_fn((), x, pattern=A)
    >>>  # output is of shape (5, 3, 32, 16), the same as
    >>>  # `(x.reshape((5, 3, 32 * 16)) @ A.reshape((5, 32 * 16, 32 * 16))
    >>>  #  ).reshape(x.shape)`
    >>>
    >>>
    >>>  # 3) NWW binary adjacency matrix
    >>>  A = random.bernoulli(random.PRNGKey(2), 0.5, (5, 16, 16))
    >>>  # `A[n, w1, w2] == True`
    >>>  # means an edge between rows `w1` and `w2` in image `n`.
    >>>
    >>>  init_fn, apply_fn, kernel_fn = stax.Aggregate(aggregate_axis=(3,),
    >>>                                                batch_axis=0,
    >>>                                                channel_axis=1)
    >>>
    >>>  out = apply_fn((), x, pattern=A)
    >>>  # output is of shape (5, 3, 32, 16), the same as
    >>>  # `(x.reshape((5, 3 * 32, 16)) @ A).reshape(x.shape)`
    >>>
    >>>
    >>>  # 4) Infinite width example
    >>>  x1 = random.normal(random.PRNGKey(1), (5, 3, 32))  # NCH
    >>>  x2 = random.normal(random.PRNGKey(2), (2, 3, 32))  # NCH
    >>>
    >>>  # 1) NHH binary adjacency matrices
    >>>  A1 = random.bernoulli(random.PRNGKey(2), 0.5, (5, 32, 32))
    >>>  A2 = random.bernoulli(random.PRNGKey(2), 0.5, (2, 32, 32))
    >>>
    >>>  _, _, kernel_fn_id = stax.Identity()
    >>>
    >>>  _, _, kernel_fn_agg = stax.Aggregate(aggregate_axis=2,
    >>>                                       batch_axis=0,
    >>>                                       channel_axis=1)
    >>>
    >>>  nngp = kernel_fn_id(x1, x2, get='nngp', channel_axis=1)
    >>>  # initial NNGP of shape (5, 2, 32, 32)
    >>>  K_agg = kernel_fn_agg(x1, x2, get='nngp', pattern=(A1, A2))
    >>>  # output NNGP of same shape (5, 2, 32, 32):
    >>>  # `K_agg[n1, n2] == A1[n1].T @ nngp[n1, n2] @ A2[n2]`

  Args:
    aggregate_axis:
      axes (non-batch and non-channel) to aggregate adjacent vertices over.
    batch_axis:
      batch axis for `inputs`. Defaults to `0`, the leading axis.
    channel_axis:
      channel axis for `inputs`. Defaults to `-1`, the trailing axis. For
      `kernel_fn`, channel size is considered to be infinite.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  init_fn = lambda rng, input_shape: (input_shape, ())

  def get_agg_axes(ndim: int) -> List[int]:
    if aggregate_axis is None:
      _batch_axis, _channel_axis = utils.mod((batch_axis, channel_axis), ndim)
      agg_axes = [i for i in range(ndim)
                  if i not in (_batch_axis, _channel_axis)]
    else:
      agg_axes = utils.canonicalize_axis(aggregate_axis, ndim)
    return agg_axes

  def get_dimension_numbers(ndim: int) -> lax.DotDimensionNumbers:
    agg_axes = get_agg_axes(ndim)
    return (agg_axes, (range(1, len(agg_axes) + 1))), ((batch_axis,), (0,))

  def apply_fn(params,
               inputs: np.ndarray, *,
               pattern: np.ndarray = None,
               **kwargs):
    """Compute the transformed tensors after an aggregation layer.

    Args:
      params:
        Not used.
      inputs:
        An input `N+2`-D tensor of shape `(batch, X_1, ..., X_N, channels)`
        (subject to `batch_axis` and `channel_axis`).
      pattern:
        An `2K+1`-D tensor of shape `(batch, X_i1, ..., X_iK, X_i1, ..., X_iK)`,
        with the batch leading dimension, and no channel dimension, where
        `K = len(aggregate_axes)`.
      **kwargs:
        unused.

    Returns:
      An `N+2`-D tensor of shape of the same shape as `inputs`.
    """
    if pattern is None:
      return inputs

    del params
    ndim = inputs.ndim
    dn = get_dimension_numbers(ndim)
    out = lax.dot_general(inputs, pattern.astype(inputs.dtype), dn)

    out_c_axis = utils.axis_after_dot(channel_axis % ndim, dn[0][0], dn[1][0])
    out_b_axis = utils.axis_after_dot(batch_axis % ndim, dn[0][0], dn[1][0])
    agg_axes = get_agg_axes(ndim)
    out = np.moveaxis(out,
                      [out_b_axis, out_c_axis] + list(range(-len(agg_axes), 0)),
                      [batch_axis, channel_axis] + agg_axes)
    return out

  @_requires(batch_axis=batch_axis,
             channel_axis=channel_axis,
             diagonal_spatial=_Diagonal(input=_Bool.NO, output=_Bool.NO))  # pytype:disable=wrong-keyword-args
  def kernel_fn(k: NTTree[Kernel],
                *,
                pattern: Tuple[Optional[np.ndarray],
                               Optional[np.ndarray]] = (None, None),
                **kwargs):
    """Compute the transformed kernels after an aggregation kernel layer.

      Specifically, the `nngp`/`ntk` is a `2N+2`-D tensor of shape
      `(B_1, B_2, X_1, X_1, ..., X_N, X_N)`. This tensor will
      be aggregated (via matrix multiplication) on the left by `pattern[0]` of
      shape `(B_1, X_i1, ..., X_iK)` and on the right by `pattern[1]` of shape
      `(B_2, X_i1, ..., X_iK)`. Ignoring the batch dimensions, the return
      `nngp/ntk` is `pattern[0].T @ nngp/ntk @ pattern[1]`

    """
    pattern1, pattern2 = pattern
    ndim = len(k.shape1)
    return k.dot_general(
        other1=pattern1,
        other2=pattern2,
        is_lhs=False,
        dimension_numbers=get_dimension_numbers(ndim)
    ).replace(
        batch_axis=batch_axis % ndim,
        channel_axis=channel_axis % ndim
    )

  return init_fn, apply_fn, kernel_fn


@layer
@supports_masking(remask_kernel=True)
def Dense(
    out_dim: int,
    W_std: float = 1.,
    b_std: float = 0.,
    parameterization: str = 'ntk',
    batch_axis: int = 0,
    channel_axis: int = -1) -> InternalLayer:
  r"""Layer constructor function for a dense (fully-connected) layer.

  Based on `jax.experimental.stax.Dense`.

  Args:
    out_dim:
      The output feature / channel dimension. This is ignored in by the
      `kernel_fn` in `"ntk"` parameterization.

    W_std:
      Specifies the standard deviation of the weights.

    b_std:
      Specifies the standard deviation of the biases.

    parameterization:
      Either `"ntk"` or `"standard"`.

      Under `"ntk"` parameterization (https://arxiv.org/abs/1806.07572, page 3),
      weights and biases are initialized as
      :math:`W_{ij} \sim \mathcal{N}(0,1)`, :math:`b_i \sim \mathcal{N}(0,1)`,
      and the finite width layer equation is
      :math:`z_i = \sigma_W / \sqrt{N} \sum_j W_{ij} x_j + \sigma_b b_i`.

      Under `"standard"` parameterization (https://arxiv.org/abs/2001.07301),
      weights and biases are initialized as :math:`W_{ij} \sim \mathcal{N}(0,
      W_{std}^2/N)`,
      :math:`b_i \sim \mathcal{N}(0,\sigma_b^2)`, and the finite width layer
      equation is
      :math:`z_i = \sum_j W_{ij} x_j + b_i`.

    batch_axis:
      Specifies which axis is contains different elements of the batch.
      Defaults to `0`, the leading axis.

    channel_axis: Specifies which axis contains the features / channels.
      Defaults to `-1`, the trailing axis. For `kernel_fn`, channel size is
      considered to be infinite.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  # TODO(jaschasd): after experimentation, evaluate whether to change default
  # parameterization from "ntk" to "standard"

  parameterization = parameterization.lower()

  def ntk_init_fn(rng, input_shape):
    _channel_axis = channel_axis % len(input_shape)
    output_shape = (input_shape[:_channel_axis] + (out_dim,)
                    + input_shape[_channel_axis + 1:])
    rng1, rng2 = random.split(rng)
    W = random.normal(rng1, (input_shape[_channel_axis], out_dim))

    b_shape = [1] * len(input_shape)
    b_shape[channel_axis] = out_dim
    b = random.normal(rng2, b_shape)

    return output_shape, (W, b)

  def standard_init_fn(rng, input_shape):
    output_shape, (W, b) = ntk_init_fn(rng, input_shape)
    return output_shape, (W * W_std / np.sqrt(input_shape[channel_axis]),
                          b * b_std)

  if parameterization == 'ntk':
    init_fn = ntk_init_fn
  elif parameterization == 'standard':
    init_fn = standard_init_fn
  else:
    raise ValueError(f'Parameterization not supported: {parameterization}')

  def apply_fn(params, inputs, **kwargs):
    W, b = params
    prod = np.moveaxis(np.tensordot(W, inputs, (0, channel_axis)),
                       0, channel_axis)

    if parameterization == 'ntk':
      norm = W_std / np.sqrt(inputs.shape[channel_axis])
      outputs = norm * prod + b_std * b
    elif parameterization == 'standard':
      outputs = prod  + b
    else:
      raise ValueError(f'Parameterization not supported: {parameterization}')

    return outputs

  @_requires(batch_axis=batch_axis,
             channel_axis=channel_axis,
             diagonal_spatial=_Diagonal())  # pytype:disable=wrong-keyword-args
  def kernel_fn(k: Kernel, **kwargs):
    """Compute the transformed kernels after a `Dense` layer."""
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    def fc(x):
      return _affine(x, W_std, b_std)

    if parameterization == 'ntk':
      cov1, nngp, cov2 = map(fc, (cov1, nngp, cov2))
      if ntk is not None:
        ntk = nngp + W_std**2 * ntk
    elif parameterization == 'standard':
      input_width = k.shape1[channel_axis]
      if ntk is not None:
        ntk = input_width * nngp + 1. + W_std**2 * ntk
      cov1, nngp, cov2 = map(fc, (cov1, nngp, cov2))

    return k.replace(cov1=cov1,
                     nngp=nngp,
                     cov2=cov2,
                     ntk=ntk,
                     is_gaussian=True,
                     is_input=False)

  def mask_fn(mask, input_shape):
    return np.all(mask, axis=channel_axis, keepdims=True)

  return init_fn, apply_fn, kernel_fn, mask_fn


@layer
@supports_masking(remask_kernel=True)
def Conv(
    out_chan: int,
    filter_shape: Sequence[int],
    strides: Sequence[int] = None,
    padding: str = Padding.VALID.name,
    W_std: float = 1.0,
    b_std: float = 0.0,
    dimension_numbers: Tuple[str, str, str] = None,
    parameterization: str = 'ntk'
) -> InternalLayer:
  """Layer construction function for a general convolution layer.

  Based on `jax.experimental.stax.GeneralConv`.

  Args:
    out_chan:
      The number of output channels / features of the convolution. This is
      ignored in by the `kernel_fn` in NTK parameterization.
    filter_shape:
      The shape of the filter. The shape of the tuple should agree with the
      number of spatial dimensions in `dimension_numbers`.
    strides:
      The stride of the convolution. The shape of the tuple should agree with
      the number of spatial dimensions in `dimension_numbers`.
    padding:
      Specifies padding for the convolution. Can be one of `"VALID"`, `"SAME"`,
      or `"CIRCULAR"`. `"CIRCULAR"` uses periodic convolutions.
    W_std:
      The standard deviation of the weights.
    b_std:
      The standard deviation of the biases.
    dimension_numbers:
      Specifies which axes should be convolved over. Should match the
      specification in `jax.lax.conv_general_dilated`.
    parameterization:
      Either `"ntk"` or `"standard"`. These parameterizations are the direct
      analogues for convolution of the corresponding parameterizations for
      `Dense` layers.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  return _Conv(out_chan, filter_shape, strides, padding, W_std, b_std,
               dimension_numbers, parameterization, False, True)


@layer
@supports_masking(remask_kernel=True)
def ConvTranspose(
    out_chan: int,
    filter_shape: Sequence[int],
    strides: Sequence[int] = None,
    padding: str = Padding.VALID.name,
    W_std: float = 1.0,
    b_std: float = 0.0,
    dimension_numbers: Tuple[str, str, str] = None,
    parameterization: str = 'ntk'
) -> InternalLayer:
  """Layer construction function for a general transpose convolution layer.

  Based on `jax.experimental.stax.GeneralConvTranspose`.

  Args:
    out_chan:
      The number of output channels / features of the convolution. This is
      ignored in by the `kernel_fn` in `"ntk"` parameterization.
    filter_shape:
      The shape of the filter. The shape of the tuple should agree with the
      number of spatial dimensions in `dimension_numbers`.
    strides:
      The stride of the convolution. The shape of the tuple should agree with
      the number of spatial dimensions in `dimension_nubmers`.
    padding:
      Specifies padding for the convolution. Can be one of `"VALID"`, `"SAME"`,
      or `"CIRCULAR"`. `"CIRCULAR"` uses periodic convolutions.
    W_std:
      standard deviation of the weights.
    b_std:
      standard deviation of the biases.
    dimension_numbers:
      Specifies which axes should be convolved over. Should match the
      specification in `jax.lax.conv_general_dilated`.
    parameterization:
      Either `"ntk"` or `"standard"`. These parameterizations are the direct
      analogues for convolution of the corresponding parameterizations for
      `Dense` layers.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  return _Conv(out_chan, filter_shape, strides, padding, W_std, b_std,
               dimension_numbers, parameterization, True, True)


@layer
@supports_masking(remask_kernel=True)
def ConvLocal(
    out_chan: int,
    filter_shape: Sequence[int],
    strides: Sequence[int] = None,
    padding: str = Padding.VALID.name,
    W_std: float = 1.0,
    b_std: float = 0.0,
    dimension_numbers: Tuple[str, str, str] = None,
    parameterization: str = 'ntk'
) -> InternalLayer:
  """Layer construction function for a general unshared convolution layer.

  Also known and "Locally connected networks" or LCNs, these are equivalent to
  convolutions except for having separate (unshared) kernels at different
  spatial locations.

  Args:
    out_chan:
      The number of output channels / features of the convolution. This is
      ignored in by the `kernel_fn` in `"ntk"` parameterization.
    filter_shape:
      The shape of the filter. The shape of the tuple should agree with the
      number of spatial dimensions in `dimension_numbers`.
    strides:
      The stride of the convolution. The shape of the tuple should agree with
      the number of spatial dimensions in `dimension_numbers`.
    padding:
      Specifies padding for the convolution. Can be one of `"VALID"`, `"SAME"`,
      or `"CIRCULAR"`. `"CIRCULAR"` uses periodic convolutions.
    W_std:
      standard deviation of the weights.
    b_std:
      standard deviation of the biases.
    dimension_numbers:
      Specifies which axes should be convolved over. Should match the
      specification in `jax.lax.conv_general_dilated`.
    parameterization:
      Either `"ntk"` or `"standard"`. These parameterizations are the direct
      analogues for convolution of the corresponding parameterizations for
      `Dense` layers.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  return _Conv(out_chan, filter_shape, strides, padding, W_std, b_std,
               dimension_numbers, parameterization, False, False)


def _Conv(
    out_chan: int,
    filter_shape: Sequence[int],
    strides: Optional[Sequence[int]],
    padding: str,
    W_std: float,
    b_std: float,
    dimension_numbers: Optional[Tuple[str, str, str]],
    parameterization: str,
    transpose: bool,
    shared_weights: bool
) -> InternalLayer:
  """Layer construction function for a general convolution layer.

  Based on `jax.experimental.stax.GeneralConv`.

  Args:
    out_chan:
      The number of output channels / features of the convolution. This is
      ignored in by the `kernel_fn` in NTK parameterization.
    filter_shape: The shape of the filter.
      The shape of the tuple should agree with the number of spatial dimensions
      in `dimension_numbers`.
    strides:
      The stride of the convolution. The shape of the tuple should agree with
      the number of spatial dimensions in `dimension_numbers`.
    padding:
      Specifies padding for the convolution. Can be one of `"VALID"`, `"SAME"`,
      or `"CIRCULAR"`. `"CIRCULAR"` uses periodic convolutions.
    W_std:
      The standard deviation of the weights.
    b_std:
      The standard deviation of the biases.
    dimension_numbers:
      Specifies which axes should be convolved over. Should match the
      specification in `jax.lax.dot_general_dilated`.
    parameterization:
      Either `"ntk"` or `"standard"`. These parameterizations are the direct
      analogues for convolution of the corresponding parameterizations for
      `Dense` layers.
    transpose:
      `True` to use transpose convolution.
    shared_weights:
      `True` to share weights (regular CNNs); otherwise different weights at
      different spatial locations (locally connected networks, LCNs).

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """

  parameterization = parameterization.lower()

  if dimension_numbers is None:
    dimension_numbers = _get_dimension_numbers(len(filter_shape), False)

  lhs_spec, rhs_spec, out_spec = dimension_numbers

  one = (1,) * len(filter_shape)
  strides = strides or one

  padding = Padding(padding)
  if padding == Padding.CIRCULAR:
    apply_padding = Padding.VALID
    init_padding = padding.SAME
  else:
    init_padding = apply_padding = padding

  init_args = dict(dimension_numbers=dimension_numbers,
                   out_chan=out_chan,
                   filter_shape=filter_shape,
                   strides=strides,
                   padding=init_padding.name,
                   W_init=random.normal,
                   b_init=random.normal)
  if transpose:
    if not shared_weights:
      raise NotImplementedError('Unshared transpose CNN not implemented.')

    lax_conv = lax.conv_transpose
    ntk_init_fn, _ = ostax.GeneralConvTranspose(**init_args)
  else:
    if shared_weights:
      lax_conv = lax.conv_general_dilated
      ntk_init_fn, _ = ostax.GeneralConv(**init_args)
    else:
      lax_conv = functools.partial(utils.conv_local_general_dilated,
                                   filter_shape=filter_shape)
      def ntk_init_fn(rng, input_shape):
        """Adapted from `jax.experimental.GeneralConv`."""
        filter_shape_iter = iter(filter_shape)
        conv_kernel_shape = [out_chan if c == 'O' else
                             input_shape[lhs_spec.index('C')] if c == 'I' else
                             next(filter_shape_iter) for c in rhs_spec]

        output_shape = eval_shape(
            lambda lhs, rhs: lax.conv_general_dilated(
                lhs=lhs,
                rhs=rhs,
                window_strides=strides,
                padding=init_padding.name,
                dimension_numbers=dimension_numbers
            ),
            ShapedArray(input_shape, np.float32),
            ShapedArray(conv_kernel_shape, np.float32)
        ).shape

        kernel_shape = [out_chan if c == 'O' else
                        onp.prod(conv_kernel_shape) // out_chan if c == 'I' else
                        output_shape[out_spec.index(c)] for c in rhs_spec]
        bias_shape = [output_shape[i] if c != 'N' else 1
                      for i, c in enumerate(out_spec)]
        k1, k2 = random.split(rng)
        W, b = random.normal(k1, kernel_shape), random.normal(k2, bias_shape)
        return output_shape, (W, b)

  def get_fan_in(input_shape):
    return input_shape[lhs_spec.index('C')] * onp.prod(filter_shape)

  def standard_init_fn(rng, input_shape):
    output_shape, (W, b) = ntk_init_fn(rng, input_shape)
    norm = W_std / np.sqrt(get_fan_in(input_shape))
    return output_shape, (W * norm, b * b_std)

  if parameterization == 'ntk':
    init_fn = ntk_init_fn
  elif parameterization == 'standard':
    init_fn = standard_init_fn
  else:
    raise ValueError(f'Parameterization not supported: {parameterization}.')

  def apply_fn(params, inputs, **kwargs):
    W, b = params

    if parameterization == 'ntk':
      norm = W_std / np.sqrt(get_fan_in(inputs.shape))
      b_rescale = b_std
    elif parameterization == 'standard':
      norm = 1.
      b_rescale = 1.

    if padding == Padding.CIRCULAR and not transpose:
      spatial_axes = tuple(lhs_spec.index(c)
                           for c in rhs_spec if c not in ('I', 'O'))
      inputs = _same_pad_for_filter_shape(inputs, filter_shape, strides,
                                          spatial_axes)

    res = norm * lax_conv(
        inputs,
        W,
        strides,
        apply_padding.name,
        dimension_numbers=dimension_numbers)

    if padding == Padding.CIRCULAR and transpose:
      out_shape = eval_shape(lambda x: lax.conv_transpose(
          lhs=x,
          rhs=W,
          strides=strides,
          padding=Padding.SAME.name,
          dimension_numbers=dimension_numbers
      ), inputs).shape
      spatial_axes = tuple(out_spec.index(c)
                           for c in rhs_spec if c not in ('I', 'O'))
      res = _same_pad_for_filter_shape_transpose(res, spatial_axes, out_shape)

    return res + b_rescale * b

  @_requires(batch_axis=lhs_spec.index('N'),
             channel_axis=lhs_spec.index('C'),
             diagonal_spatial=_Diagonal(
                 output=_Bool.NO if shared_weights else _Bool.MAYBE))  # pytype:disable=wrong-keyword-args
  def kernel_fn(k: Kernel, **kwargs):
    """Compute the transformed kernels after a conv layer."""
    cov1, nngp, cov2, ntk, is_reversed = (k.cov1, k.nngp, k.cov2, k.ntk,
                                          k.is_reversed)

    input_spec = tuple(c for c in lhs_spec if c not in ('N', 'C'))
    conv_spec = tuple(c for c in rhs_spec if c not in ('I', 'O'))
    input_to_filter_permutation = tuple(conv_spec.index(c) for c in input_spec)

    filter_shape_kernel = tuple(filter_shape[p] for p in
                                input_to_filter_permutation)
    strides_kernel = tuple(strides[p] for p in
                           input_to_filter_permutation)

    if k.diagonal_spatial:
      conv_kernel = (_conv_kernel_diagonal_spatial_transpose
                     if transpose else _conv_kernel_diagonal_spatial)

    else:
      if shared_weights:
        if is_reversed:
          filter_shape_kernel = filter_shape_kernel[::-1]
          strides_kernel = strides_kernel[::-1]

        is_reversed = not is_reversed

      if transpose:
        conv_kernel = _conv_kernel_full_spatial_transpose
      else:
        if shared_weights:
          conv_kernel = _conv_kernel_full_spatial_shared
        else:
          conv_kernel = _conv_kernel_full_spatial_unshared

    def conv_unscaled(lhs, batch_ndim):
      lhs = conv_kernel(lhs,
                        filter_shape_kernel,
                        strides_kernel,
                        padding,
                        batch_ndim)
      return lhs

    def affine(out, scale, shift, batch_ndim):
      if out is not None:
        out *= scale
        if k.diagonal_spatial or shared_weights:
          out += shift

        else:
          idx = (Ellipsis,)
          for i in range(batch_ndim, out.ndim, 2):
            shape = [1] * out.ndim
            size = out.shape[i]
            shape[i] = size
            idx += (np.arange(size).reshape(shape),) * 2
          out = ops.index_add(out, idx, shift)

      return out

    def conv(lhs, batch_ndim):
      out = conv_unscaled(lhs, batch_ndim)
      out = affine(out, W_std**2, b_std**2, batch_ndim)
      return out

    cov1 = conv(cov1, 1 if k.diagonal_batch else 2)
    cov2 = conv(cov2, 1 if k.diagonal_batch else 2)

    if parameterization == 'ntk':
      nngp = conv(nngp, 2)
      if ntk is not None:
        ntk = W_std**2 * conv_unscaled(ntk, 2) + nngp

    elif parameterization == 'standard':
      nngp_unscaled = conv_unscaled(nngp, 2)
      if ntk is not None:
        ntk = (get_fan_in(k.shape1) * nngp_unscaled +
               W_std ** 2 * conv_unscaled(ntk, 2))
        ntk = affine(ntk, 1, 1., 2)
      nngp = affine(nngp_unscaled, W_std**2, b_std**2, 2)

    res = k.replace(cov1=cov1,
                    nngp=nngp,
                    cov2=cov2,
                    ntk=ntk,
                    is_gaussian=True,
                    is_reversed=is_reversed,
                    batch_axis=out_spec.index('N'),
                    channel_axis=out_spec.index('C'),
                    is_input=False)

    # Reorder output spatial dimensions if the finite layer does so.
    # TODO(romann): make more efficient / lazy.
    out_spec_kernel = tuple(c for c in out_spec if c not in ('N', 'C'))
    in_to_out_permutation = tuple(out_spec_kernel.index(c) for c in input_spec)
    res = res.transpose(in_to_out_permutation)

    return res

  def mask_fn(mask, input_shape):
    batch_axis, channel_axis = lhs_spec.index('N'), lhs_spec.index('C')

    # Collapse channel dimension of masks, since an FC layer is applied at each
    # spatial location.
    mask = np.all(mask, axis=channel_axis, keepdims=True)

    if transpose:
      rhs_shape = list(filter_shape)
      for c in ('O', 'I'):
        rhs_shape.insert(rhs_spec.index(c), 1)
      rhs = np.ones(rhs_shape)
      # TODO(romann): revisit after https://github.com/google/jax/issues/4012.
      mask = lax.conv_transpose(
          mask.astype(rhs.dtype),
          rhs,
          strides,
          init_padding.name,
          dimension_numbers=dimension_numbers).astype(mask.dtype)

    else:
      mask = _pool_mask(mask, filter_shape, strides, init_padding,
                        batch_axis, channel_axis)
      mask = np.transpose(mask, (out_spec.index(c) for c in lhs_spec))

    return mask

  return init_fn, apply_fn, kernel_fn, mask_fn


@layer
def FanOut(num: int) -> InternalLayer:
  """Layer construction function for a fan-out layer.

  This layer takes an input and produces `num` copies that can be fed into
  different branches of a neural network (for example with residual
  connections).

  Args:
    num: The number of going edges to fan out into.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  init_fn, apply_fn = ostax.FanOut(num)
  kernel_fn = lambda k, **kwargs: [k] * num
  return init_fn, apply_fn, kernel_fn


@layer
@supports_masking(remask_kernel=False)
def FanInSum() -> InternalLayer:
  """Layer construction function for a fan-in sum layer.

  This layer takes a number of inputs (e.g. produced by `FanOut`) and sums the
  inputs to produce a single output.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  init_fn, apply_fn = ostax.FanInSum

  def kernel_fn(ks: Kernels, **kwargs) -> Kernel:
    ks, is_reversed = _proprocess_kernels_for_fan_in(ks)
    if not all([k.shape1 == ks[0].shape1 and
                k.shape2 == ks[0].shape2 for k in ks[1:]]):
      raise ValueError('All shapes should be equal in `FanInSum/FanInProd`.')

    is_gaussian = all(k.is_gaussian for k in ks)
    if not is_gaussian and len(ks) != 1:
      # TODO(xlc): FanInSum/FanInConcat could allow non-Gaussian inputs, but
      # we need to propagate the mean of the random variables as well.
      raise NotImplementedError('`FanInSum` layer along the non-channel axis is'
                                ' only implemented for the case if all input'
                                ' layers guaranteed to be mean-zero Gaussian,'
                                ' i.e. having all `is_gaussian set to `True`.')

    _mats_sum = lambda mats: None if mats[0] is None else sum(mats)

    cov1s = [k.cov1 for k in ks]
    cov2s = [k.cov2 for k in ks]
    nngps = [k.nngp for k in ks]
    ntks = [k.ntk for k in ks]
    cov1, cov2, nngp, ntk = map(_mats_sum, (cov1s, cov2s, nngps, ntks))

    return Kernel(cov1=cov1,
                  cov2=cov2,
                  nngp=nngp,
                  ntk=ntk,
                  x1_is_x2=ks[0].x1_is_x2,
                  is_gaussian=is_gaussian,
                  is_reversed=is_reversed,
                  is_input=ks[0].is_input,
                  diagonal_batch=ks[0].diagonal_batch,
                  diagonal_spatial=ks[0].diagonal_spatial,
                  shape1=ks[0].shape1,
                  shape2=ks[0].shape2,
                  batch_axis=ks[0].batch_axis,
                  channel_axis=ks[0].channel_axis,
                  mask1=None,
                  mask2=None)  # pytype:disable=wrong-keyword-args

  def mask_fn(mask, input_shape):
    return _sum_masks(mask)

  return init_fn, apply_fn, kernel_fn, mask_fn


@layer
@supports_masking(remask_kernel=False)
def FanInProd() -> InternalLayer:
  """Layer construction function for a fan-in product layer.

  This layer takes a number of inputs (e.g. produced by `FanOut`) and
  elementwisely multiply the inputs to produce a single output.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  init_fn, _ = ostax.FanInSum

  def apply_fn(params, inputs, **kwargs):
    return functools.reduce(np.multiply, inputs)

  def kernel_fn(ks: Kernels, **kwargs) -> Kernel:
    ks, is_reversed = _proprocess_kernels_for_fan_in(ks)
    if not all([k.shape1 == ks[0].shape1 and
                k.shape2 == ks[0].shape2 for k in ks[1:]]):
      raise ValueError('All shapes should be equal in `FanInProd`.')

    is_gaussian = len(ks) == 1 and ks[0].is_gaussian

    def _mats_prod(nngps, ntks):
      if None in ntks:
        return functools.reduce(np.multiply, nngps), None

      nngp_prod, ntk_prod = 1., 0.
      for nngp, ntk in zip(nngps, ntks):
        ntk_prod = ntk_prod * nngp + nngp_prod * ntk
        nngp_prod *= nngp
      return nngp_prod, ntk_prod

    cov1s = [k.cov1 for k in ks]
    cov2s = [k.cov2 for k in ks]
    nngps = [k.nngp for k in ks]
    ntks = [k.ntk for k in ks]

    cov1 = functools.reduce(np.multiply, cov1s)
    cov2 = None if None in cov2s else functools.reduce(np.multiply, cov2s)
    nngp, ntk = _mats_prod(nngps, ntks)

    return Kernel(cov1=cov1,
                  cov2=cov2,
                  nngp=nngp,
                  ntk=ntk,
                  x1_is_x2=ks[0].x1_is_x2,
                  is_gaussian=is_gaussian,
                  is_reversed=is_reversed,
                  is_input=ks[0].is_input,
                  diagonal_batch=ks[0].diagonal_batch,
                  diagonal_spatial=ks[0].diagonal_spatial,
                  shape1=None,
                  shape2=None,
                  batch_axis=ks[0].batch_axis,
                  channel_axis=ks[0].channel_axis,
                  mask1=None,
                  mask2=None)  # pytype:disable=wrong-keyword-args

  def mask_fn(mask, input_shape):
    return _sum_masks(mask)

  return init_fn, apply_fn, kernel_fn, mask_fn


@layer
@supports_masking(remask_kernel=False)
def FanInConcat(axis: int = -1) -> InternalLayer:
  """Layer construction function for a fan-in concatenation layer.

  Based on `jax.experimental.stax.FanInConcat`.

  Args:
    axis: Specifies the axis along which input tensors should be concatenated.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  init_fn, apply_fn = ostax.FanInConcat(axis)

  def kernel_fn(ks: Kernels, **kwargs) -> Kernel:
    ks, is_reversed = _proprocess_kernels_for_fan_in(ks)

    diagonal_batch = ks[0].diagonal_batch
    diagonal_spatial = ks[0].diagonal_spatial

    shape1, shape2 = ks[0].shape1, ks[0].shape2

    ndim = len(shape1)
    _axis = axis % ndim
    batch_axis = ks[0].batch_axis
    channel_axis = ks[0].channel_axis

    new_shape1 = shape1[:_axis] + shape1[_axis + 1:]
    new_shape2 = shape2[:_axis] + shape2[_axis + 1:]
    for k in ks:
      k_shape1 = k.shape1[:_axis] + k.shape1[_axis + 1:]
      k_shape2 = k.shape2[:_axis] + k.shape2[_axis + 1:]
      if k_shape1 != new_shape1 or k_shape2 != new_shape2:
        raise ValueError('Non-`axis` shapes should be equal in `FanInConcat`.')

    # Check if inputs are independent Gaussians.
    if _axis != channel_axis:
      is_gaussian = all(k.is_gaussian for k in ks)
      if not is_gaussian:
        # TODO(xlc): FanInSum/FanInConcat could allow non-Gaussian inputs, but
        # we need to propagate the mean of the random variables as well.
        raise NotImplementedError(
            '`FanInConcat` layer along the non-channel axis is only implemented'
            'for the case if all input layers guaranteed to be mean-zero '
            'Gaussian, i.e. having all `is_gaussian set to `True`.')
    else:
      # TODO(romann): allow to apply nonlinearity after
      # channelwise concatenation.
      # TODO(romann): support concatenating different channelwise masks.
      is_gaussian = False

    if _axis == batch_axis:
      warnings.warn(f'Concatenation along the batch axis ({_axis}) gives '
                    f'inconsistent covariances when batching - '
                    f'proceed with caution.')

    spatial_axes = tuple(i for i in range(ndim)
                         if i not in (channel_axis, batch_axis))
    # Change spatial axis according to the kernel `is_reversed`.
    if _axis in spatial_axes and is_reversed:
      _axis = spatial_axes[::-1][spatial_axes.index(_axis)]

    # Map activation tensor axis to the covariance tensor axis.
    tensor_axis_to_kernel_axis = {
        **{
            batch_axis: 0,
            channel_axis: -1,
        },
        **{
            spatial_axis: idx + 1
            for idx, spatial_axis in enumerate(spatial_axes)
        }
    }

    _axis = tensor_axis_to_kernel_axis[_axis]
    widths = [k.shape1[channel_axis] for k in ks]

    cov1 = _concat_kernels([k.cov1 for k in ks], _axis,
                           diagonal_batch, diagonal_spatial, widths)
    cov2 = _concat_kernels([k.cov2 for k in ks], _axis,
                           diagonal_batch, diagonal_spatial, widths)
    nngp = _concat_kernels([k.nngp for k in ks], _axis,
                           False, diagonal_spatial, widths)
    ntk = _concat_kernels([k.ntk for k in ks], _axis,
                          False, diagonal_spatial, widths)

    return Kernel(cov1=cov1,
                  cov2=cov2,
                  nngp=nngp,
                  ntk=ntk,
                  x1_is_x2=ks[0].x1_is_x2,
                  is_gaussian=is_gaussian,
                  is_reversed=is_reversed,
                  is_input=ks[0].is_input,
                  diagonal_batch=diagonal_batch,
                  diagonal_spatial=diagonal_spatial,
                  shape1=None,
                  shape2=None,
                  batch_axis=batch_axis,
                  channel_axis=channel_axis,
                  mask1=None,
                  mask2=None)  # pytype:disable=wrong-keyword-args

  def mask_fn(mask, input_shape):
    return _concat_masks(mask, input_shape, axis)

  return init_fn, apply_fn, kernel_fn, mask_fn


@layer
@supports_masking(remask_kernel=True)
def AvgPool(
    window_shape: Sequence[int],
    strides: Sequence[int] = None,
    padding: str = Padding.VALID.name,
    normalize_edges: bool = False,
    batch_axis: int = 0,
    channel_axis: int = -1) -> InternalLayer:
  """Layer construction function for an average pooling layer.

  Based on `jax.experimental.stax.AvgPool`.

  Args:
    window_shape: The number of pixels over which pooling is to be performed.
    strides: The stride of the pooling window. `None` corresponds to a stride of
      `(1, 1)`.
    padding: Can be `VALID`, `SAME`, or `CIRCULAR` padding. Here `CIRCULAR`
      uses periodic boundary conditions on the image.
    normalize_edges: `True` to normalize output by the effective receptive
      field, `False` to normalize by the window size. Only has effect at the
      edges when `SAME` padding is used. Set to `True` to retain correspondence
      to `ostax.AvgPool`.
    batch_axis: Specifies the batch dimension. Defaults to `0`, the leading
      axis.
    channel_axis: Specifies the channel / feature dimension. Defaults to `-1`,
      the trailing axis. For `kernel_fn`, channel size is considered to be
      infinite.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  return _Pool(Pooling.AVG, window_shape, strides, padding, normalize_edges,
               batch_axis, channel_axis)


@layer
@supports_masking(remask_kernel=True)
def SumPool(
    window_shape: Sequence[int],
    strides: Sequence[int] = None,
    padding: str = Padding.VALID.name,
    batch_axis: int = 0,
    channel_axis: int = -1) -> InternalLayer:
  """Layer construction function for a 2D sum pooling layer.

  Based on `jax.experimental.stax.SumPool`.

  Args:
    window_shape: The number of pixels over which pooling is to be performed.
    strides: The stride of the pooling window. `None` corresponds to a stride of
      `(1, ..., 1)`.
    padding: Can be `VALID`, `SAME`, or `CIRCULAR` padding. Here `CIRCULAR`
      uses periodic boundary conditions on the image.
    batch_axis: Specifies the batch dimension. Defaults to `0`, the leading
      axis.
    channel_axis: Specifies the channel / feature dimension. Defaults to `-1`,
      the trailing axis. For `kernel_fn`, channel size is considered to be
      infinite.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  return _Pool(Pooling.SUM, window_shape, strides, padding, False,
               batch_axis, channel_axis)


def _Pool(
    pool_type: Pooling,
    window_shape: Sequence[int],
    strides: Optional[Sequence[int]],
    padding: str,
    normalize_edges: bool,
    batch_axis: int,
    channel_axis: int) -> InternalLayer:
  """Layer construction function for a 2D pooling layer.

  Based on `jax.experimental.stax.AvgPool` and `jax.experimental.stax.SumPool`.

  Args:
    pool_type: specifies whether average or sum pooling should be performed.
      (`Pooling.AVG` or `Pooling.SUM`)
    window_shape: The number of pixels over which pooling is to be performed.
    strides: The stride of the pooling window. `None` corresponds to a stride of
      `(1, 1)`.
    padding: Can be `VALID`, `SAME`, or `CIRCULAR` padding. Here `CIRCULAR`
      uses periodic boundary conditions on the image.
    normalize_edges: `True` to normalize output by the effective receptive
      field, `False` to normalize by the window size. Only has effect at the
      edges when `SAME` padding is used. Set to `True` to retain correspondence
      to `ostax.AvgPool`.
    batch_axis: Specifies the batch dimension. Defaults to `0`, the leading
      axis.
    channel_axis: Specifies the channel / feature dimension. Defaults to `-1`,
      the trailing axis. For `kernel_fn`, channel size is considered to be
      infinite.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """

  strides = strides or (1,) * len(window_shape)
  padding = Padding(padding)

  if pool_type == Pooling.AVG:
    pool_fn = ostax.AvgPool
  elif pool_type == Pooling.SUM:
    pool_fn = ostax.SumPool
  else:
    raise ValueError('Invalid pooling type {}'.format(pool_type))

  spec = ''.join(c for c in string.ascii_uppercase
                 if c not in ('N', 'C'))[:len(strides)]
  for a in sorted((batch_axis, channel_axis % (2 + len(strides)))):
    if a == batch_axis:
      spec = spec[:a] + 'N' + spec[a:]
    else:
      spec = spec[:a] + 'C' + spec[a:]

  if padding == Padding.CIRCULAR:
    init_fn, _ = pool_fn(window_shape, strides, Padding.SAME.name, spec)
    _, apply_fn_0 = pool_fn(window_shape, strides, Padding.VALID.name, spec)

    def apply_fn(params, inputs, **kwargs):
      non_spatial_axes = (batch_axis, channel_axis % inputs.ndim)
      spatial_axes = tuple(i for i in range(inputs.ndim)
                           if i not in non_spatial_axes)
      inputs = _same_pad_for_filter_shape(inputs, window_shape, strides,
                                          spatial_axes)
      res = apply_fn_0(params, inputs, **kwargs)
      return res

  elif normalize_edges or pool_type == Pooling.SUM:
    init_fn, apply_fn = pool_fn(window_shape, strides, padding.name, spec)

  else:
    def rescaler(dims, strides, padding):
      del dims, strides, padding  # Unused.
      return lambda outputs, inputs, spec: outputs / onp.prod(window_shape)

    pool_fn = ostax._pooling_layer(lax.add, 0., rescaler)
    init_fn, apply_fn = pool_fn(window_shape, strides, padding.name, spec)

  @_requires(batch_axis=batch_axis,
             channel_axis=channel_axis,
             diagonal_spatial=_Diagonal(input=_Bool.MAYBE))  # pytype:disable=wrong-keyword-args
  def kernel_fn(k: Kernel, **kwargs) -> Kernel:
    """Kernel transformation."""
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    if k.diagonal_spatial:
      window_shape_kernel = window_shape
      strides_kernel = strides
    else:
      window_shape_kernel = _double_tuple(
          window_shape[::(-1 if k.is_reversed else 1)])
      strides_kernel = _double_tuple(strides[::(-1 if k.is_reversed else 1)])

    def pool(mat, batch_ndim):
      if mat is None or mat.ndim == 0:
        return mat

      out = _pool_kernel(mat, pool_type, window_shape_kernel, strides_kernel,
                         padding, normalize_edges, batch_ndim)

      if k.diagonal_spatial and pool_type == Pooling.AVG:
        _window_shape = (1,) * batch_ndim + tuple(window_shape)
        _strides = (1,) * batch_ndim + tuple(strides)
        out = _normalize(mat, out, normalize_edges, padding, _strides,
                         _window_shape)
      return out

    nngp = pool(nngp, 2)
    ntk = pool(ntk, 2)
    cov1 = pool(cov1, 1 if k.diagonal_batch else 2)
    cov2 = pool(cov2, 1 if k.diagonal_batch else 2)
    return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

  def mask_fn(mask, input_shape):
    _check_is_implemented(mask, channel_axis)
    return _pool_mask(mask, window_shape, strides, padding,
                      batch_axis, channel_axis)

  return init_fn, apply_fn, kernel_fn, mask_fn


@layer
@supports_masking(remask_kernel=False)
def GlobalSumPool(batch_axis: int = 0, channel_axis: int = -1) -> InternalLayer:
  """Layer construction function for a global sum pooling layer.

  Sums over and removes (`keepdims=False`) all spatial dimensions, preserving
  the order of batch and channel axes.

  Args:
    batch_axis: Specifies the batch dimension. Defaults to `0`, the leading
      axis.
    channel_axis: Specifies the channel / feature dimension. Defaults to `-1`,
      the trailing axis. For `kernel_fn`, channel size is considered to be
      infinite.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  return _GlobalPool(Pooling.SUM, batch_axis, channel_axis)


@layer
@supports_masking(remask_kernel=False)
def GlobalAvgPool(batch_axis: int = 0, channel_axis: int = -1) -> InternalLayer:
  """Layer construction function for a global average pooling layer.

  Averages over and removes (`keepdims=False`) all spatial dimensions,
  preserving the order of batch and channel axes.

  Args:
    batch_axis: Specifies the batch dimension. Defaults to `0`, the leading
      axis.
    channel_axis: Specifies the channel / feature dimension. Defaults to `-1`,
      the trailing axis. For `kernel_fn`, channel size is considered to be
      infinite.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  return _GlobalPool(Pooling.AVG, batch_axis, channel_axis)


def _GlobalPool(
    pool_type: Pooling,
    batch_axis: int,
    channel_axis: int) -> InternalLayer:
  """Layer construction function for a global pooling layer.

  Pools over and removes (`keepdims=False`) all spatial dimensions, preserving
    the order of batch and channel axes.

  Args:
    pool_type: specifies whether average or sum pooling should be performed.
      (`Pooling.AVG` or `Pooling.SUM`).
    batch_axis: Specifies the batch dimension. Defaults to `0`, the leading
      axis.
    channel_axis: Specifies the channel / feature dimension. Defaults to `-1`,
      the trailing axis. For `kernel_fn`, channel size is considered to be
      infinite.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """

  if pool_type == Pooling.AVG:
    pool_fn = lambda x, axis, mask: _mean_and_var(x, axis, mask=mask)[0]
  elif pool_type == Pooling.SUM:
    pool_fn = lambda x, axis, mask: np.sum(x, axis)
  else:
    raise ValueError(f'Invalid pooling type {pool_type}.')

  def init_fn(rng, input_shape):
    ndim = len(input_shape)
    non_spatial_axes = (batch_axis % ndim, channel_axis % ndim)
    output_shape = tuple(input_shape[i] for i in range(ndim)
                         if i in non_spatial_axes)
    return output_shape, ()

  def apply_fn(params, inputs, mask=None, **kwargs):
    non_spatial_axes = (batch_axis % inputs.ndim, channel_axis % inputs.ndim)
    spatial_axes = tuple(i for i in range(inputs.ndim)
                         if i not in non_spatial_axes)
    out = pool_fn(inputs, spatial_axes, mask)
    return out

  @_requires(batch_axis=batch_axis,
             channel_axis=channel_axis,
             diagonal_spatial=_Diagonal(input=_Bool.MAYBE, output=_Bool.YES))  # pytype:disable=wrong-keyword-args
  def kernel_fn(k: Kernel, **kwargs):
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    def _pool(mat, batch_ndim, mask=None):
      if mat is None:
        return mat
      spatial_axes = tuple(range(batch_ndim, mat.ndim))
      out = pool_fn(mat, axis=spatial_axes, mask=mask)
      if k.diagonal_spatial and pool_type == Pooling.AVG:
        out /= utils.size_at(mat, spatial_axes)
      return out

    mask11, mask12, mask22 = k._get_mask_prods(k.mask1, k.mask2)

    cov1 = _pool(cov1, 1 if k.diagonal_batch else 2, mask11)
    cov2 = _pool(cov2, 1 if k.diagonal_batch else 2, mask22)
    nngp = _pool(nngp, 2, mask12)
    ntk = _pool(ntk, 2, mask12)

    ndim = len(k.shape1)
    batch_first = batch_axis % ndim < channel_axis % ndim
    return k.replace(cov1=cov1,
                     nngp=nngp,
                     cov2=cov2,
                     ntk=ntk,
                     batch_axis=0 if batch_first else 1,
                     channel_axis=1 if batch_first else 0,
                     is_reversed=False)

  def mask_fn(mask, input_shape):
    _check_is_implemented(mask, channel_axis)
    non_spatial_axes = (batch_axis % mask.ndim, channel_axis % mask.ndim)
    spatial_axes = tuple(i for i in range(mask.ndim)
                         if i not in non_spatial_axes)
    return np.all(mask, spatial_axes)

  return init_fn, apply_fn, kernel_fn, mask_fn


@layer
@supports_masking(remask_kernel=False)
def Flatten(batch_axis: int = 0, batch_axis_out: int = 0) -> InternalLayer:
  """Layer construction function for flattening all non-batch dimensions.

  Based on `jax.experimental.stax.Flatten`, but allows to specify batch axes.

  Args:
    batch_axis: Specifies the input batch dimension. Defaults to `0`, the
      leading axis.
    batch_axis_out: Specifies the output batch dimension. Defaults to `0`, the
      leading axis.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  if batch_axis_out in (0, -2):
    batch_axis_out = 0
    channel_axis_out = 1
  elif batch_axis_out in (1, -1):
    batch_axis_out = 1
    channel_axis_out = 0
  else:
    raise ValueError(
        f'`batch_axis_out` must be 0 or 1, got {batch_axis_out}.')

  def get_output_shape(input_shape):
    batch_size = input_shape[batch_axis]
    channel_size = functools.reduce(
        op.mul,
        input_shape[:batch_axis] + input_shape[(batch_axis + 1)
                                               or len(input_shape):],
        1
    )
    if batch_axis_out == 0:
      return batch_size, channel_size
    return channel_size, batch_size

  def init_fn(rng, input_shape):
    output_shape = get_output_shape(input_shape)
    return output_shape, ()

  def apply_fn(params, inputs, **kwargs):
    output_shape = get_output_shape(inputs.shape)
    inputs = np.moveaxis(inputs, batch_axis, -batch_axis_out)
    return inputs.reshape(output_shape)

  @_requires(batch_axis=batch_axis,
             channel_axis=None,
             diagonal_spatial=_Diagonal(output=_Bool.YES))  # pytype:disable=wrong-keyword-args
  def kernel_fn(k: Kernel, **kwargs):
    """Compute kernels."""
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    def trace(x, batch_ndim):
      if x is None:
        return x

      if k.diagonal_spatial:
        spatial_axes = tuple(range(x.ndim)[batch_ndim:])
        x = np.mean(x, spatial_axes)

      else:
        while x.ndim > batch_ndim:
          x = np.trace(x, axis1=-2, axis2=-1) / x.shape[-1]

      return x

    cov1 = trace(cov1, 1 if k.diagonal_batch else 2)
    cov2 = trace(cov2, 1 if k.diagonal_batch else 2)
    nngp = trace(nngp, 2)
    ntk = trace(ntk, 2)

    return k.replace(cov1=cov1,
                     nngp=nngp,
                     cov2=cov2,
                     ntk=ntk,
                     is_gaussian=False,
                     is_reversed=False,
                     batch_axis=batch_axis_out,
                     channel_axis=channel_axis_out,
                     diagonal_spatial=False)

  def mask_fn(mask, input_shape):
    mask = np.broadcast_to(mask, input_shape)
    output_shape = get_output_shape(mask.shape)
    return np.moveaxis(mask, batch_axis, batch_axis_out).reshape(output_shape)

  return init_fn, apply_fn, kernel_fn, mask_fn


@layer
@supports_masking(remask_kernel=False)
def Identity() -> InternalLayer:
  """Layer construction function for an identity layer.

  Based on `jax.experimental.stax.Identity`.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  init_fn, apply_fn = ostax.Identity
  kernel_fn = lambda k, **kwargs: k
  return init_fn, apply_fn, kernel_fn


class PositionalEmbedding(enum.Enum):
  """Type of positional embeddings to use in a `GlobalSelfAttention` layer."""
  NONE = 'NONE'
  SUM = 'SUM'
  CONCAT = 'CONCAT'


class AttentionMechanism(enum.Enum):
  """Type of nonlinearity to use in a `GlobalSelfAttention` layer."""
  SOFTMAX = 'SOFTMAX'
  IDENTITY = 'IDENTITY'
  ABS = 'ABS'
  RELU = 'RELU'

  def fn(self):
    return {
        'softmax': ostax.softmax,
        'identity': lambda x: x,
        'abs': np.abs,
        'relu': lambda x: np.maximum(x, 0.)
    }[self.name.lower()]


@layer
@supports_masking(remask_kernel=True)
def GlobalSelfAttention(
    n_chan_out: int,
    n_chan_key: int,
    n_chan_val: int,
    n_heads: int,
    linear_scaling: bool = True,
    W_key_std: float = 1.0,
    W_value_std: float = 1.0,
    W_query_std: float = 1.0,
    W_out_std: float = 1.0,
    b_std: float = 0.0,
    attention_mechanism: str = AttentionMechanism.SOFTMAX.name,
    pos_emb_type: str = PositionalEmbedding.NONE.name,
    pos_emb_p_norm: float = 2,
    pos_emb_decay_fn: Callable[[float], float] = None,
    n_chan_pos_emb: int = None,
    W_pos_emb_std: float = 1.0,
    val_pos_emb: bool = False,
    batch_axis: int = 0,
    channel_axis: int = -1) -> InternalLayer:
  """Layer construction function for (global) scaled dot-product self-attention.

  Infinite width results based on https://arxiv.org/abs/2006.10540.

  Two versions of attention are available (the version to be used is
  determined by the argument `linear_scaling`):

  1. `False`: this is the standard scaled dot-product attention, i.e.,
  the dot product between keys and queries is scaled by the squared root
  of their dimension. The expression for `nngp`/`ntk` involves an integral
  with no known closed form and thus call to `kernel_fn` results in an error.

  2. `True`: scaling the dot products between keys and queries by their
  dimension instead of the square root of the same quantity, AND tying the key
  and query weight matrices. This makes the `nngp`/`ntk` analytically tractable
  but for the price that, unlike in the `False` case, the dot products of keys
  and queries converge to a constant. Because this constant would be zero
  if the key and query weights were independent, the variant where these
  two weight matrices are tied was implemented resulting in non-constant
  attention weights.

  The final computation for single head is then
  :math:`f_h (x) + attention_mechanism(<scaling> Q(x) K(x)^T) V(x)`
  and the output of this layer is computed as
  :math:`f(x) = concat[f_1(x) , ... , f_{<n_{heads}>} (x)] W_{out} + b`
  where the shape of of `b` is `(n_chan_out,)`, i.e., single bias per channel

  The `kernel_fn` computes the limiting kernel of the outputs of this layer
  as the number of heads and the number of feature dimensions of keys/queries
  goes to infinity.

  Args:
    n_chan_out:
      number of feature dimensions of outputs.
    n_chan_key:
      number of feature dimensions of keys/queries.
    n_chan_val:
      number of feature dimensions of values.
    n_heads:
      number of attention heads.
    linear_scaling:
      if `True`, the dot products between keys and queries are scaled by
      `1 / n_chan_key` and the key and query weight matrices are tied;
      if `False`, the dot products are scaled by `1 / sqrt(n_chan_key)` and
      the key and query matrices are independent.
    W_key_std:
      init standard deviation of the key weights values. Due to NTK
      parameterization, influences computation only through the product
      `W_key_std * W_query_std`.
    W_value_std:
      init standard deviation of the value weights values. Due to NTK
      parameterization, influences computation only through the product
      `W_out_std * W_value_std`.
    W_query_std:
      init standard deviation of the query weights values; if `linear_scaling`
      is `True` (and thus key and query weights are tied - see above) then keys
      are computed with `WK = W_key_std * W / sqrt(n_chan_in)` and queries are
      computed with `WQ = W_query_std * W / sqrt(n_chan_in)` weight matrices.
      Due to NTK parameterization, influences computation only through the
      product `W_key_std * W_query_std`.
    W_out_std:
      initial standard deviation of the output weights values. Due to NTK
      parameterization, influences computation only through the product
      `W_out_std * W_value_std`.
    b_std:
      initial standard deviation of the bias values.
    attention_mechanism:
      a string, `"SOFTMAX"`, `"IDENTITY"`, `"ABS"`, or `"RELU"`, the
      transformation applied to dot product attention weights.
    pos_emb_type:
      a string, `"NONE"`, `"SUM"`, or `"CONCAT"`, the type of positional
      embeddings to use. In the infinite-width limit, `"SUM"` and `"CONCAT"`
      are equivalent up to a scaling constant. Keep in mind that all `Dense`
      sub-layers of the attention layer use the NTK parameterization, and weight
      variances are always inversely proportional to the input channel size,
      which leads to different effective variances when using `"SUM"` and
      `"CONCAT"` embeddings, even if all variance scales like `W_key_std` etc.
      are the same.
    pos_emb_p_norm:
      use the unnormalized L-`p` distance to the power of `p` (with
      `p == pos_emb_p_norm`) to compute pairwise distances for positional
      embeddings (see `pos_emb_decay_fn` for details). Used only if
      `pos_emb_type != "NONE"`  and `pos_emb_decay_fn is not None`.
    pos_emb_decay_fn:
      a function applied to the L-`p` distance to the power of `p` (with
      `p == pos_emb_p_norm`) distance between two spatial positions to produce
      the positional embeddings covariance matrix (e.g. power decay,
      exponential decay, etc.). `None` is equivalent to an indicator function
      `lambda d: d == 0`, and returns a diagonal covariance matrix. Used only
      if `pos_emb_type != "NONE"`.
    n_chan_pos_emb:
      number of channels in positional embeddings. `None` means use the same
      number of channels as in the layer inputs. Can be used to tune the
      contribution of positional embeddings relative to contribution of inputs
      if `pos_emb_type == "CONCAT"`. Used only if `pos_emb_type != "NONE"`.
      Will trigger an error if `pos_emb_type == "SUM"`  and `n_chan_pos_emb` is
      not `None` or does not match the layer inputs channel size at runtime.
    W_pos_emb_std:
      init standard deviation of the random positional embeddings. Can be used
      to tune the contribution of positional embeddings relative to the
      contribution of inputs. Used only if `pos_emb_type != "NONE"`. To tune
      the _relative_ (to the inputs) contribution, you can either use
      `n_chan_pos_emb` when `pos_emb_type == "CONCAT"`, or, if
      `pos_emb_type == "CONCAT"`, adjust `W_key_std` etc. relative to
      `W_pos_emb_std`, to keep the total output variance fixed.
    val_pos_emb:
      `True` indicates using positional embeddings when computing all of the
      keys/queries/values matrices, `False` makes them only used for keys and
      queries, but not values. Used only if `pos_emb_type != "NONE"`.
    batch_axis:
      Specifies the batch dimension. Defaults to `0`, the leading axis.
    channel_axis:
      Specifies the channel / feature dimension. Defaults to `-1`, the trailing
      axis. For `kernel_fn`, channel size is considered to be infinite.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.

  Raises:
    NotImplementedError: If `linear_scaling` is `False`, calling `kernel_fn`
      will result in an error as there is no known analytic expression for the
      kernel for `attention_mechanism != "IDENTITY"`.

    NotImplementedError: If `apply_fn` is called with `pos_emb_decay_fn != None`
      , since custom `pos_emb_decay_fn` is only implemented in the infinite
      width regime currently.

  """
  QK_std = W_query_std * W_key_std
  OV_std = W_out_std * W_value_std

  pos_emb_type = PositionalEmbedding(pos_emb_type)
  attention_mechanism = AttentionMechanism(attention_mechanism)

  @functools.lru_cache(1)
  def get_pos_emb_L(spatial_shape):
    with jax.core.eval_context():
      size = utils.size_at(spatial_shape)
      R = _pos_emb_pdist(spatial_shape, pos_emb_p_norm, pos_emb_decay_fn)
      R = utils.unzip_axes(R)
      L = np.linalg.cholesky(np.reshape(R, (size,) * 2)).reshape(R.shape)
      return L

  def init_fn(rng, input_shape):
    _channel_axis = channel_axis % len(input_shape)
    output_shape = (input_shape[:_channel_axis] + (n_chan_out,) +
                    input_shape[_channel_axis + 1:])

    rng_Q, rng_K, rng_V, rng_O, rng_b, rng_pe = random.split(rng, 6)
    rand = random.normal

    n_chan_in_keys = n_chan_in_vals = input_shape[channel_axis]

    # Generate and add / append positional embeddings.
    if pos_emb_type == PositionalEmbedding.NONE:
      pos_emb = None
    else:
      # `None` means positional embeddings have the same number of channels
      # as inputs.
      _n_chan_pos_emb = (n_chan_in_keys if n_chan_pos_emb is None
                         else n_chan_pos_emb)

      pos_emb_shape = list(input_shape)
      pos_emb_shape[channel_axis] = _n_chan_pos_emb
      pos_emb_shape[batch_axis] = 1
      pos_emb = rand(rng_pe, shape=pos_emb_shape)

      if pos_emb_type == PositionalEmbedding.CONCAT:
        n_chan_in_keys += _n_chan_pos_emb
        if val_pos_emb:
          n_chan_in_vals += _n_chan_pos_emb

    key_matrices = rand(rng_K, shape=(n_heads, n_chan_in_keys, n_chan_key))
    val_matrices = rand(rng_V, shape=(n_heads, n_chan_in_vals, n_chan_val))
    W_out = rand(rng_O, shape=(n_chan_val * n_heads, n_chan_out))

    b_shape = [1] * len(input_shape)
    b_shape[_channel_axis] = n_chan_out
    b = rand(rng_b, shape=b_shape)

    if linear_scaling:
      query_matrices = None
      warnings.warn('Linear scaling attention used -> query initialization '
                    'ignored, tying the weights '
                    '(see docstring for more details).')
    else:
      query_matrices = rand(rng_Q, (n_heads, n_chan_in_keys, n_chan_key))

    return (output_shape,
            (query_matrices, key_matrices, val_matrices, W_out, b, pos_emb))

  def apply_fn(params: PyTree,
               inputs: np.ndarray,
               mask: np.ndarray = None,
               **kwargs) -> np.ndarray:
    query_matrices, key_matrices, val_matrices, W_out, b, pos_emb = params

    spatial_shape, spatial_axes = utils.shape_and_axes(
        inputs, (batch_axis, channel_axis))
    n = inputs.shape[batch_axis]

    if pos_emb is not None:
      # Generate positional embeddings.
      if pos_emb_decay_fn is not None:
        L = get_pos_emb_L(spatial_shape)
        first = tuple(range(L.ndim // 2))
        last = tuple(range(L.ndim // 2, L.ndim))
        pos_emb = np.tensordot(L, pos_emb, (last, spatial_axes))
        pos_emb = np.moveaxis(pos_emb, first, spatial_axes)

      # Mask positional embeddings.
      if mask is not None:
        pos_emb = np.where(mask, np.zeros((), pos_emb.dtype), pos_emb)

      pos_emb *= W_pos_emb_std

    # Add / concat positional embeddings.
    if pos_emb_type == PositionalEmbedding.SUM:
      inputs_val = None if val_pos_emb else inputs
      inputs = pos_emb + inputs

    elif pos_emb_type == PositionalEmbedding.CONCAT:
      inputs_val = inputs if not val_pos_emb else None
      _n_chan_pos_emb = (inputs.shape[channel_axis] if n_chan_pos_emb is None
                         else n_chan_pos_emb)
      _channel_axis = channel_axis % inputs.ndim
      pos_emb = np.broadcast_to(
          pos_emb,
          inputs.shape[:_channel_axis] + (_n_chan_pos_emb,) +
          inputs.shape[_channel_axis + 1:])
      inputs = np.concatenate([inputs, pos_emb], axis=channel_axis)

    elif pos_emb_type == PositionalEmbedding.NONE:
      inputs_val = None

    # Prepare separate inputs for values if asked to not add positional
    # embeddings to values.
    if inputs_val is not None:
      inputs_val = np.moveaxis(inputs_val, (batch_axis, channel_axis), (0, -1))
      inputs_val = inputs_val.reshape((n, -1, inputs_val.shape[-1]))

    # Flatten all spatial dimensions and make input of shape
    # `(batch_size, total_spatial_size, n_channels)`.
    inputs = np.moveaxis(inputs, (batch_axis, channel_axis), (0, -1))
    inputs = inputs.reshape((n, -1, inputs.shape[-1]))

    def _inputs_dot(matrices, _inputs=inputs):
      ret = np.dot(_inputs, matrices)
      return np.moveaxis(ret, 2, 0)

    # Drop positional embedding information for value matrices if requested.
    if inputs_val is not None:
      values = _inputs_dot(val_matrices, inputs_val)
      n_chan_in = inputs_val.shape[-1]
    else:
      values = _inputs_dot(val_matrices)
      n_chan_in = inputs.shape[-1]

    keys = _inputs_dot(key_matrices)
    if linear_scaling:
      queries = keys
    else:
      queries = _inputs_dot(query_matrices)

    G_mat = np.matmul(queries, np.moveaxis(keys, -1, -2))
    norm = inputs.shape[-1] * n_chan_key ** (1 if linear_scaling else 0.5)
    G_mat *= QK_std / norm

    if mask is not None:
      mask = np.all(mask, axis=channel_axis, keepdims=True)
      mask = np.moveaxis(mask, (batch_axis, channel_axis), (0, -1))
      mask = mask.reshape((1, mask.shape[0], 1, -1))

      if attention_mechanism == AttentionMechanism.SOFTMAX:
        G_mat = np.where(mask, _NEG_INF, G_mat)
      elif attention_mechanism in (AttentionMechanism.IDENTITY,
                                   AttentionMechanism.RELU,
                                   AttentionMechanism.ABS):
        G_mat = np.where(mask, np.zeros((), G_mat.dtype), G_mat)
      else:
        raise NotImplementedError(attention_mechanism, mask)

    G_mat = attention_mechanism.fn()(G_mat)
    heads = np.matmul(G_mat, values)
    heads = np.moveaxis(heads, 0, -1)
    heads = np.reshape(heads, heads.shape[:-2] + (-1,))

    outputs = np.matmul(heads, W_out)
    outputs *= OV_std / (n_chan_val * n_heads * n_chan_in) ** 0.5

    outputs = np.reshape(outputs, (n,) + spatial_shape + (n_chan_out,))
    outputs = np.moveaxis(outputs, (0, -1), (batch_axis, channel_axis))
    return outputs + b_std * b

  @_requires(batch_axis=batch_axis,
             channel_axis=channel_axis,
             diagonal_spatial=_Diagonal(input=_Bool.NO))  # pytype:disable=wrong-keyword-args
  def kernel_fn(k: Kernel, **kwargs):
    # Generate (optional) positional embedding covariances.
    R1, R12, R2 = _get_all_pos_emb(k, pos_emb_type, pos_emb_p_norm,
                                   pos_emb_decay_fn)

    def _get_interpolation_coefficients():
      input_weight, pos_emb_weight = 1, W_pos_emb_std**2

      if pos_emb_type == PositionalEmbedding.CONCAT:
        # Reweight based on relative widths of inputs and channels.
        n_chan_input = k.shape1[channel_axis]
        _n_chan_pos_emb = (k.shape1[channel_axis] if n_chan_pos_emb is None
                           else n_chan_pos_emb)
        n_chan_total = n_chan_input + _n_chan_pos_emb

        input_weight *= n_chan_input / n_chan_total
        pos_emb_weight *= _n_chan_pos_emb / n_chan_total

      return input_weight, pos_emb_weight

    def weighted_sum(x, y, x_weight, y_weight):
      if x is None or y is None:
        return x
      return x_weight * x + y_weight * y

    # Generate kernel interpolations.
    kern_weight, pos_emb_weight = _get_interpolation_coefficients()

    cov1_interp = weighted_sum(k.cov1, R1, kern_weight, pos_emb_weight)
    cov2_interp = weighted_sum(k.cov2, R2, kern_weight, pos_emb_weight)

    if val_pos_emb or (not linear_scaling and
                       attention_mechanism == AttentionMechanism.IDENTITY):
      # These interpolations need to be computed in `d^-1/2` scaling even if
      # positional embeddings aren't used in `values`.
      nngp_interp = weighted_sum(k.nngp, R12, kern_weight, pos_emb_weight)
      ntk_interp = weighted_sum(k.ntk, R12, kern_weight, pos_emb_weight)

    if linear_scaling:

      def _get_weighting(mat, mask):
        if mat is None:
          return None

        if not k.diagonal_batch:
          mat = np.moveaxis(np.diagonal(mat, axis1=0, axis2=1), -1, 0)

        if mask is not None:
          mask = np.all(mask, axis=channel_axis, keepdims=True)
          mask = np.squeeze(np.moveaxis(mask, (batch_axis, channel_axis),
                                        (0, -1)), -1)
          if k.is_reversed:
            mask = np.moveaxis(mask,
                               range(1, mask.ndim),
                               range(mask.ndim -1, 0, -1))
          mask = utils.interleave_ones(mask, 1, mask.ndim, x_first=False)
          if attention_mechanism == AttentionMechanism.SOFTMAX:
            mat = np.where(mask, _NEG_INF, mat)
          else:
            mat = np.where(mask, np.zeros((), mat.dtype), mat)

        if attention_mechanism == AttentionMechanism.SOFTMAX:
          axes = tuple(range(mat.ndim))
          return attention_mechanism.fn()(QK_std * mat, axis=axes[2::2])
        else:
          return attention_mechanism.fn()(QK_std * mat)

      def _weigh_kernel(mat, G1, G2=None):
        if mat is not None and mat.ndim != 0:
          G2 = G1 if G2 is None else G2

          # Spatial axes
          G1_dims = tuple(range(1, G1.ndim))
          G2_dims = tuple(range(G1.ndim, G1.ndim + G2.ndim - 1))
          mat_dims = utils.zip_flat(G1_dims[1::2], G2_dims[1::2])
          res_dims = utils.zip_flat(G1_dims[::2], G2_dims[::2])

          G1_dims = (0,) + G1_dims

          # Batch axes
          if mat.ndim % 2:  # Even number of spatial axes + 1 or 2 batch axes
            G2_dims = (0,) + G2_dims
            mat_dims = (0,) + mat_dims
            res_dims = (0,) + res_dims

          else:
            G2_dims = (-1,) + G2_dims
            mat_dims = (0, -1) + mat_dims
            res_dims = (0, -1) + res_dims

          mat = np.einsum(G1, G1_dims, mat, mat_dims, G2, G2_dims, res_dims,
                          optimize=True)
        return _affine(mat, OV_std, b_std)

      G1 = _get_weighting(cov1_interp, k.mask1)
      G2 = _get_weighting(cov2_interp, k.mask2)

      cov1 = _weigh_kernel(cov1_interp if val_pos_emb else k.cov1, G1)
      cov2 = _weigh_kernel(cov2_interp if val_pos_emb else k.cov2, G2)

      nngp = _weigh_kernel(nngp_interp if val_pos_emb else k.nngp, G1, G2)
      if k.ntk is None:
        ntk = None
      else:
        ntk = _weigh_kernel(ntk_interp if val_pos_emb else k.ntk,
                            G1, G2) + 2 * (nngp - b_std**2)

    elif attention_mechanism == AttentionMechanism.IDENTITY:

      def dot(lhs, rhs, diagonal_batch=k.diagonal_batch):
        if lhs is None:
          return None

        c_axes = tuple(range(1 if diagonal_batch else 2, lhs.ndim))
        if rhs is None:
          return np.sum(lhs**2, axis=c_axes, keepdims=True)

        rhs = np.broadcast_to(rhs, lhs.shape)
        b_axes = (0,) if diagonal_batch else (0, 1)
        res = lax.dot_general(lhs, rhs, ((c_axes, c_axes), (b_axes, b_axes)))
        return res.reshape(res.shape + (1,) * len(c_axes))

      dot11 = dot(cov1_interp, None if val_pos_emb else k.cov1)
      dot12 = dot(nngp_interp, None if val_pos_emb else k.nngp, False)
      dot22 = dot(cov2_interp, None if val_pos_emb else k.cov2)

      std = QK_std * OV_std

      nngp = _affine(dot12 * nngp_interp, std, b_std)
      cov1 = _affine(dot11 * cov1_interp, std, b_std)
      cov2 = _affine(None if dot22 is None else dot22 * cov2_interp, std, b_std)

      if ntk_interp is not None:
        if val_pos_emb or pos_emb_type == PositionalEmbedding.NONE:
          nngp_dot_ntk = dot(nngp_interp, ntk_interp, False)
          ntk = 2 * nngp_dot_ntk

        else:
          nngp_dot_ntk_1 = dot(nngp_interp, k.ntk, False)
          nngp_dot_ntk_2 = dot(k.nngp, ntk_interp, False)
          ntk = (nngp_dot_ntk_1 + nngp_dot_ntk_2)

        ntk = _affine(
            ntk * nngp_interp + dot12 * (ntk_interp + 4 * nngp_interp),
            std,
            b_std)

      else:
        ntk = None

    else:
      raise NotImplementedError(f'No known closed form expression for square '
                                f'root scaling and {attention_mechanism} '
                                f'attention mechanism.')

    return k.replace(cov1=cov1,
                     nngp=nngp,
                     cov2=cov2,
                     ntk=ntk,
                     is_gaussian=True)

  def mask_fn(mask, input_shape):
    return np.all(mask, channel_axis, keepdims=True)

  return init_fn, apply_fn, kernel_fn, mask_fn


@layer
@supports_masking(remask_kernel=False)
def LayerNorm(
    axis: Axes = -1,
    eps: float = 1e-12,
    batch_axis: int = 0,
    channel_axis: int = -1) -> InternalLayer:
  """Layer normalisation.

  Args:
    axis:
      dimensions over which to normalize.
    eps:
      (small) positive constant to be added to the variance estimates in order
      to prevent division by zero.
    batch_axis:
      batch dimension. Defaults to `0`, the leading axis.
    channel_axis:
      channel / feature dimension. Defaults to `-1`, the trailing axis. For
      `kernel_fn`, channel size is considered to be infinite.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  def init_fn(rng, input_shape):
    return input_shape, ()

  def apply_fn(params, inputs, mask=None, **kwargs):
    _axis = utils.canonicalize_axis(axis, inputs)
    mean, var = _mean_and_var(inputs, _axis, keepdims=True, mask=mask,
                              get_var=True)
    return (inputs - mean) / np.sqrt(eps + var)

  @_requires(batch_axis=batch_axis, channel_axis=channel_axis)
  def kernel_fn(k: Kernel, **kwargs):
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    if not k.is_gaussian:
      raise NotImplementedError('LayerNorm only implemented for Gaussian '
                                'inputs.')

    ndim = len(k.shape1)
    _channel_axis = channel_axis % ndim
    _batch_axis = batch_axis % ndim
    _axis = utils.canonicalize_axis(axis, k.shape1)

    if _channel_axis not in _axis:
      raise ValueError(f'Normalisation over channels (axis {_channel_axis})'
                       f'necessary for convergence to an asymptotic kernel; '
                       f'got axis={_axis}.')

    _axis.remove(_channel_axis)

    spatial_axes = tuple(i for i in range(len(k.shape1))
                         if i not in (_channel_axis, batch_axis))

    # Batch axis
    if _batch_axis in _axis:
      kernel_axis = (0,)
      _axis.remove(_batch_axis)
    else:
      kernel_axis = ()

    # Spatial axes
    kernel_axis += tuple(
        1 + spatial_axes[::(-1 if k.is_reversed else 1)].index(i)
        for i in _axis)

    # Prepare masks for normalization
    def prepare_mask(m):
      if m is None:
        return m

      if m.shape[channel_axis] != 1:
        raise NotImplementedError('`LayerNorm` with different per-channel masks'
                                  'not implemented in the infinite limit.')

      m = np.squeeze(m, channel_axis)
      if k.is_reversed:
        m = np.moveaxis(m, range(1, m.ndim), range(m.ndim - 1, 0, -1))

      return m

    prod11, prod12, prod22 = _get_diagonal_outer_prods(
        eps + cov1,
        cov2 if cov2 is None else eps + cov2,
        k.diagonal_batch,
        k.diagonal_spatial,
        op.mul,
        axis=kernel_axis,
        mask1=prepare_mask(k.mask1),
        mask2=prepare_mask(k.mask2),
    )

    nngp /= np.sqrt(prod12)

    if ntk is not None:
      ntk /= np.sqrt(prod12)

    cov1 /= np.sqrt(prod11)
    if cov2 is not None:
      cov2 /= np.sqrt(prod22)

    return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

  return init_fn, apply_fn, kernel_fn


@layer
@supports_masking(remask_kernel=False)
def Dropout(rate: float, mode: str = 'train') -> InternalLayer:
  """Dropout layer.

  Based on `jax.experimental.stax.Dropout`.

  Args:
    rate:
      Specifies the keep `rate`, e.g. `rate=1` is equivalent to keeping all
      neurons.
    mode:
      Either `"train"` or `"test"`.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  if mode not in ('test', 'train'):
    raise ValueError('The `mode` must be either "test"  or "train".')
  if rate <= 0. or rate > 1.:
    raise ValueError('The `rate` must be > 0. and <= 1.')

  init_fn, apply_fn = ostax.Dropout(rate, mode=mode)
  kernel_fn_test = lambda k, **kwargs: k

  @_requires(use_dropout=True)
  def kernel_fn_train(k: Kernel, **kwargs):
    """kernel_fn for `train` mode."""
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    if k.is_input:
      raise ValueError('Dropout cannot be applied to the input layer.')

    factor = 1./rate

    cov1 = _diag_mul(cov1, factor, k.diagonal_batch, k.diagonal_spatial)
    cov2 = _diag_mul(cov2, factor, k.diagonal_batch, k.diagonal_spatial)

    new_factor = np.where(k.x1_is_x2, factor, 1.)
    nngp = _diag_mul(nngp, new_factor, False, k.diagonal_spatial)
    ntk = _diag_mul(ntk, new_factor, False, k.diagonal_spatial)

    # TODO(xlc): under which condition could we leave `is_gaussian` unchanged?
    return k.replace(cov1=cov1,
                     nngp=nngp,
                     cov2=cov2,
                     ntk=ntk,
                     is_gaussian=False)

  kernel_fn = kernel_fn_test if mode == 'test' else kernel_fn_train

  return init_fn, apply_fn, kernel_fn


# NONLINEARITIES / ACTIVATION FUNCTIONS


@layer
@supports_masking(remask_kernel=True)
def Erf(
    a: float = 1.,
    b: float = 1.,
    c: float = 0.,
    do_backprop: bool = False) -> InternalLayer:
  """Affine transform of `Erf` nonlinearity, i.e. `a * Erf(b * x) + c`.

  Args:
    a: output scale.
    b: input scale.
    c: output shift.
    do_backprop: set to `True` if you want to backpropagate through the kernel.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  def fn(x):
    return a * erf(b * x) + c

  _requires(diagonal_spatial=_Diagonal())  # pytype:disable=wrong-keyword-args
  def kernel_fn(k: Kernel) -> Kernel:
    k *= b

    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    cov1_denom = 1 + 2 * cov1
    cov2_denom = None if cov2 is None else 1 + 2 * cov2

    prod11, prod12, prod22 = _get_diagonal_outer_prods(cov1_denom,
                                                       cov2_denom,
                                                       k.diagonal_batch,
                                                       k.diagonal_spatial,
                                                       op.mul)

    def nngp_ntk_fn(
        nngp: np.ndarray,
        prod: np.ndarray,
        ntk: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
      if ntk is not None:
        dot_sigma = 4 / (np.pi * np.sqrt(prod - 4 * nngp ** 2))
        ntk *= dot_sigma
      nngp = _arcsin(2 * nngp / np.sqrt(prod), do_backprop) * 2 / np.pi
      return nngp, ntk

    def nngp_fn_diag(nngp: np.ndarray, denom: np.ndarray) -> np.ndarray:
      return np.arcsin(2 * nngp / denom) * 2 / np.pi

    nngp, ntk = nngp_ntk_fn(nngp, prod12, ntk)

    if k.diagonal_batch and k.diagonal_spatial:
      cov1 = nngp_fn_diag(cov1, cov1_denom)
      if cov2 is not None:
        cov2 = nngp_fn_diag(cov2, cov2_denom)
    else:
      cov1, _ = nngp_ntk_fn(cov1, prod11)
      if cov2 is not None:
        cov2, _ = nngp_ntk_fn(cov2, prod22)

    k = k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)
    return a * k + c

  return _elementwise(fn, f'Erf({a}, {b}, {c})', kernel_fn)


def Sigmoid_like():
  """A sigmoid like function `f(x) = .5 * erf(x / 2.4020563531719796) + .5`.

  The constant `2.4020563531719796` is chosen so that the squared loss between
  this function and the ground truth sigmoid is minimized on the interval
  `[-5, 5]`; see
  https://gist.github.com/SiuMath/679e8bb4bce13d5f2383a27eca649575.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  return Erf(a=0.5, b=1/2.4020563531719796, c=0.5)


@layer
@supports_masking(remask_kernel=False)
def Gelu(
    do_backprop: bool = False) -> InternalLayer:
  """Gelu function.

  Args:
    do_backprop: set to `True` if you want to backpropagate through the kernel.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  def fn(x):
    return 0.5 * x * (1. + erf(x / np.sqrt(2.)))

  _requires(diagonal_spatial=_Diagonal())  # pytype:disable=wrong-keyword-args
  def kernel_fn(k: Kernel) -> Kernel:
    """Compute kernels after a `Gelu` layer; NNGP see `arXiv:2002.08517`."""
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    cov1_plus_1 = cov1 + 1
    cov2_plus_1 = None if cov2 is None else cov2 + 1

    prod11_plus_1, prod12_plus_1, prod22_plus_1 = _get_diagonal_outer_prods(
        cov1_plus_1, cov2_plus_1, k.diagonal_batch, k.diagonal_spatial, op.mul)
    prod11, prod12, prod22 = _get_diagonal_outer_prods(
        cov1, cov2, k.diagonal_batch, k.diagonal_spatial, op.mul)

    def nngp_ntk_fn(nngp: np.ndarray,
                    prod: np.ndarray,
                    prod_plus_1: np.ndarray,
                    ntk: np.ndarray = None
                    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
      delta_squared = prod_plus_1 - nngp**2
      delta = _safe_sqrt(delta_squared)
      ratio = nngp / _safe_sqrt(prod_plus_1)
      new_nngp = (nngp**2 + prod * delta_squared) / (prod_plus_1 * delta)
      new_nngp += nngp * _arcsin(ratio, do_backprop)
      new_nngp /= 2 * np.pi
      new_nngp += 0.25 * nngp

      if ntk is not None:
        second_term = 0.25 + _arcsin(ratio, do_backprop) / (2 * np.pi)
        first_term = 1 / delta_squared + (1 - prod) / prod_plus_1 + 1
        first_term *= nngp / delta / (2. * np.pi)
        dot_sigma = first_term + second_term
        ntk *= dot_sigma
      return new_nngp, ntk

    def nngp_fn_diag(nngp: np.ndarray) -> np.ndarray:
      new_nngp = nngp / ((nngp + 1.) * np.sqrt(1. + 2. * nngp))
      new_nngp += _arcsin(nngp / (nngp + 1), do_backprop) / 2
      new_nngp /= np.pi
      new_nngp += 0.25
      new_nngp *= nngp
      return new_nngp

    nngp, ntk = nngp_ntk_fn(nngp, prod12, prod12_plus_1, ntk)

    if k.diagonal_batch and k.diagonal_spatial:
      cov1 = nngp_fn_diag(cov1)
      if cov2 is not None:
        cov2 = nngp_fn_diag(cov2)
    else:
      cov1, _ = nngp_ntk_fn(cov1, prod11, prod11_plus_1)
      if cov2 is not None:
        cov2, _ = nngp_ntk_fn(cov2, prod22, prod22_plus_1)

    return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

  return _elementwise(fn, 'Gelu', kernel_fn)


@layer
@supports_masking(remask_kernel=True)
def Sin(
    a: float = 1.,
    b: float = 1.,
    c: float = 0.) -> InternalLayer:
  """Affine transform of `Sin` nonlinearity, i.e. `a sin(b*x + c)`.

  Args:
    a: output scale.
    b: input scale.
    c: input phase shift.
  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  def fn(x):
    return a * np.sin(b * x + c)

  _requires(diagonal_spatial=_Diagonal())  # pytype:disable=wrong-keyword-args
  def kernel_fn(k: Kernel) -> Kernel:
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    sum11, sum12, sum22 = _get_diagonal_outer_prods(cov1,
                                                    cov2,
                                                    k.diagonal_batch,
                                                    k.diagonal_spatial,
                                                    op.add)
    half_a_square = a**2 / 2.

    def nngp_ntk_fn(nngp, sum_, ntk=None):
      s1 = np.exp(b ** 2 * (-0.5 * sum_ + nngp))
      s2 = np.exp(b ** 2 * (-0.5 * sum_ - nngp)) * np.cos(2 * c)
      nngp = half_a_square * (s1 - s2)
      if ntk is not None:
        ntk *= half_a_square * b**2 * (s1 + s2)
      return nngp, ntk

    def nngp_fn_diag(nngp):
      return half_a_square *(1. - np.exp(-b**2 * nngp) * np.cos(2 * c))

    nngp, ntk = nngp_ntk_fn(nngp, sum12, ntk)

    if k.diagonal_batch and k.diagonal_spatial:
      cov1 = nngp_fn_diag(sum11)
      if cov2 is not None:
        cov2 = nngp_fn_diag(sum22)
    else:
      cov1, _ = nngp_ntk_fn(cov1, sum11)
      if cov2 is not None:
        cov2, _ = nngp_ntk_fn(cov2, sum22)

    return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

  return _elementwise(fn, f'Sin({a}, {b}, {c})', kernel_fn)


@layer
@supports_masking(remask_kernel=True)
def Rbf(
    gamma: float = 1.0) -> InternalLayer:
  """Dual activation function for normalized RBF or squared exponential kernel.

  Dual activation function is `f(x) = sqrt(2)*sin(sqrt(2*gamma) x + pi/4)`.
  NNGP kernel transformation correspond to (with input dimension `d`)
  `k = exp(- gamma / d * ||x - x'||^2) = exp(- gamma*(q11 + q22 - 2 * q12))`.

  Args:
    gamma:
      related to characteristic length-scale (l) that controls width of the
      kernel, where `gamma = 1 / (2 l^2)`.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  def fn(x):
    return np.sqrt(2) * np.sin(np.sqrt(2 * gamma) * x + np.pi/4)

  _requires(diagonal_spatial=_Diagonal())  # pytype:disable=wrong-keyword-args
  def kernel_fn(k: Kernel) -> Kernel:
    """Compute new kernels after an `Rbf` layer."""
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    sum11, sum12, sum22 = _get_diagonal_outer_prods(cov1,
                                                    cov2,
                                                    k.diagonal_batch,
                                                    k.diagonal_spatial,
                                                    op.add)

    def nngp_ntk_fn(nngp, sum_, ntk):
      s1 = np.exp(gamma * (-sum_ + 2 * nngp))
      nngp = s1
      if ntk is not None:
        ntk *= 2 * gamma * s1
      return nngp, ntk

    def nngp_fn_diag(nngp):
      return np.ones_like(nngp)

    nngp, ntk = nngp_ntk_fn(nngp, sum12, ntk)

    if k.diagonal_batch and k.diagonal_spatial:
      cov1 = nngp_fn_diag(sum11)
      if cov2 is not None:
        cov2 = nngp_fn_diag(sum22)
    else:
      cov1, _ = nngp_ntk_fn(cov1, sum11, None)
      if cov2 is not None:
        cov2, _ = nngp_ntk_fn(cov2, sum22, None)

    return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

  return _elementwise(fn, f'Rbf({gamma})', kernel_fn)


@layer
@supports_masking(remask_kernel=False)
def ABRelu(
    a: float,
    b: float,
    do_backprop: bool = False,
    do_stabilize: bool = False) -> InternalLayer:
  """ABReLU nonlinearity, i.e. `a * min(x, 0) + b * max(x, 0)`.

  Args:
    a: slope for `x < 0`.
    b: slope for `x > 0`.
    do_backprop: set to `True` if you want to backpropagate through the kernel.
    do_stabilize: set to `True` for very deep networks.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  def fn(x):
    return a * np.minimum(x, 0) + b * np.maximum(x, 0)

  _requires(diagonal_spatial=_Diagonal())  # pytype:disable=wrong-keyword-args
  def kernel_fn(k: Kernel) -> Kernel:
    """Compute new kernels after an `ABRelu` layer.

    See https://arxiv.org/pdf/1711.09090.pdf for the leaky ReLU derivation.
    """
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    if do_stabilize:
      factor = np.maximum(np.max(np.abs(nngp)), 1e-12)
      nngp /= factor
      cov1 /= factor
      if cov2 is not None:
        cov2 /= factor

    prod11, prod12, prod22 = _get_diagonal_outer_prods(cov1,
                                                       cov2,
                                                       k.diagonal_batch,
                                                       k.diagonal_spatial,
                                                       op.mul)

    def nngp_ntk_fn(nngp, prod, ntk=None):
      cosines = nngp / _safe_sqrt(prod)
      angles = _arccos(cosines, do_backprop)

      dot_sigma = (a**2 + b**2 - (a - b)**2 * angles / np.pi) / 2
      nngp = ((a - b) ** 2 * _sqrt(prod - nngp ** 2, do_backprop) / (2 * np.pi)
              + dot_sigma * nngp)

      if ntk is not None:
        ntk *= dot_sigma

      return nngp, ntk

    def nngp_fn_diag(nngp):
      return (a**2 + b**2) / 2 * nngp

    nngp, ntk = nngp_ntk_fn(nngp, prod12, ntk=ntk)

    if k.diagonal_batch and k.diagonal_spatial:
      cov1 = nngp_fn_diag(cov1)
      if cov2 is not None:
        cov2 = nngp_fn_diag(cov2)
    else:
      cov1, _ = nngp_ntk_fn(cov1, prod11)
      if cov2 is not None:
        cov2, _ = nngp_ntk_fn(cov2, prod22)

    if do_stabilize:
      nngp *= factor
      cov1 *= factor
      if cov2 is not None:
        cov2 *= factor

    return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

  return _elementwise(fn, f'ABReLU({a}, {b})', kernel_fn)


def Relu(
    do_backprop: bool = False,
    do_stabilize: bool = False) -> InternalLayer:
  """ReLU nonlinearity.

  Args:
    do_backprop: set to `True` if you want to backpropagate through the kernel.
    do_stabilize: set to `True` for very deep networks.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  return ABRelu(0, 1, do_backprop, do_stabilize)


def LeakyRelu(
    alpha: float,
    do_backprop: bool = False,
    do_stabilize: bool = False) -> InternalLayer:
  """Leaky ReLU nonlinearity, i.e. `alpha * min(x, 0) + max(x, 0)`.

  Args:
    alpha: slope for `x < 0`.
    do_backprop: set to `True` if you want to backpropagate through the kernel.
    do_stabilize: set to `True` for very deep networks.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  return ABRelu(alpha, 1, do_backprop, do_stabilize)


def Abs(
    do_backprop: bool = False,
    do_stabilize: bool = False) -> InternalLayer:
  """Absolute value nonlinearity.

  Args:
    do_backprop: set to `True` if you want to backpropagate through the kernel.
    do_stabilize: set to `True` for very deep networks.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  return ABRelu(-1, 1, do_backprop, do_stabilize)


@layer
@supports_masking(remask_kernel=True)
def ElementwiseNumerical(
    fn: Callable[[float], float],
    deg: int,
    df: Callable[[float], float] = None,
    do_backprop: bool = False) -> InternalLayer:
  """Activation function using numerical integration.

  Supports general activation functions using Gauss-Hermite quadrature.

  Args:
    fn: activation function.
    deg: number of sample points and weights for quadrature. It must be >= 1.
      We observe for smooth activations deg=25 is a good place to start.
      For non-smooth activation functions (e.g. ReLU, Abs) quadrature is not
      recommended (for now use `nt.monte_carlo_kernel_fn`). Due to bivariate
      integration, compute time and memory scale as O(deg**2) for more
      precision. See eq (13) in
      https://mathworld.wolfram.com/Hermite-GaussQuadrature.html
      for error estimates in the case of 1d Gauss-Hermite quadrature.
    df: optional, derivative of the activation funtion(`fn`). If not provided,
      it is computed by `jax.grad`. Providing analytic derivative can speed up
      the NTK computations.
    do_backprop: set to `True` if you want to backpropagate through the kernel.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  warnings.warn(
      f'Numerical Activation Layer with fn={fn}, deg={deg} used!'
      'Note that numerical error is controlled by `deg` and for a given'
      'tolerance level, required `deg` will highly be dependent on the choice'
      'of `fn`.')

  quad_points = osp.special.roots_hermite(deg)

  if df is None:
    warnings.warn(
        'Using JAX autodiff to compute the `fn` derivative for NTK. Beware of '
        'https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where.')
    df = np.vectorize(grad(fn))

  _requires(diagonal_spatial=_Diagonal())  # pytype:disable=wrong-keyword-args
  def kernel_fn(k: Kernel) -> Kernel:
    """Kernel transformation of activation function using quadrature."""
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    d1 = _get_diagonal(cov1, k.diagonal_batch, k.diagonal_spatial)
    d2 = _get_diagonal(cov2, k.diagonal_batch, k.diagonal_spatial)

    end_axis = 1 if k.diagonal_spatial else cov1.ndim
    q11 = utils.interleave_ones(d1, 0, end_axis, True)
    q22 = utils.interleave_ones(d1 if d2 is None else d2, 0, end_axis, False)

    def nngp_ntk_fn(nngp, q11, q22, ntk=None):
      """Simple Gauss-Hermite quadrature routine."""
      xs, ws = quad_points
      grid = np.outer(ws, ws)
      x = xs.reshape((xs.shape[0],) + (1,) * (nngp.ndim + 1))
      y = xs.reshape((1, xs.shape[0]) + (1,) * nngp.ndim)
      xy_axes = (0, 1)

      nngp = np.expand_dims(nngp, xy_axes)
      q11, q22 = np.expand_dims(q11, xy_axes), np.expand_dims(q22, xy_axes)

      def integrate(f):
        fvals = f(_sqrt(2*q11, do_backprop) * x) * f(
            nngp/_sqrt(q11/2, do_backprop) * x + _sqrt(
                2*(q22 - nngp**2/q11), do_backprop)* y)
        return np.tensordot(grid, fvals, (xy_axes, xy_axes)) / np.pi

      if ntk is not None:
        ntk *= integrate(df)
      nngp = integrate(fn)
      return nngp, ntk

    def nngp_fn_diag(nngp):
      xs, ws = quad_points
      x = xs.reshape((xs.shape[0],) + (1,) * nngp.ndim)
      x_axes = (0,)
      nngp = np.expand_dims(nngp, x_axes)
      fval = fn(_sqrt(2 * nngp, do_backprop) * x) ** 2
      return np.tensordot(ws, fval, (x_axes, x_axes)) / np.sqrt(np.pi)

    nngp, ntk = nngp_ntk_fn(nngp, q11, q22, ntk)

    if k.diagonal_batch and k.diagonal_spatial:
      cov1 = nngp_fn_diag(cov1)
      if cov2 is not None:
        cov2 = nngp_fn_diag(cov2)

    else:
      start_axis = 1 if k.diagonal_batch else 0
      q11 = utils.interleave_ones(d1, start_axis, end_axis, True)
      q22 = utils.interleave_ones(d1, start_axis, end_axis, False)
      cov1, _ = nngp_ntk_fn(cov1, q11, q22)

      if cov2 is not None:
        q11 = utils.interleave_ones(d2, start_axis, end_axis, True)
        q22 = utils.interleave_ones(d2, start_axis, end_axis, False)
        cov2, _ = nngp_ntk_fn(cov2, q11, q22)

    return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

  return _elementwise(fn, f'ElementwiseNumerical({fn},deg={deg})', kernel_fn)


# INTERNAL UTILITIES


_CONV_KERNEL_DIMENSION_NUMBERS = ('NCHW', 'OIHW', 'NCHW')


_INPUT_REQ = 'input_req'


_DEFAULT_INPUT_REQ = frozendict.frozendict({'diagonal_batch': True,
                                            'diagonal_spatial': False,
                                            'batch_axis': 0,
                                            'use_dropout': False,
                                            'channel_axis': -1,
                                            'mask_constant': None})


class _Bool(enum.IntEnum):
  """Helper trinary logic class."""
  NO = 0
  MAYBE = 1
  YES = 2

  def __and__(self, other: '_Bool') -> '_Bool':
    return min(self, other)

  __rand__ = __and__


@dataclasses.dataclass
class _Diagonal:
  """Helps decide whether to allow the kernel to contain diagonal entries only.

  The intended behavior is to be diagonal-only iff
    a) output off-diagonal entries are all zeros, and
    b) diagonal-only `Kernel` is sufficient for all steps of computation.

  Note that currently this parameter is shared between all parallel branches,
  even if this is excessive, and it is defined once for the whole network and
  does not change from layer to layer, even if it could be possible.

  Must be endowed with
    1) A commutative, associative, idempotent `AND` (`&`) operation,
      corresponding to combining requirements of two layers in parallel.
    2) An associative composition `>>` operation, corresponding to the
      requirement of a composition of two layers.

  Attributes:
    input:
      specifies whether inputs to given layer can contain only diagonal
      entries. `_Bool.YES` means "yes"; `_Bool.MAYBE` means iff off-diagonal
      entries are zero. `_Bool.NO` means "no". When traversing the network
      tree from inputs to outputs (as well as parallel branches from left/right
      to right/left) can only decrease.
    output:
      specifies whether any outputs (starting from this layer to the output of
      the network) can contain only diagonal entries. `_Bool.YES` means yes;
      `_Bool.MAYBE` means "yes" after current layer, but may become "no"
      further in the network. `_Bool.NO` means "no".
  """

  input: _Bool = _Bool.YES
  output: _Bool = _Bool.NO

  def __rshift__(self, other: '_Diagonal') -> '_Diagonal':
    """Associative composition (`self >> other`) operation.

    Returns the requirement satisfied by composition `other(self(.))`.
    """
    if self.output == _Bool.YES:
      return self

    if self.output > _Bool.NO and other.input > _Bool.NO:
      input = self.input
    elif self.output == _Bool.NO and other.input < _Bool.YES:
      input = _Bool.NO
    else:
      input = min(self.input, other.input)

    return _Diagonal(input=input, output=other.output)  # pytype:disable=wrong-keyword-args

  def __and__(self, other: '_Diagonal') -> '_Diagonal':
    """Commutative, associative, and idempotent `AND` operation.

    Returns the largest value allowed both `self` and `other`.
    """
    return _Diagonal(input=self.input & other.input,
                     output=self.output & other.output)  # pytype:disable=wrong-keyword-args

  def __bool__(self) -> bool:
    """Convert to `diagonal_spatial` / `diagonal_batch` `Kernel` attribute."""
    return self.input == _Bool.YES and self.output > _Bool.NO

  def __lshift__(self, other: '_Diagonal') -> '_Diagonal':
    """Associative composition (`self << other`) operation.

    Returns the value allowed by composition `self(other(.))`.
    """
    return other >> self

  __rand__ = __and__


def _get_input_req_attr(
    kernel_fns: List[LayerKernelFn],
    fold: Callable[[_Diagonal, _Diagonal], _Diagonal]) -> Dict[str, Any]:
  """Gets requirements of the combined layer based on individual requirements.

  Specifically, gets the requirements / allowances to the inputs to a `serial`
  or `parallel` sequence of layers based on requirements of each layer, setting
  requirements / allowances to the most / least demanding among all layers.

  Args:
    kernel_fns:
      list of `kernel_fn`s fed to the `kernel_fns` (e.g. a list of
      convolutional layers and nonlinearities to be chained together with the
      `serial` combinator) or evaluated in parallel (`parallel` combinator).
    fold:
      binary associative operator to combine allowances of consecutive
      individual `kernel_fn`s. Can be only `operator.rshift` (`>>`), i.e.
      composition (corresponding to `serial`) or `operator.and_`, (`&`), i.e.
      `AND` (corresponding to `parallel`).

  Returns:
    A `dict` with combined requirements / allowances.
  """
  req = {}

  for f in kernel_fns:
    req_f = getattr(f, _INPUT_REQ, {})

    for k, v in req_f.items():
      if k == 'use_dropout':
        if k in req and req[k] != v:
          raise ValueError('`use_dropout` is a single whole-network attribute '
                           'and cannot be set to different values.')
        req[k] = v

      elif k in ('batch_axis', 'channel_axis'):
        if k not in req:
          req[k] = v
        else:
          if fold is op.and_:
            if k in req and req[k] != v:
              if (req[k] >= 0 and v >= 0) or (req[k] < 0 and v < 0):
                warnings.warn(f'For `kernel_fn`, `{k}` parameters must match in'
                              f' all parallel branches, got {req[k]} and {v}. '
                              f'This WILL lead to [silent] errors if '
                              f'`kernel_fn` is called.')
              else:
                warnings.warn(f'Got potentially mismatching `{k}` values in '
                              f'parallel branches: {req[k]} and {v}.')

          elif fold is not op.rshift:
            raise ValueError(fold)

      elif k in ('diagonal_batch', 'diagonal_spatial'):
        if k in req:
          req[k] = fold(req[k], v)
        else:
          req[k] = v

      else:
        raise NotImplementedError(k)

  return req


_T = TypeVar('_T')


def _double_tuple(x: Iterable[_T]) -> Tuple[_T, ...]:
  return tuple(v for v in x for _ in range(2))


def _cov_diag_batch_diag_spatial(x: np.ndarray,
                                 batch_axis: int,
                                 channel_axis: int) -> np.ndarray:
  ret = np.sum(x ** 2, axis=channel_axis)
  new_batch_axis = batch_axis - (1 if batch_axis > channel_axis else 0)
  ret = np.moveaxis(ret, new_batch_axis, 0)
  return ret


def _cov_diag_batch_full_spatial(x: np.ndarray,
                                 batch_axis: int,
                                 channel_axis: int) -> np.ndarray:
  ret = lax.dot_general(x, x,
                        (((channel_axis,), (channel_axis,)),
                         ((batch_axis,), (batch_axis,)))
                        )
  ret = utils.zip_axes(ret, 1)
  return ret


def _cov_full_batch_full_spatial(x1: np.ndarray,
                                 x2: np.ndarray,
                                 batch_axis: int,
                                 channel_axis: int) -> np.ndarray:
  ret = np.tensordot(x1, x2, (channel_axis, channel_axis))
  new_batch_axis = batch_axis - (1 if batch_axis > channel_axis else 0)
  ret = np.moveaxis(ret, (new_batch_axis, x1.ndim - 1 + new_batch_axis), (0, 1))
  ret = utils.zip_axes(ret, 2)
  return ret


def _cov_full_batch_diag_spatial(x1: np.ndarray,
                                 x2: np.ndarray,
                                 batch_axis: int,
                                 channel_axis: int) -> np.ndarray:
  diag_axes = tuple(i for i in range(x1.ndim)
                    if i != batch_axis and i != channel_axis)
  ret = lax.dot_general(x1, x2,
                        (((channel_axis,), (channel_axis,)),
                         (diag_axes, diag_axes))
                        )
  ret = np.moveaxis(ret, (-2, -1), (0, 1))
  return ret


def _cov_diag_batch(x: np.ndarray,
                    diagonal_spatial: bool,
                    batch_axis: int,
                    channel_axis: int) -> np.ndarray:
  if diagonal_spatial:
    ret = _cov_diag_batch_diag_spatial(x, batch_axis, channel_axis)
  else:
    ret = _cov_diag_batch_full_spatial(x, batch_axis, channel_axis)
  return ret / x.shape[channel_axis]


def _cov(
    x1: np.ndarray,
    x2: Optional[np.ndarray],
    diagonal_spatial: bool,
    batch_axis: int,
    channel_axis: int) -> Optional[np.ndarray]:
  """Computes uncentered covariance (nngp) between two batches of inputs.

  Args:
    x1: a (2+S)D (S >= 0) `np.ndarray` of shape
      `(batch_size_1, <S spatial dimensions>, n_channels)`. `batch_size_1`,
      `n_channels` may be in different positions based on `batch_axis` and
      `channel_axis`.
    x2: an optional `np.ndarray` that has the same shape as `a` apart from
      possibly different batch (`batch_size_2`) dimension. `None` means
      `x2 == x1`.
    diagonal_spatial: Specifies whether only the diagonals of the
      location-location covariances will be computed,
      (`diagonal_spatial == True`,
       `nngp.shape == (batch_size_1, batch_size_2, height, width, depth, ...)`),
      or the full covariance
      (`diagonal_spatial == False`,
       `nngp.shape == (batch_size_1, batch_size_2, height, height,
                       width, width, depth, depth, ...)`).
    batch_axis: Specifies which axis is the batch axis.
    channel_axis: Specifies which axis is the channel / feature axis.
      For `kernel_fn`, channel size is considered to be infinite.
  Returns:
    Matrix of uncentred batch covariances with shape
    `(batch_size_1, batch_size_2, <S spatial dimensions>)`
    if `diagonal_spatial` is `True`, or
    `(batch_size_1, batch_size_2, <2*S spatial dimensions>)`
    if `diagonal_spatial` is `False`.
  """
  x2 = x1 if x2 is None else x2

  if diagonal_spatial:
    ret = _cov_full_batch_diag_spatial(x1, x2, batch_axis, channel_axis)

  else:
    ret = _cov_full_batch_full_spatial(x1, x2, batch_axis, channel_axis)

  return ret / x1.shape[channel_axis]


@utils.nt_tree_fn(2)
def _inputs_to_kernel(
    x1: np.ndarray,
    x2: Optional[np.ndarray],
    *,
    diagonal_batch: bool,
    diagonal_spatial: Union[bool, _Diagonal],
    compute_ntk: bool,
    batch_axis: int,
    channel_axis: Optional[int],
    mask_constant: Optional[float],
    eps: float = 1e-12,
    **kwargs
) -> Kernel:
  """Transforms (batches of) inputs to a `Kernel`.

  This is a private function. Docstring and example are for internal reference.

  The kernel contains the empirical covariances between different inputs and
  their entries (e.g. pixels, words, entries in a time series etc.) necessary
  to compute the covariance of the Gaussian Process corresponding to an
  infinite Bayesian or continuous gradient descent trained neural network.

  The smallest necessary number of covariance entries is tracked. For example,
  all networks are assumed to have i.i.d. weights along the channel / feature
  / logits dimensions, hence covariance between different entries along these
  dimensions is known to be 0 and is not tracked.

  Example:
    >>> x = np.ones((10, 32, 16, 3))
    >>> o = _inputs_to_kernel(x, None,
    >>>                       diagonal_batch=True,
    >>>                       diagonal_spatial=False,
    >>>                       compute_ntk=True,
    >>>                       batch_axis=0,
    >>>                       channel_axis=-1)
    >>> o.cov1.shape, o.ntk.shape
    (10, 32, 32, 16, 16), (10, 10, 32, 32, 16, 16)
    >>> o = _inputs_to_kernel(x, None,
    >>>                       diagonal_batch=True,
    >>>                       diagonal_spatial=True,
    >>>                       compute_ntk=True,
    >>>                       batch_axis=0,
    >>>                       channel_axis=-1)
    >>> o.cov1.shape, o.ntk.shape
    (10, 32, 16), (10, 10, 32, 16)
    >>> x1 = np.ones((10, 128))
    >>> x2 = np.ones((20, 128))
    >>> o = _inputs_to_kernel(x1, x2,
    >>>                       diagonal_batch=True,
    >>>                       diagonal_spatial=True,
    >>>                       compute_ntk=False,
    >>>                       batch_axis=0,
    >>>                       channel_axis=-1)
    >>> o.cov1.shape, o.nngp.shape
    (10,), (10, 20)

  Args:
    x1: an `(S+2)`-dimensional `np.ndarray` of shape
      `(batch_size_1, height, width, depth, ..., n_channels)` with `S` spatial
      dimensions (`S >= 0`). Dimensions may be in different order based on
      `batch_axis` and `channel_axis`.
    x2: an optional `np.ndarray` with the same shape as `x1` apart
      from possibly different batch size. `None` means `x2 == x1`.
    diagonal_batch: Specifies whether `cov1` and `cov2` store only
      the diagonal of the sample-sample covariance
      (`diagonal_batch == True`,
       `cov1.shape == (batch_size_1, ...)`),
      or the full covariance
      (`diagonal_batch == False`,
       `cov1.shape == (batch_size_1, batch_size_1, ...)`).
    diagonal_spatial: Specifies whether all (`cov1`, `ntk`, etc.)
      input covariance matrcies should store only the diagonals of the
      location-location covariances
      (`diagonal_spatial == True`,
       `nngp.shape == (batch_size_1, batch_size_2, height, width, depth, ...)`),
      or the full covariance
      (`diagonal_spatial == False`,
       `nngp.shape == (batch_size_1, batch_size_2, height, height,
                       width, width, depth, depth, ...)`).
    compute_ntk: `True` to compute both NTK and NNGP kernels,
      `False` to only compute NNGP.
    batch_axis: Specifies which axis is the batch axis.
    channel_axis: Specifies which axis is the channel / feature axis.
      For `kernel_fn`, channel size is considered to be infinite.
    mask_constant: an optional `float`, the value in inputs to be considered as
      masked (e.g. padding in a batch of sentences). `None` means no masking.
      Can also be `np.nan`, `np.inf` etc. Beware of floating point precision
      errors and try to use an atypical for inputs value.
    eps: a small number used to check whether x1 and x2 are the same up to
      `eps`.
    **kwargs: other arguments passed to all intermediary `kernel_fn` calls (not
      used here).

  Returns:
    The `Kernel` object containing inputs covariance[s].
  """

  if not (isinstance(x1, np.ndarray) and
          (x2 is None or isinstance(x2, np.ndarray))):
    raise TypeError(('Wrong input types given. Found `x1` of type '
                     f'{type(x1)} and `x2` of type {type(x2)}, need both to be'
                     f'`np.ndarray`s (`x2` can be `None`).'))

  batch_axis %= x1.ndim
  diagonal_spatial = bool(diagonal_spatial)

  if batch_axis != 0:
    # TODO(romann): add support or clear error for batching.
    warnings.warn(f'!!! Non-leading (!= 0) batch dimension in the '
                  f'input layer is not supported for batching '
                  f'kernels, got batch_axis = {batch_axis}. !!!')

  if channel_axis is None:
    def flatten(x):
      if x is None:
        return x
      return np.moveaxis(x, batch_axis, 0).reshape((x.shape[batch_axis], -1))

    x1, x2 = flatten(x1), flatten(x2)
    batch_axis, channel_axis = 0, 1
    diagonal_spatial = False

  else:
    channel_axis %= x1.ndim

  def get_x_cov_mask(x):
    if x is None:
      return None, None, None

    if x.ndim < 2:
      raise ValueError(f'Inputs must be at least 2D (a batch dimension and a '
                       f'channel/feature dimension), got {x.ndim}.')

    x = utils.get_masked_array(x, mask_constant)
    x, mask = x.masked_value, x.mask

    # TODO(schsam): Think more about dtype automatic vs manual dtype promotion.
    x = x.astype(np.float64)

    if diagonal_batch:
      cov = _cov_diag_batch(x, diagonal_spatial, batch_axis, channel_axis)
    else:
      cov = _cov(x, x, diagonal_spatial, batch_axis, channel_axis)

    return x, cov, mask

  x1, cov1, mask1 = get_x_cov_mask(x1)
  x2, cov2, mask2 = get_x_cov_mask(x2)
  nngp = _cov(x1, x2, diagonal_spatial, batch_axis, channel_axis)

  ntk = np.zeros((), nngp.dtype) if compute_ntk else None
  is_gaussian = False
  is_reversed = False
  x1_is_x2 = utils.x1_is_x2(x1, x2, eps=eps)
  is_input = False

  return Kernel(cov1=cov1,
                cov2=cov2,
                nngp=nngp,
                ntk=ntk,
                x1_is_x2=x1_is_x2,
                is_gaussian=is_gaussian,
                is_reversed=is_reversed,
                is_input=is_input,
                diagonal_batch=diagonal_batch,
                diagonal_spatial=diagonal_spatial,
                shape1=x1.shape,
                shape2=x1.shape if x2 is None else x2.shape,
                batch_axis=batch_axis,
                channel_axis=channel_axis,
                mask1=mask1,
                mask2=mask2)  # pytype:disable=wrong-keyword-args


def _propagate_shape(init_fn: InitFn,
                     apply_fn: ApplyFn,
                     shaped: ShapedArray,
                     **kwargs) -> ShapedArray:
  """Statically, abstractly, evaluate the init_fn to get shape information."""
  def init_and_apply(rng, x):
    _, params = init_fn(rng, tree_map(lambda x: x.shape, x))
    return apply_fn(params, x, rng=rng, **kwargs)
  akey = ShapedArray((2,), np.uint32)
  try:
    shaped = eval_shape(init_and_apply, akey, shaped)
  except NotImplementedError:
    # Some layers do not implement an `apply_fn` and in this case we keep the
    # shape constant.
    pass

  if isinstance(shaped, utils.MaskedArray):
    shaped = shaped.masked_value  # pytype: disable=attribute-error

  return shaped


def _set_shapes(init_fn: InitFn,
                apply_fn: ApplyFn,
                in_kernel: NTTree[Kernel],
                out_kernel: NTTree[Kernel],
                **kwargs
                ) -> NTTree[Kernel]:
  """Apply a kernel_fn to a Kernel propagating side information."""

  get_shape1_fn = utils.nt_tree_fn()(lambda k:
                                     ShapedArray(k.shape1, k.nngp.dtype))
  get_shape2_fn = utils.nt_tree_fn()(lambda k:
                                     ShapedArray(k.shape2, k.nngp.dtype))
  shape1 = get_shape1_fn(in_kernel)
  shape2 = get_shape2_fn(in_kernel)

  kwargs1, kwargs2 = utils.split_kwargs(kwargs)

  shape1 = _propagate_shape(init_fn, apply_fn, shape1, **kwargs1)
  shape2 = _propagate_shape(init_fn, apply_fn, shape2, **kwargs2)

  set_shape_fn = utils.nt_tree_fn()(
      lambda k, s1, s2: k.replace(shape1=s1.shape, shape2=s2.shape))

  return set_shape_fn(out_kernel, shape1, shape2)


def _fuse_requirements(kernel_fn_reqs, default_reqs, **user_reqs):
  # Override static requirements with explicit user-specified requirements,
  # but only if they are less demanding, raise an error otherwise.
  kernel_fn_reqs = dict(kernel_fn_reqs)
  for k, v_user in user_reqs.items():
    if v_user is not None:
      if k in kernel_fn_reqs:
        v_kernel = kernel_fn_reqs[k]
        if (v_user is True and
            (v_kernel is False or
             (isinstance(kernel_fn_reqs[k], _Diagonal) and
              kernel_fn_reqs[k].input == _Bool.NO))):
          raise ValueError(f'Asked to compute `kernel_fn` output with '
                           f'`{k} == {v_user}`, while `kernel_fn` '
                           f'requires `{k} == {kernel_fn_reqs[k]}`.')

      kernel_fn_reqs[k] = v_user

  # Fill unspecified requirements with defaults.
  for k, v_user in default_reqs.items():
    if k not in kernel_fn_reqs:
      kernel_fn_reqs[k] = v_user

  return frozendict.frozendict(kernel_fn_reqs)


def _preprocess_kernel_fn(
    init_fn: InitFn,
    apply_fn: ApplyFn,
    kernel_fn: LayerKernelFn) -> AnalyticKernelFn:
  """Returns a `kernel_fn` with additional arguments.

  Args:
    init_fn: layer parameters initialization function. Used for shape
      inference.
    apply_fn: layer forward-prop function. Used for shape inference.
    kernel_fn: the `Kernel` -> `Kernel` layer propagation function.

  Returns:
    A new `kernel_fn` that does the same computation but accepts additional
    arguments to flexibly specify the required computation, and can be applied
    to either a `Kernel' or a pair of `np.ndarrray`s.
  """
  # Set empty requirements if none specified.
  if not hasattr(kernel_fn, _INPUT_REQ):
    kernel_fn = _requires()(kernel_fn)

  def kernel_fn_kernel(kernel, **kwargs):
    out_kernel = kernel_fn(kernel, **kwargs)
    return _set_shapes(init_fn, apply_fn, kernel, out_kernel, **kwargs)

  def kernel_fn_x1(x1, x2, get, **kwargs):
    # Get input requirements requested by network layers, user, or defaults.
    kernel_fn_reqs = getattr(kernel_fn, _INPUT_REQ)
    reqs = _fuse_requirements(kernel_fn_reqs, _DEFAULT_INPUT_REQ, **kwargs)
    compute_ntk = (get is None) or ('ntk' in get)

    if x2 is None:
      x2 = tree_map(lambda x: None, x1)
    kernel = _inputs_to_kernel(x1, x2, compute_ntk=compute_ntk, **reqs)
    out_kernel = kernel_fn(kernel, **kwargs)
    return _set_shapes(init_fn, apply_fn, kernel, out_kernel, **kwargs)

  @utils.get_namedtuple('AnalyticKernel')
  def kernel_fn_any(x1_or_kernel: Union[NTTree[np.ndarray], NTTree[Kernel]],
                    x2: NTTree[np.ndarray] = None,
                    get: Get = None,
                    *,
                    pattern: Tuple[Optional[np.ndarray],
                                   Optional[np.ndarray]] = None,
                    mask_constant: float = None,
                    diagonal_batch: bool = None,
                    diagonal_spatial: bool = None,
                    **kwargs):
    """Returns the `Kernel` resulting from applying `kernel_fn` to given inputs.

    Args:
      x1_or_kernel:
        either an NTTree of the first batch of inputs.
      x2:
        an optional NTTree of `np.ndarray` with the second batch of inputs.
        `None` means `x2 == x1` or `x1_or_kernel is Kernel`.
      get:
        either `None`, a string, or a tuple of strings specifying which data
        should be returned by the kernel function. Can be "nngp", "ntk", "cov1",
        "cov2", "is_gaussian", "is_reversed", "diagonal_batch",
        "diagonal_spatial", etc.
      pattern:
        either `None` or a tuple of two `np.ndarray`. The
        `pattern = (pattern1, pattern2)` is used to specify how the nodes in a
        graphical network is aggregated.
      mask_constant:
        an optional `float`, the value in inputs to be considered
        as masked (e.g. padding in a batch of sentences). `None` means no
        masking. Can also be `np.nan`, `np.inf` etc. Beware of floating point
        precision errors and try to use an atypical for inputs value.
      diagonal_batch:
        an optional boolean specifying whether `cov1` and `cov2` in all
        intermediary layers should store only the diagonal of the
        sample-sample covariance
        (`diagonal_batch == True`,
         `cov1.shape == (batch_size_1, ...)`),
        or the full covariance
        (`diagonal_batch == False`,
         `cov1.shape == (batch_size_1, batch_size_1, ...)`).
        Defaults to least compute-heavy setting necessary to compute the output
        `nngp` [and `ntk`] covariance.
      diagonal_spatial:
        an optional boolean specifying whether all (`cov1`, `ntk`, etc.)
        covariance matrcies in all intermediary layers should store only the
        diagonals of the location-location covariances
        (`diagonal_spatial == True`,
         `nngp.shape == (batch_size_1, batch_size_2, height, width, ...)`),
        or the full covariance
        (`diagonal_spatial == False`,
         `nngp.shape == (batch_size_1, batch_size_2, height, height,
                         width, width, ...)`).
        Defaults to least compute-heavy setting necessary to compute the output
        `nngp` [and `ntk`] covariance.
      **kwargs:
        other arguments passed to all intermediary `kernel_fn` calls.

    Returns:
      If `get` is a string, returns the requested `np.ndarray`. If `get` is a
      tuple, returns an `AnalyticKernel` namedtuple containing only the
      requested information. If `get` is `None` then a `Kernel` object is
      returned containing all the data.
    """
    if utils.is_nt_tree_of(x1_or_kernel, Kernel):
      return kernel_fn_kernel(x1_or_kernel,
                              pattern=pattern,
                              diagonal_batch=diagonal_batch,
                              diagonal_spatial=diagonal_spatial,
                              **kwargs)

    return kernel_fn_x1(x1_or_kernel, x2, get,
                        pattern=pattern,
                        diagonal_batch=diagonal_batch,
                        diagonal_spatial=diagonal_spatial,
                        mask_constant=mask_constant,
                        **kwargs)

  setattr(kernel_fn_any, _INPUT_REQ, getattr(kernel_fn, _INPUT_REQ))
  return kernel_fn_any


def _elementwise(fn: Optional[Callable[[float], float]],
                 name: str,
                 kernel_fn: Optional[LayerKernelFn]) -> InternalLayer:
  init_fn = lambda rng, input_shape: (input_shape, ())

  def apply_fn(params, inputs, **kwargs):
    if fn is None:
      raise NotImplementedError(fn)
    return fn(inputs)  # pytype:disable=not-callable

  def new_kernel_fn(k: Kernel, **kwargs) -> Kernel:
    if kernel_fn is None:
      raise NotImplementedError(kernel_fn)

    if not k.is_gaussian:
      raise ValueError('The input to the activation function must be Gaussian, '
                       'i.e. a random affine transform is required before the '
                       'activation function.')
    k = kernel_fn(k)  # pytype:disable=not-callable
    return k.replace(is_gaussian=False)

  init_fn.__name__ = apply_fn.__name__ = new_kernel_fn.__name__ = name
  return init_fn, apply_fn, new_kernel_fn


def _arccos(x, do_backprop):
  if do_backprop:
    # https://github.com/google/jax/issues/654
    x = np.where(np.abs(x) >= 1, np.sign(x), x)
  else:
    x = np.clip(x, -1, 1)
  return np.arccos(x)


def _sqrt(x, do_backprop):
  if do_backprop:
    # https://github.com/google/jax/issues/654
    x = np.where(x <= 0, 0, x)
  else:
    x = np.maximum(x, 0)
  return np.sqrt(x)


def _safe_sqrt(x):
  return np.sqrt(np.maximum(x, 1e-20))


def _arcsin(x, do_backprop):
  if do_backprop:
    # https://github.com/google/jax/issues/654
    x = np.where(np.abs(x) >= 1, np.sign(x), x)
  else:
    x = np.clip(x, -1, 1)
  return np.arcsin(x)


def _get_diagonal(
    cov: Optional[np.ndarray],
    diagonal_batch: bool,
    diagonal_spatial: bool) -> Optional[np.ndarray]:
  """Extracts the diagonal of `cov` over all (sample, spatial) dimensions.

  Adapts computation if `cov` already stores only the diagonal along some
  dimensions based on `diagonal_batch` and `diagonal_spatial`.
  """
  if cov is None:
    return cov

  batch_ndim = 1 if diagonal_batch else 2
  start_axis = 2 - batch_ndim
  end_axis = batch_ndim if diagonal_spatial else cov.ndim
  cov = utils.unzip_axes(cov, start_axis, end_axis)
  return utils.diagonal_between(cov, start_axis, end_axis)


def _get_diagonal_outer_prods(cov1: np.ndarray,
                              cov2: Optional[np.ndarray],
                              diagonal_batch: bool,
                              diagonal_spatial: bool,
                              operation: Callable[[float, float], float],
                              axis: Sequence[int] = (),
                              mask1: Optional[np.ndarray] = None,
                              mask2: Optional[np.ndarray] = None
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Gets outer products of diagonals `cov1, cov1`, `cov1, cov2`, `cov2, cov2`.

  `prod11[x1, x2, h1, h2, ...]` =
   cov1[x1, [x1,], h1, [h1,], ...] * cov1[x2, [x2,], h2, [h2,], ...]`,
  `prod12[x1, x2, h1, h2, ...]` =
   cov1[x1, [x1,], h1, [h1,], ...] * cov2[x2, [x2,], h2, [h2,], ...]`,
  `prod22[x1, x2, h1, h2, ...]` =
   cov2[x1, [x1,], h1, [h1,], ...] * cov2[x2, [x2,], h2, [h2,], ...]`.

  Exact shapes of `cov1` and `cov2` are defined by `diagonal_batch` and
    `diagonal_spatial`.
  """
  axis = utils.canonicalize_axis(axis, cov1)

  cov1 = _get_diagonal(cov1, diagonal_batch, diagonal_spatial)
  cov2 = _get_diagonal(cov2, diagonal_batch, diagonal_spatial)

  cov1, _ = _mean_and_var(cov1, axis=axis, keepdims=True, mask=mask1)
  cov2, _ = _mean_and_var(cov2, axis=axis, keepdims=True, mask=mask2)

  end_axis = 1 if diagonal_spatial else cov1.ndim
  prod12 = utils.outer_prod(cov1, cov2, 0, end_axis, operation)

  start_axis = 1 if diagonal_batch else 0
  prod11 = utils.outer_prod(cov1, cov1, start_axis, end_axis, operation)
  prod22 = (utils.outer_prod(cov2, cov2, start_axis, end_axis, operation)
            if cov2 is not None else prod11)

  return prod11, prod12, prod22


def _affine(
    mat: Optional[np.ndarray],
    W_std: float,
    b_std: float) -> Optional[np.ndarray]:
  """Get covariances of affine outputs if inputs have covariances `nngp`.

  The output is assumed to be `xW + b`, where `x` is the input, `W` is a matrix
  of i.i.d. Gaussian weights with std `W_std`, `b` is a vector of i.i.d.
  Gaussian biases with std `b_std`.

  Args:
    mat:
      a `np.ndarray` containing sample-[sample-]position[-position] covariances
      of inputs.
    W_std:
      standard deviation of a fully-connected layer weights.
    b_std:
      standard deviation of a fully-connected layer biases.

  Returns:
    a `np.ndarray` containing sample-[sample-]position[-position] covariances
    of FC outputs. Has the same shape as `nngp`.
  """
  if mat is None:
    return mat

  return  W_std**2 * mat + b_std**2


def _proprocess_kernels_for_fan_in(ks: Kernels) -> Tuple[List[Kernel], bool]:
  # Check diagonal requirements.
  if not all(k.diagonal_batch == ks[0].diagonal_batch and
             k.diagonal_spatial == ks[0].diagonal_spatial and
             k.batch_axis == ks[0].batch_axis and
             k.channel_axis == ks[0].channel_axis
             for k in ks[1:]):
    raise NotImplementedError('`FanIn` layers are only implemented for the '
                              'case if all input layers output the same layout '
                              'of covariance matrices, i.e. having all '
                              'matching `diagonal_batch` and '
                              '`diagonal_spatial` and other attributes.')

  # If kernels have different spatial axes order, transpose some of them.
  n_kernels = len(ks)
  n_reversed = sum(ker.is_reversed for ker in ks)
  ks = list(ks)

  if n_reversed > n_kernels / 2:
    is_reversed = True
    for i in range(n_kernels):
      if not ks[i].is_reversed:
        ks[i] = ks[i].reverse()

  else:
    is_reversed = False
    for i in range(n_kernels):
      if ks[i].is_reversed:
        ks[i] = ks[i].reverse()

  # Warnings.
  warnings.warn('`FanIn` layers assume independent inputs which is not verified'
                ' in the code. Please make sure to have at least one `Dense` / '
                '`Conv` / `GlobalSelfAttention` etc. layer in each branch.')

  return ks, is_reversed


def _concat_kernels(
    mats: Sequence[Optional[np.ndarray]],
    axis: int,
    diagonal_batch: bool,
    diagonal_spatial: bool,
    widths: Sequence[int]) -> Optional[np.ndarray]:
  """Compute the covariance of concatenated activations with given covariances.

  Args:
    mats: Covariance tensors of the same shape.
    axis: Specifies the axis along which the covariances (not activations) are
      concatenated. `-1` corresponds to averaging.
    diagonal_batch: Specifies whether `cov1` and `cov2` store only
      the diagonal of the sample-sample covariance
      (`diagonal_batch == True`,
       `cov1.shape == (batch_size_1, ...)`),
      or the full covariance
      (`diagonal_batch == False`,
       `cov1.shape == (batch_size_1, batch_size_1, ...)`).
    diagonal_spatial: Specifies whether only the diagonals of the
      location-location covariances will be computed,
      (`diagonal_spatial == True`,
       `nngp.shape == (batch_size_1, batch_size_2, height, width, depth, ...)`),
      or the full covariance
      (`diagonal_spatial == False`,
       `nngp.shape == (batch_size_1, batch_size_2, height, height,
                       width, width, depth, depth, ...)`).
    widths: list of integer channel widths of the finite model inputs.
  Returns:
    A new `np.ndarray` representing covariance between concatenated activations.
  """
  if mats[0] is None:
    return None

  n_mats = len(mats)
  mat_ndim = mats[0].ndim

  # Averaging if concatenating along features or diagonalized dimension.
  if axis == -1:
    if all(w == widths[0] for w in widths):
      widths = [1] * len(widths)
    mat = sum(mats[i] * widths[i] for i in range(n_mats)) / sum(widths)

  # Simple concatenation along the axis if the axis is not duplicated.
  elif ((axis == 0 and diagonal_batch) or
        (axis != 0 and diagonal_spatial)):
    concat_axis = axis + (0 if diagonal_batch else 1)
    mat = np.concatenate(mats, concat_axis)

  # 2D concatenation with insertion of 0-blocks if the axis is present twice.
  else:
    rows = []
    pad_axis = max(0, 2 * axis - (1 if diagonal_batch else 0))
    for i, mat in enumerate(mats):
      pads = [(0, 0)] * mat_ndim
      pads[pad_axis] = (
          sum(mats[j].shape[pad_axis] for j in range(i)),
          sum(mats[j].shape[pad_axis] for j in range(i + 1, n_mats))
      )
      rows.append(np.pad(mat, pads))
    mat = np.concatenate(rows, pad_axis + 1)

  return mat


def _same_pad_for_filter_shape(
    x: np.ndarray,
    filter_shape: Sequence[int],
    strides: Sequence[int],
    axes: Sequence[int],
    mode: str = 'wrap',
) -> np.ndarray:
  """Pad an array to imitate `SAME` padding with `VALID`.

  See `Returns` section for details. This function is usually needed to
    implement `CIRCULAR` padding using `VALID` padding.

  Args:
    x: `np.ndarray` to pad, e.g. a 4D `NHWC` image.
    filter_shape: tuple of positive integers, the convolutional filters spatial
      shape (e.g. `(3, 3)` for a 2D convolution).
    strides: tuple of positive integers, the convolutional spatial strides, e.g.
      e.g. `(1, 1)` for a 2D convolution.
    axes: tuple of non-negative integers, the spatial axes to apply
      convolution over (e.g. `(1, 2)` for an `NHWC` image).
    mode: a string, padding mode, for all options see
      https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html.
  Returns:
    A `np.ndarray` of the same dimensionality as `x` padded to a potentially
    larger shape such that a `"VALID"` convolution with `filter_shape` applied
    to `x` over `axes` outputs an array of the same shape as `x`.
  """
  axes_shape = tuple(np.size(x, axis) for axis in axes)
  axes_pads = lax.padtype_to_pads(axes_shape, filter_shape, strides,
                                  Padding.SAME.name)
  pads = [(0, 0)] * x.ndim
  for i, axis in enumerate(axes):
    pads[axis] = axes_pads[i]
  x = np.pad(x, pads, mode)
  return x


def _same_pad_for_filter_shape_transpose(
    x: np.ndarray,
    axes: Sequence[int],
    out_shape: Sequence[int]
) -> np.ndarray:
  """Transpose of the `_same_pad_for_filter_shape` function.

  Unpads (crops) the array and fills each coordinate with the sum of all
  elements at positions where the current element would appear during
  `CIRCULAR` padding.

  Args:
    x:
      `np.ndarray` to pad, e.g. a 4D `NHWC` image.
    axes:
      non-negative integers, the spatial axes to apply convolution
      over (e.g. `(1, 2)` for an `NHWC` image).
    out_shape:
      target shape after cropping.

  Returns:
    A `np.ndarray` of shape `output_shape`.
  """
  window_dimensions = tuple(
      int(onp.ceil(x.shape[i] / out_shape[i])) // 2 * 2 + 1
      if i in axes else 1 for i in range(x.ndim))

  dilation = tuple(out_shape[i] if i in axes else 1 for i in range(x.ndim))

  x = lax.reduce_window(
      operand=x,
      init_value=onp.zeros((), x.dtype),
      computation=lax.add,
      window_dimensions=window_dimensions,
      window_strides=(1,) * x.ndim,
      padding=Padding.SAME.name,
      window_dilation=dilation
  )

  if x.shape != out_shape:
    pads = [((x.shape[i] - out_shape[i]) // 2,
             (x.shape[i] - out_shape[i]) - (x.shape[i] - out_shape[i]) // 2)
            for i in range(x.ndim)]
    slices = []
    for axis in range(x.ndim):
      if axis in axes:
        slices += [slice(pads[axis][0], x.shape[axis] - pads[axis][1])]
      else:
        slices += [slice(None)]
    x = x[tuple(slices)]
  return x


def _pool_transpose(
    x: np.ndarray,
    filter_shape: Sequence[int],
    strides: Sequence[int],
    axes: Sequence[int],
    padding: Padding
) -> np.ndarray:
  """Transpose convolution with an all-ones filter."""
  n_spatial = len(axes)
  x = np.moveaxis(x, axes, range(-n_spatial, 0))
  split = -n_spatial or x.ndim
  x_preshape = x.shape[:split]
  x = x.reshape((-1, 1) + x.shape[split:])
  rhs = np.ones(tuple(filter_shape) + (1, 1), x.dtype)
  x = lax.conv_transpose(x,
                         rhs,
                         strides,
                         padding.name,
                         dimension_numbers=_get_dimension_numbers(n_spatial))
  x = x.reshape(x_preshape + x.shape[2:])
  x = np.moveaxis(x, range(-n_spatial, 0), axes)
  return x


def _get_dimension_numbers(
    n: int,
    channels_first: bool = True
) -> Tuple[str, str, str]:
  spatial_dims = ''.join(c for c in string.ascii_uppercase
                         if c not in ('N', 'C', 'I', 'O'))[:n]
  if channels_first:
    lhs_spec = 'NC' + spatial_dims
  else:
    lhs_spec = 'N' + spatial_dims + 'C'
  dimension_numbers = (lhs_spec, spatial_dims + 'IO', lhs_spec)
  return dimension_numbers


def _conv_kernel_full_spatial_shared(
    lhs: Optional[np.ndarray],
    filter_shape: Sequence[int],
    strides: Sequence[int],
    padding: Padding,
    batch_ndim: int
) -> Optional[np.ndarray]:
  """Compute covariance of the CNN outputs given inputs with covariance `lhs`.

  Used when `kernel.diagonal_spatial == False`.

  Args:
    lhs:
      a `(2*S+batch_ndim)`-dimensional `np.ndarray` containing
      sample-[sample-]position-position covariances of CNN inputs, where `S` is
      the number of spatial dimensions (e.g. 2 for images). Has shape
      `(batch_size_1, [batch_size_2,] height, height, width, width, depth,
      depth, ...)`.
    filter_shape:
      positive integers, the convolutional filters spatial shape
      (e.g. `(3, 3)` for a 2D convolution).
    strides:
      positive integers, the CNN strides (e.g. `(1, 1)` for a 2D
      convolution).
    padding:
      a `Padding` enum, e.g. `Padding.CIRCULAR`.
    batch_ndim:
      number of batch dimensions, 1 or 2.

  Returns:
    a `(2*S+batch_ndim)`-dimensional `np.ndarray` containing
    sample-[sample-]position-position covariances of CNN outputs, where `S` is
    the number of spatial dimensions (e.g. 2 for images). Has shape
    `(batch_size_1, [batch_size_2,] new_width, new_width, new_height,
    new_height, new_depth, new_depth, ...)`.
  """
  if lhs is None or lhs.ndim == 0:
    return lhs

  if padding == Padding.CIRCULAR:
    spatial_axes = tuple(range(batch_ndim, lhs.ndim))
    total_filter_shape = _double_tuple(filter_shape)
    total_strides = _double_tuple(strides)
    lhs = _same_pad_for_filter_shape(lhs,
                                     total_filter_shape,
                                     total_strides,
                                     spatial_axes)

  def lax_conv(lhs, rhs, strides, padding):
    return lax.conv_general_dilated(
        lhs, rhs, strides, padding,
        dimension_numbers=_CONV_KERNEL_DIMENSION_NUMBERS,
        feature_group_count=lhs.shape[
            _CONV_KERNEL_DIMENSION_NUMBERS[0].index('C')])

  def get_n_channels(batch_and_channels: int) -> int:
    """Get the hardware-friendly channel size for depthwise convolution.

    Args:
      batch_and_channels: total size of non-spatial dimensions.

    Returns:
      Suggested number of channels for depthwise-separable convolution.
    """
    platform = xla_bridge.get_backend().platform
    if platform in ['gpu', 'tpu']:
      n_channels = batch_and_channels

      # Find smallest `n_channels > 1` that divides `batch_and_features`; use
      # depthwise-separable CNN. For `n_channels == 1` CuDNN appears to invoke a
      # different algorithm (`void cudnn::detail::implicit_convolve_sgemm`) than
      # in any other case (`conv2d_c1_k1_nchw_hw_packed_kernel`), and the latter
      # seems many-fold faster.
      # For TPU, start with `n_channels >= 128`. Beware of precision errors:
      # TODO(romann): revisit based on b/154160868.
      n_channels_min = 2 if platform == 'gpu' else 128

      for n_c in range(n_channels_min, batch_and_channels):
        if batch_and_channels % n_c == 0:
          n_channels = n_c
          break

    elif platform == 'cpu':
      # For CPU minimal channels seems best. Transpose convolution does not
      # support depthwise operations.
      n_channels = 1

    else:
      raise NotImplementedError(platform)
    return n_channels

  out = _conv_kernel_full_spatial_loop(lhs, filter_shape, strides, padding,
                                       lax_conv, get_n_channels)
  return out


def _conv_kernel_full_spatial_unshared(
    lhs: Optional[np.ndarray],
    filter_shape: Sequence[int],
    strides: Sequence[int],
    padding: Padding,
    batch_ndim: int,
) -> Optional[np.ndarray]:
  """Compute covariance of unshared CNN given inputs with covariance `lhs`.

  Used when `kernel.diagonal_spatial == False`. Has the same outputs on the
  spatial diagonal as `_conv_kernel_full_spatial_shared`, but `0` in all
  off-spatial-diagonal entries. The diagonal entries are computed via calling
  ``_conv_kernel_diagonal_spatial`.

  Args:
    lhs:
      a `(2*S+batch_ndim)`-dimensional `np.ndarray` containing
      sample-[sample-]position-position covariances of CNN inputs, where `S` is
      the number of spatial dimensions (e.g. 2 for images). Has shape
      `(batch_size_1, [batch_size_2,] height, height, width, width, depth,
      depth, ...)`.
    filter_shape:
      positive integers, the convolutional filters spatial shape
      (e.g. `(3, 3)` for a 2D convolution).
    strides:
      positive integers, the CNN strides (e.g. `(1, 1)` for a 2D
      convolution).
    padding:
      a `Padding` enum, e.g. `Padding.CIRCULAR`.
    batch_ndim:
      number of batch dimensions, 1 or 2.

  Returns:
    a `(2*S+batch_ndim)`-dimensional `np.ndarray` containing
    sample-[sample-]position-position covariances of CNN outputs, where `S` is
    the number of spatial dimensions (e.g. 2 for images). Has shape
    `(batch_size_1, [batch_size_2,] new_width, new_width, new_height,
    new_height, new_depth, new_depth, ...)`.
  """
  if lhs is None or lhs.ndim == 0:
    return lhs

  lhs = utils.unzip_axes(lhs, batch_ndim)
  lhs_diag = utils.diagonal_between(lhs, batch_ndim)
  out_diag = _conv_kernel_diagonal_spatial(lhs_diag, filter_shape, strides,
                                           padding, batch_ndim)
  out_diag_flat = out_diag.reshape((onp.prod(out_diag.shape[:batch_ndim]), -1))
  out_flat = vmap(np.diag)(out_diag_flat)
  out = out_flat.reshape(out_diag.shape[:batch_ndim] +
                         out_diag.shape[batch_ndim:] * 2)
  out = utils.zip_axes(out, batch_ndim)
  return out


def _conv_kernel_full_spatial_transpose(
    lhs: Optional[np.ndarray],
    filter_shape: Sequence[int],
    strides: Sequence[int],
    padding: Padding,
    batch_ndim: int
) -> Optional[np.ndarray]:
  """Compute covariance of the CNN transpose given inputs with covariance `lhs`.

  Used when `kernel.diagonal_spatial == False`.

  Args:
    lhs:
      a `(2*S+batch_ndim)`-dimensional `np.ndarray` containing
      sample-[sample-]position-position covariances of CNN inputs, where `S` is
      the number of spatial dimensions (e.g. 2 for images). Has shape
      `(batch_size_1, [batch_size_2,] height, height, width, width, depth,
      depth, ...)`.
    filter_shape:
      positive integers, the convolutional filters spatial shape
      (e.g. `(3, 3)` for a 2D convolution).
    strides:
      positive integers, the CNN strides (e.g. `(1, 1)` for a 2D
      convolution).
    padding:
      a `Padding` enum, e.g. `Padding.CIRCULAR`.
    batch_ndim:
      number of batch dimensions, 1 or 2.

  Returns:
    a `(2*S+batch_ndim)`-dimensional `np.ndarray` containing
    sample-[sample-]position-position covariances of CNN outputs, where `S` is
    the number of spatial dimensions (e.g. 2 for images). Has shape
    `(batch_size_1, [batch_size_2,] new_width, new_width, new_height,
    new_height, new_depth, new_depth, ...)`.
  """
  if lhs is None or lhs.ndim == 0:
    return lhs

  def lax_conv(lhs, rhs, strides, padding):
    return lax.conv_transpose(
        lhs, rhs, strides, padding,
        dimension_numbers=_CONV_KERNEL_DIMENSION_NUMBERS)

  def get_n_channels(batch_and_channels: int) -> int:
    """Transpose convolution does not support depthwise separable filters."""
    return 1

  out = _conv_kernel_full_spatial_loop(lhs, filter_shape, strides, padding,
                                       lax_conv, get_n_channels)

  if padding == Padding.CIRCULAR:
    spatial_axes = tuple(range(batch_ndim, out.ndim))
    total_filter_shape = _double_tuple(filter_shape)
    total_strides = _double_tuple(strides)
    out_shape = eval_shape(lambda x: _pool_transpose(x,
                                                     total_filter_shape,
                                                     total_strides,
                                                     spatial_axes,
                                                     Padding.SAME), lhs).shape
    out = _same_pad_for_filter_shape_transpose(
        x=out,
        axes=spatial_axes,
        out_shape=utils.reverse_zipped(out_shape, batch_ndim))
  return out


def _conv_kernel_full_spatial_loop(
    lhs: np.ndarray,
    filter_shape: Sequence[int],
    strides: Sequence[int],
    padding: Padding,
    lax_conv: Callable,
    get_n_channels: Callable[[int], int]
) -> np.ndarray:
  padding = Padding.VALID if padding == Padding.CIRCULAR else padding

  def get_rhs(n_channels: int, filter_size: int) -> np.ndarray:
    rhs = np.diag(np.full((filter_size,), 1. / filter_size, lhs.dtype))
    rhs_shape = ()
    for c in _CONV_KERNEL_DIMENSION_NUMBERS[1]:
      if c == 'O':
        rhs_shape += (n_channels,)
      elif c == 'I':
        rhs_shape += (1,)
      else:
        rhs_shape += (filter_size,)
    rhs = np.broadcast_to(rhs, rhs_shape)
    return rhs

  batch_ndim = lhs.ndim - len(filter_shape) * 2
  for i in range(lhs.ndim - 1, batch_ndim, -2):
    spatial_i = (i - batch_ndim) // 2

    lhs = np.moveaxis(lhs, (i - 1, i), (-2, -1))
    preshape = lhs.shape[:-2]
    n_channels = get_n_channels(utils.size_at(preshape))
    lhs = lhs.reshape((-1, n_channels, lhs.shape[-2], lhs.shape[-1]))

    rhs = get_rhs(n_channels, filter_shape[spatial_i])
    lhs = lax_conv(lhs, rhs, (strides[spatial_i],) * 2, padding.name)
    lhs = lhs.reshape(preshape + lhs.shape[-2:])

  return lhs


def _conv_kernel_diagonal_spatial(
    lhs: Optional[np.ndarray],
    filter_shape: Sequence[int],
    strides: Sequence[int],
    padding: Padding,
    batch_ndim: int
) -> Optional[np.ndarray]:
  """Compute covariance of the CNN outputs given inputs with covariance `lhs`.

  Used when `kernel.diagonal_spatial == True`.

  Args:
    lhs:
      an `(S+batch_ndim)`-dimensional `np.ndarray` containing
      sample-sample-(same position) covariances of CNN inputs. Has `batch_ndim`
      batch and `S` spatial dimensions with the shape of `(batch_size_1,
      [batch_size_2,] height, width, depth, ...)`.
    filter_shape:
      tuple of positive integers, the convolutional filters spatial shape
      (e.g. `(3, 3)` for a 2D convolution).
    strides:
      tuple of positive integers, the CNN strides (e.g. `(1, 1)` for a 2D
      convolution).
    padding:
      a `Padding` enum, e.g. `Padding.CIRCULAR`.
    batch_ndim:
      number of leading batch dimensions, 1 or 2.

  Returns:
    an `(S+batch_ndim)`-dimensional `np.ndarray` containing
    sample-sample-(same position) covariances of CNN outputs. Has `batch_ndim`
    batch and `S` spatial dimensions with the shape of `(batch_size_1,
    [batch_size_2,] new_height, new_width, new_depth, ...)`.
  """
  if lhs is None or lhs.ndim == 0:
    return lhs

  spatial_axes = tuple(range(batch_ndim, lhs.ndim))
  apply_padding = Padding.VALID if padding == Padding.CIRCULAR else padding

  if padding == Padding.CIRCULAR:
    lhs = _same_pad_for_filter_shape(lhs, filter_shape, strides, spatial_axes)

  lhs = lax.reduce_window(
      operand=lhs,
      init_value=onp.zeros((), lhs.dtype),
      computation=lax.add,
      window_dimensions=(1,) * batch_ndim + tuple(filter_shape),
      window_strides=(1,) * batch_ndim + tuple(strides),
      padding=apply_padding.name)

  filter_size = functools.reduce(op.mul, filter_shape, 1)
  return lhs / filter_size


def _conv_kernel_diagonal_spatial_transpose(
    lhs: Optional[np.ndarray],
    filter_shape: Sequence[int],
    strides: Sequence[int],
    padding: Padding,
    batch_ndim: int
) -> Optional[np.ndarray]:
  """Compute covariance of the CNN transpose given inputs with covariance `lhs`.

  Used when `kernel.diagonal_spatial == True`.

  Args:
    lhs:
      an `(S+batch_ndim)`-dimensional `np.ndarray` containing
      sample-sample-(same position) covariances of CNN inputs. Has `batch_ndim`
      batch and `S` spatial dimensions with the shape of `(batch_size_1,
      [batch_size_2,] height, width, depth, ...)`.
    filter_shape:
      tuple of positive integers, the convolutional filters spatial shape
      (e.g. `(3, 3)` for a 2D convolution).
    strides:
      tuple of positive integers, the CNN strides (e.g. `(1, 1)` for a 2D
      convolution).
    padding:
      a `Padding` enum, e.g. `Padding.CIRCULAR`.
    batch_ndim:
      number of leading batch dimensions, 1 or 2.

  Returns:
    an `(S+batch_ndim)`-dimensional `np.ndarray` containing
    sample-sample-(same position) covariances of CNN outputs. Has `batch_ndim`
    batch and `S` spatial dimensions with the shape of `(batch_size_1,
    [batch_size_2,] new_height, new_width, new_depth, ...)`.
  """
  if lhs is None or lhs.ndim == 0:
    return lhs

  spatial_axes = tuple(range(batch_ndim, lhs.ndim))
  apply_padding = Padding.VALID if padding == Padding.CIRCULAR else padding

  out = _pool_transpose(lhs, filter_shape, strides, spatial_axes, apply_padding)

  if padding == Padding.CIRCULAR:
    out_shape = eval_shape(lambda x: _pool_transpose(
        x,
        filter_shape,
        strides,
        spatial_axes,
        padding.SAME), lhs).shape
    out = _same_pad_for_filter_shape_transpose(out, spatial_axes, out_shape)

  filter_size = functools.reduce(op.mul, filter_shape, 1)
  return out / filter_size


def _pool_kernel(
    lhs: np.ndarray,
    pool_type: Pooling,
    window_shape: Sequence[int],
    strides: Sequence[int],
    padding: Padding,
    normalize_edges: bool,
    batch_ndim: int
) -> np.ndarray:
  """Get covariances of pooling outputs given inputs covariances `lhs`.

  Args:
    lhs:
      a `(2*S+batch_ndim)`-dimensional `np.ndarray` containing
      sample-[sample-]position-position covariances of pooling inputs, where `S`
      is the number of spatial dimensions (e.g. 2 for images). Has shape
      `(batch_size_1, [batch_size_2,]
        height, height, width, width, depth, depth, ...)`.
    pool_type:
      a `Pooling` enum, e.g. `Pooling.AVG`.
    window_shape:
      tuple of positive integers, the pooling spatial shape (e.g. `(3, 3)`).
    strides:
      tuple of positive integers, the pooling strides, e.g. `(1, 1)`.
    padding:
      a `Padding` enum, e.g. `Padding.CIRCULAR`.
    normalize_edges:
      `True` to normalize output by the effective receptive field, `False` to
      normalize by the window size. Only has effect at the edges when `SAME`
      padding is used. Set to `True` to retain correspondence to
      `ostax.AvgPool`.
    batch_ndim:
      number of leading batch dimensions, 1 or 2.

  Returns:
      a `(2*S+batch_ndim)`-dimensional `np.ndarray` containing
      sample-[sample-]position-position covariances of pooling outputs, where
      `S` is the number of spatial dimensions (e.g. 2 for images). Has shape
      `(batch_size_1, [batch_size_2,]
        height, height, width, width, depth, depth, ...)`.
  """
  if padding == Padding.CIRCULAR:
    spatial_axes = tuple(range(batch_ndim, lhs.ndim))
    lhs = _same_pad_for_filter_shape(lhs, window_shape, strides, spatial_axes)
    padding = Padding.VALID

  window_shape = (1,) * batch_ndim + tuple(window_shape)
  strides = (1,) * batch_ndim + tuple(strides)

  out = lax.reduce_window(lhs, 0., lax.add, window_shape, strides, padding.name)

  if pool_type == Pooling.AVG:
    out = _normalize(lhs, out, normalize_edges, padding, strides, window_shape)

  return out


def _normalize(lhs, out, normalize_edges, padding, strides, window_shape):
  if padding == Padding.SAME and normalize_edges:
    # `SAME` padding in `jax.experimental.stax.AvgPool` normalizes by actual
    # window size, which is smaller at the edges.
    one = np.ones_like(lhs, lhs.dtype)
    window_sizes = lax.reduce_window(one, 0., lax.add, window_shape, strides,
                                     padding.name)
    out /= window_sizes
  else:
    out /= onp.prod(window_shape)
  return out


def _diag_mul_full_spatial(
    x: np.ndarray,
    factor: float,
    diagonal_batch: bool) -> np.ndarray:
  if diagonal_batch:
    idx = (slice(None),)
    batch_ndim = 1
  else:
    if x.shape[0] != x.shape[1]:
      return x
    idx = ()
    batch_ndim = 2

  ndims = x.ndim // 2
  for i in range(ndims):
    shape = [1] * ndims
    size = x.shape[2 - batch_ndim + 2 * i]
    shape[i] = size
    idx += (np.arange(size).reshape(shape),) * 2

  x = ops.index_mul(x, idx, factor)
  return x


def _diag_mul_diagonal_spatial(
    x: np.ndarray,
    factor: float,
    diagonal_batch: bool) -> np.ndarray:
  if diagonal_batch:
    x *= factor

  else:
    if x.shape[0] != x.shape[1]:
      return x
    idx = np.diag_indices(x.shape[0]) + (Ellipsis,)
    x = ops.index_mul(x, idx, factor)

  return x


def _diag_mul(
    x: Optional[np.ndarray],
    factor: float,
    diagonal_batch: bool,
    diagonal_spatial: bool) -> Optional[np.ndarray]:
  if x is None:
    return x

  if diagonal_spatial:
    return _diag_mul_diagonal_spatial(x, factor, diagonal_batch)

  return _diag_mul_full_spatial(x, factor, diagonal_batch)


# MASKING


_NEG_INF = -1e20  # softmax raises an error if all entries are -np.inf


def _check_is_implemented(mask: np.ndarray, channel_axis: int) -> None:
  if mask.shape[channel_axis] != 1:
    raise NotImplementedError(
        'Different channel-wise masks as inputs to '
        'pooling layers are not yet supported. Please '
        'let us know about your use case at '
        'https://github.com/google/neural-tangents/issues/new')


def _mean_and_var(
    x: Optional[np.ndarray],
    axis: Axes = None,
    dtype: np.dtype = None,
    out: None = None,
    ddof: int = 0,
    keepdims: bool = False,
    mask: np.ndarray = None,
    get_var: bool = False
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
  """`np.mean` and `np.var` taking the `mask` information into account."""
  var = None
  if x is None:
    return x, var

  if mask is None:
    mean = np.mean(x, axis, dtype, out, keepdims)
    if get_var:
      var = np.var(x, axis, dtype, out, ddof, keepdims)

  else:
    axis = tuple(utils.canonicalize_axis(axis, x))
    size = utils.size_at(x, axis)
    mask = np.broadcast_to(mask, x.shape)
    mask_size = np.count_nonzero(mask, axis)
    for i in axis:
      mask_size = np.expand_dims(mask_size, i)
    size -= mask_size
    size = np.maximum(size, 1)

    mean = np.sum(x, axis=axis, keepdims=True) / size
    if not keepdims:
      mean = np.squeeze(mean, axis)

    if get_var:
      var = np.sum((x - mean)**2, axis=axis, keepdims=True) / (size - ddof)
      if not keepdims:
        var = np.squeeze(var, axis)

  return mean, var


def _sum_masks(masks: List[Optional[np.ndarray]]) -> Optional[np.ndarray]:
  def add_two_masks(mask1, mask2):
    if mask1 is None:
      return mask2

    if mask2 is None:
      return mask1

    return mask1 & mask2

  mask = functools.reduce(add_two_masks, masks, None)
  return mask


def _map_tuples(fn: Callable, tuples: Iterable[Tuple]) -> Tuple:
  return tuple(map(fn, zip(*(t for t in tuples))))


def _concat_masks(
    masks: List[Optional[np.ndarray]],
    input_shapes: Sequence[Sequence[int]],
    axis: int) -> Optional[np.ndarray]:
  """Returns a mask which is a concatenation of `masks`.

  Since elements of `masks` can have any shapes broadcastable to respective
  elements of `input_shapes`, their concatenation may require broadcasting and
  cannot be done with a single `np.concatenate` call.

  Args:
    masks: list of masks to concatenate.
    input_shapes: list of input shapes to which the masks are applied.
    axis: concatenation axis.

  Returns:
    A single `np.ndarray` mask applicable to the concatenated inputs.
  """
  if len(masks) != len(input_shapes):
    raise ValueError(f'Number of masks ({len(masks)}) and inputs '
                     f'({len(input_shapes)}) don\'t match, please file a bug at'
                     f' https://github.com/google/neural-tangents/issues/new.')

  if all(m is None for m in masks):
    return None

  axis %= len(input_shapes[0])

  # Expand the concatenation dimension of each mask.
  masks = [m if m is None else np.broadcast_to(
      m,
      (m.shape[:axis] +
       tuple(input_shapes[i][axis: axis + 1]) +
       m.shape[axis + 1:]))
           for i, m in enumerate(masks)]

  # Max shape to broadcast all masks to along non-concat dimension.
  max_shape = _map_tuples(max, (m.shape for m in masks if m is not None))

  # Shape of the mask to replace `None` masks with.
  max_shapes = [tuple(map(min, max_shape, i)) for i in input_shapes]

  masks = [
      (np.broadcast_to(
          m,
          max_shape[:axis] + m.shape[axis: axis + 1] + max_shape[axis + 1:])
       if m is not None
       else np.zeros_like(max_shapes[i], dtype=np.bool_))
      for i, m in enumerate(masks)
  ]

  return np.concatenate(masks, axis)


def _pool_mask(
    mask: np.ndarray,
    window_shape: Sequence[int],
    strides: Sequence[int],
    padding: Padding,
    batch_axis: int,
    channel_axis: int) -> np.ndarray:
  window_shape = list(window_shape)
  strides = list(strides)

  non_spatial_axes = utils.canonicalize_axis((batch_axis, channel_axis), mask)
  for i in non_spatial_axes:
    window_shape.insert(i, 1)
    strides.insert(i, 1)

  # Get the output shape.
  out_shape = eval_shape(lambda x: lax.reduce_window(
      operand=x,
      init_value=np.zeros((), x.dtype),
      computation=op.or_,
      window_dimensions=window_shape,
      window_strides=strides,
      padding=padding.name
  ), mask).shape

  # If shapes don't match, stride through the mask.
  if mask.shape != out_shape:
    pads = lax.padtype_to_pads(mask.shape, window_shape, strides, padding.name)
    slices = ()
    for i in range(mask.ndim):
      start = - pads[i][0] + (window_shape[i] - 1) // 2
      end = start + 1 + (out_shape[i] - 1) * strides[i]
      slices += (slice(start, end, strides[i]),)

    mask = mask[slices]
    if mask.shape != out_shape:
      raise ValueError(f'Mask shape must be {out_shape}, but got {mask.shape}, '
                       f'please submit a bug to '
                       f'https://github.com/google/neural-tangents/issues/new.')
  return mask


# POSITIONAL EMBEDDINGS


def _pos_emb_identity(shape: Sequence[int]) -> np.ndarray:
  size = utils.size_at(shape)
  R = np.eye(size).reshape(tuple(shape) * 2)
  R = utils.zip_axes(R)
  return R


def _pos_emb_pdist(shape: Sequence[int],
                   pos_emb_p_norm: Optional[float],
                   pos_emb_decay_fn: Optional[Callable[[float], float]]
                   ) -> np.ndarray:
  if pos_emb_decay_fn is None:
    # Identity / one-hot positional embeddings.
    return _pos_emb_identity(shape)

  # Pairwise distance-based positional embeddings.
  ndim = len(shape)
  R = np.zeros((1,) * (ndim * 2))
  for axis in range(ndim):
    d = np.arange(shape[axis])
    pd = utils.outer_prod(d, d, 0, d.ndim, op.sub)
    pd = pd.reshape((1,) * (2 * axis) +
                    pd.shape +
                    (1,) * (2 * (ndim - axis - 1)))
    R += np.abs(pd) ** pos_emb_p_norm

  R = pos_emb_decay_fn(R)
  return R


def _get_all_pos_emb(k: Kernel,
                     pos_emb_type: PositionalEmbedding,
                     pos_emb_p_norm: float,
                     pos_emb_decay_fn: Optional[Callable[[float], float]]
                     ) -> Tuple[Optional[np.ndarray],
                                Optional[np.ndarray],
                                Optional[np.ndarray]]:
  if pos_emb_type == PositionalEmbedding.NONE:
    return None, None, None

  shape, _ = utils.shape_and_axes(k.shape1, (k.batch_axis, k.channel_axis))
  R = _pos_emb_pdist(shape, pos_emb_p_norm, pos_emb_decay_fn)

  if k.is_reversed:
    R = utils.reverse_zipped(R)

  batch_ndim = 1 if k.diagonal_batch else 2
  R11 = np.expand_dims(R, tuple(range(batch_ndim)))
  R12 = R11 if batch_ndim == 2 else np.expand_dims(R, (0, 1))
  R22 = None if k.cov2 is None else R11

  mask11, mask12, mask22 = k._get_mask_prods(k.mask1, k.mask2)
  R11 = utils.mask(R11, mask11)
  R12 = utils.mask(R12, mask12)
  R22 = utils.mask(R22, mask22)
  return R11, R12, R22
