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

"""Linear functions."""

import enum
import functools
import operator as op
import string
from typing import Callable, Iterable, Optional, Sequence, Tuple, TypeVar, Union
import warnings

import jax
from jax import lax
from jax import numpy as np
from jax import ops
from jax import random
from jax import ShapeDtypeStruct, ShapedArray, eval_shape, vmap
import jax.example_libraries.stax as ostax
import numpy as onp
from .requirements import Bool, Diagonal, get_diagonal_outer_prods, layer, mean_and_var, requires, supports_masking
from ..utils import utils
from ..utils.kernel import Kernel
from ..utils.typing import Axes, InternalLayer, InternalLayerMasked, PyTree


# Enums


class Padding(enum.Enum):
  """Type of padding in pooling and convolutional layers.

  Attributes:
    CIRCULAR:
      circular padding, as if the input were a torus.

    SAME:
      same, a.k.a. zero padding.

    VALID:
      valid, a.k.a. no padding.
  """
  CIRCULAR = 'CIRCULAR'
  SAME = 'SAME'
  VALID = 'VALID'


class _Pooling(enum.Enum):
  """Type of pooling in pooling layers.

  Attributes:
    AVG:
      average pooling, the output is normalized by the input receptive field
      size.

    SUM:
      sum pooling, no normalization.
  """
  AVG = 'AVG'
  SUM = 'SUM'


class AggregateImplementation(enum.Enum):
  """Implementation of the :obj:`Aggregate` layer.

  See :obj:`Aggregate` docstring for details.

  Attributes:
    DENSE:
      Is recommended for dense graphs, where the number of edges `E` is
      proportional to the number of vertices `V` to the power of 1.5 or more.

    SPARSE:
      Is recommended for sparse graphs, where `E ~ O(V)` or less.
  """
  DENSE = 'DENSE'
  SPARSE = 'SPARSE'


# LAYERS


@layer
@supports_masking(remask_kernel=False)
def Identity() -> InternalLayer:
  """Identity (no-op).

  Based on :obj:`jax.example_libraries.stax.Identity`.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  init_fn, apply_fn = ostax.Identity
  kernel_fn = lambda k, **kwargs: k
  return init_fn, apply_fn, kernel_fn


@layer
@supports_masking(remask_kernel=False)
def DotGeneral(
    *,
    lhs: Optional[Union[np.ndarray, float]] = None,
    rhs: Optional[Union[np.ndarray, float]] = None,
    dimension_numbers: lax.DotDimensionNumbers = (((), ()), ((), ())),
    precision: Optional[lax.Precision] = None,
    batch_axis: int = 0,
    channel_axis: int = -1
) -> InternalLayerMasked:
  r"""Constant (non-trainable) rhs/lhs Dot General.

  Dot General allows to express any linear transformation on the inputs,
  including but not limited to matrix multiplication, pooling, convolutions,
  permutations, striding, masking etc (but specialized implementations are
  typically much more efficient).

  Returned `apply_fn` is calling
  `jax.lax.dot_general(inputs, rhs, dimension_numbers, precision)` or
  `jax.lax.dot_general(lhs, inputs, dimension_numbers, precision)`, depending
  on whether `lhs` or `rhs` is specified (not `None`).

  Example:
    >>> from jax import random
    >>> import jax.numpy as np
    >>> from neural_tangents import stax
    >>> #
    >>> # Two time series stacked along the second (H) dimension.
    >>> x = random.normal(random.PRNGKey(1), (5, 2, 32, 3))  # NHWC
    >>> #
    >>> # Multiply all outputs by a scalar:
    >>> nn = stax.serial(
    >>>     stax.Conv(128, (1, 3)),
    >>>     stax.Relu(),
    >>>     stax.DotGeneral(rhs=2.),  # output shape is (5, 2, 30, 128)
    >>>     stax.GlobalAvgPool()      # (5, 128)
    >>> )
    >>> #
    >>> # Subtract second time series from the first one:
    >>> nn = stax.serial(
    >>>     stax.Conv(128, (1, 3)),
    >>>     stax.Relu(),
    >>>     stax.DotGeneral(
    >>>         rhs=np.array([1., -1.]),
    >>>         dimension_numbers=(((1,), (0,)), ((), ()))),  # (5, 30, 128)
    >>>     stax.GlobalAvgPool()                              # (5, 128)
    >>> )
    >>> #
    >>> # Flip outputs with each other
    >>> nn = stax.serial(
    >>>     stax.Conv(128, (1, 3)),
    >>>     stax.Relu(),
    >>>     stax.DotGeneral(
    >>>         lhs=np.array([[0., 1.], [1., 0.]]),
    >>>         dimension_numbers=(((1,), (1,)), ((), ()))),  # (5, 2, 30, 128)
    >>>     stax.GlobalAvgPool()                              # (5, 128)
    >>> )

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
      a tuple of tuples of the form `((lhs_contracting_dims,
      rhs_contracting_dims), (lhs_batch_dims, rhs_batch_dims))`.

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
  diagonal_spatial = Diagonal(
      input=Bool.YES
      if (input_cs in ((), (batch_axis,)) or (rhs is None and lhs is None))
      else Bool.NO)

  @requires(diagonal_batch=diagonal_batch,
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
    aggregate_axis: Optional[Axes] = None,
    batch_axis: int = 0,
    channel_axis: int = -1,
    to_dense: Optional[Callable[[np.ndarray], np.ndarray]] = lambda p: p,
    implementation: str = AggregateImplementation.DENSE.value
) -> InternalLayer:
  r"""Aggregation operator (graphical neural network).

  See e.g.
  "`Graph Neural Tangent Kernel: Fusing Graph Neural Networks with Graph Kernels
  <https://arxiv.org/abs/1905.13192>`_".

  Specifically, each `N+2`-D `input` of shape `(batch, X_1, ..., X_N, channels)`
  (subject to `batch_axis` and `channel_axis`) is accompanied by an array
  `pattern` specifying the directed edges (arcs, arrows) of the graph. The
  format of `pattern` depends on `implementation`:

  `implementation = "DENSE"`:
    Is recommended for dense graphs, where the number of
    edges `E` is proportional to the number of vertices `V` to the power of 1.5
    or more. In this case, `pattern` is a [weighted] adjacency 2-adjacency
    `2K+1`-D tensor of shape `(batch, X_i1, ..., X_iK, X_i1, ..., X_iK)` (i.e.
    leading batch dimensions, repeated spatial dimensions, no channel dimension)
    and the output tensor is
    `lax.dot_general(inputs, pattern, ((aggregate_axes, range(1, K + 1)),
    (batch_axis,), (0,)))` with the `batch_axis` and `channel_axis` preserved.
    `K = len(aggregate_axes)`.

    Having `pattern[n, i1, ..., iK, j1, ..., jK] == w` represents a directed
    edge (arc) from tail pixel / token `(i1, ..., iK)` to head `(j1, ..., jK)`
    with weight `w` in an individual input sample `n`. The `apply_fn` of this
    layer replaces all vertices with the (weighted) sum of all direct
    predecessors to the given vertex.

    Note that individual inputs can have more than `K` dimensions (e.g.
    channels, other coordinates), in which case slices along these coordinates
    are processed in the same way independently.

    This implementation uses matrix multiplication, and for a graph with `V`
    vertices and `E` edges, `apply_fn` costs `O(V^2)` memory and time, while
    `kernel_fn` costs `O(V^2)` memory and `O(V^3)` time.

    The adjacency tensor `pattern` can be specified in a sparse format. If
    you provide a `to_dense` function (defaults to identity), then `pattern` is
    decoded into a dense representation as described above
    (`pattern_dense = to_dense(pattern)`) each time `apply_fn` or `kernel_fn`
    are called. This avoids storing the whole graph in the dense format in
    advance, but only convert it to dense format on the fly, for each
    individual batch `x` / `(x1, x2)`. However, this does not improve the
    runtime or memory of the `Aggregate` layer (in fact makes it a bit slower
    due to an extra `to_dense` call).

  `implementation = "SPARSE"`:
    Is recommended for sparse graphs, where `E ~ O(V)` or less. In this case,
    `pattern` must be an integer array of shape `(batch, n_edges, K, 2)`,
    specifying `n_edges` directed edges (arcs) of weight `w = 1` for each of
    the `batch` input samples (if `K == 1` `pattern` can also have the shape
    `(batch, n_edges, 2)`). Trailing dimension of size 2 corresponds to tails
    (sources, senders) and heads (targets, receivers). Edges can be repeated,
    which is interpreted as having their weight be the number of repetitions.
    If any of the `K` coordinates of a given vertex in `heads` is negative
    (e.g. `-1`), it is discarded. This can be used for padding, when different
    input samples have different `n_edges`. Note that this means you can't use
    negative indexing to specify vertices.

    This implementation uses :obj:`jax.ops.segment_sum` instead of matrix
    multiplication. This makes `apply_fn` cost `O(V + E)` memory and `O(V + E)`
    time, and `kernel_fn` cost `O(V^2)` memory and `O(V^2 + E^2 + V * E)` time.
    This is beneficial for sparse graphs, i.e. `E << V^2`, but detrimental for
    dense graphs (when `E ~ V^2`).

  See Also:
    `AggregateTest` in `tests/stax_test.py` for examples and conversion between
    sparse and dense patterns.

  Example:
    >>> # 1D inputs
    >>> x = random.normal(random.PRNGKey(1), (5, 3, 32))  # NCH
    >>> #
    >>> # 1) NHH dense binary adjacency matrix
    >>> A = random.bernoulli(random.PRNGKey(2), 0.5, (5, 32, 32))
    >>> # `A[n, h1, h2] == True`
    >>> # means an edge between tokens `h1` and `h2` in sample `n`.
    >>> #
    >>> init_fn, apply_fn, kernel_fn = stax.Aggregate(aggregate_axis=2,
    >>>                                               batch_axis=0,
    >>>                                               channel_axis=1)
    >>> #
    >>> out = apply_fn((), x, pattern=A)
    >>> # output is the same as `x @ A` of shape (5, 3, 32)
    >>> #
    >>> # Sparse NHH binary pattern with 10 edges
    >>> n_edges = 10
    >>> A_sparse = random.randint(random.PRNGKey(3),
    >>>                           shape=(x.shape[0], n_edges, 1, 2),
    >>>                           minval=0,
    >>>                           maxval=x.shape[2])
    >>> #
    >>> # Setting `implementation="SPARSE"` to invoke the segment sum
    >>> # implementation.
    >>> init_fn, apply_fn, kernel_fn = stax.Aggregate(aggregate_axis=2,
    >>>                                               batch_axis=0,
    >>>                                               channel_axis=1,
    >>>                                               implementation="SPARSE")
    >>> #
    >>> out = apply_fn((), x, pattern=A_sparse)
    >>> # output is of shape (5, 3, 32), computed via `jax.ops.segment_sum`.
    >>> #
    >>> # 2D inputs
    >>> x = random.normal(random.PRNGKey(1), (5, 3, 32, 16))  # NCHW
    >>> #
    >>> # 2) NHWHW dense binary adjacency matrix
    >>> A = random.bernoulli(random.PRNGKey(2), 0.5, (5, 32, 16, 32, 16))
    >>> # `A[n, h1, w1, h2, w2] == True`
    >>> # means an edge between pixels `(h1, w1)` and `(h2, w2)` in image `n`.
    >>> #
    >>> init_fn, apply_fn, kernel_fn = stax.Aggregate(aggregate_axis=(2, 3),
    >>>                                               batch_axis=0,
    >>>                                               channel_axis=1)
    >>> #
    >>> out = apply_fn((), x, pattern=A)
    >>> # output is of shape (5, 3, 32, 16), the same as
    >>> # `(x.reshape((5, 3, 32 * 16)) @ A.reshape((5, 32 * 16, 32 * 16))
    >>> #  ).reshape(x.shape)`
    >>> #
    >>> # 3) NWW binary adjacency matrix
    >>> A = random.bernoulli(random.PRNGKey(2), 0.5, (5, 16, 16))
    >>> # `A[n, w1, w2] == True`
    >>> # means an edge between rows `w1` and `w2` in image `n`.
    >>> #
    >>> init_fn, apply_fn, kernel_fn = stax.Aggregate(aggregate_axis=(3,),
    >>>                                               batch_axis=0,
    >>>                                               channel_axis=1)
    >>> #
    >>> out = apply_fn((), x, pattern=A)
    >>> # output is of shape (5, 3, 32, 16), the same as
    >>> # `(x.reshape((5, 3 * 32, 16)) @ A).reshape(x.shape)`
    >>> #
    >>> # 4) Infinite width example
    >>> x1 = random.normal(random.PRNGKey(1), (5, 3, 32))  # NCH
    >>> x2 = random.normal(random.PRNGKey(2), (2, 3, 32))  # NCH
    >>> #
    >>> # NHH binary adjacency matrices
    >>> A1 = random.bernoulli(random.PRNGKey(2), 0.5, (5, 32, 32))
    >>> A2 = random.bernoulli(random.PRNGKey(2), 0.5, (2, 32, 32))
    >>> #
    >>> _, _, kernel_fn_id = stax.Identity()
    >>> #
    >>> _, _, kernel_fn_agg = stax.Aggregate(aggregate_axis=2,
    >>>                                      batch_axis=0,
    >>>                                      channel_axis=1)
    >>> #
    >>> nngp = kernel_fn_id(x1, x2, get='nngp', channel_axis=1)
    >>> # initial NNGP of shape (5, 2, 32, 32)
    >>> K_agg = kernel_fn_agg(x1, x2, get='nngp', pattern=(A1, A2))
    >>> # output NNGP of same shape (5, 2, 32, 32):
    >>> # `K_agg[n1, n2] == A1[n1].T @ nngp[n1, n2] @ A2[n2]`

  Args:
    aggregate_axis:
      axes (non-batch and non-channel) to aggregate predecessor vertices over.

    batch_axis:
      batch axis for `inputs`. Defaults to `0`, the leading axis.

    channel_axis:
      channel axis for `inputs`. Defaults to `-1`, the trailing axis. For
      `kernel_fn`, channel size is considered to be infinite.

    to_dense:
      Ignored unless `implementation == "DENSE"`. A function to convert
      potentially sparse `pattern` matrices into dense `2K+1`-D tensors of shape
      `(batch, X_i1, ..., X_iK, X_i1, ..., X_iK)`, with the batch leading
      dimension, and no channel dimension, where `K = len(aggregate_axes)`.
      Will be called on input `pattern` (or a pair `(pattern1, pattern2)`)
      every time  `apply_fn` or `kernel_fn` is called. Defaults to identity,
      meaning that `pattern` is expected in the dense format.

    implementation:
      `"DENSE"` or `"SPARSE"`, specifying which implementation to use.
      `"DENSE"` uses matrix multiplications and is recommended for dense graphs
      (`E ~> O(V^1.5)`), while `"SPARSE"` uses :obj:`jax.ops.segment_sum` and is
      recommended for sparse graphs (`E ~< O(V)`). Note that different
      `implementation` require different `pattern` array format - see the
      :obj:`Aggregate` layer docstring above for details.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  implementation = AggregateImplementation(implementation)
  if implementation == AggregateImplementation.SPARSE:
    warnings.warn('Negative indices in `pattern` are considered as padding '
                  '(i.e. ignored), unlike typical numpy negative indexing.')

  init_fn = lambda rng, input_shape: (input_shape, ())

  def get_agg_axes(ndim: int) -> Tuple[Tuple[int, ...], int, int]:
    _batch_axis, _channel_axis = utils.mod((batch_axis, channel_axis), ndim)
    if aggregate_axis is None:
      agg_axes = tuple(i for i in range(ndim)
                       if i not in (_batch_axis, _channel_axis))
    else:
      agg_axes = tuple(utils.canonicalize_axis(aggregate_axis, ndim))
    return agg_axes, _batch_axis, _channel_axis

  def get_dimension_numbers(ndim: int) -> lax.DotDimensionNumbers:
    agg_axes, batch_axis, _ = get_agg_axes(ndim)
    agg_ndim = len(agg_axes)
    return (agg_axes, (range(1, agg_ndim + 1))), ((batch_axis,), (0,))

  @functools.partial(vmap, in_axes=(0, None))
  def make_indices(index_array, agg_shape):
    index_array = np.moveaxis(index_array, -1, 0)
    raveled = np.ravel_multi_index(index_array, agg_shape, 'wrap')
    # We mask edges where either sender or receiver is negative.
    return np.where(np.all(index_array >= 0, axis=0), raveled, -1)

  def get_senders_receivers(pattern, batch_size: int, agg_ndim: int):
    """Unpack `pattern` and make sure it has correct shape."""
    if pattern.shape[-1] != 2:
      raise ValueError('`pattern` must have a trailing dimension of 2, got '
                       f'{pattern.shape[-1]}.')
    s, r = pattern[..., 0], pattern[..., 1]

    # Allow for `(batch, n_edges, 2)` shape for single aggregation
    # dimension `K == 1`.
    if agg_ndim == 1 and s.ndim == 2:
      s, r = np.expand_dims(s, -1), np.expand_dims(r, -1)

    if s.ndim != 3:
      raise ValueError(f'Tails and heads need to be 3-dimensional, '
                       f'got {s.ndim}.')

    if s.shape[2] != agg_ndim:
      raise ValueError(f'Trailing dimension of tails and heads need to have '
                       f'the same size as the number of aggregate axes of '
                       f'`aggregate_axis` ({agg_ndim}), got {s.shape[2]}.')

    if s.shape[0] != batch_size:
      raise ValueError(f'Tails and heads need to have leading dimension equal '
                       f'to batch size, got {s.shape[0]}.')

    return s, r

  def apply_fn(params,
               inputs: np.ndarray,
               *,
               pattern: Optional[np.ndarray] = None,
               **kwargs):
    """Compute the transformed tensors after an aggregation layer.

    Args:
      params:
        Not used.

      inputs:
        An input `N+2`-D tensor of shape `(batch, X_1, ..., X_N, channels)`
        (subject to `batch_axis` and `channel_axis`).

      pattern:
        A tensor specifying the directed edges between `inputs`. The shape and
        type of `pattern` depends on `implementation` (see docstring of
        `stax.Aggregate` above).

        `implementation == "DENSE"`:
          `pattern` must be a (float) `2K+1`-D tensor of shape
          `(batch, X_i1, ..., X_iK, X_i1, ..., X_iK)`, with the batch leading
          dimension, and no channel dimension, where `K = len(aggregate_axes)`.
          Can have another shape (e.g. a sparse matrix), as long as
          `to_dense(pattern)` has the correct (dense) shape (if `nt.batch` is
          used, the leading dimension of `pattern` must be the batch dimension,
          of size `batch`).

        `implementation == "SPARSE"`:
          `pattern` must be an integer array of shape `(batch, n_edges, K, 2)`,
          specifying tail and head (source and target / sender and receiver)
          vertices along the trailing dimension (if `K == 1`, `pattern` is also
          allowed to have the shape `(batch, n_edges, 2)`).

        `pattern=None` means identity adjacency, i.e. `apply_fn` is an identity
          function.

      **kwargs:
        unused.

    Returns:
      An `N+2`-D tensor of shape of the same shape as `inputs`.
    """
    if pattern is None:
      return inputs

    del params

    ndim = inputs.ndim
    agg_axes, batch_axis, channel_axis = get_agg_axes(ndim)
    agg_ndim = len(agg_axes)

    if implementation == AggregateImplementation.DENSE:
      # Dense implementation through matrix multiplication.
      pattern = to_dense(pattern)

      dn = get_dimension_numbers(ndim)
      out = lax.dot_general(inputs, pattern.astype(inputs.dtype), dn)

      # Put back potentially displaced batch and channel axes.
      out_c_axis = utils.axis_after_dot(channel_axis % ndim, dn[0][0], dn[1][0])
      out_b_axis = utils.axis_after_dot(batch_axis % ndim, dn[0][0], dn[1][0])

      out = np.moveaxis(out,
                        (out_b_axis, out_c_axis) + tuple(range(-agg_ndim, 0)),
                        (batch_axis, channel_axis) + agg_axes)

    elif implementation == AggregateImplementation.SPARSE:
      # Sparse implementation through `jax.ops.segment_sum`.
      s, r = get_senders_receivers(pattern, inputs.shape[batch_axis], agg_ndim)

      # Canonicalize axes
      src_axes = (batch_axis,) + agg_axes + (channel_axis,)
      dst_axes = (0,) + tuple(range(1, agg_ndim + 1)) + (-1,)

      inputs = np.moveaxis(inputs, src_axes, dst_axes)
      input_shape = inputs.shape
      inputs = inputs.reshape((inputs.shape[0],
                               functools.reduce(
                                   op.mul, inputs.shape[1:agg_ndim + 1], 1))
                              + inputs.shape[agg_ndim + 1:])

      agg_shape = input_shape[1:agg_ndim + 1]
      s, r = make_indices(s, agg_shape), make_indices(r, agg_shape)

      @vmap
      def pass_messages(s, r, inputs):
        n_nodes = inputs.shape[0]
        sender_in = inputs[s]
        messages = ops.segment_sum(sender_in, r, num_segments=n_nodes)
        return messages

      out = pass_messages(s, r, inputs)
      out = out.reshape(input_shape)
      out = np.moveaxis(out, dst_axes, src_axes)

    else:
      raise ValueError(f'Unrecognized `implementation == {implementation}.')

    return out

  @requires(batch_axis=batch_axis,
            channel_axis=channel_axis,
            diagonal_spatial=Diagonal(input=Bool.NO, output=Bool.NO))
  def kernel_fn(k: Kernel,
                *,
                pattern: Tuple[Optional[np.ndarray],
                               Optional[np.ndarray]] = (None, None),
                **kwargs):
    """Compute the transformed kernels after an aggregation kernel layer.

      Specifically, the `nngp`/`ntk` is a `2N+2`-D tensor of shape
      `(B_1, B_2, X_1, X_1, ..., X_N, X_N)`.

      If `implementation == "DENSE"`, this tensor will be aggregated
      (via matrix multiplication) on the left by `to_dense(pattern[0])` of
      shape `(B_1, X_i1, ..., X_iK)` and on the right by `to_dense(pattern[1])`
      of shape `(B_2, X_i1, ..., X_iK)`. Ignoring the batch dimensions, the
      output `nngp/ntk` is `pattern[0].T @ nngp/ntk @ pattern[1]`.

      If `implementation == "SPARSE"`, result is computed using
      `jax.ops.segment_sum` given `pattern[0]` and `pattern[1]` as integer
      arrays of shapes `(B_1, n_edges_1, K, 2)` and `(B_2, n_edges_2, K, 2)`
      respectively.
    """
    pattern1, pattern2 = pattern

    if pattern1 is None and pattern2 is None:
      return k

    if pattern1 is None or pattern2 is None:
      raise NotImplementedError(
          'Having exactly one of two `pattern1/2=None` is not implemented. '
          'Please file a bug at '
          'https://github.com/google/neural-tangents/issues/new.')

    ndim = len(k.shape1)
    agg_axes, batch_axis, channel_axis = get_agg_axes(ndim)
    agg_ndim = len(agg_axes)
    agg_shape = tuple(k.shape1[a] for a in agg_axes)
    agg_size = functools.reduce(op.mul, agg_shape, 1)

    def bucket_axes(ndim, start_axis):
      """Bucket kernel axes into batch, aggregate, and non-aggregate."""
      ndim_spatial = (ndim - start_axis) // 2
      agg_1 = tuple(
          a - int(batch_axis < a) - int(channel_axis < a) + start_axis
          for a in agg_axes)
      agg_2 = tuple(
          a + ndim_spatial
          for a in agg_1)
      non_agg_1 = tuple(
          a for a in range(start_axis, start_axis + ndim_spatial)
          if a not in agg_1)
      non_agg_2 = tuple(
          a for a in range(start_axis + ndim_spatial, ndim)
          if a not in agg_2)
      return tuple(range(start_axis)), agg_1, agg_2, non_agg_1, non_agg_2

    if implementation == AggregateImplementation.DENSE:
      # Dense implementation through matrix multiplication.
      pattern1 = None if pattern1 is None else to_dense(pattern1)
      pattern2 = None if pattern2 is None else to_dense(pattern2)

      k = k.dot_general(
          other1=pattern1,
          other2=pattern2,
          is_lhs=False,
          dimension_numbers=get_dimension_numbers(ndim)
      )

      # Put back potentially displaced axes.
      def transpose(k, diagonal_batch):
        if k is None or k.ndim == 0:
          return k

        start_axis = 1 if diagonal_batch else 2

        k = utils.unzip_axes(k, start_axis)
        b, agg_1, agg_2, non_agg_1, non_agg_2 = bucket_axes(k.ndim, start_axis)
        permutation = b + non_agg_1 + agg_1 + non_agg_2 + agg_2
        k = np.transpose(k, onp.argsort(permutation))
        return utils.zip_axes(k, start_axis)

      k = k.replace(
          cov1=transpose(k.cov1, k.diagonal_batch),
          cov2=transpose(k.cov2, k.diagonal_batch),
          nngp=transpose(k.nngp, False),
          ntk=transpose(k.ntk, False),
          batch_axis=batch_axis % ndim,
          channel_axis=channel_axis % ndim
      )

    elif implementation == AggregateImplementation.SPARSE:
      # Sparse implementation through `jax.ops.segment_sum`.
      def pass_messages(s1, s2, r1, r2, k):
        v1, v2 = k.shape[:2]

        def send(s, r, num_segments):
          return ops.segment_sum(s, r, num_segments=num_segments)

        send_inner = vmap(functools.partial(send, num_segments=v2), (0, None))

        k = k[s1[:, None], s2[None, :]]
        k = send_inner(k, r2)
        k = send(k, r1, num_segments=v1)
        return k

      pass_messages_self = vmap(pass_messages)
      pass_messages_cross = vmap(vmap(pass_messages,
                                      (None, 0, None, 0, 0)),
                                 (0, None, 0, None, 0))

      s1, r1 = get_senders_receivers(pattern1, k.shape1[batch_axis], agg_ndim)
      s2, r2 = get_senders_receivers(pattern2, k.shape2[batch_axis], agg_ndim)

      s1, r1 = make_indices(s1, agg_shape), make_indices(r1, agg_shape)
      s2, r2 = make_indices(s2, agg_shape), make_indices(r2, agg_shape)

      def agg(k, diagonal_batch, s1, r1, s2, r2):
        if k is None or k.ndim == 0:
          return k

        start_axis = 1 if diagonal_batch else 2
        k = utils.unzip_axes(k, start_axis)
        b, agg_1, agg_2, non_agg_1, non_agg_2 = bucket_axes(k.ndim, start_axis)
        permutation = b + agg_1 + agg_2 + non_agg_1 + non_agg_2
        k = np.transpose(k, permutation)
        k_shape = k.shape
        k = k.reshape(
            k.shape[:start_axis] +
            (agg_size,) * 2 +
            k.shape[start_axis + 2 * len(agg_axes):]
        )
        fn = pass_messages_self if diagonal_batch else pass_messages_cross
        k = fn(s1, s2, r1, r2, k)
        k = k.reshape(k_shape)
        k = np.transpose(k, onp.argsort(permutation))
        return utils.zip_axes(k, start_axis)

      nngp = agg(k.nngp, False, s1, r1, s2, r2)
      ntk = agg(k.ntk, False, s1, r1, s2, r2)
      cov1 = agg(k.cov1, k.diagonal_batch, s1, r1, s1, r1)
      cov2 = agg(k.cov2, k.diagonal_batch, s2, r2, s2, r2)
      k = k.replace(nngp=nngp, ntk=ntk, cov1=cov1, cov2=cov2)

    else:
      raise ValueError(f'Unregocnized `implementation == {implementation}.')

    return k

  return init_fn, apply_fn, kernel_fn


@layer
@supports_masking(remask_kernel=True)
def Dense(
    out_dim: int,
    W_std: float = 1.,
    b_std: Optional[float] = None,
    batch_axis: int = 0,
    channel_axis: int = -1,
    parameterization: str = 'ntk',
    s: Tuple[int, int] = (1, 1),
) -> InternalLayerMasked:
  r"""Dense (fully-connected, matrix product).

  Based on :obj:`jax.example_libraries.stax.Dense`.

  Args:
    out_dim:
      The output feature / channel dimension. This is ignored in by the
      `kernel_fn` in `"ntk"` parameterization.

    W_std:
      Specifies the standard deviation of the weights.

    b_std:
      Specifies the standard deviation of the biases. `None` means no bias.

    batch_axis:
      Specifies which axis is contains different elements of the batch.
      Defaults to `0`, the leading axis.

    channel_axis: Specifies which axis contains the features / channels.
      Defaults to `-1`, the trailing axis. For `kernel_fn`, channel size is
      considered to be infinite.

    parameterization:
      Either `"ntk"` or `"standard"`.

      Under `"ntk"` parameterization (page 3 in "`Neural Tangent Kernel:
      Convergence and Generalization in Neural Networks
      <https://arxiv.org/abs/1806.07572>`_"),
      weights and biases are initialized as
      :math:`W_{ij} \sim \mathcal{N}(0,1)`, :math:`b_i \sim \mathcal{N}(0,1)`,
      and the finite width layer equation is
      :math:`z_i = \sigma_W / \sqrt{N} \sum_j W_{ij} x_j + \sigma_b b_i`, where
      `N` is `out_dim`.

      Under `"standard"` parameterization ("`On the infinite width limit of
      neural networks with a standard parameterization
      <https://arxiv.org/abs/2001.07301>`_".),
      weights and biases are initialized as :math:`W_{ij} \sim \mathcal{N}(0,
      W_{std}^2/N)`,
      :math:`b_i \sim \mathcal{N}(0,\sigma_b^2)`, and the finite width layer
      equation is
      :math:`z_i = \frac{1}{s} \sum_j W_{ij} x_j + b_i`, where `N` is `out_dim`.

      `N` corresponds to the respective variable in
      "`On the infinite width limit of neural networks with a standard
      parameterization <https://arxiv.org/abs/2001.07301>`_".

    s:
      only applicable when `parameterization="standard"`. A tuple of integers
      specifying the width scalings of the input and the output of the layer,
      i.e. the weight matrix `W` of the layer has shape
      `(s[0] * in_dim, s[1] * out_dim)`, and the bias has size `s[1] * out_dim`.

      .. note::
        We need `s[0]` (scaling of the previous layer) to infer `in_dim` from
        `input_shape`. Further, for the bottom layer, `s[0]` must be `1`, and
        for all other layers `s[0]` must be equal to `s[1]` of the previous
        layer. For the top layer, `s[1]` is expected to be `1` (recall that the
        output size is `s[1] * out_dim`, and in common infinite network
        research input and output sizes are considered fixed).

      `s` corresponds to the respective variable in
      "`On the infinite width limit of neural networks with a standard
      parameterization <https://arxiv.org/abs/2001.07301>`_".

      For `parameterization="ntk"`, or for standard, finite-width networks
      corresponding to He initialization, `s=(1, 1)`.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  # TODO(jaschasd): after experimentation, evaluate whether to change default
  # parameterization from "ntk" to "standard"

  parameterization = parameterization.lower()

  def _init_fn(rng, input_shape, out_dim):
    _channel_axis = channel_axis % len(input_shape)
    output_shape = (input_shape[:_channel_axis] + (out_dim,)
                    + input_shape[_channel_axis + 1:])
    rng1, rng2 = random.split(rng)
    W = random.normal(rng1, (input_shape[_channel_axis], out_dim))

    if b_std is None:
      b = None
    else:
      b_shape = [1] * len(input_shape)
      b_shape[channel_axis] = out_dim
      b = random.normal(rng2, b_shape)

    return output_shape, (W, b)

  def ntk_init_fn(rng, input_shape):
    return _init_fn(rng, input_shape, out_dim)

  def standard_init_fn(rng, input_shape):
    output_shape, (W, b) = _init_fn(rng, input_shape, out_dim * s[1])
    W *= W_std / (input_shape[channel_axis] / s[0])**0.5
    b = None if b is None else b * b_std
    return output_shape, (W, b)

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
      norm = W_std / inputs.shape[channel_axis]**0.5
      outputs = norm * prod
      if b is not None:
        outputs += b_std * b
    elif parameterization == 'standard':
      outputs = prod / s[0]**0.5
      if b is not None:
        outputs += b
    else:
      raise ValueError(f'Parameterization not supported: {parameterization}')

    return outputs

  @requires(batch_axis=batch_axis,
            channel_axis=channel_axis,
            diagonal_spatial=Diagonal())
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
      input_width = k.shape1[channel_axis] / s[0]
      if ntk is not None:
        ntk = input_width * nngp + W_std**2 * ntk
        if b_std is not None:
          ntk += 1.
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
    strides: Optional[Sequence[int]] = None,
    padding: str = Padding.VALID.name,
    W_std: float = 1.0,
    b_std: Optional[float] = None,
    dimension_numbers: Optional[Tuple[str, str, str]] = None,
    parameterization: str = 'ntk',
    s: Tuple[int, int] = (1, 1),
) -> InternalLayerMasked:
  """General convolution.

  Based on :obj:`jax.example_libraries.stax.GeneralConv`.

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
      specification in :obj:`jax.lax.conv_general_dilated`.

    parameterization:
      Either `"ntk"` or `"standard"`. These parameterizations are the direct
      analogues for convolution of the corresponding parameterizations for
      :obj:`Dense` layers.

    s:
      A tuple of integers, a direct convolutional analogue of the respective
      parameters for the :obj:`Dense` layer.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  return _Conv(out_chan, filter_shape, strides, padding, W_std, b_std,
               dimension_numbers, parameterization, s, False, True)


@layer
@supports_masking(remask_kernel=True)
def ConvTranspose(
    out_chan: int,
    filter_shape: Sequence[int],
    strides: Optional[Sequence[int]] = None,
    padding: str = Padding.VALID.name,
    W_std: float = 1.0,
    b_std: Optional[float] = None,
    dimension_numbers: Optional[Tuple[str, str, str]] = None,
    parameterization: str = 'ntk',
    s: Tuple[int, int] = (1, 1),
) -> InternalLayerMasked:
  """General transpose convolution.

  Based on :obj:`jax.example_libraries.stax.GeneralConvTranspose`.

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
      specification in :obj:`jax.lax.conv_general_dilated`.

    parameterization:
      Either `"ntk"` or `"standard"`. These parameterizations are the direct
      analogues for convolution of the corresponding parameterizations for
      :obj:`Dense` layers.

    s:
      A tuple of integers, a direct convolutional analogue of the respective
      parameters for the :obj:`Dense` layer.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  return _Conv(out_chan, filter_shape, strides, padding, W_std, b_std,
               dimension_numbers, parameterization, s, True, True)


@layer
@supports_masking(remask_kernel=True)
def ConvLocal(
    out_chan: int,
    filter_shape: Sequence[int],
    strides: Optional[Sequence[int]] = None,
    padding: str = Padding.VALID.name,
    W_std: float = 1.0,
    b_std: Optional[float] = None,
    dimension_numbers: Optional[Tuple[str, str, str]] = None,
    parameterization: str = 'ntk',
    s: Tuple[int, int] = (1, 1),
) -> InternalLayerMasked:
  """General unshared convolution.

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
      standard deviation of the biases. `None` means no bias.

    dimension_numbers:
      Specifies which axes should be convolved over. Should match the
      specification in :obj:`jax.lax.conv_general_dilated`.

    parameterization:
      Either `"ntk"` or `"standard"`. These parameterizations are the direct
      analogues for convolution of the corresponding parameterizations for
      :obj:`Dense` layers.

    s:
      A tuple of integers, a direct convolutional analogue of the respective
      parameters for the :obj:`Dense` layer.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  return _Conv(out_chan, filter_shape, strides, padding, W_std, b_std,
               dimension_numbers, parameterization, s, False, False)


def _Conv(
    out_chan: int,
    filter_shape: Sequence[int],
    strides: Optional[Sequence[int]],
    padding: str,
    W_std: float,
    b_std: Optional[float],
    dimension_numbers: Optional[Tuple[str, str, str]],
    parameterization: str,
    s: Tuple[int, int],
    transpose: bool,
    shared_weights: bool
) -> InternalLayerMasked:
  """General convolution.

  Based on :obj:`jax.example_libraries.stax.GeneralConv`.

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
      The standard deviation of the biases. `None` means no bias.

    dimension_numbers:
      Specifies which axes should be convolved over. Should match the
      specification in :obj:`jax.lax.dot_general_dilated`.

    parameterization:
      Either `"ntk"` or `"standard"`. These parameterizations are the direct
      analogues for convolution of the corresponding parameterizations for
      `Dense` layers.

    s:
      A tuple of integers, a direct convolutional analogue of the respective
      parameters for the :obj:`Dense` layer.

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

  if parameterization == 'ntk':
    out_chan_arg = out_chan
  elif parameterization == 'standard':
    out_chan_arg = out_chan * s[1]
  else:
    raise ValueError(parameterization)
  init_args = dict(dimension_numbers=dimension_numbers,
                   out_chan=out_chan_arg,
                   filter_shape=filter_shape,
                   strides=strides,
                   padding=init_padding.name,
                   W_init=random.normal,
                   b_init=random.normal)

  def get_ntk_init_fn(ostax_init_fn):
    def ntk_init_fn(rng, input_shape):
      output_shape, (W, b) = ostax_init_fn(rng, input_shape)
      if b_std is None:
        b = None
      return output_shape, (W, b)
    return ntk_init_fn

  if transpose:
    if not shared_weights:
      raise NotImplementedError('Unshared transpose CNN not implemented.')

    lax_conv = lax.conv_transpose
    ostax_init_fn, _ = ostax.GeneralConvTranspose(**init_args)
    ntk_init_fn = get_ntk_init_fn(ostax_init_fn)

  else:
    if shared_weights:
      lax_conv = lax.conv_general_dilated
      ostax_init_fn, _ = ostax.GeneralConv(**init_args)
      ntk_init_fn = get_ntk_init_fn(ostax_init_fn)

    else:
      lax_conv = functools.partial(lax.conv_general_dilated_local,
                                   filter_shape=filter_shape)
      def ntk_init_fn(rng, input_shape):
        """Adapted from :obj:`jax.example_libraries.stax.GeneralConv`."""
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
        W = random.normal(k1, kernel_shape)
        b = None if b_std is None else random.normal(k2, bias_shape)
        return output_shape, (W, b)

  def get_fan_in(input_shape):
    return input_shape[lhs_spec.index('C')] * onp.prod(filter_shape)

  def standard_init_fn(rng, input_shape):
    output_shape, (W, b) = ntk_init_fn(rng, input_shape)
    norm = W_std / (get_fan_in(input_shape) / s[0])**0.5
    return output_shape, (W * norm, None if b_std is None else b * b_std)

  if parameterization == 'ntk':
    init_fn = ntk_init_fn
  elif parameterization == 'standard':
    init_fn = standard_init_fn
  else:
    raise ValueError(f'Parameterization not supported: {parameterization}.')

  def apply_fn(params, inputs, **kwargs):
    W, b = params

    if parameterization == 'ntk':
      norm = W_std / get_fan_in(inputs.shape)**0.5
      b_rescale = b_std
    elif parameterization == 'standard':
      norm = 1. / s[0]**0.5
      b_rescale = 1.
    else:
      raise NotImplementedError(parameterization)

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

    if b is not None:
      res += b_rescale * b
    return res

  @requires(batch_axis=lhs_spec.index('N'),
            channel_axis=lhs_spec.index('C'),
            diagonal_spatial=Diagonal(
                output=Bool.NO if shared_weights else Bool.MAYBE))
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
        if shift is not None:
          if k.diagonal_spatial or shared_weights:
            out += shift

          else:
            idx = (Ellipsis,)
            for i in range(batch_ndim, out.ndim, 2):
              shape = [1] * out.ndim
              size = out.shape[i]
              shape[i] = size
              idx += (np.arange(size).reshape(shape),) * 2
            out = out.at[idx].add(shift)

      return out

    b_std_sq = None if b_std is None else b_std**2

    def conv(lhs, batch_ndim):
      out = conv_unscaled(lhs, batch_ndim)
      out = affine(out, W_std**2, b_std_sq, batch_ndim)
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
        ntk = (get_fan_in(k.shape1) / s[0] * nngp_unscaled +
               W_std ** 2 * conv_unscaled(ntk, 2))
        if b_std is not None:
          ntk = affine(ntk, 1, 1., 2)
      nngp = affine(nngp_unscaled, W_std**2, b_std_sq, 2)

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

      # TODO(romann): revisit based on http://b/235531081.
      rhs = np.ones(
          rhs_shape,
          dtype=None if jax.default_backend() == 'gpu' else mask.dtype)
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
@supports_masking(remask_kernel=True)
def AvgPool(window_shape: Sequence[int],
            strides: Optional[Sequence[int]] = None,
            padding: str = Padding.VALID.name,
            normalize_edges: bool = False,
            batch_axis: int = 0,
            channel_axis: int = -1) -> InternalLayerMasked:
  """Average pooling.

  Based on :obj:`jax.example_libraries.stax.AvgPool`.

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
  return _Pool(_Pooling.AVG, window_shape, strides, padding, normalize_edges,
               batch_axis, channel_axis)


@layer
@supports_masking(remask_kernel=True)
def SumPool(window_shape: Sequence[int],
            strides: Optional[Sequence[int]] = None,
            padding: str = Padding.VALID.name,
            batch_axis: int = 0,
            channel_axis: int = -1) -> InternalLayerMasked:
  """Sum pooling.

  Based on :obj:`jax.example_libraries.stax.SumPool`.

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
  return _Pool(_Pooling.SUM, window_shape, strides, padding, False,
               batch_axis, channel_axis)


def _Pool(
    pool_type: _Pooling,
    window_shape: Sequence[int],
    strides: Optional[Sequence[int]],
    padding: str,
    normalize_edges: bool,
    batch_axis: int,
    channel_axis: int) -> InternalLayerMasked:
  """General pooling.

  Based on :obj:`jax.example_libraries.stax.AvgPool` and
  :obj:`jax.example_libraries.stax.SumPool`.

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

  if pool_type == _Pooling.AVG:
    pool_fn = ostax.AvgPool
  elif pool_type == _Pooling.SUM:
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

  elif normalize_edges or pool_type == _Pooling.SUM:
    init_fn, apply_fn = pool_fn(window_shape, strides, padding.name, spec)

  else:
    def rescaler(dims, strides, padding):
      del dims, strides, padding  # Unused.
      return lambda outputs, inputs, spec: outputs / onp.prod(window_shape)

    pool_fn = _pooling_layer(lax.add, 0., rescaler)
    init_fn, apply_fn = pool_fn(window_shape, strides, padding.name, spec)

  @requires(batch_axis=batch_axis,
            channel_axis=channel_axis,
            diagonal_spatial=Diagonal(input=Bool.MAYBE))
  def kernel_fn(k: Kernel, **kwargs) -> Kernel:
    """Kernel transformation."""
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    if k.diagonal_spatial:
      window_shape_kernel = window_shape
      strides_kernel = strides
    else:
      window_shape_kernel = utils.double_tuple(
          window_shape[::(-1 if k.is_reversed else 1)])
      strides_kernel = utils.double_tuple(strides[::(-1 if k.is_reversed else 1)])

    def pool(mat, batch_ndim):
      if mat is None or mat.ndim == 0:
        return mat

      out = _pool_kernel(mat, pool_type, window_shape_kernel, strides_kernel,
                         padding, normalize_edges, batch_ndim)

      if k.diagonal_spatial and pool_type == _Pooling.AVG:
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
def GlobalSumPool(
    batch_axis: int = 0,
    channel_axis: int = -1
) -> InternalLayerMasked:
  """Global sum pooling.

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
  return _GlobalPool(_Pooling.SUM, batch_axis, channel_axis)


@layer
@supports_masking(remask_kernel=False)
def GlobalAvgPool(
    batch_axis: int = 0,
    channel_axis: int = -1
) -> InternalLayerMasked:
  """Global average pooling.

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
  return _GlobalPool(_Pooling.AVG, batch_axis, channel_axis)


def _GlobalPool(
    pool_type: _Pooling,
    batch_axis: int,
    channel_axis: int
) -> InternalLayerMasked:
  """General global pooling.

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

  if pool_type == _Pooling.AVG:
    pool_fn = lambda x, axis, mask: mean_and_var(x, axis, mask=mask)[0]
  elif pool_type == _Pooling.SUM:
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

  @requires(batch_axis=batch_axis,
            channel_axis=channel_axis,
            diagonal_spatial=Diagonal(input=Bool.MAYBE, output=Bool.YES))
  def kernel_fn(k: Kernel, **kwargs):
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    def _pool(mat, batch_ndim, mask=None):
      if mat is None:
        return mat
      spatial_axes = tuple(range(batch_ndim, mat.ndim))
      out = pool_fn(mat, axis=spatial_axes, mask=mask)
      if k.diagonal_spatial and pool_type == _Pooling.AVG:
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
def Flatten(
    batch_axis: int = 0,
    batch_axis_out: int = 0
) -> InternalLayerMasked:
  """Flattening all non-batch dimensions.

  Based on :obj:`jax.example_libraries.stax.Flatten`, but allows to specify
  batch axes.

  Args:
    batch_axis:
      Specifies the input batch dimension. Defaults to `0`, the leading axis.

    batch_axis_out:
      Specifies the output batch dimension. Defaults to `0`, the leading axis.

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

  @requires(batch_axis=batch_axis,
            channel_axis=None,
            diagonal_spatial=Diagonal(output=Bool.YES))
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


class PositionalEmbedding(enum.Enum):
  """Type of positional embeddings to use in a :obj:`GlobalSelfAttention` layer.

  Attributes:
    NONE:
      no additional positional embeddings.

    SUM:
      positional embeddings are added to activations.

    CONCAT:
      positional embeddings are concatenated with activations.
  """
  NONE = 'NONE'
  SUM = 'SUM'
  CONCAT = 'CONCAT'


class AttentionMechanism(enum.Enum):
  """Type of nonlinearity to use in a :obj:`GlobalSelfAttention` layer.

  Attributes:
    SOFTMAX:
      attention weights are computed by passing the dot product between keys
      and queries through :obj:`jax.nn.softmax`.

    IDENTITY:
      attention weights are the dot product between keys and queries.

    ABS:
      attention weights are computed by passing the dot product between keys
      and queries through :obj:`jax.numpy.abs`.

    RELU:
      attention weights are computed by passing the dot product between keys
      and queries through :obj:`jax.nn.relu`.
  """
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
    b_std: Optional[float] = None,
    attention_mechanism: str = AttentionMechanism.SOFTMAX.name,
    pos_emb_type: str = PositionalEmbedding.NONE.name,
    pos_emb_p_norm: float = 2,
    pos_emb_decay_fn: Optional[Callable[[float], float]] = None,
    n_chan_pos_emb: Optional[int] = None,
    W_pos_emb_std: float = 1.0,
    val_pos_emb: bool = False,
    batch_axis: int = 0,
    channel_axis: int = -1) -> InternalLayerMasked:
  """Global scaled dot-product self-attention.

  Infinite width results based on
  "`Infinite attention: NNGP and NTK for deep attention networks
  <https://arxiv.org/abs/2006.10540>`_".

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
  `f_h (x) + attention_mechanism(<scaling> Q(x) K(x)^T) V(x)`
  and the output of this layer is computed as
  `f(x) = concat[f_1(x) , ... , f_{<n_{heads}>} (x)] W_{out} + b`
  where the shape of `b` is `(n_chan_out,)`, i.e., single bias per channel.

  The `kernel_fn` computes the limiting kernel of the outputs of this layer
  as the number of heads and the number of feature dimensions of keys/queries
  goes to infinity.

  For details, please see "`Infinite attention: NNGP and NTK for deep attention
  networks <https://arxiv.org/abs/2006.10540>`_".

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
      initial standard deviation of the bias values. `None` means no bias.

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

    if b_std is None:
      b = None
    else:
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
               mask: Optional[np.ndarray] = None,
               **kwargs) -> np.ndarray:
    query_matrices, key_matrices, val_matrices, W_out, b, pos_emb = params

    spatial_shape, spatial_axes = _shape_and_axes(inputs.shape,
                                                  (batch_axis, channel_axis))
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
    if b is not None:
      outputs += b_std * b
    return outputs

  @requires(batch_axis=batch_axis,
            channel_axis=channel_axis,
            diagonal_spatial=Diagonal(input=Bool.NO))
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
                            G1, G2) + 2 * (nngp if b_std is None
                                           else (nngp - b_std**2))

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
    mean, var = mean_and_var(inputs, _axis, keepdims=True, mask=mask,
                             get_var=True)
    return (inputs - mean) / np.sqrt(eps + var)

  @requires(batch_axis=batch_axis, channel_axis=channel_axis)
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

    prod11, prod12, prod22 = get_diagonal_outer_prods(
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
  """Dropout.

  Based on :obj:`jax.example_libraries.stax.Dropout`.

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

  @requires(use_dropout=True)
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


@layer
@supports_masking(remask_kernel=True)
def ImageResize(
    shape: Sequence[int],
    method: Union[str, jax.image.ResizeMethod],
    antialias: bool = True,
    precision: lax.Precision = lax.Precision.HIGHEST,
    batch_axis: int = 0,
    channel_axis: int = -1
) -> InternalLayerMasked:
  """Image resize function mimicking :obj:`jax.image.resize`.

  Docstring adapted from
  https://jax.readthedocs.io/en/latest/_modules/jax/_src/image/scale.html#resize
  Note two changes:

    1. Only `"linear"` and `"nearest"` interpolation methods are supported;

    2. Set `shape[i]` to `-1` if you want dimension `i` of `inputs` unchanged.

  The `method` argument expects one of the following resize methods:

  `ResizeMethod.NEAREST`, `"nearest"`:
    `Nearest neighbor interpolation`_. The values of `antialias` and `precision`
    are ignored.

  `ResizeMethod.LINEAR`, `"linear"`, `"bilinear"`, `"trilinear"`, `"triangle"`:
    `Linear interpolation`_. If `antialias` is `True`, uses a triangular
    filter when downsampling.

  The following methods are NOT SUPPORTED in `kernel_fn` (only `init_fn` and
  `apply_fn` work):

  `ResizeMethod.CUBIC`, `"cubic"`, `"bicubic"`, `"tricubic"`:
    `Cubic interpolation`_, using the Keys cubic kernel.

  `ResizeMethod.LANCZOS3`, `"lanczos3"`:
    `Lanczos resampling`_, using a kernel of radius 3.

  `ResizeMethod.LANCZOS5`, `"lanczos5"`:
    `Lanczos resampling`_, using a kernel of radius 5.

  .. _Nearest neighbor interpolation:
    https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation
  .. _Linear interpolation: https://en.wikipedia.org/wiki/Bilinear_interpolation
  .. _Cubic interpolation: https://en.wikipedia.org/wiki/Bicubic_interpolation
  .. _Lanczos resampling: https://en.wikipedia.org/wiki/Lanczos_resampling

  Args:
    shape:
      the output shape, as a sequence of integers with length equal to
      the number of dimensions of `image`. Note that :func:`resize` does not
      distinguish spatial dimensions from batch or channel dimensions, so this
      includes all dimensions of the image. To leave a certain dimension
      (e.g. batch or channel) unchanged, set the respective entry to `-1`.

      .. note::
        Setting a `shape` entry to the respective size of the `input` also
        works, but will make `kernel_fn` computation much more expensive with
        no benefit. Further, note that `kernel_fn` does not support resizing the
        `channel_axis`, therefore `shape[channel_axis]` should be set to `-1`.

    method:
      the resizing method to use; either a `ResizeMethod` instance or a
      string. Available methods are: `"LINEAR"`, `"NEAREST"`. Other methods
      like `"LANCZOS3"`, `"LANCZOS5"`, `"CUBIC"` only work for `apply_fn`, but
      not `kernel_fn`.

    antialias:
      should an antialiasing filter be used when downsampling? Defaults to
      `True`. Has no effect when upsampling.

    precision:
      `np.einsum` precision.

    batch_axis:
      batch axis for `inputs`. Defaults to `0`, the leading axis.

    channel_axis:
      channel axis for `inputs`. Defaults to `-1`, the trailing axis. For
      `kernel_fn`, channel size is considered to be infinite.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  def _shape(input_shape):
    return tuple(s if s != -1 else input_shape[i] for i, s in enumerate(shape))

  def init_fn(rng, input_shape):
    return _shape(input_shape), ()

  def apply_fn(params, x, **kwargs):
    return jax.image.resize(image=x,
                            shape=_shape(x.shape),
                            method=method,
                            antialias=antialias,
                            precision=precision)

  def mask_fn(mask, input_shape):
    """Behavior of interpolation with masking.

    Interpolation (except for "NEAREST") is done in float format:
    https://github.com/google/jax/issues/3811. Float converted back to bool
    rounds up all non-zero elements to `True`, so naively resizing the `mask`
    will mark any output that has at least one contribution from a masked
    input as fully masked. This can lead to mask growing unexpectedly, e.g.
    consider a 5x5 image with a single masked pixel in the center:

      >>> mask = np.array([[0, 0, 0, 0, 0],
      >>>                 [0, 0, 0, 0, 0],
      >>>                 [0, 0, 1, 0, 0],
      >>>                 [0, 0, 0, 0, 0],
      >>>                 [0, 0, 0, 0, 0]], dtype=np.bool_)

    Downsampling this mask to 2x2 will mark all output pixels as masked!

      >>> jax.image.resize(mask, (2, 2), method='bilinear').astype(np.bool_)
      DeviceArray([[ True,  True],
                   [ True,  True]], dtype=bool)

    Therefore, througout `stax` we rather follow the convention of marking
    outputs as masked if they _only_ have contributions from masked elements
    (in other words, we don't let the mask destroy information; let content
    have preference over mask). For this we invert the mask before and after
    resizing, to round up unmasked outputs instead.
    """
    return ~jax.image.resize(image=~mask,
                             shape=_shape(mask.shape),
                             method=method,
                             antialias=antialias,
                             precision=precision).astype(np.bool_)

  batch_axis, channel_axis = utils.mod((batch_axis, channel_axis), shape)

  diagonal_batch = shape[batch_axis] == -1
  diagonal_spatial = Diagonal(
      input=Bool.NO
      if any(shape[i] != -1 for i in range(len(shape))
             if i not in (batch_axis, channel_axis))
      else Bool.YES)

  @requires(batch_axis=batch_axis,
            channel_axis=channel_axis,
            diagonal_batch=diagonal_batch,
            diagonal_spatial=diagonal_spatial)
  def kernel_fn(k: Kernel, **kwargs) -> Kernel:
    if isinstance(method, str):
      _method = jax.image.ResizeMethod.from_string(method)

    if _method not in (jax.image.ResizeMethod.LINEAR,
                       jax.image.ResizeMethod.NEAREST):
      raise NotImplementedError(
          f'Only "linear" (`jax.image.ResizeMethod.LINEAR`) and '
          f'"nearest" (`jax.image.ResizeMethod.NEAREST`) interpolation is '
          f'supported in `kernel_fn`, got {_method}.')

    if shape[channel_axis] != -1:
      raise ValueError(f'Resizing the channel axis {channel_axis} is not '
                       f'well-defined in the infinite-width limit. Please '
                       f'either set `shape[channel_axis] = -1` or file '
                       f'an issue describing your use case at '
                       f'https://github.com/google/neural-tangents/issues/new.')

    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk
    diagonal_spatial = k.diagonal_spatial

    def resize(k, shape1, shape2, diagonal_batch):
      if k is None or k.ndim == 0:
        return k

      k_shape = (shape1[batch_axis],)
      if not diagonal_batch:
        k_shape += (shape2[batch_axis],)

      for i, (s1, s2) in enumerate(zip(shape1, shape2)):
        if i not in (batch_axis, channel_axis):
          k_shape += (s1,)
          if not diagonal_spatial:
            k_shape += (s2,)

      return jax.image.resize(image=k,
                              shape=k_shape,
                              method=_method,
                              antialias=antialias,
                              precision=precision)

    shape1 = _shape(k.shape1)
    shape2 = _shape(k.shape2)

    k = k.replace(cov1=resize(cov1, shape1, shape1, k.diagonal_batch),
                  nngp=resize(nngp, shape1, shape2, False),
                  cov2=resize(cov2, shape2, shape2, k.diagonal_batch),
                  ntk=resize(ntk, shape1, shape2, False))
    return k

  return init_fn, apply_fn, kernel_fn, mask_fn


@layer
@supports_masking(remask_kernel=False)
def Index(
    idx: utils.SliceType,
    batch_axis: int = 0,
    channel_axis: int = -1
) -> InternalLayerMasked:
  """Index into the array mimicking :class:`numpy.ndarray` indexing.

  Args:
    idx:
      a `slice` object that would result from indexing an array as `x[idx]`.
      To create this object, use the helper object :obj:`Slice`, i.e. pass
      `idx=stax.Slice[1:10, :, ::-1]` (which is equivalent to passing an
      explicit `idx=(slice(1, 10, None), slice(None), slice(None, None, -1)`.

    batch_axis:
      batch axis for `inputs`. Defaults to `0`, the leading axis.

    channel_axis:
      channel axis for `inputs`. Defaults to `-1`, the trailing axis. For
      `kernel_fn`, channel size is considered to be infinite.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.

  Raises:
    NotImplementedError:
      If the `channel_axis` (infinite width) is indexed
      (except for `:` or `...`) in the kernel regime (`kernel_fn`).

    NotImplementedError:
      If the `batch_axis` is indexed with an integer (as opposed to a tuple or
      slice) in the kernel regime (`kernel_fn`), since the library currently
      requires there always to be `batch_axis` in  the kernel regime (while
      indexing with integers removes the respective axis).

    ValueError:
      If `init_fn` is called on a shape with dummy axes (with sizes like `-1`
      or `None`), that are indexed with non-trivial (not `:` or `...`) slices.
      For indexing, the size of the respective axis needs to be specified.

  Example:
    >>> from neural_tangents import stax
    >>> #
    >>> init_fn, apply_fn, kernel_fn = stax.serial(
    >>>     stax.Conv(128, (3, 3)),
    >>>     stax.Relu(),
    >>>     # Select every other element from the batch (leading axis), cropped
    >>>     # to the upper-left 4x4 corner.
    >>>     stax.Index(idx=stax.Slice[::2, :4, :4])
    >>>     stax.Conv(128, (2, 2)),
    >>>     stax.Relu(),
    >>>     # Select the first row. Notice that the image becomes 1D.
    >>>     stax.Index(idx=stax.Slice[:, 0, ...])
    >>>     stax.Conv(128, (2,))
    >>>     stax.GlobalAvgPool(),
    >>>     stax.Dense(10)
    >>> )
  """
  def init_fn(rng, input_shape):
    return utils.slice_shape(input_shape, idx), ()

  def apply_fn(params, x, **kwargs):
    return x[idx]

  def mask_fn(mask, input_shape):
    return mask[idx]

  @requires(batch_axis=batch_axis, channel_axis=channel_axis)
  def kernel_fn(k: Kernel, **kwargs) -> Kernel:
    return k[idx]

  return init_fn, apply_fn, kernel_fn, mask_fn


class _Slice:

  def __getitem__(self, idx: utils.SliceType) -> utils.SliceType:
    return idx


Slice = _Slice()
"""A helper object to pass the slicing index `idx` to the :obj:`Index` layer.

Since we cannot pass slice specifications like `1, :, 2:8:3` as function
arguments, pass `Slice[1, :, 2:8:3] == (1, slice(None), slice(2, 8, 3))`
instead.
"""


# INTERNAL UTILITIES


_CONV_KERNEL_DIMENSION_NUMBERS = ('NCHW', 'OIHW', 'NCHW')


def _affine(
    mat: Optional[np.ndarray],
    W_std: float,
    b_std: Optional[float]) -> Optional[np.ndarray]:
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
      `None` means no bias.

  Returns:
    a `np.ndarray` containing sample-[sample-]position[-position] covariances
    of FC outputs. Has the same shape as `nngp`.
  """
  if mat is not None:
    mat *= W_std**2

    if b_std is not None:
      mat += b_std**2

  return mat


def _same_pad_for_filter_shape(
    x: np.ndarray,
    filter_shape: Sequence[int],
    strides: Sequence[int],
    axes: Sequence[int],
    mode: str = 'wrap',
) -> np.ndarray:
  """Padding imitating :attr:`Padding.SAME` padding with :attr:`Padding.VALID`.

  See `Returns` section for details. This function is usually needed to
    implement :attr:`Padding.CIRCULAR` padding using :attr:`Padding.VALID`
    padding.

  Args:
    x:
      `np.ndarray` to pad, e.g. a 4D `NHWC` image.

    filter_shape:
      tuple of positive integers, the convolutional filters spatial shape (e.g.
      `(3, 3)` for a 2D convolution).

    strides:
      tuple of positive integers, the convolutional spatial strides, e.g.
      `(1, 1)` for a 2D convolution.

    axes:
      tuple of non-negative integers, the spatial axes to apply convolution
      over (e.g. `(1, 2)` for an `NHWC` image).

    mode:
      a string, padding mode, for all options see
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
    total_filter_shape = utils.double_tuple(filter_shape)
    total_strides = utils.double_tuple(strides)
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
    platform = jax.default_backend()
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
  `_conv_kernel_diagonal_spatial`.

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
    total_filter_shape = utils.double_tuple(filter_shape)
    total_strides = utils.double_tuple(strides)
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
    lax_conv: Callable[
        [np.ndarray, np.ndarray, Tuple[int, ...], str], np.ndarray],
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
    pool_type: _Pooling,
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

  if pool_type == _Pooling.AVG:
    out = _normalize(lhs, out, normalize_edges, padding, strides, window_shape)

  return out


def _normalize(lhs, out, normalize_edges, padding, strides, window_shape):
  if padding == Padding.SAME and normalize_edges:
    # `SAME` padding in :obj:`jax.example_libraries.stax.AvgPool` normalizes by
    # actual window size, which is smaller at the edges.
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

  x = x.at[idx].mul(factor)
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
    x = x.at[idx].mul(factor)

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


def _vmap_2d(fn: Callable[[float, float, float], float],
             cov12: np.ndarray,
             var1: np.ndarray,
             var2: Optional[np.ndarray],
             diagonal_batch: bool,
             diagonal_spatial: bool) -> np.ndarray:
  """Effectively a "2D vmap" of `fn(cov12, var1, var2)`.

  Applicable for all possible kernel layouts.

  Args:
    fn:
      scalar-valued, elementwise `fn(cov12, var1, var2)` function to apply.

    cov12:
      covariance tensor (`q12`), `nngp`/`ntk`/`cov1`/`cov2`, of shape
      `(N1[, N2])`, `(N1[, N2], X, Y, ...)`, `(N1[, N2], X, X, Y, Y, ...)`
      depending on `diagonal_batch`, `diagonal_spatial`, and the number of
      spatial dimensions.

    var1:
      variance tensor (`q11`), has shape `(N1[, X, Y, ...])`.

    var2:
      variance tensor (`q22`), has shape `(N1[, X, Y, ...])`.

    diagonal_batch:
      `True` if `cov12` has only one batch dimension.

    diagonal_spatial:
      `True` if `cov12` has spatial dimensions appearing once (vs twice).

  Returns:
    Resulting array `[fn(cov12[i, j], var1[i], var2[j])]_{i j}`. Has the same
    shape as `cov12`.
  """
  batch_ndim = 1 if diagonal_batch else 2
  start = 2 - batch_ndim
  cov_end = batch_ndim if diagonal_spatial else cov12.ndim
  _cov12 = utils.make_2d(cov12, start, cov_end)

  var_end = 1 if diagonal_spatial else var1.ndim
  var1 = var1.reshape(var1.shape[:start] + (-1,) + var1.shape[var_end:])
  var2 = var1 if var2 is None else var2.reshape(var2.shape[:start] + (-1,) +
                                                var2.shape[var_end:])

  fn = vmap(
      vmap(
          np.vectorize(fn),
          in_axes=(start, None, start),
          out_axes=start
      ),
      in_axes=(start, start, None),
      out_axes=start
  )
  out = fn(_cov12, var1, var2)  # type: np.ndarray
  out_shape = (cov12.shape[:start] +
               cov12.shape[start:cov_end:2] +
               cov12.shape[start + 1:cov_end:2] +
               cov12.shape[cov_end:])
  out = out.reshape(out_shape)
  out = utils.zip_axes(out, start, cov_end)
  return out


# MASKING


_NEG_INF = -1e20  # softmax raises an error if all entries are -np.inf


def _check_is_implemented(mask: np.ndarray, channel_axis: int) -> None:
  if mask.shape[channel_axis] != 1:
    raise NotImplementedError(
        'Different channel-wise masks as inputs to '
        'pooling layers are not yet supported. Please '
        'let us know about your use case at '
        'https://github.com/google/neural-tangents/issues/new')


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


def _pooling_layer(reducer, init_val, rescaler=None):
  """Adapted from :obj:`jax.example_libraries.stax`."""

  def PoolingLayer(window_shape, strides=None, padding='VALID', spec=None):
    """Pooling."""
    window_shape = tuple(window_shape)
    strides = strides or (1,) * len(window_shape)
    rescale = rescaler(window_shape, strides, padding) if rescaler else None

    if spec is None:
      non_spatial_axes = 0, len(window_shape) + 1
    else:
      non_spatial_axes = spec.index('N'), spec.index('C')

    for i in sorted(non_spatial_axes):
      window_shape = window_shape[:i] + (1,) + window_shape[i:]
      strides = strides[:i] + (1,) + strides[i:]

    def init_fun(rng, input_shape):
      padding_vals = lax.padtype_to_pads(input_shape, window_shape,
                                         strides, padding)
      ones = (1,) * len(window_shape)
      out_shape = lax.reduce_window_shape_tuple(
          input_shape, window_shape, strides, padding_vals, ones, ones)
      return out_shape, ()

    def apply_fun(params, inputs, **kwargs):
      out = lax.reduce_window(inputs, init_val, reducer, window_shape,
                              strides, padding)
      return rescale(out, inputs, spec) if rescale else out
    return init_fun, apply_fun

  return PoolingLayer


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

  shape, _ = _shape_and_axes(k.shape1, (k.batch_axis, k.channel_axis))
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


def _shape_and_axes(
    x: Tuple[int, ...],
    ignore_axes: Iterable[int] = ()
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
  ndim = len(x)
  ignore_axes = tuple(i % ndim for i in ignore_axes)
  axes = tuple(i for i in range(ndim) if i not in ignore_axes)
  shape = tuple(x[i] for i in axes)
  return shape, axes
