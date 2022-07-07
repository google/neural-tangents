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

"""Requirement management for :obj:`~neural_tangents.stax` layers."""

import enum
from typing import Callable, Optional, Tuple, Union, Sequence, Type
import warnings

import frozendict
import jax
from jax import lax
from jax import numpy as np
from jax import eval_shape, ShapedArray
from jax.tree_util import tree_map, tree_all
from ..utils import utils
import dataclasses
from ..utils import dataclasses as nt_dataclasses
from ..utils.kernel import Kernel
from ..utils.typing import AnalyticKernelFn, Axes, Get, InitFn, ApplyFn, InternalLayer, Layer, LayerKernelFn, NTTree, PyTree
import numpy as onp


# Public decorators


def layer(layer_fn: Callable[..., InternalLayer]) -> Callable[..., Layer]:
  """A convenience decorator to be added to all public layers.

  Used in :obj:`~neural_tangents.stax.Relu` etc.

  Makes the `kernel_fn` of the layer work with both input
  :class:`jax.numpy.ndarray` (when the layer is the first one applied to
  inputs), and with :class:`~neural_tangents.Kernel` for intermediary layers.
  Also adds optional arguments to the `kernel_fn` to allow specifying the
  computation and returned results with more flexibility.

  Args:
    layer_fn: Layer function returning triple `(init_fn, apply_fn, kernel_fn)`.

  Returns:
    A function with the same signature as `layer` with `kernel_fn` now
    accepting :class:`jax.numpy.ndarray` as inputs if needed, and accepts
    optional `get`, `diagonal_batch`, `diagonal_spatial` arguments.
  """
  name = layer_fn.__name__

  @utils.wraps(layer_fn)
  def new_layer_fns(*args, **kwargs):
    init_fn, apply_fn, kernel_fn = layer_fn(*args, **kwargs)
    kernel_fn = _preprocess_kernel_fn(init_fn, apply_fn, kernel_fn)
    init_fn.__name__ = apply_fn.__name__ = kernel_fn.__name__ = name
    return init_fn, apply_fn, kernel_fn

  return new_layer_fns


def requires(**static_reqs):
  """Returns a decorator that augments `kernel_fn` with consistency checks.

  Use this to specify your `kernel_fn` input kernel requirements.

  See Also:
    :class:`Diagonal`, :class:`Bool`.

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
                   (isinstance(v, Diagonal) and v.input == Bool.NO))):
                raise ValueError(f'{kernel_fn} requires `{key} == {v}`, but '
                                 f'input kernel has `{key} == True`, hence '
                                 f'does not contain sufficient information. '
                                 f'Please recompute the input kernel with '
                                 f'`{key} == {v}`.')

            elif key in ('batch_axis', 'channel_axis'):
              ndim = len(k.shape1)  # pytype: disable=attribute-error  # preserve-union-macros
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

    _set_req(new_kernel_fn, frozendict.frozendict(static_reqs))
    return new_kernel_fn

  return req


def supports_masking(remask_kernel: bool):
  """Returns a decorator that turns layers into layers supporting masking.

  Specifically:

  1. `init_fn` is left unchanged.

  2. `apply_fn` is turned from a function that accepts a `mask=None` keyword
  argument (which indicates `inputs[mask]` must be masked), into a function
  that accepts a `mask_constant=None` keyword argument (which indicates
  `inputs[inputs == mask_constant]` must be masked).

  3. `kernel_fn` is modified to

    3.a. propagate the `kernel.mask1` and `kernel.mask2` through intermediary
    layers, and,

    3.b. if `remask_kernel == True`, zeroes-out covariances between entries of
    which at least one is masked.

  4. If the decorated layers has a `mask_fn`, it is used to propagate masks
  forward through the layer, in both `apply_fn` and `kernel_fn`. If not, it is
  assumed the mask remains unchanged.

  Must be applied before the `layer` decorator.

  See Also:
    Example of masking application in `examples/imdb.py`.

  Args:
    remask_kernel:
      `True` to zero-out kernel covariance entries between masked inputs after
      applying `kernel_fn`. Some layers don't need this and setting
      `remask_kernel=False` can save compute.

  Returns:
    A decorator that turns functions returning
    `(init_fn, apply_fn, kernel_fn[, mask_fn])`
    into functions returning
    `(init_fn, apply_fn_with_masking, kernel_fn_with_masking)`.
  """
  def supports_masking(layer):

    @utils.wraps(layer)
    def layer_with_masking(*args, **kwargs) -> InternalLayer:
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
        masked_inputs = tree_map(
            lambda x: _get_masked_array(x, mask_constant),
            inputs,
            is_leaf=lambda x: isinstance(x, (np.ndarray, MaskedArray)))

        is_leaf = lambda x: isinstance(x, MaskedArray)
        inputs = tree_map(
            lambda x: x.masked_value,
            masked_inputs,
            is_leaf=is_leaf)
        mask = tree_map(
            lambda x: x.mask,
            masked_inputs,
            is_leaf=is_leaf)

        outputs = apply_fn(params, inputs, mask=mask, **kwargs)
        outputs_mask = mask_fn(mask,
                               inputs.shape if isinstance(inputs, np.ndarray)
                               else [i.shape for i in inputs])
        if outputs_mask is None:
          return outputs
        return MaskedArray(outputs, outputs_mask)  # pytype:disable=wrong-arg-count

      def kernel_fn_with_masking(k: NTTree[Kernel], **user_reqs):
        is_leaf = lambda k: isinstance(k, Kernel)
        mask1 = tree_map(lambda k: k.mask1, k, is_leaf=is_leaf)
        shape1 = tree_map(lambda k: k.shape1, k, is_leaf=is_leaf)
        mask2 = tree_map(lambda k: k.mask2, k, is_leaf=is_leaf)
        shape2 = tree_map(lambda k: k.shape2, k, is_leaf=is_leaf)

        mask1, mask2 = mask_fn(mask1, shape1), mask_fn(mask2, shape2)

        k = kernel_fn(k, **user_reqs)  # type: Kernel

        if remask_kernel:
          remask_fn = lambda k, m1, m2: k.mask(m1, m2)
        else:
          remask_fn = lambda k, m1, m2: k.replace(mask1=m1, mask2=m2)

        k = tree_map(remask_fn, k, mask1, mask2, is_leaf=is_leaf)
        return k

      if _has_req(kernel_fn):
        _set_req(kernel_fn_with_masking, get_req(kernel_fn))

      return init_fn, apply_fn_with_masking, kernel_fn_with_masking

    return layer_with_masking

  return supports_masking


def unmask_fn(fn: ApplyFn) -> ApplyFn:
  """Make a function returning a `MaskedArray` return a `np.ndarray`.

  Useful if you pass `masked_constant` to your `apply_fn` in order to have
  variable-length inputs. In this case `apply_fn` returns a `MaskedArray`
  that stores the information about which entries are masked (for convenient
  chaining with further functions operating on masked inputs). This decorator
  replaces the output `MaskedArray` with an `np.ndarray` where masked
  entries are zeroed-out, which is convenient to pass to functions operating on
  arrays, such as :obj:`~neural_tangents.monte_carlo_kernel_fn` or
  :obj:`~neural_tangents.empirical_kernel_fn`.

  .. warning::
    In some cases you may want to define your own custom unmasking behavior,
    e.g. one that normalizes the values based on the number of non-zero entries.

  See Also:
    :class:`MaskedArray`, and an example masking application in
    `examples/imdb.py`.

  Args:
    fn: function returning a :class:`MaskedArray`.

  Returns:
    Function of same signature as `fn`, where the output :class:`MaskedArray` is
    replaced with the :class:`jax.numpy.ndarray` with masked entries zeroed-out.
  """
  def unmask(x: Union[MaskedArray, np.ndarray]) -> np.ndarray:
    if isinstance(x, MaskedArray):
      x = utils.mask(x.masked_value, x.mask)
    return x

  def is_leaf(x) -> bool:
    return isinstance(x, (np.ndarray, MaskedArray))

  @utils.wraps(fn)
  def fn_no_mask(*args, **kwargs):
    out = fn(*args, **kwargs)
    out = tree_map(unmask, out, is_leaf=is_leaf)
    return out

  return fn_no_mask


# INTERNAL UTILITIES


@nt_dataclasses.dataclass
class MaskedArray:
  """A dataclass representing a masked :class:`jax.numpy.ndarray` or a `PyTree`.

  This type may be returned by an `apply_fn` if you provide the
  `masked_constant` argument, i.e. indicate that values of `x` equal to
  `masked_constant` are considered as masked. In this case the output of the
  `apply_fn` will be a :class:`MaskedArray`, containing information about which
  output entries are considered masked.

  See Also:
    :obj:`unmask_fn`, and an example masking application in `examples/imdb.py`.

  Attributes:
    masked_value:
      :class:`jax.numpy.ndarray` or a `PyTree` with values.

    mask:
      a boolean :class:`jax.numpy.ndarray` or a `PyTree` with `True` indicating
      that the respective entry in `masked_value` is considered masked.
  """
  masked_value: PyTree
  mask: PyTree


def _get_masked_array(
    x: Union[None, np.ndarray, ShapedArray, MaskedArray],
    mask_constant: Optional[float] = None
) -> MaskedArray:
  """Return `x` with entries equal to `mask_constant` zeroed-out, and the mask.

  The mask returned is a boolean `np.ndarray` with masked indices having `True`.

  Args:
    x:
      `np.ndarray` to mask. If `x` is a :class:`MaskedArray`, treat it as
      `(masked_x, mask)` and pass it through.

    mask_constant: an optional `float`, the value in inputs to be considered as
      masked (e.g. padding in a batch of sentences). `None` means no masking.
      Can also be `np.nan`, `np.inf` etc.

  Returns:
    A :class:`MaskedArray` of `(masked_x, boolean_mask)`.
  """

  if x is None:
    mask_mat = None

  elif isinstance(x, MaskedArray):
    x, mask_mat = x.masked_value, x.mask

  elif isinstance(x, (onp.ndarray, np.ndarray, float, int)):
    if mask_constant is None:
      mask_mat = None
    else:
      mask_mat = lax.cond(np.isnan(mask_constant),
                          np.isnan,
                          lambda x: x == mask_constant,
                          x)
  else:
    raise TypeError(x, type(x))

  x = utils.mask(x, mask_mat)
  return MaskedArray(x, mask_mat)  # pytype: disable=wrong-arg-count


_INPUT_REQ = 'input_req'


def get_req(
    f: Callable,
    default: Optional[frozendict.frozendict] = None) -> frozendict.frozendict:
  return getattr(f, _INPUT_REQ, default)


def _set_req(f: Callable, req: frozendict.frozendict):
  setattr(f, _INPUT_REQ, req)


def _has_req(f: Callable) -> bool:
  return hasattr(f, _INPUT_REQ)


_DEFAULT_INPUT_REQ = frozendict.frozendict(
    {
        'diagonal_batch': True,
        'diagonal_spatial': False,
        'batch_axis': 0,
        'use_dropout': False,
        'channel_axis': -1,
        'mask_constant': None
    }
)


class Bool(enum.IntEnum):
  """Helper trinary logic class. See :class:`Diagonal` for details.

  Attributes:
    NO:
      `False`.

    MAYBE:
      Maybe.

    YES:
      `True`.

  """
  NO = 0
  MAYBE = 1
  YES = 2

  def __and__(self, other: 'Bool') -> 'Bool':
    return min(self, other)

  __rand__ = __and__


@dataclasses.dataclass(frozen=True)
class Diagonal:
  """Helps decide whether to allow the kernel to contain diagonal entries only.

  The intended behavior is to be diagonal-only iff
    a) output off-diagonal entries are all zeros, and

    b) diagonal-only :class:`~neural_tangents.Kernel` is sufficient for all
    steps of computation.

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
      entries. :attr:`Bool.YES` means "yes"; :attr:`Bool.MAYBE` means iff
      off-diagonal entries are zero. :attr:`Bool.NO` means "no". When
      traversing the network tree from inputs to outputs (as well as parallel
      branches from left/right to right/left) can only decrease.

    output:
      specifies whether any outputs (starting from this layer to the output of
      the network) can contain only diagonal entries. :attr:`Bool.YES` means
      yes; :attr:`Bool.MAYBE` means "yes" after current layer, but may become
      "no" further in the network. :attr:`Bool.NO` means "no".
  """

  input: Bool = Bool.YES
  output: Bool = Bool.NO

  def __rshift__(self, other: 'Diagonal') -> 'Diagonal':
    """Associative composition (`self >> other`) operation.

    Args:
      other:
        lhs.

    Returns:
      The requirement satisfied by composition `other(self(.))`.
    """
    if self.output == Bool.YES:
      return self

    if self.output > Bool.NO and other.input > Bool.NO:
      input = self.input
    elif self.output == Bool.NO and other.input < Bool.YES:
      input = Bool.NO
    else:
      input = min(self.input, other.input)

    return Diagonal(input=input, output=other.output)

  def __and__(self, other: 'Diagonal') -> 'Diagonal':
    """Commutative, associative, and idempotent `AND` operation.

    Args:
      other:
        lhs/rhs.

    Returns:
       The largest value allowed both `self` and `other`.
    """
    return Diagonal(input=self.input & other.input,
                    output=self.output & other.output)

  def __bool__(self) -> bool:
    """Convert to `diagonal_spatial` / `diagonal_batch` `Kernel` attribute."""
    return self.input == Bool.YES and self.output > Bool.NO

  def __lshift__(self, other: 'Diagonal') -> 'Diagonal':
    """Associative composition (`self << other`) operation.

    Args:
      other:
        lhs.

    Returns:
      The value allowed by composition `self(other(.))`.
    """
    return other >> self

  __rand__ = __and__


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
    x1:
      a (2+S)D (S >= 0) `np.ndarray` of shape
      `(batch_size_1, <S spatial dimensions>, n_channels)`. `batch_size_1`,
      `n_channels` may be in different positions based on `batch_axis` and
      `channel_axis`.

    x2:
      an optional `np.ndarray` that has the same shape as `a` apart from
      possibly different batch (`batch_size_2`) dimension. `None` means
      `x2 == x1`.

    diagonal_spatial:
      Specifies whether only the diagonals of the
      location-location covariances will be computed,
      (`diagonal_spatial == True`,
       `nngp.shape == (batch_size_1, batch_size_2, height, width, depth, ...)`),
      or the full covariance
      (`diagonal_spatial == False`,
       `nngp.shape == (batch_size_1, batch_size_2, height, height,
                       width, width, depth, depth, ...)`).

    batch_axis:
      Specifies which axis is the batch axis.

    channel_axis:
      Specifies which axis is the channel / feature axis. For `kernel_fn`,
      channel size is considered to be infinite.

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


def _inputs_to_kernel(
    x1: np.ndarray,
    x2: Optional[np.ndarray],
    *,
    diagonal_batch: bool,
    diagonal_spatial: Union[bool, Diagonal],
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
    >>>                      diagonal_batch=True,
    >>>                      diagonal_spatial=False,
    >>>                      compute_ntk=True,
    >>>                      batch_axis=0,
    >>>                      channel_axis=-1)
    >>> o.cov1.shape, o.ntk.shape
    (10, 32, 32, 16, 16), (10, 10, 32, 32, 16, 16)
    >>> o = _inputs_to_kernel(x, None,
    >>>                      diagonal_batch=True,
    >>>                      diagonal_spatial=True,
    >>>                      compute_ntk=True,
    >>>                      batch_axis=0,
    >>>                      channel_axis=-1)
    >>> o.cov1.shape, o.ntk.shape
    (10, 32, 16), (10, 10, 32, 16)
    >>> x1 = np.ones((10, 128))
    >>> x2 = np.ones((20, 128))
    >>> o = _inputs_to_kernel(x1, x2,
    >>>                      diagonal_batch=True,
    >>>                      diagonal_spatial=True,
    >>>                      compute_ntk=False,
    >>>                      batch_axis=0,
    >>>                      channel_axis=-1)
    >>> o.cov1.shape, o.nngp.shape
    (10,), (10, 20)

  Args:
    x1:
      an `(S+2)`-dimensional `np.ndarray` of shape
      `(batch_size_1, height, width, depth, ..., n_channels)` with `S` spatial
      dimensions (`S >= 0`). Dimensions may be in different order based on
      `batch_axis` and `channel_axis`.

    x2:
      an optional `np.ndarray` with the same shape as `x1` apart from possibly
      different batch size. `None` means `x2 == x1`.

    diagonal_batch:
      Specifies whether `cov1` and `cov2` store only
      the diagonal of the sample-sample covariance
      (`diagonal_batch == True`,
       `cov1.shape == (batch_size_1, ...)`),
      or the full covariance
      (`diagonal_batch == False`,
       `cov1.shape == (batch_size_1, batch_size_1, ...)`).

    diagonal_spatial:
      Specifies whether all (`cov1`, `ntk`, etc.) input covariance matrcies
      should store only the diagonals of the location-location covariances
      (`diagonal_spatial == True`,
       `nngp.shape == (batch_size_1, batch_size_2, height, width, depth, ...)`),
      or the full covariance
      (`diagonal_spatial == False`,
       `nngp.shape == (batch_size_1, batch_size_2, height, height,
                       width, width, depth, depth, ...)`).

    compute_ntk:
      `True` to compute both NTK and NNGP kernels, `False` to only compute NNGP.

    batch_axis:
      Specifies which axis is the batch axis.

    channel_axis:
      Specifies which axis is the channel / feature axis. For `kernel_fn`,
      channel size is considered to be infinite.

    mask_constant:
      an optional `float`, the value in inputs to be considered as masked (e.g.
      padding in a batch of sentences). `None` means no masking. Can also be
      `np.nan`, `np.inf` etc. Beware of floating point precision errors and try
      to use an atypical for inputs value.

    eps:
      a small number used to check whether x1 and x2 are the same up to `eps`.

    **kwargs:
      other arguments passed to all intermediary `kernel_fn` calls (not used
      here).

  Returns:
    The :class:`~neural_tangents.Kernel` object containing inputs covariance[s].
  """

  if not (isinstance(x1, (onp.ndarray, np.ndarray)) and
          (x2 is None or isinstance(x2, (onp.ndarray, np.ndarray)))):
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

    x = _get_masked_array(x, mask_constant)
    x, mask = x.masked_value, x.mask

    # TODO(schsam): Think more about dtype automatic vs manual dtype promotion.
    x = x.astype(jax.dtypes.canonicalize_dtype(np.float64))

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

  if isinstance(shaped, MaskedArray):
    shaped = shaped.masked_value

  return shaped


def _set_shapes(init_fn: InitFn,
                apply_fn: ApplyFn,
                in_kernel: NTTree[Kernel],
                out_kernel: NTTree[Kernel],
                **kwargs
                ) -> NTTree[Kernel]:
  """Apply a kernel_fn to a Kernel propagating side information."""
  is_leaf = lambda k: isinstance(k, Kernel)

  shape1 = tree_map(lambda k: ShapedArray(k.shape1, k.nngp.dtype),
                    in_kernel, is_leaf=is_leaf)
  shape2 = tree_map(lambda k: ShapedArray(k.shape2, k.nngp.dtype),
                    in_kernel, is_leaf=is_leaf)

  kwargs1, kwargs2 = utils.split_kwargs(kwargs)

  shape1 = _propagate_shape(init_fn, unmask_fn(apply_fn), shape1, **kwargs1)
  shape2 = _propagate_shape(init_fn, unmask_fn(apply_fn), shape2, **kwargs2)

  set_shape_fn = lambda k, s1, s2: k.replace(shape1=s1.shape, shape2=s2.shape)

  return tree_map(set_shape_fn, out_kernel, shape1, shape2, is_leaf=is_leaf)


def _fuse_requirements(
    kernel_fn_reqs,
    default_reqs,
    **user_reqs
) -> frozendict.frozendict:
  # Override static requirements with explicit user-specified requirements,
  # but only if they are less demanding, raise an error otherwise.
  kernel_fn_reqs = dict(kernel_fn_reqs)
  for k, v_user in user_reqs.items():
    if v_user is not None:
      if k in kernel_fn_reqs:
        v_kernel = kernel_fn_reqs[k]
        if (v_user is True and
            (v_kernel is False or
             (isinstance(kernel_fn_reqs[k], Diagonal) and
              kernel_fn_reqs[k].input == Bool.NO))):
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
    kernel_fn: LayerKernelFn
) -> AnalyticKernelFn:
  """Returns a `kernel_fn` with additional arguments.

  Args:
    init_fn: layer parameters initialization function. Used for shape inference.

    apply_fn: layer forward-prop function. Used for shape inference.

    kernel_fn: the `Kernel` -> `Kernel` layer propagation function.

  Returns:
    A new `kernel_fn` that does the same computation but accepts additional
    arguments to flexibly specify the required computation, and can be applied
    to either a `Kernel' or a pair of `np.ndarrray`s.
  """
  # Set empty requirements if none specified.
  if not _has_req(kernel_fn):
    kernel_fn = requires()(kernel_fn)

  def kernel_fn_kernel(kernel, **kwargs):
    out_kernel = kernel_fn(kernel, **kwargs)
    return _set_shapes(init_fn, apply_fn, kernel, out_kernel, **kwargs)

  def kernel_fn_x1(x1, x2, get, **kwargs):
    # Get input requirements requested by network layers, user, or defaults.
    kernel_fn_reqs = get_req(kernel_fn)
    reqs = _fuse_requirements(kernel_fn_reqs, _DEFAULT_INPUT_REQ, **kwargs)
    compute_ntk = (get is None) or ('ntk' in get)

    if x2 is None:
      x2 = tree_map(lambda x: None, x1)

    def input_fn(x1, x2):
      return _inputs_to_kernel(x1, x2, compute_ntk=compute_ntk, **reqs)
    kernel = tree_map(input_fn, x1, x2)

    out_kernel = kernel_fn(kernel, **kwargs)
    return _set_shapes(init_fn, apply_fn, kernel, out_kernel, **kwargs)

  @utils.get_namedtuple('AnalyticKernel')
  def kernel_fn_any(x1_or_kernel: Union[NTTree[np.ndarray], NTTree[Kernel]],
                    x2: Optional[NTTree[np.ndarray]] = None,
                    get: Optional[Get] = None,
                    *,
                    pattern: Optional[Tuple[Optional[np.ndarray],
                                            Optional[np.ndarray]]] = None,
                    mask_constant: Optional[float] = None,
                    diagonal_batch: Optional[bool] = None,
                    diagonal_spatial: Optional[bool] = None,
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
    def all_of(x, cls: Type) -> bool:

      def is_leaf(x) -> bool:
        return isinstance(x, (Kernel, np.ndarray, onp.ndarray))

      return tree_all(
          tree_map(
              lambda x: isinstance(x, cls),
              x,
              is_leaf=is_leaf)
          )

    if all_of(x1_or_kernel, Kernel) and x2 is None:
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

  _set_req(kernel_fn_any, get_req(kernel_fn))
  return kernel_fn_any


def get_diagonal(
    cov: Optional[np.ndarray],
    diagonal_batch: bool,
    diagonal_spatial: bool
) -> Optional[np.ndarray]:
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


def get_diagonal_outer_prods(
    cov1: np.ndarray,
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

  cov1 = get_diagonal(cov1, diagonal_batch, diagonal_spatial)
  cov2 = get_diagonal(cov2, diagonal_batch, diagonal_spatial)

  cov1, _ = mean_and_var(cov1, axis=axis, keepdims=True, mask=mask1)
  cov2, _ = mean_and_var(cov2, axis=axis, keepdims=True, mask=mask2)

  end_axis = 1 if diagonal_spatial else cov1.ndim
  prod12 = utils.outer_prod(cov1, cov2, 0, end_axis, operation)

  start_axis = 1 if diagonal_batch else 0
  prod11 = utils.outer_prod(cov1, cov1, start_axis, end_axis, operation)
  prod22 = (utils.outer_prod(cov2, cov2, start_axis, end_axis, operation)
            if cov2 is not None else prod11)

  return prod11, prod12, prod22


def mean_and_var(
    x: Optional[np.ndarray],
    axis: Optional[Axes] = None,
    dtype: Optional[np.dtype] = None,
    out: Optional[None] = None,
    ddof: int = 0,
    keepdims: bool = False,
    mask: Optional[np.ndarray] = None,
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

