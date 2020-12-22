# Copyright 2019 Google LLC
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

"""General-purpose internal utilities."""

from collections import namedtuple
import functools
import inspect
import operator
import types
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Sized, Tuple, Union
from .typing import Axes, PyTree
import warnings

from . import dataclasses
from jax import lax
from jax import random
from jax.lib import xla_bridge
import jax.numpy as np
from jax.tree_util import tree_all, tree_map
from .kernel import Kernel
import numpy as onp


def is_list_or_tuple(x):
  # We do not want to return True if x is a subclass of list or tuple since
  # otherwise this will return true for namedtuples.
  return type(x) == list or type(x) == tuple


def is_nt_tree_of(x, dtype):
  if isinstance(x, dtype):
    return True
  if not is_list_or_tuple(x):
    return False
  return all(is_nt_tree_of(_x, dtype) for _x in x)


def nt_tree_fn(nargs: int = None,
               tree_structure_argnum: int = None,
               reduce: Callable = lambda x: x):
  """Convert a function that acts on single inputs to one that acts on trees.

  `nt_tree_fn` treats the first `nargs` arguments as NTTrees and the remaining
  arguments as broadcasted over the tree structure. `nt_tree_fn` then calls the
  function on each leaf of the tree. Each node of the tree optionally calls a
  reduce function over the values of its children.

  If `tree_structure_argnum` is None then each of the NTTrees must have the same
  structure. If `tree_structure_argnum` is an integer then then a specific tree
  is used to infer the structure.

  Args:
    nargs: The number of arguments to be treated as NTTrees. If `nargs` is None
      then all of the arguments are used. `nargs` can also be negative which
      follows numpy's semantics for array indexing.
    tree_structure_argnum: The argument used to infer the tree structure to be
      traversed. If `tree_structure_argnum` is None then a check is performed to
      ensure that all trees have the same structure.
    reduce: A callable that is applied recursively by each internal tree node
      to its children.

  Returns:
    A decorator `tree_fn` that transforms a function, `fn`, from acting on
    leaves to acting on NTTrees.
  """

  def check_tree_structure(args):
    """Ensure the structure of the trees in each of the `nargs` is the same."""
    if any(is_list_or_tuple(x) for x in args):
      if not all(type(x) == type(args[0]) for x in args[1:]):
        raise TypeError(f'Inconsistent NTTree structure found. '
                        f'Node Types: {[type(x) for x in args]}.')

      """
        Regarding the use of zip, consider an example `x1 = x2 = (1, (1, 1))`.
        We would like to determine whether these two trees have the same
        structure.

        On the first recurrence `x1` and `x2` are both tuples so the check
        passes and `zip(*args) = [(1, 1), ((1, 1), (1, 1))]` so that
        `(check_tree_structure(x) for x in zip(x1, x2))` will first check that
        the first element of `x1` has the same tree structure as the first
        element of `x2` and then the second element and so on.
      """
      for x in zip(*args):
        check_tree_structure(x)

  def tree_fn(fn):
    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
      _nargs = len(args) if nargs is None else nargs
      recurse, norecurse = args[:_nargs], args[_nargs:]

      structure_argnum = tree_structure_argnum
      if structure_argnum is None:
        check_tree_structure(recurse)
        structure_argnum = 0

      if is_list_or_tuple(args[structure_argnum]):
        list_or_tuple = type(args[structure_argnum])
        return reduce(list_or_tuple(
            wrapped_fn(*(xs + norecurse), **kwargs) for xs in zip(*recurse)))
      return fn(*args, **kwargs)
    return wrapped_fn
  return tree_fn


def all_none(x: Any, attr: str = None) -> bool:
  get_fn = (lambda x: x) if attr is None else lambda x: getattr(x, attr)
  return tree_all(tree_map(lambda x: get_fn(x) is None, x))


def canonicalize_get(get):
  if get is None:
    return True, get

  if not get:
    # NOTE(schsam): It seems slightly nicer to not support the empty-tuple
    # case. Happy to add support later, if there's a use-case.
    raise ValueError('"get" must be non-empty.')

  get_is_not_tuple = isinstance(get, str)
  if get_is_not_tuple:
    get = (get,)

  get = tuple(s.lower() for s in get)
  if len(set(get)) < len(get):
    raise ValueError('All entries in "get" must be unique. Got {}'.format(get))
  return get_is_not_tuple, get


_KERNEL_NAMED_TUPLE_CACHE: Dict[Any, Any] = {}


def named_tuple_factory(name, get):
  key = (name, get)
  if key in _KERNEL_NAMED_TUPLE_CACHE:
    return _KERNEL_NAMED_TUPLE_CACHE[key]
  else:
    _KERNEL_NAMED_TUPLE_CACHE[key] = namedtuple(name, get)
    return named_tuple_factory(name, get)


def _output_to_dict(output):
  if isinstance(output, dict):
    return output

  if isinstance(output, Kernel):
    return output.asdict()

  if hasattr(output, '_asdict'):
    return output._asdict()

  if isinstance(output, types.GeneratorType):
    return (_output_to_dict(out) for out in output)

  raise ValueError(type(output))


def wraps(f):
  def wrapper(g):
    @functools.wraps(f)
    def h(*args, **kwargs):
      return g(*args, **kwargs)

    h.__signature__ = inspect.signature(f)
    return h
  return wrapper


def get_namedtuple(name):
  def getter_decorator(fn):
    try:
      argspec = inspect.getfullargspec(fn)
      get_index = argspec.args.index('get')
      defaults = argspec.defaults
    except:
      raise ValueError('`get_namedtuple` functions must have a `get` argument.')

    @wraps(fn)
    def getter_fn(*args, **kwargs):
      canonicalized_args = list(args)

      if 'get' in kwargs:
        get_is_not_tuple, get = canonicalize_get(kwargs['get'])
        kwargs['get'] = get
      elif get_index < len(args):
        get_is_not_tuple, get = canonicalize_get(args[get_index])
        canonicalized_args[get_index] = get
      elif defaults is None:
        raise ValueError(
            '`get_namedtuple` function must have a `get` argument provided or '
            'set by default.')
      else:
        get_is_not_tuple, get = canonicalize_get(defaults[get_index -
                                                          len(args)])

      fn_out = fn(*canonicalized_args, **kwargs)

      @nt_tree_fn()
      def canonicalize_output(out):
        if get is None:
          if isinstance(out, dict):
            ReturnType = named_tuple_factory(name, tuple(out.keys()))
            out = ReturnType(*out.values())
          return out

        out = _output_to_dict(out)

        if get_is_not_tuple:
          if isinstance(out, types.GeneratorType):
            return (output[get[0]] for output in out)
          else:
            return out[get[0]]

        ReturnType = named_tuple_factory(name, get)
        if isinstance(out, types.GeneratorType):
          return (ReturnType(*tuple(output[g] for g in get)) for output in out)
        else:
          return ReturnType(*tuple(out[g] for g in get))

      return canonicalize_output(fn_out)

    return getter_fn

  return getter_decorator


@nt_tree_fn(nargs=2, reduce=lambda x: np.all(np.array(x)))
def x1_is_x2(x1: np.ndarray,
             x2: np.ndarray = None,
             eps: float = 1e-12) -> Union[bool, np.ndarray]:
  if not isinstance(x1, np.ndarray):
    raise TypeError('`x1` must be an ndarray. A {} is found.'.format(type(x1)))

  if x2 is None:
    return True

  if x1 is x2:
    return True

  if x1.shape != x2.shape:
    return False

  if xla_bridge.get_backend().platform == 'tpu':
    eps = 1e-4

  return np.all(np.abs(x1 - x2) < eps)


def _get_ndim(x: Union[int, Sized, np.ndarray]) -> int:
  """Get number of dimensions given number of dimensions / shape / array."""
  if hasattr(x, 'ndim'):
    n = x.ndim
  elif hasattr(x, '__len__'):
    n = len(x)
  elif isinstance(x, int):
    n = x
  else:
    raise TypeError(x, type(x))
  return n


def mod(axis: Axes, x: Union[int, Sized, np.ndarray]) -> List[int]:
  """Makes `axis` non-negative given number of dimensions / shape / array."""
  n = _get_ndim(x)
  if isinstance(axis, int):
    axis = [axis]
  return [i % n for i in axis]


def canonicalize_axis(axis: Axes,
                      x: Union[int, Sized, np.ndarray]) -> List[int]:
  """Converts axis into a sorted non-negative list.

  Args:
    axis: input axis.
    x: array / shape / number of dimensions.

  Returns:
    A sorted list of integer axes.
  """
  axis = [axis] if isinstance(axis, int) else list(axis)
  n = _get_ndim(x)
  return list(set(onp.arange(n)[axis]))


def zip_axes(x: np.ndarray,
             start_axis: int = 0,
             end_axis: int = None) -> np.ndarray:
  """Zip (interleave) axes starting from `start_axis`.

  Changes the shape as follows:
  `[..., X, Y, Z, ..., X, Y, Z, ...] -> [..., X, X, ..., Y, Y, ..., Z, Z, ...]`

  Args:
    x: `np.ndarray` with an even number of dimensions following `start_axis`.
    start_axis: `int`, number of axis from which to zip (interleave).
    end_axis: `int`, number of axis until which to zip (interleave).

  Returns:
    A `np.ndarray` with a new shape.
  """
  return _zip_axes(x, start_axis, end_axis, unzip=False)


def unzip_axes(x: np.ndarray,
               start_axis: int = 0,
               end_axis: int = None) -> np.ndarray:
  """Unzip (de-interleave) axes starting from `start_axis`.

  Changes the shape as follows:
  `[..., X, X, ..., Y, Y, ..., Z, Z, ...] -> [..., X, Y, Z, ..., X, Y, Z, ...]`

  Args:
    x: `np.ndarray` with an even number of dimensions following `start_axis`.
    start_axis: `int`, number of axis from which to unzip (de-interleave).
    end_axis: `int`, number of axis until which to unzip (de-interleave).

  Returns:
    A `np.ndarray` with a new shape.
  """
  return _zip_axes(x, start_axis, end_axis, unzip=True)


def _zip_axes(x: np.ndarray,
              start_axis: int = 0,
              end_axis: int = None,
              unzip: bool = False) -> np.ndarray:
  """Zip/unzip (interleave/de-interleave) axes starting from `start_axis`.

  Changes the shape as follows:
    If `unzip == True`:
    `[..., X, X, ..., Y, Y, ..., Z, Z, ...] -> [..., X, Y, Z, ..., X, Y, Z, ..]`
    If `unzip == False`:
    `[..., X, Y, Z, ..., X, Y, Z, ...] -> [..., X, X, ..., Y, Y, ..., Z, Z, ..]`

  Args:
    x: `np.ndarray` with an even number of dimensions following `start_axis`.
    start_axis: `int`, number of axis from which to zip/unzip.
    end_axis: `int`, number of axis until which to zip/unzip.
    unzip: `bool`, set to `True` to unzip instead of zip.

  Returns:
    A `np.ndarray` with a new shape.
  """
  if end_axis is None:
    end_axis = x.ndim

  half_ndim, ragged = divmod(end_axis - start_axis, 2)
  if ragged:
    raise ValueError(
        f'Need even number of axes to zip, got {end_axis - start_axis}.')

  odd_axes = range(start_axis + 1, end_axis, 2)
  last_axes = range(end_axis - half_ndim, end_axis)

  if unzip:
    x = np.moveaxis(x, odd_axes, last_axes)
  else:
    x = np.moveaxis(x, last_axes, odd_axes)
  return x


def transpose_zipped(x: np.ndarray) -> np.ndarray:
  return np.moveaxis(x, range(1, x.ndim, 2), range(0, x.ndim, 2))


def diagonal_between(x: np.ndarray,
                     start_axis: int = 0,
                     end_axis: int = None) -> np.ndarray:
  """Returns the diagonal along all dimensions between start and end axes."""
  if end_axis is None:
    end_axis = x.ndim

  half_ndim, ragged = divmod(end_axis - start_axis, 2)
  if ragged:
    raise ValueError(
        f'Need even number of axes to flatten, got {end_axis - start_axis}.')
  if half_ndim == 0:
    return x

  side_shape = x.shape[start_axis:start_axis + half_ndim]
  side_size = size_at(side_shape)

  shape_2d = x.shape[:start_axis] + (side_size, side_size) + x.shape[end_axis:]
  shape_result = x.shape[:start_axis] + side_shape + x.shape[end_axis:]

  x = np.diagonal(x.reshape(shape_2d), axis1=start_axis, axis2=start_axis+1)
  x = np.moveaxis(x, -1, start_axis)
  return x.reshape(shape_result)


def zip_flat(x, y):
  return tuple(c for xy in zip(x, y) for c in xy)


def interleave_ones(x, start_axis, end_axis, x_first):
  x_axes = x.shape[start_axis:end_axis]
  ones = (1,) * (end_axis - start_axis)
  shape = x.shape[:start_axis]
  if x_first:
    shape += zip_flat(x_axes, ones)
  else:
    shape += zip_flat(ones, x_axes)
  shape += x.shape[end_axis:]
  return x.reshape(shape)


def outer_prod(x, y, start_axis, end_axis, prod_op):
  if y is None:
    y = x
  x = interleave_ones(x, start_axis, end_axis, True)
  y = interleave_ones(y, start_axis, end_axis, False)
  return prod_op(x, y)


def reverse_zipped(
    x: Union[np.ndarray, Sequence[int]],
    start_axis: int = 0) -> Union[np.ndarray, Sequence[int]]:
  if x is not None:
    ndim = _get_ndim(x)
    source_axes = tuple(j
                        for i in range(ndim - 2, start_axis - 1, -2)
                        for j in (i, i + 1))

    if isinstance(x, np.ndarray):
      target_axes = range(start_axis, ndim)
      x = np.moveaxis(x, source_axes, target_axes)
    else:
      x = x[:start_axis] + type(x)(x[i] for i in source_axes)
  return x


@dataclasses.dataclass
class MaskedArray:
  masked_value: np.ndarray
  mask: np.ndarray
  shape: Tuple[int, ...] = dataclasses.field(init=False, pytree_node=False)
  ndim: int = dataclasses.field(init=False, pytree_node=False)

  def __post_init__(self):
    super().__setattr__('shape', self.masked_value.shape)
    super().__setattr__('ndim', self.masked_value.ndim)

  astuple = ...  # type: Callable[[], Tuple[np.ndarray, np.ndarray, Tuple[int, ...], int]]


@nt_tree_fn(nargs=1)
def get_masked_array(x: np.ndarray,
                     mask_constant: float = None) -> MaskedArray:
  """Return `x` with entries equal to `mask_constant` zeroed-out, and the mask.

  The mask returned is a boolean `np.ndarray` with masked indices having `True`.

  Args:
    x: `np.ndarray` to mask. If `x` is a `MaskedInput`, treat it as
      `(masked_x, mask)` and pass it through.
    mask_constant: an optional `float`, the value in inputs to be considered as
      masked (e.g. padding in a batch of sentences). `None` means no masking.
      Can also be `np.nan`, `np.inf` etc.

  Returns:
    A `MaskedArray` of `(masked_x, boolean_mask)`.
  """

  if x is None:
    mask_mat = None

  elif isinstance(x, MaskedArray):
    x, mask_mat, _, _ = x.astuple()

  elif isinstance(x, np.ndarray):
    if mask_constant is None:
      mask_mat = None
    else:
      mask_mat = lax.cond(np.isnan(mask_constant),
                          np.isnan,
                          lambda x: x == mask_constant,
                          x)
  else:
    raise TypeError(x, type(x))

  x = mask(x, mask_mat)
  return MaskedArray(x, mask_mat)  # pytype: disable=wrong-arg-count


def mask(x: Optional[np.ndarray], mask_mat: Optional[np.ndarray]):
  if x is None or mask_mat is None:
    return x
  return np.where(mask_mat, np.zeros((), x.dtype), x)


def size_at(x: Union[np.ndarray, Sequence[int]],
            axes: Iterable[int] = None) -> int:
  if hasattr(x, 'shape'):
    x = x.shape

  if axes is None:
    axes = range(len(x))

  return functools.reduce(operator.mul, [x[a] for a in axes], 1)


def shape_and_axes(
    x: Union[np.ndarray, Sequence[int]],
    ignore_axes: Iterable[int] = ()) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
  if hasattr(x, 'shape'):
    x = x.shape
  ndim = len(x)
  ignore_axes = tuple(i % ndim for i in ignore_axes)
  axes = tuple(i for i in range(ndim) if i not in ignore_axes)
  shape = tuple(x[i] for i in axes)
  return shape, axes


def get_res_batch_dims(contracting_dims: Iterable[int],
                       batch_dims: Iterable[int]) -> List[int]:
  res_batch_dims = [2 * b - i for i, b in enumerate(batch_dims)]
  for i, b in enumerate(batch_dims):
    for c in contracting_dims:
      if b > c:
        res_batch_dims[i] -= 2
  return res_batch_dims


def dot_general(lhs: np.ndarray,
                rhs: np.ndarray,
                contracting_dims: Axes,
                batch_dims: Axes,
                precision=None) -> np.ndarray:
  """`jax.lax.dot_general` with preserved dims order and shared lhs / rhs dims.

  Precisely, returns `jax.lax.dot_general(lhs, rhs, dimension_numbers)` where
  `dimension_numbers == ((contracting_dims, contracting_dims),
                         (batch_dims, batch_dims))`,
  but preserves the dimension order in the output. See XLA's
   `DotGeneral<https://www.tensorflow.org/xla/operation_semantics#dotgeneral>`.

  Args:
    lhs: array.
    rhs: array, must have the same dimensionality as `lhs`.
    contracting_dims: contracting dimensions.
    batch_dims: batch dimensions.
    precision: Optional. Either `None`, which means the default precision for
      the backend, or a `Precision` enum value.

  Returns:
    Dot product result with preserved dimension order.
  """
  if lhs.ndim != rhs.ndim:
    raise ValueError(f'`lhs` and `rhs` must have the same dimensionality, got'
                     f'`lhs.ndim == {lhs.ndim}` and `rhs.ndim == {rhs.ndim}`.')

  contracting_dims = canonicalize_axis(contracting_dims, lhs)
  batch_dims = canonicalize_axis(batch_dims, lhs)

  n_batch_dims = len(batch_dims)
  leading_batch_dims = range(n_batch_dims)

  dimension_numbers = ((contracting_dims, contracting_dims),
                       (batch_dims, batch_dims))

  prod = lax.dot_general(lhs, rhs, dimension_numbers, precision)
  prod = zip_axes(prod, n_batch_dims)

  res_batch_dims = get_res_batch_dims(contracting_dims, batch_dims)
  prod = np.moveaxis(prod, leading_batch_dims, res_batch_dims)
  return prod


def axis_after_dot(axis: int,
                   contracting_dims: Sequence[int],
                   batch_dims: Sequence[int],
                   lhs_ndim: int = None) -> int:
  if axis in batch_dims:
    return batch_dims.index(axis)

  return (
      axis -
      sum(1 for i in contracting_dims if i < axis) +
      sum(1 for i in batch_dims if i > axis) +
      (0 if lhs_ndim is None
       else lhs_ndim - len(batch_dims) - len(contracting_dims))
  )


def conv_general_dilated_local(
    lhs: np.ndarray,
    rhs: np.ndarray,
    window_strides: Sequence[int],
    padding: str,
    filter_shape: Sequence[int],
    lhs_dilation: Sequence[int] = None,
    rhs_dilation: Sequence[int] = None,
    dimension_numbers: Tuple[str, str, str] = None,
    precision: lax.Precision = None
) -> np.ndarray:
  """General n-dimensional unshared convolution operator with optional dilation.

  Also known as locally connected layer, the operation is equivalent to
  convolution with a separate (unshared) `rhs` kernel used at each output
  spatial location. Docstring below adapted from `jax.lax.conv_general_dilated`.

  See Also:
    https://www.tensorflow.org/xla/operation_semantics#conv_convolution

  Args:
    lhs: a rank `n+2` dimensional input array.
    rhs: a rank `n+2` dimensional array of kernel weights. Unlike in regular
      CNNs, its spatial coordinates (`H`, `W`, ...) correspond to output spatial
      locations, while input spatial locations are fused with the input channel
      locations in the single `I` dimension, in the oreder of
      `"C" + ''.join(c for c in rhs_spec if c not in 'OI')` order, where
      `rhs_spec = dimension_numbers[1]`. For example, if `rhs_spec == "WHIO",
      the unfolded kernel shape is
      `"[output W][output H]{I[receptive window W][receptive window H]}O"`.
    window_strides: a sequence of `n` integers, representing the inter-window
      strides.
    padding: either the string `'SAME'`, the string `'VALID'`, or a sequence of
      `n` `(low, high)` integer pairs that give the padding to apply before and
      after each spatial dimension.
    filter_shape: a sequence of `n` integers, representing the receptive window
      spatial shape in the order as specified in
      `rhs_spec = dimension_numbers[1]`.
    lhs_dilation: `None`, or a sequence of `n` integers, giving the
      dilation factor to apply in each spatial dimension of `lhs`. LHS dilation
      is also known as transposed convolution.
    rhs_dilation: `None`, or a sequence of `n` integers, giving the
      dilation factor to apply in each input spatial dimension of `rhs`.
      RHS dilation is also known as atrous convolution.
    dimension_numbers: either `None`, a `ConvDimensionNumbers` object, or
      a 3-tuple `(lhs_spec, rhs_spec, out_spec)`, where each element is a string
      of length `n+2`.
    precision: Optional. Either ``None``, which means the default precision for
      the backend, or a ``lax.Precision`` enum value (``Precision.DEFAULT``,
      ``Precision.HIGH`` or ``Precision.HIGHEST``).

  Returns:
    An array containing the unshared convolution result.

  In the string case of `dimension_numbers`, each character identifies by
  position:

  - the batch dimensions in `lhs`, `rhs`, and the output with the character
    'N',
  - the feature dimensions in `lhs` and the output with the character 'C',
  - the input and output feature dimensions in rhs with the characters 'I'
    and 'O' respectively, and
  - spatial dimension correspondences between `lhs`, `rhs`, and the output using
    any distinct characters.

  For example, to indicate dimension numbers consistent with the `conv` function
  with two spatial dimensions, one could use `('NCHW', 'OIHW', 'NCHW')`. As
  another example, to indicate dimension numbers consistent with the TensorFlow
  Conv2D operation, one could use `('NHWC', 'HWIO', 'NHWC')`. When using the
  latter form of convolution dimension specification, window strides are
  associated with spatial dimension character labels according to the order in
  which the labels appear in the `rhs_spec` string, so that `window_strides[0]`
  is matched with the dimension corresponding to the first character
  appearing in rhs_spec that is not `'I'` or `'O'`.

  If `dimension_numbers` is `None`, the default is `('NCHW', 'OIHW', 'NCHW')`
  (for a 2D convolution).
  """
  lhs_precision = (precision[0]
                   if (isinstance(precision, tuple) and len(precision) == 2)
                   else precision)

  out = lax.conv_general_dilated_patches(
      lhs=lhs,
      filter_shape=filter_shape,
      window_strides=window_strides,
      padding=padding,
      lhs_dilation=lhs_dilation,
      rhs_dilation=rhs_dilation,
      dimension_numbers=dimension_numbers,
      precision=lhs_precision
  )
  _, rhs_spec, out_spec = dimension_numbers

  lhs_c_dims, rhs_c_dims = [out_spec.index('C')], [rhs_spec.index('I')]
  lhs_b_dims, rhs_b_dims = [], []

  for i, c in enumerate(out_spec):
    if c not in ('N', 'C'):
      lhs_b_dims += [i]
      rhs_b_dims += [rhs_spec.index(c)]

  dn = ((lhs_c_dims, rhs_c_dims), (lhs_b_dims, rhs_b_dims))
  out = lax.dot_general(out, rhs, dimension_numbers=dn)
  out = np.moveaxis(out, (-2, -1), (out_spec.index('N'), out_spec.index('C')))
  return out


def make_2d(x: Optional[np.ndarray],
            start_axis: int = 0,
            end_axis: int = None) -> Optional[np.ndarray]:
  """Makes `x` 2D from `start_axis` to `end_axis`, preserving other axes.

  `x` is assumed to follow the (`X, X, Y, Y, Z, Z`) axes layout.

  Example:
    >>> x = np.ones((1, 2, 3, 3, 4, 4))
    >>> make_2d(x).shape == (12, 24)
    >>>
    >>> make_2d(x, 2).shape == (1, 2, 12, 12)
    >>>
    >>> make_2d(x, 2, 4).shape == (1, 2, 3, 3, 4, 4)
  """
  if x is None:
    return x

  if end_axis is None:
    end_axis = x.ndim

  x = unzip_axes(x, start_axis, end_axis)

  half_ndim = (end_axis - start_axis) // 2
  x = x.reshape(x.shape[:start_axis] +
                (size_at(x.shape[start_axis:start_axis + half_ndim]),
                 size_at(x.shape[start_axis + half_ndim:end_axis])) +
                x.shape[end_axis:])
  return x


def is_on_cpu(x: PyTree) -> bool:
  def _arr_is_on_cpu(x: np.ndarray) -> bool:
    # TODO(romann): revisit when https://github.com/google/jax/issues/1431 and
    # https://github.com/google/jax/issues/1432 are fixed.
    if hasattr(x, 'device_buffer'):
      return 'cpu' in str(x.device_buffer.device()).lower()

    if isinstance(x, np.ndarray):
      return True

    raise NotImplementedError(type(x))

  return tree_all(tree_map(_arr_is_on_cpu, x))


def _read_keys(key, x1, x2):
  """Read dropout key.

  `key` might be a tuple of two rng keys or a single rng key or None. In
  either case, `key` will be mapped into two rng keys `key1` and `key2` to
  make sure `(x1==x2) == (key1==key2)`.
  """

  if key is None or all_none(x2):
    key1 = key2 = key
  elif isinstance(key, tuple) and len(key) == 2:
    key1, key2 = key
    new_key = np.where(x1_is_x2(key1, key2),
                       random.fold_in(key2, 1), key2)
    key2 = np.where(x1_is_x2(x1, x2), key1, new_key)
    warnings.warn('The value of `key[1]` might be replaced by a new value if '
                  'key[0] == key[1] and x1 != x2 or key[0] != key[1] and '
                  'x1 == x2.')
  elif isinstance(key, np.ndarray):
    key1 = key
    key2 = np.where(x1_is_x2(x1, x2), key1, random.fold_in(key, 1))
  else:
    raise TypeError(type(key))
  return key1, key2


def split_kwargs(kwargs, x1=None, x2=None):
  """Splitting `kwargs`.

     Specifically,
       1. if kwarg is an rng key, it will be split into two keys.
       2. else if it is a tuple of length two, the tuple will be split into two
          parts, one for kwargs1 and the other for kwargs2.
       3. else it is copied to kwargs1 and kwargs2.

  """
  kwargs1 = {}
  kwargs2 = {}
  for k, v in kwargs.items():
    if x2 is not None and k == 'rng':
      key1, key2 = _read_keys(v, x1, x2)
      kwargs1[k] = key1
      kwargs2[k] = key2
    elif isinstance(v, tuple) and len(v) == 2:
      kwargs1[k] = v[0]
      kwargs2[k] = v[1]
    else:
      kwargs1[k] = kwargs2[k] = v

  return kwargs1, kwargs2
