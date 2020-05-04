# Lint as: python3

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
import operator as op
import types
from typing import Any, Optional, Union, List, Dict, Callable, Tuple

from . import dataclasses
from jax import lax
from jax.lib import xla_bridge
import jax.numpy as np
from .kernel import Kernel


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
            '`get_namedtuple` function must have a `get` argument provided or'
            'set by default.')
      else:
        get_is_not_tuple, get = canonicalize_get(defaults[get_index -
                                                          len(args)])

      fn_out = fn(*canonicalized_args, **kwargs)

      if get is None:
        if isinstance(fn_out, dict):
          ReturnType = named_tuple_factory(name, tuple(fn_out.keys()))
          fn_out = ReturnType(*fn_out.values())
        return fn_out

      fn_out = _output_to_dict(fn_out)

      if get_is_not_tuple:
        if isinstance(fn_out, types.GeneratorType):
          return (output[get[0]] for output in fn_out)
        else:
          return fn_out[get[0]]

      ReturnType = named_tuple_factory(name, get)
      if isinstance(fn_out, types.GeneratorType):
        return (ReturnType(*tuple(output[g] for g in get)) for output in fn_out)
      else:
        return ReturnType(*tuple(fn_out[g] for g in get))

    return getter_fn

  return getter_decorator


def x1_is_x2(x1, x2=None, eps=1e-12):
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


def is_array(x: Union[None, float, np.ndarray]) -> bool:
  return (isinstance(x, np.ndarray) and
          hasattr(x, 'shape') and
          x.shape != () and
          x.shape != [])


def canonicalize_axis(axis, x):
  if axis is None:
    return None

  axis = [axis] if isinstance(axis, int) else list(axis)
  if hasattr(x, 'ndim'):
    ndim = x.ndim
  elif isinstance(x, tuple):
    ndim = len(x)
  else:
    raise TypeError(x, type(x))
  return tuple(set(np.arange(ndim)[axis]))


def zip_axes(x, start_axis=0, end_axis=-1):
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


def unzip_axes(x, start_axis=0, end_axis=-1):
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


def _zip_axes(x, start_axis=0, end_axis=-1, unzip=False):
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
  if end_axis == -1:
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


def diagonal_between(x, start_axis=0, end_axis=-1):
  """Returns the diagonal along all dimensions between start and end axes."""
  if end_axis == -1:
    end_axis = x.ndim
  half_ndim, ragged = divmod(end_axis - start_axis, 2)
  if ragged:
    raise ValueError(
        f'Need even number of axes to flatten, got {end_axis - start_axis}.')
  if half_ndim == 0:
    return x

  side_shape = x.shape[start_axis:start_axis + half_ndim]
  side_size = functools.reduce(op.mul, side_shape, 1)

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


_array_or_list = Union[Optional[np.ndarray], List[Optional[np.ndarray]]]


@dataclasses.dataclass
class MaskedArray:
  masked_value: _array_or_list
  mask: _array_or_list

  def astuple(self) -> Tuple[Any, ...]:
    return dataclasses.astuple(self)


def get_masked_array(x: _array_or_list,
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
  if isinstance(x, list):
    fields = zip(*(get_masked_array(_x, mask_constant).astuple() for _x in x))  # pytype: disable=attribute-error
    return MaskedArray(*(list(f) for f in fields))

  if x is None:
    mask = None

  elif isinstance(x, MaskedArray):
    x, mask = x.astuple()

  elif isinstance(x, np.ndarray):
    if mask_constant is None:
      mask = None
    else:
      id_fn = lambda m: m
      mask = lax.cond(np.isnan(mask_constant),
                      np.isnan(x), id_fn,
                      x == mask_constant, id_fn)
  else:
    raise TypeError(x, type(x))

  if mask is not None:
    x = np.where(mask, np.zeros((), x.dtype), x)  # type: ignore

  return MaskedArray(x, mask)  # type: ignore
