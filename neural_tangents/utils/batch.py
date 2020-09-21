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

"""Batch kernel computations serially or in parallel.

This module contains a decorator `batch` that can be applied to any `kernel_fn`
of signature `kernel_fn(x1, x2, *args, **kwargs)`. The decorated function
performs the same computation by batching over `x1` and `x2` and concatenating
the result, allowing to both use multiple accelerators and stay within memory
limits.

Note that you typically should not apply the `jax.jit` decorator to the
resulting `batched_kernel_fn`, as its purpose is explicitly serial execution in
order to save memory. Further, you do not need to apply `jax.jit` to the input
`kernel_fn` function, as it is JITted internally.

Example:
  >>>  from jax import numpy as np
  >>>  import neural_tangents as nt
  >>>  from neural_tangents import stax
  >>>
  >>>  # Define some kernel function.
  >>>  _, _, kernel_fn = stax.serial(stax.Dense(1), stax.Relu(), stax.Dense(1))
  >>>
  >>>  # Compute the kernel in batches, in parallel.
  >>>  kernel_fn_batched = nt.batch(kernel_fn, device_count=-1, batch_size=5)
  >>>
  >>>  # Generate dummy input data.
  >>>  x1, x2 = np.ones((40, 10)), np.ones((80, 10))
  >>>  kernel_fn_batched(x1, x2) == kernel_fn(x1, x2)  # True!
"""


from typing import Callable, Tuple, Union, Dict, Any, TypeVar, Iterable, Optional
from functools import partial
import warnings
from jax.api import device_put, devices
from jax.api import jit
from jax.api import pmap
from jax.interpreters.pxla import ShardedDeviceArray
from jax.lib import xla_bridge
from jax import random
import jax.numpy as np
from jax.tree_util import tree_all
from jax.tree_util import tree_map
from jax.tree_util import tree_multimap, tree_flatten, tree_unflatten
from neural_tangents.utils.kernel import Kernel
from neural_tangents.utils import utils
from neural_tangents.utils.typing import KernelFn, NTTree

import numpy as onp


def batch(kernel_fn: KernelFn,
          batch_size: int = 0,
          device_count: int = -1,
          store_on_device: bool = True) -> KernelFn:
  """Returns a function that computes a kernel in batches over all devices.

  Note that you typically should not apply the `jax.jit` decorator to the
  resulting `batched_kernel_fn`, as its purpose is explicitly serial execution
  in order to save memory. Further, you do not need to apply `jax.jit` to the
  input `kernel_fn` function, as it is JITted internally.

  Args:
    kernel_fn:
      A function that computes a kernel on two batches,
      `kernel_fn(x1, x2, *args, **kwargs)`. Here `x1` and `x2` are
      `np.ndarray`s of shapes `(n1,) + input_shape` and `(n2,) + input_shape`.
      The kernel function should return a `PyTree`.
    batch_size:
      specifies the size of each batch that gets processed per physical device.
      Because we parallelize the computation over columns it should be the case
      that `x1.shape[0]` is divisible by `device_count * batch_size` and
      `x2.shape[0]` is divisible by `batch_size`.
    device_count:
      specifies the number of physical devices to be used. If
      `device_count == -1` all devices are used. If `device_count == 0`, no
      device parallelism is used (a single default device is used).
    store_on_device:
      specifies whether the output should be kept on device or brought back to
      CPU RAM as it is computed. Defaults to `True`. Set to `False` to store
      and concatenate results using CPU RAM, allowing to compute larger kernels.

  Returns:
    A new function with the same signature as `kernel_fn` that computes the
    kernel by batching over the dataset in parallel with the specified
    `batch_size` using `device_count` devices.
  """
  input_req = getattr(kernel_fn, 'input_req', {})
  dropout_in_analytic_kernel = input_req.get('use_dropout', False)
  use_multidevice = device_count > 0 or (device_count == -1 and
                                         xla_bridge.device_count() > 1)
  use_serial = bool(batch_size)
  if use_multidevice:
    kernel_fn = _parallel(kernel_fn, use_serial,
                          dropout_in_analytic_kernel, device_count)
  else:
    kernel_fn = _jit_or_pmap_broadcast(kernel_fn, 0)

  if not use_serial:
    return kernel_fn

  return _serial(kernel_fn, batch_size, store_on_device)


# INTERNAL UTILITIES


_Carry = TypeVar('_Carry')
_Input = TypeVar('_Input')
_Output = TypeVar('_Output')


def _scan(f: Callable[[_Carry, _Input], Tuple[_Carry, _Output]],
          init: _Carry,
          xs: Iterable[_Input]) -> Tuple[_Carry, _Output]:
  """Implements an unrolled version of scan.

  Based on `jax.lax.scan` and has a similar API.

  TODO(schsam): We introduce this function because lax.scan currently has a
  higher peak memory usage than the unrolled version. We will aim to swap this
  out for lax.scan when issue #1273 and related have been resolved.
  """
  carry = init
  ys = []
  flat_xs, tree_def = tree_flatten(xs)
  for flat_x in zip(*flat_xs):
    x = tree_unflatten(tree_def, flat_x)
    carry, y = f(carry, x)
    ys += [y]

  return carry, tree_multimap(lambda *y: np.stack(y), *ys)


def _flatten_batch_dimensions(k: np.ndarray,
                              discard_axis: int = None) -> np.ndarray:
  """Takes a kernel that has been evaluated in batches and flattens."""

  if discard_axis is not None:
    if k.ndim % 2:
      k = np.take(k, 0, axis=discard_axis)
      return np.reshape(k, (-1,) + k.shape[2:])

    if discard_axis == 1:
      return np.reshape(k, (k.shape[0] * k.shape[1],) + k.shape[2:])

    return k[0]

  else:
    if k.ndim % 2:
      return np.reshape(k, (k.shape[0] * k.shape[1],) + k.shape[2:])

    k = np.transpose(k, (0, 2, 1, 3) + tuple(range(4, k.ndim)))
    return np.reshape(k,
                      (k.shape[0] * k.shape[1],
                       k.shape[2] * k.shape[3]) + k.shape[4:])


@utils.nt_tree_fn(nargs=1)
def _flatten_kernel_dict(k: Dict[str, Any],
                         x2_is_none: bool,
                         is_parallel: bool) -> Dict[str, Any]:
  if 'nngp' in k:
    # We only use `batch_size` to compute `shape1` and `shape2` for the batch.
    # This only happens if k_dict came from a `Kernel` in which case it must
    # have 'nngp'. I do think there is a failure case if the user called
    # >>> batched_kernel_fn(x1, x2, get=('ntk', 'shape1'))
    # but I don't think this will get hit ever (and certainly before we rework
    # this code).
    batch_size = {'1': k['nngp'].shape[0], '2': k['nngp'].shape[1]}

  if 'diagonal_batch' in k and not k['diagonal_batch']:
    raise NotImplementedError('Batching not implemented for '
                              '`diagonal_batch == False`.')

  for key, value in k.items():
    if key == 'cov1':
      k[key] = _flatten_batch_dimensions(value, 1)
    elif key == 'cov2':
      if x2_is_none:
        k[key] = None
      else:
        k[key] = _flatten_batch_dimensions(value, 0)
    elif key == 'x1_is_x2':
      k[key] = value[(0,) * value.ndim]
    elif key == 'mask1':
      if value is None:
        k[key] = None
      else:
        k[key] = _flatten_batch_dimensions(value, 1)
    elif key == 'mask2':
      if value is None or x2_is_none:
        k[key] = None
      else:
        k[key] = -_flatten_batch_dimensions(value, 0)
    elif key in ('shape1', 'shape2'):
      if key == 'shape2' and is_parallel:
        continue
      batch_axis = k['batch_axis']
      shape = value
      k[key] = (shape[:batch_axis] +
                (shape[batch_axis] * batch_size[key[-1]],) +
                shape[batch_axis + 1:])
    elif isinstance(k[key], np.ndarray):
      k[key] = _flatten_batch_dimensions(value)
    else:
      pass
  return k


@utils.nt_tree_fn(nargs=1)
def _flatten_kernel(k: Kernel,
                    x2_is_none: bool,
                    is_parallel: bool) -> Kernel:
  """Flattens a kernel array or a `Kernel` along the batch dimension."""

  # pytype: disable=attribute-error
  if hasattr(k, '_asdict'):
    return k._replace(**_flatten_kernel_dict(k._asdict(), x2_is_none,
                                             is_parallel))

  elif isinstance(k, Kernel):
    return Kernel(**_flatten_kernel_dict(k.asdict(), x2_is_none, is_parallel))
  # pytype:enable=attribute-error

  elif isinstance(k, np.ndarray):
    return _flatten_batch_dimensions(k)

  raise TypeError(f'Expected kernel to be either a namedtuple, `Kernel`, or '
                  f'`np.ndarray`, got {type(k)}.')


@utils.nt_tree_fn(nargs=1)
def _reshape_kernel_for_pmap(k: Kernel,
                             device_count: int,
                             n1_per_device: int) -> Kernel:
  # pytype: disable=attribute-error
  cov2 = k.cov2
  if cov2 is None:
    cov2 = k.cov1
  cov2 = np.broadcast_to(cov2, (device_count,) + cov2.shape)

  mask2 = k.mask2
  if mask2 is None and k.mask1 is not None:
    mask2 = k.mask1
  if mask2 is not None:
    mask2 = np.broadcast_to(mask2, (device_count,) + mask2.shape)

  x1_is_x2 = np.broadcast_to(k.x1_is_x2, (device_count,) + k.x1_is_x2.shape)

  nngp, ntk, cov1 = [
      np.reshape(x, (device_count, n1_per_device,) + x.shape[1:]) for x in
      (k.nngp, k.ntk, k.cov1)]

  return k.replace(
      nngp=nngp,
      ntk=ntk,
      cov1=cov1,
      cov2=cov2,
      x1_is_x2=x1_is_x2,
      shape1=(n1_per_device,) + k.shape1[1:],
      mask2=mask2)


@utils.nt_tree_fn()
def _set_cov2_is_none(k: Kernel) -> Kernel:
  return k.replace(cov2=None)
  # pytype: enable=attribute-error


def _serial(kernel_fn: KernelFn,
            batch_size: int,
            store_on_device: bool = True) -> KernelFn:
  """Returns a function that computes a kernel in batches serially.

  This function computes the kernel over data in batches where each batch is
  processed sequentially with a given batch size. If serial detects that the
  kernel function is the result of `_parallel` (that is, if the kernel is
  distributed over multiple devices) then serial adjusts the batch size so that
  each device processes chunks of work that have batch_size x batch_size.

  The dataset size must divide the effective batch size. If parallelism is used
  this means that `|x1|` must divide `batch_size * device_count` and `|x2|` must
  divide `batch_size`.

  Args:
    kernel_fn:
      A function that computes a kernel between two datasets,
      `kernel_fn(x1, x2)` or the compositional kernel for an input kernel
      `kernel_fn(kernel_in)`. Here x1 and x2 are `np.ndarray`s of floats of
      shape `(n1,) + input_shape` and `(n2,) + input_shape`; `kernel_in` is a
      `Kernel` object. The kernel function should return a `PyTree`.
    batch_size:
      Integer specifying the size of batches in which to split the data.
    store_on_device:
      A boolean that species whether the computed kernel should be kept on
      device or brought back to CPU as it is computed. Defaults to `True`.

  Returns:
    A new function with the same signature as kernel_fn that computes the kernel
    by batching over the dataset serially with the specified batch_size.
  """

  device_count = max(getattr(kernel_fn, 'device_count', 1), 1)

  if not store_on_device:
    _kernel_fn = kernel_fn
    @utils.wraps(_kernel_fn)
    def kernel_fn(x1, x2=None, *args, **kwargs):
      return device_put(_kernel_fn(x1, x2, *args, **kwargs), devices('cpu')[0])

  flatten = partial(_flatten_kernel, is_parallel=False)

  def serial_fn_x1(x1: NTTree[np.ndarray],
                   x2: NTTree[Optional[np.ndarray]] = None,
                   *args,
                   **kwargs) -> NTTree[Kernel]:

    x2_is_none = utils.all_none(x2)
    if x2_is_none:
      # TODO(schsam): Only compute the upper triangular part of the kernel.
      x2 = x1

    @utils.nt_tree_fn(reduce=lambda x: x[0])
    def get_n1_n2(x1, x2):
      n1, n2 = x1.shape[0], x2.shape[0]
      return n1, n2
    n1, n2 = get_n1_n2(x1, x2)

    (n1_batches, n1_batch_size, n2_batches, n2_batch_size) = \
        _get_n_batches_and_batch_sizes(n1, n2, batch_size, device_count)

    @utils.nt_tree_fn(nargs=1)
    def batch_input(x, batch_count, batch_size):
      input_shape = x.shape[1:]
      return np.reshape(x, (batch_count, batch_size,) + input_shape)

    x1s = batch_input(x1, n1_batches, n1_batch_size)
    x2s = batch_input(x2, n2_batches, n2_batch_size)

    kwargs_np1 = {}
    kwargs_np2 = {}
    kwargs_other = {}

    for k, v in kwargs.items():
      if _is_np_ndarray(v):
        if k == 'rng':
          key1, key2 = random.split(v)
          v1 = random.split(key1, n1_batches)
          v2 = random.split(key2, n2_batches)
        else:
          assert isinstance(v, tuple) and len(v) == 2
          v1 = np.reshape(v[0], (n1_batches, n1_batch_size,) + v[0].shape[1:])
          v2 = np.reshape(v[1], (n2_batches, n2_batch_size,) + v[1].shape[1:])
        kwargs_np1[k] = v1
        kwargs_np2[k] = v2
      else:
        kwargs_other[k] = v

    def row_fn(_, x1):
      return _, _scan(col_fn, x1, (x2s, kwargs_np2))[1]

    def col_fn(x1, x2):
      x1, kwargs1 = x1
      x2, kwargs2 = x2
      kwargs_merge = {
          **kwargs_other,
          **dict((k, (kwargs1[k], kwargs2[k])) for k in kwargs1)
      }
      return (x1, kwargs1), kernel_fn(x1, x2, *args, **kwargs_merge)

    _, kernel = _scan(row_fn, 0, (x1s, kwargs_np1))
    return flatten(kernel, x2_is_none)

  def serial_fn_kernel(k: NTTree[Kernel], *args, **kwargs) -> NTTree[Kernel]:
    # pytype: disable=attribute-error
    def get_n1_n2(k):
      if utils.is_list_or_tuple(k):
        # TODO(schsam): We might want to check for consistency here, but I can't
        # imagine a case where we could get inconsistent kernels.
        return get_n1_n2(k[0])
      return k.nngp.shape[:2]
    # pytype: enable=attribute-error
    n1, n2 = get_n1_n2(k)

    (n1_batches, n1_batch_size,
     n2_batches, n2_batch_size) = _get_n_batches_and_batch_sizes(n1, n2,
                                                                 batch_size,
                                                                 device_count)

    n1s = np.arange(0, n1, n1_batch_size)
    n2s = np.arange(0, n2, n2_batch_size)

    @utils.nt_tree_fn(nargs=1)
    def slice_kernel(k, n1, n2):
      return k.slice(n1, n2)

    kwargs_np1 = {}
    kwargs_np2 = {}

    kwargs_other = {}
    for key, v in kwargs.items():
      if _is_np_ndarray(v):
        assert isinstance(v, tuple) and len(v) == 2
        v1 = np.reshape(v[0], (n1_batches, n1_batch_size,) + v[0].shape[1:])
        v2 = np.reshape(v[1], (n2_batches, n2_batch_size,) + v[1].shape[1:])
        kwargs_np1[key] = v1
        kwargs_np2[key] = v2
      else:
        kwargs_other[key] = v

    def row_fn(_, n1):
      return _, _scan(col_fn, n1, (n2s, kwargs_np2))[1]

    def col_fn(n1, n2):
      # NOTE(schsam): If we end up wanting to enable jit-of-batch then we will
      # probably have to change this to dynamic slicing.
      n1, kwargs1 = n1
      n2, kwargs2 = n2
      kwargs_merge = {
          **kwargs_other,
          **dict((key, (kwargs1[key], kwargs2[key])) for key in kwargs1)
      }
      n1_slice = slice(n1, n1 + n1_batch_size)
      n2_slice = slice(n2, n2 + n2_batch_size)
      in_kernel = slice_kernel(k, n1_slice, n2_slice)
      return (n1, kwargs1), kernel_fn(in_kernel, *args, **kwargs_merge)

    cov2_is_none = utils.nt_tree_fn(reduce=lambda k: all(k))(lambda k:
                                                             k.cov2 is None)(k)
    _, k = _scan(row_fn, 0, (n1s, kwargs_np1))
    if cov2_is_none:
      k = _set_cov2_is_none(k)
    return flatten(k, cov2_is_none)

  @utils.wraps(kernel_fn)
  def serial_fn(x1_or_kernel: Union[NTTree[np.ndarray], NTTree[Kernel]],
                x2: NTTree[Optional[np.ndarray]] = None,
                *args,
                **kwargs) -> NTTree[Kernel]:
    if utils.is_nt_tree_of(x1_or_kernel, np.ndarray):
      return serial_fn_x1(x1_or_kernel, x2, *args, **kwargs)
    elif utils.is_nt_tree_of(x1_or_kernel, Kernel):
      if x2 is not None:
        raise ValueError(f'`x2` must be `None`, got {x2}.')
      return serial_fn_kernel(x1_or_kernel, *args, **kwargs)
    else:
      raise TypeError(x1_or_kernel, type(x1_or_kernel))

  return serial_fn


def _parallel(kernel_fn: KernelFn,
              use_serial: bool = True,
              dropout_in_analytic_kernel: bool = False,
              device_count: int = -1,
              ) -> KernelFn:
  """Returns a function that computes a kernel in batches in parallel.

  When batching in parallel, the data is split over a set number of devices.
  The number of devices must be less than or equal to the number of physical
  devices. Moreover, the dataset size needs to divide the device count.

  Given two datasets `x1` and `x2`, parallel splits the kernel calculation over
  devices such that each device computes a batch of rows of shape
  `[|x1| / device_count, |x2|]`.

  Args:
    kernel_fn:
      A function that computes a kernel between two datasets,
      `kernel_fn(x1, x2)` or the compositional kernel for an input kernel
      `kernel_fn(kernel_in)`. Here `x1` and `x2` are `np.ndarray`s of floats of
      shape `(n1,) + input_shape` and `(n2,) + input_shape`; `kernel_in` is a
      Kernel object. The kernel function should return a `PyTree`.
    use_serial:
      Whether `serial` will be called after `_parallel`. The only use case is to
      make sure when `dropout` is used in the analytic/empirical kernel, the
      batch size in each device is square.
    dropout_in_analytic_kernel:
      whether `dropout` is used in the analytic kernel. See `use_serial` above
      for the only use case.
    device_count:
      Integer specifying the number of devices over which to split the data. If
      `device_count == 0`, the computation is parallelized over all available
      devices.

  Returns:
    A new function with the same signature as kernel_fn that computes the kernel
    by batching over the dataset in parallel over a specified number of cores.
  """

  if device_count == -1:
    device_count = xla_bridge.device_count()

  def _check_dropout(n1, n2, kwargs):
    dropout_in_empirical_kernel = getattr(kwargs, 'rng', None) is not None
    if n1 == n2 and (dropout_in_empirical_kernel or
                     dropout_in_analytic_kernel) and not use_serial:
      raise NotImplementedError(
          'Batching for empirical / analytic kernels with dropout'
          ' is not implemented for non-square batch size. '
          'Using `serial` (i.e. use a non-zero batch_size in the '
          '`batch` function.) could enforce square batch size in each device.')

  def _get_n_per_device(n1, n2):
    _device_count = device_count

    n1_per_device, ragged = divmod(n1, device_count)
    if n1_per_device and ragged:
      raise ValueError(
          ('Dataset size ({}) must divide number of '
           'physical devices ({}).').format(n1, device_count))
    elif not n1_per_device:
      _device_count = ragged
      n1_per_device = 1

    return n1_per_device, _device_count

  def parallel_fn_x1(x1, x2=None, *args, **kwargs):
    x2_is_none = utils.all_none(x2)
    if x2_is_none:
      # TODO(schsam): Only compute the upper triangular part of the kernel.
      x2 = x1

    def get_batch_size(x):
      if utils.is_list_or_tuple(x):
        return get_batch_size(x[0])
      return x.shape[0]

    n1 = get_batch_size(x1)
    n2 = n1 if x2_is_none else get_batch_size(x2)

    _check_dropout(n1, n2, kwargs)
    n1_per_device, _device_count = _get_n_per_device(n1, n2)

    _kernel_fn = _jit_or_pmap_broadcast(kernel_fn, _device_count)

    @utils.nt_tree_fn()
    def batch_data(x):
      input_shape = x.shape[1:]
      return np.reshape(x, (_device_count, n1_per_device,) + input_shape)

    for k, v in kwargs.items():
      if _is_np_ndarray(v):
        assert isinstance(v, tuple) and len(v) == 2
        v0 = np.reshape(v[0], (_device_count, n1_per_device,) + v[0].shape[1:])
        kwargs[k] = (v0, v[1])

    x1 = batch_data(x1)

    kernel = _kernel_fn(x1, x2, *args, **kwargs)
    return _flatten_kernel(kernel, x2_is_none, True)

  def parallel_fn_kernel(kernel, *args, **kwargs):
    @utils.nt_tree_fn(reduce=lambda shapes: shapes[0])
    def get_batch_sizes(k):
      n1 = n2 = k.cov1.shape[0]
      if k.cov2 is not None:
        n2 = k.cov2.shape[0]
      return n1, n2

    n1, n2 = get_batch_sizes(kernel)
    _check_dropout(n1, n2, kwargs)
    n1_per_device, _device_count = _get_n_per_device(n1, n2)

    _kernel_fn = _jit_or_pmap_broadcast(kernel_fn, _device_count)

    cov2_is_none = utils.nt_tree_fn(reduce=lambda k:
                                    all(k))(lambda k: k.cov2 is None)(kernel)
    kernel = _reshape_kernel_for_pmap(kernel, _device_count, n1_per_device)
    kernel = _kernel_fn(kernel, *args, **kwargs)
    if cov2_is_none:
      kernel = _set_cov2_is_none(kernel)
    return _flatten_kernel(kernel, cov2_is_none, True)

  @utils.wraps(kernel_fn)
  def parallel_fn(x1_or_kernel, x2=None, *args, **kwargs):
    if utils.is_nt_tree_of(x1_or_kernel, np.ndarray):
      return parallel_fn_x1(x1_or_kernel, x2, *args, **kwargs)
    elif utils.is_nt_tree_of(x1_or_kernel, Kernel):
      assert not x2
      return parallel_fn_kernel(x1_or_kernel, *args, **kwargs)
    raise NotImplementedError()

  # Set function attributes so that `serial` can detect whether or not it is
  # acting on a parallel function.
  parallel_fn.device_count = device_count
  return parallel_fn


def _get_n_batches_and_batch_sizes(n1: int,
                                   n2: int,
                                   batch_size: int,
                                   device_count: int
                                   ) -> Tuple[int, int, int, int]:
  # TODO(romann): if dropout batching works for different batch sizes, relax.
  max_serial_batch_size = onp.gcd(n1, n2) // device_count

  n2_batch_size = min(batch_size, max_serial_batch_size)
  if n2_batch_size != batch_size:
    warnings.warn(
        'Batch size is reduced from requested %d to effective %d to '
        'fit the dataset.' % (batch_size, n2_batch_size))

  n1_batch_size = n2_batch_size * device_count
  n1_batches, ragged = divmod(n1, n1_batch_size)
  if ragged:
    # TODO(schsam): Relax this constraint.
    msg = ('Number of rows of kernel must divide batch size. Found n1 = {} '
           'and batch size = {}.').format(n1, n1_batch_size)
    if device_count > 1:
      msg += (' Note that device parallelism was detected and so the batch '
              'size was expanded by a factor of {}.'.format(device_count))
    raise ValueError(msg)

  n2_batches, ragged = divmod(n2, n2_batch_size)
  if ragged:
    # TODO(schsam): Relax this constraint.
    raise ValueError(('Number of columns of kernel must divide batch '
                      'size. Found n2 = {} '
                      'and batch size = {}').format(n2, n2_batch_size))
  return n1_batches, n1_batch_size, n2_batches, n2_batch_size


def _is_np_ndarray(x) -> bool:
  if x is None:
    return False
  return tree_all(tree_map(lambda y: isinstance(y, np.ndarray), x))


def _get_jit_or_pmap_broadcast() -> Callable[[Callable, int], Callable]:
  """Initializes a cache of pmapped functions closed over non-`np.ndarray` args.

  Returns:
    A `jit_or_pmap_broadcast` function allowing to jit or pmap a function as a
    closure over all non-`np.ndarray` args, all `kwargs`, while broadcasting
    all `np.ndarray`s in `args` except the first one.
  """
  cache = {}

  def jit_or_pmap_broadcast(f: Callable, device_count: int = -1) -> Callable:
    """Pmap `f` over the first argument by closing over or broadcasting others.

    Args:
      f:
        function to pmap. First argument must be an `np.ndarray` or a Kernel.
        In either case, ndarrays should have a leading axis having the size of
        `device_count`.
      device_count:
        number of XLA devices. `-1` means all available devices. `0` means to
        just `jit` the function.

    Returns:
      A function of the same signature as `f` pmapped over the `np.ndarray`s in
      the first argument. Other arguments are either closed over
      (non-`np.ndarray`s in `args` and all `kwargs`) or broadcasted to
      `(device_count,) + old_shape` (for `np.ndarray`s). If `device_count == 0`,
      `f` is closed over and jitted over all non-array arguments and all
      `kwargs`.

    Raises:
      An error if `kwargs` have a `np.ndarray`.
      TODO(romann): treat `np.ndarray`s in `kwargs` when JAX allows it. See
      https://github.com/google/jax/issues/912
    """
    key = (f, device_count)

    if device_count == -1:
      device_count = xla_bridge.device_count()

    # TODO(romann): adapt this when JAX allows `axis_in` for `pmap`.
    def broadcast(arg: np.ndarray) -> np.ndarray:
      if device_count == 0:
        return arg
      # If the argument has already been sharded, no need to broadcast it.
      if isinstance(arg, ShardedDeviceArray) and arg.shape[0] == device_count:
        return arg
      return np.broadcast_to(arg, (device_count,) + arg.shape)

    @utils.wraps(f)
    def f_pmapped(x_or_kernel: Union[np.ndarray, Kernel], *args, **kwargs):
      args_np, args_np_idxs = [], []
      args_other = {}

      # TODO(romann): treat `np.ndarray`s in `kwargs` when JAX allows it.
      # https://github.com/google/jax/issues/912
      # Filter out `np.ndarray`s from other arguments.
      for i, arg in enumerate(args):
        if _is_np_ndarray(arg):
          args_np.append(arg)
          args_np_idxs.append(i)
        else:
          args_other[i] = arg
      kwargs_np = {}
      kwargs_other = {}
      for k, v in kwargs.items():
        if _is_np_ndarray(v):
          assert isinstance(v, tuple), len(v) == 2
          kwargs_np[k] = (v[0], broadcast(v[1]))
        else:
          kwargs_other[k] = v

      # Check cache before jitting.
      _key = key + \
          tuple(args_other.items()) + \
          tuple(kwargs_other.items())
      if _key in cache:
        _f = cache[_key]
      else:
        # Define a `np.ndarray`-only function as a closure over other arguments.
        def _f(_x_or_kernel, *_args_np, **_kwargs_np):
          # Merge args.
          _args_np = {i: _arg_np for i, _arg_np in zip(args_np_idxs, _args_np)}
          _args = {**_args_np, **args_other}
          _args = tuple(v for k, v in sorted(_args.items()))
          _kwargs = {**_kwargs_np, **kwargs_other}
          return f(_x_or_kernel, *_args, **_kwargs)

        _f = jit(_f) if device_count == 0 else pmap(_f)
        cache[_key] = _f

      # Broadcast `np.ndarray` arguments and apply the new function to them.
      args_np = tree_map(broadcast, args_np)
      return _f(x_or_kernel, *args_np, **kwargs_np)

    return f_pmapped

  return jit_or_pmap_broadcast


_jit_or_pmap_broadcast = _get_jit_or_pmap_broadcast()
