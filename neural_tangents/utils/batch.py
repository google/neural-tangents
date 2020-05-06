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

"""Batch kernel calculations serially or in parallel."""

from functools import partial
import warnings
from jax.api import device_get
from jax.api import jit
from jax.api import pmap
from jax.interpreters.pxla import ShardedDeviceArray
from jax.lib import xla_bridge
import jax.numpy as np
from jax.tree_util import tree_all
from jax.tree_util import tree_map
from jax.tree_util import tree_multimap
from neural_tangents.utils.kernel import Kernel
from neural_tangents.utils import utils
import numpy as onp


def _scan(f, init, xs, store_on_device):
  """Implements an unrolled version of scan.

  Based on `jax.lax.scan` and has an identical API.

  TODO(schsam): We introduce this function because lax.scan currently has a
  higher peak memory usage than the unrolled version. We will aim to swap this
  out for lax.scan when issue #1273 and related have been resolved.
  """

  stack = np.stack if store_on_device else jit(np.stack, backend='cpu')

  carry = init
  ys = []
  for x in xs:
    carry, y = f(carry, x)
    ys += [y]

  return carry, tree_multimap(lambda *y: stack(y), *ys)


def _flatten_batch_dimensions(k, discard_axis=None):
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


def _flatten_kernel_dict(k_dict, x2_is_none, store_on_device, is_parallel):
  fl = (_flatten_batch_dimensions if store_on_device else
      jit(_flatten_batch_dimensions, static_argnums=(1,), backend='cpu'))

  if 'nngp' in k_dict:
    # We only use `batch_size` to compute `shape1` and `shape2` for the batch.
    # This only happens if k_dict came from a `Kernel` in which case it must
    # have 'nngp'. I do think there is a failure case if the user called
    # >>> batched_kernel_fn(x1, x2, get=('ntk', 'shape1'))
    # but I don't think this will get hit ever (and certainly before we rework
    # this code).
    batch_size = {'1': k_dict['nngp'].shape[0], '2': k_dict['nngp'].shape[1]}

  if 'diagonal_batch' in k_dict and not k_dict['diagonal_batch']:
    raise NotImplementedError('Batching not implemented for '
                              '`diagonal_batch == False`.')

  for key, value in k_dict.items():
    if key == 'cov1':
      k_dict[key] = fl(value, 1)

    elif key == 'cov2':
      if x2_is_none:
        k_dict[key] = None
      else:
        k_dict[key] = fl(value, 0)

    elif key == 'x1_is_x2':
      k_dict[key] = value[(0,) * value.ndim]

    elif key == 'mask1':
      if value is None:
        k_dict[key] = None
      else:
        k_dict[key] = fl(value, 1)

    elif key == 'mask2':
      if value is None or x2_is_none:
        k_dict[key] = None
      else:
        k_dict[key] = fl(value, 0)

    elif key in ('shape1', 'shape2'):
      if key == 'shape2' and is_parallel:
        continue
      batch_axis = k_dict['batch_axis']
      shape = value
      k_dict[key] = (shape[:batch_axis] +
                     (shape[batch_axis] * batch_size[key[-1]],) +
                     shape[batch_axis + 1:])
    elif isinstance(k_dict[key], np.ndarray):
      k_dict[key] = fl(value, None)
    else:
      pass
  return k_dict


def _flatten_kernel(k, x2_is_none, store_on_device, is_parallel):
  """Flattens a kernel array or a `Kernel` along the batch dimension."""
  if hasattr(k, '_asdict'):
    return k._replace(**_flatten_kernel_dict(
        k._asdict(), x2_is_none, store_on_device, is_parallel))

  if isinstance(k, Kernel):
    return Kernel(**_flatten_kernel_dict(
        k.asdict(), x2_is_none, store_on_device, is_parallel))

  if isinstance(k, np.ndarray):
    return _flatten_batch_dimensions(k)

  raise TypeError(
      ('Expected kernel to be either a namedtuple, Kernel, or `np.ndarray`, '
       'got %s.') % type(k))


def _reshape_kernel_for_pmap(k, device_count, n1_per_device):
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


def _serial(kernel_fn, batch_size, store_on_device=True):
  """Returns a function that computes a kernel in batches serially.

  This function computes the kernel over data in batches where each batch is
  processed sequentially with a given batch size. If serial detects that the
  kernel function is the result of `_parallel` (that is, if the kernel is
  distributed over multiple devices) then serial adjusts the batch size so that
  each device processes chunks of work that have batch_size x batch_size.

  The dataset size must divide the effective batch size. If parallelism is used
  this means that |x1| must divide batch_size * device_count and |x2| must
  divide batch_size.

  Args:
    kernel_fn: A function that computes a kernel between two datasets,
        `kernel_fn(x1, x2)` or the compositional kernel for an input kernel
        `kernel_fn(kernel_in)`. Here x1 and x2 are `np.ndarray`s of floats of
        shape [n1] + input_shape and [n2] + input_shape; `kernel_in` is a Kernel
        object. The kernel function should return a PyTree.
    batch_size: Integer specifying the size of batches in which to split the
        data.
    store_on_device: A boolean that species whether the computed kernel should
        be kept on device or brought back to CPU as it is computed. Defaults to
        True.

  Returns:
    A new function with the same signature as kernel_fn that computes the kernel
    by batching over the dataset serially with the specified batch_size.
  """

  device_count = max(getattr(kernel_fn, 'device_count', 1), 1)

  if not store_on_device:
    _kernel_fn = kernel_fn
    @utils.wraps(_kernel_fn)
    def kernel_fn(x1, x2=None, *args, **kwargs):
      return device_get(_kernel_fn(x1, x2, *args, **kwargs))

  flatten = partial(_flatten_kernel,
                    store_on_device=store_on_device,
                    is_parallel=False)

  def serial_fn_x1(x1, x2=None, *args, **kwargs):
    # TODO(xlc): Make batch + dropout work reasonably well.
    if 'key' in kwargs:
      raise NotImplementedError('Batching for the empirical kernel with dropout'
                                ' is not implemented. ')
    x2_is_none = x2 is None
    if x2_is_none:
      # TODO(schsam): Only compute the upper triangular part of the kernel.
      x2 = x1

    n1, n2 = x1.shape[0], x2.shape[0]
    (n1_batches, n1_batch_size,
     n2_batches, n2_batch_size) = _get_n_batches_and_batch_sizes(n1, n2,
                                                                 batch_size,
                                                                 device_count)

    input_shape = x1.shape[1:]
    x1s = np.reshape(x1, (n1_batches, n1_batch_size,) + input_shape)
    x2s = np.reshape(x2, (n2_batches, n2_batch_size,) + input_shape)

    def row_fn(_, x1):
      return _, _scan(col_fn, x1, x2s, store_on_device)[1]

    def col_fn(x1, x2):
      return x1, kernel_fn(x1, x2, *args, **kwargs)

    _, kernel = _scan(row_fn, 0, x1s, store_on_device)
    return flatten(kernel, x2_is_none)

  def serial_fn_kernel(kernel, *args, **kwargs):
    n1, n2 = kernel.nngp.shape[:2]
    (n1_batches, n1_batch_size,
     n2_batches, n2_batch_size) = _get_n_batches_and_batch_sizes(n1, n2,
                                                                 batch_size,
                                                                 device_count)

    n1s = np.arange(0, n1, n1_batch_size)
    n2s = np.arange(0, n2, n2_batch_size)

    def row_fn(_, n1):
      return _, _scan(col_fn, n1, n2s, store_on_device)[1]

    def col_fn(n1, n2):
      # NOTE(schsam): If we end up wanting to enable jit-of-batch then we will
      # probably have to change this to dynamic slicing.
      n1_slice = slice(n1, n1 + n1_batch_size)
      n2_slice = slice(n2, n2 + n2_batch_size)
      in_kernel = kernel.slice(n1_slice, n2_slice)
      return n1, kernel_fn(in_kernel, *args, **kwargs)

    cov2_is_none = kernel.cov2 is None
    _, kernel = _scan(row_fn, 0, n1s, store_on_device)
    if cov2_is_none:
      kernel = kernel.replace(cov2=None)
    return flatten(kernel, cov2_is_none)

  @utils.wraps(kernel_fn)
  def serial_fn(x1_or_kernel, x2=None, *args, **kwargs):
    if isinstance(x1_or_kernel, np.ndarray):
      return serial_fn_x1(x1_or_kernel, x2, *args, **kwargs)
    elif isinstance(x1_or_kernel, Kernel):
      assert not x2
      return serial_fn_kernel(x1_or_kernel, *args, **kwargs)
    else:
      raise NotImplementedError()

  return serial_fn


def _parallel(kernel_fn, device_count=-1):
  """Returns a function that computes a kernel in batches in parallel.

  When batching in parallel, the data is split over a set number of devices.
  The number of devices must be less than or equal to the number of physical
  devices. Moreover, the dataset size needs to divide the device count.

  Given two datasets x1 and x2, parallel splits the kernel calculation over
  devices such that each device computes a batch of rows of shape
  [|x1| / device_count, |x2|].

  Args:
    kernel_fn: A function that computes a kernel between two datasets,
        `kernel_fn(x1, x2)` or the compositional kernel for an input kernel
        `kernel_fn(kernel_in)`. Here x1 and x2 are `np.ndarray`s of floats of
        shape [n1] + input_shape and [n2] + input_shape; `kernel_in` is a Kernel
        object. The kernel function should return a PyTree.
    device_count: Integer specifying the number of devices over which to split
        the data. If device_count = 0, the computation is parallelized over all
        available devices.

  Returns:
    A new function with the same signature as kernel_fn that computes the kernel
    by batching over the dataset in parallel over a specified number of cores.
  """
  kernel_fn = _jit_or_pmap_broadcast(kernel_fn, device_count)
  if device_count == -1:
    device_count = xla_bridge.device_count()

  def parallel_fn_x1(x1, x2=None, *args, **kwargs):
    if 'key' in kwargs:
      raise NotImplementedError('Batching for the empirical kernel with dropout'
                                ' is not implemented. ')
    x2_is_none = x2 is None
    if x2_is_none:
      # TODO(schsam): Only compute the upper triangular part of the kernel.
      x2 = x1

    n1 = x1.shape[0]

    assert x1.shape[1:] == x2.shape[1:]
    input_shape = x1.shape[1:]

    _device_count = device_count

    n1_per_device, ragged = divmod(n1, device_count)
    if n1_per_device and ragged:
      raise ValueError(
          ('Dataset size ({}) must divide number of '
           'physical devices ({}).').format(n1, device_count))
    elif not n1_per_device:
      _device_count = ragged
      n1_per_device = 1

    x1 = np.reshape(x1, (_device_count, n1_per_device,) + input_shape)
    kernel = kernel_fn(x1, x2, *args, **kwargs)
    return _flatten_kernel(kernel, x2_is_none, True, True)

  def parallel_fn_kernel(kernel, *args, **kwargs):
    n1 = kernel.cov1.shape[0]

    _device_count = device_count

    n1_per_device, ragged = divmod(n1, device_count)
    if n1_per_device and ragged:
      raise ValueError(
          ('Dataset size ({}) must divide number of '
           'physical devices ({}).').format(n1, device_count))
    elif not n1_per_device:
      _device_count = ragged
      n1_per_device = 1

    cov2_is_none = kernel.cov2 is None
    kernel = _reshape_kernel_for_pmap(kernel, _device_count, n1_per_device)
    kernel = kernel_fn(kernel, *args, **kwargs)
    if cov2_is_none:
      kernel = kernel.replace(cov2=None)
    return _flatten_kernel(kernel, cov2_is_none, True, True)

  @utils.wraps(kernel_fn)
  def parallel_fn(x1_or_kernel, x2=None, *args, **kwargs):
    if isinstance(x1_or_kernel, np.ndarray):
      return parallel_fn_x1(x1_or_kernel, x2, *args, **kwargs)
    elif isinstance(x1_or_kernel, Kernel):
      assert not x2
      return parallel_fn_kernel(x1_or_kernel, *args, **kwargs)
    raise NotImplementedError()

  # Set function attributes so that `serial` can detect whether or not it is
  # acting on a parallel function.
  parallel_fn.device_count = device_count
  return parallel_fn


def batch(kernel_fn, batch_size=0, device_count=-1, store_on_device=True):
  """Returns a function that computes a kernel in batches over all devices.

  Args:
    :kernel_fn: A function that computes a kernel between two datasets,
      `kernel_fn(x1, x2)`. Here `x1` and `x2` are `np.ndarray`s of floats of
      shape `[n1,] + input_shape` and `[n2,] + input_shape`. The kernel
      function should return a PyTree.
    :batch_size: Integer specifying the size of each batch that gets processed
      per physical device. Because we parallelize the computation over columns
      it should be the case that `|x1|` is divisible by
      device_count * batch_size and `|x2|` is divisible by batch_size.
    :device_count: Integer specifying the number of physical devices to be
      mapped over. If device_count = -1 all devices are used. If
      device_count = 0, no device parallelism is used.
    :store_on_device: A boolean that species whether the computed kernel should
      be kept on device or brought back to CPU as it is computed. Defaults to
      True.

  Returns:
    A new function with the same signature as kernel_fn that computes the kernel
    by batching over the dataset in parallel with the specified batch_size.
  """
  if (device_count == -1 and xla_bridge.device_count() > 1) or device_count > 0:
    kernel_fn = _parallel(kernel_fn, device_count)
  else:
    kernel_fn = _jit_or_pmap_broadcast(kernel_fn, device_count=0)

  if not batch_size:
    return kernel_fn

  return _serial(kernel_fn, batch_size, store_on_device)


def _get_n_batches_and_batch_sizes(n1, n2, batch_size, device_count):
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


def _is_np_ndarray(x):
  return tree_all(tree_map(lambda y: isinstance(y, np.ndarray), x))


def _get_jit_or_pmap_broadcast():
  """Initializes a cache of pmapped functions closed over non-`np.ndarray` args.

  Returns:
    A `jit_or_pmap_broadcast` function allowing to jit or pmap a function as a
      closure over all non-`np.ndarray` args, all `kwargs`, while broadcasting
      all `np.ndarray`s in `args` except the first one.
  """
  cache = {}

  def jit_or_pmap_broadcast(f, device_count=-1):
    """Pmap `f` over the first argument by closing over or broadcasting others.

    Args:
      f: function to pmap. First argument must be an `np.ndarray` or a Kernel.
        In either case, ndarrays should have a leading axis having the size of
        `device_count`.
      device_count: number of XLA devices. `-1` means all available devices. `0`
        means to just `jit` the function.

    Returns:
      A function of the same signature as `f` pmapped over the ndarrays in the
      first argument. Other arguments are either closed over (non-`np.ndarray`s
      in `args` and all `kwargs`) or broadcasted to
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
    def broadcast(arg):
      if device_count == 0:
        return arg
      # If the argument has already been sharded, no need to broadcast it.
      if isinstance(arg, ShardedDeviceArray) and arg.shape[0] == device_count:
        return arg
      return np.broadcast_to(arg, (device_count,) + arg.shape)

    @utils.wraps(f)
    def f_pmapped(x_or_kernel, *args, **kwargs):
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

      # Check cache before jitting.
      _key = key + \
          tuple(args_other.items()) + \
          tuple(kwargs.items())
      if _key in cache:
        _f = cache[_key]
      else:
        # Define a `np.ndarray`-only function as a closure over other arguments.
        def _f(_x_or_kernel, *_args_np):
          # Merge args.
          _args_np = {i: _arg_np for i, _arg_np in zip(args_np_idxs, _args_np)}
          _args = {**_args_np, **args_other}
          _args = tuple(v for k, v in sorted(_args.items()))
          return f(_x_or_kernel, *_args, **kwargs)

        _f = jit(_f) if device_count == 0 else pmap(_f)
        cache[_key] = _f

      # Broadcast `np.ndarray` arguments and apply the new function to them.
      args_np = tree_map(broadcast, args_np)
      return _f(x_or_kernel, *args_np)

    return f_pmapped

  return jit_or_pmap_broadcast


_jit_or_pmap_broadcast = _get_jit_or_pmap_broadcast()
