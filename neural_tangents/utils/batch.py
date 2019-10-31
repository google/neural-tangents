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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from functools import partial

from jax.api import device_get
from jax.api import jit
from jax.api import pmap
from jax.lib import xla_bridge
import jax.numpy as np
from jax.tree_util import tree_all
from jax.tree_util import tree_map
from jax.tree_util import tree_multimap
from .kernel import Kernel


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


def _flatten_kernel(k, x2_is_none, store_on_device):
  """Flattens a kernel array or a `Kernel` along the batch dimension."""
  fl = (_flatten_batch_dimensions if store_on_device else
        jit(_flatten_batch_dimensions, static_argnums=(1,), backend='cpu'))
  if hasattr(k, '_asdict'):
    k_dict = k._asdict()
    k = k._replace(**dict((key, 0.) for key in k_dict))
    for key, value in k_dict.items():
      if key == 'var1':
        k_dict[key] = fl(value, 1)
      elif key == 'var2':
        if x2_is_none:
          k_dict[key] = None
        else:
          k_dict[key] = fl(value, 0)
      elif key in ('is_height_width', 'is_gaussian'):
        # NOTE(schsam): Currently we have to make these values concrete so that
        # batched analytic kernels compose.
        k_dict[key] = bool(value[(0,) * value.ndim])
      elif key in ('marginal', 'cross'):
        k_dict[key] = int(value[(0,) * value.ndim])
      elif key == 'shape1':
        if any([x.ndim > 2 for x in value]):
          raise ValueError((
              'After batching, shape arrays expected to be either'
              ' one- or two-dimensional.'))
        k_dict[key] = tuple(int(x[(0,) * x.ndim]) if i > 0 else
                            int(np.sum(x[:, 0])) if x.ndim == 2 else
                            int(np.sum(x)) for i, x in enumerate(value))
      elif key == 'shape2':
        if any([x.ndim > 2 for x in value]):
          raise ValueError((
              'After batching, shape arrays expected to be either'
              ' one- or two-dimensional.'))
        k_dict[key] = tuple(int(x[(0,) * x.ndim]) if i > 0 else
                            int(np.sum(x[0])) if x.ndim == 2 else
                            int(x[0]) for i, x in enumerate(value))
      else:
        k_dict[key] = fl(value, None)
    return k._replace(**k_dict)

  if isinstance(k, np.ndarray):
    return _flatten_batch_dimensions(k)

  raise TypeError(
      'Expected kernel to be either a namedtuple or a `np.ndarray`, got %s.'
      % type(k))


def _move_kernel_to_cpu(k):
  """Moves data in a kernel from an accelerator to the CPU."""
  if hasattr(k, '_asdict') and hasattr(k, '_replace'):
    return k._replace(
        **dict([(k, v) if not isinstance(v, np.ndarray) else
                (k, device_get(v)) for k, v in k._asdict().items()]))
  elif isinstance(k, np.ndarray):
    return device_get(k)
  else:
    raise TypeError(
        'Expected kernel to be either a namedtuple or a `np.ndarray`, got %s.'
        % type(k)
    )


def _slice_kernel(kernel, n1_slice, n2_slice):
  assert isinstance(kernel, Kernel)
  var1 = kernel.var1[n1_slice]
  var2 = kernel.var1[n2_slice] if kernel.var2 is None else kernel.var2[n2_slice]
  return kernel._replace(
      var1=var1,
      nngp=kernel.nngp[n1_slice, n2_slice],
      var2=var2,
      ntk=kernel.ntk[n1_slice, n2_slice],
      shape1=(var1.shape[0],) + kernel.shape1[1:],
      shape2=(var2.shape[0],) + kernel.shape2[1:])


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

  is_parallel = hasattr(kernel_fn, 'is_parallel')
  if is_parallel:
    device_count = kernel_fn.device_count

  if not store_on_device:
    _kernel_fn = kernel_fn
    def kernel_fn(x1, x2=None, *args, **kwargs):
      return _move_kernel_to_cpu(_kernel_fn(x1, x2, *args, **kwargs))

  flatten = partial(_flatten_kernel, store_on_device=store_on_device)

  def serial_fn_x1(x1, x2=None, *args, **kwargs):
    x2_is_none = x2 is None
    if x2_is_none:
      # TODO(schsam): Only compute the upper triangular part of the kernel.
      x2 = x1

    n1 = x1.shape[0]
    n2 = x2.shape[0]
    input_shape = x1.shape[1:]

    n1_batch_size = batch_size if not is_parallel else batch_size * device_count
    n1_batches, ragged = divmod(n1, n1_batch_size)
    if ragged:
      # TODO(schsam): Relax this constraint.
      msg = ('Number of examples in x1 must divide batch size. Found |x1| = {} '
             'and batch size = {}.').format(n1, n1_batch_size)
      if is_parallel:
        msg += (' Note that device parallelism was detected and so the batch '
                'size was expanded by a factor of {}.'.format(device_count))
      raise ValueError(msg)

    n2_batches, ragged = divmod(n2, batch_size)
    if ragged:
      # TODO(schsam): Relax this constraint.
      raise ValueError((
          'Number of examples in x2 must divide batch size. Found |x2| = {} '
          'and batch size = {}').format(n2, batch_size))

    x1s = np.reshape(x1, (n1_batches, n1_batch_size,) + input_shape)
    x2s = np.reshape(x2, (n2_batches, batch_size,) + input_shape)

    def row_fn(_, x1):
      return _, _scan(col_fn, x1, x2s, store_on_device)[1]

    def col_fn(x1, x2):
      return x1, kernel_fn(x1, x2, *args, **kwargs)

    _, kernel = _scan(row_fn, 0, x1s, store_on_device)
    return flatten(kernel, x2_is_none)

  def serial_fn_kernel(kernel, *args, **kwargs):
    n1 = kernel.var1.shape[0]
    n2 = n1 if kernel.var2 is None else kernel.var2.shape[0]

    n1_batch_size = batch_size if not is_parallel else batch_size * device_count
    n1_batches, ragged = divmod(n1, n1_batch_size)
    if ragged:
      # TODO(schsam): Relax this constraint.
      msg = ('Number of rows of kernel must divide batch size. Found n1 = {} '
             'and batch size = {}.').format(n1, n1_batch_size)
      if is_parallel:
        msg += (' Note that device parallelism was detected and so the batch '
                'size was expanded by a factor of {}.'.format(device_count))
      raise ValueError(msg)

    n2_batches, ragged = divmod(n2, batch_size)
    if ragged:
      # TODO(schsam): Relax this constraint.
      raise ValueError((
          'Number of columns of kernel must divide batch size. Found n2 = {} '
          'and batch size = {}').format(n2, batch_size))

    n1s = np.arange(0, n1, n1_batch_size)
    n2s = np.arange(0, n2, batch_size)

    def row_fn(_, n1):
      return _, _scan(col_fn, n1, n2s, store_on_device)[1]

    def col_fn(n1, n2):
      # NOTE(schsam): If we end up wanting to enable jit-of-batch then we will
      # probably have to change this to dynamic slicing.
      n1_slice = slice(n1, n1 + n1_batch_size)
      n2_slice = slice(n2, n2 + batch_size)
      in_kernel = _slice_kernel(kernel, n1_slice, n2_slice)
      return n1, kernel_fn(in_kernel, *args, **kwargs)

    var2_is_none = kernel.var2 is None
    _, kernel = _scan(row_fn, 0, n1s, store_on_device)
    if var2_is_none:
      kernel = kernel._replace(var2=None)
    return flatten(kernel, var2_is_none)

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
    return _flatten_kernel(kernel, x2_is_none, True)

  def parallel_fn_kernel(kernel, *args, **kwargs):
    n1 = kernel.var1.shape[0]

    _device_count = device_count

    n1_per_device, ragged = divmod(n1, device_count)
    if n1_per_device and ragged:
      raise ValueError(
          ('Dataset size ({}) must divide number of '
           'physical devices ({}).').format(n1, device_count))
    elif not n1_per_device:
      _device_count = ragged
      n1_per_device = 1

    kernel_dict = kernel._asdict()

    var2 = kernel_dict['var2']
    var2_is_none = var2 is None
    if var2 is None:
      var2 = kernel_dict['var1']
    kernel_dict['var2'] = np.broadcast_to(var2, (_device_count,) + var2.shape)

    for k, v in kernel_dict.items():
      if k in ('nngp', 'ntk', 'var1'):
        kernel_dict[k] = \
            np.reshape(v, (_device_count, n1_per_device,) + v.shape[1:])
      if k in ('shape1',):
        kernel_dict[k] = (n1_per_device,) + v[1:]
    kernel = kernel_fn(Kernel(**kernel_dict), *args, **kwargs)
    if var2_is_none:
      kernel = kernel._replace(var2=None)
    return _flatten_kernel(kernel, var2_is_none, True)

  def parallel_fn(x1_or_kernel, x2=None, *args, **kwargs):
    if isinstance(x1_or_kernel, np.ndarray):
      return parallel_fn_x1(x1_or_kernel, x2, *args, **kwargs)
    elif isinstance(x1_or_kernel, Kernel):
      assert not x2
      return parallel_fn_kernel(x1_or_kernel, *args, **kwargs)
    raise NotImplementedError()

  # Set function attributes so that `serial` can detect whether or not it is
  # acting on a parallel function.
  parallel_fn.is_parallel = True
  parallel_fn.device_count = device_count

  return parallel_fn


def batch(kernel_fn, batch_size=0, device_count=-1, store_on_device=True):
  """Returns a function that computes a kernel in batches over all devices.

  Args:
    kernel_fn: A function that computes a kernel between two datasets,
        kernel_fn(x1, x2). Here x1 and x2 are `np.ndarray`s of floats of shape
        [n1,] + input_shape and [n2,] + input_shape. The kernel function
        should return a PyTree.
    batch_size: Integer specifying the size of each batch that gets processed
        per physical device. Because we parallelize the computation over columns
        it should be the case that |x1| is divisible by
        device_count * batch_size and |x2| is divisible by batch_size.
    device_count: Integer specifying the number of physical devices to be mapped
        over. If device_count = -1 all devices are used. If device_count = 0,
        no device parallelism is used.
    store_on_device: A boolean that species whether the computed kernel should
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


def _is_np_ndarray(x):
  return tree_all(tree_map(lambda y: isinstance(y, np.ndarray), x))


def _merge_dicts(a, b):
  # TODO(schsam): Replace by {**a, **b} when Python 2 is depricated.
  merged = dict(a)
  merged.update(b)
  return merged


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
      return np.broadcast_to(arg, (device_count,) + arg.shape)

    def f_pmapped(x_or_kernel, *args, **kwargs):
      args_np, args_np_idxs = [], []
      args_other = {}

      is_input_kernel = isinstance(x_or_kernel, Kernel)
      x_or_kernel_np = {}
      x_or_kernel_other = {}

      if is_input_kernel:
        kernel_dict = x_or_kernel._asdict()
        for k, v in kernel_dict.items():
          if isinstance(v, np.ndarray):
            x_or_kernel_np[k] = v
          else:
            x_or_kernel_other[k] = v
      else:
        x_or_kernel_np = x_or_kernel

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
          tuple(kwargs.items()) + \
          tuple(x_or_kernel_other.items())
      if _key in cache:
        _f = cache[_key]
      else:
        # Define a `np.ndarray`-only function as a closure over other arguments.
        def _f(_x_or_kernel_np, *_args_np):
          # Merge Kernel.
          if is_input_kernel:
            _x_or_kernel_np = _merge_dicts(_x_or_kernel_np, x_or_kernel_other)
            _x_or_kernel_np = Kernel(**_x_or_kernel_np)
          # Merge args.
          _args_np = {i: _arg_np for i, _arg_np in zip(args_np_idxs, _args_np)}
          _args = _merge_dicts(_args_np, args_other)
          _args = tuple(v for k, v in sorted(_args.items()))
          return f(_x_or_kernel_np, *_args, **kwargs)

        _f = jit(_f) if device_count == 0 else pmap(_f)
        cache[_key] = _f

      # Broadcast `np.ndarray` arguments and apply the new function to them.
      args_np = tree_map(broadcast, args_np)
      return _f(x_or_kernel_np, *args_np)

    return f_pmapped

  return jit_or_pmap_broadcast


_jit_or_pmap_broadcast = _get_jit_or_pmap_broadcast()
