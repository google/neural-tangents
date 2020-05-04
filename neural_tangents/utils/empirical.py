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

"""Compute the empirical NTK and approximate functions via Taylor series."""

from functools import partial
import operator

from absl import flags
from jax import random
from jax.api import eval_shape
from jax.api import jacobian
from jax.api import jvp
from jax.api import vjp
from jax.config import config
import jax.numpy as np
from jax.tree_util import tree_multimap
from jax.tree_util import tree_reduce
from neural_tangents.utils import flags as internal_flags
from neural_tangents.utils import utils
from neural_tangents.utils.typing import ApplyFn, EmpiricalKernelFn, PyTree
from typing import Callable

config.parse_flags_with_absl()  # NOTE(schsam): Is this safe?


FLAGS = flags.FLAGS


def linearize(f: Callable[..., np.ndarray],
              params: PyTree) -> Callable[..., np.ndarray]:
  """Returns a function `f_lin`, the first order taylor approximation to `f`.

  Example:
    >>> # Compute the MSE of the first order Taylor series of a function.
    >>> f_lin = linearize(f, params)
    >>> mse = np.mean((f(new_params, x) - f_lin(new_params, x)) ** 2)

  Args:
    f: A function that we would like to linearize. It should have the signature
       `f(params, *args, **kwargs)` where params is a PyTree and `f` should
       return an `np.ndarray`.
    params: Initial parameters to the function that we would like to take the
       Taylor series about. This can be any structure that is compatible
       with the JAX tree operations.
  Returns:
    A function `f_lin(new_params, *args, **kwargs)` whose signature is the same
    as f. Here `f_lin` implements the first-order taylor series of `f` about
    `params`.
  """
  def f_lin(p, *args, **kwargs):
    dparams = tree_multimap(lambda x, y: x - y, p, params)
    f_params_x, proj = jvp(lambda param: f(param, *args, **kwargs),
                           (params,), (dparams,))
    return f_params_x + proj
  return f_lin


def taylor_expand(f: Callable[..., np.ndarray],
                  params: PyTree,
                  degree: int) -> Callable[..., np.ndarray]:
  """Returns a function `f_tayl`, Taylor approximation to `f` of order `degree`.

  Example:
    >>> # Compute the MSE of the third order Taylor series of a function.
    >>> f_tayl = taylor_expand(f, params, 3)
    >>> mse = np.mean((f(new_params, x) - f_tayl(new_params, x)) ** 2)

  Args:
    f: A function that we would like to Taylor expand. It should have the
      signature `f(params, *args, **kwargs)` where `params` is a PyTree, and
      `f` returns a `np.ndarray`.
    params: Initial parameters to the function that we would like to take the
      Taylor series about. This can be any structure that is compatible
      with the JAX tree operations.
    degree: The degree of the Taylor expansion.

  Returns:
    A function `f_tayl(new_params, *args, **kwargs)` whose signature is the
    same as `f`. Here `f_tayl` implements the degree-order taylor series of `f`
    about `params`.

  """

  def taylorize_r(f, params, dparams, degree, current_degree):
    """Recursive function to accumulate contributions to the Taylor series."""
    if current_degree == degree:
      return f(params)

    def f_jvp(p):
      _, val_jvp = jvp(f, (p,), (dparams,))
      return val_jvp

    df = taylorize_r(f_jvp, params, dparams, degree, current_degree+1)
    return f(params) + df / (current_degree + 1)

  def f_tayl(p, *args, **kwargs):
    dparams = tree_multimap(lambda x, y: x - y, p, params)
    return taylorize_r(lambda param: f(param, *args, **kwargs),
                       params, dparams, degree, 0)

  return f_tayl


# Empirical Kernel


def flatten_features(kernel: np.ndarray) -> np.ndarray:
  """Flatten an empirical kernel."""
  if kernel.ndim == 2:
    return kernel
  assert kernel.ndim % 2 == 0
  half_shape = (kernel.ndim - 1) // 2
  n1, n2 = kernel.shape[:2]
  feature_size = int(np.prod(kernel.shape[2 + half_shape:]))
  transposition = ((0,) + tuple(i + 2 for i in range(half_shape)) +
                   (1,) + tuple(i + 2 + half_shape for i in range(half_shape)))
  kernel = np.transpose(kernel, transposition)
  return np.reshape(kernel, (feature_size *  n1, feature_size * n2))


def empirical_implicit_ntk_fn(f: ApplyFn) -> EmpiricalKernelFn:
  """Computes the ntk without batching for inputs x1 and x2.

  The Neural Tangent Kernel is defined as :math:`J(X_1) J(X_2)^T` where
  :math:`J` is the jacobian :math:`df / dparams^T`. Computing the NTK directly
  involves instantiating the jacobian which takes
  `O(dataset_size * output_dim * parameters)` memory. It turns out it is
  substantially more efficient (especially as the number of parameters grows)
  to compute the NTK implicitly.

  The implicit kernel is derived by observing that:
    :math:`Theta = J(X_1) J(X_2)^T = d[J(X_1) J(X_2)^T v] / d[v^T]`,
  for a vector :math:`v`. This allows the computation of the NTK to be phrased
  as: :math:`a(v) = J(X_2)^T v`, which is computed by a vector-Jacobian product;
  :math:`b(v) = J(X_1) a(v)` which is computed by a Jacobian-vector product; and
  :math:`Theta = d[b(v)] / d[v^T]` which is computed by taking the Jacobian of
  :math:`b(v)`.

  Args:
    f: The function whose NTK we are computing. `f` should have the signature
      `f(params, inputs)` and should return an `np.ndarray` of outputs with
      shape `[|inputs|, output_dim]`.
  Returns:
    A function ntk_fn that computes the empirical ntk.
  """

  def ntk_fn(x1, x2, params, keys=None, **apply_fn_kwargs):
    """Computes the empirical ntk.

    Args:
      x1: A first `np.ndarray` of inputs, of shape [n1, ...], over which we
        would like to compute the NTK.
      x2: A second `np.ndarray` of inputs, of shape [n2, ...], over which we
        would like to compute the NTK.
      params: A PyTree of parameters about which we would like to compute the
        neural tangent kernel.
      keys: None or a PRNG key or a tuple of PRNG keys or a (2, 2) array and
        dtype uint32. If `key == None`, then the function `f` is deterministic
        and requires no PRNG key; else if `keys` is a single PRNG key, then x1
        and x2 must be the same and share the same PRNG key; else x1 and x2 use
        two different PRNG keys.
      **apply_fn_kwargs: keyword arguments passed to `apply_fn`.

    Returns:
      A `np.ndarray` of shape [n1, n2] + output_shape + output_shape.
    """
    key1, key2 = _read_keys(keys)
    # TODO: find a good way to check utils.x1_is_x2(x1, x2) == (key1==key2)
    if x2 is None:
      x2 = x1

    f1 = _get_f_params(f, x1, key1, **apply_fn_kwargs)
    f2 = _get_f_params(f, x2, key2, **apply_fn_kwargs)

    def delta_vjp_jvp(delta):
      def delta_vjp(delta):
        return vjp(f2, params)[1](delta)
      return jvp(f1, (params,), delta_vjp(delta))[1]

    # Since we are taking the Jacobian of a linear function (which does not
    # depend on its coefficients), it is more efficient to substitute fx_dummy
    # for the outputs of the network. fx_dummy has the same shape as the output
    # of the network on a single piece of input data.
    fx2_struct = eval_shape(f2, params)
    fx_dummy = np.ones(fx2_struct.shape, fx2_struct.dtype)

    ntk = jacobian(delta_vjp_jvp)(fx_dummy)
    ndim = len(fx2_struct.shape)
    ordering = (0, ndim) + tuple(range(1, ndim)) + \
       tuple(x + ndim for x in range(1, ndim))
    return np.transpose(ntk, ordering)

  return ntk_fn


def empirical_direct_ntk_fn(f: ApplyFn) -> EmpiricalKernelFn:
  """Computes the ntk without batching for inputs x1 and x2.

  The Neural Tangent Kernel is defined as :math:`J(X_1) J(X_2)^T` where
  :math:`J` is the jacobian :math:`df/dparams`.

  Args:
    :f: The function whose NTK we are computing. `f` should have the signature
     `f(params, inputs)` and should return an `np.ndarray` of outputs with shape
     `[|inputs|, output_dim]`.

  Returns:
    A function `ntk_fn` that computes the empirical ntk.
  """
  def sum_and_contract(j1, j2):
    def contract(x, y):
      param_count = int(np.prod(x.shape[2:]))
      x = np.reshape(x, x.shape[:2] + (param_count,))
      y = np.reshape(y, y.shape[:2] + (param_count,))
      return np.dot(x, np.transpose(y, (0, 2, 1)))

    return tree_reduce(operator.add, tree_multimap(contract, j1, j2))

  def ntk_fn(x1, x2, params, keys=None, **apply_fn_kwargs):
    """Computes the empirical ntk.

    Args:
      x1: A first `np.ndarray` of inputs, of shape [n1, ...], over which we
        would like to compute the NTK.
      x2: A second `np.ndarray` of inputs, of shape [n2, ...], over which we
        would like to compute the NTK.
      params: A PyTree of parameters about which we would like to compute the
        neural tangent kernel.
      keys: None or a PRNG key or a tuple of PRNG keys or a (2, 2) array and
        dtype uint32. If `key == None`, then the function `f` is deterministic
        and requires no PRNG key; else if `keys` is a single PRNG key, then x1
        and x2 share the same PRNG key; else x1 and x2 use two different PRNG
        keys.
      **apply_fn_kwargs: keyword arguments passed to `apply_fn`.

    Returns:
      A `np.ndarray` of shape [n1, n2] + output_shape + output_shape.
    """
    key1, key2 = _read_keys(keys)

    f1 = _get_f_params(f, x1, key1, **apply_fn_kwargs)
    jac_fn1 = jacobian(f1)
    j1 = jac_fn1(params)
    if x2 is None:
      j2 = j1
    else:
      f2 = _get_f_params(f, x2, key2, **apply_fn_kwargs)
      jac_fn2 = jacobian(f2)
      j2 = jac_fn2(params)

    ntk = sum_and_contract(j1, j2)
    # TODO: If we care, this will not work if the output is not of
    # shape [n, output_dim].
    return np.transpose(ntk, (0, 2, 1, 3))

  return ntk_fn


empirical_ntk_fn = (empirical_implicit_ntk_fn
                    if FLAGS.tangents_optimized else
                    empirical_direct_ntk_fn)


def empirical_nngp_fn(f: ApplyFn) -> EmpiricalKernelFn:
  """Returns a function to draw a single sample the NNGP of a given network `f`.

  This method assumes that slices of the random network outputs along the last
  dimension are i.i.d. (which is true for e.g. classifiers with a dense
  readout layer, or true for outputs of a CNN layer with the `NHWC` data
  format. As a result it treats outputs along that dimension as independent
  samples and only reports covariance along other dimensions.

  Note that the `ntk_monte_carlo` makes no such assumption and returns the full
  covariance.

  Args:
    :f: a function computing the output of the neural network.
      From `jax.experimental.stax`: "takes params, inputs, and an rng key and
      applies the layer".

  Returns:
     A function to draw a single sample the NNGP of a given network `f`.
  """
  def nngp_fn(x1, x2, params, keys=None, **apply_fn_kwargs):
    """Sample a single NNGP of a given network `f` on given inputs and `params`.

    This method assumes that slices of the random network outputs along the last
    dimension are i.i.d. (which is true for e.g. classifiers with a dense
    readout layer, or true for outputs of a CNN layer with the `NHWC` data
    format. As a result it treats outputs along that dimension as independent
    samples and only reports covariance along other dimensions.

    Note that the `ntk` method makes no such assumption and returns the full
    covariance.

    Args:
      x1: a `np.ndarray` of shape `[batch_size_1] + input_shape`.
      x2: an optional `np.ndarray` with shape `[batch_size_2] + input_shape`.
        `None` means `x2 == x1`.
      params: A PyTree of parameters about which we would like to compute the
        NNGP.
      keys: None or a PRNG key or a tuple of PRNG keys or a (2, 2) array and
        dtype uint32. If `key == None`, then the function `f` is deterministic
        and requires no PRNG key; else if `keys` is a single PRNG key, then x1
        and x2 share the same PRNG key; else x1 and x2 use two different PRNG
        keys.
      **apply_fn_kwargs: keyword arguments passed to `apply_fn`.

    Returns:
      A Monte Carlo estimate of the NNGP, a `np.ndarray` of shape
      `[batch_size_1] + output_shape[:-1] + [batch_size_2] + output_shape[:-1]`.
    """
    key1, key2 = _read_keys(keys)

    def output(x, rng):
      out = f(params, x, rng=rng, **apply_fn_kwargs)
      masked_output = utils.get_masked_array(out)
      return masked_output.masked_value

    out1 = output(x1, key1)
    if x2 is None:
      out2 = out1
    else:
      out2 = output(x2, key2)

    out2 = np.expand_dims(out2, -1)
    nngp_12 = np.dot(out1, out2) / out1.shape[-1]
    return np.squeeze(nngp_12, -1)

  return nngp_fn


def empirical_kernel_fn(f: ApplyFn) -> EmpiricalKernelFn:
  """Returns a function that computes single draws from NNGP and NT kernels."""

  kernel_fns = {
      'nngp': empirical_nngp_fn(f),
      'ntk': empirical_ntk_fn(f)
  }

  @utils.get_namedtuple('EmpiricalKernel')
  def kernel_fn(x1, x2, params, get=None, keys=None, **apply_fn_kwargs):
    """Returns a draw from the requested empirical kernels.

    Args:
      x1: An ndarray of shape [n1,] + input_shape.
      x2: An ndarray of shape [n2,] + input_shape.
      params: A PyTree of parameters for the function `f`.
      get: either None, a string, a tuple of strings specifying which data
        should be returned by the kernel function. Can be "nngp" or "ntk". If
        `None` then both "nngp" and "ntk" are returned.
      keys: None or a PRNG key or a tuple of PRNG keys or a (2, 2) array and
        dtype uint32. If `key == None`, then the function `f` is deterministic
        and requires no PRNG key; else if `keys` is a single PRNG key, then x1
        and x2 share the same PRNG key; else x1 and x2 use two different PRNG
        keys.
      **apply_fn_kwargs: keyword arguments passed to `apply_fn`.

    Returns:
      If `get` is a string, returns the requested `np.ndarray`. If `get` is a
      tuple, returns an `EmpiricalKernel` namedtuple containing only the
      requested information.
    """
    if get is None:
      get = ('nngp', 'ntk')
    return {g: kernel_fns[g](x1, x2, params, keys, **apply_fn_kwargs)
            for g in get}

  return kernel_fn


# INTERNAL UTILITIES


def _read_keys(keys):
  if keys is None or (isinstance(keys, np.ndarray) and keys.shape == (2,)):
    key1 = key2 = keys
  elif isinstance(keys, tuple):
    # assuming x1 and x2 using key1 and key2, resp.
    key1, key2 = keys
  elif isinstance(keys, np.ndarray) and keys.shape == (2, 2):
    key1, key2 = keys[0], keys[1]
  else:
    raise ValueError('`keys` must be one of the following: `None`, a PRNG '
                     'key, a tuple of PRNG keys or a (2, 2) array and dtype '
                     'unint32')
  return key1, key2


def _get_f_params(f, x, rng, **apply_fn_kwargs):
  def _f(p):
    out = f(p, x, rng=rng, **apply_fn_kwargs)
    # TODO(romann): normalize properly if output is masked.
    out = utils.get_masked_array(out)
    return out.masked_value
  return _f
