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


import operator
from absl import flags
from jax import random
from jax.api import eval_shape
from jax.api import jacobian
from jax.api import jvp
from jax.api import vjp
from functools import partial
from jax.config import config
import jax.numpy as np
from jax.tree_util import tree_multimap
from jax.tree_util import tree_reduce
from neural_tangents.utils import flags as internal_flags
from neural_tangents.utils import utils

config.parse_flags_with_absl()  # NOTE(schsam): Is this safe?


FLAGS = flags.FLAGS


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


def linearize(f, params):
  """Returns a function f_lin, the first order taylor approximation to f.

  Example:
    >>> # Compute the MSE of the first order Taylor series of a function.
    >>> f_lin = linearize(f, params)
    >>> mse = np.mean((f(new_params, x) - f_lin(new_params, x)) ** 2)

  Args:
    f: A function that we would like to linearize. It should have the signature
       f(params, inputs) where params and inputs are `np.ndarray`s and f should
       return an `np.ndarray`.
    params: Initial parameters to the function that we would like to take the
            Taylor series about. This can be any structure that is compatible
            with the JAX tree operations.

  Returns:
    A function f_lin(new_params, inputs) whose signature is the same as f.
    Here f_lin implements the first-order taylor series of f about params.
  """
  def f_lin(p, x):
    dparams = tree_multimap(lambda x, y: x - y, p, params)
    f_params_x, proj = jvp(lambda param: f(param, x), (params,), (dparams,))
    return f_params_x + proj
  return f_lin


def taylor_expand(f, params, degree):
  """Returns a function f_tayl, the Taylor approximation to f of degree degree.

  Example:
    >>> # Compute the MSE of the third order Taylor series of a function.
    >>> f_tayl = taylor_expand(f, params, 3)
    >>> mse = np.mean((f(new_params, x) - f_tayl(new_params, x)) ** 2)

  Args:
    f: A function that we would like to Taylor expand. It should have the
       signature f(params, inputs) where params is a PyTree, inputs is an
       `np.ndarray`, and f returns an `np.ndarray`.
    params: Initial parameters to the function that we would like to take the
            Taylor series about. This can be any structure that is compatible
            with the JAX tree operations.
    degree: The degree of the Taylor expansion.

  Returns:
    A function f_tayl(new_params, inputs) whose signature is the same as f.
    Here f_tayl implements the degree-order taylor series of f about params.
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

  def f_tayl(p, x):
    dparams = tree_multimap(lambda x, y: x - y, p, params)
    return taylorize_r(lambda param: f(param, x), params, dparams, degree, 0)

  return f_tayl


# Empirical Kernel


def flatten_features(kernel):
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


def empirical_implicit_ntk_fn(f):
  """Computes the ntk without batching for inputs x1 and x2.

  The Neural Tangent Kernel is defined as J(X_1)^T J(X_2) where J is the
  jacobian df/dparams. Computing the NTK directly involves directly
  instantiating the jacobian which takes
  O(dataset_size * output_dim * parameters) memory. It turns out it is
  substantially more efficient (especially as the number of parameters grows)
  to compute the NTK implicitly.

  This involves using JAX's autograd to compute derivatives of linear functions
  (which do not depend on the inputs). Thus, we find it more efficient to refer
  to fx_dummy for the outputs of the network. fx_dummy has the same shape as
  the output of the network on a single piece of input data.

  TODO(schsam): Write up a better description of the implicit method.

  Args:
    f: The function whose NTK we are computing. f should have the signature
       f(params, inputs) and should return an `np.ndarray` of outputs with shape
       [|inputs|, output_dim].

  Returns:
    A function ntk_fn that computes the empirical ntk.
  """

  def ntk_fn(x1, x2, params, keys=None):
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

    Returns:
      A `np.ndarray` of shape [n1, n2] + output_shape + output_shape.
    """
    key1, key2 = _read_keys(keys)
    # TODO(xlc): find a good way to check utils.x1_is_x2(x1, x2) == (key1==key2)
    if x2 is None:
      x2 = x1

    f_dummy = partial(f, rng=random.PRNGKey(1))
    fx2_struct = eval_shape(f_dummy, params, x2)
    fx_dummy = np.ones(fx2_struct.shape, fx2_struct.dtype)
    def delta_vjp_jvp(delta):
      def delta_vjp(delta):
        return vjp(lambda p: f(p, x2, rng=key2), params)[1](delta)
      return jvp(lambda p: f(p, x1, rng=key1), (params,), delta_vjp(delta))[1]

    ntk = jacobian(delta_vjp_jvp)(fx_dummy)
    ndim = len(fx2_struct.shape)
    ordering = (0, ndim) + tuple(range(1, ndim)) + \
        tuple(x + ndim for x in range(1, ndim))
    return np.transpose(ntk, ordering)

  return ntk_fn


def empirical_direct_ntk_fn(f):
  """Computes the ntk without batching for inputs x1 and x2.

  The Neural Tangent Kernel is defined as J(X_1)^T J(X_2) where J is the
  jacobian df/dparams.

  Args:
    f: The function whose NTK we are computing. f should have the signature
       f(params, inputs) and should return an `np.ndarray` of outputs with shape
       [|inputs|, output_dim].

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

  def ntk_fn(x1, x2, params, keys=None):
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

    Returns:
      A `np.ndarray` of shape [n1, n2] + output_shape + output_shape.
    """
    key1, key2 = _read_keys(keys)
    f1 = partial(f, rng=key1)
    jac_fn1 = jacobian(f1)
    j1 = jac_fn1(params, x1)
    if x2 is None:
      j2 = j1
    else:
      f2 = partial(f, rng=key2)
      jac_fn2 = jacobian(f2)
      j2 = jac_fn2(params, x2)

    ntk = sum_and_contract(j1, j2)
    # TODO(schsam): If we care, this will not work if the output is not of
    # shape [n, output_dim].
    return np.transpose(ntk, (0, 2, 1, 3))

  return ntk_fn


empirical_ntk_fn = (empirical_implicit_ntk_fn
                    if FLAGS.tangents_optimized else
                    empirical_direct_ntk_fn)


def empirical_nngp_fn(f):
  """Returns a function to draw a single sample the NNGP of a given network `f`.

  This method assumes that slices of the random network outputs along the last
    dimension are i.i.d. (which is true for e.g. classifiers with a dense
    readout layer, or true for outputs of a CNN layer with the `NHWC` data
    format. As a result it treats outputs along that dimension as independent
    samples and only reports covariance along other dimensions.

  Note that the `ntk_monte_carlo` makes no such assumption and returns the full
    covariance.

  Args:
    f: a function computing the output of the neural network.
      From `jax.experimental.stax`: "takes params, inputs, and an rng key and
      applies the layer".

  Returns:
     A function to draw a single sample the NNGP of a given network `f`.
  """
  def nngp_fn(x1, x2, params, keys=None):
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

    Returns:
      A Monte Carlo estimate of the NNGP, a `np.ndarray` of shape
      `[batch_size_1] + output_shape[:-1] + [batch_size_2] + output_shape[:-1]`.
    """
    key1, key2 = _read_keys(keys)
    out1 = f(params, x1, rng=key1)
    if x2 is None:
      out2 = out1
    else:
      out2 = f(params, x2, rng=key2)

    out2 = np.expand_dims(out2, -1)
    nngp_12 = np.dot(out1, out2) / out1.shape[-1]
    return np.squeeze(nngp_12, -1)

  return nngp_fn


def empirical_kernel_fn(f):
  """Returns a function that computes single draws from NNGP and NT kernels."""

  kernel_fns = {
      'nngp': empirical_nngp_fn(f),
      'ntk': empirical_ntk_fn(f)
  }

  @utils.get_namedtuple('EmpiricalKernel')
  def kernel_fn(x1, x2, params, get=None, keys=None):
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

    Returns:
      If `get` is a string, returns the requested `np.ndarray`. If `get` is a
      tuple, returns an `EmpiricalKernel` namedtuple containing only the
      requested information.
    """
    if get is None:
      get = ('nngp', 'ntk')
    return {g: kernel_fns[g](x1, x2, params, keys) for g in get}

  return kernel_fn
