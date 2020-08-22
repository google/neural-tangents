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

"""Compute empirical NNGP and NTK; approximate functions via Taylor series.

NNGP and NTK are comuted using `empirical_nngp_fn`, `empirical_ntk_fn` (or
`empirical_direct_ntk_fn`), or `empirical_kernel_fn` (for both).

WARNING: resulting kernel shape is *nearly* `zip(f(x1).shape, f(x2).shape)`
subject to `trace_axes` and `diagonal_axes` parameters, which make certain
assumptions about the outputs `f(x)` that may only be true in the infinite width
/ infinite number of samples limit, or may not apply to your architecture. For
most precise results in the context of linearized training dynamics of a
specific finite-width network, set both `trace_axes=()` and `diagonal_axes=()`
to obtain the kernel exactly of shape `zip(f(x1).shape, f(x2).shape)`. Please
refer to individual functions' docstrings for details.
"""

import operator
from typing import Union, Callable, Optional, Tuple, Dict
import warnings


from jax import random
from jax.api import eval_shape
from jax.api import jacobian
from jax.api import jvp
from jax.api import vjp
import jax.numpy as np
from jax.tree_util import tree_multimap
from jax.tree_util import tree_reduce
from neural_tangents.utils import utils
from neural_tangents.utils.typing import ApplyFn, EmpiricalKernelFn, PyTree, PRNGKey, Axes


def linearize(f: Callable[..., np.ndarray],
              params: PyTree) -> Callable[..., np.ndarray]:
  """Returns a function `f_lin`, the first order taylor approximation to `f`.

  Example:
    >>> # Compute the MSE of the first order Taylor series of a function.
    >>> f_lin = linearize(f, params)
    >>> mse = np.mean((f(new_params, x) - f_lin(new_params, x)) ** 2)

  Args:
    f:
      A function that we would like to linearize. It should have the signature
      `f(params, *args, **kwargs)` where params is a `PyTree` and `f` should
      return an `np.ndarray`.
    params:
      Initial parameters to the function that we would like to take the
      Taylor series about. This can be any structure that is compatible with the
      JAX tree operations.

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
    f:
      A function that we would like to Taylor expand. It should have the
      signature `f(params, *args, **kwargs)` where `params` is a `PyTree`, and
      `f` returns a `np.ndarray`.
    params:
      Initial parameters to the function that we would like to take the Taylor
      series about. This can be any structure that is compatible with the JAX
      tree operations.
    degree:
      The degree of the Taylor expansion.

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


def empirical_implicit_ntk_fn(f: ApplyFn,
                              trace_axes: Axes = (-1,),
                              diagonal_axes: Axes = ()
                              ) -> EmpiricalKernelFn:
  r"""Returns a function to draw a single sample the NTK of a given network `f`.

  The Neural Tangent Kernel is defined as :math:`J(X_1) J(X_2)^T` where
  :math:`J` is the Jacobian :math:`df / dparams^T`. Computing the NTK directly
  involves instantiating the jacobian which takes
  `O(dataset_size * output_dim * parameters)` memory. It turns out it is
  substantially more efficient (especially as the number of parameters grows)
  to compute the NTK implicitly.

  The implicit kernel is derived by observing that:
  :math:`\Theta = J(X_1) J(X_2)^T = d[J(X_1) J(X_2)^T v] / d[v^T]`,
  for a vector :math:`v`. This allows the computation of the NTK to be phrased
  as: :math:`a(v) = J(X_2)^T v`, which is computed by a vector-Jacobian product;
  :math:`b(v) = J(X_1) a(v)` which is computed by a Jacobian-vector product; and
  :math:`\Theta = d[b(v)] / d[v^T]` which is computed by taking the Jacobian of
  :math:`b(v)`.

  Args:
    f:
      the function whose NTK we are computing. `f` should have the signature
      `f(params, inputs[, rng])` and should return `np.ndarray` of outputs.
    trace_axes:
      output axes to trace the output kernel over, i.e. compute only the trace
      of the covariance along the respective pair of axes (one pair for each
      axis in `trace_axes`). This allows to save space and compute if you are
      only interested in the respective trace, but also improve approximation
      accuracy if you know that covariance along these pairs of axes converges
      to a `constant * identity matrix` in the limit of interest (e.g.
      infinite width or infinite `n_samples`). A common use case is the channel
      / feature / logit axis, since activation slices along such axis are i.i.d.
      and the respective covariance along the respective pair of axes indeed
      converges to a constant-diagonal matrix in the infinite width or infinite
      `n_samples` limit.
      Also related to "contracting dimensions" in XLA terms.
      (https://www.tensorflow.org/xla/operation_semantics#dotgeneral)
    diagonal_axes:
      output axes to diagonalize the output kernel over, i.e. compute only the
      diagonal of the covariance along the respective pair of axes (one pair for
      each axis in `diagonal_axes`). This allows to save space and compute, if
      off-diagonal values along these axes are not needed, but also improve
      approximation accuracy if their limiting value is known theoretically,
      e.g. if they vanish in the limit of interest (e.g. infinite
      width or infinite `n_samples`). If you further know that on-diagonal
      values converge to the same constant in your limit of interest, you should
      specify these axes in `trace_axes` instead, to save even more compute and
      gain even more accuracy. A common use case is computing the variance
      (instead of covariance) along certain axes.
      Also related to "batch dimensions" in XLA terms.
      (https://www.tensorflow.org/xla/operation_semantics#dotgeneral)

  Returns:
    A function `ntk_fn` that computes the empirical ntk.
  """
  def ntk_fn(x1: np.ndarray,
             x2: Optional[np.ndarray],
             params: PyTree,
             **apply_fn_kwargs) -> np.ndarray:
    """Computes a single sample of the empirical NTK (implicit differentiation).

    Args:
      x1:
        first batch of inputs.
      x2:
        second batch of inputs. `x2=None` means `x2=x1`. `f(x2)` must have a
        matching shape with `f(x1)` on `trace_axes` and `diagonal_axes`.
      params:
        A `PyTree` of parameters about which we would like to compute the
        neural tangent kernel.
      **apply_fn_kwargs:
        keyword arguments passed to `apply_fn`. `apply_fn_kwargs` will be split
        into `apply_fn_kwargs1` and `apply_fn_kwargs2` by the `_split_kwargs`
        function which will be passed to `apply_fn`. In particular, the rng key
        in `apply_fn_kwargs`, will be split into two different (if `x1 != x2`)
        or same (if `x1 == x2`) rng keys. See the `_read_key` function for more
        details.

    Returns:
      A single sample of the empirical NTK. The shape of the kernel is "almost"
      `zip(f(x1).shape, f(x2).shape)` except for:
      1) `trace_axes` are absent as they are contracted over.
      2) `diagonal_axes` are present only once.
      All other axes are present twice.
    """

    apply_fn_kwargs1, apply_fn_kwargs2 = _split_kwargs(apply_fn_kwargs, x1, x2)
    f1 = _get_f_params(f, x1, **apply_fn_kwargs1)
    f2 = f1 if x2 is None else _get_f_params(f, x2, **apply_fn_kwargs2)

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
    return _trace_and_diagonal(ntk, trace_axes, diagonal_axes)

  return ntk_fn


def empirical_direct_ntk_fn(f: ApplyFn,
                            trace_axes: Axes = (-1,),
                            diagonal_axes: Axes = ()
                            ) -> EmpiricalKernelFn:
  """Returns a function to draw a single sample the NTK of a given network `f`.

  The Neural Tangent Kernel is defined as :math:`J(X_1) J(X_2)^T` where
  :math:`J` is the Jacobian :math:`df/dparams`. This function instatiates the
  Jacobians directly and computes their outer product.

  Args:
    f:
      the function whose NTK we are computing. `f` should have the signature
      `f(params, inputs[, rng])` and should return an `np.ndarray` outputs.
    trace_axes:
      output axes to trace the output kernel over, i.e. compute only the trace
      of the covariance along the respective pair of axes (one pair for each
      axis in `trace_axes`). This allows to save space and compute if you are
      only interested in the respective trace, but also improve approximation
      accuracy if you know that covariance along these pairs of axes converges
      to a `constant * identity matrix` in the limit of interest (e.g.
      infinite width or infinite `n_samples`). A common use case is the channel
      / feature / logit axis, since activation slices along such axis are i.i.d.
      and the respective covariance along the respective pair of axes indeed
      converges to a constant-diagonal matrix in the infinite width or infinite
      `n_samples` limit.
      Also related to "contracting dimensions" in XLA terms.
      (https://www.tensorflow.org/xla/operation_semantics#dotgeneral)
    diagonal_axes:
      output axes to diagonalize the output kernel over, i.e. compute only the
      diagonal of the covariance along the respective pair of axes (one pair for
      each axis in `diagonal_axes`). This allows to save space and compute, if
      off-diagonal values along these axes are not needed, but also improve
      approximation accuracy if their limiting value is known theoretically,
      e.g. if they vanish in the limit of interest (e.g. infinite
      width or infinite `n_samples`). If you further know that on-diagonal
      values converge to the same constant in your limit of interest, you should
      specify these axes in `trace_axes` instead, to save even more compute and
      gain even more accuracy. A common use case is computing the variance
      (instead of covariance) along certain axes.
      Also related to "batch dimensions" in XLA terms.
      (https://www.tensorflow.org/xla/operation_semantics#dotgeneral)

  Returns:
    A function `ntk_fn` that computes the empirical ntk.
  """
  def sum_and_contract(j1, j2, output_ndim):
    _diagonal_axes = utils.canonicalize_axis(diagonal_axes, output_ndim)
    _trace_axes = utils.canonicalize_axis(trace_axes, output_ndim)

    def contract(x, y):
      param_axes = list(range(x.ndim))[output_ndim:]
      contract_axes = _trace_axes + param_axes
      return utils.dot_general(x, y, contract_axes, _diagonal_axes)

    return tree_reduce(operator.add, tree_multimap(contract, j1, j2))

  def ntk_fn(x1: np.ndarray,
             x2: Optional[np.ndarray],
             params: PyTree,
             **apply_fn_kwargs) -> np.ndarray:
    """Computes a single sample of the empirical NTK (jacobian outer product).

    Args:
      x1:
        first batch of inputs.
      x2:
        second batch of inputs. `x2=None` means `x2=x1`. `f(x2)` must have a
        matching shape with `f(x1)` on `trace_axes` and `diagonal_axes`.
      params:
        A `PyTree` of parameters about which we would like to compute the
        neural tangent kernel.
      **apply_fn_kwargs:
        keyword arguments passed to `apply_fn`. `apply_fn_kwargs` will be split
        into `apply_fn_kwargs1` and `apply_fn_kwargs2` by the `_split_kwargs`
        function which will be passed to `apply_fn`. In particular, the rng key
        in `apply_fn_kwargs`, will be split into two different (if `x1!=x2`) or
        same (if `x1==x2`) rng keys. See the `_read_key` function for more
        details.

    Returns:
      A single sample of the empirical NTK. The shape of the kernel is "almost"
      `zip(f(x1).shape, f(x2).shape)` except for:
      1) `trace_axes` are absent as they are contracted over.
      2) `diagonal_axes` are present only once.
      All other axes are present twice.
    """

    apply_fn_kwargs1, apply_fn_kwargs2 = _split_kwargs(apply_fn_kwargs, x1, x2)
    f1 = _get_f_params(f, x1, **apply_fn_kwargs1)
    jac_fn1 = jacobian(f1)
    j1 = jac_fn1(params)
    if x2 is None:
      j2 = j1
    else:
      f2 = _get_f_params(f, x2, **apply_fn_kwargs2)
      jac_fn2 = jacobian(f2)
      j2 = jac_fn2(params)

    fx1 = eval_shape(f1, params)
    ntk = sum_and_contract(j1, j2, fx1.ndim)
    return ntk / utils.size_at(fx1, trace_axes)

  return ntk_fn


empirical_ntk_fn = empirical_implicit_ntk_fn


def empirical_nngp_fn(f: ApplyFn,
                      trace_axes: Axes = (-1,),
                      diagonal_axes: Axes = ()
                      ) -> EmpiricalKernelFn:
  """Returns a function to draw a single sample the NNGP of a given network `f`.

  Args:
    f:
      the function whose NNGP we are computing. `f` should have the signature
      `f(params, inputs[, rng])` and should return an `np.ndarray` outputs.
    trace_axes:
      output axes to trace the output kernel over, i.e. compute only the trace
      of the covariance along the respective pair of axes (one pair for each
      axis in `trace_axes`). This allows to save space and compute if you are
      only interested in the respective trace, but also improve approximation
      accuracy if you know that covariance along these pairs of axes converges
      to a `constant * identity matrix` in the limit of interest (e.g.
      infinite width or infinite `n_samples`). A common use case is the channel
      / feature / logit axis, since activation slices along such axis are i.i.d.
      and the respective covariance along the respective pair of axes indeed
      converges to a constant-diagonal matrix in the infinite width or infinite
      `n_samples` limit.
      Also related to "contracting dimensions" in XLA terms.
      (https://www.tensorflow.org/xla/operation_semantics#dotgeneral)
    diagonal_axes:
      output axes to diagonalize the output kernel over, i.e. compute only the
      diagonal of the covariance along the respective pair of axes (one pair for
      each axis in `diagonal_axes`). This allows to save space and compute, if
      off-diagonal values along these axes are not needed, but also improve
      approximation accuracy if their limiting value is known theoretically,
      e.g. if they vanish in the limit of interest (e.g. infinite
      width or infinite `n_samples`). If you further know that on-diagonal
      values converge to the same constant in your limit of interest, you should
      specify these axes in `trace_axes` instead, to save even more compute and
      gain even more accuracy. A common use case is computing the variance
      (instead of covariance) along certain axes.
      Also related to "batch dimensions" in XLA terms.
      (https://www.tensorflow.org/xla/operation_semantics#dotgeneral)

  Returns:
     A function to draw a single sample the NNGP of a given network `f`.
  """
  def nngp_fn(x1: np.ndarray,
              x2: Optional[np.ndarray],
              params: PyTree,
              **apply_fn_kwargs) -> np.ndarray:
    """Computes a single sample of the empirical NNGP.

    Args:
      x1:
        first batch of inputs.
      x2:
        second batch of inputs. `x2=None` means `x2=x1`. `f(x2)` must have a
        matching shape with `f(x1)` on `trace_axes` and `diagonal_axes`.
      params:
        A `PyTree` of parameters about which we would like to compute the
        neural tangent kernel.
      **apply_fn_kwargs:
        keyword arguments passed to `apply_fn`. `apply_fn_kwargs` will be split
        into `apply_fn_kwargs1` and `apply_fn_kwargs2` by the `_split_kwargs`
        function which will be passed to `apply_fn`. In particular, the rng key
        in `apply_fn_kwargs`, will be split into two different (if `x1!=x2`) or
        same (if `x1==x2`) rng keys. See the `_read_key` function for more
        details.

    Returns:
      A single sample of the empirical NNGP. The shape of the kernel is "almost"
      `zip(f(x1).shape, f(x2).shape)` except for:
      1) `trace_axes` are absent as they are contracted over.
      2) `diagonal_axes` are present only once.
      All other axes are present twice.
    """

    def output(x, **kwargs):
      out = f(params, x, **kwargs)
      masked_output = utils.get_masked_array(out)
      return masked_output.masked_value

    apply_fn_kwargs1, apply_fn_kwargs2 = _split_kwargs(apply_fn_kwargs, x1, x2)

    out1 = output(x1, **apply_fn_kwargs1)
    if x2 is None:
      out2 = out1
    else:
      out2 = output(x2, **apply_fn_kwargs2)

    dot = utils.dot_general(out1, out2, trace_axes, diagonal_axes)
    return dot / utils.size_at(out1, trace_axes)

  return nngp_fn


def empirical_kernel_fn(f: ApplyFn,
                        trace_axes: Axes = (-1,),
                        diagonal_axes: Axes = ()
                        ) -> EmpiricalKernelFn:
  """Returns a function that computes single draws from NNGP and NT kernels.

  Args:
    f:
      the function whose NTK we are computing. `f` should have the signature
      `f(params, inputs[, rng])` and should return an `np.ndarray` outputs.
    trace_axes:
      output axes to trace the output kernel over, i.e. compute only the trace
      of the covariance along the respective pair of axes (one pair for each
      axis in `trace_axes`). This allows to save space and compute if you are
      only interested in the respective trace, but also improve approximation
      accuracy if you know that covariance along these pairs of axes converges
      to a `constant * identity matrix` in the limit of interest (e.g.
      infinite width or infinite `n_samples`). A common use case is the channel
      / feature / logit axis, since activation slices along such axis are i.i.d.
      and the respective covariance along the respective pair of axes indeed
      converges to a constant-diagonal matrix in the infinite width or infinite
      `n_samples` limit.
      Also related to "contracting dimensions" in XLA terms.
      (https://www.tensorflow.org/xla/operation_semantics#dotgeneral)
    diagonal_axes:
      output axes to diagonalize the output kernel over, i.e. compute only the
      diagonal of the covariance along the respective pair of axes (one pair for
      each axis in `diagonal_axes`). This allows to save space and compute, if
      off-diagonal values along these axes are not needed, but also improve
      approximation accuracy if their limiting value is known theoretically,
      e.g. if they vanish in the limit of interest (e.g. infinite
      width or infinite `n_samples`). If you further know that on-diagonal
      values converge to the same constant in your limit of interest, you should
      specify these axes in `trace_axes` instead, to save even more compute and
      gain even more accuracy. A common use case is computing the variance
      (instead of covariance) along certain axes.
      Also related to "batch dimensions" in XLA terms.
      (https://www.tensorflow.org/xla/operation_semantics#dotgeneral)

  Returns:
    A function to draw a single sample the NNGP and NTK empirical kernels of a
    given network `f`.
  """
  kernel_fns = {
      'nngp': empirical_nngp_fn(f, trace_axes, diagonal_axes),
      'ntk': empirical_ntk_fn(f, trace_axes, diagonal_axes)
  }

  @utils.get_namedtuple('EmpiricalKernel')
  def kernel_fn(x1: np.ndarray,
                x2: Optional[np.ndarray],
                get: Union[None, str, Tuple[str, ...]],
                params: PyTree,
                **apply_fn_kwargs) -> Dict[str, np.ndarray]:
    """Computes a single sample of the empirical kernel of type `get`.

    Args:
      x1:
        first batch of inputs.
      x2:
        second batch of inputs. `x2=None` means `x2=x1`. `f(x2)` must have a
        matching shape with `f(x1)` on `trace_axes` and `diagonal_axes`.
      get:
        type of the empirical kernel. `get=None` means `get=("nngp", "ntk")`.
        Can be a string (`"nngp"`) or a tuple of strings (`("ntk", "nngp")`).
      params:
        A `PyTree` of parameters about which we would like to compute the
        neural tangent kernel.
      **apply_fn_kwargs:
        keyword arguments passed to `apply_fn`. `apply_fn_kwargs` will be split
        into `apply_fn_kwargs1` and `apply_fn_kwargs2` by the `_split_kwargs`
        function which will be passed to `apply_fn`. In particular, the rng key
        in `apply_fn_kwargs`, will be split into two different (if `x1!=x2`) or
        same (if `x1==x2`) rng keys. See the `_read_key` function for more
        details.

    Returns:
      A single sample of the empirical kernel. The shape is "almost"
      `zip(f(x1).shape, f(x2).shape)` except for:
      1) `trace_axes` are absent as they are contracted over.
      2) `diagonal_axes` are present only once.
      All other axes are present twice.

      If `get` is a string, returns the requested `np.ndarray`. If `get` is a
      tuple, returns an `EmpiricalKernel` namedtuple containing the
      requested information.
    """
    if get is None:
      get = ('nngp', 'ntk')

    return {g: kernel_fns[g](x1, x2, params, **apply_fn_kwargs)
            for g in get}  # pytype: disable=wrong-arg-count
  return kernel_fn


# INTERNAL UTILITIES


def _read_keys(key, x1, x2):
  """Read dropout key.

     `key` might be a tuple of two rng keys or a single rng key or None. In
     either case, `key` will be mapped into two rng keys `key1` and `key2` to
     make sure `(x1==x2) == (key1==key2)`.
  """

  if key is None or x2 is None:
    key1 = key2 = key
  elif isinstance(key, tuple) and len(key) == 2:
    key1, key2 = key
    new_key = np.where(utils.x1_is_x2(key1, key2),
                       random.fold_in(key2, 1), key2)
    key2 = np.where(utils.x1_is_x2(x1, x2), key1, new_key)
    warnings.warn('The value of `key[1]` might be replaced by a new value if '
                  'key[0] == key[1] and x1 != x2 or key[0] != key[1] and '
                  'x1 == x2.')
  elif isinstance(key, np.ndarray):
    key1 = key
    key2 = np.where(utils.x1_is_x2(x1, x2), key1, random.fold_in(key, 1))
  else:
    raise TypeError(type(key))
  return key1, key2


def _get_f_params(f, x, **apply_fn_kwargs):
  def _f(p):
    out = f(p, x, **apply_fn_kwargs)
    # TODO(romann): normalize properly if output is masked.
    out = utils.get_masked_array(out)
    return out.masked_value
  return _f


def _trace_and_diagonal(ntk: np.ndarray,
                        trace_axes: Axes,
                        diagonal_axes: Axes) -> np.ndarray:
  """Extract traces and diagonals along respective pairs of axes from the `ntk`.

  Args:
    ntk:
      input empirical NTK of shape `(N1, X, Y, Z, ..., N2, X, Y, Z, ...)`.
    trace_axes:
      axes (among `X, Y, Z, ...`) to trace over, i.e. compute the trace along
      and remove the  respective pairs of axes from the `ntk`.
    diagonal_axes:
      axes (among `X, Y, Z, ...`) to take the diagonal along, i.e. extract the
      diagonal along the respective pairs of axes from the `ntk` (and hence
      reduce the resulting `ntk` axes count by 2).
  Returns:
    An array of shape, for example, `(N1, N2, Y, Z, Z, ...)` if
    `trace_axes=(1,)` (`X` axes removed), and `diagonal_axes=(2,)` (`Y` axes
    replaced with a single `Y` axis).
  """
  if ntk.ndim % 2 == 1:
    raise ValueError('Expected an even-dimensional kernel. Please file a bug at'
                     'https://github.com/google/neural-tangents/issues/new')

  output_ndim = ntk.ndim // 2

  trace_axes = utils.canonicalize_axis(trace_axes, output_ndim)
  diagonal_axes = utils.canonicalize_axis(diagonal_axes, output_ndim)

  n_diag, n_trace = len(diagonal_axes), len(trace_axes)
  contract_size = utils.size_at(ntk.shape[:output_ndim], trace_axes)

  for i, c in enumerate(reversed(trace_axes)):
    ntk = np.trace(ntk, axis1=c, axis2=output_ndim + c - i)

  for i, d in enumerate(diagonal_axes):
    axis1 = d - i
    axis2 = output_ndim + d - 2 * i - n_trace
    for c in trace_axes:
      if c < d:
        axis1 -= 1
        axis2 -= 1
    ntk = np.diagonal(ntk, axis1=axis1, axis2=axis2)

  ntk = utils.zip_axes(ntk, 0, ntk.ndim - n_diag)
  res_diagonal_axes = utils.get_res_batch_dims(trace_axes, diagonal_axes)
  ntk = np.moveaxis(ntk, range(-n_diag, 0), res_diagonal_axes)
  return ntk / contract_size


def _split_kwargs(kwargs, x1, x2):
  """Spliting `kwargs`.

     Specifically,
       1. if kwarg is an rng key, it will be split into two keys.
       2. else if it is a tuple of length two, the tuple will be split into two
          parts, one for kwargs1 and the other for kwargs2.
       3. else it is copied to kwargs1 and kwargs2.

  """
  kwargs1 = {}
  kwargs2 = {}
  for k, v in kwargs.items():
    if k == 'rng':
      key1, key2 = _read_keys(v, x1, x2)
      kwargs1[k] = key1
      kwargs2[k] = key2
    elif isinstance(v, tuple) and len(v) == 2:
      kwargs1[k] = v[0]
      kwargs2[k] = v[1]
    else:
      kwargs1[k] = kwargs2[k] = v

  return kwargs1, kwargs2
