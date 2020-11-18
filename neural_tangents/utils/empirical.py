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

All functions in this module are applicable to any JAX functions of proper
signatures (not only those from `nt.stax`).

NNGP and NTK are computed using `empirical_nngp_fn`, `empirical_ntk_fn`, or
 `empirical_kernel_fn` (for both). The kernels have a very specific output
shape convention that may be unexpected. Further, NTK has multiple
implementations that may perform differently depending on the task.
Please read individual functions' docstrings.

Example:
  >>>  from jax import random
  >>>  import neural_tangents as nt
  >>>  from neural_tangents import stax
  >>>
  >>>  key1, key2, key3 = random.split(random.PRNGKey(1), 3)
  >>>  x_train = random.normal(key1, (20, 32, 32, 3))
  >>>  y_train = random.uniform(key1, (20, 10))
  >>>  x_test = random.normal(key2, (5, 32, 32, 3))
  >>>
  >>>  # A narrow CNN.
  >>>  init_fn, f, _ = stax.serial(
  >>>      stax.Conv(32, (3, 3)),
  >>>      stax.Relu(),
  >>>      stax.Conv(32, (3, 3)),
  >>>      stax.Relu(),
  >>>      stax.Conv(32, (3, 3)),
  >>>      stax.Flatten(),
  >>>      stax.Dense(10)
  >>>  )
  >>>
  >>>  _, params = init_fn(key3, x_train.shape)
  >>>
  >>>  # Default setting: reducing over logits; pass `vmap_axes=0` because the
  >>>  # network is iid along the batch axis, no BatchNorm. Use default
  >>>  # `implementation=1` since the network has few trainable parameters.
  >>>  kernel_fn = nt.empirical_kernel_fn(f, trace_axes=(-1,),
  >>>                                     vmap_axes=0, implementation=1)
  >>>
  >>>  # (5, 20) np.ndarray test-train NNGP/NTK
  >>>  nngp_test_train = kernel_fn(x_test, x_train, 'nngp', params)
  >>>  ntk_test_train = kernel_fn(x_test, x_train, 'ntk', params)
  >>>
  >>>  # Full kernel: not reducing over logits.
  >>>  kernel_fn = nt.empirical_kernel_fn(f, trace_axes=(), vmap_axes=0)
  >>>
  >>>  # (5, 20, 10, 10) np.ndarray test-train NNGP/NTK namedtuple.
  >>>  k_test_train = kernel_fn(x_test, x_train, params)
  >>>
  >>>  # A wide FCN with lots of parameters
  >>>  init_fn, f, _ = stax.serial(
  >>>      stax.Flatten(),
  >>>      stax.Dense(1024),
  >>>      stax.Relu(),
  >>>      stax.Dense(1024),
  >>>      stax.Relu(),
  >>>      stax.Dense(10)
  >>>  )
  >>>
  >>>  _, params = init_fn(key3, x_train.shape)
  >>>
  >>>  # Use implicit differentiation in NTK: `implementation=2` to reduce
  >>>  # memory cost, since the network has many trainable parameters.
  >>>  ntk_fn = nt.empirical_ntk_fn(f, vmap_axes=0, implementation=2)
  >>>
  >>>  # (5, 5) np.ndarray test-test NTK
  >>>  ntk_test_train = ntk_fn(x_test, None, params)
  >>>
  >>>  # Compute only output variances:
  >>>  nngp_fn = nt.empirical_nngp_fn(f, diagonal_axes=(0,))
  >>>
  >>>  # (20,) np.ndarray train-train diagonal NNGP
  >>>  nngp_train_train_diag = nngp_fn(x_train, None, params)
"""

import operator
from typing import Union, Callable, Optional, Tuple, Dict
from jax.api import eval_shape, jacobian, jvp, vjp, vmap, _std_basis, _unravel_array_into_pytree, linear_transpose
import jax.numpy as np
from jax.tree_util import tree_flatten, tree_unflatten, tree_multimap, tree_reduce, tree_map
from neural_tangents.utils import utils
from neural_tangents.utils.typing import ApplyFn, EmpiricalKernelFn, NTTree, PyTree, Axes, VMapAxes


def linearize(f: Callable[..., PyTree],
              params: PyTree) -> Callable[..., PyTree]:
  """Returns a function `f_lin`, the first order taylor approximation to `f`.

  Example:
    >>> # Compute the MSE of the first order Taylor series of a function.
    >>> f_lin = linearize(f, params)
    >>> mse = np.mean((f(new_params, x) - f_lin(new_params, x)) ** 2)

  Args:
    f:
      A function that we would like to linearize. It should have the signature
      `f(params, *args, **kwargs)` where params is a `PyTree` and `f` should
      return a `PyTree`.
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
    dparams = _sub(p, params)
    f_params_x, proj = jvp(lambda param: f(param, *args, **kwargs),
                           (params,), (dparams,))
    return _add(f_params_x, proj)
  return f_lin


def taylor_expand(f: Callable[..., PyTree],
                  params: PyTree,
                  degree: int) -> Callable[..., PyTree]:
  """Returns a function `f_tayl`, Taylor approximation to `f` of order `degree`.

  Example:
    >>> # Compute the MSE of the third order Taylor series of a function.
    >>> f_tayl = taylor_expand(f, params, 3)
    >>> mse = np.mean((f(new_params, x) - f_tayl(new_params, x)) ** 2)

  Args:
    f:
      A function that we would like to Taylor expand. It should have the
      signature `f(params, *args, **kwargs)` where `params` is a `PyTree`, and
      `f` returns a `PyTree`.
    params:
      Initial parameters to the function that we would like to take the Taylor
      series about. This can be any structure that is compatible with the JAX
      tree operations.
    degree:
      The degree of the Taylor expansion.

  Returns:
    A function `f_tayl(new_params, *args, **kwargs)` whose signature is the
    same as `f`. Here `f_tayl` implements the `degree`-order taylor series of
    `f` about `params`.
  """
  def taylorize_r(f, params, dparams, degree, current_degree):
    """Recursive function to accumulate contributions to the Taylor series."""
    if current_degree == degree:
      return f(params)

    def f_jvp(p):
      _, val_jvp = jvp(f, (p,), (dparams,))
      return val_jvp

    df = taylorize_r(f_jvp, params, dparams, degree, current_degree + 1)
    return _add(f(params), _div(df, (current_degree + 1)))

  def f_tayl(p, *args, **kwargs):
    dparams = _sub(p, params)
    return taylorize_r(lambda param: f(param, *args, **kwargs),
                       params, dparams, degree, 0)

  return f_tayl


# Empirical Kernel


def empirical_kernel_fn(
    f: ApplyFn,
    trace_axes: Axes = (-1,),
    diagonal_axes: Axes = (),
    vmap_axes: VMapAxes = None,
    implementation: int = 1
) -> EmpiricalKernelFn:
  r"""Returns a function that computes single draws from NNGP and NT kernels.

  WARNING: resulting kernel shape is *nearly* `zip(f(x1).shape, f(x2).shape)`
  subject to `trace_axes` and `diagonal_axes` parameters, which make certain
  assumptions about the outputs `f(x)` that may only be true in the infinite
  width / infinite number of samples limit, or may not apply to your
  architecture. For most precise results in the context of linearized training
  dynamics of a specific finite-width network, set both `trace_axes=()` and
  `diagonal_axes=()` to obtain the kernel exactly of shape
  `zip(f(x1).shape, f(x2).shape)`.

  For networks with multiple (i.e. lists, tuples, PyTrees) outputs, in principal
  the empirical kernels will have terms measuring the covariance between the
  outputs. Here, we ignore these cross-terms and consider each output
  separately. Please raise an issue if this feature is important to you.

  Args:
    f:
      the function whose NTK we are computing. `f` should have the signature
      `f(params, inputs, **kwargs)` and should return an `np.ndarray` outputs.
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
    vmap_axes:
      applicable only to NTK.

      A triple of `(in_axes, out_axes, kwargs_axes)`
      passed to `vmap` to evaluate the empirical NTK in parallel ove these axes.
      Precisely, providing this argument implies that `f(params, x, **kwargs)`
      equals to a concatenation along `out_axes` of `f` applied to slices of
      `x` and `**kwargs` along `in_axes` and `kwargs_axes`. In other words, it
      certifies that `f` can be evaluated as a `vmap` with `out_axes=out_axes`
      over `x` (along `in_axes`) and those arguments in `**kwargs` that are
      present in `kwargs_axes.keys()` (along `kwargs_axes.values()`).

      For example if `_, f, _ = nt.stax.Aggregate()`, `f` is called via
      `f(params, x, pattern=pattern)`. By default, inputs `x`, patterns
      `pattern`, and outputs of `f` are all batched along the leading `0`
      dimension, and each output `f(params, x, pattern=pattern)[i]` only
      depends on the inputs `x[i]` and `pattern[i]`. In this case, we can
      pass `vmap_axes=(0, 0, dict(pattern=0)` to specify along which dimensions
      inputs, outputs, and keyword arguments are batched respectively.

      This allows us to evaluate Jacobians much more
      efficiently. If `vmap_axes` is not a triple, it is interpreted as
      `in_axes = out_axes = vmap_axes, kwargs_axes = {}`. For example a very
      common use case is `vmap_axes=0` for a neural network with leading (`0`)
      batch dimension, both for inputs and outputs, and no interactions between
      different elements of the batch (e.g. no BatchNorm, and, in the case of
      `nt.stax`, also no Dropout). However, if there is interaction between
      batch elements or no concept of a batch axis at all, `vmap_axes` must be
      set to `None`, to avoid wrong (and potentially silent) results.
    implementation:
      applicable only to NTK.

      `1` or `2`.

      `1` directly instantiates Jacobians and computes their outer
      product.

      `2` uses implicit differentiation to avoid instantiating whole
      Jacobians at once. The implicit kernel is derived by observing that:
      :math:`\Theta = J(X_1) J(X_2)^T = [J(X_1) J(X_2)^T](I)`,
      i.e. a linear function :math:`[J(X_1) J(X_2)^T]` applied to an identity
      matrix :math:`I`. This allows the computation of the NTK to be
      phrased as: :math:`a(v) = J(X_2)^T v`, which is computed by a
      vector-Jacobian product; :math:`b(v) = J(X_1) a(v)` which is computed by
      a Jacobian-vector product; and :math:`\Theta = [b(v)] / d[v^T](I)` which
      is computed via a `vmap` of :math:`b(v)` over columns of the identity
      matrix :math:`I`.

      It is best to benchmark each method on your specific task. We suggest
      using `1` unless you get OOMs due to large number of trainable parameters,
      otherwise - `2`.

  Returns:
    A function to draw a single sample the NNGP and NTK empirical kernels of a
    given network `f`.
  """
  kwargs = dict(f=f,
                trace_axes=trace_axes,
                diagonal_axes=diagonal_axes)

  kernel_fns = {
      'nngp': empirical_nngp_fn(**kwargs),
      'ntk': empirical_ntk_fn(**kwargs,
                              vmap_axes=vmap_axes,
                              implementation=implementation)
  }

  @utils.get_namedtuple('EmpiricalKernel')
  def kernel_fn(x1: NTTree[np.ndarray],
                x2: Optional[NTTree[np.ndarray]],
                get: Union[None, str, Tuple[str, ...]],
                params: PyTree,
                **apply_fn_kwargs) -> NTTree[Dict[str, np.ndarray]]:
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
        into `apply_fn_kwargs1` and `apply_fn_kwargs2` by the `split_kwargs`
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

    out_dict = {g: kernel_fns[g](x1, x2, params, **apply_fn_kwargs)
                for g in get}
    out_dict = _dict_of_tree_to_tree_of_dict(out_dict, get)

    return out_dict

  return kernel_fn


def empirical_nngp_fn(f: ApplyFn,
                      trace_axes: Axes = (-1,),
                      diagonal_axes: Axes = ()
                      ) -> Callable[[NTTree[np.ndarray],
                                     Optional[NTTree[np.ndarray]],
                                     PyTree],
                                    NTTree[np.ndarray]]:
  """Returns a function to draw a single sample the NNGP of a given network `f`.

  The Neural Network Gaussian Process (NNGP) kernel is defined as
  :math:`f(X_1) f(X_2)^T`, i.e. the outer product of the function outputs.

  WARNING: resulting kernel shape is *nearly* `zip(f(x1).shape, f(x2).shape)`
  subject to `trace_axes` and `diagonal_axes` parameters, which make certain
  assumptions about the outputs `f(x)` that may only be true in the infinite
  width / infinite number of samples limit, or may not apply to your
  architecture. For most precise results in the context of linearized training
  dynamics of a specific finite-width network, set both `trace_axes=()` and
  `diagonal_axes=()` to obtain the kernel exactly of shape
  `zip(f(x1).shape, f(x2).shape)`.

  For networks with multiple (i.e. lists, tuples, PyTrees) outputs, in principal
  the empirical kernels will have terms measuring the covariance between the
  outputs. Here, we ignore these cross-terms and consider each output
  separately. Please raise an issue if this feature is important to you.

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
        into `apply_fn_kwargs1` and `apply_fn_kwargs2` by the `split_kwargs`
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
      return utils.nt_tree_fn()(lambda x: x.masked_value)(masked_output)

    kwargs1, kwargs2 = utils.split_kwargs(apply_fn_kwargs, x1, x2)

    out1 = output(x1, **kwargs1)
    out2 = output(x2, **kwargs2) if not utils.all_none(x2) else out1

    @utils.nt_tree_fn()
    def contract(out1, out2):
      dot = utils.dot_general(out1, out2, trace_axes, diagonal_axes)
      return dot / utils.size_at(out1, trace_axes)

    return contract(out1, out2)

  return nngp_fn


def empirical_ntk_fn(f: ApplyFn,
                     trace_axes: Axes = (-1,),
                     diagonal_axes: Axes = (),
                     vmap_axes: VMapAxes = None,
                     implementation: int = 1
                     ) -> Callable[[NTTree[np.ndarray],
                                    Optional[NTTree[np.ndarray]],
                                    PyTree],
                                   NTTree[np.ndarray]]:
  r"""Returns a function to draw a single sample the NTK of a given network `f`.

  The Neural Tangent Kernel is defined as :math:`J(X_1) J(X_2)^T` where
  :math:`J` is the Jacobian :math:`df/dparams` of shape
  `full_output_shape + params.shape`.

  For best performance:
  1) pass `x2=None` if `x1 == x2;
  2) prefer square batches (i.e `x1.shape == x2.shape`);
  3) make sure to set `vmap_axes` correctly.
  4) try different `implementation` values.

  WARNING: Resulting kernel shape is *nearly* `zip(f(x1).shape, f(x2).shape)`
  subject to `trace_axes` and `diagonal_axes` parameters, which make certain
  assumptions about the outputs `f(x)` that may only be true in the infinite
  width / infinite number of samples limit, or may not apply to your
  architecture. For most precise results in the context of linearized training
  dynamics of a specific finite-width network, set both `trace_axes=()` and
  `diagonal_axes=()` to obtain the kernel exactly of shape
  `zip(f(x1).shape, f(x2).shape)`.

  For networks with multiple (i.e. lists, tuples, PyTrees) outputs, in principal
  the empirical kernels will have terms measuring the covariance between the
  outputs. Here, we ignore these cross-terms and consider each output
  separately. Please raise an issue if this feature is important to you.

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
    vmap_axes:
      A triple of `(in_axes, out_axes, kwargs_axes)`
      passed to `vmap` to evaluate the empirical NTK in parallel ove these axes.
      Precisely, providing this argument implies that `f(params, x, **kwargs)`
      equals to a concatenation along `out_axes` of `f` applied to slices of
      `x` and `**kwargs` along `in_axes` and `kwargs_axes`. In other words, it
      certifies that `f` can be evaluated as a `vmap` with `out_axes=out_axes`
      over `x` (along `in_axes`) and those arguments in `**kwargs` that are
      present in `kwargs_axes.keys()` (along `kwargs_axes.values()`).

      For example if `_, f, _ = nt.stax.Aggregate()`, `f` is called via
      `f(params, x, pattern=pattern)`. By default, inputs `x`, patterns
      `pattern`, and outputs of `f` are all batched along the leading `0`
      dimension, and each output `f(params, x, pattern=pattern)[i]` only
      depends on the inputs `x[i]` and `pattern[i]`. In this case, we can
      pass `vmap_axes=(0, 0, dict(pattern=0)` to specify along which dimensions
      inputs, outputs, and keyword arguments are batched respectively.

      This allows us to evaluate Jacobians much more
      efficiently. If `vmap_axes` is not a triple, it is interpreted as
      `in_axes = out_axes = vmap_axes, kwargs_axes = {}`. For example a very
      common use case is `vmap_axes=0` for a neural network with leading (`0`)
      batch dimension, both for inputs and outputs, and no interactions between
      different elements of the batch (e.g. no BatchNorm, and, in the case of
      `nt.stax`, also no Dropout). However, if there is interaction between
      batch elements or no concept of a batch axis at all, `vmap_axes` must be
      set to `None`, to avoid wrong (and potentially silent) results.
    implementation:
      `1` or `2`.

      `1` directly instantiates Jacobians and computes their outer
      product.

      `2` uses implicit differentiation to avoid instantiating whole
      Jacobians at once. The implicit kernel is derived by observing that:
      :math:`\Theta = J(X_1) J(X_2)^T = [J(X_1) J(X_2)^T](I)`,
      i.e. a linear function :math:`[J(X_1) J(X_2)^T]` applied to an identity
      matrix :math:`I`. This allows the computation of the NTK to be
      phrased as: :math:`a(v) = J(X_2)^T v`, which is computed by a
      vector-Jacobian product; :math:`b(v) = J(X_1) a(v)` which is computed by
      a Jacobian-vector product; and :math:`\Theta = [b(v)] / d[v^T](I)` which
      is computed via a `vmap` of :math:`b(v)` over columns of the identity
      matrix :math:`I`.

      It is best to benchmark each method on your specific task. We suggest
      using `1` unless you get OOMs due to large number of trainable parameters,
      otherwise - `2`.

  Returns:
    A function `ntk_fn` that computes the empirical ntk.
  """
  kwargs = dict(f=f,
                trace_axes=trace_axes,
                diagonal_axes=diagonal_axes,
                vmap_axes=vmap_axes)

  if implementation == 1:
    return _empirical_direct_ntk_fn(**kwargs)

  if implementation == 2:
    return _empirical_implicit_ntk_fn(**kwargs)

  raise ValueError(implementation)


def _empirical_implicit_ntk_fn(f: ApplyFn,
                               trace_axes: Axes = (-1,),
                               diagonal_axes: Axes = (),
                               vmap_axes: VMapAxes = None
                               ) -> Callable[[NTTree[np.ndarray],
                                              Optional[NTTree[np.ndarray]],
                                              PyTree],
                                             NTTree[np.ndarray]]:
  """Compute NTK implicitly without instantiating full Jacobians."""

  def ntk_fn(x1: NTTree[np.ndarray],
             x2: Optional[NTTree[np.ndarray]],
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
        into `apply_fn_kwargs1` and `apply_fn_kwargs2` by the `split_kwargs`
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
    kwargs1, kwargs2 = utils.split_kwargs(apply_fn_kwargs, x1, x2)
    fx1 = eval_shape(f, params, x1, **kwargs1)
    x_axis, fx_axis, kw_axes = _canonicalize_axes(vmap_axes, x1, fx1, **kwargs1)

    keys = apply_fn_kwargs.keys()
    args1 = (kwargs1[k] for k in keys)
    args2 = (kwargs1[k] if k in kw_axes and kwargs2[k] is None else kwargs2[k]
             for k in keys)

    def get_ntk(x1, x2, *args):
      args1, args2 = args[:len(args) // 2], args[len(args) // 2 :]
      _kwargs1 = {k: v for k, v in zip(keys, args1)}
      _kwargs2 = {k: v for k, v in zip(keys, args2)}

      f1 = _get_f_params(f, x1, x_axis, fx_axis, kw_axes, **_kwargs1)
      f2 = f1 if utils.all_none(x2) else _get_f_params(
          f, x2, x_axis, fx_axis, kw_axes, **_kwargs2)

      def delta_vjp_jvp(delta):
        def delta_vjp(delta):
          return vjp(f2, params)[1](delta)
        return jvp(f1, (params,), delta_vjp(delta))[1]

      fx1, fx2 = eval_shape(f1, params), eval_shape(f2, params)
      eye = _std_basis(fx1)
      ntk = vmap(linear_transpose(delta_vjp_jvp, fx2))(eye)
      ntk = tree_map(lambda fx12: _unravel_array_into_pytree(fx1, 0, fx12), ntk)
      ntk = _diagonal(ntk, fx1)
      return ntk

    if x_axis is not None or kw_axes:
      x2 = x1 if utils.all_none(x2) else x2

      kw_in_axes = [kw_axes[k] if k in kw_axes else None for k in keys]
      in_axes1 = [x_axis, None] + kw_in_axes + [None] * len(kw_in_axes)
      in_axes2 = [None, x_axis] + [None] * len(kw_in_axes) + kw_in_axes

      get_ntk = vmap(vmap(get_ntk,
                          in_axes1,
                          fx_axis),
                     in_axes2,
                     _add(fx_axis, _ndim(fx1)))

    return _trace_and_diagonal(get_ntk(x1, x2, *args1, *args2),
                               trace_axes, diagonal_axes)

  return ntk_fn


def _empirical_direct_ntk_fn(f: ApplyFn,
                             trace_axes: Axes = (-1,),
                             diagonal_axes: Axes = (),
                             vmap_axes: VMapAxes = None
                             ) -> Callable[[NTTree[np.ndarray],
                                            Optional[NTTree[np.ndarray]],
                                            PyTree],
                                           NTTree[np.ndarray]]:
  """Compute NTK by directly instantiating Jacobians and contracting."""

  @utils.nt_tree_fn(tree_structure_argnum=0)
  def sum_and_contract(fx, j1, j2):
    ndim = fx.ndim
    size = utils.size_at(fx, trace_axes)

    _diagonal_axes = utils.canonicalize_axis(diagonal_axes, ndim)
    _trace_axes = utils.canonicalize_axis(trace_axes, ndim)

    def contract(x, y):
      param_axes = list(range(x.ndim))[ndim:]
      contract_axes = _trace_axes + param_axes
      return utils.dot_general(x, y, contract_axes, _diagonal_axes) / size

    return tree_reduce(operator.add, tree_multimap(contract, j1, j2))

  def ntk_fn(x1: NTTree[np.ndarray],
             x2: Optional[NTTree[np.ndarray]],
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
        into `apply_fn_kwargs1` and `apply_fn_kwargs2` by the `split_kwargs`
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
    kwargs1, kwargs2 = utils.split_kwargs(apply_fn_kwargs, x1, x2)
    fx1 = eval_shape(f, params, x1, **kwargs1)
    x_axis, fx_axis, kw_axes = _canonicalize_axes(vmap_axes, x1, fx1, **kwargs1)

    keys = apply_fn_kwargs.keys()
    args1, args2 = (kwargs1[k] for k in keys), (kwargs2[k] for k in keys)

    def j_fn(x, *args):
      _kwargs = {k: v for k, v in zip(keys, args)}
      fx = _get_f_params(f, x, x_axis, fx_axis, kw_axes, **_kwargs)
      jx = jacobian(fx)(params)
      return jx

    if x_axis is not None or kw_axes:
      in_axes = [x_axis] + [kw_axes[k] if k in kw_axes else None for k in keys]
      j_fn = vmap(j_fn, in_axes=in_axes, out_axes=fx_axis)

    j1 = j_fn(x1, *args1)
    j2 = j_fn(x2, *args2) if not utils.all_none(x2) else j1
    ntk = sum_and_contract(fx1, j1, j2)
    return ntk

  return ntk_fn


# INTERNAL UTILITIES


@utils.nt_tree_fn(nargs=1)
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


def _dict_of_tree_to_tree_of_dict(out_dict, get):
  # If the elements of an output dict are tuples then change the representation
  # to be a tuple of dicts instead. This occurs when the output of a network is
  # is a parallel layer.

  return tree_multimap(lambda *x: dict((g, v) for g, v in zip(get, x)),
                       *[out_dict[g] for g in get])


def _get_f_params(f, x, x_axis, fx_axis, kw_axes, **apply_fn_kwargs):
  x = _expand_dims(x, x_axis)

  apply_fn_kwargs = {
      k: _expand_dims(v, kw_axes[k]) if k in kw_axes else v
      for k, v in apply_fn_kwargs.items()
  }

  def _f(p):
    fx = f(p, x, **apply_fn_kwargs)
    fx = utils.get_masked_array(fx)
    # TODO(romann): normalize properly if output is masked.

    get_masked = utils.nt_tree_fn()(lambda o: o.masked_value)
    fx = get_masked(fx)
    return _squeeze(fx, fx_axis)

  return _f


def _expand_dims(x, axis):
  if axis is None or x is None:
    return x
  return tree_multimap(np.expand_dims, x, axis)


def _add(x, y):
  if x is None or y is None:
    return None
  return tree_multimap(operator.add, x, y)


def _sub(x, y):
  return tree_multimap(operator.sub, x, y)


def _div(x, y):
  return tree_map(lambda x: x / y, x)


def _squeeze(x, axis, take=False):
  if axis is None:
    return x
  if take:
    return tree_multimap(lambda x, axis: np.take(x, 0, axis), x, axis)
  return tree_multimap(np.squeeze, x, axis)


@utils.nt_tree_fn()
def _ndim(x):
  return x.ndim


def _mod(x, y):
  return tree_multimap(operator.mod, x, y)


def _diagonal(ntk, fx):
  ntk_flat, _ = tree_flatten(ntk)
  fx_flat, fx_tree = tree_flatten(fx)
  n = len(fx_flat)
  diag = [ntk_flat[i * (n + 1)] for i in range(n)]
  return tree_unflatten(fx_tree, diag)


def _canonicalize_axes(vmap_axes: Optional[VMapAxes],
                       x: NTTree[np.ndarray],
                       fx: NTTree[np.ndarray],
                       **kwargs) -> VMapAxes:
  if isinstance(vmap_axes, tuple) and len(vmap_axes) == 3:
    x_axis, fx_axis, kw_axes = vmap_axes
  else:
    x_axis, fx_axis, kw_axes = vmap_axes, vmap_axes, {}

  x_axis = _mod(x_axis, _ndim(x)) if x_axis is not None else None
  fx_axis = _mod(fx_axis, _ndim(fx)) if fx_axis is not None else None
  kw_axes = _mod(kw_axes, {k: _ndim(kwargs[k]) for k in kw_axes})
  return x_axis, fx_axis, kw_axes
