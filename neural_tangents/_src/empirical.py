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
signatures (not only those from :obj:`~neural_tangents.stax`).

NNGP and NTK are computed using :obj:`~neural_tangents.empirical_nngp_fn`,
:obj:`~neural_tangents.empirical_ntk_fn`, or
:obj:`~neural_tangents.empirical_kernel_fn` (for both). The kernels have a very
specific output shape convention that may be unexpected. Further, NTK has
multiple implementations that may perform differently depending on the task.
Please read individual functions' docstrings.

For details, please see "`Fast Finite Width Neural Tangent Kernel
<https://arxiv.org/abs/2206.08720>`_".

Example:
  >>> from jax import random
  >>> import neural_tangents as nt
  >>> from neural_tangents import stax
  >>> #
  >>> key1, key2, key3 = random.split(random.PRNGKey(1), 3)
  >>> x_train = random.normal(key1, (20, 32, 32, 3))
  >>> y_train = random.uniform(key1, (20, 10))
  >>> x_test = random.normal(key2, (5, 32, 32, 3))
  >>> #
  >>> # A narrow CNN.
  >>> init_fn, f, _ = stax.serial(
  >>>     stax.Conv(32, (3, 3)),
  >>>     stax.Relu(),
  >>>     stax.Conv(32, (3, 3)),
  >>>     stax.Relu(),
  >>>     stax.Conv(32, (3, 3)),
  >>>     stax.Flatten(),
  >>>     stax.Dense(10)
  >>> )
  >>> #
  >>> _, params = init_fn(key3, x_train.shape)
  >>> #
  >>> # Default setting: reducing over logits; pass `vmap_axes=0` because the
  >>> # network is iid along the batch axis, no BatchNorm. Use default
  >>> # `implementation=nt.NtkImplementation.JACOBIAN_CONTRACTION` (`1`).
  >>> kernel_fn = nt.empirical_kernel_fn(
  >>>     f, trace_axes=(-1,), vmap_axes=0,
  >>>     implementation=nt.NtkImplementation.JACOBIAN_CONTRACTION)
  >>> #
  >>> # (5, 20) np.ndarray test-train NNGP/NTK
  >>> nngp_test_train = kernel_fn(x_test, x_train, 'nngp', params)
  >>> ntk_test_train = kernel_fn(x_test, x_train, 'ntk', params)
  >>> #
  >>> # Full kernel: not reducing over logits. Use structured derivatives
  >>> # `implementation=nt.NtkImplementation.STRUCTURED_DERIVATIVES` (`3`) for
  >>> # typically faster computation and lower memory cost.
  >>> kernel_fn = nt.empirical_kernel_fn(
  >>>     f, trace_axes=(), vmap_axes=0,
  >>>     implementation=nt.NtkImplementation.STRUCTURED_DERIVATIVES)
  >>> #
  >>> # (5, 20, 10, 10) np.ndarray test-train NNGP/NTK namedtuple.
  >>> k_test_train = kernel_fn(x_test, x_train, None, params)
  >>> #
  >>> # A wide FCN with lots of parameters and many (`100`) outputs.
  >>> init_fn, f, _ = stax.serial(
  >>>     stax.Flatten(),
  >>>     stax.Dense(1024),
  >>>     stax.Relu(),
  >>>     stax.Dense(1024),
  >>>     stax.Relu(),
  >>>     stax.Dense(100)
  >>> )
  >>> #
  >>> _, params = init_fn(key3, x_train.shape)
  >>> #
  >>> # Use ntk-vector products
  >>> # (`implementation=nt.NtkImplementation.NTK_VECTOR_PRODUCTS`) since the
  >>> # network has many parameters relative to the cost of forward pass,
  >>> # large outputs.
  >>> ntk_fn = nt.empirical_ntk_fn(
  >>>     f, vmap_axes=0,
  >>>     implementation=nt.NtkImplementation.NTK_VECTOR_PRODUCTS)
  >>> #
  >>> # (5, 5) np.ndarray test-test NTK
  >>> ntk_test_test = ntk_fn(x_test, None, params)
  >>> #
  >>> # Compute only output variances:
  >>> nngp_fn = nt.empirical_nngp_fn(f, diagonal_axes=(0,))
  >>> #
  >>> # (20,) np.ndarray train-train diagonal NNGP
  >>> nngp_train_train_diag = nngp_fn(x_train, None, params)
"""

import enum
import functools
import operator
from typing import Callable, Dict, KeysView, List, Optional, Set, Tuple, TypeVar, Union, Iterable
import warnings

import jax
from jax import core, lax
from jax import eval_shape, jacobian, jvp, vjp, vmap
from jax import linear_transpose
from jax import linear_util as lu
from jax.core import Jaxpr, JaxprEqn, Literal, ShapedArray, Value, Var
from jax.interpreters import ad, xla
from jax.interpreters.ad import UndefinedPrimal, Zero
import jax.numpy as np
from jax.tree_util import tree_flatten, tree_map, tree_reduce, tree_structure, tree_transpose, tree_unflatten
from jax.util import safe_map as map, safe_zip as zip
import numpy as onp
from .utils import rules
from .utils import utils
from .utils.typing import ApplyFn, Axes, EmpiricalGetKernelFn, EmpiricalKernelFn, PyTree, VMapAxes, VMapAxisTriple


# LINEARIZATION AND TAYLOR EXPANSION


def linearize(f: ApplyFn, params: PyTree) -> ApplyFn:
  """Returns a function `f_lin`, the first order taylor approximation to `f`.

  Example:
    >>> # Compute the MSE of the first order Taylor series of a function.
    >>> f_lin = linearize(f, params)
    >>> mse = np.mean((f(new_params, x) - f_lin(new_params, x)) ** 2)

  Args:
    f:
      A function that we would like to linearize. It should have the signature
      `f(params, *args, **kwargs)` where `params` is a `PyTree` and `f` should
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


def taylor_expand(f: ApplyFn, params: PyTree, degree: int) -> ApplyFn:
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


# NNGP


def empirical_nngp_fn(
    f: ApplyFn,
    trace_axes: Axes = (-1,),
    diagonal_axes: Axes = ()
) -> EmpiricalKernelFn:
  """Returns a function to draw a single sample the NNGP of a given network `f`.

  The Neural Network Gaussian Process (NNGP) kernel is defined as
  :math:`f(X_1) f(X_2)^T`, i.e. the outer product of the function outputs.

  .. warning::
    Resulting kernel shape is *nearly* `zip(f(x1).shape, f(x2).shape)`
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
      the function whose NNGP we are computing. It should have the signature
      `f(params, x, **kwargs)` where `params` is a `PyTree`, `x` is a `PyTree`,
      and `f` should also return a `PyTree`.

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
  def nngp_fn(x1: PyTree,
              x2: Optional[PyTree],
              params: PyTree,
              **apply_fn_kwargs) -> PyTree:
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
      return f(params, x, **kwargs)

    kwargs1, kwargs2 = utils.split_kwargs(apply_fn_kwargs, x1, x2)

    out1 = output(x1, **kwargs1)
    out2 = output(x2, **kwargs2) if not utils.all_none(x2) else out1

    def contract(out1: np.ndarray, out2: np.ndarray) -> np.ndarray:
      dot = _dot_general(out1, out2, trace_axes, diagonal_axes)
      return dot / utils.size_at(out1, trace_axes)

    return tree_map(contract, out1, out2)

  return nngp_fn


# NTK


class NtkImplementation(enum.IntEnum):
  """Implementation method of the underlying finite width NTK computation.

  Below is a very brief summary of each method. For details, please see "`Fast
  Finite Width Neural Tangent Kernel <https://arxiv.org/abs/2206.08720>`_".

  Attributes:
    AUTO:
      (or `0`) evaluates FLOPs of all other methods at compilation time,
      and selects the fastest method. However, at the time it only works
      correctly on TPUs, and on CPU/GPU can return wrong results, which is why
      it is not the default. TODO(romann): revisit based on http://b/202218145.

    JACOBIAN_CONTRACTION:
      (or `1`) computes the NTK as the outer product of two Jacobians, each
      computed using reverse-mode Autodiff (vector-Jacobian products, VJPs).
      When JITted, the contraction is performed in a layerwise fashion, so that
      entire Jacobians aren't necessarily instantiated in memory at once, and
      the memory usage of the method can be lower than memory needed to
      instantiate the two Jacobians. This method is best suited for networks
      with small outputs (such as scalar outputs for binary classification or
      regression, as opposed to 1000 ImageNet classes), and an expensive
      forward pass relative to the number of parameters (such as CNNs, where
      forward pass reuses a small filter bank many times). It is also the the
      most reliable method, since its implementation is simplest, and
      reverse-mode Autodiff is most commonly used and well tested elsewhere.
      For this reason it is set as the default.

    NTK_VECTOR_PRODUCTS:
      (or `2`) computes the NTK as a sequence of NTK-vector products, similarly
      to how a Jacobian is computed as a sequence of Jacobian-vector products
      (JVPs) or vector-Jacobian products (VJPs). This amounts to using both
      forward (JVPs) and reverse (VJPs) mode Autodiff, and allows to eliminate
      the Jacobian contraction at the expense of additional forward passes.
      Therefore this method is recommended for networks with a cheap forward
      pass relative to the number of parameters (e.g. fully-connected networks,
      where each parameter matrix is used only once in the forward pass), and
      networks with large outputs (e.g. 1000 ImageNet classes). Memory
      requirements of this method are same as :attr:`JACOBIAN_CONTRACTION`
      (`1`). Due to reliance of forward-mode Autodiff, this method is slightly
      more prone to JAX and XLA bugs than :attr:`JACOBIAN_CONTRACTION` (`1`),
      but overall is quite simple and reliable.

    STRUCTURED_DERIVATIVES:
      (or `3`) uses a custom JAX interpreter to compute the NTK more
      efficiently than other methods. It traverses the computational graph of a
      function in the same order as during reverse-mode Autodiff, but instead
      of computing VJPs, it directly computes MJJMPs,
      "matrix-Jacobian-Jacobian-matrix" products, which arise in the
      computation of an NTK. Each MJJMP computation relies on the structure in
      the Jacobians, hence the name. This method can be dramatically faster
      (up to several orders of magnitude) then other methods on fully-connected
      networks, and is usually faster or equivalent on CNNs, Transformers, and
      other architectures, but exact speedup (e.g. from no speedup to 10X)
      depends on each specific setting. It can also use less memory than other
      methods. In our experience it consistently outperforms other methods in
      most settings. However, its implementation is significantly more complex
      (hence bug-prone), and it doesn't yet support functions using more exotic
      JAX primitives (e.g. :obj:`jax.checkpoint`, parallel collectives such as
      :obj:`jax.lax.psum`, compiled loops like :obj:`jax.lax.scan`, etc.), which
      is why it is highly-recommended to try, but not set as the default yet.
  """
  AUTO = 0
  JACOBIAN_CONTRACTION = 1
  NTK_VECTOR_PRODUCTS = 2
  STRUCTURED_DERIVATIVES = 3


DEFAULT_NTK_IMPLEMENTATION = NtkImplementation.JACOBIAN_CONTRACTION
"""Default user-facing empirical NTK implementation.

We default to `JACOBIAN_CONTRACTION` since it's the most straightforward and
reliable method, virtually guaranteed to compute the correct result.
"""


_DEFAULT_TESTING_NTK_IMPLEMENTATION = NtkImplementation.STRUCTURED_DERIVATIVES
"""Default empirical NTK implementation used in `tests`.

We default to `STRUCTURED_DERIVATIVES` since it is the fastest but also most
complex method, hence benefiting from additional testing against infinite-width
results.
"""


_DEFAULT_NTK_J_RULES: bool = True
"""Says whether to use custom Jacobian rules in `STRUCTURED_DERIVATIVES` (`3`).

Useful for debugging and testing. Theoretically should be set to `True`, but if
some Jacobian rule is implemented suboptimally, trying out `False` could improve
performance.
"""


_DEFAULT_NTK_S_RULES: bool = True
"""Says whether to use structure rules in `STRUCTURED_DERIVATIVES` (`3`).

Useful for debugging and testing. In practice should be set to `True`, and
setting it to `False` can lead to dramatic deterioration of performance.
"""


_DEFAULT_NTK_FWD: Optional[bool] = None
"""Says whether to use forward mode in `STRUCTURED_DERIVATIVES` (`3`) Jacobians.

Useful for debugging and testing, but for best performance should be set to
`None`, i.e. to selecting forward or reverse mode AD automatically based on
input/output sizes.
"""


def _empirical_auto_ntk_fn(**kwargs) -> EmpiricalGetKernelFn:
  """Compute NTK by automatically selecting the best implementation.

  Returns wrong FLOPS on CPU and GPU when JITting.

  TODO(romann): revisit based on http://b/202218145.
  """
  cache = {}

  def ntk_fn(
      x1: PyTree,
      x2: Optional[PyTree],
      params: PyTree,
      **apply_fn_kwargs
  ) -> np.ndarray:
    """Computes a single sample of the automatic empirical NTK.

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
    shapes = tree_map(np.shape, (x1, x2, params, apply_fn_kwargs))
    shapes = _to_tuple_tree(shapes)

    if shapes not in cache:
      best_ntk_fn = None
      best_flops = onp.inf
      for implementation in NtkImplementation:
        if implementation != NtkImplementation.AUTO:
          ntk_fn = empirical_ntk_fn(**kwargs, implementation=implementation)
          flops = _get_flops(ntk_fn, True, x1, x2, params, **apply_fn_kwargs)
          print(f'impl={implementation}, flops={flops}')
          if flops < best_flops:
            best_flops = flops
            best_ntk_fn = ntk_fn

      if best_ntk_fn is None:
        raise ValueError('This should not happen.')
      cache[shapes] = best_ntk_fn

    return cache[shapes](x1, x2, params, **apply_fn_kwargs)

  return ntk_fn


def _jacobian_contraction_ntk_fn(
    f: ApplyFn,
    trace_axes: Axes,
    diagonal_axes: Axes,
    vmap_axes: VMapAxes,
    **kwargs
) -> EmpiricalKernelFn:
  """Compute NTK by directly instantiating Jacobians and contracting."""

  def sum_and_contract(fx, j1, j2):
    ndim = fx.ndim
    size = utils.size_at(fx, trace_axes)

    _diagonal_axes = utils.canonicalize_axis(diagonal_axes, ndim)
    _trace_axes = utils.canonicalize_axis(trace_axes, ndim)

    def contract(x, y):
      param_axes = list(range(x.ndim))[ndim:]
      contract_axes = _trace_axes + param_axes
      return _dot_general(x, y, contract_axes, _diagonal_axes) / size

    return tree_reduce(operator.add, tree_map(contract, j1, j2))

  def ntk_fn(
      x1: PyTree,
      x2: Optional[PyTree],
      params: PyTree,
      **apply_fn_kwargs
  ) -> np.ndarray:
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
    args1, args2, fx1, fx2, fx_axis, keys, kw_axes, x_axis = _get_args(
        f, apply_fn_kwargs, params, vmap_axes, x1, x2)

    def j_fn(x, *args):
      _kwargs = {k: v for k, v in zip(keys, args)}
      fx = _get_f_params(f, x, x_axis, fx_axis, kw_axes, **_kwargs)
      jx = jacobian(fx)(params)
      return jx

    if not utils.all_none(x_axis) or not utils.all_none(kw_axes):
      in_axes = [x_axis] + [kw_axes[k] if k in kw_axes else None for k in keys]
      j_fn = vmap(j_fn, in_axes=in_axes, out_axes=fx_axis)

    j1 = j_fn(x1, *args1)
    j2 = j_fn(x2, *args2) if not utils.all_none(x2) else j1
    ntk = tree_map(sum_and_contract, fx1, j1, j2)
    return ntk

  return ntk_fn


def _ntk_vector_products_ntk_fn(
    f: ApplyFn,
    trace_axes: Axes,
    diagonal_axes: Axes,
    vmap_axes: VMapAxes,
    **kwargs
) -> EmpiricalKernelFn:
  """Compute NTK via NTK-vector products."""

  def ntk_fn(
      x1: PyTree,
      x2: Optional[PyTree],
      params: PyTree,
      **apply_fn_kwargs
  ) -> np.ndarray:
    """Computes a single sample of the empirical NTK with NTK-vector products.

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
    args1, args2, fx1, fx2, fx_axis, keys, kw_axes, x_axis = _get_args(
        f, apply_fn_kwargs, params, vmap_axes, x1, x2)

    def get_ntk(x1, x2, *args):
      f1, f2 = _get_f1_f2(f, keys, x_axis, fx_axis, kw_axes, args, x1, x2)

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

    if not utils.all_none(x_axis) or not utils.all_none(kw_axes):
      x2 = x1 if utils.all_none(x2) else x2

      kw_in_axes = [kw_axes[k] if k in kw_axes else None for k in keys]
      in_axes1 = [x_axis, None] + kw_in_axes + [None] * len(kw_in_axes)
      in_axes2 = [None, x_axis] + [None] * len(kw_in_axes) + kw_in_axes

      get_ntk = vmap(vmap(get_ntk,
                          in_axes1,
                          fx_axis),
                     in_axes2,
                     _add(fx_axis, _ndim(fx1)))

    ntk = get_ntk(x1, x2, *args1, *args2)
    ntk = tree_map(lambda x: _trace_and_diagonal(x, trace_axes, diagonal_axes),
                   ntk)
    return ntk

  return ntk_fn


def _structured_derivatives_ntk_fn(
    f: ApplyFn,
    trace_axes: Axes,
    diagonal_axes: Axes,
    vmap_axes: VMapAxes,
    _j_rules: bool,
    _s_rules: bool,
    _fwd: Optional[bool]
) -> EmpiricalKernelFn:
  """Compute NTK by using structured derivatives."""

  def sum_and_contract(
      fx1: np.ndarray,
      fx2: np.ndarray,
      fx_axis,
      df_dys_1: List[Union[np.ndarray, Zero]],
      df_dys_2: List[Union[np.ndarray, Zero]],
      dy_dws_1: List[Tuple[np.ndarray, rules.Structure]],
      dy_dws_2: List[Tuple[np.ndarray, rules.Structure]],
      dtype: np.dtype
  ):
    ndim = fx1.ndim
    size = utils.size_at(fx1, trace_axes)

    _diagonal_axes = utils.canonicalize_axis(diagonal_axes, ndim)
    _trace_axes = utils.canonicalize_axis(trace_axes, ndim)

    def contract(df_dys_1, df_dys_2, dy_dws_1, dy_dws_2):
      ntk = np.zeros((), dtype=dtype)

      for df_dy_1, dy_dw_1_ in zip(df_dys_1, dy_dws_1):
        for df_dy_2, dy_dw_2_ in zip(df_dys_2, dy_dws_2):
          if isinstance(dy_dw_1_, Zero) or isinstance(dy_dw_2_, Zero):
            continue

          dy_dw_1: np.ndarray
          s1: rules.Structure
          dy_dw_1, s1 = dy_dw_1_

          dy_dw_2: np.ndarray
          s2: rules.Structure
          dy_dw_2, s2 = dy_dw_2_

          df_dy_dims_1, df_dy_dims_2, out_dims = _get_dims(df_dy_1,
                                                           df_dy_2,
                                                           ndim,
                                                           _trace_axes,
                                                           _diagonal_axes)

          if len(s1.out_trace) != len(s2.out_trace):
            raise NotImplementedError('Different number of trace_axes 1/2.')

          for i, (id_1, id_2) in enumerate(zip(s1.out_trace, s2.out_trace)):
            axis_id = df_dy_1.ndim + df_dy_2.ndim + i
            y_axis_1 = id_1 % (df_dy_1.ndim - ndim)
            y_axis_2 = id_2 % (df_dy_2.ndim - ndim)
            df_dy_dims_1[ndim + y_axis_1] = axis_id
            df_dy_dims_2[ndim + y_axis_2] = axis_id

          dy_dw_dims_1 = list(range(-dy_dw_1.ndim, 0))
          dy_dw_dims_2 = list(range(-dy_dw_2.ndim, 0))

          if fx_axis is not None:
            df_dy_1 = np.moveaxis(df_dy_1, 0, fx_axis)
            df_dy_2 = np.moveaxis(df_dy_2, 0, fx_axis)

            dy_dw_dims_1[0] = df_dy_dims_1[fx_axis]
            dy_dw_dims_2[0] = df_dy_dims_2[fx_axis]
            ix_1, ix_2 = 1, 1

          else:
            ix_1, ix_2 = 0, 0

          if len(s1.out_diagonal) != len(s2.out_diagonal):
            raise NotImplementedError('Different number of diagonal_axes 1/2.')

          for i, (id_1, id_2) in enumerate(zip(s1.out_diagonal,
                                               s2.out_diagonal)):
            # TODO(romann): compute based on array dimensions.
            axis_shift = -100_000  # Huge axis shift to ensure unique axis ids.

            axis_id = (-axis_shift -df_dy_1.ndim - df_dy_2.ndim - dy_dw_1.ndim
                       - dy_dw_2.ndim - i)

            df_dy_dims_1[ndim + id_1] = axis_id
            dy_dw_dims_1[ix_1 + id_1] = axis_id

            df_dy_dims_2[ndim + id_2] = axis_id
            dy_dw_dims_2[ix_2 + id_2] = axis_id

          for i in range(ndim, df_dy_1.ndim):
            if i - ndim not in (s1.out_trace +
                                s1.out_diagonal +
                                s1.out_broadcast):
              dy_dw_dims_1[ix_1] = df_dy_dims_1[i]
            ix_1 += 1

          for i in range(ndim, df_dy_2.ndim):
            if i - ndim not in (s2.out_trace +
                                s2.out_diagonal +
                                s2.out_broadcast):
              dy_dw_dims_2[ix_2] = df_dy_dims_2[i]
            ix_2 += 1

          _check_einsum_no_broadcast(
              arrays=[df_dy_1, dy_dw_1, dy_dw_2, df_dy_2],
              dims=[df_dy_dims_1, dy_dw_dims_1, dy_dw_dims_2, df_dy_dims_2]
          )

          ntk_l = np.einsum(
              df_dy_1, df_dy_dims_1,
              dy_dw_1, dy_dw_dims_1,
              dy_dw_2, dy_dw_dims_2,
              df_dy_2, df_dy_dims_2,
              out_dims
          )
          ntk += ntk_l

      return ntk

    ntk = tree_reduce(
        operator.add,
        tree_map(
            contract,
            df_dys_1, df_dys_2, dy_dws_1, dy_dws_2,
            is_leaf=
            lambda x: (x == [] or
                       (isinstance(x, list) and isinstance(x[0], np.ndarray)))),
        np.zeros((), dtype)
    )
    ntk /= size
    ntk_shape = _ntk_shape(fx1.shape, fx2.shape, trace_axes, diagonal_axes)
    ntk = np.broadcast_to(ntk, ntk_shape)  # if ntk is 0.
    return ntk

  def ntk_fn(
      x1: PyTree,
      x2: Optional[PyTree],
      params: PyTree,
      **apply_fn_kwargs
  ) -> np.ndarray:
    """Computes a single sample of the structured derivatives NTK.

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
    args1, args2, fx1, fx2, fx_axis, keys, kw_axes, x_axis = _get_args(
        f, apply_fn_kwargs, params, vmap_axes, x1, x2)

    def j_fn(x, *args):
      _kwargs = {k: v for k, v in zip(keys, args)}
      fx = _get_f_params(f, x, x_axis, fx_axis, kw_axes, **_kwargs)
      df_dys, dy_dws = _get_df_dys_and_dy_dws(fn=fx, params=params,
                                              _j_rules=_j_rules,
                                              _s_rules=_s_rules, _fwd=_fwd)
      return df_dys, dy_dws

    if not utils.all_none(x_axis) or not utils.all_none(kw_axes):
      in_axes = [x_axis] + [kw_axes[k] if k in kw_axes else None for k in keys]
      j_fn = vmap(j_fn, in_axes=in_axes, out_axes=0)

    df_dys_1, dy_dws_1 = j_fn(x1, *args1)
    df_dys_2, dy_dws_2 = j_fn(x2, *args2) if not utils.all_none(x2) else (
        df_dys_1, dy_dws_1)

    fx_axis, dtype = _get_fx_axis_and_dtype(fx1, fx_axis, params)
    ntk = tree_map(
        functools.partial(
            sum_and_contract,
            dy_dws_1=dy_dws_1,
            dy_dws_2=dy_dws_2,
            dtype=dtype),
        fx1,
        fx2,
        fx_axis,
        df_dys_1,
        df_dys_2,
    )

    return ntk

  return ntk_fn


_implementation_to_ntk_fn = {
    NtkImplementation.AUTO: _empirical_auto_ntk_fn,
    NtkImplementation.JACOBIAN_CONTRACTION: _jacobian_contraction_ntk_fn,
    NtkImplementation.NTK_VECTOR_PRODUCTS: _ntk_vector_products_ntk_fn,
    NtkImplementation.STRUCTURED_DERIVATIVES: _structured_derivatives_ntk_fn,
}


def empirical_ntk_fn(
    f: ApplyFn,
    trace_axes: Axes = (-1,),
    diagonal_axes: Axes = (),
    vmap_axes: VMapAxes = None,
    implementation: Union[NtkImplementation, int] = DEFAULT_NTK_IMPLEMENTATION,
    _j_rules: bool = _DEFAULT_NTK_J_RULES,
    _s_rules: bool = _DEFAULT_NTK_S_RULES,
    _fwd: Optional[bool] = _DEFAULT_NTK_FWD,
) -> EmpiricalKernelFn:
  r"""Returns a function to draw a single sample the NTK of a given network `f`.

  The Neural Tangent Kernel is defined as :math:`J(X_1) J(X_2)^T` where
  :math:`J` is the Jacobian :math:`df/dparams` of shape
  `full_output_shape + params.shape`.

  For best performance:
  1) pass `x2=None` if `x1 == x2;
  2) prefer square batches (i.e `x1.shape == x2.shape`);
  3) make sure to set `vmap_axes` correctly.
  4) try different `implementation` values.

  .. warning::
    Resulting kernel shape is *nearly* `zip(f(x1).shape, f(x2).shape)`
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
      the function whose NTK we are computing. It should have the signature
      `f(params, x, **kwargs)` where `params` is a `PyTree`, `x` is a `PyTree`,
      and `f` should also return a `PyTree`.

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
      An :class:`NtkImplementation` value (or an :class:`int`  `0`, `1`, `2`,
      or `3`). See the :class:`NtkImplementation` docstring for details.

    _j_rules:
      Internal debugging parameter, applicable only when
      `implementation` is :attr:`~NtkImplementation.STRUCTURED_DERIVATIVES`
      (`3`) or :attr:`~NtkImplementation.AUTO` (`0`). Set to `True` to allow
      custom Jacobian rules for intermediary primitive `dy/dw` computations for
      MJJMPs (matrix-Jacobian-Jacobian-matrix products). Set to `False` to use
      JVPs or VJPs, via JAX's :obj:`jax.jacfwd` or :obj:`jax.jacrev`. Custom
      Jacobian rules (`True`) are expected to be not worse, and sometimes better
      than automated alternatives, but in case of a suboptimal implementation
      setting it to `False` could improve performance.

    _s_rules:
      Internal debugging parameter, applicable only when
      `implementation` is :attr:`~NtkImplementation.STRUCTURED_DERIVATIVES`
      (`3`) or :attr:`~NtkImplementation.AUTO` (`0`). Set to `True` to allow
      efficient MJJMp rules for structured `dy/dw` primitive Jacobians. In
      practice should be set to `True`, and setting it to `False` can lead to
      dramatic deterioration of performance.

    _fwd:
      Internal debugging parameter, applicable only when
      `implementation` is :attr:`~NtkImplementation.STRUCTURED_DERIVATIVES`
      (`3`) or :attr:`~NtkImplementation.AUTO` (`0`). Set to `True` to allow
      :obj:`jax.jvp` in intermediary primitive Jacobian `dy/dw` computations,
      `False` to always use :obj:`jax.vjp`. `None` to decide automatically
      based on input/output sizes. Applicable when `_j_rules=False`, or when a
      primitive does not have a Jacobian rule. Should be set to `None` for best
      performance.

  Returns:
    A function `ntk_fn` that computes the empirical ntk.
  """
  return _implementation_to_ntk_fn[implementation](
      f=f,
      trace_axes=trace_axes,
      diagonal_axes=diagonal_axes,
      vmap_axes=vmap_axes,
      _j_rules=_j_rules,
      _s_rules=_s_rules,
      _fwd=_fwd
  )


# JOINT NNGP/NTK KERNEL FUNCTION


def empirical_kernel_fn(
    f: ApplyFn,
    trace_axes: Axes = (-1,),
    diagonal_axes: Axes = (),
    vmap_axes: VMapAxes = None,
    implementation: Union[NtkImplementation, int] = DEFAULT_NTK_IMPLEMENTATION,
    _j_rules: bool = _DEFAULT_NTK_J_RULES,
    _s_rules: bool = _DEFAULT_NTK_S_RULES,
    _fwd: Optional[bool] = _DEFAULT_NTK_FWD,
) -> EmpiricalGetKernelFn:
  r"""Returns a function that computes single draws from NNGP and NT kernels.

  .. warning::
    Resulting kernel shape is *nearly* `zip(f(x1).shape, f(x2).shape)`
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
      the function whose kernel(s) (NNGP and/or NTK) we are computing. It
      should have the signature `f(params, x, **kwargs)` where `params` is a
      `PyTree`, `x` is a  `PyTree`, and `f` should also return a `PyTree`.

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
      Applicable only to NTK, an :class:`NtkImplementation` value (or an
      :class:`int`  `0`, `1`, `2`, or `3`). See the :class:`NtkImplementation`
      docstring for details.

    _j_rules:
      Internal debugging parameter, applicable only to NTK when
      `implementation` is :attr:`~NtkImplementation.STRUCTURED_DERIVATIVES`
      (`3`) or :attr:`~NtkImplementation.AUTO` (`0`). Set to `True` to allow
      custom Jacobian rules for intermediary primitive `dy/dw` computations for
      MJJMPs (matrix-Jacobian-Jacobian-matrix products). Set to `False` to use
      JVPs or VJPs, via JAX's :obj:`jax.jacfwd` or :obj:`jax.jacrev`. Custom
      Jacobian rules (`True`) are expected to be not worse, and sometimes better
      than automated alternatives, but in case of a suboptimal implementation
      setting it to `False` could improve performance.

    _s_rules:
      Internal debugging parameter, applicable only to NTK when
      `implementation` is :attr:`~NtkImplementation.STRUCTURED_DERIVATIVES`
      (`3`) or :attr:`~NtkImplementation.AUTO` (`0`). Set to `True` to allow
      efficient MJJMp rules for structured `dy/dw` primitive Jacobians. In
      practice should be set to `True`, and setting it to `False` can lead to
      dramatic deterioration of performance.

    _fwd:
      Internal debugging parameter, applicable only to NTK when
      `implementation` is :attr:`~NtkImplementation.STRUCTURED_DERIVATIVES`
      (`3`) or :attr:`~NtkImplementation.AUTO` (`0`). Set to `True` to allow
      :obj:`jax.jvp` in intermediary primitive Jacobian `dy/dw` computations,
      `False` to always use :obj:`jax.vjp`. `None` to decide automatically
      based on input/output sizes. Applicable when `_j_rules=False`, or when a
      primitive does not have a Jacobian rule. Should be set to `None` for best
      performance.

  Returns:
    A function to draw a single sample the NNGP and NTK empirical kernels of a
    given network `f`.
  """
  kwargs = dict(
      f=f,
      trace_axes=trace_axes,
      diagonal_axes=diagonal_axes
  )

  ntk_kwargs = dict(
      vmap_axes=vmap_axes,
      implementation=implementation,
      _j_rules=_j_rules,
      _s_rules=_s_rules,
      _fwd=_fwd,
  )

  kernel_fns = {
      'nngp': empirical_nngp_fn(**kwargs),
      'ntk': empirical_ntk_fn(**kwargs, **ntk_kwargs)
  }

  @utils.get_namedtuple('EmpiricalKernel')
  def kernel_fn(
      x1: PyTree,
      x2: Optional[PyTree],
      get: Union[None, str, Tuple[str, ...]],
      params: PyTree,
      **apply_fn_kwargs
  ) -> PyTree:
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


# NTK-VECTOR PRODUCT FUNCTION


def empirical_ntk_vp_fn(
    f: ApplyFn,
    x1: PyTree,
    x2: Optional[PyTree],
    params: PyTree,
    **apply_fn_kwargs
) -> Callable[[PyTree], PyTree]:
  """Returns an NTK-vector product function.

  The function computes NTK-vector product without instantiating the NTK, and
  has the runtime equivalent to `(N1 + N2)` forward passes through `f`, and
  memory equivalent to evaluating a vector-Jacobian product of `f`.

  For details, please see section L of "`Fast Finite Width Neural Tangent Kernel
  <https://arxiv.org/abs/2206.08720>`_".

  Example:
    >>> from jax import random
    >>> import neural_tangents as nt
    >>> from neural_tangents import stax
    >>> #
    >>> k1, k2, k3, k4 = random.split(random.PRNGKey(1), 4)
    >>> x1 = random.normal(k1, (20, 32, 32, 3))
    >>> x2 = random.normal(k2, (10, 32, 32, 3))
    >>> #
    >>> # Define a forward-pass function `f`.
    >>> init_fn, f, _ = stax.serial(
    >>>     stax.Conv(32, (3, 3)),
    >>>     stax.Relu(),
    >>>     stax.Conv(32, (3, 3)),
    >>>     stax.Relu(),
    >>>     stax.Conv(32, (3, 3)),
    >>>     stax.Flatten(),
    >>>     stax.Dense(10)
    >>> )
    >>> #
    >>> # Initialize parameters.
    >>> _, params = init_fn(k3, x1.shape)
    >>> #
    >>> # NTK-vp function. Can/should be JITted.
    >>> ntk_vp_fn = empirical_ntk_vp_fn(f, x1, x2, params)
    >>> #
    >>> # Cotangent vector
    >>> cotangents = random.normal(k4, f(params, x2).shape)
    >>> #
    >>> # NTK-vp output
    >>> ntk_vp = ntk_vp_fn(cotangents)
    >>> #
    >>> # Output has same shape as `f(params, x1)`.
    >>> assert ntk_vp.shape == f(params, x1).shape

  Args:
    f:
      forward-pass function of signature `f(params, x)`.

    x1:
      first batch of inputs.

    x2:
      second batch of inputs. `x2=None` means `x2=x1`.

    params:
      A `PyTree` of parameters about which we would like to compute the neural
      tangent kernel.

    **apply_fn_kwargs:
      keyword arguments passed to `f`. `apply_fn_kwargs` will be split into
      `apply_fn_kwargs1` and `apply_fn_kwargs2` by the `split_kwargs` function
      which will be passed to `f`. In particular, the rng key in
      `apply_fn_kwargs`, will be split into two different (if `x1!=x2`) or same
      (if `x1==x2`) rng keys. See the `_read_key` function for more details.

  Returns:
    An NTK-vector product function accepting a `PyTree` of cotangents of shape
    and structure of `f(params, x2)`, and returning the NTK-vector product of
    shape and structure of `f(params, x1)`.
  """
  args1, args2, fx1, fx2, fx_axis, keys, kw_axes, x_axis = _get_args(
      f, apply_fn_kwargs, params, None, x1, x2)

  f1, f2 = _get_f1_f2(f, keys, x_axis, fx_axis, kw_axes, args1 + args2, x1, x2)

  def ntk_vp_fn(cotangents: PyTree) -> PyTree:
    """Computes a single empirical NTK-vector product.

    Args:
      cotangents:
        a `PyTree` of cotangents. Must have the same shape and tree structure
        as `f(params, x2)`.

    Returns:
      A single NTK-vector product of shape and tree structure of
      `f(params, x1)`.
    """
    vjp_out = vjp(f2, params)[1](cotangents)
    jvp_out = jvp(f1, (params,), vjp_out)[1]
    return jvp_out

  return ntk_vp_fn


# INTERNAL UTILITIES


def _trace_and_diagonal(
    ntk: np.ndarray,
    trace_axes: Axes,
    diagonal_axes: Axes
) -> np.ndarray:
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
    raise ValueError('Expected an even-dimensional kernel.')

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
  res_diagonal_axes = _get_res_batch_dims(trace_axes, diagonal_axes)
  ntk = np.moveaxis(ntk, range(-n_diag, 0), res_diagonal_axes)
  return ntk / contract_size


def _dict_of_tree_to_tree_of_dict(
    out_dict: Dict[str, PyTree],
    get: Tuple[str, ...]
) -> PyTree:
  # If the elements of an output dict are tuples then change the representation
  # to be a tuple of dicts instead. This occurs when the output of a network is
  # a parallel layer.
  return tree_map(lambda *x: dict((g, v) for g, v in zip(get, x)),
                  *[out_dict[g] for g in get])


def _get_f_params(
    f: Callable,
    x: PyTree,
    x_axis: PyTree,
    fx_axis: PyTree,
    kw_axes: Dict[str, PyTree],
    **apply_fn_kwargs
) -> Callable[[PyTree], PyTree]:
  x = _expand_dims(x, x_axis)

  apply_fn_kwargs = {
      k: _expand_dims(v, kw_axes[k]) if k in kw_axes else v
      for k, v in apply_fn_kwargs.items()
  }

  def _f(p: PyTree) -> PyTree:
    fx = f(p, x, **apply_fn_kwargs)
    return _squeeze(fx, fx_axis)

  return _f


def _get_args(
    f: Callable,
    apply_fn_kwargs: Dict[str, PyTree],
    params: PyTree,
    vmap_axes: VMapAxes,
    x1: PyTree,
    x2: PyTree
):
  kwargs1, kwargs2 = utils.split_kwargs(apply_fn_kwargs, x1, x2)

  fx1 = eval_shape(f, params, x1, **kwargs1)
  fx2 = fx1 if utils.all_none(x2) else eval_shape(f, params, x2, **kwargs2)

  x_axis, fx_axis, kw_axes = _canonicalize_axes(vmap_axes, x1, fx1, **kwargs1)

  keys = apply_fn_kwargs.keys()
  args1 = tuple(kwargs1[k] for k in keys)
  args2 = tuple(kwargs2[k] for k in keys)
  return args1, args2, fx1, fx2, fx_axis, keys, kw_axes, x_axis


def _get_f1_f2(
    f: Callable,
    keys: KeysView[str],
    x_axis: PyTree,
    fx_axis: PyTree,
    kw_axes: Dict[str, PyTree],
    args: Tuple,
    x1: PyTree,
    x2: Optional[PyTree]
) -> Tuple[Callable[[PyTree], PyTree], Callable[[PyTree], PyTree]]:
  args1, args2 = args[:len(args) // 2], args[len(args) // 2:]
  _kwargs1 = {k: v for k, v in zip(keys, args1)}
  _kwargs2 = {k: v for k, v in zip(keys, args2)}
  f1 = _get_f_params(f, x1, x_axis, fx_axis, kw_axes, **_kwargs1)
  f2 = f1 if utils.all_none(x2) else _get_f_params(
      f, x2, x_axis, fx_axis, kw_axes, **_kwargs2)
  return f1, f2


_ArrayOrShape = TypeVar('_ArrayOrShape', np.ndarray, ShapedArray)


def _check_einsum_no_broadcast(arrays: List[np.ndarray], dims: List[List[int]]):
  """Check that all matching einsum contracting axis sizes are equal.

  Einsum allows silent broadcasting, and this function helps ensure it doesn't
  happen.
  """
  for idx_1, (a1, dims_1) in enumerate(zip(arrays, dims)):
    if len(set(dims_1)) != len(dims_1):
      raise ValueError(f'Dimensions {idx_1} contain duplicate axes: '
                       f'{dims_1}.')

    for ax_1, dim_1 in enumerate(dims_1):
      sz_idx_1 = a1.shape[ax_1]
      for idx_2, (a2, dims_2) in enumerate(zip(arrays, dims)):
        if dim_1 in dims_2:
          ax_2 = dims_2.index(dim_1)
          sz_idx_2 = a2.shape[ax_2]
          if sz_idx_2 != sz_idx_1:
            raise ValueError(f'Arrays {idx_1} and {idx_2} mismatch '
                             f'sizes at {ax_1} and {ax_2}: '
                             f'{sz_idx_1} != {sz_idx_2}')


def _expand_dims_array(x: _ArrayOrShape, axis: int) -> _ArrayOrShape:
  def expand(x: np.ndarray) -> np.ndarray:
    return np.expand_dims(x, axis)

  if isinstance(x, ShapedArray):
    return eval_shape(expand, x)

  if isinstance(x, np.ndarray):
    return expand(x)

  raise TypeError(type(x), x)


def _expand_dims(
    x: Union[Optional[PyTree], UndefinedPrimal],
    axis: Optional[PyTree]
) -> Optional[PyTree]:
  if axis is None or x is None or isinstance(x, UndefinedPrimal):
    return x
  return tree_map(_expand_dims_array, x, axis)


def _add(x: Optional[PyTree], y: Optional[PyTree]) -> Optional[PyTree]:
  if x is None or y is None:
    return None
  return tree_map(operator.add, x, y)


def _sub(x: PyTree, y: PyTree) -> PyTree:
  return tree_map(operator.sub, x, y)


def _div(x: PyTree, y: int) -> PyTree:
  return tree_map(lambda x: x / y, x)


def _squeeze(x: PyTree, axis: Optional[PyTree]) -> PyTree:
  if axis is None:
    return x

  def squeeze(
      x: np.ndarray,
      axis: Union[None, int, Tuple[int, ...]]
  ) -> np.ndarray:
    """`np.squeeze` analog working with 0-sized axes."""
    if isinstance(axis, int):
      axis = (axis,)

    non_zero_axes = tuple()
    shift = 0

    for a in sorted(axis):
      if x.shape[a - shift] == 0:
        new_shape = x.shape[:a] + x.shape[a + 1:]
        if utils.size_at(new_shape) == 0:
          x = x.reshape(new_shape)
        else:
          x = np.zeros(new_shape, x.dtype)

        shift += 1
      else:
        non_zero_axes += (a - shift,)

    return np.squeeze(x, non_zero_axes)

  return tree_map(squeeze, x, axis)


def _ndim(x: PyTree) -> PyTree:
  return tree_map(lambda x: x.ndim, x)


def _mod(
    x: Optional[PyTree],
    y: PyTree
) -> PyTree:
  if x is None:
    return None
  return tree_map(operator.mod, x, y)


def _diagonal(ntk: PyTree, fx: PyTree) -> PyTree:
  ntk_flat, _ = tree_flatten(ntk)
  fx_flat, fx_tree = tree_flatten(fx)
  n = len(fx_flat)
  diag = [ntk_flat[i * (n + 1)] for i in range(n)]
  return tree_unflatten(fx_tree, diag)


def _canonicalize_axes(
    vmap_axes: Optional[VMapAxes],
    x: PyTree,
    fx: PyTree,
    **kwargs
) -> VMapAxisTriple:
  if isinstance(vmap_axes, tuple) and len(vmap_axes) == 3:
    x_axis, fx_axis, kw_axes = vmap_axes
  else:
    x_axis, fx_axis, kw_axes = vmap_axes, vmap_axes, {}

  if isinstance(x_axis, int):
    x_axis = tree_map(lambda _: x_axis, x)

  if isinstance(fx_axis, int):
    fx_axis = tree_map(lambda _: fx_axis, fx)

  if isinstance(kw_axes, int):
    kw_axes = tree_map(lambda _: kw_axes, kwargs)

  x_axis = _mod(x_axis, _ndim(x))
  fx_axis = _mod(fx_axis, _ndim(fx))
  kw_axes = _mod(kw_axes, {k: _ndim(kwargs[k]) for k in kw_axes})
  return x_axis, fx_axis, kw_axes


def _to_tuple_tree(x: PyTree) -> Tuple:
  """Replace all lists and dictionaries with tuples in a PyTree for hashing."""
  if isinstance(x, (tuple, list)):
    return tuple(_to_tuple_tree(x_i) for x_i in x)

  if isinstance(x, dict):
    return tuple((k, _to_tuple_tree(v)) for k, v in sorted(x.items()))

  return x


def _ntk_shape(fx1_shape, fx2_shape, trace_axes: Axes, diagonal_axes: Axes):
  ntk_shape = ()

  trace_axes = utils.canonicalize_axis(trace_axes, fx1_shape)
  diagonal_axes = utils.canonicalize_axis(diagonal_axes, fx1_shape)

  for i, (a1, a2) in enumerate(zip(fx1_shape, fx2_shape)):
    if i not in trace_axes:
      if i in diagonal_axes:
        assert a1 == a2
        ntk_shape += (a1,)
      else:
        ntk_shape += (a1, a2)
    else:
      assert a1 == a2
  return ntk_shape


def _get_dims(
    df_dy_1: np.ndarray,
    df_dy_2: np.ndarray,
    ndim: int,
    trace_axes: Axes,
    diagonal_axes: Axes
) -> Tuple[List[int], List[int], List[int]]:
  df_dy_dims_1 = list(range(df_dy_1.ndim))
  df_dy_dims_2 = list(range(df_dy_1.ndim, df_dy_1.ndim + df_dy_2.ndim))

  out_dims = []

  for i in range(ndim):
    if i in trace_axes:
      assert df_dy_1.shape[i] == df_dy_2.shape[i]
      df_dy_dims_2[i] = df_dy_dims_1[i]

    elif i in diagonal_axes:
      assert df_dy_1.shape[i] == df_dy_2.shape[i]
      df_dy_dims_2[i] = df_dy_dims_1[i]
      out_dims += [df_dy_dims_1[i]]

    else:
      out_dims += [df_dy_dims_1[i], df_dy_dims_2[i]]

  return df_dy_dims_1, df_dy_dims_2, out_dims


def _vmap(f: Callable, in_axes, out_axes, squeeze_out: bool = True) -> Callable:
  """An expand-then-squeeze `vmap` for `f` expecting/returning batch dims."""
  in_axes_plus_1 = tree_map(lambda x: x if x in (None, -1) else x + 1, in_axes)

  @utils.wraps(f)
  def f_vmapped(*args):
    args = tree_map(_expand_dims, args, in_axes_plus_1,
                    is_leaf=lambda x: isinstance(x, np.ndarray))
    out = vmap(f, in_axes, out_axes)(*args)
    if squeeze_out:
      out_axes_plus_1 = tree_map(
          lambda x: x if x in (None, -1) else x + 1, out_axes)
      out = _squeeze(out, out_axes_plus_1)
    return out

  return f_vmapped


def _get_fx_axis_and_dtype(fx, fx_axis, params: PyTree):
  if fx_axis is None:
    fx_axis = tree_map(lambda x: None, fx)
  # Set the default type to be the least common type ancestor.
  dtypes, _ = tree_flatten(tree_map(np.dtype, params))
  if not dtypes:
    dtype = None
  else:
    dtype = functools.reduce(np.promote_types, dtypes)
  return fx_axis, dtype


def _unravel_dfs(dfs: PyTree, params: PyTree, y: PyTree) -> PyTree:
  dfs = tree_map(functools.partial(_unravel_array_into_pytree, y, 0), dfs)

  if tree_structure(dfs).num_leaves > 0:
    dfs = tree_transpose(tree_structure(tree_map(lambda x, y: [x] * len(y),
                                                 params,
                                                 dfs)),
                         tree_structure(y), dfs)

  if tree_structure(dfs).num_leaves == 0:
    dfs = tree_map(lambda x: dfs, y)
  return dfs


class _MODE(enum.Enum):
  """`F` - final output; `Y` - intermediary pre-activations; `W` - weights."""
  DF_DY = 'DF_DY'
  DY_DW = 'DY_DW'


def _get_df_dys_and_dy_dws(
    fn: Callable[[PyTree], PyTree],
    params: PyTree,
    _j_rules: bool,
    _s_rules: bool,
    _fwd: Optional[bool]
) -> Tuple[PyTree, PyTree]:
  """Computes primitive output cotangents (`df/dy`) and Jacobians (`dy/dw`)."""
  def primals_out_and_pullback(mode: _MODE) -> PyTree:
    return _get_primals_out_and_pullback(fn, mode, _j_rules, _s_rules, _fwd,
                                         params)

  primals_out, pullback_df_dy = primals_out_and_pullback(_MODE.DF_DY)
  df_dys = vmap(pullback_df_dy)(_std_basis(primals_out))
  df_dys = _unravel_dfs(df_dys[0], params, primals_out)

  _, pullback_dy_dw = primals_out_and_pullback(_MODE.DY_DW)
  dy_dws = pullback_dy_dw(primals_out)  # values of `primals_out` don't matter.
  dy_dws = dy_dws[0]

  return df_dys, dy_dws


def _get_primals_out_and_pullback(
    fn: Callable[[PyTree], PyTree],
    mode: _MODE,
    _j_rules: bool,
    _s_rules: bool,
    _fwd: Optional[bool],
    *primals_in: PyTree
) -> Tuple[PyTree, Callable]:
  """Adapted from `jax.interpreters.ad`.

  Returns outputs of `fn` and the "pullback" function, which is similar to the
  regular pullback function (computing cotangents to `primals_in` given output
  cotangents), but collects and returns other quantities.
  """
  primals_in_flat, in_tree = tree_flatten(primals_in)
  fn_flat, out_tree = jax.flatten_fun_nokwargs(lu.wrap_init(fn), in_tree)

  # TODO(romann): handle call primitives more gracefully.
  with jax.disable_jit():
    outs = ad.linearize(fn_flat, *primals_in_flat, has_aux=False)

  primals_out, pvals, jaxpr, consts = outs
  primals_out = tree_unflatten(out_tree(), primals_out)

  def pullback_fn(*cts_in: PyTree):
    cts_in, _ = tree_flatten(cts_in)
    cts_in = tuple(ct for ct, pval in zip(cts_in, pvals) if not pval.is_known())
    dummy_args = [UndefinedPrimal(v.aval) for v in jaxpr.invars]
    cts_out = _backward_pass(jaxpr, mode=mode, consts=consts,
                             primals_in=dummy_args, cotangents_in=cts_in,
                             _j_rules=_j_rules, _s_rules=_s_rules, _fwd=_fwd)
    return tree_unflatten(in_tree, cts_out)

  return primals_out, pullback_fn


def _backward_pass(
    jaxpr: Jaxpr,
    mode: _MODE,
    consts: List[Value],
    primals_in: List[UndefinedPrimal],
    cotangents_in: Tuple[np.ndarray, ...],
    _j_rules: bool,
    _s_rules: bool,
    _fwd: Optional[bool]
) -> Union[List[List[Union[np.ndarray, Zero]]],
           List[List[Tuple[np.ndarray, rules.Structure]]]]:
  """Similar to and adapted from `jax.interpreters.ad.backward_pass`.

  Traverses the computational graph in the same order as the above, but collects
  and returns _not_ the cotangents wrt `jaxpr.invars`, but rather primitive
  output cotangents (`df/dy`) and Jacobians (`dy/dw`). Precisely:

    `mode=_MODE.DF_DY`: cotangents wrt outputs of equations where `jaxpr.invars`
    are inputs.

    `mode=_MODE.DY_DF`: Jacobians (of outputs wrt inputs that are within
    `jaxpr.invars`) of equations to which `jaxpr.invars` are inputs. Jacobians
    are accompanied by their `rules.Structure` metadata.

  The above are then efficiently contracted with each other elsewhere to compute
  the NTK.
  """

  def read_cotangent(v: Var) -> Union[np.ndarray, Zero]:
    return ct_env.pop(v, Zero(v.aval))

  primal_env: Dict[Var, np.ndarray] = {}
  map(functools.partial(_write_primal, primal_env), jaxpr.constvars, consts)
  map(functools.partial(_write_primal, primal_env), jaxpr.invars, primals_in)

  ct_env: Dict[Var, np.ndarray] = {}
  map(functools.partial(_write_cotangent, 'outvars', ct_env),
      jaxpr.outvars, cotangents_in)

  # List of `df_dy`s or `dy_dw`s for each variable in `jaxpr.invars`.
  outs = [[] for _ in jaxpr.invars]

  if mode == _MODE.DY_DW:
    invar_to_structure = rules.get_structure_cache(jaxpr, _s_rules=_s_rules)
    vars_needing_cts_in = set()
  elif mode == _MODE.DF_DY:
    vars_needing_cts_in = _get_vars_needing_cts_in(jaxpr)
  else:
    raise ValueError(f'Unrecognized mode {mode}.')

  for eqn in jaxpr.eqns[::-1]:
    # Do regular backprop.
    cts_in, invals = _backprop_step(
        eqn=eqn,
        primal_env=primal_env,
        ct_env=ct_env,
        read_cotangent=read_cotangent,
        do_write_cotangents=any(
            not isinstance(i, Literal) and i in vars_needing_cts_in
            for i in eqn.invars
        )
    )

    # Compute `df_dy`s or `dy_dw`s.
    for i_eqn, eq_invar in enumerate(eqn.invars):
      if eq_invar in jaxpr.invars:
        i_jaxpr = jaxpr.invars.index(eq_invar)
        inval = invals[i_eqn].aval

        if mode == _MODE.DF_DY:
          if not isinstance(cts_in, Zero):
            if eqn.primitive == lax.reshape_p:
              cts_in = cts_in.reshape(inval.shape)
            cts_in = cts_in.astype(inval.dtype)
          outs[i_jaxpr] += [cts_in]

        elif mode == _MODE.DY_DW:
          structure = rules.get_structure(
              eqn=eqn,
              invals=[v.aval for v in eqn.invars],
              idx=i_eqn,
              _s_rules=_s_rules
          )
          structure &= invar_to_structure[eq_invar]

          if eqn.primitive == lax.reshape_p:
            cts_in = ShapedArray(inval.shape, inval.dtype)
          elif hasattr(cts_in, 'aval'):
            cts_in = cts_in.aval

          trimmed_invals = _trim_invals(invals, structure)
          if not isinstance(cts_in, ShapedArray):
            raise TypeError(cts_in)
          trimmed_cts_in = _trim_cotangents(cts_in, structure)

          if _s_rules:
            eqn = _trim_eqn(eqn, i_eqn, trimmed_invals, trimmed_cts_in)

          def j_fn(invals):
            return _get_jacobian(eqn=eqn,
                                 cts_in=trimmed_cts_in,
                                 invals=invals,
                                 idx=i_eqn,
                                 _fwd=_fwd,
                                 _j_rules=_j_rules)

          for in_d, out_d in zip(structure.in_diagonal, structure.out_diagonal):
            in_axes = [
                None
                if isinstance(invals[ix], UndefinedPrimal)
                else i
                for ix, i in enumerate(in_d)]
            j_fn = _vmap(j_fn, in_axes=(in_axes,), out_axes=out_d)

          dy_dw = j_fn(trimmed_invals)
          outs[i_jaxpr] += [(dy_dw, structure)]

        else:
          raise ValueError(f'Unrecognized mode {mode}.')

  # If output contains any of `primals_in`, this "identity" primitive is not
  # present in `jaxpr.eqns`. Below we treat this case by passing `cotangents_in`
  # as `df_dy`, and an identity matrix as `dy_dw`.
  for i_in, v_out in enumerate(jaxpr.outvars):
    for i_eqn, v in enumerate(jaxpr.invars):
      if v == v_out:
        if mode == _MODE.DF_DY:
          if v in ct_env:
            df_dy = cotangents_in[i_in]
          else:
            df_dy = v.aval

          outs[i_eqn] += [df_dy]
          break

        elif mode == _MODE.DY_DW:
          # Identity function
          structure = rules.get_id_structure(v.aval, _s_rules)
          structure &= invar_to_structure[v]

          # Identity Jacobian
          trimmed_invals = _trim_invals([UndefinedPrimal(v.aval)], structure)
          if not isinstance(v.aval, ShapedArray):
            raise TypeError(v.aval)
          trimmed_cts_in = _trim_cotangents(v.aval, structure)
          dy_dw = _get_jacobian(
              eqn=None,
              cts_in=trimmed_cts_in,
              invals=trimmed_invals,
              idx=0,
              _j_rules=_j_rules,
              _fwd=_fwd,
          )
          outs[i_eqn] += [(dy_dw, structure)]

        else:
          raise ValueError(f'Unrecognized mode {mode}.')

  return outs


def _get_vars_needing_cts_in(jaxpr: Jaxpr) -> Set[Var]:
  """Get a set of variables that need cotangents for structured derivatives.

  Specifically, returns variables which are outputs of equations to which
  `jaxpr.invars` are inputs. Cotangents `df/dy` to these variables are needed
  elsewhere to compute the NTK.
  """
  need_cts: Set[Var] = set()

  def visit(vs: Set[Var]):
    if len(vs) == 0:
      return

    next_visit = set()

    for e in jaxpr.eqns:
      if any(v in e.invars for v in vs):
        for o in e.outvars:
          if o not in need_cts:
            need_cts.add(o)
            next_visit.add(o)

    visit(next_visit)

  visit(set(jaxpr.invars))

  # `invars` don't need cotangents in `STRUCTURED_DERIVATIVES` mode.
  assert all(i not in need_cts for i in jaxpr.invars)
  return need_cts


def _backprop_step(
    eqn: JaxprEqn,
    primal_env: Dict[Var, np.ndarray],
    ct_env: Dict[Var, np.ndarray],
    read_cotangent: Callable[[Var], Union[np.ndarray, Zero]],
    do_write_cotangents: bool = True
) -> Tuple[Union[np.ndarray, Zero],
           List[Union[np.ndarray, UndefinedPrimal]]]:
  """Adapted from `jax.interpreters.ad`."""
  invals = map(functools.partial(_read_primal, primal_env), eqn.invars)
  cts_in = map(read_cotangent, eqn.outvars)
  if not eqn.primitive.multiple_results:
    cts_in = cts_in[0]
  else:
    raise NotImplementedError(
        f'Primitives with multiple outputs are not supported. '
        f'Please file a bug at '
        f'https://github.com/google/neural-tangents/issues. '
        f'Got {len(eqn.outvars)} outputs for {eqn}, with input '
        f'cotangents {cts_in}.')

  if do_write_cotangents:
    cts_out = _eqn_vjp_fn(eqn, cts_in, *invals)
    cts_out = [Zero(v.aval) for v in eqn.invars] if cts_out is Zero else cts_out
    map(functools.partial(_write_cotangent, eqn.primitive, ct_env),
        eqn.invars, cts_out)
  return cts_in, invals


def _trim_cotangents(
    cts_in: ShapedArray,
    structure: rules.Structure
) -> ShapedArray:
  cts_in = _trim_axis(
      cts_in,
      structure.out_trace + structure.out_broadcast + structure.out_diagonal)
  cts_in: ShapedArray
  return cts_in


def _trim_invals(
    invals: List[Union[np.ndarray, UndefinedPrimal]],
    structure: rules.Structure,
) -> List[Union[np.ndarray, UndefinedPrimal]]:
  trimmed_invals = list(invals)

  for i in structure.in_trace_idxs:
    trimmed_invals[i] = _trim_axis(trimmed_invals[i], structure.in_trace)

  for ax in structure.in_broadcast:
    trimmed_invals[structure.in_broadcast_idx] = _trim_axis(
        trimmed_invals[structure.in_broadcast_idx], ax)

  for ax in structure.out_broadcast:
    for i in structure.out_broadcast_idxs:
      trimmed_invals[i] = _trim_axis(trimmed_invals[i], ax)

  for i in range(len(trimmed_invals)):
    for in_d in sorted([axis[i] for axis in structure.in_diagonal
                        if axis[i] is not None],
                       reverse=True):
      if isinstance(trimmed_invals[i], UndefinedPrimal):
        trimmed_invals[i] = _trim_axis(trimmed_invals[i], in_d)

  return trimmed_invals


def _trim_eqn(
    eqn: JaxprEqn,
    idx: int,
    trimmed_invals: List[Union[np.ndarray, UndefinedPrimal]],
    trimmed_cts_in: ShapedArray
) -> JaxprEqn:
  if eqn.primitive in rules.EQN_PARAMS_RULES:
    # Copy the equation parameters to modify.
    trimmed_invals_e = [i.aval if isinstance(i, UndefinedPrimal) else i for i in
                        trimmed_invals]
    params = rules.EQN_PARAMS_RULES[eqn.primitive](
        params=dict(eqn.params),
        idx=idx,
        trimmed_invals=trimmed_invals_e,
        trimmed_cts_in=trimmed_cts_in
    )
    eqn = eqn.replace(params=params)

  return eqn


def _trim_axis(
    x: Union[UndefinedPrimal, ShapedArray, np.ndarray],
    axis: Union[int, Tuple[int, ...]],
) -> Union[UndefinedPrimal, ShapedArray]:
  """Trim `axis` of `x` to be of length `1`. `x` is only used for shape."""
  if isinstance(axis, int):
    axis = (axis,)

  if isinstance(x, UndefinedPrimal):
    return UndefinedPrimal(_trim_axis(x.aval, axis))

  if isinstance(x, (ShapedArray, np.ndarray)):
    return ShapedArray([1 if i in axis else x.shape[i]
                        for i in range(x.ndim)], dtype=x.dtype)

  raise TypeError(type(x), x)


def _eqn_jvp_fn(
    eqn: Optional[JaxprEqn],
    idx: int,
    tangents: np.ndarray,
    *invals
) -> np.ndarray:
  """Perform a JVP for `eqn`."""
  if eqn is None:
    # Identity function
    return tangents

  new_tangents = []
  new_invals = []

  for i_dx, i in enumerate(invals):
    if i_dx == idx:
      inval = np.zeros(i.aval.shape, i.aval.dtype)
      tangent = tangents
    else:
      inval = i
      aval = i.aval if hasattr(i, 'aval') else ShapedArray(i.shape, i.dtype)
      tangent = Zero(aval)
      if isinstance(inval, (UndefinedPrimal, ShapedArray)):
        inval = np.zeros(aval.shape, aval.dtype)

    new_invals.append(inval)
    new_tangents.append(tangent)

  jvp_fn = ad.primitive_jvps[eqn.primitive]
  return jvp_fn(new_invals, new_tangents, **eqn.params)[1]


def _eqn_vjp_fn(
    eqn: Optional[JaxprEqn],
    cts_in: np.ndarray,
    *invals
) -> Tuple[np.ndarray, ...]:
  """Perform a VJP for `eqn`. Adapted from `jax.interpreters.ad`."""
  if eqn is None:
    # Identity function
    return cts_in,

  traceback = eqn.source_info.traceback
  with ad.source_info_util.user_context(traceback):
    if eqn.primitive.call_primitive or eqn.primitive.map_primitive:
      cts_in_avals = [v.aval for v in eqn.outvars]
      call_jaxpr, params = core.extract_call_jaxpr(eqn.primitive, eqn.params)
      cts_out = ad.get_primitive_transpose(eqn.primitive)(
          params, call_jaxpr, invals, cts_in, cts_in_avals, ())
    elif eqn.primitive in ad.reducing_transposes:
      cts_out = ad.reducing_transposes[eqn.primitive](
          (), cts_in, *invals, **eqn.params)
    else:
      cts_out = ad.get_primitive_transpose(eqn.primitive)(cts_in, *invals,
                                                          **eqn.params)
  return cts_out


def _get_jacobian(
    eqn: Optional[JaxprEqn],
    cts_in: ShapedArray,
    invals: List[Union[np.ndarray, UndefinedPrimal]],
    idx: int,
    _j_rules: bool,
    _fwd: Optional[bool],
) -> np.ndarray:
  """Get the (structured) `eqn` output Jacobian wrt `eqn.invars[idx]`."""
  if eqn is None:
    primitive = None
  else:
    primitive = eqn.primitive

  inval_shape = invals[idx].aval.shape
  cts_in_shape = cts_in.shape

  if primitive == xla.xla_call_p:
    raise NotImplementedError(
        f'Call primitives {eqn} not supported. Please file a bug at '
        f'https://github.com/google/neural-tangents/issues.'
    )

  if primitive not in rules.JACOBIAN_RULES:
    warnings.warn(f'No Jacobian rule found for {primitive}.')

  if primitive in rules.JACOBIAN_RULES and _j_rules:
    # Custom Jacobian rule.
    invals_j = [i.aval if isinstance(i, UndefinedPrimal) else i for i in invals]
    dy_dw = rules.JACOBIAN_RULES[primitive](eqn, idx, invals_j, cts_in)

  else:
    # Vanilla Jacobian evaluation.
    if _get_fwd(_fwd, cts_in_shape, inval_shape):
      # Forward mode.
      out_axes = -1
      inputs = invals[idx].aval
      def jac_fn(tangents):
        return _eqn_jvp_fn(eqn, idx, tangents, *invals)

    else:
      # Reverse mode.
      out_axes = 0
      inputs = cts_in
      def jac_fn(cotangents):
        return _eqn_vjp_fn(eqn, cotangents, *invals)[idx]

    eye = _std_basis(inputs)
    dy_dw = vmap(jac_fn, out_axes=out_axes)(eye)
    dy_dw = dy_dw.reshape(cts_in_shape + inval_shape)

  assert dy_dw.shape == cts_in_shape + inval_shape, (
      dy_dw.shape, cts_in_shape, inval_shape)

  return dy_dw


def _write_cotangent(
    prim: core.Primitive,
    ct_env: Dict[Var, np.ndarray],
    v: Var,
    ct: Union[np.ndarray, Zero]
):
  """Adapted from `jax.interpreters.ad`."""
  assert ct is not Zero, (prim, v.aval)
  if ct is None or type(v) is Literal:
    return

  if type(ct) is Zero:
    return

  ct_env[v] = ad.add_tangents(ct_env[v], ct) if v in ct_env else ct
  if ad.config.jax_enable_checks:
    ct_aval = core.get_aval(ct_env[v])
    joined_aval = core.lattice_join(
        v.aval, ct_aval).strip_weak_type().strip_named_shape()
    assert v.aval.strip_weak_type().strip_named_shape() == joined_aval, (
        prim, v.aval, ct_aval)


def _read_primal(
    env: Dict[Var, np.ndarray],
    v: Union[Var, Literal],
    str_match: bool = False
) -> Union[np.ndarray, UndefinedPrimal]:
  if type(v) is Literal:
    return v.val

  if v in env:
    return env[v]

  if str_match:
    for v_ in env:
      if str(v) == str(v_):
        return env[v_]

  return UndefinedPrimal(v.aval)


def _write_primal(
    env: Dict[Var, np.ndarray],
    v: Var,
    val: Union[np.ndarray, UndefinedPrimal]
):
  if not ad.is_undefined_primal(val):
    env[v] = val


def _get_fwd(
    _fwd: Optional[bool],
    cts_in_shape: Tuple[int, ...],
    inval_shape: Tuple[int, ...]
) -> bool:
  if _fwd is None:
    out_size = onp.prod(cts_in_shape)
    in_size = onp.prod(inval_shape)
    _fwd = out_size > in_size
  return _fwd


def _get_flops(f: Callable, optimize: bool, *a, **kw) -> float:
  m = jax.xla_computation(f)(*a, **kw)
  client = jax.lib.xla_bridge.get_backend()
  if optimize:
    m = client.compile(m).hlo_modules()[0]
  else:
    m = m.as_hlo_module()
  analysis = jax.lib.xla_client._xla.hlo_module_cost_analysis(client, m)
  return analysis['flops']


def _std_basis(pytree: PyTree) -> PyTree:
  """Similar to `jax.api._std_basis` without host-side ops."""
  leaves, _ = tree_flatten(pytree)
  ndim = sum(map(np.size, leaves))
  dtype = jax.dtypes.result_type(*leaves)
  flat_basis = np.eye(ndim, dtype=dtype)
  return _unravel_array_into_pytree(pytree, 1, flat_basis)


def _unravel_array_into_pytree(
    pytree: PyTree,
    axis: int,
    arr: np.ndarray
) -> PyTree:
  """Similar to `jax.api._unravel_array_into_pytree` without host-side ops."""
  leaves, treedef = tree_flatten(pytree)
  if arr.ndim > 0:
    axis %= arr.ndim
  shapes = [arr.shape[:axis] + np.shape(l) + arr.shape[axis+1:] for l in leaves]
  parts = np.split(arr, onp.cumsum([np.size(l) for l in leaves[:-1]]), axis)
  reshaped_parts = [np.reshape(x, shape) for x, shape in zip(parts, shapes)]
  return tree_unflatten(treedef, reshaped_parts)


def _get_res_batch_dims(
    contracting_dims: Iterable[int],
    batch_dims: Iterable[int]
) -> List[int]:
  res_batch_dims = [2 * b - i for i, b in enumerate(batch_dims)]
  for i, b in enumerate(batch_dims):
    for c in contracting_dims:
      if b > c:
        res_batch_dims[i] -= 2
  return res_batch_dims


def _dot_general(
    lhs: np.ndarray,
    rhs: np.ndarray,
    contracting_dims: Axes,
    batch_dims: Axes,
    precision=None
) -> np.ndarray:
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

  contracting_dims = utils.canonicalize_axis(contracting_dims, lhs)
  batch_dims = utils.canonicalize_axis(batch_dims, lhs)

  n_batch_dims = len(batch_dims)
  leading_batch_dims = range(n_batch_dims)

  dimension_numbers = ((contracting_dims, contracting_dims),
                       (batch_dims, batch_dims))

  prod = lax.dot_general(lhs, rhs, dimension_numbers, precision)
  prod = utils.zip_axes(prod, n_batch_dims)

  res_batch_dims = _get_res_batch_dims(contracting_dims, batch_dims)
  prod = np.moveaxis(prod, leading_batch_dims, res_batch_dims)
  return prod
