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

"""Functions to make predictions on the train/test set using NTK/NNGP.

Most functions in this module accept training data as inputs and return a new
function `predict_fn` that computes predictions on the train set / given test
set / timesteps.

.. warning::
  `trace_axes` parameter supplied to prediction functions must match the
  respective parameter supplied to the function used to compute the kernel.
  Namely, this is the same `trace_axes` used to compute the empirical kernel
  (`utils/empirical.py`; `diagonal_axes` must be `()`), or `channel_axis` in the
  output of the top layer used to compute the closed-form kernel (`stax.py`;
  note that closed-form kernels currently only support a single `channel_axis`).
"""


import collections
from functools import lru_cache
from typing import Callable, Dict, Generator, Iterable, NamedTuple, Optional, Tuple, Union, Any

import jax
from jax import grad
from jax.experimental import ode
import jax.numpy as np
import jax.scipy as sp
from jax.tree_util import tree_all, tree_map
import numpy as onp
import scipy as osp
from typing_extensions import Protocol
from .utils import dataclasses, utils
from .utils.typing import Axes, Get, KernelFn


PyTree = Any


ArrayOrScalar = Union[None, int, float, np.ndarray]
"""Alias for optional arrays or scalars."""


class PredictFn(Protocol):
  """A type alias for a predictor function."""

  def __call__(
      self,
      t: Optional[ArrayOrScalar] = None,
      fx_train_0: ArrayOrScalar = 0.,
      fx_test_0: Optional[ArrayOrScalar] = None,
      k_test_train: Optional[np.ndarray] = None
  ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    ...


def gradient_descent_mse(
    k_train_train: np.ndarray,
    y_train: np.ndarray,
    learning_rate: float = 1.,
    diag_reg: float = 0.,
    diag_reg_absolute_scale: bool = False,
    trace_axes: Axes = (-1,)
) -> PredictFn:
  r"""Predicts the outcome of function space gradient descent training on MSE.

  Solves in closed form for the continuous-time version of gradient descent.

  Uses the closed-form solution for gradient descent on an MSE loss in function
  space detailed in [*,**] given a Neural Tangent or Neural Network Gaussian
  Process Kernel over the dataset. Given NNGP or NTK, this function will return
  a function that predicts the time evolution for function space points at
  arbitrary time[s] (training step[s]) `t`. Note that these time[s] (step[s])
  are continuous and are interpreted in units of the `learning_rate` so
  `absolute_time = learning_rate * t`, and the scales of `learning_rate` and `t`
  are interchangeable.

  Note that first invocation of the returned `predict_fn` will be slow and
  allocate a lot of memory for its whole lifetime, as either eigendecomposition
  (`t` is a scalar or an array) or Cholesky factorization (`t=None`) of
  `k_train_train` is performed and cached for future invocations (or both, if
  the function is called on both finite and infinite (`t=None`) times).

  [*] "`Neural Tangent Kernel: Convergence and Generalization in Neural Networks
  <https://arxiv.org/abs/1806.07572>`_"

  [**] "`Wide Neural Networks of Any Depth Evolve as Linear
  Models Under Gradient Descent <https://arxiv.org/abs/1902.06720>`_"

  Example:
    >>> import neural_tangents as nt
    >>> #
    >>> t = 1e-7
    >>> kernel_fn = nt.empirical_ntk_fn(f)
    >>> k_train_train = kernel_fn(x_train, None, params)
    >>> k_test_train = kernel_fn(x_test, x_train, params)
    >>> #
    >>> predict_fn = nt.predict.gradient_descent_mse(k_train_train, y_train)
    >>> #
    >>> fx_train_0 = f(params, x_train)
    >>> fx_test_0 = f(params, x_test)
    >>> #
    >>> fx_train_t, fx_test_t = predict_fn(t, fx_train_0, fx_test_0,
    >>>                                    k_test_train)

  Args:
    k_train_train:
      kernel on the training data. Must have the shape of
      `zip(y_train.shape, y_train.shape)` with `trace_axes` absent.

    y_train:
      targets for the training data.

    learning_rate:
      learning rate, step size.

    diag_reg:
      a scalar representing the strength of the diagonal regularization for
      `k_train_train`, i.e. computing `k_train_train + diag_reg * I` during
      Cholesky factorization or eigendecomposition.

    diag_reg_absolute_scale:
      `True` for `diag_reg` to represent regularization in absolute units,
      `False` to be `diag_reg * np.mean(np.trace(k_train_train))`.

    trace_axes:
      `f(x_train)` axes such that `k_train_train` lacks these pairs of
      dimensions and is to be interpreted as :math:`\Theta \otimes I`, i.e.
      block-diagonal along `trace_axes`. These can can be specified either to
      save space and compute, or to even improve approximation accuracy of the
      infinite-width or infinite-samples limit, since in these limits the
      covariance along channel / feature / logit axes indeed converges to a
      constant-diagonal matrix. However, if you target linearized dynamics of a
      specific finite-width network, `trace_axes=()` will yield most accurate
      result.

  Returns:
    A function of signature
    `predict_fn(t, fx_train_0, fx_test_0, k_test_train)` that
    returns output train [and test] set[s] predictions at time[s] `t`.
  """
  _, odd, first, _ = _get_axes(k_train_train)
  trace_axes = utils.canonicalize_axis(trace_axes, y_train)
  trace_axes = tuple(-y_train.ndim + a for a in trace_axes)
  n_t_axes, n_non_t_axes = len(trace_axes), y_train.ndim - len(trace_axes)
  last_t_axes = tuple(range(-n_t_axes, 0))
  non_t_axes = tuple(range(-y_train.ndim, -n_t_axes))

  @lru_cache(1)
  def get_predict_fn_inf():
    with jax.core.eval_context():
      solve = _get_cho_solve(k_train_train, diag_reg, diag_reg_absolute_scale)

    def predict_fn_inf(fx_train_0, fx_test_0, k_test_train):
      fx_train_t = y_train.astype(k_train_train.dtype)
      if fx_test_0 is None:
        return fx_train_t

      rhs = y_train if fx_train_0 is None else y_train - fx_train_0
      dfx_test = np.tensordot(k_test_train, solve(rhs, trace_axes),
                              (odd, first))
      dfx_test = np.moveaxis(dfx_test, last_t_axes, trace_axes)
      fx_test_t = fx_test_0 + dfx_test

      if fx_train_0 is None:
        return fx_test_t
      return fx_train_t, fx_test_t

    return predict_fn_inf

  @lru_cache(1)
  def get_predict_fn_finite():
    with jax.core.eval_context():
      expm1_fn, inv_expm1_fn = _get_fns_in_eigenbasis(
          k_train_train,
          diag_reg,
          diag_reg_absolute_scale,
          (_make_expm1_fn(y_train.size),
           _make_inv_expm1_fn(y_train.size))
      )

    rhs_shape = tuple(y_train.shape[a] for a in trace_axes)

    def predict_fn_finite(t, fx_train_0, fx_test_0, k_test_train):
      t = np.array(t) * learning_rate
      t_shape, t_ndim = t.shape, t.ndim
      first_t_axes = tuple(range(t_ndim))
      t = t.reshape((-1, 1))

      rhs = -y_train if fx_train_0 is None else fx_train_0 - y_train
      rhs = np.moveaxis(rhs, trace_axes, last_t_axes).reshape(
          (-1,) + rhs_shape)
      shape = t_shape + k_train_train.shape[1::2] + rhs_shape

      if fx_train_0 is not None:
        dfx_train = expm1_fn(rhs, t).reshape(shape)
        dfx_train = np.moveaxis(dfx_train, last_t_axes, trace_axes)
        fx_train_t = np.expand_dims(fx_train_0, first_t_axes) + dfx_train

      if fx_test_0 is not None:
        dfx_test = inv_expm1_fn(rhs, t).reshape(shape)
        dfx_test = np.tensordot(k_test_train, dfx_test, (odd, non_t_axes))
        dfx_test = np.moveaxis(
            dfx_test,
            tuple(range(n_non_t_axes, n_non_t_axes + t_ndim)) + last_t_axes,
            tuple(range(t_ndim)) + trace_axes)
        fx_test_t = np.expand_dims(fx_test_0, first_t_axes) + dfx_test

      if fx_train_0 is not None and fx_test_0 is not None:
        return fx_train_t, fx_test_t
      if fx_test_0 is None:
        return fx_train_t
      return fx_test_t

    return predict_fn_finite

  def predict_fn(
      t: Optional[ArrayOrScalar] = None,
      fx_train_0: ArrayOrScalar = 0.,
      fx_test_0: Optional[ArrayOrScalar] = None,
      k_test_train: Optional[np.ndarray] = None
  ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Return output predictions on train [and test] set[s] at time[s] `t`.

    Args:
      t:
        a scalar of array of scalars of any shape. `t=None` is treated as
        infinity and returns the same result as `t=np.inf`, but is computed
        using identity or linear solve for train and test predictions
        respectively instead of eigendecomposition, saving time and precision.
        Equivalent of training steps (but can be fractional).

      fx_train_0:
        output of the network at `t == 0` on the training set. `fx_train_0=None`
        means to not compute predictions on the training set.

      fx_test_0:
        output of the network at `t == 0` on the test set. `fx_test_0=None`
        means to not compute predictions on the test set.

      k_test_train:
        kernel relating test data with training data. Must have the shape of
        `zip(y_test.shape, y_train.shape)` with `trace_axes` absent. Pass
        `k_test_train=None` if you only need non-regularized (`diag_reg=0`)
        predictions on the training set. For regularized train-set predictions,
        pass `k_test_train=k_train_train`.

    Returns:
      `fx_train_t` or `(fx_train_t, fx_test_t)` if `fx_test_0 != None` with
      potentially additional leading time dimensions matching `t.shape`.

    Raises:
      ValueError: if `fx_test_0` is not `None`, but `k_test_train` is `None`.
    """
    _check_inputs(fx_train_0, fx_test_0, k_test_train)

    # Infinite time
    if t is None:
      return get_predict_fn_inf()(fx_train_0, fx_test_0, k_test_train)

    # Finite time
    return get_predict_fn_finite()(t, fx_train_0, fx_test_0, k_test_train)

  return predict_fn


@dataclasses.dataclass
class ODEState:
  """ODE state dataclass holding outputs and auxiliary variables.

  Attributes:
    fx_train:
      training set outputs.

    fx_test:
      test set outputs.

    qx_train:
      training set auxiliary state variable (e.g. momentum).

    qx_test:
      test set auxiliary state variable (e.g. momentum).
  """
  fx_train: Optional[np.ndarray] = None
  fx_test: Optional[np.ndarray] = None
  qx_train: Optional[np.ndarray] = None
  qx_test: Optional[np.ndarray] = None


class PredictFnODE(Protocol):
  """A type alias for a predictor function operating on an `ODEState`."""

  def __call__(
      self,
      t: Optional[ArrayOrScalar] = None,
      fx_train_or_state_0: Union[ArrayOrScalar, ODEState] = 0.,
      fx_test_0: Optional[ArrayOrScalar] = None,
      k_test_train: Optional[np.ndarray] = None
  ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], ODEState]:
    ...


def gradient_descent(
    loss: Callable[[np.ndarray, np.ndarray], float],
    k_train_train: np.ndarray,
    y_train: np.ndarray,
    learning_rate: float = 1.,
    momentum: Optional[float] = None,
    trace_axes: Axes = (-1,)
) -> PredictFnODE:
  r"""Predicts the outcome of function space training using gradient descent.

  Uses an ODE solver. If `momentum != None`, solves a continuous-time version of
  gradient descent with momentum.

  .. note::
    We use standard momentum as opposed to Nesterov momentum.

  Solves the function space ODE for [momentum] gradient descent with a given
  `loss` (detailed in "`Wide Neural Networks of Any Depth Evolve as Linear
  Models Under Gradient Descent <https://arxiv.org/abs/1902.06720>`_".) given a
  Neural Tangent Kernel[s] over the dataset[s] at arbitrary time[s] (step[s])
  `t`. Note that for gradient descent `absolute_time = learning_rate * t` and
  the scales of the learning rate and query step[s] `t` are interchangeable.
  However, the momentum gradient descent ODE is solved in the units of
  `learning_rate**0.5`, and therefore `absolute_time = learning_rate**0.5 * t`,
  hence the `learning_rate` and training time[s] (step[s]) `t` scales are not
  interchangeable.

  Example:
    >>> import neural_tangents as nt
    >>> #
    >>> t = 1e-7
    >>> learning_rate = 1e-2
    >>> momentum = 0.9
    >>> #
    >>> kernel_fn = nt.empirical_ntk_fn(f)
    >>> k_test_train = kernel_fn(x_test, x_train, params)
    >>> #
    >>> from jax.nn import log_softmax
    >>> cross_entropy = lambda fx, y_hat: -np.mean(log_softmax(fx) * y_hat)
    >>> predict_fn = nt.redict.gradient_descent(
    >>>     cross_entropy, k_train_train, y_train, learning_rate, momentum)
    >>> #
    >>> fx_train_0 = f(params, x_train)
    >>> fx_test_0 = f(params, x_test)
    >>> #
    >>> fx_train_t, fx_test_t = predict_fn(t, fx_train_0, fx_test_0,
    >>>                                    k_test_train)

  Args:
    loss:
      a loss function whose signature is `loss(f(x_train), y_train)`. Note:
      the loss function should treat the batch and output dimensions
      symmetrically.

    k_train_train:
      kernel on the training data. Must have the shape of
      `zip(y_train.shape, y_train.shape)` with `trace_axes` absent.

    y_train:
      targets for the training data.

    learning_rate:
      learning rate, step size.

    momentum:
      momentum scalar.

    trace_axes:
      `f(x_train)` axes such that `k_train_train` lacks these pairs of
      dimensions and is to be interpreted as :math:`\Theta \otimes I`, i.e.
      block-diagonal along `trace_axes`. These can can be specified either to
      save space and compute, or to even improve approximation accuracy of the
      infinite-width or infinite-samples limit, since in these limits the
      covariance along channel / feature / logit axes indeed converges to a
      constant-diagonal matrix. However, if you target linearized dynamics of a
      specific finite-width network, `trace_axes=()` will yield most accurate
      result.

  Returns:
    A function that returns output train [and test] set[s] predictions at
    time[s] `t`.
  """
  _, odd, _, _ = _get_axes(k_train_train)
  trace_axes = utils.canonicalize_axis(trace_axes, y_train)
  non_t_axes = tuple(a for a in range(y_train.ndim) if a not in trace_axes)
  last_t_axes = range(-len(trace_axes), 0)

  dtype = k_train_train.dtype
  grad_loss = grad(lambda fx: loss(fx, y_train))

  if momentum is not None:
    learning_rate **= 0.5
    momentum = (momentum - 1.0) / learning_rate

  def get_state_0(fx_train_or_state_0, fx_test_0, fx_test_shape):
    if isinstance(fx_train_or_state_0, ODEState):
      fx_train_0 = fx_train_or_state_0.fx_train
      fx_test_0 = fx_train_or_state_0.fx_test
      qx_train_0 = fx_train_or_state_0.qx_train
      qx_test_0 = fx_train_or_state_0.qx_test
    else:
      fx_train_0 = fx_train_or_state_0
      qx_train_0 = qx_test_0 = None

    if fx_train_0 is None:
      fx_train_0 = np.zeros_like(y_train, dtype)
    else:
      fx_train_0 = np.broadcast_to(fx_train_0, y_train.shape)

    if fx_test_0 is not None:
      fx_test_0 = np.broadcast_to(fx_test_0, fx_test_shape)

    if momentum is None:
      if qx_train_0 is not None or qx_test_0 is not None:
        raise ValueError('Got passed momentum state variables, while '
                         '`momentum is None`.')
    else:
      qx_train_0 = (np.zeros_like(y_train, dtype) if qx_train_0 is None else
                    np.broadcast_to(qx_train_0, y_train.shape))
      qx_test_0 = (None if fx_test_0 is None else
                   (np.zeros(fx_test_shape, dtype) if qx_test_0 is None
                    else np.broadcast_to(qx_test_0, fx_test_shape)))

    return ODEState(fx_train_0, fx_test_0, qx_train_0, qx_test_0)  # pytype: disable=wrong-arg-count

  def get_dstate_dt(k_test_train):
    def dstate_dt(state_t: ODEState, unused_t) -> ODEState:
      fx_train_t, fx_test_t, qx_train_t, qx_test_t = (
          state_t.fx_train, state_t.fx_test, state_t.qx_train, state_t.qx_test)

      dy_df_t = grad_loss(fx_train_t)

      fx_train_t = -np.moveaxis(
          np.tensordot(k_train_train, dy_df_t, (odd, non_t_axes)),
          last_t_axes, trace_axes
      )
      if fx_test_t is not None:
        fx_test_t = -np.moveaxis(
            np.tensordot(k_test_train, dy_df_t, (odd, non_t_axes)),
            last_t_axes, trace_axes
        )

      if momentum is None:
        return ODEState(fx_train_t, fx_test_t)

      fx_train_t += momentum * qx_train_t
      if qx_test_t is not None:
        fx_test_t += momentum * qx_test_t

      return ODEState(qx_train_t, qx_test_t, fx_train_t, fx_test_t)  # pytype: disable=wrong-arg-count
    return dstate_dt

  def predict_fn(
      t: Optional[ArrayOrScalar] = None,
      fx_train_or_state_0: Union[ArrayOrScalar, ODEState] = 0.,
      fx_test_0: Optional[ArrayOrScalar] = None,
      k_test_train: Optional[np.ndarray] = None
  ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], ODEState]:
    """Return output predictions on train [and test] set[s] at time[s] `t`.

    Args:
      t:
        a scalar or array of scalars of any shape in strictly increasing order.
        `t=None` is equivalent to `t=np.inf` and may not converge. Equivalent of
        training steps (but can be fractional).

      fx_train_or_state_0:
        either (a) output of the network at `t == 0` on the training set or (b)
        complete ODE state (`predict.ODEState`). Pass an ODE state if you want
        to operate on the full ODE state instead of output variables only
        (useful for inspecting auxiliary variables or resuming an optimizer with
        auxiliary variables from a specific state. Note that only
        `momentum != None` optimizer currently has auxiliary variables. To
        initialize an ODE state from scratch, call
        `predict.ODEState(fx_train_0, fx_test_0)`. If an ODE state is passed, an
        ODE state is returned. `fx_train_0=None` means to not compute
        predictions on the training set.

      fx_test_0:
        output of the network at `t == 0` on the test set. `fx_test_0=None`
        means to not compute predictions on the test set.

      k_test_train:
        kernel relating test data with training data. Must have the shape of
        `zip(y_test.shape, y_train.shape)` with `trace_axes` absent. Pass
        `k_test_train=None` if you only need predictions on the training set.

    Returns:
      `fx_train_t` or `(fx_train_t, fx_test_t)` if `fx_test_0 != None` with
      potentially additional leading time dimensions matching `t.shape`.
      Alternatively can return an `ODEState` at time[s] `t`.

    Raises:
      ValueError: if `fx_test_0` is not `None`, but `k_test_train` is `None`.
    """
    _check_inputs(fx_train_or_state_0, fx_test_0, k_test_train)

    t = np.array(t if t is not None else np.inf, dtype) * learning_rate
    t_shape = t.shape
    t = t.reshape((-1,))

    # ODE solver requires `t[0]` to be the time where `fx_train_0` [and
    # `fx_test_0`] are evaluated, but also a strictly increasing sequence of
    # timesteps, so we always temporarily append an [almost] `0` at the start.
    t0 = np.where(t[0] == 0,
                  np.full((1,), -1e-24, t.dtype),
                  np.zeros((1,), t.dtype))
    t = np.concatenate([t0, t])

    # Solve the ODE.
    fx_test_shape = _get_fx_test_shape(y_train, k_test_train, trace_axes)
    state_0 = get_state_0(fx_train_or_state_0, fx_test_0, fx_test_shape)
    state_t = ode.odeint(get_dstate_dt(k_test_train), state_0, t)

    # Remove the added `t0`.
    trim = lambda x: x[1:].reshape(t_shape + x.shape[1:])
    trim_tree = lambda tree: tree_map(trim, tree)
    state_t = trim_tree(state_t)

    # `ODEState` -> `ODEState`
    if isinstance(fx_train_or_state_0, ODEState):
      return state_t

    # `np.ndarray` -> `np.ndarray`
    fx_train_t, fx_test_t = state_t.fx_train, state_t.fx_test

    if fx_train_or_state_0 is not None and fx_test_0 is None:
      return fx_train_t
    if fx_test_0 is not None and fx_train_or_state_0 is None:
      return fx_test_t
    return fx_train_t, fx_test_t

  return predict_fn


class Gaussian(NamedTuple):
  """A `(mean, covariance)` convenience namedtuple.

  Attributes:
    mean:
      Mean of shape equal to the shape of the function outputs.

    covariance:
      Covariance of shape equal to the shape of the respective NTK/NNGP kernel.
  """
  mean: np.ndarray
  covariance: np.ndarray


def gp_inference(
    k_train_train,
    y_train: np.ndarray,
    diag_reg: float = 0.,
    diag_reg_absolute_scale: bool = False,
    trace_axes: Axes = (-1,)):
  r"""Compute the mean and variance of the 'posterior' of NNGP/NTK/NTKGP.

  NNGP - the exact posterior of an infinitely wide Bayesian NN. NTK - exact
  distribution of an infinite ensemble of infinitely wide NNs trained with
  gradient flow for infinite time. NTKGP - posterior of a GP (Gaussian process)
  with the NTK covariance (see
  "`Bayesian Deep Ensembles via the Neural Tangent Kernel
  <https://arxiv.org/abs/2007.05864>`_" for how this can correspond to infinite
  ensembles of infinitely wide NNs as well).

  Note that first invocation of the returned `predict_fn` will be slow and
  allocate a lot of memory for its whole lifetime, as a Cholesky factorization
  of `k_train_train.nngp` or `k_train_train.ntk` (or both) is performed and
  cached for future invocations.

  Args:
    k_train_train:
      train-train kernel. Can be (a) :class:`jax.numpy.ndarray`,
      (b) `Kernel` namedtuple, (c) :class:`~neural_tangents.Kernel` object.
      Must contain the necessary `nngp` and/or `ntk` kernels for arguments
      provided to the returned `predict_fn` function. For example, if you
      request to compute posterior test [only] NTK covariance in future
      `predict_fn` invocations, `k_train_train` must contain both `ntk` and
      `nngp` kernels.

    y_train:
      train targets.

    diag_reg:
      a scalar representing the strength of the diagonal regularization for
      `k_train_train`, i.e. computing `k_train_train + diag_reg * I` during
      Cholesky factorization.

    diag_reg_absolute_scale:
      `True` for `diag_reg` to represent regularization in absolute units,
      `False` to be `diag_reg * np.mean(np.trace(k_train_train))`.

    trace_axes:
      `f(x_train)` axes such that `k_train_train`,
      `k_test_train`[, and `k_test_test`] lack these pairs of dimensions and
      are to be interpreted as :math:`\Theta \otimes I`, i.e. block-diagonal
      along `trace_axes`. These can can be specified either to save space and
      compute, or to even improve approximation accuracy of the infinite-width
      or infinite-samples limit, since in these limits the covariance along
      channel / feature / logit axes indeed converges to a  constant-diagonal
      matrix. However, if you target linearized dynamics of a specific
      finite-width network, `trace_axes=()` will yield most accurate result.

  Returns:
    A function of signature `predict_fn(get, k_test_train, k_test_test)`
    computing 'posterior' Gaussian distribution (mean or mean and covariance)
    on a given test set.
  """
  even, odd, first, last = _get_axes(_get_first(k_train_train))
  trace_axes = utils.canonicalize_axis(trace_axes, y_train)

  @lru_cache(2)
  def solve(g: str):
    k_dd = _get_attr(k_train_train, g)
    return _get_cho_solve(k_dd, diag_reg, diag_reg_absolute_scale)

  @lru_cache(2)
  def k_inv_y(g: str):
    return solve(g)(y_train, trace_axes)

  @utils.get_namedtuple('Gaussians')
  def predict_fn(get: Optional[Get] = None,
                 k_test_train=None,
                 k_test_test=None
                 ) -> Dict[str, Union[np.ndarray, Gaussian]]:
    """`test`-set posterior given respective covariance matrices.

    Args:
      get:
        string, the mode of the Gaussian process, either "nngp", "ntk", "ntkgp",
        (see "`Bayesian Deep Ensembles via the Neural Tangent Kernel
        <https://arxiv.org/abs/2007.05864>`_") or a tuple, or `None`. If `None`
        then both `nngp` and `ntk` predictions are returned.

      k_test_train:
        test-train kernel. Can be (a) :class:`jax.numpy.ndarray`,
        (b) `Kernel` namedtuple, (c) :class:`~neural_tangents.Kernel` object.
        Must contain the necessary `nngp` and/or `ntk` kernels for arguments
        provided to the returned `predict_fn` function. For example, if you
        request to compute posterior test [only] NTK covariance, `k_test_train`
        must contain both `ntk` and `nngp` kernels. If `None`, returns
        predictions on the training set. Note that train-set outputs are always
        `N(y_train, 0)` and mostly returned for API consistency.

      k_test_test:
        test-test kernel. Can be (a) :class:`jax.numpy.ndarray`,
        (b) `Kernel` namedtuple, (c) :class:`~neural_tangents.Kernel` object.
        Must contain the necessary `nngp` and/or `ntk` kernels for arguments
        provided to the returned `predict_fn` function. Provide if you want to
        compute test-test posterior covariance. `k_test_test=None` means to not
        compute it. If `k_test_train is None`, pass any non-`None` value (e.g.
        `True`) if you want to get non-regularized (`diag_reg=0`) train-train
        posterior covariance. Note that non-regularized train-set outputs will
        always be the zero-variance Gaussian `N(y_train, 0)` and mostly
        returned for API consistency. For regularized train-set posterior
        outputs according to a positive `diag_reg`, pass
        `k_test_train=k_train_train`, and, optionally,
        `k_test_test=nngp_train_train`.

    Returns:
      Either a :class:`Gaussian` `(mean, variance)` namedtuple or `mean` of the
      GP posterior on the `test` set.
    """
    if get is None:
      get = ('nngp', 'ntk')

    out = {}

    for g in get:
      k = g if g != 'ntkgp' else 'ntk'
      k_dd = _get_attr(k_train_train, k)
      k_td = None if k_test_train is None else _get_attr(k_test_train, k)

      if k_td is None:
        # Train set predictions.
        y = y_train.astype(k_dd.dtype)
      else:
        # Test set predictions.
        y = np.tensordot(k_td, k_inv_y(k), (odd, first))
        y = np.moveaxis(y, range(-len(trace_axes), 0), trace_axes)

      if k_test_test is not None:
        if k_td is None:
          out[g] = Gaussian(y, np.zeros_like(k_dd, k_dd.dtype))
        else:
          if (g == 'ntk' and
              (not hasattr(k_train_train, 'nngp') or
               not hasattr(k_test_train, 'nngp'))):
            raise ValueError(
                'If `"ntk" in get`, and `k_test_test is not None`, '
                'and `k_test_train is not None`, i.e. you request the '
                'NTK posterior covariance on the test set, you need '
                'both NTK and NNGP train-train and test-train matrices '
                'contained in `k_test_train` and `k_train_train`. '
                'Hence they must be `namedtuple`s with `nngp` and '
                '`ntk` attributes.')

          #  kernel of wide NN at initialization
          g_init = 'nngp' if g != 'ntkgp' else 'ntk'

          k_td_g_inv_y = solve(k)(_get_attr(k_test_train, g_init), even)
          k_tt = _get_attr(k_test_test, g_init)

          if g == 'nngp' or g == 'ntkgp':
            cov = np.tensordot(k_td, k_td_g_inv_y, (odd, first))
            cov = k_tt - utils.zip_axes(cov)
            out[g] = Gaussian(y, cov)

          elif g == 'ntk':
            term_1 = solve(g)(k_td, even)
            cov = np.tensordot(_get_attr(k_train_train, 'nngp'), term_1,
                               (odd, first))
            cov = np.tensordot(term_1, cov, (first, first))

            term_2 = np.tensordot(k_td, k_td_g_inv_y, (odd, first))
            term_2 += np.moveaxis(term_2, first, last)
            cov = utils.zip_axes(cov - term_2) + k_tt
            out[g] = Gaussian(y, cov)

          else:
            raise ValueError(g)

      else:
        out[g] = y

    return out

  return predict_fn


_Kernel = collections.namedtuple('Kernel', 'nngp ntk')
"""Helper type to fit cache dictionaries to `get` API."""
_Kernel.__new__.__defaults__ = (None,) * len(_Kernel._fields)


def gradient_descent_mse_ensemble(
    kernel_fn: KernelFn,
    x_train: np.ndarray,
    y_train: np.ndarray,
    learning_rate: float = 1.,
    diag_reg: float = 0.0,
    diag_reg_absolute_scale: bool = False,
    trace_axes: Axes = (-1,),
    **kernel_fn_train_train_kwargs):
  r"""Predicts the gaussian embedding induced by gradient descent on MSE loss.

  This is equivalent to an infinite ensemble of infinite-width networks after
  marginalizing out the initialization, if `kernel_fn` is the kernel function of
  the infinite-width network. Note that `kernel_fn` can in principle also be an
  empirical / Monte Carlo finite-width kernel function, but in this case the
  returned output will not have a simple interpretation (unless these functions
  are used to approximate the infinite-width kernel).

  Note that first invocation of the returned `predict_fn` will be slow and
  allocate a lot of memory for its whole lifetime, as the kernel computation,
  and either eigendecomposition (`t` is a scalar or an array) or Cholesky
  factorization (`t=None`) of `kernel_fn(x_train, None, get)` is performed and
  cached for future invocations (or both, if the function is called on both
  finite and infinite (`t=None`) times).

  Args:
    kernel_fn:
      A kernel function that computes NNGP and/or NTK. Must have a signature
      `kernel_fn(x1, x2, get, **kernel_fn_kwargs)` and return a
      :class:`~neural_tangents.Kernel` object or a `namedtuple` with `nngp`
      and/or `ntk` attributes. Therefore, it can be an `AnalyticKernelFn`, but
      also a `MonteCarloKernelFn`, or an `EmpiricalKernelFn` (but only
      `nt.empirical_kernel_fn` and not `nt.empirical_ntk_fn` or
      `nt.empirical_nngp_fn`, since the latter two do not accept a `get`
      argument). Note that for meaningful outputs, the kernel function must
      represent or at least approximate the infinite-width kernel.

    x_train:
      training inputs.

    y_train:
      training targets.

    learning_rate:
      learning rate, step size.

    diag_reg:
      a scalar representing the strength of the diagonal regularization for
      `kernel_fn(x_train, None, get)`, i.e. computing
      `kernel_fn(x_train, None, get) + diag_reg * I` during Cholesky
      factorization or eigendecomposition.

    diag_reg_absolute_scale:
      `True` for `diag_reg` to represent regularization in absolute units,
      `False` to be
      `diag_reg * np.mean(np.trace(kernel_fn(x_train, None, get)))`.

    trace_axes:
      `f(x_train)` axes such that `kernel_fn(x_train, None, get)`,
      `kernel_fn(x_test, x_train, get)`[, and `kernel_fn(x_test, None, get)`]
      lack these pairs of dimensions and are to be interpreted as
      :math:`\Theta \otimes I`, i.e. block-diagonal along `trace_axes`. These
      can can be specified either to save space and compute, or to even improve
      approximation accuracy of the infinite-width or infinite-samples limit,
      since in these limits the covariance along channel / feature / logit
      axes indeed converges to a constant-diagonal matrix. However, if you
      target linearized dynamics of a specific finite-width network,
      `trace_axes=()` will yield most accurate result.

    **kernel_fn_train_train_kwargs:
      optional keyword arguments passed to `kernel_fn`. For train-train kernel,
      these are passed to `kernel_fn` without changes. For test-test kernel,
      they are passed to `kernel_fn`, unless overwritten by a similar
      `**kernel_fn_test_test_kwargs` arguments passed to the `predict_fn`
      function call. Finally, for test-train kernel, values that are tuples of
      arrays (destined for calls of the finite-width network on training and
      testing data) will be tuples of values combined from
      `**kernel_fn_train_train_kwargs` and `**kernel_fn_test_test_kwargs`, and
      all other values must match.

  Returns:
    A function with signature `predict_fn(t, x_test, get, compute_cov)`
    returning either mean or mean and covariance of the infinite ensemble of
    infinite-width networks outputs on `x_test` at time[s] `t`, in the `get`
    regime (`"nngp"`, `"ntk"`, or `("nngp", "ntk")`).
  """
  expm1 = _make_expm1_fn(y_train.size)
  inv_expm1 = _make_inv_expm1_fn(y_train.size)

  trace_axes = utils.canonicalize_axis(trace_axes, y_train)
  trace_axes = tuple(-y_train.ndim + a for a in trace_axes)
  n_trace_axes = len(trace_axes)
  last_t_axes = range(-n_trace_axes, 0)
  trace_shape = tuple(y_train.shape[a] for a in trace_axes)

  y_train_flat = np.moveaxis(y_train, trace_axes, last_t_axes).reshape(
      (-1,) + trace_shape)

  k_dd_cache = {}

  def get_k_train_train(get: Tuple[str, ...]) -> _Kernel:
    if len(get) == 1:
      get = get[0]
      if get not in k_dd_cache:
        k_dd_cache[get] = kernel_fn(x_train, None, get,
                                    **kernel_fn_train_train_kwargs)

    elif len(get) == 2:
      if not any(g in k_dd_cache for g in get):
        k_dd_cache.update(
            kernel_fn(x_train, None, get,
                      **kernel_fn_train_train_kwargs)._asdict())
      else:
        for g in get:
          if g not in k_dd_cache:
            k_dd_cache[g] = kernel_fn(x_train, None, g,
                                      **kernel_fn_train_train_kwargs)

    else:
      raise ValueError(get)
    return _Kernel(**k_dd_cache)

  @lru_cache(2)
  def eigenspace(get: str):
    k_dd = getattr(get_k_train_train((get,)), get)
    k_dd = _add_diagonal_regularizer(utils.make_2d(k_dd), diag_reg,
                                     diag_reg_absolute_scale)
    evals, evecs = np.linalg.eigh(k_dd)
    evals = np.expand_dims(evals, 0)
    return evals, evecs

  @lru_cache(4)
  def predict_inf(get: Get):
    _, get = utils.canonicalize_get(get)
    k_dd = get_k_train_train(get)
    return gp_inference(k_dd, y_train, diag_reg, diag_reg_absolute_scale,
                        trace_axes)

  def get_kernels(get: Get, x_test: Optional[np.ndarray],
                  compute_cov: bool,
                  **kernel_fn_test_test_kwargs):
    get = _get_dependency(get, compute_cov)
    k_dd = get_k_train_train(get)
    if x_test is None:
      k_td = None
      nngp_tt = compute_cov or None
    else:
      args_train, _ = utils.split_kwargs(kernel_fn_train_train_kwargs, x_train)
      args_test, _ = utils.split_kwargs(kernel_fn_test_test_kwargs, x_test)

      def is_array(x):
        return tree_all(tree_map(
            lambda x: isinstance(x, (onp.ndarray, np.ndarray)), x))

      kwargs_td = dict(kernel_fn_train_train_kwargs)
      kwargs_tt = dict(kernel_fn_train_train_kwargs)

      for k in kernel_fn_test_test_kwargs:
        v_tt = kernel_fn_test_test_kwargs[k]
        v_dd = kernel_fn_train_train_kwargs[k]

        if is_array(v_dd) and is_array(v_tt):
          if (isinstance(v_dd, tuple) and len(v_dd) == 2 and
              isinstance(v_tt, tuple) and len(v_tt) == 2):
            v_td = (args_test[k], args_train[k])
          else:
            v_td = v_tt

        elif v_dd != v_tt:
          raise ValueError(f'Same keyword argument {k} of `kernel_fn` is set to'
                           f'different values {v_dd} != {v_tt} when computing '
                           f'the train-train and test-train/test-test kernels. '
                           f'If this is your intention, please submit a feature'
                           f' request at '
                           f'https://github.com/google/neural-tangents/issues')

        else:
          v_td = v_tt

        kwargs_td[k] = v_td
        kwargs_tt[k] = v_tt

      k_td = kernel_fn(x_test, x_train, get, **kwargs_td)

      if compute_cov:
        nngp_tt = kernel_fn(x_test, None, 'nngp', **kwargs_tt)
      else:
        nngp_tt = None
    return k_dd, k_td, nngp_tt

  @utils.get_namedtuple('Gaussians')
  def predict_fn(t: Optional[ArrayOrScalar] = None,
                 x_test: Optional[np.ndarray] = None,
                 get: Optional[Get] = None,
                 compute_cov: bool = False,
                 **kernel_fn_test_test_kwargs) -> Dict[str, Gaussian]:
    """Return output mean and covariance on the test set at time[s] `t`.

    Args:
      t:
        a scalar of array of scalars of any shape. `t=None` is treated as
        infinity and returns the same result as `t=np.inf`, but is computed
        using linear solve for test predictions instead of eigendecomposition,
        saving time and precision.

      x_test:
        test inputs. `None` means to return non-regularized (`diag_reg=0`)
        predictions on the train-set inputs. For regularized predictions, pass
        `x_test=x_train`.

      get:
        string, the mode of the Gaussian process, either "nngp" or "ntk", or a
        tuple. `get=None` is equivalent to `get=("nngp", "ntk")`.

      compute_cov:
        if `True` computing both `mean` and `variance` and only `mean`
        otherwise.

      **kernel_fn_test_test_kwargs:
        optional keyword arguments passed to `kernel_fn`. See also
        `kernel_fn_train_train_kwargs` argument of the parent function.

    Returns:
      `fx_test_mean_t` or `(fx_test_mean_t, fx_test_cov_t)` if
      `compute_cov == True` with potentially additional leading time dimensions.
    """
    if get is None:
      get = ('nngp', 'ntk')

    # train-train, test-train, test-test.
    k_dd, k_td, nngp_tt = get_kernels(get, x_test, compute_cov,
                                      **kernel_fn_test_test_kwargs)

    # Infinite time.
    if t is None:
      return predict_inf(get)(get=get, k_test_train=k_td,
                              k_test_test=nngp_tt)

    # Finite time.
    t = np.array(t) * learning_rate
    t_shape = t.shape
    t = t.reshape((-1, 1))

    def reshape_mean(mean):
      k = _get_first(k_dd if k_td is None else k_td)
      mean = mean.reshape(t_shape + k.shape[::2] + trace_shape)
      mean = np.moveaxis(mean, last_t_axes, trace_axes)
      return mean

    def reshape_cov(cov):
      k = _get_first(k_dd if k_td is None else k_td)
      cov_shape_t = t_shape + k.shape[::2] * 2
      return utils.zip_axes(cov.reshape(cov_shape_t), len(t_shape))

    out = {}

    for g in get:
      evals, evecs = eigenspace(g)

      # Training set.
      if k_td is None:
        mean = np.einsum(
            'ji,ti,ki,k...->tj...',
            evecs, -expm1(evals, t), evecs, y_train_flat,
            optimize=_optimize())

      # Test set.
      else:
        neg_inv_expm1 = -inv_expm1(evals, t)
        ktd_g = utils.make_2d(getattr(k_td, g))
        mean = np.einsum(
            'lj,ji,ti,ki,k...->tl...',
            ktd_g, evecs, neg_inv_expm1, evecs, y_train_flat,
            optimize=_optimize())

      mean = reshape_mean(mean)

      if nngp_tt is not None:
        nngp_dd = utils.make_2d(k_dd.nngp)

        # Training set.
        if k_td is None:
          if g == 'nngp':
            cov = np.einsum(
                'ji,ti,ki->tjk',
                evecs,
                (np.maximum(evals, 0.) *
                 np.exp(- 2 * np.maximum(evals, 0.) * t / y_train.size)),
                evecs,
                optimize=_optimize())

          elif g == 'ntk':
            exp = np.einsum(
                'mi,ti,ki->tmk',
                evecs,
                np.exp(-np.maximum(evals, 0.) * t / y_train.size),
                evecs,
                optimize=_optimize())
            cov = np.einsum(
                'tmk,kl,tnl->tmn',
                exp,
                nngp_dd,
                exp,
                optimize=_optimize())

          else:
            raise ValueError(g)

        # Test set.
        else:
          _nngp_tt = np.expand_dims(utils.make_2d(nngp_tt), 0)

          if g == 'nngp':
            cov = _nngp_tt - np.einsum(
                'mj,ji,ti,ki,lk->tml',
                ktd_g, evecs, -inv_expm1(evals, 2 * t), evecs, ktd_g,
                optimize=_optimize())

          elif g == 'ntk':
            term_1 = np.einsum(
                'mi,ti,ki,lk->tml',
                evecs, neg_inv_expm1, evecs, ktd_g,
                optimize=_optimize())
            term_2 = np.einsum(
                'mj,ji,ti,ki,lk->tml',
                ktd_g, evecs, neg_inv_expm1, evecs, utils.make_2d(k_td.nngp),
                optimize=_optimize())
            term_2 += np.moveaxis(term_2, 1, 2)
            cov = np.einsum(
                'tji,jk,tkl->til',
                term_1, nngp_dd, term_1,
                optimize=_optimize())
            cov += -term_2 + _nngp_tt

          else:
            raise ValueError(g)

        out[g] = Gaussian(mean, reshape_cov(cov))

      else:
        out[g] = mean

    return out

  return predict_fn


def max_learning_rate(
    ntk_train_train: np.ndarray,
    y_train_size: Optional[int] = None,
    momentum=0.,
    eps: float = 1e-12) -> float:
  r"""Computes the maximal feasible learning rate for infinite width NNs.

  The network is assumed to be trained using mini-/full-batch GD + momentum
  with mean squared loss. The loss is assumed to have the form
  `1/(2 * batch_size * output_size) \|f(train_x) - train_y\|^2`. For vanilla SGD
  (i.e. `momentum = 0`) the maximal feasible learning rate is the largest `\eta`
  such that the operator `(I - \eta / (batch_size * output_size) * NTK)` is a
  contraction, which is `2 * batch_size * output_size * lambda_max(NTK)`. When
  `momentum > 0`, we use
  `2 * (1 + momentum) * batch_size * output_size * lambda_max(NTK)` (see
  *The Dynamics of Momentum* section in
  "`Why Momentum Really Works <https://distill.pub/2017/momentum/>`_").

  Args:
    ntk_train_train:
      analytic or empirical NTK on the training data.

    y_train_size:
      total training set output size, i.e.
      `f(x_train).size ==  y_train.size`. If `output_size=None` it is inferred
      from `ntk_train_train.shape` assuming `trace_axes=()`.

    momentum:
      The `momentum` for momentum optimizers.

    eps:
      a float to avoid zero divisor.

  Returns:
    The maximal feasible learning rate for infinite width NNs.
  """
  ntk_train_train = utils.make_2d(ntk_train_train)
  factor = ntk_train_train.shape[0] if y_train_size is None else y_train_size

  if _is_on_cpu(ntk_train_train):
    max_eva = osp.linalg.eigvalsh(ntk_train_train,
                                  eigvals=(ntk_train_train.shape[0] - 1,
                                           ntk_train_train.shape[0] - 1))[-1]
  else:
    max_eva = np.linalg.eigvalsh(ntk_train_train)[-1]
  lr = 2 * (1 + momentum) * factor / (max_eva + eps)
  return lr


# INTERNAL UTILITIES


def _optimize() -> str:
  """Return contraction order for `np.einsum` based on platform.

  Introduced after https://github.com/google/jax/pull/7512 since TPU seems to
  be more precise in `greeedy` mode.
  """
  return 'greedy' if jax.default_backend() == 'tpu' else 'optimal'


def _get_dependency(get: Get, compute_cov: bool) -> Tuple[str, ...]:
  """Figure out dependency for get."""
  _, get = utils.canonicalize_get(get)
  for g in get:
    if g not in ['nngp', 'ntk']:
      raise NotImplementedError(
          'Can only get either "nngp" or "ntk" predictions, got %s.' % g)
  get_dependency = ()
  if 'nngp' in get or ('ntk' in get and compute_cov):
    get_dependency += ('nngp',)
  if 'ntk' in get:
    get_dependency += ('ntk',)
  return get_dependency


def _get_fns_in_eigenbasis(
    k_train_train: np.ndarray,
    diag_reg: float,
    diag_reg_absolute_scale: bool,
    fns: Iterable[Callable[[np.ndarray, np.ndarray], np.ndarray]]
) -> Generator[Callable[[np.ndarray, np.ndarray], np.ndarray], None, None]:
  """Build functions of a matrix in its eigenbasis.

  Args:
    k_train_train:
      an n x n matrix.

    diag_reg:
      diagonal regularizer strength.

    diag_reg_absolute_scale:
      `True` to use absolute (vs relative to mean trace) regulatization.

    fns:
      a sequence of functions that add on the eigenvalues (evals, dt) ->
      modified_evals.

  Returns:
    A tuple of functions that act as functions of the matrix mat
    acting on vectors: `transform(vec, dt) = fn(mat, dt) @ vec`
  """
  k_train_train = utils.make_2d(k_train_train)
  k_train_train = _add_diagonal_regularizer(k_train_train, diag_reg,
                                            diag_reg_absolute_scale)
  evals, evecs = np.linalg.eigh(k_train_train)
  evals = np.expand_dims(evals, 0)

  def to_eigenbasis(fn):
    """Generates a transform given a function on the eigenvalues."""
    def new_fn(y_train, t):
      return np.einsum('ji,ti,ki,k...->tj...',
                       evecs, fn(evals, t), evecs, y_train,
                       optimize=_optimize())

    return new_fn

  return (to_eigenbasis(fn) for fn in fns)


def _add_diagonal_regularizer(A: np.ndarray,
                              diag_reg: float,
                              diag_reg_absolute_scale: bool) -> np.ndarray:
  dimension = A.shape[0]
  if not diag_reg_absolute_scale:
    diag_reg *= np.trace(A) / dimension
  return A + diag_reg * np.eye(dimension)


def _get_cho_solve(A: np.ndarray,
                   diag_reg: float,
                   diag_reg_absolute_scale: bool,
                   lower: bool = False) -> Callable[[np.ndarray, Axes],
                                                    np.ndarray]:
  x_non_channel_shape = A.shape[1::2]
  A = utils.make_2d(A)
  A = _add_diagonal_regularizer(A, diag_reg, diag_reg_absolute_scale)
  C = sp.linalg.cho_factor(A, lower)

  def cho_solve(b: np.ndarray, b_axes: Axes) -> np.ndarray:
    b_axes = utils.canonicalize_axis(b_axes, b)
    last_b_axes = range(-len(b_axes), 0)
    x_shape = x_non_channel_shape + tuple(b.shape[a] for a in b_axes)

    b = np.moveaxis(b, b_axes, last_b_axes)
    b = b.reshape((A.shape[1], -1))

    x = sp.linalg.cho_solve(C, b)
    x = x.reshape(x_shape)
    return x

  return cho_solve


def _get_fx_test_shape(y_train: np.ndarray,
                       k_test_train: np.ndarray,
                       y_axes: Axes) -> Tuple[int, ...]:
  if k_test_train is None:
    return y_train.shape

  shape = list(k_test_train.shape[::2])
  y_axes = utils.canonicalize_axis(y_axes, y_train)
  for i, c in enumerate(y_train.shape):
    if i in y_axes:
      shape.insert(i, c)
  return tuple(shape)


def _make_expm1_fn(normalization: float):

  def expm1_fn(evals: np.ndarray, t: np.ndarray):
    # Since our matrix really should be positive semidefinite,
    # we can threshold the eigenvalues to squash ones that are negative
    # for numerical reasons.
    return np.expm1(-np.maximum(evals, 0.) * t / normalization)

  return expm1_fn


def _make_inv_expm1_fn(normalization: float):
  expm1_fn = _make_expm1_fn(normalization)

  def _inv_expm1_fn(evals: np.ndarray, t: np.ndarray):
    return expm1_fn(evals, t) / np.abs(evals)

  return _inv_expm1_fn


def _check_inputs(fx_train_or_state_0: Union[ArrayOrScalar, ODEState],
                  fx_test_0: ArrayOrScalar,
                  k_test_train: Optional[np.ndarray]):
  if isinstance(fx_train_or_state_0, ODEState):
    if fx_test_0 is not None:
      raise ValueError('`fx_test_0` is included in `ODEState` and must be set '
                       'to `None`.')

    fx_train_0 = fx_train_or_state_0.fx_train
    fx_test_0 = fx_train_or_state_0.fx_test

  else:
    fx_train_0 = fx_train_or_state_0

  if fx_train_0 is None and fx_test_0 is None:
    raise ValueError('Both `fx_train_0` and `fx_test_0` are `None`, i.e. no '
                     'predictions will be computed.')
  if fx_test_0 is not None and k_test_train is None:
    raise ValueError('To get predictions on the test set, please provide '
                     '`k_test_train` kernel to the parent function.')


def _get_axes(x: np.ndarray):
  n = x.ndim
  return (
      tuple(range(0, n, 2)),
      tuple(range(1, n, 2)),
      tuple(range(0, n // 2)),
      tuple(range(n // 2, n))
  )


def _get_first(k) -> np.ndarray:
  if isinstance(k, (onp.ndarray, np.ndarray)):
    return k

  for g in ('nngp', 'ntk'):
    if hasattr(k, g):
      v = getattr(k, g)
      if v is not None:
        return v

  raise ValueError(k)


def _get_attr(k, g: str) -> np.ndarray:
  if isinstance(k, (onp.ndarray, np.ndarray)):
    return k
  return getattr(k, g)


def _is_on_cpu(x: PyTree) -> bool:
  def _arr_is_on_cpu(x: np.ndarray) -> bool:
    # TODO(romann): revisit when https://github.com/google/jax/issues/1431 and
    # https://github.com/google/jax/issues/1432 are fixed.
    if hasattr(x, 'device_buffer'):
      return 'cpu' in str(x.device_buffer.device()).lower()

    if isinstance(x, (onp.ndarray, np.ndarray)):
      return True

    raise NotImplementedError(type(x))

  return tree_all(tree_map(_arr_is_on_cpu, x))
