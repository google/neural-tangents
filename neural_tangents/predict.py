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
"""Functions to make predictions on the test set using NTK kernel."""


import collections
import functools

from jax.api import grad
from jax.api import jit
import jax.numpy as np
import jax.scipy as sp

from jax.tree_util import tree_all
from jax.tree_util import tree_map
from neural_tangents.utils import empirical
from neural_tangents.utils.utils import canonicalize_get
from neural_tangents.utils.utils import get_namedtuple
from neural_tangents.utils.utils import named_tuple_factory
import scipy as osp
from scipy.integrate._ode import ode

from neural_tangents.utils.typing import KernelFn
from typing import Union, Tuple, Callable, Iterable, Dict, Any


Gaussian = collections.namedtuple('Gaussian', 'mean covariance')


# pylint: disable=g-bare-generic
def gradient_descent_mse(
    g_dd: np.ndarray,
    y_train: np.ndarray,
    g_td: np.ndarray = None,
    diag_reg: float = 0.) -> Callable:
  """Predicts the outcome of function space gradient descent training on MSE.

  Analytically solves for the continuous-time version of gradient descent.

  Uses the analytic solution for gradient descent on an MSE loss in function
  space detailed in [*] given a Neural Tangent Kernel over the dataset. Given
  NTKs, this function will return a function that predicts the time evolution
  for function space points at arbitrary times. Note that times are continuous
  and are measured in units of the learning rate so t = learning_rate * steps.

  [*] https://arxiv.org/abs/1806.07572

  Example:
    >>> from neural_tangents import empirical_ntk_fn
    >>> from neural_tangents import predict
    >>>
    >>> train_time = 1e-7
    >>> kernel_fn = empirical_ntk_fn(f)
    >>> g_td = kernel_fn(x_test, x_train, params)
    >>>
    >>> predict_fn = predict.gradient_descent_mse(g_dd, y_train, g_td)
    >>>
    >>> fx_train_initial = f(params, x_train)
    >>> fx_test_initial = f(params, x_test)
    >>>
    >>> fx_train_final, fx_test_final = predict_fn(
    >>>          fx_train_initial, fx_test_initial, train_time)

  Args:
    g_dd: A kernel on the training data. The kernel should be an `np.ndarray`
      of shape [n_train * output_dim, n_train * output_dim] or
      [n_train, n_train]. In the latter case, the kernel is assumed to be block
      diagonal over the logits.
    y_train: A `np.ndarray` of shape [n_train, output_dim] of targets for the
      training data.
    g_td: A Kernel relating training data with test data. The kernel should be
      an `np.ndarray` of shape [n_test * output_dim, n_train * output_dim] or
      [n_test, n_train]. Note; g_td should have been created in the convention
      kernel_fn(x_train, x_test, params).
    diag_reg: A float, representing the strength of the regularization.

  Returns:
    A function that predicts outputs after t = learning_rate * steps of
    training.

    If g_td is None:
      The function returned is predict(fx, t). Here fx is an `np.ndarray` of
      network outputs and has shape [n_train, output_dim], t is a floating point
      time. predict(fx, t) returns an `np.ndarray` of predictions of shape
      [n_train, output_dim].

    If g_td is not None:
      If a test set Kernel is specified then it returns a function,
      predict(fx_train, fx_test, t). Here fx_train and fx_test are ndarays of
      network outputs and have shape [n_train, output_dim] and
      [n_test, output_dim] respectively and t is a floating point time.
      predict(fx_train, fx_test, t) returns a tuple of predictions of shape
      [n_train, output_dim] and [n_test, output_dim] for train and test points
      respectively.
  """

  g_dd = empirical.flatten_features(g_dd)

  normalization = y_train.size
  output_dimension = y_train.shape[-1]
  expm1_fn, inv_expm1_fn = (_make_expm1_fn(normalization),
                            _make_inv_expm1_fn(normalization))

  def fl(fx):
    """Flatten outputs."""
    return np.reshape(fx, (-1,))

  def ufl(fx):
    """Unflatten outputs."""
    return np.reshape(fx, (-1, output_dimension))

  # Check to see whether the kernel has a logit dimension.
  if y_train.size > g_dd.shape[-1]:
    out_dim, ragged = divmod(y_train.size, g_dd.shape[-1])
    if ragged or out_dim != y_train.shape[-1]:
      raise ValueError()
    fl = lambda x: x
    ufl = lambda x: x

  g_dd_plus_reg = _add_diagonal_regularizer(g_dd, diag_reg)
  expm1_dot_vec, inv_expm1_dot_vec = _eigen_fns(g_dd_plus_reg,
                                                (expm1_fn, inv_expm1_fn))

  if g_td is None:

    def train_predict(dt, fx=0.0):
      gx_train = fl(fx - y_train)
      dgx = expm1_dot_vec(gx_train, dt)
      return ufl(dgx) + fx

    return train_predict

  g_td = empirical.flatten_features(g_td)

  def predict_using_kernel(dt, fx_train=0., fx_test=0.):
    gx_train = fl(fx_train - y_train)
    dgx = expm1_dot_vec(gx_train, dt)
    # Note: consider use a linalg solve instead of the eigeninverse
    # dfx = sp.linalg.solve(g_dd, dgx, sym_pos=True)
    dfx = inv_expm1_dot_vec(gx_train, dt)
    dfx = np.dot(g_td, dfx)
    return ufl(dgx) + fx_train, fx_test + ufl(dfx)

  return predict_using_kernel


def gradient_descent(
    g_dd: np.ndarray,
    y_train: np.ndarray,
    loss: Callable[[np.ndarray, np.ndarray], float],
    g_td: np.ndarray = None) -> Callable:
  """Predicts the outcome of function space gradient descent training on `loss`.

  Solves for continuous-time gradient descent using an ODE solver.

  Solves the function space ODE for continuous gradient descent with a given
  loss (detailed in [*]) given a Neural Tangent Kernel over the dataset. This
  function returns a function that predicts the time evolution for function
  space points at arbitrary times. Note that times are continuous and are
  measured in units of the learning rate so that t = learning_rate * steps.

  This function uses the scipy ode solver with the 'dopri5' algorithm.

  [*] https://arxiv.org/abs/1902.06720

  Example:
    >>> from neural_tangents import empirical_ntk_fn
    >>> from jax.experimental import stax
    >>> from neural_tangents import predict
    >>>
    >>> train_time = 1e-7
    >>> kernel_fn = empirical_ntk_fn(f)
    >>> g_td = kernel_fn(x_test, x_train, params)
    >>>
    >>> from jax.experimental import stax
    >>> cross_entropy = lambda fx, y_hat: -np.mean(stax.logsoftmax(fx) * y_hat)
    >>> predict_fn = predict.gradient_descent(
    >>>     g_dd, y_train, cross_entropy, g_td)
    >>>
    >>> fx_train_initial = f(params, x_train)
    >>> fx_test_initial = f(params, x_test)
    >>>
    >>> fx_train_final, fx_test_final = predict_fn(
    >>>     fx_train_initial, fx_test_initial, train_time)

  Args:
    g_dd: A Kernel on the training data. The kernel should be an `np.ndarray`
      of shape [n_train * output_dim, n_train * output_dim] or
      [n_train, n_train]. In the latter case it is assumed that the kernel is
      block diagonal over the logits.
    y_train: A `np.ndarray` of shape [n_train, output_dim] of labels for the
      training data.
    loss: A loss function whose signature is loss(fx, y_hat) where fx is an
      `np.ndarray` of function space output_dim of the network and y_hat are
      targets. Note: the loss function should treat the batch and output
      dimensions symmetrically.
    g_td: A Kernel relating training data with test data. The kernel should be
      an `np.ndarray` of shape [n_test * output_dim, n_train * output_dim] or
      [n_test, n_train]. Note: g_td should have been created in the convention
      kernel_fn(x_test, x_train, params).

  Returns:
    A function that predicts outputs after t = learning_rate * steps of
    training.

    If g_td is None:
      The function returned is predict(fx, t). Here fx is an `np.ndarray` of
      network outputs and has shape [n_train, output_dim], t is a floating point
      time. predict(fx, t) returns an `np.ndarray` of predictions of shape
      [n_train, output_dim].

    If g_td is not None:
      If a test set Kernel is specified then it returns a function,
      predict(fx_train, fx_test, t). Here fx_train and fx_test are ndarays of
      network outputs and have shape [n_train, output_dim] and
      [n_test, output_dim] respectively and t is a floating point time.
      predict(fx_train, fx_test, t) returns a tuple of predictions of shape
      [n_train, output_dim] and [n_test, output_dim] for train and test points
      respectively.
  """

  output_dimension = y_train.shape[-1]

  g_dd = empirical.flatten_features(g_dd)

  def fl(fx):
    """Flatten outputs."""
    return np.reshape(fx, (-1,))

  def ufl(fx):
    """Unflatten outputs."""
    return np.reshape(fx, (-1, output_dimension))

  # These functions are used inside the integrator only if the kernel is
  # diagonal over the logits.
  ifl = lambda x: x
  iufl = lambda x: x

  # Check to see whether the kernel has a logit dimension.
  if y_train.size > g_dd.shape[-1]:
    out_dim, ragged = divmod(y_train.size, g_dd.shape[-1])
    if ragged or out_dim != y_train.shape[-1]:
      raise ValueError()
    ifl = fl
    iufl = ufl

  y_train = np.reshape(y_train, (-1))
  grad_loss = grad(functools.partial(loss, y_hat=y_train))

  if g_td is None:
    dfx_dt = lambda unused_t, fx: -ifl(np.dot(g_dd, iufl(grad_loss(fx))))

    def predict(dt, fx=0.):
      r = ode(dfx_dt).set_integrator('dopri5')
      r.set_initial_value(fl(fx), 0)
      r.integrate(dt)

      return ufl(r.y)
  else:
    g_td = empirical.flatten_features(g_td)

    def dfx_dt(unused_t, fx, train_size):
      fx_train = fx[:train_size]
      dfx_train = -ifl(np.dot(g_dd, iufl(grad_loss(fx_train))))
      dfx_test = -ifl(np.dot(g_td, iufl(grad_loss(fx_train))))
      return np.concatenate((dfx_train, dfx_test), axis=0)

    def predict(dt, fx_train=0., fx_test=0.):
      r = ode(dfx_dt).set_integrator('dopri5')

      fx = fl(np.concatenate((fx_train, fx_test), axis=0))
      train_size, output_dim = fx_train.shape
      r.set_initial_value(fx, 0).set_f_params(train_size * output_dim)
      r.integrate(dt)
      fx = ufl(r.y)

      return fx[:train_size], fx[train_size:]

  return predict


def momentum(
    g_dd: np.ndarray,
    y_train: np.ndarray,
    loss: Callable[[np.ndarray, np.ndarray], np.ndarray],
    learning_rate: float,
    g_td: np.ndarray = None,
    momentum: float = 0.9) -> Tuple[Callable, Callable, Callable]:
  r"""Predicts the outcome of function space training using momentum descent.

  Solves a continuous-time version of standard momentum instead of
  Nesterov momentum using an ODE solver.

  Solves the function space ODE for momentum with a given loss (detailed
  in [*]) given a Neural Tangent Kernel over the dataset. This function returns
  a triplet of functions that initialize state variables, predicts the time
  evolution for function space points at arbitrary times and retrieves the
  function-space outputs from the state. Note that times are continuous and are
  measured in units of the learning rate so that
  t = \sqrt(learning_rate) * steps.

  This function uses the scipy ode solver with the 'dopri5' algorithm.

  [*] https://arxiv.org/abs/1902.06720

  Example:
    >>> from neural_tangents import empirical_ntk_fn
    >>> from neural_tangents import predict
    >>>
    >>> train_time = 1e-7
    >>> learning_rate = 1e-2
    >>>
    >>> kernel_fn = empirical_ntk_fn(f)
    >>> g_td = kernel_fn(x_test, x_train, params)
    >>>
    >>> from jax.experimental import stax
    >>> cross_entropy = lambda fx, y_hat: -np.mean(stax.logsoftmax(fx) * y_hat)
    >>> init_fn, predict_fn, get_fn = predict.momentum(
    >>>                   g_dd, y_train, cross_entropy, learning_rate, g_td)
    >>>
    >>> fx_train_initial = f(params, x_train)
    >>> fx_test_initial = f(params, x_test)
    >>>
    >>> lin_state = init_fn(fx_train_initial, fx_test_initial)
    >>> lin_state = predict_fn(lin_state, train_time)
    >>> fx_train_final, fx_test_final = get_fn(lin_state)

  Args:
    g_dd: Kernel on the training data. The kernel should be an `np.ndarray` of
      shape [n_train * output_dim, n_train * output_dim].
    y_train: A `np.ndarray` of shape [n_train, output_dim] of labels for the
      training data.
    loss: A loss function whose signature is loss(fx, y_hat) where fx an
      `np.ndarray` of function space outputs of the network and y_hat are
      labels. Note: the loss function should treat the batch and output
      dimensions symmetrically.
    learning_rate:  A float specifying the learning rate.
    g_td: Kernel relating training data with test data. Should be an
      `np.ndarray` of shape [n_test * output_dim, n_train * output_dim]. Note:
      g_td should have been created in the convention g_td = kernel_fn(x_test,
      x_train, params).
    momentum: float specifying the momentum.

  Returns:
    Functions to predicts outputs after t = \sqrt(learning_rate) * steps of
    training. Generically three functions are returned, an init_fn that creates
    auxiliary velocity variables needed for optimization and packs them into
    a state variable, a predict_fn that computes the time-evolution of the state
    for some dt, and a get_fn that extracts the predictions from the state.

    If g_td is None:
      init_fn(fx_train): Takes a single `np.ndarray` of shape
      [n_train, output_dim] and returns a tuple containing the output_dim as
      an int and an `np.ndarray` of shape [2 * n_train * output_dim].

      predict_fn(state, dt): Takes a state described above and a floating point
      time. Returns a new state with the same type and shape.

      get_fn(state): Takes a state and returns an `np.ndarray` of shape
      [n_train, output_dim].

    If g_td is not None:
      init_fn(fx_train, fx_test): Takes two `np.ndarray`s of shape
      [n_train, output_dim] and [n_test, output_dim] respectively. Returns a
      tuple with an int giving 2 * n_train * output_dim, an int containing the
      output_dim, and an `np.ndarray` of shape
      [2 * (n_train + n_test) * output_dim].

      predict_fn(state, dt): Takes a state described above and a floating point
      time. Returns a new state with the same type and shape.

      get_fn(state): Takes a state and returns two `np.ndarray` of shape
      [n_train, output_dim] and [n_test, output_dim] respectively.
  """
  output_dimension = y_train.shape[-1]

  g_dd = empirical.flatten_features(g_dd)

  momentum = (momentum - 1.0) / np.sqrt(learning_rate)

  def fl(fx):
    """Flatten outputs."""
    return np.reshape(fx, (-1,))

  def ufl(fx):
    """Unflatten outputs."""
    return np.reshape(fx, (-1, output_dimension))

  # These functions are used inside the integrator only if the kernel is
  # diagonal over the logits.
  ifl = lambda x: x
  iufl = lambda x: x

  # Check to see whether the kernel has a logit dimension.
  if y_train.size > g_dd.shape[-1]:
    out_dim, ragged = divmod(y_train.size, g_dd.shape[-1])
    if ragged or out_dim != y_train.shape[-1]:
      raise ValueError()
    ifl = fl
    iufl = ufl

  y_train = np.reshape(y_train, (-1))
  grad_loss = grad(functools.partial(loss, y_hat=y_train))

  if g_td is None:

    def dr_dt(unused_t, r):
      fx, qx = np.split(r, 2)
      dfx = qx
      dqx = momentum * qx - ifl(np.dot(g_dd, iufl(grad_loss(fx))))
      return np.concatenate((dfx, dqx), axis=0)

    def init_fn(fx_train=0.):
      fx_train = fl(fx_train)
      qx_train = np.zeros_like(fx_train)
      return np.concatenate((fx_train, qx_train), axis=0)

    def predict_fn(state, dt):
      solver = ode(dr_dt).set_integrator('dopri5')
      solver.set_initial_value(state, 0)
      solver.integrate(dt)

      return solver.y

    def get_fn(state):
      return ufl(np.split(state, 2)[0])

  else:
    g_td = empirical.flatten_features(g_td)

    def dr_dt(unused_t, r, train_size):
      train, test = r[:train_size], r[train_size:]
      fx_train, qx_train = np.split(train, 2)
      _, qx_test = np.split(test, 2)
      dfx_train = qx_train
      dqx_train = \
          momentum * qx_train - ifl(np.dot(g_dd, iufl(grad_loss(fx_train))))
      dfx_test = qx_test
      dqx_test = \
          momentum * qx_test - ifl(np.dot(g_td, iufl(grad_loss(fx_train))))
      return np.concatenate((dfx_train, dqx_train, dfx_test, dqx_test), axis=0)

    def init_fn(fx_train=0., fx_test=0.):
      train_size = fx_train.shape[0]
      fx_train, fx_test = fl(fx_train), fl(fx_test)
      qx_train = np.zeros_like(fx_train)
      qx_test = np.zeros_like(fx_test)
      return (2 * train_size * output_dimension,
              np.concatenate((fx_train, qx_train, fx_test, qx_test), axis=0))

    def predict_fn(state, dt):
      train_size, state = state
      solver = ode(dr_dt).set_integrator('dopri5')
      solver.set_initial_value(state, 0).set_f_params(train_size)
      solver.integrate(dt)

      return train_size, solver.y

    def get_fn(state):
      train_size, state = state
      train, test = state[:train_size], state[train_size:]
      return ufl(np.split(train, 2)[0]), ufl(np.split(test, 2)[0])

  return init_fn, predict_fn, get_fn


def gp_inference(kernel_fn: KernelFn,
                 x_train: np.ndarray,
                 y_train: np.ndarray,
                 x_test: np.ndarray,
                 get: Union[str, Tuple[str, ...]],
                 diag_reg: Union[float, Iterable[float]] = 0.,
                 compute_cov: bool = False,
                 **kernel_fn_kwargs) -> Union[Gaussian, np.ndarray]:
  """Compute the mean and variance of the `posterior` of NNGP and NTK.

  Note that this method is equivalent to `gradient_descent_mse_gp` at infinite
  time.

  Example:
    >>> predict = gradient_descent_mse_gp(kernel_fn, x_train, y_train, x_test,
    >>>                                   get, diag_reg, compute_cov)
    >>> predict(np.inf) == predict(None) == gp_inference(kernel_fn, x_train,
    >>>     y_train, x_test, get, diag_reg, compute_cov)

  Args:
    kernel_fn: A kernel function that computes NNGP and NTK.
    x_train: A `np.ndarray`, representing the training data.
    y_train: A `np.ndarray`, representing the labels of the training data.
    x_test: A `np.ndarray`, representing the test data.
    get: string, the mode of the Gaussian process, either "nngp" or "ntk", or a
      tuple, or None. If `None` then both `nngp` and `ntk` predictions are
      returned.
    diag_reg: A float or iterable of floats, representing the strength of the
      regularization.
    compute_cov: A boolean. If `True` computing both `mean` and `variance` and
    :**kernel_fn_kwargs: optional keyword arguments passed to `kernel_fn`.
      only `mean` otherwise.
  Returns:
    Either a Gaussian(`mean`, `variance`) namedtuple or `mean` of the GP
    posterior or generator function returning Gaussian or `mean` corresponding
    to diag_reg values.
  """
  if get is None:
    get = ('nngp', 'ntk')
  kdd, ktd, nngp_tt = _get_matrices(kernel_fn, x_train, x_test, get,
                                    compute_cov, **kernel_fn_kwargs)
  gp_inference_mat = (_gp_inference_mat_jit_cpu if _is_on_cpu(kdd) else
                      _gp_inference_mat_jit)
  try:
    iterator = iter(diag_reg)
  except TypeError:
    # diag_reg is a number.
    return gp_inference_mat(kdd, ktd, nngp_tt, y_train, get, diag_reg)

  def iter_func():
    for diag_reg in iterator:
      yield gp_inference_mat(kdd, ktd, nngp_tt, y_train, get, diag_reg)
  return iter_func()


@get_namedtuple('Gaussians')
def _gp_inference_mat(kdd: np.ndarray,
                      ktd: np.ndarray,
                      nngp_tt: np.ndarray,
                      y_train: np.ndarray,
                      get: Union[str, Tuple[str, ...]],
                      diag_reg: float) -> Union[Gaussian, np.ndarray]:
  """Compute the mean and variance of the `posterior` of NNGP and NTK.

  Args:
    kdd: A train-train `Kernel` namedtuple.
    ktd: A test-train `Kernel` namedtuple.
    nngp_tt: A test-test `nngp` kernel.
    y_train: A `np.ndarray`, representing the train targets.
    get: string, the mode of the Gaussian process, either "nngp" or "ntk", or a
      tuple, or `None`. If `None` then both `nngp` and `ntk` predictions are
      returned.
    diag_reg: A float, representing the strength of the regularization.

  Returns:
    Either a Gaussian(`mean`, `variance`) namedtuple or `mean` of the GP
    posterior.
  """
  out = {}
  if get is None:
    get = ('nngp', 'ntk')
  if 'nngp' in get:
    op = _inv_operator(kdd.nngp, diag_reg)
    pred_mean = _mean_prediction(op, ktd.nngp, y_train)
    if nngp_tt is not None:
      pred_cov = _nngp_cov(op, ktd.nngp, nngp_tt)
    out['nngp'] = (
        Gaussian(pred_mean, pred_cov) if nngp_tt is not None else pred_mean)

  if 'ntk' in get:
    op = _inv_operator(kdd.ntk, diag_reg)
    pred_mean = _mean_prediction(op, ktd.ntk, y_train)
    if nngp_tt is not None:
      pred_cov = _ntk_cov(op, ktd.ntk, kdd.nngp, ktd.nngp, nngp_tt)
    out['ntk'] = (Gaussian(pred_mean, pred_cov) if nngp_tt is not None
                  else pred_mean)

  return out


_gp_inference_mat_jit = jit(_gp_inference_mat, static_argnums=(4,))


_gp_inference_mat_jit_cpu = jit(_gp_inference_mat, static_argnums=(4,),
                                backend='cpu')


def _get_matrices(kernel_fn, x_train, x_test, get, compute_cov,
                  **kernel_fn_kwargs):
  get = _get_dependency(get, compute_cov)
  kdd = kernel_fn(x_train, None, get, **kernel_fn_kwargs)
  ktd = kernel_fn(x_test, x_train, get, **kernel_fn_kwargs)
  if compute_cov:
    nngp_tt = kernel_fn(x_test, x_test, 'nngp', **kernel_fn_kwargs)
  else:
    nngp_tt = None
  return kdd, ktd, nngp_tt


# TODO(schsam): Refactor this method to make use of @getter.
def gradient_descent_mse_gp(kernel_fn: KernelFn,
                            x_train: np.ndarray,
                            y_train: np.ndarray,
                            x_test: np.ndarray,
                            get: Union[str, Tuple[str, ...]],
                            diag_reg: float = 0.0,
                            compute_cov: bool = False,
                            **kernel_fn_kwargs) -> Callable:
  """Predicts the gaussian embedding induced by gradient descent with mse loss.

  This is equivalent to an infinite ensemble of networks after marginalizing
  out the initialization.

  Args:
    kernel_fn: A kernel function that computes NNGP and NTK.
    x_train: A `np.ndarray`, representing the training data.
    y_train: A `np.ndarray`, representing the labels of the training data.
    x_test: A `np.ndarray`, representing the test data.
    get: string, the mode of the Gaussian process, either "nngp" or "ntk", or
      a tuple.
    diag_reg: A float, representing the strength of the regularization.
    compute_cov: A boolean. If `True` computing both `mean` and `variance` and
      only `mean` otherwise.
    :**kernel_fn_kwargs: optional keyword arguments passed to `kernel_fn`.

  Returns:
    A function that predicts the gaussian parameters at t:
    predict(t) -> Gaussian(mean, variance).
    If compute_cov is False, only returns the mean.
  """
  if get is None:
    get = ('nngp', 'ntk')

  if isinstance(get, str):
    # NOTE(schsam): This seems like an ugly solution that involves an extra
    # indirection. It might be nice to clean it up.
    def _predict(t=None):
      return gradient_descent_mse_gp(
          kernel_fn,
          x_train,
          y_train,
          x_test,
          diag_reg=diag_reg,
          get=(get,),
          compute_cov=compute_cov,
          **kernel_fn_kwargs
      )(t)[0]

    return _predict

  _, get = canonicalize_get(get)

  normalization = y_train.size
  op_fn = _make_inv_expm1_fn(normalization)

  eigenspace: Dict[str, Any] = {}

  kdd, ktd, nngp_tt = _get_matrices(kernel_fn, x_train, x_test, get,
                                    compute_cov, **kernel_fn_kwargs)
  gp_inference_mat = (_gp_inference_mat_jit_cpu if _is_on_cpu(kdd) else
                      _gp_inference_mat_jit)

  @_jit_cpu(kdd)
  def predict(t=None):
    """`t=None` is equivalent to infinite time and calls `gp_inference`."""
    if t is None:
      return gp_inference_mat(kdd, ktd, nngp_tt, y_train, get, diag_reg)

    if not eigenspace:
      for g in get:
        k = kdd.nngp if g == 'nngp' else kdd.ntk
        k_dd_plus_reg = _add_diagonal_regularizer(k, diag_reg)
        eigenspace[g] = np.linalg.eigh(k_dd_plus_reg)

    out = {}

    if 'nngp' in get:
      evals, evecs = eigenspace['nngp']
      op_evals = -op_fn(evals, t)
      pred_mean = _mean_prediction_einsum(evecs, op_evals, ktd.nngp, y_train)
      if compute_cov:
        op_evals_x2 = -op_fn(evals, 2 * t)
        pred_cov = nngp_tt - np.einsum(
            'mj,ji,i,ki,lk->ml',
            ktd.nngp,
            evecs,
            op_evals_x2,
            evecs,
            ktd.nngp,
            optimize=True)

      out['nngp'] = Gaussian(pred_mean, pred_cov) if compute_cov else pred_mean

    if 'ntk' in get:
      evals, evecs = eigenspace['ntk']
      op_evals = -op_fn(evals, t)
      pred_mean = _mean_prediction_einsum(evecs, op_evals, ktd.ntk, y_train)
      if compute_cov:
        # inline the covariance calculation with einsum.
        term_1 = np.einsum(
            'mi,i,ki,lk->ml', evecs, op_evals, evecs, ktd.ntk, optimize=True)
        pred_cov = np.einsum(
            'ji,jk,kl->il', term_1, kdd.nngp, term_1, optimize=True)
        term_2 = np.einsum(
            'mj,ji,i,ki,lk->ml',
            ktd.ntk,
            evecs,
            op_evals,
            evecs,
            ktd.nngp,
            optimize=True)
        term_2 += np.transpose(term_2)
        pred_cov += -term_2 + nngp_tt

      out['ntk'] = Gaussian(pred_mean, pred_cov) if compute_cov else pred_mean

    returntype = named_tuple_factory('Gaussians', get)
    return returntype(*tuple(out[g] for g in get))

  return predict


## Utility functions
def _get_dependency(get, compute_cov):
  """Figure out dependency for get."""
  _, get = canonicalize_get(get)
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


def _eigen_fns(
    mat: np.ndarray, fns: Tuple[Callable, ...]) -> Tuple[Callable, ...]:
  """Build functions of a matrix in its eigenbasis.

  Args:
    mat: an n x n matrix
    fns: a sequence of functions that add on the eigenvalues (evals, dt) ->
      modified_evals

  Returns:
    A tuple of functions that act as functions of the matrix mat
    acting on vectors: `transform(vec, dt) = fn(mat, dt) @ vec`
  """
  evals, evecs = np.linalg.eigh(mat)

  def transform(fn):
    """Generates a transform given a function on the eigenvalues."""
    def _(vec, dt):
      return np.einsum(
          'ji,i,ki,k...->j...',
          evecs, fn(evals, dt), evecs, vec, optimize=True)

    return _

  return tuple(transform(fn) for fn in fns)


def _add_diagonal_regularizer(covariance, diag_reg=0.):
  dimension = covariance.shape[0]
  reg = np.trace(covariance) / dimension
  return covariance + diag_reg * reg * np.eye(dimension)


def _inv_operator(g_dd, diag_reg=0.0):
  g_dd_plus_reg = _add_diagonal_regularizer(g_dd, diag_reg)
  return lambda vec: sp.linalg.solve(g_dd_plus_reg, vec, sym_pos=True)


def _make_flatten_uflatten(g_td, y_train):
  """Create the flatten and unflatten utilities."""
  output_dimension = y_train.shape[-1]

  def fl(fx):
    """Flatten outputs."""
    return np.reshape(fx, (-1,))

  def ufl(fx):
    """Unflatten outputs."""
    return np.reshape(fx, (-1, output_dimension))

  if y_train.size > g_td.shape[-1]:
    out_dim, ragged = divmod(y_train.size, g_td.shape[-1])
    if ragged or out_dim != output_dimension:
      raise ValueError('The batch size of `y_train` must be the same as the'
                       ' last dimension of `g_td`')
    fl = lambda x: x
    ufl = lambda x: x
  return fl, ufl


def _mean_prediction(
    op: Callable, g_td: np.ndarray, y_train: np.ndarray) -> np.ndarray:
  """Compute the mean prediction of a Gaussian process.

  Args:
    op: Some vector operator that projects the data along the relevant
      directions, op(vec, dt) = M^{-1} @ (I - E^(-M dt)) @ vec
    g_td: A kernel relating training data with test data. The kernel should be
      an `np.ndarray` of shape [n_test * output_dim, n_train * output_dim] or
      [n_test, n_train].
    y_train: An `np.ndarray` of shape [n_train, output_dim] of targets for the
      training data.

  Returns:
    The mean prediction of the GP. `g_td @ op @ y_train`.
  """
  fl, ufl = _make_flatten_uflatten(g_td, y_train)

  mean_pred = op(fl(y_train))
  mean_pred = np.dot(g_td, mean_pred)
  return ufl(mean_pred)


def _mean_prediction_einsum(evecs, op_evals, g_td, y_train):
  """Einsum powered version of _mean_prediction."""
  fl, ufl = _make_flatten_uflatten(g_td, y_train)

  mean_pred = np.einsum(
      'lj,ji,i,ki,k...->l...',
      g_td, evecs, op_evals, evecs, fl(y_train), optimize=True)
  return ufl(mean_pred)


def _ntk_cov(op, ntk_td, nngp_dd, nngp_td, nngp_tt):
  """Compute the covariance in the ntk approximation."""
  # op(vec) here should compute \Theta^{-1} @ (I - e^{-\Theta dt}) @ vec
  # for the time dependent case and
  # op(vec) = \Theta^{-1} @ vec for the infinite time case.
  # below implements Equation 15 from https://arxiv.org/abs/1902.06720
  term_1 = op(np.transpose(ntk_td))
  cov = np.dot(nngp_dd, term_1)
  cov = np.dot(np.transpose(term_1), cov)
  term_2 = np.dot(ntk_td, op(np.transpose(nngp_td)))
  term_2 += np.transpose(term_2)
  cov += (-term_2 + nngp_tt)
  return cov


def _nngp_cov(op, g_td, g_tt):
  """Compute the covariance in the nngp approximation."""
  # op(vec) here should compute K^{-1} @ (I - e^{-2 K dt}) @ vec
  # for the time dependent case or
  # op(vec) = K^{-1} @ vec
  # for infinite time.
  # below implements Equation S23 from https://arxiv.org/abs/1902.06720
  cov = op(np.transpose(g_td))
  return g_tt - np.dot(g_td, cov)


def _make_expm1_fn(normalization):

  def expm1_fn(evals, dt):
    # Since our maxtrix really should be positive semidefinite,
    # we can threshold the eigenvalues to squash ones that are negative
    # for numerical reasons.
    return np.expm1(-np.maximum(evals, 0.) * dt / normalization)

  return expm1_fn


def _make_inv_expm1_fn(normalization):
  expm1_fn = _make_expm1_fn(normalization)

  def _inv_expm1_fn(evals, dt):
    return expm1_fn(evals, dt) / np.abs(evals)

  return _inv_expm1_fn


def _arr_is_on_cpu(x):
  # TODO(romann): revisit when https://github.com/google/jax/issues/1431 and
  # https://github.com/google/jax/issues/1432 are fixed.
  if hasattr(x, 'device_buffer'):
    return 'cpu' in str(x.device_buffer.device()).lower()

  if isinstance(x, np.ndarray):
    return True

  raise NotImplementedError(type(x))


def _is_on_cpu(x):
  return tree_all(tree_map(_arr_is_on_cpu, x))


def _jit_cpu(x):
  def jit_cpu(f):
    if _is_on_cpu(x):
      return jit(f, backend='cpu')
    return jit(f)
  return jit_cpu


def max_learning_rate(
    kdd: np.ndarray, num_outputs: int = -1, eps: float = 1e-12) -> float:
  r"""Computing the maximal feasible learning rate for infinite width NNs.

  The network is assumed to be trained using SGD or full-batch GD with mean
  squared loss. The loss is assumed to have the form
  :math:`1/(2 * batch_size * num_outputs) \|f(train_x) - train_y\|^2`. The
  maximal feasible learning rate is the largest `\eta` such that the operator
  :math:`(I - \eta / (batch_size * num_outputs) * NTK)` is a contraction, which
  is :math:`2 * batch_size * num_outputs * \lambda_max(NTK)`.

  Args:
    kdd: The analytic or empirical NTK of (train_x, train_x).
    num_outputs: The number of outputs of the neural network. If `kdd` is the
      analytic ntk, `num_outputs` must be provided. Otherwise `num_outputs=-1`
      and `num_outputs` is computed via the size of `kdd`.
    eps: A float to avoid zero divisor.

  Returns:
    The maximal feasible learning rate for infinite width NNs.
  """

  if kdd.ndim not in [2, 4]:
    raise ValueError('`kdd` must be a 2d or 4d tensor.')
  if kdd.ndim == 2 and num_outputs == -1:
    raise ValueError('`num_outputs` must be provided for theoretical kernel.')
  if kdd.ndim == 2:
    factor = kdd.shape[0] * num_outputs
  else:
    kdd = empirical.flatten_features(kdd)
    factor = kdd.shape[0]
  if kdd.shape[0] != kdd.shape[1]:
    raise ValueError('`kdd` must be a square matrix.')
  if _is_on_cpu(kdd):
    max_eva = osp.linalg.eigvalsh(
        kdd,
        eigvals=(kdd.shape[0] - 1, kdd.shape[0] - 1))[-1]
  else:
    max_eva = np.linalg.eigvalsh(kdd)[-1]
  lr = 2 * factor / (max_eva + eps)
  return lr
