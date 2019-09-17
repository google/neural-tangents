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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
from jax.api import grad
from jax.lib import xla_bridge
import jax.numpy as np
import jax.scipy as sp
from neural_tangents.utils import empirical
from neural_tangents.utils.kernel import Kernel
from scipy.integrate._ode import ode


def gradient_descent_mse(g_dd, y_train, g_td=None):
  """Predicts the outcome of function space gradient descent training on MSE.

  Analytically solves for the continuous-time version of gradient descent.

  Uses the analytic solution for gradient descent on an MSE loss in function
  space detailed in [*] given a Neural Tangent Kernel over the dataset. Given
  NTKs, this function will return a function that predicts the time evolution
  for function space points at arbitrary times. Note that times are continuous
  and are measured in units of the learning rate so t = learning_rate * steps.

  [*] https://arxiv.org/abs/1806.07572

  Example:
    ```python
    >>> from neural_tangents import predict
    >>>
    >>> train_time = 1e-7
    >>> ker_fun = empirical(f)
    >>> g_td = ker_fun(x_test, x_train, params)
    >>>
    >>> predict_fn = predict.gradient_descent_mse(g_dd, y_train, g_td)
    >>>
    >>> fx_train_initial = f(params, x_train)
    >>> fx_test_initial = f(params, x_test)
    >>>
    >>> fx_train_final, fx_test_final = predict_fn(
    >>>          fx_train_initial, fx_test_initial, train_time)
    ```

  Args:
    g_dd: A kernel on the training data. The kernel should be an `np.ndarray` of
      shape [n_train * output_dim, n_train * output_dim] or [n_train, n_train].
      In the latter case, the kernel is assumed to be block diagonal over the
      logits.
    y_train: A `np.ndarray` of shape [n_train, output_dim] of targets for the
      training data.
    g_td: A Kernel relating training data with test data. The kernel should be
      an `np.ndarray` of shape [n_test * output_dim, n_train * output_dim] or
      [n_test, n_train].
      Note: g_td should have been created in the convention ker_fun(x_train,
        x_test, params).

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

  g_dd = _canonicalize_kernel_to_ntk(g_dd)
  g_td = _canonicalize_kernel_to_ntk(g_td)

  g_dd = empirical.flatten_features(g_dd)

  # TODO(schsam): Eventually, we may want to handle non-symmetric kernels for
  # e.g. masking. Additionally, once JAX supports eigh on TPU, we probably want
  # to switch to JAX's eigh.
  if xla_bridge.get_backend().platform == 'tpu':
    eigh = np.onp.linalg.eigh
  else:
    eigh = np.linalg.eigh

  evals, evecs = eigh(g_dd)
  ievecs = np.transpose(evecs)

  normalization = y_train.size
  output_dimension = y_train.shape[-1]

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

  def predict(gx, dt):
    gx_ = np.diag(np.exp(-evals * dt / normalization))
    gx_ = np.dot(evecs, gx_)
    gx_ = np.dot(gx_, ievecs)
    gx_ = np.dot(gx_, gx)
    return gx_

  if g_td is None:
    return lambda dt, fx=0.: ufl(predict(fl(fx - y_train), dt)) + y_train

  g_td = empirical.flatten_features(g_td)
  mevals = np.diag(1.0 / evals)
  inverse = np.dot(np.dot(evecs, mevals), ievecs)

  def predict_using_kernel(dt, fx_train=0., fx_test=0.):
    gx_train = fl(fx_train - y_train)
    dgx = predict(gx_train, dt) - gx_train
    dfx = np.dot(inverse, dgx)
    dfx = np.dot(g_td, dfx)
    return ufl(dgx) + fx_train, fx_test + ufl(dfx)

  return predict_using_kernel


def gradient_descent(g_dd, y_train, loss, g_td=None):
  """Predicts the outcome of function space gradient descent training on `loss`.

  Solves for continuous-time gradient descent using an ODE solver.

  Solves the function space ODE for continuous gradient descent with a given
  loss (detailed in [*]) given a Neural Tangent Kernel over the dataset. This
  function returns a function that predicts the time evolution for function
  space points at arbitrary times. Note that times are continuous and are
  measured in units of the learning rate so that t = learning_rate * steps.

  This function uses the scipy ode solver with the 'dopri5' algorithm.

  [*] https://arxiv.org/abs/1806.07572

  Example:
    ```python
    >>> from jax.experimental import stax
    >>> from neural_tangents import predict
    >>>
    >>> train_time = 1e-7
    >>> ker_fun = empirical(f)
    >>> g_td = ker_fun(x_test, x_train, params)
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
    ```
  Args:
    g_dd: A Kernel on the training data. The kernel should be an `np.ndarray` of
      shape [n_train * output_dim, n_train * output_dim] or [n_train, n_train].
      In the latter case it is assumed that the kernel is block diagonal over
      the logits.
    y_train: A `np.ndarray` of shape [n_train, output_dim] of labels for the
      training data.
    loss: A loss function whose signature is loss(fx, y_hat) where fx is an
      `np.ndarray` of function space output_dim of the network and y_hat are
      targets. Note: the loss function should treat the batch and output
      dimensions symmetrically.
    g_td: A Kernel relating training data with test data. The kernel should be
      an `np.ndarray` of shape [n_test * output_dim, n_train * output_dim] or
      [n_test, n_train]. Note: g_td should have been created in the convention
      ker_fun(x_test, x_train, params).

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

  g_dd = _canonicalize_kernel_to_ntk(g_dd)
  g_td = _canonicalize_kernel_to_ntk(g_td)

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


def momentum(g_dd, y_train, loss, learning_rate, g_td=None, momentum=0.9):
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

  [*] https://arxiv.org/abs/1806.07572

  Example:
    ```python
    >>> train_time = 1e-7
    >>> learning_rate = 1e-2
    >>>
    >>> ker_fun = empirical(f)
    >>> g_td = ker_fun(x_test, x_train, params)
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
    ```python

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
      g_td should have been created in the convention
      g_td = ker_fun(x_test, x_train, params).
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
  g_dd = _canonicalize_kernel_to_ntk(g_dd)
  g_td = _canonicalize_kernel_to_ntk(g_td)

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
      state = state

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


def _canonicalize_kernel_to_ntk(k):
  if k is None or isinstance(k, np.ndarray):
    return k
  if isinstance(k, Kernel):
    return k.ntk
  raise ValueError(
      'Expected kernel to either be a `Kernel`, a `np.ndarry`, or `None`. '
      'Found {}.'.format(type(k)))


def gp_inference(ker_fun, x_train, y_train, x_test, diag_reg=0., mode='NNGP',
                 compute_var=False):
  """Compute the mean and variance of the `posterior` of NNGP and NTK.

  Args:
    ker_fun: A kernel function that computes NNGP and NTK.
    x_train: A `np.ndarray`, representing the training data.
    y_train: A `np.ndarray`, representing the labels of the training data.
    x_test: A `np.ndarray`, representing the test data.
    diag_reg: A float, representing the strength of the regularization.
    mode: The mode of the Gaussian process, either 'NNGP' or `NTK`.
    compute_var: A boolean. If `True` computing both `mean` and `variance` and
      only `mean` othorwise.
  Returns:
    Either `mean, variance` or `mean` of the GP posterior.
  """
  if mode not in ['NNGP', 'NTK']:
    raise ValueError('The `mode` must be either `NNGP` or `NTK`.')

  kdd, ktd = ker_fun(x_train, None), ker_fun(x_test, x_train)
  if mode == 'NNGP':
    pred_mean = _mean_prediction(kdd.nngp, ktd.nngp, y_train, diag_reg)
  else:
    pred_mean = _mean_prediction(kdd.ntk, ktd.ntk, y_train, diag_reg)
  if not compute_var:
    return pred_mean

  ktt = ker_fun(x_test, None)
  if mode == 'NNGP':
    var = _nngp_var(kdd.nngp, ktd.nngp, ktt.nngp, diag_reg)
  else:
    var = _ntk_var(kdd.ntk, ktd.ntk, kdd.nngp, ktd.nngp, ktt.nngp, diag_reg)

  return pred_mean, var


def _add_diagonal_regularizer(covariance, diag_reg=0.):
  dimension = covariance.shape[0]
  reg = np.trace(covariance) / dimension
  return covariance + diag_reg * reg * np.eye(dimension)


def _mean_prediction(g_dd, g_td, y_train, diag_reg=0.):
  """Compute the mean prediction of a Gaussian process.

  Args:
    g_dd: A kernel on the training data. The kernel should be an `np.ndarray` of
      shape [n_train * output_dim, n_train * output_dim] or [n_train, n_train].
      In the latter case, the kernel is assumed to be block diagonal over the
      logits.
    g_td: A kernel relating training data with test data. The kernel should be
      an `np.ndarray` of shape [n_test * output_dim, n_train * output_dim] or
      [n_test, n_train].
    y_train: An `np.ndarray` of shape [n_train, output_dim] of targets for the
      training data.
    diag_reg: strength of regularization.
  Returns:
    The mean prediction of the GP. When `diag_reg=0.`, returns
    `g_td * g_dd^{-1} * y_train`.
  """
  output_dimension = y_train.shape[-1]
  def fl(fx):
    """Flatten outputs."""
    return np.reshape(fx, (-1,))

  def ufl(fx):
    """Unflatten outputs."""
    return np.reshape(fx, (-1, output_dimension))

  if y_train.size > g_dd.shape[-1]:
    out_dim, ragged = divmod(y_train.size, g_dd.shape[-1])
    if ragged or out_dim != output_dimension:
      raise ValueError('The batch size of `y_train` must be the same as the'
                       ' last dimension of `g_dd`')
    fl = lambda x: x
    ufl = lambda x: x
  g_dd_plus_reg = _add_diagonal_regularizer(g_dd, diag_reg)
  mean_pred = sp.linalg.solve(g_dd_plus_reg, fl(y_train), sym_pos=True)
  mean_pred = np.dot(g_td, mean_pred)
  return ufl(mean_pred)


def _ntk_var(ntk_dd, ntk_td, nngp_dd, nngp_td, nngp_tt, diag_reg=0.):
  ntk_dd_plus_reg = _add_diagonal_regularizer(ntk_dd, diag_reg)
  var = sp.linalg.solve(ntk_dd_plus_reg, np.transpose(ntk_td), sym_pos=True)
  var = np.dot(nngp_dd, var)
  var -= 2. * np.transpose(nngp_td)
  var = sp.linalg.solve(ntk_dd_plus_reg, var, sym_pos=True)
  var = np.dot(ntk_td, var) + nngp_tt
  return var


def _nngp_var(g_dd, g_td, g_tt, diag_reg=0.):
  g_dd_plus_reg = _add_diagonal_regularizer(g_dd, diag_reg)
  var = sp.linalg.solve(g_dd_plus_reg, np.transpose(g_td), sym_pos=True)
  return g_tt - np.dot(g_td, var)

