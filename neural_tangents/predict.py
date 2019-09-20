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
import collections
import functools
from jax.api import grad
from jax.lib import xla_bridge
import jax.numpy as np
import jax.scipy as sp
from neural_tangents.utils import empirical
from neural_tangents.utils.kernel import Kernel
from scipy.integrate._ode import ode

Gaussian = collections.namedtuple('Gaussian', 'mean covariance')


def gradient_descent_mse(g_dd, y_train, g_td=None, diag_reg=0.):
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
      [n_test, n_train]. Note; g_td should have been created in the convention
      ker_fun(x_train, x_test, params).
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

  g_dd = _canonicalize_kernel_to_ntk(g_dd)
  g_td = _canonicalize_kernel_to_ntk(g_td)

  g_dd = empirical.flatten_features(g_dd)

  normalization = y_train.size
  output_dimension = y_train.shape[-1]
  expm1_func, inv_expm1_func = (_make_expm1_func(normalization),
                                _make_inv_expm1_func(normalization))

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
  expm1_dot_vec, inv_expm1_dot_vec = _eigenfuncs(g_dd_plus_reg,
                                                 (expm1_func, inv_expm1_func))

  if g_td is None:

    def train_predict(dt, fx=0.0):
      gx_train = fl(fx - y_train)
      dgx = expm1_dot_vec(gx_train, dt)
      return ufl(dgx) + y_train

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
        g_td should have been created in the convention g_td = ker_fun(x_test,
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


def gp_inference(ker_fun,
                 x_train,
                 y_train,
                 x_test,
                 diag_reg=0.,
                 mode='NNGP',
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
      only `mean` otherwise.

  Returns:
    Either a Gaussian(`mean`, `variance`) namedtuple or `mean` of the GP
    posterior.
  """
  if mode not in ['NNGP', 'NTK']:
    raise ValueError('The `mode` must be either `NNGP` or `NTK`.')

  kdd, ktd = ker_fun(x_train, None), ker_fun(x_test, x_train)
  if mode == 'NNGP':
    op = _inv_operator(kdd.nngp, diag_reg)
    pred_mean = _mean_prediction(op, ktd.nngp, y_train)
  else:
    op = _inv_operator(kdd.ntk, diag_reg)
    pred_mean = _mean_prediction(op, ktd.ntk, y_train)
  if not compute_var:
    return pred_mean

  ktt = ker_fun(x_test, None)
  if mode == 'NNGP':
    var = _nngp_var(op, ktd.nngp, ktt.nngp)
  else:
    var = _ntk_var(op, ktd.ntk, kdd.nngp, ktd.nngp, ktt.nngp)

  return Gaussian(pred_mean, var)


def gradient_descent_mse_gp(ker_fun,
                            x_train,
                            y_train,
                            x_test,
                            diag_reg=0.0,
                            mode='NTK',
                            compute_var=False):
  """Predicts the gaussian embedding induced by gradient descent with mse loss.

  This is equivalent to an infinite ensemble of networks after marginalizing
  out the initialization.

  Args:
    ker_fun: A kernel function that computes NNGP and NTK.
    x_train: A `np.ndarray`, representing the training data.
    y_train: A `np.ndarray`, representing the labels of the training data.
    x_test: A `np.ndarray`, representing the test data.
    diag_reg: A float, representing the strength of the regularization.
    mode: The mode of the Gaussian process, either 'NNGP' or `NTK`.
    compute_var: A boolean. If `True` computing both `mean` and `variance` and
      only `mean` otherwise.

  Returns:
    A function that predicts the gaussian parameters at t:
      prediction(t) -> Gaussian(mean, variance).
      If compute_var is False, only returns the mean.
  """
  if mode not in ['NNGP', 'NTK']:
    raise ValueError('The `mode` must be either `NNGP` or `NTK`.')

  kdd = ker_fun(x_train, None)
  ktd = ker_fun(x_test, x_train)
  ktt = ker_fun(x_test, None)
  normalization = y_train.size
  op_func = _make_inv_expm1_func(normalization)

  if mode == 'NNGP':
    k_dd_plus_reg = _add_diagonal_regularizer(kdd.nngp, diag_reg)
    evals, evecs = _eigh(k_dd_plus_reg)

    def prediction(t):
      """The Gaussian prediction at finite time for the NNGP."""
      op_evals = -op_func(evals, 2 * t)
      pred_mean = _mean_prediction_einsum(evecs, op_evals, ktd.nngp, y_train)
      if not compute_var:
        return pred_mean
      # inline the variance calculation with an einsum.
      var = ktt.nngp - np.einsum(
          'mj,ji,i,ki,lk->ml',
          ktd.nngp, evecs, op_evals, evecs, ktd.nngp, optimize=True)

      return Gaussian(pred_mean, var)

  else:  # mode == 'NTK'
    g_dd_plus_reg = _add_diagonal_regularizer(kdd.ntk, diag_reg)
    evals, evecs = _eigh(g_dd_plus_reg)

    def prediction(t):
      """The Gaussian prediction at finite time for the NTK."""
      op_evals = -op_func(evals, t)
      pred_mean = _mean_prediction_einsum(evecs, op_evals, ktd.ntk, y_train)
      if not compute_var:
        return pred_mean
      # inline the covariance calculation with einsum.
      var = np.einsum(
          'mj,ji,i,ki,lk->ml',
          kdd.nngp, evecs, op_evals, evecs, ktd.ntk, optimize=True)
      var -= 2. * np.transpose(ktd.nngp)
      var = np.einsum(
          'mj,ji,i,ki,kl->ml',
          ktd.ntk, evecs, op_evals, evecs, var, optimize=True)
      var = var + ktt.nngp

      return Gaussian(pred_mean, var)

  return prediction


## Utility functions


def _eigh(mat):
  """Platform specific eigh."""
  # TODO(schsam): Eventually, we may want to handle non-symmetric kernels for
  # e.g. masking. Additionally, once JAX supports eigh on TPU, we probably want
  # to switch to JAX's eigh.
  if xla_bridge.get_backend().platform == 'tpu':
    eigh = np.onp.linalg.eigh
  else:
    eigh = np.linalg.eigh

  return eigh(mat)


def _eigenfuncs(mat, funcs):
  """Build functions of a matrix in its eigenbasis.

  Args:
    mat: an n x n matrix
    funcs: a sequence of functions that add on the eigenvalues (evals, dt) ->
      modified_evals

  Returns:
    A tuple of functions that act as functions of the matrix mat
      acting on vectors:
        transform(vec, dt) = func(mat, dt) @ vec
  """
  evals, evecs = _eigh(mat)

  def transform(func):
    """Generates a transform given a function on the eigenvalues."""
    def _(vec, dt):
      return np.einsum(
          'ji,i,ki,k...->j...',
          evecs, func(evals, dt), evecs, vec, optimize=True)

    return _

  return tuple(transform(func) for func in funcs)


def _canonicalize_kernel_to_ntk(k):
  if k is None or isinstance(k, np.ndarray):
    return k
  if isinstance(k, Kernel):
    return k.ntk
  raise ValueError(
      'Expected kernel to either be a `Kernel`, a `np.ndarry`, or `None`. '
      'Found {}.'.format(type(k)))


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


def _mean_prediction(op, g_td, y_train):
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
    The mean prediction of the GP.  `g_td @ op @ y_train`.
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


def _ntk_var(op, ntk_td, nngp_dd, nngp_td, nngp_tt):
  """Compute the covariance in the ntk approximation."""
  # op(vec) here should compute \Theta^{-1} @ (I - e^{-\Theta dt}) @ vec
  # for the time dependent case and
  # op(vec) = \Theta^{-1} @ vec for the infinite time case.
  # below implements Equation 15 from 1902.06720
  var = op(np.transpose(ntk_td))
  var = np.dot(nngp_dd, var)
  var -= 2. * np.transpose(nngp_td)
  var = op(var)
  var = np.dot(ntk_td, var) + nngp_tt
  return var


def _nngp_var(op, g_td, g_tt):
  """Compute the covariance in the nngp approximation."""
  # op(vec) here should compute K^{-1} @ (I - e^{-2 K dt}) @ vec
  # for the time dependent case or
  # op(vec) = K^{-1} @ vec
  # for infinite time.
  # below implements Equation S23 from 1902.06720
  var = op(np.transpose(g_td))
  return g_tt - np.dot(g_td, var)


def _make_expm1_func(normalization):

  def expm1_func(evals, dt):
    # Since our maxtrix really should be positive semidefinite,
    # we can threshold the eigenvalues to squash ones that are negative
    # for numerical reasons.
    return np.expm1(-np.maximum(evals, 0.) * dt / normalization)

  return expm1_func


def _make_inv_expm1_func(normalization):
  expm1_func = _make_expm1_func(normalization)

  def _inv_expm1_func(evals, dt):
    return expm1_func(evals, dt) / np.abs(evals)

  return _inv_expm1_func