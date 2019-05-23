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

"""Code to linearize neural networks and compute empirical kernels.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from jax.api import grad
from jax.api import jacobian
from jax.api import jit
from jax.api import jvp
from jax.api import vjp

import jax.numpy as np

from jax.tree_util import tree_map
from jax.tree_util import tree_multimap

import numpy as onp

from scipy.integrate._ode import ode


def linearize(f, params):
  """Returns a function f_lin, the first order taylor approximation to f.

  Example:
    >>> # Compute the MSE of the first order Taylor series of a function.
    >>> f_lin = linearize(f, params)
    >>> mse = np.mean((f(new_params, x) - f_lin(new_params, x)) ** 2)

  Args:
    f: A function that we would like to linearize. It should have the signature
       f(params, inputs) where params and inputs are ndarrays and f should
       return an ndarray.
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


def _batch_kernel(kernel_fn, x1, x2, batch_size):
  """Takes a kernel function and computes it over a dataset in batches.

  Args:
    kernel_fn: A function that computes a kernel between two datasets,
               kernel_fn(x1, x2)
    x1: A first ndarray of inputs of shape [n1, ...] over which we would like to
        compute the kernel.
    x2: A second ndarray of inputs of shape [n2, ...] to use in the kernel
        computation. Can be None in which case x2 = x1.
    batch_size: The size of batches in which to split the data.

  Returns:
    An ndarray of size [n1 * output_dim, n2 * output_dim].
  """
  x1s = np.split(x1, range(batch_size, x1.shape[0], batch_size))
  if x2 is None:
    batches = len(x1s)
    # pylint: disable=g-complex-comprehension
    kernel = [[
        kernel_fn(xi, xj) for xj in x1s[:i + 1]] for i, xi in enumerate(x1s)]
    kernel = [[
        kernel[i][j]  if j <= i else
        np.transpose(kernel[j][i])
        for j in range(batches)]
              for i in range(batches)]
  else:
    x2s = np.split(x2, range(batch_size, x2.shape[0], batch_size))
    # pylint: disable=g-complex-comprehension
    kernel = [[kernel_fn(x1, x2) for x2 in x2s] for x1 in x1s]

  return np.vstack([np.hstack(k) for k in kernel])


def _compute_ntk(f, fx_dummy, params, x1, x2):
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
       f(params, inputs) and should return an ndarray of outputs with shape
       [|inputs|, output_dim].
    fx_dummy: A dummy evaluation of f on a single input that we use to
       instantiate an ndarray with the correct shape
       (aka [|inputs|, output_dim]).
       It should be possible at some point to use JAX's tracing mechanism to do
       this more efficiently.
    params: A set of parameters about which we would like to compute the neural
       tangent kernel. This should be any structure that can be mapped over by
       JAX's tree utilities.
    x1: A first ndarray of inputs, of shape [n1, ...], over which we would like
       to compute the NTK.
    x2: A second ndarray of inputs, of shape [n2, ...], over which we would like
       to compute the NTK.

  Returns:
    An ndarray containing the NTK with shape [n * output_dim, m * output_dim].
  """
  fx_dummy = np.concatenate([fx_dummy] * len(x2))
  output_dim = fx_dummy.shape[1]
  def dzdt(delta):
    _, dfdw = vjp(lambda p: f(p, x2), params)
    dfdw, = dfdw(delta)
    def z(t):
      p = tree_multimap(
          np.add, params, tree_map(lambda x: t * x, dfdw))
      return f(p, x1)
    _, dzdot = jvp(z, (0.0,), (1.0,))
    return dzdot
  theta = jacobian(dzdt)(fx_dummy)
  return np.reshape(theta, (len(x1) * output_dim, len(x2) * output_dim))


def ntk(f, batch_size=None):
  """Computes the neural tangent kernel.

  Example:
    >>> theta = ntk(f, batch_size=64)
    >>> k_dd = theta(params, x_train)
    >>> k_td = theta(params, x_test, x_train)

  Args:
    f: A function whose NTK we would like to compute.
    batch_size: int. Size of batches of inputs to use in computing the NTK.

  Returns:
    A function ntk_fun(params, x1, x2) that computes the NTK for a
    specific choice of parameters and inputs. Here x1 and x2 are ndarrays of
    shape [n1, ...] and [n2, ...] respectively and f(params, xi) has shape
    [ni, output_dim]; params should be a PyTree. The function will return an
    an ndarray of shape [n1 * output_dim, n2 * output_dim].
  """
  # NOTE(schsam): Can we move the jit outside?
  kernel_fn = jit(functools.partial(_compute_ntk, f))
  if batch_size is None or batch_size <= 0:

    def ntk_fun(params, x1, x2=None):
      if x2 is None: x2 = x1
      # @Optimization
      # Can we compute this using a shaped array or some other jax magic?
      fx_dummy = f(params, x2[:1])
      return kernel_fn(fx_dummy, params, x1, x2)
    return ntk_fun

  def ntk_fun_batched(params, x1, x2=None):
    if x2 is None:
      fx_dummy = f(params, x1[:1])
    else:
      fx_dummy = f(params, x2[:1])

    n = functools.partial(kernel_fn, fx_dummy, params)
    return _batch_kernel(n, x1, x2, batch_size)

  return ntk_fun_batched


def analytic_mse_predictor(g_dd, y_train, g_td=None):
  """Predicts the outcome of function space training with an MSE loss.

  Uses the analytic solution for gradient descent on an MSE loss in function
  space detailed in [*] given a Neural Tangent Kernel over the dataset. Given
  NTKs, this function will return a function that predicts the time evolution
  for function space points at arbitrary times. Note that times are continuous
  and are measured in units of the learning rate so t = learning_rate * steps.

  [*] https://arxiv.org/abs/1806.07572

  Example:
    >>> train_time = 1e-7
    >>> kernel_fn = ntk(f)
    >>> g_dd = compute_spectrum(kernel_fn(params, x_train))
    >>> g_td = kernel_fn(params, x_test, x_train)
    >>>
    >>> predict_fn = analytic_mse_predictor(g_dd, train_y, g_td)
    >>>
    >>> fx_train_initial = f(params, x_train)
    >>> fx_test_initial = f(params, x_test)
    >>>
    >>> fx_train_final, fx_test_final = predict_fn(
    >>>          fx_train_initial, fx_test_initial, train_time)

  Args:
    g_dd: A kernel on the training data. The kernel should be an ndarray of
      shape [n_train * output_dim, n_train * output_dim].
    y_train: An ndarray of shape [n_train, output_dim] of targets for the
      training data.
    g_td: A Kernel relating training data with test data. The kernel should be
      an ndarray of shape [n_test * output_dim, n_train * output_dim].
      Note: g_td should have been created in the convention
      kernel_fn(params, x_train, x_test).

  Returns:
    A function that predicts outputs after t = learning_rate * steps of
    training.

    If g_td is None:
      The function returned is predict(fx, t). Here fx is an ndarray of network
      outputs and has shape [n_train, output_dim], t is a floating point time.
      predict(fx, t) returns an ndarray of predictions of shape
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

  # TODO(schsam): Eventually, we may want to handle non-symmetric kernels for
  # e.g. masking. Additionally, once JAX supports eigh on GPU, we probably want
  # to switch to JAX's eigh.
  evals, evecs = onp.linalg.eigh(g_dd)
  ievecs = np.transpose(evecs)
  inverse = onp.linalg.inv(g_dd)
  normalization = g_dd.shape[1]

  def fl(fx):
    """Flatten outputs."""
    return np.reshape(fx, (-1,))

  def ufl(fx, output_dim):
    """Unflatten outputs."""
    return np.reshape(fx, (-1, output_dim))

  def predict(gx, dt):
    gx_ = np.diag(np.exp(-evals * dt / normalization))
    gx_ = np.dot(evecs, gx_)
    gx_ = np.dot(gx_, ievecs)
    gx_ = np.dot(gx_, gx)
    return gx_

  if g_td is None:
    return lambda fx, dt: \
        ufl(predict(fl(fx - y_train), dt), fx.shape[-1]) + y_train

  def predict_using_kernel(fx_train, fx_test, dt):
    output_dim = fx_train.shape[-1]
    gx_train = fl(fx_train - y_train)
    dgx = predict(gx_train, dt) - gx_train
    dfx = np.dot(inverse, dgx)
    dfx = np.dot(g_td, dfx)
    return ufl(dgx, output_dim) + fx_train, fx_test + ufl(dfx, output_dim)

  return predict_using_kernel


def gradient_descent_predictor(g_dd, y_train, loss, g_td=None):
  """Predicts the outcome of function space training using gradient descent.

  Solves the function space ODE for gradient descent with a given loss (detailed
  in [*]) given a Neural Tangent Kernel over the dataset. This function returns
  a function that predicts the time evolution for function space points at
  arbitrary times. Note that times are continuous and are measured in units of
  the learning rate so that t = learning_rate * steps.

  This function uses the scipy ode solver with the 'dopri5' algorithm.

  [*] https://arxiv.org/abs/1806.07572

  Example:
    >>> train_time = 1e-7
    >>> kernel_fn = ntk(f)
    >>> g_dd = compute_spectrum(kernel_fn(params, x_train))
    >>> g_td = kernel_fn(params, x_test, x_train)
    >>>
    >>> from jax.experimental import stax
    >>> cross_entropy = lambda fx, y_hat: -np.mean(stax.logsoftmax(fx) * y_hat)
    >>> predict_fn = gradient_descent_predictor(
    >>>                   g_dd, train_y, cross_entropy, g_td)
    >>>
    >>> fx_train_initial = f(params, x_train)
    >>> fx_test_initial = f(params, x_test)
    >>>
    >>> fx_train_final, fx_test_final = predict_fn(
    >>>          fx_train_initial, fx_test_initial, train_time)

  Args:
    g_dd: A Kernel on the training data. The kernel should be an ndarray of
      shape [n_train * output_dim, n_train * output_dim].
    y_train: An ndarray of shape [n_train, output_dim] of labels for the
      training data.
    loss: A loss function whose signature is loss(fx, y_hat) where fx is an
      ndarray of function space output_dim of the network and y_hat are
      targets.

      Note: the loss function should treat the batch and output dimensions
      symmetrically.
    g_td: A Kernel relating training data with test data. The kernel should be
      an ndarray of shape [n_test * output_dim, n_train * output_dim].

      Note: g_td should have been created in the convention
      kernel_fn(params, x_test, x_train).

  Returns:
    A function that predicts outputs after t = learning_rate * steps of
    training.

    If g_td is None:
      The function returned is predict(fx, t). Here fx is an ndarray of network
      outputs and has shape [n_train, output_dim], t is a floating point time.
      predict(fx, t) returns an ndarray of predictions of shape
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
  y_train = np.reshape(y_train, (-1))
  grad_loss = grad(functools.partial(loss, y_hat=y_train))

  def fl(fx):
    """Flatten outputs."""
    return np.reshape(fx, (-1,))

  def ufl(fx, output_dim):
    """Unflatten outputs."""
    return np.reshape(fx, (-1, output_dim))

  if g_td is None:
    dfx_dt = lambda unused_t, fx: -np.dot(g_dd, grad_loss(fx))
    def predict(fx, dt):
      r = ode(dfx_dt).set_integrator('dopri5')
      r.set_initial_value(fl(fx), 0)
      r.integrate(dt)

      return ufl(r.y, fx.shape[-1])
  else:
    def dfx_dt(unused_t, fx, train_size):
      fx_train = fx[:train_size]
      dfx_train = -np.dot(g_dd, grad_loss(fx_train))
      dfx_test = -np.dot(g_td, grad_loss(fx_train))
      return np.concatenate((dfx_train, dfx_test), axis=0)

    def predict(fx_train, fx_test, dt):
      r = ode(dfx_dt).set_integrator('dopri5')

      fx = fl(np.concatenate((fx_train, fx_test), axis=0))
      train_size, output_dim = fx_train.shape
      r.set_initial_value(fx, 0).set_f_params(train_size * output_dim)
      r.integrate(dt)
      fx = ufl(r.y, output_dim)

      return fx[:train_size], fx[train_size:]

  return predict


def momentum_predictor(
    g_dd, y_train, loss, learning_rate, g_td=None, momentum=0.9):
  r"""Predicts the outcome of function space training using momentum.

  Solves the function space ODE for momentum with a given loss (detailed
  in [*]) given a Neural Tangent Kernel over the dataset. This function returns
  a triplet of functions that initialize state variables, predicts the time
  evolution for function space points at arbitrary times and retrieves the
  function-space outputs from the state. Note that times are continuous and are
  measured in units of the learning rate so that
  t = \sqrt(learning_rate) * steps.

  Note: this solves a continuous version of standard momentum instead of
  Nesterov momentum.

  This function uses the scipy ode solver with the 'dopri5' algorithm.

  [*] https://arxiv.org/abs/1806.07572

  Example:
    >>> train_time = 1e-7
    >>> learning_rate = 1e-2
    >>>
    >>> kernel_fn = ntk(f)
    >>> g_dd = compute_spectrum(kernel_fn(params, x_train))
    >>> g_td = kernel_fn(params, x_test, x_train)
    >>>
    >>> from jax.experimental import stax
    >>> cross_entropy = lambda fx, y_hat: -np.mean(stax.logsoftmax(fx) * y_hat)
    >>> init_fn, predict_fn, get_fn = momentum_predictor(
    >>>                   g_dd, y_train, cross_entropy, learning_rate, g_td)
    >>>
    >>> fx_train_initial = f(params, x_train)
    >>> fx_test_initial = f(params, x_test)
    >>>
    >>> lin_state = init_fn(fx_train_initial, fx_test_initial)
    >>> lin_state = predict_fn(lin_state, train_time)
    >>> fx_train_final, fx_test_final = get_fn(lin_state)

  Args:
    g_dd: Kernel on the training data. The kernel should be an ndarray of shape
      [n_train * output_dim, n_train * output_dim].
    y_train: An ndarray of shape [n_train, output_dim] of labels for the
       training data.
    loss: A loss function whose signature is loss(fx, y_hat) where fx an ndarray
      of function space outputs of the network and y_hat are labels.

      Note: the loss function should treat the batch and output dimensions
      symmetrically.
    learning_rate:  A float specifying the learning rate.
    g_td: Kernel relating training data with test data. Should be an ndarray of
      shape [n_test * output_dim, n_train * output_dim]. Note: g_td should
      have been created in the convention
      g_td = kernel_fn(params, x_test, x_train).
    momentum: float specifying the momentum.

  Returns:
    Functions to predicts outputs after t = \sqrt(learning_rate) * steps of
    training. Generically three functions are returned, an init_fn that creates
    auxiliary velocity variables needed for optimization and packs them into
    a state variable, a predict_fn that computes the time-evolution of the state
    for some dt, and a get_fn that extracts the predictions from the state.

    If g_td is None:
      init_fn(fx_train): Takes a single ndarray of shape [n_train, output_dim]
        and returns a tuple containing the output_dim as an int and an ndarray
        of shape [2 * n_train * output_dim].

      predict_fn(state, dt): Takes a state described above and a floating point
        time. Returns a new state with the same type and shape.

      get_fn(state): Takes a state and returns an ndarray of shape
        [n_train, output_dim].

    If g_td is not None:
      init_fn(fx_train, fx_test): Takes two ndarrays of shape
        [n_train, output_dim] and [n_test, output_dim] respectively. Returns a
        tuple with an int giving 2 * n_train * output_dim, an int containing the
        output_dim, and an ndarray of shape
        [2 * (n_train + n_test) * output_dim].

      predict_fn(state, dt): Takes a state described above and a floating point
        time. Returns a new state with the same type and shape.

      get_fn(state): Takes a state and returns two ndarray of shape
        [n_train, output_dim] and [n_test, output_dim] respectively.
  """
  momentum = (momentum - 1.0) / np.sqrt(learning_rate)
  y_train = np.reshape(y_train, (-1))
  grad_loss = grad(functools.partial(loss, y_hat=y_train))

  def fl(fx):
    """Flatten outputs."""
    return np.reshape(fx, (-1,))

  def ufl(fx, output_dim):
    """Unflatten outputs."""
    return np.reshape(fx, (-1, output_dim))

  if g_td is None:
    def dr_dt(unused_t, r):
      fx, qx = np.split(r, 2)
      dfx = qx
      dqx = momentum * qx - np.dot(g_dd, grad_loss(fx))
      return np.concatenate((dfx, dqx), axis=0)

    def init_fn(fx_train):
      output_dim = fx_train.shape[-1]
      fx_train = fl(fx_train)
      qx_train = np.zeros_like(fx_train)
      return output_dim, np.concatenate((fx_train, qx_train), axis=0)

    def predict_fn(state, dt):
      output_dim, state = state

      solver = ode(dr_dt).set_integrator('dopri5')
      solver.set_initial_value(state, 0)
      solver.integrate(dt)

      return output_dim, solver.y

    def get_fn(state):
      output_dim, state = state
      return ufl(np.split(state, 2)[0], output_dim)

  else:
    def dr_dt(unused_t, r, train_size):
      train, test = r[:train_size], r[train_size:]
      fx_train, qx_train = np.split(train, 2)
      _, qx_test = np.split(test, 2)
      dfx_train = qx_train
      dqx_train = momentum * qx_train - np.dot(g_dd, grad_loss(fx_train))
      dfx_test = qx_test
      dqx_test = momentum * qx_test - np.dot(g_td, grad_loss(fx_train))
      return np.concatenate((dfx_train, dqx_train, dfx_test, dqx_test), axis=0)

    def init_fn(fx_train, fx_test):
      train_size, output_dim = fx_train.shape
      fx_train, fx_test = fl(fx_train), fl(fx_test)
      qx_train = np.zeros_like(fx_train)
      qx_test = np.zeros_like(fx_test)
      return (
          2 * train_size * output_dim, output_dim,
          np.concatenate((fx_train, qx_train, fx_test, qx_test), axis=0))

    def predict_fn(state, dt):
      train_size, output_dim, state = state
      solver = ode(dr_dt).set_integrator('dopri5')
      solver.set_initial_value(state, 0).set_f_params(train_size)
      solver.integrate(dt)

      return train_size, output_dim, solver.y

    def get_fn(state):
      train_size, output_dim, state = state
      train, test = state[:train_size], state[train_size:]
      return (
          ufl(np.split(train, 2)[0], output_dim),
          ufl(np.split(test, 2)[0], output_dim)
          )

  return init_fn, predict_fn, get_fn
