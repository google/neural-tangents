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

"""Tests for the Neural Tangents library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import operator

from absl.testing import absltest
from absl.testing import parameterized

from jax import test_util as jtu
from jax.api import grad
from jax.api import jacobian
from jax.config import config as jax_config

from jax.experimental import optimizers as opt

import jax.numpy as np
import jax.random as random
from jax.tree_util import tree_multimap
from jax.tree_util import tree_reduce

from neural_tangents import tangents


jax_config.parse_flags_with_absl()
FLAGS = jax_config.FLAGS


MATRIX_SHAPES = [(3, 3), (4, 4)]
OUTPUT_LOGITS = [1, 2, 3]


@opt.optimizer
def momentum(learning_rate, momentum=0.9):
  """A standard momentum optimizer for testing.

  Different from `jax.experimental.optimizers.momentum` (Nesterov).
  """
  learning_rate = opt.make_schedule(learning_rate)
  def init_fun(x0):
    v0 = np.zeros_like(x0)
    return x0, v0
  def update_fun(i, g, state):
    x, velocity = state
    velocity = momentum * velocity + g
    x = x - learning_rate(i) * velocity
    return x, velocity
  def get_params(state):
    x, _ = state
    return x
  return init_fun, update_fun, get_params


class NeuralTangentsTest(jtu.JaxTestCase):
  @parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name': '_{}'.format(shape),
          'shape': shape
      } for shape in MATRIX_SHAPES))
  def testLinearization(self, shape):
    def f(w, x):
      return np.dot(w, x)

    key = random.PRNGKey(0)
    key, split = random.split(key)
    w0 = random.normal(split, shape)
    key, split = random.split(key)
    x = random.normal(split, (shape[-1],))

    f_lin = tangents.linearize(f, w0)

    for _ in range(10):
      key, split = random.split(key)
      w = random.normal(split, shape)
      self.assertAllClose(f(w, x), f_lin(w, x), True)

  @parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name': '_shape_{}_logits_{}'.format(shape, out_logits),
          'shape': shape,
          'out_logits': out_logits
      } for shape in MATRIX_SHAPES for out_logits in OUTPUT_LOGITS))
  def testNTKAgainstDirect(self, shape, out_logits):

    def sum_and_contract(j1, j2):
      def contract(x, y):
        param_count = int(np.prod(x.shape[2:]))
        x = np.reshape(x, (-1, param_count))
        y = np.reshape(y, (-1, param_count))
        return np.dot(x, np.transpose(y))

      return tree_reduce(operator.add, tree_multimap(contract, j1, j2))

    def ntk_direct(f, params, x1, x2):
      jac_fn = jacobian(f)
      j1 = jac_fn(params, x1)

      if x2 is None:
        j2 = j1
      else:
        j2 = jac_fn(params, x2)

      return sum_and_contract(j1, j2)

    key = random.PRNGKey(0)
    data_self = random.normal(key, shape)
    data_other = random.normal(key, shape)

    key, w_split, b_split = random.split(key, 3)
    params = (random.normal(w_split, (shape[-1], out_logits)),
              random.normal(b_split, (out_logits,)))

    def f(params, x):
      w, b = params
      return np.dot(x, w) / shape[-1] + b

    g_fn = tangents.ntk(f)

    g = g_fn(params, data_self)
    g_direct = ntk_direct(f, params, data_self, data_self)
    self.assertAllClose(g, g_direct, check_dtypes=False)

    g = g_fn(params, data_other, data_self)
    g_direct = ntk_direct(f, params, data_other, data_self)
    self.assertAllClose(g, g_direct, check_dtypes=False)

  @parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name': '_shape_{}_logits_{}'.format(shape, out_logits),
          'shape': shape,
          'out_logits': out_logits
      } for shape in MATRIX_SHAPES for out_logits in OUTPUT_LOGITS))
  def testNTKBatched(self, shape, out_logits):

    key = random.PRNGKey(0)
    data_self = random.normal(key, shape)
    data_other = random.normal(key, shape)

    key, w_split, b_split = random.split(key, 3)
    params = (random.normal(w_split, (shape[-1], out_logits)),
              random.normal(b_split, (out_logits,)))

    def f(params, x):
      w, b = params
      return np.dot(x, w) / shape[-1] + b

    g_fn = tangents.ntk(f)
    g_batched_fn = tangents.ntk(f, batch_size=2)

    g = g_fn(params, data_self)
    g_batched = g_batched_fn(params, data_self)
    self.assertAllClose(g, g_batched, check_dtypes=False)

    g = g_fn(params, data_other, data_self)
    g_batched = g_batched_fn(params, data_other, data_self)
    self.assertAllClose(g, g_batched, check_dtypes=False)

  @parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name': '_shape_{}_logits_{}'.format(shape, out_logits),
          'shape': shape,
          'out_logits': out_logits
      } for shape in MATRIX_SHAPES for out_logits in OUTPUT_LOGITS))
  def testNTKMSEPrediction(self, shape, out_logits):

    key = random.PRNGKey(0)

    key, split = random.split(key)
    data_train = random.normal(split, shape)

    key, split = random.split(key)
    data_labels = np.array(
        random.bernoulli(split, shape=(shape[0], out_logits)), np.float32)

    key, split = random.split(key)
    data_test = random.normal(split, shape)

    key, w_split, b_split = random.split(key, 3)
    params = (random.normal(w_split, (shape[-1], out_logits)),
              random.normal(b_split, (out_logits,)))

    def f(params, x):
      w, b = params
      return np.dot(x, w) / shape[-1] + b

    # Regress to an MSE loss.
    loss = lambda params, x: \
        0.5 * np.mean((f(params, x) - data_labels) ** 2)

    theta = tangents.ntk(f)
    g_dd = theta(params, data_train)
    g_td = theta(params, data_test, data_train)

    predictor = tangents.analytic_mse_predictor(g_dd, data_labels, g_td)

    step_size = 1.0
    train_time = 100.0
    steps = int(train_time / step_size)

    opt_init, opt_update, get_params = opt.sgd(step_size)
    opt_state = opt_init(params)

    fx_initial_train = f(params, data_train)
    fx_initial_test = f(params, data_test)

    fx_pred_train, fx_pred_test = predictor(
        fx_initial_train, fx_initial_test, 0.0)

    # NOTE(schsam): I think at the moment stax always generates 32-bit results
    # since the weights are explicitly cast to float32.
    self.assertAllClose(fx_initial_train, fx_pred_train, False)
    self.assertAllClose(fx_initial_test, fx_pred_test, False)

    for i in range(steps):
      params = get_params(opt_state)
      opt_state = opt_update(i, grad(loss)(params, data_train), opt_state)

    params = get_params(opt_state)
    fx_train = f(params, data_train)
    fx_test = f(params, data_test)

    fx_pred_train, fx_pred_test = predictor(
        fx_initial_train, fx_initial_test, train_time)

    fx_disp_train = np.sqrt(np.mean((fx_train - fx_initial_train) ** 2))
    fx_disp_test = np.sqrt(np.mean((fx_test - fx_initial_test) ** 2))

    fx_error_train = (fx_train - fx_pred_train) / fx_disp_train
    fx_error_test = (fx_test - fx_pred_test) / fx_disp_test

    self.assertAllClose(
        fx_error_train, np.zeros_like(fx_error_train), False, 0.1, 0.1)
    self.assertAllClose(
        fx_error_test, np.zeros_like(fx_error_test), False, 0.1, 0.1)

  @parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name': '_shape_{}_logits_{}'.format(shape, out_logits),
          'shape': shape,
          'out_logits': out_logits,
      } for shape in MATRIX_SHAPES for out_logits in OUTPUT_LOGITS[1:]))
  def testNTKGDPrediction(self, shape, out_logits):

    key = random.PRNGKey(1)

    key, split = random.split(key)
    data_train = random.normal(split, shape)

    key, split = random.split(key)
    label_ids = random.randint(split, (shape[0],), 0, out_logits)
    data_labels = np.eye(out_logits)[label_ids]

    key, split = random.split(key)
    data_test = random.normal(split, shape)

    key, w_split, b_split = random.split(key, 3)
    params = (random.normal(w_split, (shape[-1], out_logits)),
              random.normal(b_split, (out_logits,)))

    def f(params, x):
      w, b = params
      return np.dot(x, w) / shape[-1] + b

    loss = lambda y, y_hat: 0.5 * np.mean((y - y_hat) ** 2)
    grad_loss = grad(lambda params, x: loss(f(params, x), data_labels))

    theta = tangents.ntk(f)
    g_dd = theta(params, data_train)
    g_td = theta(params, data_test, data_train)

    predictor = tangents.gradient_descent_predictor(
        g_dd, data_labels, loss, g_td)

    step_size = 1.0
    train_time = 100.0
    steps = int(train_time / step_size)

    opt_init, opt_update, get_params = opt.sgd(step_size)
    opt_state = opt_init(params)

    fx_initial_train = f(params, data_train)
    fx_initial_test = f(params, data_test)

    fx_pred_train, fx_pred_test = predictor(
        fx_initial_train, fx_initial_test, 0.0)

    # NOTE(schsam): I think at the moment stax always generates 32-bit results
    # since the weights are explicitly cast to float32.
    self.assertAllClose(fx_initial_train, fx_pred_train, False)
    self.assertAllClose(fx_initial_test, fx_pred_test, False)

    for i in range(steps):
      params = get_params(opt_state)
      opt_state = opt_update(i, grad_loss(params, data_train), opt_state)

    params = get_params(opt_state)
    fx_train = f(params, data_train)
    fx_test = f(params, data_test)

    fx_pred_train, fx_pred_test = predictor(
        fx_initial_train, fx_initial_test, train_time)

    # Put errors in units of RMS distance of the function values during
    # optimization.
    fx_disp_train = np.sqrt(np.mean((fx_train - fx_initial_train) ** 2))
    fx_disp_test = np.sqrt(np.mean((fx_test - fx_initial_test) ** 2))

    fx_error_train = (fx_train - fx_pred_train) / fx_disp_train
    fx_error_test = (fx_test - fx_pred_test) / fx_disp_test

    self.assertAllClose(
        fx_error_train, np.zeros_like(fx_error_train), False, 0.1, 0.1)
    self.assertAllClose(
        fx_error_test, np.zeros_like(fx_error_test), False, 0.1, 0.1)

  @parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name': '_shape_{}_logits_{}'.format(shape, out_logits),
          'shape': shape,
          'out_logits': out_logits,
      } for shape in MATRIX_SHAPES for out_logits in OUTPUT_LOGITS[1:]))
  def testNTKMomentumPrediction(self, shape, out_logits):

    key = random.PRNGKey(1)

    key, split = random.split(key)
    data_train = random.normal(split, shape)

    key, split = random.split(key)
    label_ids = random.randint(split, (shape[0],), 0, out_logits)
    data_labels = np.eye(out_logits)[label_ids]

    key, split = random.split(key)
    data_test = random.normal(split, shape)

    key, w_split, b_split = random.split(key, 3)
    params = (random.normal(w_split, (shape[-1], out_logits)),
              random.normal(b_split, (out_logits,)))

    def f(params, x):
      w, b = params
      return np.dot(x, w) / shape[-1] + b

    loss = lambda y, y_hat: 0.5 * np.mean((y - y_hat) ** 2)
    grad_loss = grad(lambda params, x: loss(f(params, x), data_labels))

    theta = tangents.ntk(f)
    g_dd = theta(params, data_train)
    g_td = theta(params, data_test, data_train)

    step_size = 1.0
    train_time = 100.0
    steps = int(train_time / np.sqrt(step_size))

    init_fn, predict_fn, get_fn = tangents.momentum_predictor(
        g_dd, data_labels, loss, step_size, g_td)

    opt_init, opt_update, get_params = momentum(step_size, 0.9)
    opt_state = opt_init(params)

    fx_initial_train = f(params, data_train)
    fx_initial_test = f(params, data_test)

    lin_state = init_fn(fx_initial_train, fx_initial_test)

    for i in range(steps):
      params = get_params(opt_state)
      opt_state = opt_update(i, grad_loss(params, data_train), opt_state)

    params = get_params(opt_state)
    fx_train = f(params, data_train)
    fx_test = f(params, data_test)

    lin_state = predict_fn(lin_state, train_time)

    fx_pred_train, fx_pred_test = get_fn(lin_state)

    # Put errors in units of RMS distance of the function values during
    # optimization.
    fx_disp_train = np.sqrt(np.mean((fx_train - fx_initial_train) ** 2))
    fx_disp_test = np.sqrt(np.mean((fx_test - fx_initial_test) ** 2))

    fx_error_train = (fx_train - fx_pred_train) / fx_disp_train
    fx_error_test = (fx_test - fx_pred_test) / fx_disp_test

    self.assertAllClose(
        fx_error_train, np.zeros_like(fx_error_train), False, 0.1, 0.1)
    self.assertAllClose(
        fx_error_test, np.zeros_like(fx_error_test), False, 0.1, 0.1)


if __name__ == '__main__':
  absltest.main()
