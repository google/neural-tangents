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

"""Tests for `utils/predict.py`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from functools import partial
from jax import test_util as jtu
from jax.api import grad
from jax.api import jit
from jax.config import config
from jax.experimental import optimizers
from jax.lib import xla_bridge
import jax.numpy as np
import jax.random as random
from neural_tangents import predict
from neural_tangents import stax
from neural_tangents.utils import empirical


config.parse_flags_with_absl()


MATRIX_SHAPES = [(3, 3), (4, 4)]
OUTPUT_LOGITS = [1, 2, 3]

RTOL = 0.1
ATOL = 0.1

if not config.read('jax_enable_x64'):
  RTOL = 0.2
  ATOL = 0.2


FLAT = 'FLAT'
POOLING = 'POOLING'

# TODO(schsam): Add a pooling test when multiple inputs are supported in
# Conv + Pooling.
TRAIN_SHAPES = [(4, 4), (4, 8), (8, 8), (6, 4, 4, 3)]
TEST_SHAPES = [(2, 4), (6, 8), (16, 8), (2, 4, 4, 3)]
NETWORK = [FLAT, FLAT, FLAT, FLAT]
OUTPUT_LOGITS = [1, 2, 3]

CONVOLUTION_CHANNELS = 256


def _build_network(input_shape, network, out_logits):
  if len(input_shape) == 1:
    assert network == 'FLAT'
    return stax.serial(
        stax.Dense(4096, W_std=1.2, b_std=0.05),
        stax.Erf(),
        stax.Dense(out_logits, W_std=1.2, b_std=0.05))
  elif len(input_shape) == 3:
    if network == 'POOLING':
      return stax.serial(
          stax.Conv(CONVOLUTION_CHANNELS, (3, 3), W_std=2.0, b_std=0.05),
          stax.GlobalAvgPool(), stax.Dense(out_logits, W_std=2.0, b_std=0.05))
    elif network == 'FLAT':
      return stax.serial(
          stax.Conv(CONVOLUTION_CHANNELS, (3, 3), W_std=2.0, b_std=0.05),
          stax.Flatten(), stax.Dense(out_logits, W_std=2.0, b_std=0.05))
    else:
      raise ValueError('Unexpected network type found: {}'.format(network))
  else:
    raise ValueError('Expected flat or image test input.')


def _empirical_kernel(key, input_shape, network, out_logits):
  init_fn, f, _ = _build_network(input_shape, network, out_logits)
  _, params = init_fn(key, (-1,) + input_shape)
  ker_fun = jit(empirical.get_ntk_fun_empirical(f))

  return params, f, partial(ker_fun, params=params)


def _theoretical_kernel(key, input_shape, network, out_logits):
  init_fn, f, _ker_fun = _build_network(input_shape, network, out_logits)
  _, params = init_fn(key, (-1,) + input_shape)
  @jit
  def ker_fun(x1, x2=None):
    k = _ker_fun(x1, x2)
    return k.ntk
  return params, f, ker_fun


KERNELS = {
    'empirical': _empirical_kernel,
    'theoretical': _theoretical_kernel,
}


@optimizers.optimizer
def momentum(learning_rate, momentum=0.9):
  """A standard momentum optimizer for testing.

  Different from `jax.experimental.optimizers.momentum` (Nesterov).
  """
  learning_rate = optimizers.make_schedule(learning_rate)
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


class PredictTest(jtu.JaxTestCase):

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name': '_train={}_test={}_network={}_logits={}_{}'.format(
              train, test, network, out_logits, name),
          'train_shape': train,
          'test_shape': test,
          'network': network,
          'out_logits': out_logits,
          'fn_and_kernel': fn
      } for train, test, network in zip(TRAIN_SHAPES, TEST_SHAPES, NETWORK)
                          for out_logits in OUTPUT_LOGITS
                          for name, fn in KERNELS.items()))
  def testNTKMSEPrediction(
      self, train_shape, test_shape, network, out_logits, fn_and_kernel):

    key = random.PRNGKey(0)

    key, split = random.split(key)
    data_train = random.normal(split, train_shape)

    key, split = random.split(key)
    data_labels = np.array(
        random.bernoulli(split, shape=(train_shape[0], out_logits)), np.float32)

    key, split = random.split(key)
    data_test = random.normal(split, test_shape)

    params, f, ntk = fn_and_kernel(key, train_shape[1:], network, out_logits)

    # Regress to an MSE loss.
    loss = lambda params, x: \
        0.5 * np.mean((f(params, x) - data_labels) ** 2)
    grad_loss = jit(grad(loss))

    g_dd = ntk(data_train, None)
    g_td = ntk(data_test, data_train)

    predictor = predict.analytic_mse(g_dd, data_labels, g_td)

    atol = ATOL
    rtol = RTOL
    step_size = 0.5

    if len(train_shape) > 2:
      # Hacky way to up the tolerance just for convolutions.
      atol = ATOL * 2
      rtol = RTOL * 2
      step_size = 0.1

    train_time = 100.0
    steps = int(train_time / step_size)

    opt_init, opt_update, get_params = optimizers.sgd(step_size)
    opt_state = opt_init(params)

    fx_initial_train = f(params, data_train)
    fx_initial_test = f(params, data_test)

    fx_pred_train, fx_pred_test = predictor(
        fx_initial_train, fx_initial_test, 0.0)

    self.assertAllClose(fx_initial_train, fx_pred_train, True)
    self.assertAllClose(fx_initial_test, fx_pred_test, True)

    for i in range(steps):
      params = get_params(opt_state)
      opt_state = opt_update(i, grad_loss(params, data_train), opt_state)

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
        fx_error_train, np.zeros_like(fx_error_train), True, rtol, atol)
    self.assertAllClose(
        fx_error_test, np.zeros_like(fx_error_test), True, rtol, atol)

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name': '_train={}_test={}_network={}_logits={}_{}'.format(
              train, test, network, out_logits, name),
          'train_shape': train,
          'test_shape': test,
          'network': network,
          'out_logits': out_logits,
          'fn_and_kernel': fn
      } for train, test, network in zip(TRAIN_SHAPES, TEST_SHAPES, NETWORK)
                          for out_logits in OUTPUT_LOGITS
                          for name, fn in KERNELS.items()))
  def testNTKGDPrediction(
      self, train_shape, test_shape, network, out_logits, fn_and_kernel):
    key = random.PRNGKey(0)

    key, split = random.split(key)
    data_train = random.normal(split, train_shape)

    key, split = random.split(key)
    data_labels = np.array(
        random.bernoulli(split, shape=(train_shape[0], out_logits)), np.float32)

    key, split = random.split(key)
    data_test = random.normal(split, test_shape)

    params, f, ntk = fn_and_kernel(key, train_shape[1:], network, out_logits)

    # Regress to an MSE loss.
    loss = lambda y, y_hat: 0.5 * np.mean((y - y_hat) ** 2)
    grad_loss = jit(grad(lambda params, x: loss(f(params, x), data_labels)))

    g_dd = ntk(data_train, None)
    g_td = ntk(data_test, data_train)

    predictor = predict.gradient_descent(g_dd, data_labels, loss, g_td)

    atol = ATOL
    rtol = RTOL
    step_size = 0.5

    if len(train_shape) > 2:
      # Hacky way to up the tolerance just for convolutions.
      atol = ATOL * 2
      rtol = RTOL * 2
      step_size = 0.1

    train_time = 100.0
    steps = int(train_time / step_size)

    opt_init, opt_update, get_params = optimizers.sgd(step_size)
    opt_state = opt_init(params)

    fx_initial_train = f(params, data_train)
    fx_initial_test = f(params, data_test)

    fx_pred_train, fx_pred_test = predictor(
        fx_initial_train, fx_initial_test, 0.0)

    self.assertAllClose(fx_initial_train, fx_pred_train, True)
    self.assertAllClose(fx_initial_test, fx_pred_test, True)

    for i in range(steps):
      params = get_params(opt_state)
      opt_state = opt_update(i, grad_loss(params, data_train), opt_state)

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
        fx_error_train, np.zeros_like(fx_error_train), True, rtol, atol)
    self.assertAllClose(
        fx_error_test, np.zeros_like(fx_error_test), True, rtol, atol)

  # TODO(schsam): Get this test passing with theoretical conv.
  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name': '_train={}_test={}_network={}_logits={}_{}'.format(
              train, test, network, out_logits, name),
          'train_shape': train,
          'test_shape': test,
          'network': network,
          'out_logits': out_logits,
          'fn_and_kernel': fn
      } for train, test, network in zip(TRAIN_SHAPES, TEST_SHAPES, NETWORK)
                          for out_logits in OUTPUT_LOGITS
                          for name, fn in KERNELS.items()
                          if len(train) == 2))
  def testNTKMomentumPrediction(
      self, train_shape, test_shape, network, out_logits, fn_and_kernel):
    key = random.PRNGKey(0)

    key, split = random.split(key)
    data_train = random.normal(split, train_shape)

    key, split = random.split(key)
    data_labels = np.array(
        random.bernoulli(split, shape=(train_shape[0], out_logits)), np.float32)

    key, split = random.split(key)
    data_test = random.normal(split, test_shape)

    params, f, ntk = fn_and_kernel(key, train_shape[1:], network, out_logits)

    # Regress to an MSE loss.
    loss = lambda y, y_hat: 0.5 * np.mean((y - y_hat) ** 2)
    grad_loss = jit(grad(lambda params, x: loss(f(params, x), data_labels)))

    g_dd = ntk(data_train, None)
    g_td = ntk(data_test, data_train)

    atol = ATOL
    rtol = RTOL
    step_size = 0.5

    if len(train_shape) > 2:
      # Hacky way to up the tolerance just for convolutions.
      atol = ATOL * 2
      rtol = RTOL * 2
      step_size = 0.1

    train_time = 100.0
    steps = int(train_time / np.sqrt(step_size))

    init, predictor, get = predict.momentum(
        g_dd, data_labels, loss, step_size, g_td)

    opt_init, opt_update, get_params = momentum(step_size, 0.9)
    opt_state = opt_init(params)

    fx_initial_train = f(params, data_train)
    fx_initial_test = f(params, data_test)

    lin_state = init(fx_initial_train, fx_initial_test)
    fx_pred_train, fx_pred_test = get(lin_state)

    self.assertAllClose(fx_initial_train, fx_pred_train, True)
    self.assertAllClose(fx_initial_test, fx_pred_test, True)

    for i in range(steps):
      params = get_params(opt_state)
      opt_state = opt_update(i, grad_loss(params, data_train), opt_state)

    params = get_params(opt_state)
    fx_train = f(params, data_train)
    fx_test = f(params, data_test)

    lin_state = predictor(lin_state, train_time)
    fx_pred_train, fx_pred_test = get(lin_state)

    fx_disp_train = np.sqrt(np.mean((fx_train - fx_initial_train) ** 2))
    fx_disp_test = np.sqrt(np.mean((fx_test - fx_initial_test) ** 2))

    fx_error_train = (fx_train - fx_pred_train) / fx_disp_train
    fx_error_test = (fx_test - fx_pred_test) / fx_disp_test

    self.assertAllClose(
        fx_error_train, np.zeros_like(fx_error_train), True, rtol, atol)
    self.assertAllClose(
        fx_error_test, np.zeros_like(fx_error_test), True, rtol, atol)


  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              '_train={}_test={}_network={}_logits={}'.format(
                  train, test, network, out_logits),
          'train_shape':
              train,
          'test_shape':
              test,
          'network':
              network,
          'out_logits':
              out_logits,
      }
                          for train, test, network in zip(
                              TRAIN_SHAPES[:-1], TEST_SHAPES[:-1], NETWORK[:-1])
                          for out_logits in OUTPUT_LOGITS))

  def testNTKMeanPrediction(
      self, train_shape, test_shape, network, out_logits):

    key = random.PRNGKey(0)

    key, split = random.split(key)
    data_train = np.cos(random.normal(split, train_shape))

    key, split = random.split(key)
    data_labels = np.array(
        random.bernoulli(split, shape=(train_shape[0], out_logits)), np.float32)

    key, split = random.split(key)
    data_test = np.cos(random.normal(split, test_shape))
    _, _, ker_fun = _build_network(train_shape[1:], network, out_logits)
    mean_pred, var = predict.gp_inference(ker_fun, data_train, data_labels,
                                          data_test, diag_reg=0., mode='NTK',
                                          compute_var=True)

    if xla_bridge.get_backend().platform == 'tpu':
      eigh = np.onp.linalg.eigh
    else:
      eigh = np.linalg.eigh

    self.assertEqual(var.shape[0], data_test.shape[0])
    min_eigh = np.min(eigh(var)[0])
    self.assertGreater(min_eigh + 1e-10, 0.)
    def mc_sampling(count=10):
      empirical_mean = 0.
      key = random.PRNGKey(100)
      for _ in range(count):
        key, split = random.split(key)
        params, f, theta = _empirical_kernel(split, train_shape[1:],
                                             network, out_logits)
        g_dd = theta(data_train, None)
        g_td = theta(data_test, data_train)
        predictor = predict.analytic_mse(g_dd, data_labels, g_td)

        fx_initial_train = f(params, data_train)
        fx_initial_test = f(params, data_test)

        _, fx_pred_test = predictor(fx_initial_train, fx_initial_test, np.inf)
        empirical_mean += fx_pred_test
      return empirical_mean / count
    atol = ATOL
    rtol = RTOL
    mean_emp = mc_sampling(100)

    self.assertAllClose(mean_pred, mean_emp, True, rtol, atol)


if __name__ == '__main__':
  jtu.absltest.main()
