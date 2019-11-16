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
import math

from jax import test_util as jtu
from jax.api import device_get
from jax.api import grad
from jax.api import jit
from jax.api import vmap
from jax.config import config
from jax.experimental import optimizers
from jax.lib import xla_bridge
import jax.numpy as np
import jax.random as random
from neural_tangents import predict
from neural_tangents import stax
from neural_tangents.utils import batch
from neural_tangents.utils import empirical

config.parse_flags_with_absl()

MATRIX_SHAPES = [(3, 3), (4, 4)]
OUTPUT_LOGITS = [1, 2, 3]

GETS = ('ntk', 'nngp', ('ntk', 'nngp'))

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
        stax.Dense(4096, W_std=1.2, b_std=0.05), stax.Erf(),
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
  _kernel_fn = empirical.empirical_kernel_fn(f)
  kernel_fn = lambda x1, x2, get: _kernel_fn(x1, x2, params, get)
  return params, f, jit(kernel_fn, static_argnums=(2,))


def _theoretical_kernel(key, input_shape, network, out_logits):
  init_fn, f, kernel_fn = _build_network(input_shape, network, out_logits)
  _, params = init_fn(key, (-1,) + input_shape)
  return params, f, jit(kernel_fn, static_argnums=(2,))


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

  def init_fn(x0):
    v0 = np.zeros_like(x0)
    return x0, v0

  def update_fn(i, g, state):
    x, velocity = state
    velocity = momentum * velocity + g
    x = x - learning_rate(i) * velocity
    return x, velocity

  def get_params(state):
    x, _ = state
    return x

  return init_fn, update_fn, get_params


class PredictTest(jtu.JaxTestCase):

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              '_train={}_network={}_logits={}_{}'.format(
                  train, network, out_logits, name),
          'train_shape':
              train,
          'network':
              network,
          'out_logits':
              out_logits,
          'fn_and_kernel':
              fn,
          'name':
              name,
      } for train, network in zip(TRAIN_SHAPES, NETWORK)
                          for out_logits in OUTPUT_LOGITS
                          for name, fn in KERNELS.items()))
  def testMaxLearningRate(self, train_shape, network, out_logits, fn_and_kernel,
                          name):

    key = random.PRNGKey(0)

    key, split = random.split(key)
    if len(train_shape) == 2:
      train_shape = (train_shape[0] * 5, train_shape[1] * 10)
    else:
      train_shape = (16, 8, 8, 3)
    x_train = random.normal(split, train_shape)

    key, split = random.split(key)
    y_train = np.array(
        random.bernoulli(split, shape=(train_shape[0], out_logits)), np.float32)

    for lr_factor in [0.5, 3.]:
      params, f, ntk = fn_and_kernel(key, train_shape[1:], network, out_logits)

      # Regress to an MSE loss.
      loss = lambda params, x: \
          0.5 * np.mean((f(params, x) - y_train) ** 2)
      grad_loss = jit(grad(loss))

      g_dd = ntk(x_train, None, 'ntk')

      steps = 20
      if name == 'theoretical':
        step_size = predict.max_learning_rate(
            g_dd, num_outputs=out_logits) * lr_factor
      else:
        step_size = predict.max_learning_rate(g_dd, num_outputs=-1) * lr_factor
      opt_init, opt_update, get_params = optimizers.sgd(step_size)
      opt_state = opt_init(params)

      def get_loss(opt_state):
        return loss(get_params(opt_state), x_train)

      init_loss = get_loss(opt_state)

      for i in range(steps):
        params = get_params(opt_state)
        opt_state = opt_update(i, grad_loss(params, x_train), opt_state)

      trained_loss = get_loss(opt_state)
      loss_ratio = trained_loss / (init_loss + 1e-12)
      if lr_factor == 3.:
        if not math.isnan(loss_ratio):
          self.assertGreater(loss_ratio, 10.)
      else:
        self.assertLess(loss_ratio, 0.1)

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              '_train={}_test={}_network={}_logits={}_{}'.format(
                  train, test, network, out_logits, name),
          'train_shape':
              train,
          'test_shape':
              test,
          'network':
              network,
          'out_logits':
              out_logits,
          'fn_and_kernel':
              fn
      } for train, test, network in zip(TRAIN_SHAPES, TEST_SHAPES, NETWORK)
                          for out_logits in OUTPUT_LOGITS
                          for name, fn in KERNELS.items()))
  def testNTKMSEPrediction(self, train_shape, test_shape, network, out_logits,
                           fn_and_kernel):

    key = random.PRNGKey(0)

    key, split = random.split(key)
    x_train = random.normal(split, train_shape)

    key, split = random.split(key)
    y_train = np.array(
        random.bernoulli(split, shape=(train_shape[0], out_logits)), np.float32)

    key, split = random.split(key)
    x_test = random.normal(split, test_shape)

    params, f, ntk = fn_and_kernel(key, train_shape[1:], network, out_logits)

    # Regress to an MSE loss.
    loss = lambda params, x: \
        0.5 * np.mean((f(params, x) - y_train) ** 2)
    grad_loss = jit(grad(loss))

    g_dd = ntk(x_train, None, 'ntk')
    g_td = ntk(x_test, x_train, 'ntk')

    predictor = predict.gradient_descent_mse(g_dd, y_train, g_td)
    predictor_train = predict.gradient_descent_mse(g_dd, y_train)

    atol = ATOL
    rtol = RTOL
    step_size = 0.1

    if len(train_shape) > 2:
      # Hacky way to up the tolerance just for convolutions.
      atol = ATOL * 2
      rtol = RTOL * 2
      step_size = 0.1

    train_time = 100.0
    steps = int(train_time / step_size)

    opt_init, opt_update, get_params = optimizers.sgd(step_size)
    opt_state = opt_init(params)

    fx_initial_train = f(params, x_train)
    fx_initial_test = f(params, x_test)

    fx_pred_train, fx_pred_test = predictor(0.0, fx_initial_train,
                                            fx_initial_test)
    fx_pred_train_only = predictor_train(0.0, fx_initial_train)

    self.assertAllClose(fx_initial_train, fx_pred_train, True)
    self.assertAllClose(fx_initial_train, fx_pred_train_only, True)
    self.assertAllClose(fx_initial_test, fx_pred_test, True)

    for i in range(steps):
      params = get_params(opt_state)
      opt_state = opt_update(i, grad_loss(params, x_train), opt_state)

    params = get_params(opt_state)
    fx_train = f(params, x_train)
    fx_test = f(params, x_test)

    fx_pred_train, fx_pred_test = predictor(train_time, fx_initial_train,
                                            fx_initial_test)
    fx_pred_train_only = predictor_train(train_time, fx_initial_train)

    fx_disp_train = np.sqrt(np.mean((fx_train - fx_initial_train)**2))
    fx_disp_test = np.sqrt(np.mean((fx_test - fx_initial_test)**2))

    fx_error_train = (fx_train - fx_pred_train) / fx_disp_train
    fx_error_train_only = (fx_pred_train_only - fx_pred_train) / fx_disp_train
    fx_error_test = (fx_test - fx_pred_test) / fx_disp_test

    self.assertAllClose(fx_error_train, np.zeros_like(fx_error_train), True,
                        rtol, atol)
    self.assertAllClose(fx_error_train_only, np.zeros_like(fx_error_train_only),
                        True, rtol, atol)
    self.assertAllClose(fx_error_test, np.zeros_like(fx_error_test), True, rtol,
                        atol)

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              '_train={}_test={}_network={}_logits={}_{}'.format(
                  train, test, network, out_logits, name),
          'train_shape':
              train,
          'test_shape':
              test,
          'network':
              network,
          'out_logits':
              out_logits,
          'fn_and_kernel':
              fn
      } for train, test, network in zip(TRAIN_SHAPES, TEST_SHAPES, NETWORK)
                          for out_logits in OUTPUT_LOGITS
                          for name, fn in KERNELS.items()))
  def testNTKGDPrediction(self, train_shape, test_shape, network, out_logits,
                          fn_and_kernel):
    key = random.PRNGKey(0)

    key, split = random.split(key)
    x_train = random.normal(split, train_shape)

    key, split = random.split(key)
    y_train = np.array(
        random.bernoulli(split, shape=(train_shape[0], out_logits)), np.float32)

    key, split = random.split(key)
    x_test = random.normal(split, test_shape)

    params, f, ntk = fn_and_kernel(key, train_shape[1:], network, out_logits)

    # Regress to an MSE loss.
    loss = lambda y, y_hat: 0.5 * np.mean((y - y_hat)**2)
    grad_loss = jit(grad(lambda params, x: loss(f(params, x), y_train)))

    g_dd = ntk(x_train, None, 'ntk')
    g_td = ntk(x_test, x_train, 'ntk')

    predictor = predict.gradient_descent(g_dd, y_train, loss, g_td)

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

    fx_initial_train = f(params, x_train)
    fx_initial_test = f(params, x_test)

    fx_pred_train, fx_pred_test = predictor(0.0, fx_initial_train,
                                            fx_initial_test)

    self.assertAllClose(fx_initial_train, fx_pred_train, True)
    self.assertAllClose(fx_initial_test, fx_pred_test, True)

    for i in range(steps):
      params = get_params(opt_state)
      opt_state = opt_update(i, grad_loss(params, x_train), opt_state)

    params = get_params(opt_state)
    fx_train = f(params, x_train)
    fx_test = f(params, x_test)

    fx_pred_train, fx_pred_test = predictor(train_time, fx_initial_train,
                                            fx_initial_test)

    fx_disp_train = np.sqrt(np.mean((fx_train - fx_initial_train)**2))
    fx_disp_test = np.sqrt(np.mean((fx_test - fx_initial_test)**2))

    fx_error_train = (fx_train - fx_pred_train) / fx_disp_train
    fx_error_test = (fx_test - fx_pred_test) / fx_disp_test

    self.assertAllClose(fx_error_train, np.zeros_like(fx_error_train), True,
                        rtol, atol)
    self.assertAllClose(fx_error_test, np.zeros_like(fx_error_test), True, rtol,
                        atol)

  # TODO(schsam): Get this test passing with theoretical conv.
  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              '_train={}_test={}_network={}_logits={}_{}'.format(
                  train, test, network, out_logits, name),
          'train_shape':
              train,
          'test_shape':
              test,
          'network':
              network,
          'out_logits':
              out_logits,
          'fn_and_kernel':
              fn
      } for train, test, network in zip(TRAIN_SHAPES, TEST_SHAPES, NETWORK)
                          for out_logits in OUTPUT_LOGITS
                          for name, fn in KERNELS.items()
                          if len(train) == 2))
  def testNTKMomentumPrediction(self, train_shape, test_shape, network,
                                out_logits, fn_and_kernel):
    key = random.PRNGKey(0)

    key, split = random.split(key)
    x_train = random.normal(split, train_shape)

    key, split = random.split(key)
    y_train = np.array(
        random.bernoulli(split, shape=(train_shape[0], out_logits)), np.float32)

    key, split = random.split(key)
    x_test = random.normal(split, test_shape)

    params, f, ntk = fn_and_kernel(key, train_shape[1:], network, out_logits)

    # Regress to an MSE loss.
    loss = lambda y, y_hat: 0.5 * np.mean((y - y_hat)**2)
    grad_loss = jit(grad(lambda params, x: loss(f(params, x), y_train)))

    g_dd = ntk(x_train, None, 'ntk')
    g_td = ntk(x_test, x_train, 'ntk')

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

    init, predictor, get = predict.momentum(g_dd, y_train, loss, step_size,
                                            g_td)

    opt_init, opt_update, get_params = momentum(step_size, 0.9)
    opt_state = opt_init(params)

    fx_initial_train = f(params, x_train)
    fx_initial_test = f(params, x_test)

    lin_state = init(fx_initial_train, fx_initial_test)
    fx_pred_train, fx_pred_test = get(lin_state)

    self.assertAllClose(fx_initial_train, fx_pred_train, True)
    self.assertAllClose(fx_initial_test, fx_pred_test, True)

    for i in range(steps):
      params = get_params(opt_state)
      opt_state = opt_update(i, grad_loss(params, x_train), opt_state)

    params = get_params(opt_state)
    fx_train = f(params, x_train)
    fx_test = f(params, x_test)

    lin_state = predictor(lin_state, train_time)
    fx_pred_train, fx_pred_test = get(lin_state)

    fx_disp_train = np.sqrt(np.mean((fx_train - fx_initial_train)**2))
    fx_disp_test = np.sqrt(np.mean((fx_test - fx_initial_test)**2))

    fx_error_train = (fx_train - fx_pred_train) / fx_disp_train
    fx_error_test = (fx_test - fx_pred_test) / fx_disp_test

    self.assertAllClose(fx_error_train, np.zeros_like(fx_error_train), True,
                        rtol, atol)
    self.assertAllClose(fx_error_test, np.zeros_like(fx_error_test), True, rtol,
                        atol)

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
  def testNTKMeanCovPrediction(self, train_shape, test_shape, network,
                               out_logits):

    key = random.PRNGKey(0)

    key, split = random.split(key)
    x_train = np.cos(random.normal(split, train_shape))

    key, split = random.split(key)
    y_train = np.array(
        random.bernoulli(split, shape=(train_shape[0], out_logits)), np.float32)

    key, split = random.split(key)
    x_test = np.cos(random.normal(split, test_shape))
    _, _, kernel_fn = _build_network(train_shape[1:], network, out_logits)
    mean_pred, cov_pred = predict.gp_inference(
        kernel_fn,
        x_train,
        y_train,
        x_test,
        'ntk',
        diag_reg=0.,
        compute_cov=True)

    if xla_bridge.get_backend().platform == 'tpu':
      eigh = np.onp.linalg.eigh
    else:
      eigh = np.linalg.eigh

    self.assertEqual(cov_pred.shape[0], x_test.shape[0])
    min_eigh = np.min(eigh(cov_pred)[0])
    self.assertGreater(min_eigh + 1e-10, 0.)

    def mc_sampling(count=10):
      key = random.PRNGKey(100)
      init_fn, f, _ = _build_network(train_shape[1:], network, out_logits)
      _kernel_fn = empirical.empirical_kernel_fn(f)
      kernel_fn = jit(lambda x1, x2, params: _kernel_fn(x1, x2, params, 'ntk'))
      collect_test_predict = []
      for _ in range(count):
        key, split = random.split(key)
        _, params = init_fn(split, train_shape)

        g_dd = kernel_fn(x_train, None, params)
        g_td = kernel_fn(x_test, x_train, params)
        predictor = predict.gradient_descent_mse(g_dd, y_train, g_td)

        fx_initial_train = f(params, x_train)
        fx_initial_test = f(params, x_test)

        _, fx_pred_test = predictor(1.0e8, fx_initial_train, fx_initial_test)
        collect_test_predict.append(fx_pred_test)
      collect_test_predict = np.array(collect_test_predict)
      mean_emp = np.mean(collect_test_predict, axis=0)
      mean_subtracted = collect_test_predict - mean_emp
      cov_emp = np.einsum(
          'ijk,ilk->jl', mean_subtracted, mean_subtracted, optimize=True) / (
              mean_subtracted.shape[0] * mean_subtracted.shape[-1])
      return mean_emp, cov_emp

    atol = ATOL
    rtol = RTOL
    mean_emp, cov_emp = mc_sampling(100)

    self.assertAllClose(mean_pred, mean_emp, True, rtol, atol)
    self.assertAllClose(cov_pred, cov_emp, True, rtol, atol)

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
  def testGPInferenceGet(self, train_shape, test_shape, network, out_logits):

    key = random.PRNGKey(0)

    key, split = random.split(key)
    x_train = np.cos(random.normal(split, train_shape))

    key, split = random.split(key)
    y_train = np.array(
        random.bernoulli(split, shape=(train_shape[0], out_logits)), np.float32)

    key, split = random.split(key)
    x_test = np.cos(random.normal(split, test_shape))
    _, _, kernel_fn = _build_network(train_shape[1:], network, out_logits)

    out = predict.gp_inference(
        kernel_fn,
        x_train,
        y_train,
        x_test,
        'ntk',
        diag_reg=0.,
        compute_cov=True)
    assert isinstance(out, predict.Gaussian)

    out = predict.gp_inference(
        kernel_fn,
        x_train,
        y_train,
        x_test,
        'nngp',
        diag_reg=0.,
        compute_cov=True)
    assert isinstance(out, predict.Gaussian)

    out = predict.gp_inference(
        kernel_fn,
        x_train,
        y_train,
        x_test, ('ntk',),
        diag_reg=0.,
        compute_cov=True)
    assert len(out) == 1 and isinstance(out[0], predict.Gaussian)

    out = predict.gp_inference(
        kernel_fn,
        x_train,
        y_train,
        x_test, ('ntk', 'nngp'),
        diag_reg=0.,
        compute_cov=True)
    assert (len(out) == 2 and isinstance(out[0], predict.Gaussian) and
            isinstance(out[1], predict.Gaussian))

    out2 = predict.gp_inference(
        kernel_fn,
        x_train,
        y_train,
        x_test, ('nngp', 'ntk'),
        diag_reg=0.,
        compute_cov=True)
    self.assertAllClose(out[0], out2[1], True)
    self.assertAllClose(out[1], out2[0], True)

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              '_train={}_test={}_network={}_logits={}_get={}'.format(
                  train, test, network, out_logits, get),
          'train_shape':
              train,
          'test_shape':
              test,
          'network':
              network,
          'out_logits':
              out_logits,
          'get':
              get,
      }
                          for train, test, network in zip(
                              TRAIN_SHAPES[:-1], TEST_SHAPES[:-1], NETWORK[:-1])
                          for out_logits in OUTPUT_LOGITS for get in GETS))
  def testInfiniteTimeAgreement(self, train_shape, test_shape, network,
                                out_logits, get):

    key = random.PRNGKey(0)

    key, split = random.split(key)
    x_train = np.cos(random.normal(split, train_shape))

    key, split = random.split(key)
    y_train = np.array(
        random.bernoulli(split, shape=(train_shape[0], out_logits)), np.float32)

    key, split = random.split(key)
    x_test = np.cos(random.normal(split, test_shape))
    _, _, kernel_fn = _build_network(train_shape[1:], network, out_logits)

    reg = 1e-7
    prediction = predict.gradient_descent_mse_gp(
        kernel_fn,
        x_train,
        y_train,
        x_test,
        get,
        diag_reg=reg,
        compute_cov=True)

    finite_prediction = prediction(np.inf)
    finite_prediction_none = prediction(None)
    gp_inference = predict.gp_inference(kernel_fn, x_train, y_train, x_test,
                                        get, reg, True)

    self.assertAllClose(finite_prediction_none, finite_prediction, True)
    self.assertAllClose(finite_prediction_none, gp_inference, True)

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
  def testZeroTimeAgreement(self, train_shape, test_shape, network, out_logits):
    """Test that the NTK and NNGP agree at t=0."""

    key = random.PRNGKey(0)

    key, split = random.split(key)
    x_train = np.cos(random.normal(split, train_shape))

    key, split = random.split(key)
    y_train = np.array(
        random.bernoulli(split, shape=(train_shape[0], out_logits)), np.float32)

    key, split = random.split(key)
    x_test = np.cos(random.normal(split, test_shape))
    _, _, ker_fun = _build_network(train_shape[1:], network, out_logits)

    reg = 1e-7
    prediction = predict.gradient_descent_mse_gp(
        ker_fun,
        x_train,
        y_train,
        x_test,
        diag_reg=reg,
        get=('NTK', 'NNGP'),
        compute_cov=True)

    zero_prediction = prediction(0.0)

    self.assertAllClose(zero_prediction.ntk, zero_prediction.nngp, True)
    reference = (np.zeros(
        (test_shape[0], out_logits)), ker_fun(x_test, x_test, get='nngp'))
    self.assertAllClose((reference,) * 2, zero_prediction, True)

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
  def testNTK_NTKNNGPAgreement(self, train_shape, test_shape, network,
                               out_logits):
    key = random.PRNGKey(0)

    key, split = random.split(key)
    x_train = np.cos(random.normal(split, train_shape))

    key, split = random.split(key)
    y_train = np.array(
        random.bernoulli(split, shape=(train_shape[0], out_logits)), np.float32)

    key, split = random.split(key)
    x_test = np.cos(random.normal(split, test_shape))
    _, _, ker_fun = _build_network(train_shape[1:], network, out_logits)

    reg = 1e-7
    prediction = predict.gradient_descent_mse_gp(
        ker_fun,
        x_train,
        y_train,
        x_test,
        diag_reg=reg,
        get='NTK',
        compute_cov=True)

    ts = np.logspace(-2, 8, 10)
    ntk_predictions = [prediction(t).mean for t in ts]

    # Create a hacked kernel function that always returns the ntk kernel
    def always_ntk(x1, x2, get=('nngp', 'ntk')):
      out = ker_fun(x1, x2, get=('nngp', 'ntk'))
      if get == 'nngp' or get == 'ntk':
        return out.ntk
      else:
        return out._replace(nngp=out.ntk)

    ntk_nngp_prediction = predict.gradient_descent_mse_gp(
        always_ntk,
        x_train,
        y_train,
        x_test,
        diag_reg=reg,
        get='NNGP',
        compute_cov=True)

    ntk_nngp_predictions = [ntk_nngp_prediction(t).mean for t in ts]

    # Test if you use the nngp equations with the ntk, you get the same mean
    self.assertAllClose(ntk_predictions, ntk_nngp_predictions, True)

    # Next test that if you go through the NTK code path, but with only
    # the NNGP kernel, we recreate the NNGP dynamics.
    reg = 1e-7
    nngp_prediction = predict.gradient_descent_mse_gp(
        ker_fun,
        x_train,
        y_train,
        x_test,
        diag_reg=reg,
        get='NNGP',
        compute_cov=True)

    # Create a hacked kernel function that always returns the nngp kernel
    def always_nngp(x1, x2, get=('nngp', 'ntk')):
      out = ker_fun(x1, x2, get=('nngp', 'ntk'))
      if get == 'nngp' or get == 'ntk':
        return out.nngp
      else:
        return out._replace(ntk=out.nngp)

    nngp_ntk_prediction = predict.gradient_descent_mse_gp(
        always_nngp,
        x_train,
        y_train,
        x_test,
        diag_reg=reg,
        get='NTK',
        compute_cov=True)

    nngp_cov_predictions = [nngp_prediction(t).covariance for t in ts]
    nngp_ntk_cov_predictions = [nngp_ntk_prediction(t).covariance for t in ts]

    # Test if you use the ntk equations with the nngp, you get the same cov
    # Although, due to accumulation of numerical errors, only roughly.
    self.assertAllClose(nngp_cov_predictions, nngp_ntk_cov_predictions, True)

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
  def testNTKPredCovPosDef(self, train_shape, test_shape, network, out_logits):
    key = random.PRNGKey(0)

    key, split = random.split(key)
    x_train = np.cos(random.normal(split, train_shape))

    key, split = random.split(key)
    y_train = np.array(
        random.bernoulli(split, shape=(train_shape[0], out_logits)), np.float32)

    key, split = random.split(key)
    x_test = np.cos(random.normal(split, test_shape))
    _, _, ker_fun = _build_network(train_shape[1:], network, out_logits)

    reg = 1e-7
    ntk_predictions = predict.gradient_descent_mse_gp(
        ker_fun,
        x_train,
        y_train,
        x_test,
        diag_reg=reg,
        get='ntk',
        compute_cov=True)

    ts = np.logspace(-2, 8, 10)

    ntk_cov_predictions = [ntk_predictions(t).covariance for t in ts]

    if xla_bridge.get_backend().platform == 'tpu':
      eigh = np.onp.linalg.eigh
    else:
      eigh = np.linalg.eigh

    check_symmetric = np.array(
        [np.max(np.abs(cov - cov.T)) for cov in ntk_cov_predictions])
    check_pos_evals = np.min(
        np.array([eigh(cov)[0] + 1e-10 for cov in ntk_cov_predictions]))

    self.assertAllClose(check_symmetric, np.zeros_like(check_symmetric), True)
    self.assertGreater(check_pos_evals, 0., True)

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
  def testTrainedEnsemblePredCov(self, train_shape, test_shape, network,
                                 out_logits):
    if xla_bridge.get_backend().platform == 'gpu' and config.read(
        'jax_enable_x64'):
      raise jtu.SkipTest('Not running GPU x64 to save time.')
    training_steps = 5000
    learning_rate = 1.0
    ensemble_size = 50

    init_fn, apply_fn, ker_fn = stax.serial(
        stax.Dense(1024, W_std=1.2, b_std=0.05), stax.Erf(),
        stax.Dense(out_logits, W_std=1.2, b_std=0.05))

    opt_init, opt_update, get_params = optimizers.sgd(learning_rate)
    opt_update = jit(opt_update)

    key = random.PRNGKey(0)
    key, = random.split(key, 1)

    key, split = random.split(key)
    x_train = np.cos(random.normal(split, train_shape))

    key, split = random.split(key)
    y_train = np.array(
        random.bernoulli(split, shape=(train_shape[0], out_logits)), np.float32)
    train = (x_train, y_train)
    key, split = random.split(key)
    x_test = np.cos(random.normal(split, test_shape))

    ensemble_key = random.split(key, ensemble_size)

    loss = jit(lambda params, x, y: 0.5 * np.mean((apply_fn(params, x) - y)**2))
    grad_loss = jit(lambda state, x, y: grad(loss)(get_params(state), x, y))

    def train_network(key):
      _, params = init_fn(key, (-1,) + train_shape[1:])
      opt_state = opt_init(params)
      for i in range(training_steps):
        opt_state = opt_update(i, grad_loss(opt_state, *train), opt_state)

      return get_params(opt_state)

    params = vmap(train_network)(ensemble_key)

    ensemble_fx = vmap(apply_fn, (0, None))(params, x_test)
    ensemble_loss = vmap(loss, (0, None, None))(params, x_train, y_train)
    ensemble_loss = np.mean(ensemble_loss)
    self.assertLess(ensemble_loss, 1e-5, True)

    mean_emp = np.mean(ensemble_fx, axis=0)
    mean_subtracted = ensemble_fx - mean_emp
    cov_emp = np.einsum(
        'ijk,ilk->jl', mean_subtracted, mean_subtracted, optimize=True) / (
            mean_subtracted.shape[0] * mean_subtracted.shape[-1])

    reg = 1e-7
    ntk_predictions = predict.gp_inference(
        ker_fn, x_train, y_train, x_test, 'ntk', reg, compute_cov=True)

    self.assertAllClose(mean_emp, ntk_predictions.mean, True, RTOL, ATOL)
    self.assertAllClose(cov_emp, ntk_predictions.covariance, True, RTOL, ATOL)

  def testPredictOnCPU(self):
    x_train = random.normal(random.PRNGKey(1), (10, 4, 5, 3))
    x_test = random.normal(random.PRNGKey(1), (8, 4, 5, 3))

    y_train = random.uniform(random.PRNGKey(1), (10, 7))

    _, _, kernel_fn = stax.serial(
        stax.Conv(1, (3, 3)), stax.Relu(), stax.Flatten(), stax.Dense(1))

    for store_on_device in [False, True]:
      for device_count in [0, 1]:
        for get in ['ntk', 'nngp', ('nngp', 'ntk'), ('ntk', 'nngp')]:
          with self.subTest(
              store_on_device=store_on_device,
              device_count=device_count,
              get=get):
            kernel_fn_batched = batch.batch(kernel_fn, 2, device_count,
                                            store_on_device)
            predictor = predict.gradient_descent_mse_gp(kernel_fn_batched,
                                                        x_train, y_train,
                                                        x_test, get, 0., True)
            gp_inference = predict.gp_inference(kernel_fn_batched, x_train,
                                                y_train, x_test, get, 0., True)

            self.assertAllClose(predictor(None), predictor(np.inf), True)
            self.assertAllClose(predictor(None), gp_inference, True)

  def testIsOnCPU(self):
    for dtype in [np.float32, np.float64]:
      with self.subTest(dtype=dtype):

        def x():
          return random.normal(random.PRNGKey(1), (2, 3), dtype)

        def x_cpu():
          return device_get(random.normal(random.PRNGKey(1), (2, 3), dtype))

        x_jit = jit(x)
        x_cpu_jit = jit(x_cpu)
        x_cpu_jit_cpu = jit(x_cpu, backend='cpu')

        self.assertTrue(predict._is_on_cpu(x_cpu()))
        self.assertTrue(predict._is_on_cpu(x_cpu_jit()))
        self.assertTrue(predict._is_on_cpu(x_cpu_jit_cpu()))

        if xla_bridge.get_backend().platform == 'cpu':
          self.assertTrue(predict._is_on_cpu(x()))
          self.assertTrue(predict._is_on_cpu(x_jit()))
        else:
          self.assertFalse(predict._is_on_cpu(x()))
          self.assertFalse(predict._is_on_cpu(x_jit()))


if __name__ == '__main__':
  jtu.absltest.main()
