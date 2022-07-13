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

"""Tests for `neural_tangents/predict.py`."""

import math

from absl.testing import absltest
from jax import grad
from jax import jit
from jax import random
from jax import vmap
from jax.config import config
from jax.example_libraries import optimizers
from jax.flatten_util import ravel_pytree
import jax.numpy as np
import jax.tree_util
import neural_tangents as nt
from neural_tangents import predict, stax
from neural_tangents._src.predict import _is_on_cpu
from tests import test_utils


config.parse_flags_with_absl()
config.update('jax_numpy_rank_promotion', 'raise')


OUTPUT_LOGITS = [2]

GETS = ('ntk', 'nngp', ('ntk', 'nngp'))

RTOL = 0.01
ATOL = 0.01

if not config.read('jax_enable_x64'):
  RTOL = 0.02
  ATOL = 0.02

FLAT = 'FLAT'
POOLING = 'POOLING'

# TODO(schsam): Add a pooling test when multiple inputs are supported in
# Conv + Pooling.
TRAIN_SIZES = [4, 8]
TEST_SIZES = [6, 2]
INPUT_SHAPES = [(8,), (4, 4, 3)]
NETWORK = [FLAT, FLAT]

CONVOLUTION_CHANNELS = 256

test_utils.update_test_tolerance()


def _build_network(input_shape, network, out_logits):
  if len(input_shape) == 1:
    assert network == FLAT
    return stax.serial(
        stax.Dense(4096, W_std=1.2, b_std=0.05), stax.Erf(),
        stax.Dense(out_logits, W_std=1.2, b_std=0.05))
  elif len(input_shape) == 3:
    if network == POOLING:
      return stax.serial(
          stax.Conv(CONVOLUTION_CHANNELS, (3, 3), W_std=2.0, b_std=0.05),
          stax.GlobalAvgPool(), stax.Dense(out_logits, W_std=2.0, b_std=0.05))
    elif network == FLAT:
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
  _kernel_fn = nt.empirical_kernel_fn(f, trace_axes=(), vmap_axes=0)
  kernel_fn = lambda x1, x2, get: _kernel_fn(x1, x2, get, params)
  return params, f, jit(kernel_fn, static_argnames='get')


def _theoretical_kernel(key, input_shape, network, out_logits):
  init_fn, f, kernel_fn = _build_network(input_shape, network, out_logits)
  _, params = init_fn(key, (-1,) + input_shape)
  return params, f, jit(kernel_fn, static_argnames='get')


KERNELS = {
    'empirical': _empirical_kernel,
    'theoretical': _theoretical_kernel,
}


class PredictTest(test_utils.NeuralTangentsTestCase):

  def _test_zero_time(self, predictor, fx_train_0, fx_test_0, g_td, momentum):
    fx_train_t0, fx_test_t0 = predictor(0.0, fx_train_0, fx_test_0, g_td)
    self.assertAllClose(fx_train_0, fx_train_t0)
    self.assertAllClose(fx_test_0, fx_test_t0)
    fx_train_only_t0 = predictor(0.0, fx_train_0, None, g_td)
    self.assertAllClose(fx_train_0, fx_train_only_t0)

    if momentum is not None:
      # Test state-based prediction
      state_0 = predict.ODEState(fx_train_0, fx_test_0)  # pytype:disable=wrong-arg-count
      state_t0 = predictor(0.0, state_0, None, g_td)
      self.assertAllClose(state_0.fx_train, state_t0.fx_train)
      self.assertAllClose(state_0.fx_test, state_t0.fx_test)

      state_train_only_0 = predict.ODEState(fx_train_0)  # pytype:disable=wrong-arg-count
      state_train_only_t0 = predictor(0.0, state_0, None, g_td)
      self.assertAllClose(state_train_only_0.fx_train,
                          state_train_only_t0.fx_train)

  def _test_inf_time(self, predictor, fx_train_0, fx_test_0, g_td, y_train):
    # Test infinite-time prediction
    pr_inf = predictor(np.inf, fx_train_0)
    self.assertAllClose(pr_inf, y_train, check_dtypes=False)
    self.assertAllClose(pr_inf, predictor(None, fx_train_0))
    self.assertAllClose(predictor(np.inf, fx_train_0, fx_test_0, g_td),
                        predictor(None, fx_train_0, fx_test_0, g_td))

  def _test_multi_step(self, predictor, fx_train_0, fx_test_0, g_td, momentum):
    # Test multi-time prediction
    ts = np.arange(6).reshape((2, 1, 3))

    fx_train_single, fx_test_single = predictor(ts, fx_train_0, fx_test_0, g_td)

    fx_train_concat, fx_test_concat = [], []
    for t in ts.ravel():
      fx_train_concat_t, fx_test_concat_t = predictor(t, fx_train_0, fx_test_0,
                                                      g_td)
      fx_train_concat += [fx_train_concat_t]
      fx_test_concat += [fx_test_concat_t]
    fx_train_concat = np.stack(fx_train_concat).reshape(
        ts.shape + fx_train_single.shape[ts.ndim:])
    fx_test_concat = np.stack(fx_test_concat).reshape(
        ts.shape + fx_test_single.shape[ts.ndim:])

    self.assertAllClose(fx_train_concat, fx_train_single)
    self.assertAllClose(fx_test_concat, fx_test_single)

    if momentum is not None:
      state_0 = predict.ODEState(fx_train_0, fx_test_0)  # pytype:disable=wrong-arg-count
      t_1 = (0, 0, 2)
      state_1 = predictor(ts[t_1], state_0, None, g_td)
      self.assertAllClose(fx_train_single[t_1], state_1.fx_train)
      self.assertAllClose(fx_test_single[t_1], state_1.fx_test)

      t_max = (-1,) * ts.ndim
      state_max = predictor(ts[t_max] - ts[t_1], state_1, None, g_td)
      self.assertAllClose(fx_train_single[t_max], state_max.fx_train)
      self.assertAllClose(fx_test_single[t_max], state_max.fx_test)

  @classmethod
  def _get_inputs(cls, out_logits, test_shape, train_shape):
    key = random.PRNGKey(0)
    key, split = random.split(key)
    x_train = random.normal(split, train_shape)
    key, split = random.split(key)
    y_train = np.array(
        random.bernoulli(split, shape=(train_shape[0], out_logits)), np.float32)
    key, split = random.split(key)
    x_test = random.normal(split, test_shape)
    return key, x_test, x_train, y_train

  @test_utils.product(
      train_size=TRAIN_SIZES,
      test_size=TEST_SIZES,
      input_shape=INPUT_SHAPES,
      network=NETWORK,
      out_logits=OUTPUT_LOGITS,
      kernel_type=list(KERNELS.keys()),
      momentum=[None, 0.9],
      learning_rate=[0.0002],
      t=[5],
      loss=['mse_analytic', 'mse'],
  )
  def testNTKGDPrediction(
      self,
      train_size,
      test_size,
      input_shape,
      network,
      out_logits,
      kernel_type,
      momentum,
      learning_rate,
      t,
      loss
  ):
    train_shape = (train_size, *input_shape)
    test_shape = (test_size, *input_shape)
    key, x_test, x_train, y_train = self._get_inputs(out_logits, test_shape,
                                                     train_shape)

    fn_and_kernel = KERNELS[kernel_type]
    params, f, ntk = fn_and_kernel(key, train_shape[1:], network, out_logits)

    g_dd = ntk(x_train, None, 'ntk')
    g_td = ntk(x_test, x_train, 'ntk')

    # Regress to an MSE loss.
    loss_fn = lambda y, y_hat: 0.5 * np.mean((y - y_hat)**2)
    grad_loss = jit(grad(lambda params, x: loss_fn(f(params, x), y_train)))

    trace_axes = () if g_dd.ndim == 4 else (-1,)
    if loss == 'mse_analytic':
      if momentum is not None:
        raise absltest.SkipTest(momentum)
      predictor = predict.gradient_descent_mse(g_dd, y_train,
                                               learning_rate=learning_rate,
                                               trace_axes=trace_axes)
    elif loss == 'mse':
      predictor = predict.gradient_descent(loss_fn, g_dd, y_train,
                                           learning_rate=learning_rate,
                                           momentum=momentum,
                                           trace_axes=trace_axes)
    else:
      raise NotImplementedError(loss)

    predictor = jit(predictor)

    fx_train_0 = f(params, x_train)
    fx_test_0 = f(params, x_test)

    self._test_zero_time(predictor, fx_train_0, fx_test_0, g_td, momentum)
    self._test_multi_step(predictor, fx_train_0, fx_test_0, g_td, momentum)
    if loss == 'mse_analytic':
      self._test_inf_time(predictor, fx_train_0, fx_test_0, g_td, y_train)

    if momentum is None:
      opt_init, opt_update, get_params = optimizers.sgd(learning_rate)
    else:
      opt_init, opt_update, get_params = optimizers.momentum(learning_rate,
                                                             momentum)

    opt_state = opt_init(params)
    for i in range(t):
      params = get_params(opt_state)
      opt_state = opt_update(i, grad_loss(params, x_train), opt_state)

    params = get_params(opt_state)

    fx_train_nn, fx_test_nn = f(params, x_train), f(params, x_test)
    fx_train_t, fx_test_t = predictor(t, fx_train_0, fx_test_0, g_td)

    self.assertAllClose(fx_train_nn, fx_train_t, rtol=RTOL, atol=ATOL)
    self.assertAllClose(fx_test_nn, fx_test_t, rtol=RTOL, atol=ATOL)

  @classmethod
  def _cov_empirical(cls, x):
    return np.einsum('itjk,itlk->tjl', x, x, optimize=True) / (x.shape[0] *
                                                               x.shape[-1])

  @test_utils.product(
      train_size=TRAIN_SIZES[:1],
      test_size=TEST_SIZES[:1],
      input_shape=INPUT_SHAPES[:1],
      out_logits=[1],
  )
  def testNTKMeanCovPrediction(
      self,
      train_size,
      test_size,
      input_shape,
      out_logits,
  ):
    train_shape = (train_size, *input_shape)
    test_shape = (test_size, *input_shape)
    key, x_test, x_train, y_train = self._get_inputs(out_logits, test_shape,
                                                     train_shape)
    init_fn, f, kernel_fn = stax.serial(
        stax.Dense(512, W_std=1.2, b_std=0.05), stax.Erf(),
        stax.Dense(out_logits, W_std=1.2, b_std=0.05))

    reg = 1e-6
    predictor = predict.gradient_descent_mse_ensemble(kernel_fn, x_train,
                                                      y_train, diag_reg=reg)
    ts = np.array([1., 5., 10.])

    fx_test_inf, cov_test_inf = predictor(ts, x_test, 'ntk', True)
    self.assertEqual(cov_test_inf.shape[1], x_test.shape[0])
    self.assertGreater(np.min(np.linalg.eigh(cov_test_inf)[0]), -1e-8)

    fx_train_inf, cov_train_inf = predictor(ts, None, 'ntk', True)
    self.assertEqual(cov_train_inf.shape[1], x_train.shape[0])
    self.assertGreater(np.min(np.linalg.eigh(cov_train_inf)[0]), -1e-8)

    _kernel_fn = nt.empirical_kernel_fn(f)
    kernel_fn = jit(lambda x1, x2, params: _kernel_fn(x1, x2, 'ntk', params))

    def predict_empirical(key):
      _, params = init_fn(key, train_shape)
      g_dd = kernel_fn(x_train, None, params)
      g_td = kernel_fn(x_test, x_train, params)
      predict_fn = predict.gradient_descent_mse(g_dd, y_train, diag_reg=reg)
      fx_train_0 = f(params, x_train)
      fx_test_0 = f(params, x_test)
      return predict_fn(ts, fx_train_0, fx_test_0, g_td)

    def predict_mc(count, key):
      key = random.split(key, count)
      fx_train, fx_test = vmap(predict_empirical)(key)
      fx_train_mean = np.mean(fx_train, axis=0, keepdims=True)
      fx_test_mean = np.mean(fx_test, axis=0, keepdims=True)

      fx_train_centered = fx_train - fx_train_mean
      fx_test_centered = fx_test - fx_test_mean

      cov_train = PredictTest._cov_empirical(fx_train_centered)
      cov_test = PredictTest._cov_empirical(fx_test_centered)

      return fx_train_mean, fx_test_mean, cov_train, cov_test

    fx_train_mc, fx_test_mc, cov_train_mc, cov_test_mc = predict_mc(4096, key)
    tol = 0.05

    assert_close = lambda a, b: self.assertAllClose(ravel_pytree(a)[0],
                                                    ravel_pytree(b)[0],
                                                    atol=tol,
                                                    rtol=tol)

    assert_close(fx_train_mc, fx_train_inf)
    assert_close(cov_train_mc, cov_train_inf)
    assert_close(cov_test_mc, cov_test_inf)
    assert_close(fx_test_mc, fx_test_inf)

  @test_utils.product(
      train_size=TRAIN_SIZES[:-1],
      test_size=TEST_SIZES[:-1],
      input_shape=INPUT_SHAPES,
      network=NETWORK[:-1],
      out_logits=OUTPUT_LOGITS,
  )
  def testGradientDescentMseEnsembleGet(
      self,
      train_size,
      test_size,
      input_shape,
      network,
      out_logits,
  ):
    train_shape = (train_size, *input_shape)
    test_shape = (test_size, *input_shape)
    _, x_test, x_train, y_train = self._get_inputs(out_logits, test_shape,
                                                   train_shape)
    _, _, kernel_fn = _build_network(train_shape[1:], network, out_logits)

    predictor = predict.gradient_descent_mse_ensemble(kernel_fn,
                                                      x_train,
                                                      y_train,
                                                      diag_reg=0.)
    for x in [None, 'x_test']:
      with self.subTest(x=x):
        x = x if x is None else x_test
        out = predictor(None, x, 'ntk', compute_cov=True)
        assert isinstance(out, predict.Gaussian)

        out = predictor(1., x, 'nngp', compute_cov=True)
        assert isinstance(out, predict.Gaussian)

        out = predictor(np.array([0., 1.]), x, ('ntk',), compute_cov=True)
        assert len(out) == 1 and isinstance(out[0], predict.Gaussian)

        out = predictor(2., x, ('ntk', 'nngp'), compute_cov=True)
        assert (len(out) == 2 and isinstance(out[0], predict.Gaussian) and
                isinstance(out[1], predict.Gaussian))

        out2 = predictor(2., x, ('nngp', 'ntk'), compute_cov=True)
        self.assertAllClose(out[0], out2[1])
        self.assertAllClose(out[1], out2[0])

  @test_utils.product(
      train_size=TRAIN_SIZES[:-1],
      test_size=TEST_SIZES[:-1],
      input_shape=INPUT_SHAPES[:-1],
      network=NETWORK[:-1],
      out_logits=OUTPUT_LOGITS,
      get=GETS
  )
  def testInfiniteTimeAgreement(
      self,
      train_size,
      test_size,
      input_shape,
      network,
      out_logits,
      get
  ):
    train_shape = (train_size, *input_shape)
    test_shape = (test_size, *input_shape)
    _, x_test, x_train, y_train = self._get_inputs(out_logits, test_shape,
                                                   train_shape)
    _, _, kernel_fn = _build_network(train_shape[1:], network, out_logits)

    reg = 0.
    predictor = predict.gradient_descent_mse_ensemble(kernel_fn,
                                                      x_train,
                                                      y_train,
                                                      diag_reg=reg)

    for x in (None, 'x_test'):
      with self.subTest(x=x):
        x = x if x is None else x_test
        fin = predictor(t=np.inf, x_test=x, get=get, compute_cov=True)
        inf = predictor(t=None, x_test=x, get=get, compute_cov=True)
        self.assertAllClose(inf, fin)
        if x is None:
          fin_x = predictor(t=np.inf, x_test=x_train, get=get, compute_cov=True)
          inf_x = predictor(t=None, x_test=x_train, get=get, compute_cov=True)
          self.assertAllClose(inf, inf_x)
          self.assertAllClose(inf_x, fin_x)

  @test_utils.product(
      train_size=TRAIN_SIZES,
      test_size=TEST_SIZES,
      input_shape=INPUT_SHAPES,
      network=NETWORK,
      out_logits=OUTPUT_LOGITS,
  )
  def testZeroTimeAgreement(
      self,
      train_size,
      test_size,
      input_shape,
      network,
      out_logits,
  ):
    """Test that the NTK and NNGP agree at t=0."""
    train_shape = (train_size, *input_shape)
    test_shape = (test_size, *input_shape)
    _, x_test, x_train, y_train = self._get_inputs(out_logits, test_shape,
                                                   train_shape)
    _, _, ker_fun = _build_network(train_shape[1:], network, out_logits)

    reg = 1e-7
    predictor = predict.gradient_descent_mse_ensemble(
        ker_fun,
        x_train,
        y_train,
        diag_reg=reg)

    for x in (None, 'x_test'):
      with self.subTest(x=x):
        x = x if x is None else x_test
        zero = predictor(t=0.0, x_test=x, get=('NTK', 'NNGP'), compute_cov=True)
        if x is None:
          k = ker_fun(x_train, None, get='nngp')
          ref = (np.zeros_like(y_train, k.dtype), k)
        else:
          ref = (np.zeros((test_shape[0], out_logits)),
                 ker_fun(x_test, None, get='nngp'))

        self.assertAllClose((ref,) * 2, zero, check_dtypes=False)
        if x is None:
          zero_x = predictor(t=0.0, x_test=x_train, get=('NTK', 'NNGP'),
                             compute_cov=True)
          self.assertAllClose((ref,) * 2, zero_x)

  @classmethod
  def _always_ntk(cls, ker_fun):
    def always_ntk(x1, x2, get=('nngp', 'ntk')):
      out = ker_fun(x1, x2, get=('nngp', 'ntk'))
      if get == 'nngp' or get == 'ntk':
        return out.ntk
      else:
        return out._replace(nngp=out.ntk)
    return always_ntk

  @test_utils.product(
      train_size=TRAIN_SIZES,
      test_size=TEST_SIZES,
      input_shape=INPUT_SHAPES,
      network=NETWORK,
      out_logits=OUTPUT_LOGITS,
  )
  def testNTK_NTKNNGPAgreement(
      self,
      train_size,
      test_size,
      input_shape,
      network,
      out_logits,
  ):
    train_shape = (train_size, *input_shape)
    test_shape = (test_size, *input_shape)
    _, x_test, x_train, y_train = self._get_inputs(out_logits, test_shape,
                                                   train_shape)
    _, _, ker_fun = _build_network(train_shape[1:], network, out_logits)

    reg = 1e-7
    predictor = predict.gradient_descent_mse_ensemble(ker_fun,
                                                      x_train,
                                                      y_train,
                                                      diag_reg=reg)

    ts = np.logspace(-2, 8, 10).reshape((5, 2))

    for t in (None, 'ts'):
      for x in (None, 'x_test'):
        with self.subTest(t=t, x=x):
          x = x if x is None else x_test
          t = t if t is None else ts

          ntk = predictor(t=t, get='ntk', x_test=x)

          # Test time broadcasting
          if t is not None:
            ntk_ind = np.array([predictor(t=t, get='ntk', x_test=x)
                                for t in t.ravel()]).reshape(
                                    t.shape + ntk.shape[2:])
            self.assertAllClose(ntk_ind, ntk)

          always_ntk = self._always_ntk(ker_fun)
          predictor_ntk = predict.gradient_descent_mse_ensemble(always_ntk,
                                                                x_train,
                                                                y_train,
                                                                diag_reg=reg)

          ntk_nngp = predictor_ntk(t=t, get='nngp', x_test=x)

          # Test if you use nngp equations with ntk, you get the same mean
          self.assertAllClose(ntk, ntk_nngp)

          # Next test that if you go through the NTK code path, but with only
          # the NNGP kernel, we recreate the NNGP dynamics.
          # Create a hacked kernel function that always returns the nngp kernel
          def always_nngp(x1, x2, get=('nngp', 'ntk')):
            out = ker_fun(x1, x2, get=('nngp', 'ntk'))
            if get == 'nngp' or get == 'ntk':
              return out.nngp
            else:
              return out._replace(ntk=out.nngp)

          predictor_nngp = predict.gradient_descent_mse_ensemble(always_nngp,
                                                                 x_train,
                                                                 y_train,
                                                                 diag_reg=reg)

          nngp_cov = predictor(t=t,
                               get='nngp',
                               x_test=x,
                               compute_cov=True).covariance

          # test time broadcasting for covariance
          nngp_ntk_cov = predictor_nngp(t=t,
                                        get='ntk',
                                        x_test=x,
                                        compute_cov=True).covariance
          if t is not None:
            nngp_ntk_cov_ind = np.array(
                [predictor_nngp(t=t,
                                get='ntk',
                                x_test=x,
                                compute_cov=True).covariance for
                 t in t.ravel()]).reshape(t.shape + nngp_cov.shape[2:])
            self.assertAllClose(nngp_ntk_cov_ind, nngp_ntk_cov)

          # Test if you use ntk equations with nngp, you get the same cov
          # Although, due to accumulation of numerical errors, only roughly.
          self.assertAllClose(nngp_cov, nngp_ntk_cov)

  @test_utils.product(
      train_size=TRAIN_SIZES,
      test_size=TEST_SIZES,
      input_shape=INPUT_SHAPES,
      network=NETWORK,
      out_logits=OUTPUT_LOGITS,
  )
  def testPredCovPosDef(
      self,
      train_size,
      test_size,
      input_shape,
      network,
      out_logits,
  ):
    train_shape = (train_size, *input_shape)
    test_shape = (test_size, *input_shape)
    _, x_test, x_train, y_train = self._get_inputs(out_logits, test_shape,
                                                   train_shape)
    _, _, ker_fun = _build_network(train_shape[1:], network, out_logits)

    ts = np.logspace(-3, 3, 10)
    predict_fn_mse_ens = predict.gradient_descent_mse_ensemble(
        ker_fun, x_train, y_train)

    for get in ('nngp', 'ntk'):
      for x in (None, 'x_test'):
        for t in (None, 'ts'):
          with self.subTest(get=get, x=x, t=t):
            cov = predict_fn_mse_ens(t=t if t is None else ts,
                                     get=get,
                                     x_test=x if x is None else x_test,
                                     compute_cov=True).covariance

            self.assertAllClose(cov, np.moveaxis(cov, -1, -2))
            self.assertGreater(np.min(np.linalg.eigh(cov)[0]), -1e-4)

  @test_utils.product(
      train_size=TRAIN_SIZES[:1],
      test_size=TEST_SIZES[:1],
      input_shape=INPUT_SHAPES[:1],
      out_logits=[1],
  )
  def testTrainedEnsemblePredCov(
      self,
      train_size,
      test_size,
      input_shape,
      out_logits
  ):
    training_steps = 1000
    learning_rate = 0.1
    ensemble_size = 1024

    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(128, W_std=1.2, b_std=0.05), stax.Erf(),
        stax.Dense(out_logits, W_std=1.2, b_std=0.05))

    opt_init, opt_update, get_params = optimizers.sgd(learning_rate)
    opt_update = jit(opt_update)

    train_shape = (train_size, *input_shape)
    test_shape = (test_size, *input_shape)

    key, x_test, x_train, y_train = self._get_inputs(out_logits, test_shape,
                                                     train_shape)
    predict_fn_mse_ens = predict.gradient_descent_mse_ensemble(
        kernel_fn,
        x_train,
        y_train,
        learning_rate=learning_rate,
        diag_reg=0.)

    train = (x_train, y_train)
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
    tol = 0.08

    for x in [None, 'x_test']:
      with self.subTest(x=x):
        x = x if x is None else x_test
        x_fin = x_train if x is None else x_test
        ensemble_fx = vmap(apply_fn, (0, None))(params, x_fin)

        mean_emp = np.mean(ensemble_fx, axis=0, keepdims=True)
        mean_subtracted = ensemble_fx - mean_emp
        cov_emp = np.einsum(
            'ijk,ilk->jl', mean_subtracted, mean_subtracted, optimize=True) / (
                mean_subtracted.shape[0] * mean_subtracted.shape[-1])

        ntk = predict_fn_mse_ens(training_steps, x, 'ntk', compute_cov=True)
        self.assertAllClose(ravel_pytree(mean_emp)[0],
                            ravel_pytree(ntk.mean)[0], rtol=tol, atol=tol)
        self.assertAllClose(cov_emp, ntk.covariance, rtol=tol, atol=tol)

  def testGradientDescentMseEnsembleTrain(self):
    key = random.PRNGKey(1)
    x = random.normal(key, (8, 4, 6, 3))
    _, _, kernel_fn = stax.serial(stax.Conv(1, (2, 2)),
                                  stax.Relu(),
                                  stax.Conv(1, (2, 1)))
    y = random.normal(key, (8, 2, 5, 1))
    predictor = predict.gradient_descent_mse_ensemble(kernel_fn, x, y,
                                                      diagonal_spatial=False)

    for t in [None, np.array([0., 1., 10.])]:
      with self.subTest(t=t):
        y_none = predictor(t, None, None, compute_cov=True)
        y_x = predictor(t, x, None, compute_cov=True)
        self.assertAllClose(y_none, y_x, rtol=0.04, atol=0.04)

  def testGpInference(self):
    reg = 1e-5
    key = random.PRNGKey(1)
    x_train = random.normal(key, (4, 2))
    init_fn, apply_fn, kernel_fn_analytic = stax.serial(
        stax.Dense(32, 2., 0.5),
        stax.Relu(),
        stax.Dense(10, 2., 0.5))
    kernel_fn_empirical = nt.empirical_kernel_fn(apply_fn)
    y_train = random.normal(key, (4, 10))
    for kernel_fn_is_analytic in [True, False]:
      if kernel_fn_is_analytic:
        kernel_fn = kernel_fn_analytic
      else:
        _, params = init_fn(key, x_train.shape)
        def kernel_fn(x1, x2, get):
          return kernel_fn_empirical(x1, x2, get, params)

      for get in [None,
                  'nngp', 'ntk',
                  ('nngp',), ('ntk',),
                  ('nngp', 'ntk'), ('ntk', 'nngp')]:
        k_dd = kernel_fn(x_train, None, get)

        gp_inference = predict.gp_inference(k_dd, y_train, diag_reg=reg)
        gd_ensemble = predict.gradient_descent_mse_ensemble(kernel_fn,
                                                            x_train,
                                                            y_train,
                                                            diag_reg=reg)
        for x_test in [None, 'x_test']:
          x_test = None if x_test is None else random.normal(key, (8, 2))
          k_td = None if x_test is None else kernel_fn(x_test, x_train, get)

          for compute_cov in [True, False]:
            with self.subTest(kernel_fn_is_analytic=kernel_fn_is_analytic,
                              get=get,
                              x_test=x_test if x_test is None else 'x_test',
                              compute_cov=compute_cov):
              if compute_cov:
                nngp_tt = (True if x_test is None else
                           kernel_fn(x_test, None, 'nngp'))
              else:
                nngp_tt = None

              out_ens = gd_ensemble(None, x_test, get, compute_cov)
              out_ens_inf = gd_ensemble(np.inf, x_test, get, compute_cov)
              tol = 0.35 if jax.default_backend() == 'tpu' else 0.08
              self.assertAllClose(out_ens_inf, out_ens, rtol=tol, atol=tol)

              if (get is not None and
                  'nngp' not in get and
                  compute_cov and
                  k_td is not None):
                with self.assertRaises(ValueError):
                  out_gp_inf = gp_inference(get=get, k_test_train=k_td,
                                            k_test_test=nngp_tt)
              else:
                out_gp_inf = gp_inference(get=get, k_test_train=k_td,
                                          k_test_test=nngp_tt)
                self.assertAllClose(out_ens, out_gp_inf)

      # Test NTKGP
      for get in [(), ('nngp',), ('ntk',), ('nngp', 'ntk'), ('ntk', 'nngp')]:
        ntkgp_get = get + ('ntkgp',)
        if 'ntk' not in get:
          get += ('ntk',)
        k_dd = kernel_fn(x_train, None, get)

        always_ntk = self._always_ntk(kernel_fn)
        always_ntk_k_dd = always_ntk(x_train, None, get)

        gp_inference = predict.gp_inference(k_dd, y_train, diag_reg=reg)

        always_ntk_gp_inference = predict.gp_inference(always_ntk_k_dd, y_train,
                                                       diag_reg=reg)
        gd_ensemble = predict.gradient_descent_mse_ensemble(kernel_fn,
                                                            x_train,
                                                            y_train,
                                                            diag_reg=reg)
        for x_test in [None, 'x_test']:
          x_test = None if x_test is None else random.normal(key, (8, 2))
          k_td = None if x_test is None else kernel_fn(x_test, x_train, get)
          always_ntk_k_td = None if x_test is None else always_ntk(x_test,
                                                                   x_train)

          for compute_cov in [True, False]:
            with self.subTest(kernel_fn_is_analytic=kernel_fn_is_analytic,
                              get=ntkgp_get,
                              x_test=x_test if x_test is None else 'x_test',
                              compute_cov=compute_cov):
              if compute_cov:
                k_tt = (True if x_test is None else
                        kernel_fn(x_test, None, get))
                always_ntk_tt = (True if x_test is None else
                                 kernel_fn(x_test, None, 'ntk'))

              else:
                k_tt = None
                always_ntk_tt = None

              if ('nngp' not in get and
                  'ntk' in ntkgp_get and
                  compute_cov and
                  k_td is not None):
                with self.assertRaises(ValueError):
                  out_gp_inf = gp_inference(get=ntkgp_get, k_test_train=k_td,
                                            k_test_test=k_tt)
              else:
                out_gp_inf = gp_inference(get=ntkgp_get, k_test_train=k_td,
                                          k_test_test=k_tt)
                out_ens = gd_ensemble(None, x_test, get, compute_cov)
                out_always_ntk_gp_inf = always_ntk_gp_inference(
                    get='nngp',
                    k_test_train=always_ntk_k_td,
                    k_test_test=always_ntk_tt)

                # Compare ntkgp predictions to nngp code, with hacked kernel
                for g in ntkgp_get:
                  self.assertAllClose(getattr(out_gp_inf, g),
                                      (getattr(out_ens, g) if g != 'ntkgp'
                                       else out_always_ntk_gp_inf))

  def testPredictOnCPU(self):
    x_train = random.normal(random.PRNGKey(1), (4, 4, 4, 2))
    x_test = random.normal(random.PRNGKey(1), (8, 4, 4, 2))

    y_train = random.uniform(random.PRNGKey(1), (4, 2))

    _, _, kernel_fn = stax.serial(
        stax.Conv(1, (3, 3)), stax.Relu(), stax.Flatten(), stax.Dense(1))

    for store_on_device in [False, True]:
      for device_count in [0, 1]:
        for get in ['ntk', 'nngp', ('nngp', 'ntk'), ('ntk', 'nngp')]:
          for x in [None, 'x_test']:
            with self.subTest(
                store_on_device=store_on_device,
                device_count=device_count,
                get=get,
                x=x):
              kernel_fn_batched = nt.batch(kernel_fn, 2, device_count,
                                           store_on_device)
              predictor = predict.gradient_descent_mse_ensemble(
                  kernel_fn_batched, x_train, y_train)

              x = x if x is None else x_test
              predict_none = predictor(None, x, get, compute_cov=True)
              predict_inf = predictor(np.inf, x, get, compute_cov=True)
              self.assertAllClose(predict_none, predict_inf)

              if x is not None:
                on_cpu = not store_on_device or jax.default_backend() == 'cpu'

                def is_on_cpu(x):
                  return jax.tree_util.tree_all(
                      jax.tree_map(
                          lambda x: 'cpu' in str(x.device_buffer.device()
                                                 ).lower(),
                          x))

                self.assertEqual(on_cpu, is_on_cpu(predict_inf))
                self.assertEqual(on_cpu, is_on_cpu(predict_none))

  def testPredictND(self):
    n_chan = 6
    key = random.PRNGKey(1)
    im_shape = (5, 4, 3)
    n_train = 2
    n_test = 2
    x_train = random.normal(key, (n_train,) + im_shape)
    y_train = random.uniform(key, (n_train, 3, 2, n_chan))
    init_fn, apply_fn, _ = stax.Conv(n_chan, (3, 2), (1, 2))
    _, params = init_fn(key, x_train.shape)
    fx_train_0 = apply_fn(params, x_train)

    for trace_axes in [(),
                       (-1,),
                       (-2,),
                       (-3,),
                       (0, 1),
                       (2, 3),
                       (2,),
                       (1, 3),
                       (0, -1),
                       (0, 0, -3),
                       (0, 1, 2, 3),
                       (0, 1, -1, 2)]:
      for ts in [None, np.arange(6).reshape((2, 3))]:
        for x in [None, 'x_test']:
          with self.subTest(trace_axes=trace_axes, ts=ts, x=x):
            t_shape = ts.shape if ts is not None else ()
            y_test_shape = t_shape + (n_test,) + y_train.shape[1:]
            y_train_shape = t_shape + y_train.shape
            x = x if x is None else random.normal(key, (n_test,) + im_shape)
            fx_test_0 = None if x is None else apply_fn(params, x)

            kernel_fn = nt.empirical_kernel_fn(apply_fn, trace_axes=trace_axes)

            kernel_fn = jit(kernel_fn, static_argnames='get')
            ntk_train_train = kernel_fn(x_train, None, 'ntk', params)
            if x is not None:
              ntk_test_train = kernel_fn(x, x_train, 'ntk', params)

            loss = lambda x, y: 0.5 * np.mean(x - y)**2
            predict_fn_mse = predict.gradient_descent_mse(ntk_train_train,
                                                          y_train,
                                                          trace_axes=trace_axes)

            predict_fn_mse_ensemble = predict.gradient_descent_mse_ensemble(
                kernel_fn, x_train, y_train, trace_axes=trace_axes,
                params=params
            )

            if x is None:
              p_train_mse = predict_fn_mse(ts, fx_train_0)
            else:
              p_train_mse, p_test_mse = predict_fn_mse(
                  ts, fx_train_0, fx_test_0, ntk_test_train)
              self.assertAllClose(y_test_shape, p_test_mse.shape)
            self.assertAllClose(y_train_shape, p_train_mse.shape)

            p_nngp_mse_ens, p_ntk_mse_ens = predict_fn_mse_ensemble(
                ts, x, ('nngp', 'ntk'), compute_cov=True)
            ref_shape = y_train_shape if x is None else y_test_shape
            self.assertAllClose(ref_shape, p_ntk_mse_ens.mean.shape)
            self.assertAllClose(ref_shape, p_nngp_mse_ens.mean.shape)

            if ts is not None:
              predict_fn = predict.gradient_descent(
                  loss, ntk_train_train, y_train, trace_axes=trace_axes)

              if x is None:
                p_train = predict_fn(ts, fx_train_0)
              else:
                p_train, p_test = predict_fn(
                    ts, fx_train_0, fx_test_0, ntk_test_train)
                self.assertAllClose(y_test_shape, p_test.shape)
              self.assertAllClose(y_train_shape, p_train.shape)

  @test_utils.product(
      train_size=TRAIN_SIZES,
      input_shape=INPUT_SHAPES,
      network=NETWORK,
      out_logits=OUTPUT_LOGITS,
      kernel_type=list(KERNELS.keys()),
      lr_factor=[0.5, 1., 3.],
      momentum=[0., 0.1, 0.5, 0.9]
  )
  def testMaxLearningRate(
      self,
      train_size,
      input_shape,
      network,
      out_logits,
      kernel_type,
      lr_factor,
      momentum
  ):

    key = random.PRNGKey(0)

    key, split = random.split(key)
    if len(input_shape) == 1:
      train_shape = (train_size * 5, input_shape[0] * 10)
    else:
      train_shape = (16, 8, 8, 3)
    x_train = random.normal(split, train_shape)

    key, split = random.split(key)
    y_train = np.array(
        random.bernoulli(split, shape=(train_shape[0], out_logits)), np.float32)

    # Regress to an MSE loss.
    loss = lambda params, x: 0.5 * np.mean((f(params, x) - y_train) ** 2)
    grad_loss = jit(grad(loss))

    def get_loss(opt_state):
      return loss(get_params(opt_state), x_train)

    steps = 30

    fn_and_kernel = KERNELS[kernel_type]
    params, f, ntk = fn_and_kernel(key, train_shape[1:], network, out_logits)
    g_dd = ntk(x_train, None, 'ntk')

    step_size = predict.max_learning_rate(
        g_dd, y_train_size=y_train.size, momentum=momentum) * lr_factor
    opt_init, opt_update, get_params = optimizers.momentum(step_size,
                                                           mass=momentum)

    opt_state = opt_init(params)

    init_loss = get_loss(opt_state)

    for i in range(steps):
      params = get_params(opt_state)
      opt_state = opt_update(i, grad_loss(params, x_train), opt_state)

    trained_loss = get_loss(opt_state)
    loss_ratio = trained_loss / (init_loss + 1e-12)
    if lr_factor < 1.:
      self.assertLess(loss_ratio, 0.1)
    elif lr_factor == 1:
      # At the threshold, the loss decays slowly
      self.assertLess(loss_ratio, 1.)
    if lr_factor > 2.:
      if not math.isnan(loss_ratio):
        self.assertGreater(loss_ratio, 10.)


class PredictKwargsTest(test_utils.NeuralTangentsTestCase):

  @test_utils.product(
      do_batch=[True, False],
      mode=['analytic', 'mc', 'empirical']
  )
  def test_kwargs(self, do_batch, mode):
    rng = random.PRNGKey(1)

    x_train = random.normal(rng, (8, 7, 10))
    x_test = random.normal(rng, (4, 7, 10))
    y_train = random.normal(rng, (8, 1))

    rng_train, rng_test = random.split(rng, 2)

    pattern_train = random.normal(rng, (8, 7, 7))
    pattern_test = random.normal(rng, (4, 7, 7))

    diag_reg = 1e-4

    if jax.default_backend() == 'tpu':
      atol = 3e-3
      rtol = 4e-2
      width = 256
    else:
      atol = 5e-4
      rtol = 1e-2
      width = 64

    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(width, W_std=2**0.5),
        stax.Relu(),
        stax.Dropout(rate=0.7),
        stax.Aggregate(),
        stax.GlobalAvgPool(),
        stax.Dense(width)
    )

    kw_dd = dict(pattern=(pattern_train, pattern_train))
    kw_td = dict(pattern=(pattern_test, pattern_train))
    kw_tt = dict(pattern=(pattern_test, pattern_test))

    if mode == 'mc':
      kernel_fn = nt.monte_carlo_kernel_fn(init_fn, apply_fn, rng, 2,
                                           batch_size=2 if do_batch else 0)

    elif mode == 'empirical':
      kernel_fn = nt.empirical_kernel_fn(apply_fn)
      if do_batch:
        raise absltest.SkipTest('Batching of empirical kernel is not '
                                'implemented with keyword arguments.')

      for kw in (kw_dd, kw_td, kw_tt):
        kw.update(dict(params=init_fn(rng, x_train.shape)[1],
                       get=('nngp', 'ntk')))

      kw_dd.update(dict(rng=(rng_train, None)))
      kw_td.update(dict(rng=(rng_test, rng_train)))
      kw_tt.update(dict(rng=(rng_test, None)))

    elif mode == 'analytic':
      if do_batch:
        kernel_fn = nt.batch(kernel_fn, batch_size=2)

    else:
      raise ValueError(mode)

    k_dd = kernel_fn(x_train, None, **kw_dd)
    k_td = kernel_fn(x_test, x_train, **kw_td)
    k_tt = kernel_fn(x_test, None, **kw_tt)

    # Infinite time NNGP/NTK.
    predict_fn_gp = predict.gp_inference(
        k_dd,
        y_train,
        diag_reg=diag_reg
    )
    out_gp = predict_fn_gp(k_test_train=k_td, k_test_test=k_tt.nngp)

    if mode == 'empirical':
      for kw in (kw_dd, kw_td, kw_tt):
        kw.pop('get')

    predict_fn_ensemble = predict.gradient_descent_mse_ensemble(
        kernel_fn,
        x_train,
        y_train,
        diag_reg=diag_reg,
        **kw_dd
    )
    out_ensemble = predict_fn_ensemble(x_test=x_test, compute_cov=True, **kw_tt)
    self.assertAllClose(out_gp, out_ensemble)

    # Finite time NTK test.
    predict_fn_mse = predict.gradient_descent_mse(k_dd.ntk, y_train)
    out_mse = predict_fn_mse(t=1.,
                             fx_train_0=None,
                             fx_test_0=0.,
                             k_test_train=k_td.ntk)
    out_ensemble = predict_fn_ensemble(t=1.,
                                       get='ntk',
                                       x_test=x_test,
                                       compute_cov=False,
                                       **kw_tt)
    self.assertAllClose(out_mse, out_ensemble, atol=atol, rtol=rtol)

    # Finite time NTK train.
    out_mse = predict_fn_mse(t=0.5,
                             fx_train_0=0.,
                             fx_test_0=None,
                             k_test_train=k_td.ntk)
    out_ensemble = predict_fn_ensemble(t=0.5,
                                       get='ntk',
                                       x_test=None,
                                       compute_cov=False,
                                       **kw_dd)
    self.assertAllClose(out_mse, out_ensemble, atol=atol, rtol=rtol)

    # Finite time NNGP test.
    predict_fn_mse = predict.gradient_descent_mse(k_dd.nngp, y_train)
    out_mse = predict_fn_mse(t=1.,
                             fx_train_0=None,
                             fx_test_0=0.,
                             k_test_train=k_td.nngp)
    out_ensemble = predict_fn_ensemble(t=1.,
                                       get='nngp',
                                       x_test=x_test,
                                       compute_cov=False,
                                       **kw_tt)
    self.assertAllClose(out_mse, out_ensemble, atol=atol, rtol=rtol)

    # Finite time NNGP train.
    out_mse = predict_fn_mse(t=5.,
                             fx_train_0=0.,
                             fx_test_0=None,
                             k_test_train=k_td.nngp)
    out_ensemble = predict_fn_ensemble(t=5.,
                                       get='nngp',
                                       x_test=None,
                                       compute_cov=False,
                                       **kw_dd)
    self.assertAllClose(out_mse, out_ensemble, atol=atol, rtol=rtol)


class IsOnCpuTest(test_utils.NeuralTangentsTestCase):

  def test_is_on_cpu(self):
    dtypes = [np.float16, np.float32]
    float64 = jax.dtypes.canonicalize_dtype(np.float64)
    if float64 != np.float32:
      dtypes += [float64]

    for dtype in dtypes:
      with self.subTest(dtype=dtype):

        def x():
          return random.normal(random.PRNGKey(1), (2, 3), dtype)

        def x_cpu():
          return jax.device_get(random.normal(random.PRNGKey(1), (2, 3), dtype))

        x_jit = jit(x)
        # x_cpu_jit = jit(x_cpu)
        x_cpu_jit_cpu = jit(x_cpu, backend='cpu')

        self.assertTrue(_is_on_cpu(x_cpu()))
        # TODO(mattjj): re-enable this when device_put under jit works
        # self.assertTrue(predict._is_on_cpu(x_cpu_jit()))
        self.assertTrue(_is_on_cpu(x_cpu_jit_cpu()))

        if jax.default_backend() == 'cpu':
          self.assertTrue(_is_on_cpu(x()))
          self.assertTrue(_is_on_cpu(x_jit()))
        else:
          self.assertFalse(_is_on_cpu(x()))
          self.assertFalse(_is_on_cpu(x_jit()))


if __name__ == '__main__':
  absltest.main()
