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

"""Tests for `utils/empirical.py`."""

from functools import partial
import unittest

from absl.testing import absltest
from jax import test_util as jtu
from jax.api import jit
from jax.config import config as jax_config
import jax.numpy as np
import jax.random as random
from neural_tangents import stax
from neural_tangents.utils import empirical
from neural_tangents.utils import test_utils
from neural_tangents.utils import utils


jax_config.parse_flags_with_absl()


TAYLOR_MATRIX_SHAPES = [(3, 3), (4, 4)]
TAYLOR_RANDOM_SAMPLES = 10

FLAT = 'FLAT'
POOLING = 'POOLING'
CONV = 'CONV'

TRAIN_SHAPES = [(4, 4), (4, 8), (8, 8), (6, 4, 4, 3), (4, 4, 4, 3),
                (4, 4, 4, 3)]
TEST_SHAPES = [(2, 4), (6, 8), (16, 8), (2, 4, 4, 3), (2, 4, 4, 3),
               (2, 4, 4, 3)]
NETWORK = [FLAT, FLAT, FLAT, FLAT, POOLING,
           CONV]
OUTPUT_LOGITS = [1, 2, 3]

CONVOLUTION_CHANNELS = 8
test_utils.update_test_tolerance()


def _build_network(input_shape, network, out_logits):
  if len(input_shape) == 1:
    assert network == FLAT
    return stax.Dense(out_logits, W_std=2.0, b_std=0.5)
  elif len(input_shape) == 3:
    if network == POOLING:
      return stax.serial(
          stax.Conv(CONVOLUTION_CHANNELS, (3, 3), W_std=2.0, b_std=0.05),
          stax.GlobalAvgPool(), stax.Dense(out_logits, W_std=2.0, b_std=0.5))
    elif network == CONV:
      return stax.serial(
          stax.Conv(CONVOLUTION_CHANNELS, (1, 2), W_std=1.5, b_std=0.1),
          stax.Relu(),
          stax.Conv(CONVOLUTION_CHANNELS, (3, 2), W_std=2.0, b_std=0.05),
      )
    elif network == FLAT:
      return stax.serial(
          stax.Conv(CONVOLUTION_CHANNELS, (3, 3), W_std=2.0, b_std=0.05),
          stax.Flatten(), stax.Dense(out_logits, W_std=2.0, b_std=0.5))
    else:
      raise ValueError('Unexpected network type found: {}'.format(network))
  else:
    raise ValueError('Expected flat or image test input.')


def _kernel_fns(key,
                input_shape,
                network,
                out_logits,
                diagonal_axes,
                trace_axes):
  init_fn, f, _ = _build_network(input_shape, network, out_logits)
  _, params = init_fn(key, (-1,) + input_shape)
  implicit_kernel_fn = empirical.empirical_implicit_ntk_fn(f, trace_axes,
                                                           diagonal_axes)
  direct_kernel_fn = empirical.empirical_direct_ntk_fn(f, trace_axes,
                                                       diagonal_axes)
  nngp_kernel_fn = empirical.empirical_nngp_fn(f, trace_axes, diagonal_axes)

  implicit_kernel_fn = jit(implicit_kernel_fn)
  direct_kernel_fn = jit(direct_kernel_fn)
  nngp_kernel_fn = jit(nngp_kernel_fn)

  return (partial(implicit_kernel_fn, params=params),
          partial(direct_kernel_fn, params=params),
          partial(nngp_kernel_fn, params=params))


KERNELS = {}
for o in OUTPUT_LOGITS:
  KERNELS['empirical_logits_{}'.format(o)] = partial(_kernel_fns, out_logits=o)


class EmpiricalTest(jtu.JaxTestCase):

  # We use a three layer deep linear network for testing.
  @classmethod
  def f(cls, x, params, do_alter, do_shift_x=True):
    w1, w2, b = params
    if do_alter:
      b *= 2.
      w1 += 5.
      w2 /= 0.9
    if do_shift_x:
      x = x * 2 + 1.
    return 0.5 * np.dot(np.dot(x.T, w1), x) + np.dot(w2, x) + b

  @classmethod
  def f_lin_exact(cls, x0, x, params, do_alter, do_shift_x=True):
    w1, w2, b = params
    f0 = EmpiricalTest.f(x0, params, do_alter, do_shift_x)
    if do_shift_x:
      x0 = x0 * 2 + 1.
      x = x * 2 + 1.
    dx = x - x0
    if do_alter:
      b *= 2.
      w1 += 5.
      w2 /= 0.9
    return f0 + np.dot(np.dot(x0.T, w1) + w2, dx)

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name': '_{}'.format(shape),
          'shape': shape
      } for shape in TAYLOR_MATRIX_SHAPES))
  def testLinearization(self, shape):
    key = random.PRNGKey(0)
    key, s1, s2, s3, = random.split(key, 4)
    w1 = random.normal(s1, shape)
    w1 = 0.5 * (w1 + w1.T)
    w2 = random.normal(s2, shape)
    b = random.normal(s3, (shape[-1],))
    params = (w1, w2, b)

    key, split = random.split(key)
    x0 = random.normal(split, (shape[-1],))

    f_lin = empirical.linearize(EmpiricalTest.f, x0)

    for _ in range(TAYLOR_RANDOM_SAMPLES):
      for do_alter in [True, False]:
        for do_shift_x in [True, False]:
          key, split = random.split(key)
          x = random.normal(split, (shape[-1],))
          self.assertAllClose(EmpiricalTest.f_lin_exact(x0, x, params, do_alter,
                                                        do_shift_x=do_shift_x),
                              f_lin(x, params, do_alter, do_shift_x=do_shift_x))

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name': '_{}'.format(shape),
          'shape': shape
      } for shape in TAYLOR_MATRIX_SHAPES))
  def testTaylorExpansion(self, shape):

    def f_2_exact(x0, x, params, do_alter, do_shift_x=True):
      w1, w2, b = params
      f_lin = EmpiricalTest.f_lin_exact(x0, x, params, do_alter, do_shift_x)
      if do_shift_x:
        x0 = x0 * 2 + 1.
        x = x * 2 + 1.
      if do_alter:
        b *= 2.
        w1 += 5.
        w2 /= 0.9
      dx = x - x0
      return f_lin + 0.5 * np.dot(np.dot(dx.T, w1), dx)

    key = random.PRNGKey(0)
    key, s1, s2, s3, = random.split(key, 4)
    w1 = random.normal(s1, shape)
    w1 = 0.5 * (w1 + w1.T)
    w2 = random.normal(s2, shape)
    b = random.normal(s3, (shape[-1],))
    params = (w1, w2, b)

    key, split = random.split(key)
    x0 = random.normal(split, (shape[-1],))

    f_lin = empirical.taylor_expand(EmpiricalTest.f, x0, 1)
    f_2 = empirical.taylor_expand(EmpiricalTest.f, x0, 2)

    for _ in range(TAYLOR_RANDOM_SAMPLES):
      for do_alter in [True, False]:
        for do_shift_x in [True, False]:
          key, split = random.split(key)
          x = random.normal(split, (shape[-1],))
          self.assertAllClose(EmpiricalTest.f_lin_exact(x0, x, params, do_alter,
                                                        do_shift_x=do_shift_x),
                              f_lin(x, params, do_alter, do_shift_x=do_shift_x))
          self.assertAllClose(f_2_exact(x0, x, params, do_alter,
                                        do_shift_x=do_shift_x),
                              f_2(x, params, do_alter, do_shift_x=do_shift_x))

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name': '_train_shape={}_test_shape={}_network={}_{}'.format(
              train, test, network, name),
          'train_shape': train,
          'test_shape': test,
          'network': network,
          'name': name,
          'kernel_fn': kernel_fn
      } for train, test, network in zip(TRAIN_SHAPES, TEST_SHAPES, NETWORK)
                          for name, kernel_fn in KERNELS.items()))
  def testNTKAgainstDirect(
      self, train_shape, test_shape, network, name, kernel_fn):
    key = random.PRNGKey(0)
    key, self_split, other_split = random.split(key, 3)
    data_self = random.normal(self_split, train_shape)
    data_other = random.normal(other_split, test_shape)

    implicit, direct, _ = kernel_fn(key, train_shape[1:], network,
                                    diagonal_axes=(), trace_axes=())

    g = implicit(data_self, None)
    g_direct = direct(data_self, None)
    self.assertAllClose(g, g_direct)

    g = implicit(data_other, data_self)
    g_direct = direct(data_other, data_self)
    self.assertAllClose(g, g_direct)

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name': '_diagonal_axes={}_trace_axes={}'.format(
              diagonal_axes, trace_axes),
          'diagonal_axes': diagonal_axes,
          'trace_axes': trace_axes,
      }
                          for diagonal_axes in [(),
                                                (0,),
                                                (0, 1),
                                                (0, 1, 2),
                                                (0, 1, 2, 3),]
                          for trace_axes in [(),
                                               (0,),
                                               (0, 1),
                                               (-1,),
                                               (1,),
                                               (0, -1),
                                               (-1, -2),
                                               (0, 1, 2, 3),
                                               (3, 1, 2, 0),
                                               (1, 2, 3),
                                               (-3, -2),
                                               (-3, -1),
                                               (-2, -4)]))
  def testAxes(self, diagonal_axes, trace_axes):
    key = random.PRNGKey(0)
    key, self_split, other_split = random.split(key, 3)
    data_self = random.normal(self_split, (4, 5, 6, 3))
    data_other = random.normal(other_split, (2, 5, 6, 3))

    _diagonal_axes = utils.canonicalize_axis(diagonal_axes, data_self)
    _trace_axes = utils.canonicalize_axis(trace_axes, data_self)

    if any(d == c for d in _diagonal_axes for c in _trace_axes):
      raise unittest.SkipTest(
          'diagonal axes must be different from channel axes.')

    implicit, direct, nngp = KERNELS['empirical_logits_3'](
        key,
        (5, 6, 3),
        CONV,
        diagonal_axes=diagonal_axes,
        trace_axes=trace_axes)

    n_marg = len(_diagonal_axes)
    n_chan = len(_trace_axes)

    g = implicit(data_self, None)
    g_direct = direct(data_self, None)
    g_nngp = nngp(data_self, None)

    self.assertAllClose(g, g_direct)
    self.assertEqual(g_nngp.shape, g.shape)
    self.assertEqual(2 * (data_self.ndim - n_chan) - n_marg, g_nngp.ndim)

    if 0 not in _trace_axes and 0 not in _diagonal_axes:
      g = implicit(data_other, data_self)
      g_direct = direct(data_other, data_self)
      g_nngp = nngp(data_other, data_self)

      self.assertAllClose(g, g_direct)
      self.assertEqual(g_nngp.shape, g.shape)
      self.assertEqual(2 * (data_self.ndim - n_chan) - n_marg, g_nngp.ndim)


if __name__ == '__main__':
  absltest.main()
