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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from functools import partial
from jax import test_util as jtu
from jax.api import jit
from jax.config import config as jax_config
import jax.numpy as np
import jax.random as random
from neural_tangents import stax
from neural_tangents.utils import empirical

jax_config.parse_flags_with_absl()


TAYLOR_MATRIX_SHAPES = [(3, 3), (4, 4)]
TAYLOR_RANDOM_SAMPLES = 10

STANDARD = 'FLAT'
POOLING = 'POOLING'

TRAIN_SHAPES = [(4, 4), (4, 8), (8, 8), (6, 4, 4, 3), (4, 4, 4, 3)]
TEST_SHAPES = [(2, 4), (6, 8), (16, 8), (2, 4, 4, 3), (2, 4, 4, 3)]
NETWORK = [STANDARD, STANDARD, STANDARD, STANDARD, POOLING]
OUTPUT_LOGITS = [1, 2, 3]

CONVOLUTION_CHANNELS = 256


def _build_network(input_shape, network, out_logits):
  if len(input_shape) == 1:
    assert network == 'FLAT'
    return stax.Dense(out_logits, W_std=2.0, b_std=0.5)
  elif len(input_shape) == 3:
    if network == 'POOLING':
      return stax.serial(
          stax.Conv(CONVOLUTION_CHANNELS, (3, 3), W_std=2.0, b_std=0.05),
          stax.GlobalAvgPool(), stax.Dense(out_logits, W_std=2.0, b_std=0.5))
    elif network == 'FLAT':
      return stax.serial(
          stax.Conv(CONVOLUTION_CHANNELS, (3, 3), W_std=2.0, b_std=0.05),
          stax.Flatten(), stax.Dense(out_logits, W_std=2.0, b_std=0.5))
    else:
      raise ValueError('Unexpected network type found: {}'.format(network))
  else:
    raise ValueError('Expected flat or image test input.')


def _kernel_fns(key, input_shape, network, out_logits):
  init_fn, f, _ = _build_network(input_shape, network, out_logits)
  _, params = init_fn(key, (-1,) + input_shape)
  implicit_kernel_fn = jit(empirical.empirical_implicit_ntk_fn(f))
  direct_kernel_fn = jit(empirical.empirical_direct_ntk_fn(f))

  return (partial(implicit_kernel_fn, params=params),
          partial(direct_kernel_fn, params=params))


KERNELS = {}
for o in OUTPUT_LOGITS:
  KERNELS['empirical_logits_{}'.format(o)] = partial(_kernel_fns, out_logits=o)


class EmpiricalTest(jtu.JaxTestCase):

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name': '_{}'.format(shape),
          'shape': shape
      } for shape in TAYLOR_MATRIX_SHAPES))
  def testLinearization(self, shape):
    # We use a three layer deep linear network for testing.
    def f(x, params):
      w1, w2, b = params
      return 0.5 * np.dot(np.dot(x.T, w1), x) + np.dot(w2, x) + b

    def f_lin_exact(x0, x, params):
      w1, w2, b = params
      f0 = f(x0, params)
      dx = x - x0
      return f0 + np.dot(np.dot(x0.T, w1) + w2, dx)

    key = random.PRNGKey(0)
    key, s1, s2, s3, = random.split(key, 4)
    w1 = random.normal(s1, shape)
    w1 = 0.5 * (w1 + w1.T)
    w2 = random.normal(s2, shape)
    b = random.normal(s3, (shape[-1],))
    params = (w1, w2, b)

    key, split = random.split(key)
    x0 = random.normal(split, (shape[-1],))

    f_lin = empirical.linearize(f, x0)

    for _ in range(TAYLOR_RANDOM_SAMPLES):
      key, split = random.split(key)
      x = random.normal(split, (shape[-1],))
      self.assertAllClose(f_lin_exact(x0, x, params), f_lin(x, params), True)

  # pylint: disable=g-complex-comprehension
  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name': '_{}'.format(shape),
          'shape': shape
      } for shape in TAYLOR_MATRIX_SHAPES))
  def testTaylorExpansion(self, shape):
    # We use a three layer deep linear network for testing.
    def f(x, params):
      w1, w2, b = params
      return 0.5 * np.dot(np.dot(x.T, w1), x) + np.dot(w2, x) + b

    def f_lin_exact(x0, x, params):
      w1, w2, b = params
      f0 = f(x0, params)
      dx = x - x0
      return f0 + np.dot(np.dot(x0.T, w1) + w2, dx)

    def f_2_exact(x0, x, params):
      w1, w2, b = params
      dx = x - x0
      return f_lin_exact(x0, x, params) + 0.5 * np.dot(np.dot(dx.T, w1), dx)

    key = random.PRNGKey(0)
    key, s1, s2, s3, = random.split(key, 4)
    w1 = random.normal(s1, shape)
    w1 = 0.5 * (w1 + w1.T)
    w2 = random.normal(s2, shape)
    b = random.normal(s3, (shape[-1],))
    params = (w1, w2, b)

    key, split = random.split(key)
    x0 = random.normal(split, (shape[-1],))

    f_lin = empirical.taylor_expand(f, x0, 1)
    f_2 = empirical.taylor_expand(f, x0, 2)

    for _ in range(TAYLOR_RANDOM_SAMPLES):
      key, split = random.split(key)
      x = random.normal(split, (shape[-1],))
      self.assertAllClose(f_lin_exact(x0, x, params), f_lin(x, params), True)
      self.assertAllClose(f_2_exact(x0, x, params), f_2(x, params), True)

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

    implicit, direct = kernel_fn(key, train_shape[1:], network)

    g = implicit(data_self, None)
    g_direct = direct(data_self, None)
    self.assertAllClose(g, g_direct, check_dtypes=False)

    g = implicit(data_other, data_self)
    g_direct = direct(data_other, data_self)
    self.assertAllClose(g, g_direct, check_dtypes=False)

if __name__ == '__main__':
  jtu.absltest.main()
