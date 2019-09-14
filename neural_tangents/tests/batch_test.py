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
from functools import partial
from jax import test_util as jtu
from jax.api import jit
from jax.config import config as jax_config
import jax.random as random
from neural_tangents import stax
from neural_tangents.utils import batch
from neural_tangents.utils import empirical
from neural_tangents.utils import utils

jax_config.parse_flags_with_absl()

STANDARD = 'FLAT'
POOLING = 'POOLING'
INTERMEDIATE_CONV = 'INTERMEDIATE_CONV'

# TODO(schsam): Add a pooling test when multiple inputs are supported in
# Conv + Pooling.
TRAIN_SHAPES = [(4, 4), (4, 8), (8, 8), (6, 4, 4, 3), (4, 4, 4, 3)]
TEST_SHAPES = [(2, 4), (6, 8), (16, 8), (2, 4, 4, 3), (2, 4, 4, 3)]
NETWORK = [STANDARD, STANDARD, STANDARD, STANDARD, INTERMEDIATE_CONV]
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
    elif network == 'INTERMEDIATE_CONV':
      return stax.Conv(CONVOLUTION_CHANNELS, (3, 3), W_std=2.0, b_std=0.05)
    else:
      raise ValueError('Unexpected network type found: {}'.format(network))
  else:
    raise ValueError('Expected flat or image test input.')


def _empirical_kernel(key, input_shape, network, out_logits):
  init_fn, f, _ = _build_network(input_shape, network, out_logits)
  _, params = init_fn(key, (-1,) + input_shape)
  ker_fun = jit(empirical.get_ntk_fun_empirical(f))

  return partial(ker_fun, params=params)


def _theoretical_kernel(unused_key, input_shape, network, just_theta):
  _, _, _ker_fun = _build_network(input_shape, network, 1)

  @jit
  def ker_fun(x1, x2=None):
    k = _ker_fun(x1, x2)
    if just_theta:
      return k.ntk
    return k

  return ker_fun


KERNELS = {}
for o in OUTPUT_LOGITS:
  KERNELS['empirical_logits_{}'.format(o)] = partial(
      _empirical_kernel, out_logits=o)
KERNELS['theoretical'] = partial(_theoretical_kernel, just_theta=True)
KERNELS['theoretical_pytree'] = partial(_theoretical_kernel, just_theta=False)


def _test_kernel_against_batched(cls, ker_fun, batched_ker_fun, train, test):

  g = ker_fun(train, None)
  g_b = batched_ker_fun(train, None)

  if isinstance(g, stax.Kernel):
    cls.assertAllClose(g.var1, g_b.var1, check_dtypes=True)
    cls.assertAllClose(g.nngp, g_b.nngp, check_dtypes=True)
    cls.assertAllClose(g.ntk, g_b.ntk, check_dtypes=True)
  else:
    cls.assertAllClose(g, g_b, check_dtypes=True)

  g = ker_fun(train, test)
  g_b = batched_ker_fun(train, test)

  if isinstance(g, stax.Kernel):
    cls.assertAllClose(g.var1, g_b.var1, check_dtypes=True)
    cls.assertAllClose(g.var2, g_b.var2, check_dtypes=True)
    cls.assertAllClose(g.nngp, g_b.nngp, check_dtypes=True)
    cls.assertAllClose(g.ntk, g_b.ntk, check_dtypes=True)
  else:
    cls.assertAllClose(g, g_b, check_dtypes=True)


class BatchTest(jtu.JaxTestCase):

  # pylint: disable=g-complex-comprehension
  @jtu.parameterized.named_parameters(
      jtu.cases_from_list(
          {
              'testcase_name':
                  '_train_shape={}_test_shape={}_network={}_{}'.format(
                      train, test, network, name),
              'train_shape':
                  train,
              'test_shape':
                  test,
              'network':
                  network,
              'name':
                  name,
              'ker_fun':
                  ker_fun
          }
          for train, test, network in zip(TRAIN_SHAPES, TEST_SHAPES, NETWORK)
          for name, ker_fun in KERNELS.items()))
  def testSerial(self, train_shape, test_shape, network, name, ker_fun):
    key = random.PRNGKey(0)
    key, self_split, other_split = random.split(key, 3)
    data_self = random.normal(self_split, train_shape)
    data_other = random.normal(other_split, test_shape)

    ker_fun = ker_fun(key, train_shape[1:], network)
    kernel_batched = batch._serial(ker_fun, batch_size=2)

    _test_kernel_against_batched(self, ker_fun, kernel_batched, data_self,
                                 data_other)

  # NOTE(schsam): Here and below we exclude tests involving convolutions and
  # empirical kernels since we need to add a batching rule to JAX to proceed.
  # I'll do that in a followup PR and then enable the tests.
  @jtu.parameterized.named_parameters(
      jtu.cases_from_list(
          {
              'testcase_name':
                  '_train_shape={}_test_shape={}_network={}_{}'.format(
                      train, test, network, name),
              'train_shape':
                  train,
              'test_shape':
                  test,
              'network':
                  network,
              'name':
                  name,
              'ker_fun':
                  ker_fun
          }
          for train, test, network in zip(TRAIN_SHAPES, TEST_SHAPES, NETWORK)
          for name, ker_fun in KERNELS.items()))
  # if (len(train) == 2 or name[:5] == 'theor')))
  def testParallel(self, train_shape, test_shape, network, name, ker_fun):
    utils.stub_out_pmap(batch, 2)

    key = random.PRNGKey(0)
    key, self_split, other_split = random.split(key, 3)
    data_self = random.normal(self_split, train_shape)
    data_other = random.normal(other_split, test_shape)

    ker_fun = ker_fun(key, train_shape[1:], network)
    kernel_batched = batch._parallel(ker_fun)

    _test_kernel_against_batched(self, ker_fun, kernel_batched, data_self,
                                 data_other)

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list(
          {
              'testcase_name':
                  '_train_shape={}_test_shape={}_network={}_{}'.format(
                      train, test, network, name),
              'train_shape':
                  train,
              'test_shape':
                  test,
              'network':
                  network,
              'name':
                  name,
              'ker_fun':
                  ker_fun
          }
          for train, test, network in zip(TRAIN_SHAPES, TEST_SHAPES, NETWORK)
          for name, ker_fun in KERNELS.items()
          if len(train) == 2))
  def testComposition(self, train_shape, test_shape, network, name, ker_fun):
    utils.stub_out_pmap(batch, 2)

    key = random.PRNGKey(0)
    key, self_split, other_split = random.split(key, 3)
    data_self = random.normal(self_split, train_shape)
    data_other = random.normal(other_split, test_shape)

    ker_fun = ker_fun(key, train_shape[1:], network)

    kernel_batched = batch._parallel(batch._serial(ker_fun, batch_size=2))
    _test_kernel_against_batched(self, ker_fun, kernel_batched, data_self,
                                 data_other)

    kernel_batched = batch._serial(batch._parallel(ker_fun), batch_size=2)
    _test_kernel_against_batched(self, ker_fun, kernel_batched, data_self,
                                 data_other)

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list(
          {
              'testcase_name':
                  '_train_shape={}_test_shape={}_network={}_{}'.format(
                      train, test, network, name),
              'train_shape':
                  train,
              'test_shape':
                  test,
              'network':
                  network,
              'name':
                  name,
              'ker_fun':
                  ker_fun
          }
          for train, test, network in zip(TRAIN_SHAPES, TEST_SHAPES, NETWORK)
          for name, ker_fun in KERNELS.items()
          if len(train) == 2))
  def testAutomatic(self, train_shape, test_shape, network, name, ker_fun):
    utils.stub_out_pmap(batch, 2)

    key = random.PRNGKey(0)
    key, self_split, other_split = random.split(key, 3)
    data_self = random.normal(self_split, train_shape)
    data_other = random.normal(other_split, test_shape)

    ker_fun = ker_fun(key, train_shape[1:], network)

    kernel_batched = batch.batch(ker_fun, batch_size=2)
    _test_kernel_against_batched(self, ker_fun, kernel_batched, data_self,
                                 data_other)

    kernel_batched = batch.batch(ker_fun, batch_size=2, store_on_device=False)
    _test_kernel_against_batched(self, ker_fun, kernel_batched, data_self,
                                 data_other)


if __name__ == '__main__':
  jtu.absltest.main()
