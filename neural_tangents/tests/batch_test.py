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

from absl.testing import absltest

from functools import partial
from jax import test_util as jtu
from jax.api import jit
from jax.config import config as jax_config
import jax.numpy as np
import jax.random as random
from jax.tree_util import tree_map
from neural_tangents import stax
from neural_tangents.utils import batch
from neural_tangents.utils import empirical
from neural_tangents.utils import test_utils
from neural_tangents.utils.kernel import Kernel


jax_config.parse_flags_with_absl()

FLAT = 'FLAT'
POOLING = 'POOLING'
INTERMEDIATE_CONV = 'INTERMEDIATE_CONV'

# TODO(schsam): Add a pooling test when multiple inputs are supported in
# Conv + Pooling.
TRAIN_SHAPES = [(2, 4), (4, 8), (8, 8), (8, 4, 4, 3), (4, 3, 3, 3)]
TEST_SHAPES = [(2, 4), (2, 8), (16, 8), (2, 4, 4, 3), (2, 3, 3, 3)]
NETWORK = [FLAT, FLAT, FLAT, FLAT, INTERMEDIATE_CONV]
OUTPUT_LOGITS = [1, 2, 3]
CONVOLUTION_CHANNELS = 4
WIDTH = 1024
RTOL = 1e-2
test_utils.update_test_tolerance(f64_tol=5e-5)


def _build_network(input_shape, network, out_logits, use_dropout):
  dropout = stax.Dropout(0.9, mode='train') if use_dropout else stax.Identity()
  if len(input_shape) == 1:
    assert network == 'FLAT'
    return stax.serial(
        stax.Dense(WIDTH, W_std=2.0, b_std=0.5), dropout,
        stax.Dense(out_logits, W_std=2.0, b_std=0.5))
  elif len(input_shape) == 3:
    if network == POOLING:
      return stax.serial(
          stax.Conv(CONVOLUTION_CHANNELS, (2, 2), W_std=2.0, b_std=0.05),
          stax.GlobalAvgPool(), dropout,
          stax.Dense(out_logits, W_std=2.0, b_std=0.5))
    elif network == FLAT:
      return stax.serial(
          stax.Conv(CONVOLUTION_CHANNELS, (2, 2), W_std=2.0, b_std=0.05),
          stax.Flatten(), dropout, stax.Dense(out_logits, W_std=2.0, b_std=0.5))
    elif network == INTERMEDIATE_CONV:
      return stax.Conv(CONVOLUTION_CHANNELS, (2, 2), W_std=2.0, b_std=0.05)
    else:
      raise ValueError('Unexpected network type found: {}'.format(network))
  else:
    raise ValueError('Expected flat or image test input.')


def _empirical_kernel(key, input_shape, network, out_logits, use_dropout):
  init_fn, f, _ = _build_network(input_shape, network, out_logits, use_dropout)
  key, split = random.split(key)
  _, params = init_fn(key, (-1,) + input_shape)
  kernel_fn = jit(empirical.empirical_ntk_fn(f))
  return partial(kernel_fn, params=params, keys=split)


def _theoretical_kernel(unused_key, input_shape, network, just_theta,
                        use_dropout):
  _, _, _kernel_fn = _build_network(input_shape, network, 1, use_dropout)

  @jit
  def kernel_fn(x1, x2=None):
    get_all = None
    k = _kernel_fn(x1, x2, 'ntk') if just_theta else _kernel_fn(x1, x2, get_all)
    return k

  return kernel_fn


KERNELS = {}
for o in OUTPUT_LOGITS:
  KERNELS['empirical_logits_{}'.format(o)] = partial(
      _empirical_kernel, out_logits=o, use_dropout=False)
KERNELS['theoretical'] = partial(
    _theoretical_kernel, just_theta=True, use_dropout=True)
KERNELS['theoretical_pytree'] = partial(
    _theoretical_kernel, just_theta=False, use_dropout=True)


def _test_kernel_against_batched(cls,
                                 kernel_fn,
                                 batched_kernel_fn,
                                 train,
                                 test,
                                 is_parallel_only=False):

  g = kernel_fn(train, None)
  g_b = batched_kernel_fn(train, None)

  if is_parallel_only and isinstance(g_b, Kernel):
    # In the parallel setting, `x1_is_x2` is not computed correctly when x1==x2.
    g_b = g_b.replace(x1_is_x2=g.x1_is_x2)

  cls.assertAllClose(g, g_b)

  g = kernel_fn(train, test)
  g_b = batched_kernel_fn(train, test)
  if is_parallel_only and isinstance(g_b, Kernel):
    g_b = g_b.replace(x1_is_x2=g.x1_is_x2)
  cls.assertAllClose(g, g_b)


class BatchTest(test_utils.NeuralTangentsTestCase):

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              '_train_shape={}_test_shape={}_network={}_{}_batch_size={}'
              .format(train, test, network, name, batch_size),
          'train_shape':
              train,
          'test_shape':
              test,
          'network':
              network,
          'name':
              name,
          'kernel_fn':
              kernel_fn,
          'batch_size':
              batch_size
      } for train, test, network in zip(TRAIN_SHAPES, TEST_SHAPES, NETWORK)
                          for name, kernel_fn in KERNELS.items()
                          for batch_size in [2, 8]))
  def testSerial(self, train_shape, test_shape, network, name, kernel_fn,
                 batch_size):
    key = random.PRNGKey(0)
    key, self_split, other_split = random.split(key, 3)
    data_self = random.normal(self_split, train_shape)
    data_other = random.normal(other_split, test_shape)
    kernel_fn = kernel_fn(key, train_shape[1:], network)
    kernel_batched = batch._serial(kernel_fn, batch_size=batch_size)

    _test_kernel_against_batched(self, kernel_fn, kernel_batched, data_self,
                                 data_other)

  # We also exclude tests for dropout + parallel. It is not clear what is the
  # best way to handle this case.
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
              'kernel_fn':
                  kernel_fn
          }
          for train, test, network in zip(TRAIN_SHAPES, TEST_SHAPES, NETWORK)
          for name, kernel_fn in KERNELS.items()))
  def testParallel(self, train_shape, test_shape, network, name, kernel_fn):
    test_utils.stub_out_pmap(batch, 2)
    key = random.PRNGKey(0)
    key, self_split, other_split = random.split(key, 3)
    data_self = random.normal(self_split, train_shape)
    data_other = random.normal(other_split, test_shape)

    kernel_fn = kernel_fn(key, train_shape[1:], network, use_dropout=False)
    kernel_batched = batch._parallel(kernel_fn)

    _test_kernel_against_batched(self, kernel_fn, kernel_batched, data_self,
                                 data_other, True)

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              '_train_shape={}_test_shape={}_network={}_{}_batch_size={}'
              .format(train, test, network, name, batch_size),
          'train_shape':
              train,
          'test_shape':
              test,
          'network':
              network,
          'name':
              name,
          'kernel_fn':
              kernel_fn,
          'batch_size':
              batch_size
      } for train, test, network in zip(TRAIN_SHAPES, TEST_SHAPES, NETWORK)
                          for name, kernel_fn in KERNELS.items()
                          for batch_size in [2, 8]))
  def testComposition(self, train_shape, test_shape, network, name, kernel_fn,
                      batch_size):
    test_utils.stub_out_pmap(batch, 2)

    key = random.PRNGKey(0)
    key, self_split, other_split = random.split(key, 3)
    data_self = random.normal(self_split, train_shape)
    data_other = random.normal(other_split, test_shape)

    kernel_fn = kernel_fn(key, train_shape[1:], network)

    kernel_batched = batch._parallel(
        batch._serial(kernel_fn, batch_size=batch_size))
    _test_kernel_against_batched(self, kernel_fn, kernel_batched, data_self,
                                 data_other)

    kernel_batched = batch._serial(
        batch._parallel(kernel_fn), batch_size=batch_size)
    _test_kernel_against_batched(self, kernel_fn, kernel_batched, data_self,
                                 data_other)

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              '_train_shape={}_test_shape={}_network={}_{}_batch_size={}'
              .format(train, test, network, name, batch_size),
          'train_shape':
              train,
          'test_shape':
              test,
          'network':
              network,
          'name':
              name,
          'kernel_fn':
              kernel_fn,
          'batch_size':
              batch_size
      } for train, test, network in zip(TRAIN_SHAPES, TEST_SHAPES, NETWORK)
                          for name, kernel_fn in KERNELS.items()
                          for batch_size in [2, 8]))
  def testAutomatic(self, train_shape, test_shape, network, name, kernel_fn,
                    batch_size):
    test_utils.stub_out_pmap(batch, 2)

    key = random.PRNGKey(0)
    key, self_split, other_split = random.split(key, 3)
    data_self = random.normal(self_split, train_shape)
    data_other = random.normal(other_split, test_shape)

    kernel_fn = kernel_fn(key, train_shape[1:], network)

    kernel_batched = batch.batch(kernel_fn, batch_size=batch_size)
    _test_kernel_against_batched(self, kernel_fn, kernel_batched, data_self,
                                 data_other)

    kernel_batched = batch.batch(
        kernel_fn, batch_size=batch_size, store_on_device=False)
    _test_kernel_against_batched(self, kernel_fn, kernel_batched, data_self,
                                 data_other)

  def _test_analytic_kernel_composition(self, batching_fn):
    # Check Fully-Connected.
    rng = random.PRNGKey(0)
    rng_self, rng_other = random.split(rng)
    x_self = random.normal(rng_self, (8, 10))
    x_other = random.normal(rng_other, (2, 10))
    Block = stax.serial(stax.Dense(256), stax.Relu())

    _, _, ker_fn = Block
    ker_fn = batching_fn(ker_fn)

    _, _, composed_ker_fn = stax.serial(Block, Block)

    ker_out = ker_fn(ker_fn(x_self))
    composed_ker_out = composed_ker_fn(x_self)
    if batching_fn == batch._parallel:
      # In the parallel setting, `x1_is_x2` is not computed correctly
      # when x1==x2.
      composed_ker_out = composed_ker_out.replace(x1_is_x2=ker_out.x1_is_x2)
    self.assertAllClose(ker_out, composed_ker_out)

    ker_out = ker_fn(ker_fn(x_self, x_other))
    composed_ker_out = composed_ker_fn(x_self, x_other)
    if batching_fn == batch._parallel:
      composed_ker_out = composed_ker_out.replace(x1_is_x2=ker_out.x1_is_x2)
    self.assertAllClose(ker_out, composed_ker_out)

    # Check convolutional + pooling.
    x_self = random.normal(rng, (8, 10, 10, 3))
    x_other = random.normal(rng, (2, 10, 10, 3))

    Block = stax.serial(stax.Conv(256, (2, 2)), stax.Relu())
    Readout = stax.serial(stax.GlobalAvgPool(), stax.Dense(10))

    block_ker_fn, readout_ker_fn = Block[2], Readout[2]
    _, _, composed_ker_fn = stax.serial(Block, Readout)
    block_ker_fn = batching_fn(block_ker_fn)
    readout_ker_fn = batching_fn(readout_ker_fn)

    ker_out = readout_ker_fn(block_ker_fn(x_self))
    composed_ker_out = composed_ker_fn(x_self)
    if batching_fn == batch._parallel:
      composed_ker_out = composed_ker_out.replace(x1_is_x2=ker_out.x1_is_x2)
    self.assertAllClose(ker_out, composed_ker_out)
    ker_out = readout_ker_fn(block_ker_fn(x_self, x_other))
    composed_ker_out = composed_ker_fn(x_self, x_other)
    if batching_fn == batch._parallel:
      composed_ker_out = composed_ker_out.replace(x1_is_x2=ker_out.x1_is_x2)
    self.assertAllClose(ker_out, composed_ker_out)

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              '_on_device={}_batch_size={}'.format(store_on_device, batch_size),
          'store_on_device':
              store_on_device,
          'batch_size':
              batch_size
      } for store_on_device in [True, False] for batch_size in [2, 8]))
  def testAnalyticKernelComposeSerial(self, store_on_device, batch_size):
    self._test_analytic_kernel_composition(
        partial(
            batch._serial,
            batch_size=batch_size,
            store_on_device=store_on_device))

  def testAnalyticKernelComposeParallel(self):
    test_utils.stub_out_pmap(batch, 2)
    self._test_analytic_kernel_composition(batch._parallel)

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              '_on_device={}_batch_size={}'.format(store_on_device, batch_size),
          'store_on_device':
              store_on_device,
          'batch_size':
              batch_size
      } for store_on_device in [True, False] for batch_size in [2, 8]))
  def testAnalyticKernelComposeAutomatic(self, store_on_device, batch_size):
    test_utils.stub_out_pmap(batch, 2)
    self._test_analytic_kernel_composition(
        partial(
            batch.batch, batch_size=batch_size,
            store_on_device=store_on_device))

  def test_jit_or_pmap_broadcast(self):

    def kernel_fn(x1,
                  x2,
                  do_flip,
                  keys,
                  do_square,
                  params,
                  _unused=None,
                  p=0.65):
      res = np.abs(np.matmul(x1, x2))
      if do_square:
        res *= res
      if do_flip:
        res = -res

      res *= random.uniform(keys) * p
      return [res, params]

    params = (np.array([1., 0.3]), (np.array([1.2]), np.array([0.5])))
    x2 = np.arange(0, 10).reshape((10,))
    keys = random.PRNGKey(1)

    kernel_fn_pmapped = batch._jit_or_pmap_broadcast(kernel_fn, device_count=0)
    x1 = np.arange(0, 10).reshape((1, 10))
    for do_flip in [True, False]:
      for do_square in [True, False]:
        with self.subTest(do_flip=do_flip, do_square=do_square, device_count=0):
          res_1 = kernel_fn(
              x1, x2, do_flip, keys, do_square, params, _unused=True, p=0.65)
          res_2 = kernel_fn_pmapped(
              x1, x2, do_flip, keys, do_square, params, _unused=True)
          self.assertAllClose(res_1, res_2)

    test_utils.stub_out_pmap(batch, 1)
    x1 = np.arange(0, 10).reshape((1, 10))
    kernel_fn_pmapped = batch._jit_or_pmap_broadcast(kernel_fn, device_count=1)
    for do_flip in [True, False]:
      for do_square in [True, False]:
        with self.subTest(do_flip=do_flip, do_square=do_square, device_count=1):
          res_1 = kernel_fn(
              x1, x2, do_flip, keys, do_square, params, _unused=False, p=0.65)
          res_2 = kernel_fn_pmapped(
              x1, x2, do_flip, keys, do_square, params, _unused=None)
          self.assertAllClose(res_1[0], res_2[0])
          self.assertAllClose(
              tree_map(partial(np.expand_dims, axis=0), res_1[1]), res_2[1])

    kernel_fn_pmapped = batch._jit_or_pmap_broadcast(kernel_fn, device_count=2)
    x1 = np.arange(0, 20).reshape((2, 10))
    test_utils.stub_out_pmap(batch, 2)

    def broadcast(arg):
      return np.broadcast_to(arg, (2,) + arg.shape)

    for do_flip in [True, False]:
      for do_square in [True, False]:
        with self.subTest(do_flip=do_flip, do_square=do_square, device_count=2):
          res_1 = kernel_fn(x1, x2, do_flip, keys, do_square, params, p=0.2)
          res_2 = kernel_fn_pmapped(
              x1, x2, do_flip, keys, do_square, params, _unused=None, p=0.2)
          self.assertAllClose(res_1[0][0], res_2[0][0])
          self.assertAllClose(res_1[0][1], res_2[0][1])
          self.assertAllClose(tree_map(broadcast, res_1[1]), res_2[1])


if __name__ == '__main__':
  absltest.main()
