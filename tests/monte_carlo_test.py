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

"""Tests for `neural_tangents/_src/monte_carlo.py`."""

from absl.testing import absltest
import jax
from jax import random
from jax.config import config
import jax.numpy as np
import neural_tangents as nt
from neural_tangents import stax
from neural_tangents._src import batching
from neural_tangents._src import monte_carlo
from tests import test_utils


config.parse_flags_with_absl()
config.update('jax_numpy_rank_promotion', 'raise')


BATCH_SIZES = [
    2,
    4,
]

WIDTH = 192

DEVICE_COUNTS = [0, 1, 2]

STORE_ON_DEVICE = [True, False]

ALL_GET = ('nngp', 'ntk', ('nngp', 'ntk'), None)

test_utils.update_test_tolerance()


def _get_inputs_and_model(width=1, n_classes=2, use_conv=True):
  key = random.PRNGKey(1)
  key, split = random.split(key)
  x1 = random.normal(key, (8, 4, 3, 2))
  x2 = random.normal(split, (4, 4, 3, 2))

  if not use_conv:
    x1 = np.reshape(x1, (x1.shape[0], -1))
    x2 = np.reshape(x2, (x2.shape[0], -1))

  init_fn, apply_fn, kernel_fn = stax.serial(
      stax.Conv(width, (3, 3)) if use_conv else stax.Dense(width),
      stax.Relu(),
      stax.Flatten(),
      stax.Dense(n_classes, 2., 0.5))
  return x1, x2, init_fn, apply_fn, kernel_fn, key


class MonteCarloTest(test_utils.NeuralTangentsTestCase):

  @test_utils.product(
      batch_size=BATCH_SIZES,
      device_count=DEVICE_COUNTS,
      store_on_device=STORE_ON_DEVICE,
      get=ALL_GET,
  )
  def test_sample_once_batch(
      self,
      batch_size,
      device_count,
      store_on_device,
      get
  ):
    test_utils.stub_out_pmap(batching, device_count)

    x1, x2, init_fn, apply_fn, _, key = _get_inputs_and_model()
    kernel_fn = nt.empirical_kernel_fn(apply_fn)

    sample_once_fn = monte_carlo._sample_once_kernel_fn(kernel_fn, init_fn)
    sample_once_batch_fn = monte_carlo._sample_once_kernel_fn(
        kernel_fn, init_fn, batch_size, device_count, store_on_device)

    one_sample = sample_once_fn(x1, x2, key, get)
    one_sample_batch = sample_once_batch_fn(x1, x2, key, get)
    self.assertAllClose(one_sample, one_sample_batch)

  @test_utils.product(
      batch_size=BATCH_SIZES,
      device_count=DEVICE_COUNTS,
      store_on_device=STORE_ON_DEVICE,
      get=ALL_GET
  )
  def test_batch_sample_once(
      self,
      batch_size,
      device_count,
      store_on_device,
      get
  ):
    test_utils.stub_out_pmap(batching, device_count)

    x1, x2, init_fn, apply_fn, _, key = _get_inputs_and_model()
    kernel_fn = nt.empirical_kernel_fn(apply_fn)
    sample_once_fn = monte_carlo._sample_once_kernel_fn(
        kernel_fn, init_fn, device_count=0)
    batch_sample_once_fn = batching.batch(sample_once_fn, batch_size,
                                          device_count, store_on_device)
    one_sample = sample_once_fn(x1, x2, key, get)
    one_batch_sample = batch_sample_once_fn(x1, x2, key, get)
    self.assertAllClose(one_sample, one_batch_sample)

  @test_utils.product(
      batch_size=BATCH_SIZES,
      device_count=DEVICE_COUNTS,
      store_on_device=STORE_ON_DEVICE
  )
  def test_sample_vs_analytic_nngp(
      self,
      batch_size,
      device_count,
      store_on_device
  ):
    test_utils.stub_out_pmap(batching, device_count)

    x1, x2, init_fn, apply_fn, stax_kernel_fn, key = _get_inputs_and_model(
        WIDTH, 256, jax.default_backend() == 'tpu')

    sample = monte_carlo.monte_carlo_kernel_fn(init_fn, apply_fn, key, 200,
                                               batch_size, device_count,
                                               store_on_device)

    ker_empirical = sample(x1, x2, 'nngp')
    ker_analytic = stax_kernel_fn(x1, x2, 'nngp')

    test_utils.assert_close_matrices(self, ker_analytic, ker_empirical, 2e-2)

  @test_utils.product(
      batch_size=BATCH_SIZES,
      device_count=DEVICE_COUNTS,
      store_on_device=STORE_ON_DEVICE
  )
  def test_monte_carlo_vs_analytic_ntk(
      self,
      batch_size,
      device_count,
      store_on_device
  ):
    test_utils.stub_out_pmap(batching, device_count)

    x1, x2, init_fn, apply_fn, stax_kernel_fn, key = _get_inputs_and_model(
        WIDTH, 2, jax.default_backend() == 'tpu')

    sample = monte_carlo.monte_carlo_kernel_fn(init_fn, apply_fn, key, 100,
                                               batch_size, device_count,
                                               store_on_device,
                                               vmap_axes=0)

    ker_empirical = sample(x1, x2, 'ntk')
    ker_analytic = stax_kernel_fn(x1, x2, 'ntk')

    test_utils.assert_close_matrices(self, ker_analytic, ker_empirical, 2e-2)

  @test_utils.product(
      batch_size=BATCH_SIZES,
      device_count=DEVICE_COUNTS,
      store_on_device=STORE_ON_DEVICE,
      get=ALL_GET
  )
  def test_monte_carlo_generator(
      self,
      batch_size,
      device_count,
      store_on_device,
      get
  ):
    test_utils.stub_out_pmap(batching, device_count)

    x1, x2, init_fn, apply_fn, stax_kernel_fn, key = _get_inputs_and_model(8, 1)
    x3, x4, _, _, _, _ = _get_inputs_and_model(8, 1)

    log_n_max = 4
    n_samples = [2**k for k in range(log_n_max)]
    sample_generator = monte_carlo.monte_carlo_kernel_fn(
        init_fn, apply_fn, key, n_samples, batch_size, device_count,
        store_on_device, vmap_axes=0)

    if get is None:
      samples_12 = sample_generator(x1, x2)
      samples_34 = sample_generator(x3, x4)

      count = 0
      for n, s_12, s_34 in zip(n_samples, samples_12, samples_34):
        sample_fn = monte_carlo.monte_carlo_kernel_fn(init_fn, apply_fn, key,
                                                      n, batch_size,
                                                      device_count,
                                                      store_on_device,
                                                      vmap_axes=0)
        sample_12 = sample_fn(x1, x2)
        sample_34 = sample_fn(x3, x4)
        self.assertAllClose(s_12, sample_12)
        self.assertAllClose(s_12, s_34)
        self.assertAllClose(s_12, sample_34)
        count += 1

      self.assertEqual(log_n_max, count)

      ker_analytic_12 = stax_kernel_fn(x1, x2, ('nngp', 'ntk'))
      ker_analytic_34 = stax_kernel_fn(x3, x4, ('nngp', 'ntk'))

    else:
      samples_12 = sample_generator(x1, x2, get)
      samples_34 = sample_generator(x3, x4, get)

      count = 0
      for n, s_12, s_34 in zip(n_samples, samples_12, samples_34):
        sample_fn = monte_carlo.monte_carlo_kernel_fn(
            init_fn, apply_fn, key, n, batch_size,
            device_count, store_on_device, vmap_axes=0)
        sample_12 = sample_fn(x1, x2, get)
        sample_34 = sample_fn(x3, x4, get)
        self.assertAllClose(s_12, sample_12)
        self.assertAllClose(s_12, s_34)
        self.assertAllClose(s_12, sample_34)
        count += 1

      self.assertEqual(log_n_max, count)

      ker_analytic_12 = stax_kernel_fn(x1, x2, get)
      ker_analytic_34 = stax_kernel_fn(x3, x4, get)

    self.assertAllClose(ker_analytic_12, s_12, atol=2., rtol=2.)
    self.assertAllClose(ker_analytic_12, ker_analytic_34)

  @test_utils.product(
      same_inputs=[True, False],
      batch_size=[1, 2]
  )
  def test_parallel_in_out_mc(self, same_inputs, batch_size):
    rng = random.PRNGKey(0)
    input_key1, input_key2, net_key = random.split(rng, 3)

    x1_1, x1_2, x1_3 = random.normal(input_key1, (3, 2, 5))
    x1 = (x1_1, (x1_2, x1_3))

    if same_inputs:
      x2 = None
    else:
      x2_1, x2_2, x2_3 = random.normal(input_key2, (3, 4, 5))
      x2 = (x2_1, (x2_2, x2_3))

    def net(N_out):
      return stax.parallel(stax.Dense(N_out),
                           stax.parallel(stax.Dense(N_out + 1),
                                         stax.Dense(N_out + 2)))

    # Check NNGP.
    init_fn, apply_fn, _ = net(WIDTH)

    nb_kernel_fn = monte_carlo.monte_carlo_kernel_fn(init_fn,
                                                     apply_fn,
                                                     net_key,
                                                     n_samples=4,
                                                     trace_axes=(-1,))

    kernel_fn = monte_carlo.monte_carlo_kernel_fn(init_fn,
                                                  apply_fn,
                                                  net_key,
                                                  n_samples=4,
                                                  batch_size=batch_size,
                                                  trace_axes=(-1,))

    self.assertAllClose(kernel_fn(x1, x2, 'nngp'), nb_kernel_fn(x1, x2, 'nngp'))


if __name__ == '__main__':
  absltest.main()
