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
"""Tests for `utils/monte_carlo.py`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from jax import test_util as jtu
from jax.config import config as jax_config
from jax.lib import xla_bridge
import jax.numpy as np
import jax.random as random
from neural_tangents import stax
from neural_tangents.utils import batch
from neural_tangents.utils import empirical
from neural_tangents.utils import monte_carlo
from neural_tangents.utils import utils

jax_config.parse_flags_with_absl()

BATCH_SIZES = [
    1,
    2,
    4,
]

DEVICE_COUNTS = [0, 1, 2]

STORE_ON_DEVICE = [True, False]

N_SAMPLES = 4

ALL_GET = ('nngp', 'ntk', ('nngp', 'ntk'), None)


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


class MonteCarloTest(jtu.JaxTestCase):

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name': '[batch_size={}, '
                           'device_count={} '
                           'store_on_device={} '
                           'get={} '
                           ']'.format(batch_size, device_count, store_on_device,
                                      get),
          'batch_size': batch_size,
          'device_count': device_count,
          'store_on_device': store_on_device,
          'get': get,
      } for batch_size in BATCH_SIZES for device_count in DEVICE_COUNTS
                          for store_on_device in STORE_ON_DEVICE
                          for get in ALL_GET))
  def test_sample_once_batch(self, batch_size, device_count, store_on_device,
                             get):
    utils.stub_out_pmap(batch, device_count)

    x1, x2, init_fn, apply_fn, _, key = _get_inputs_and_model()
    kernel_fn = empirical.empirical_kernel_fn(apply_fn)

    sample_once_fn = monte_carlo._sample_once_kernel_fn(kernel_fn, init_fn)
    sample_once_batch_fn = monte_carlo._sample_once_kernel_fn(
        kernel_fn, init_fn, batch_size, device_count, store_on_device)

    one_sample = sample_once_fn(x1, x2, key, get)
    one_sample_batch = sample_once_batch_fn(x1, x2, key, get)
    self.assertAllClose(one_sample, one_sample_batch, True)

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name': '[batch_size={}, '
                           'device_count={} '
                           'store_on_device={} '
                           'get={} '
                           ']'.format(batch_size, device_count, store_on_device,
                                      get),
          'batch_size': batch_size,
          'device_count': device_count,
          'store_on_device': store_on_device,
          'get': get,
      } for batch_size in BATCH_SIZES for device_count in DEVICE_COUNTS
                          for store_on_device in STORE_ON_DEVICE
                          for get in ALL_GET))
  def test_batch_sample_once(self, batch_size, device_count, store_on_device,
                             get):
    utils.stub_out_pmap(batch, device_count)

    x1, x2, init_fn, apply_fn, _, key = _get_inputs_and_model()
    kernel_fn = empirical.empirical_kernel_fn(apply_fn)
    sample_once_fn = monte_carlo._sample_once_kernel_fn(
        kernel_fn, init_fn, device_count=0)
    batch_sample_once_fn = batch.batch(sample_once_fn, batch_size,
                                       device_count, store_on_device)
    one_sample = sample_once_fn(x1, x2, key, get)
    one_batch_sample = batch_sample_once_fn(x1, x2, key, get)
    self.assertAllClose(one_sample, one_batch_sample, True)

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name': '[batch_size={}, '
                           'device_count={} '
                           'store_on_device={} '
                           ']'.format(batch_size, device_count, store_on_device
                                     ),
          'batch_size': batch_size,
          'device_count': device_count,
          'store_on_device': store_on_device,
      } for batch_size in BATCH_SIZES for device_count in DEVICE_COUNTS
                          for store_on_device in STORE_ON_DEVICE))
  def test_sample_vs_analytic_nngp(self, batch_size, device_count,
                                   store_on_device):
    utils.stub_out_pmap(batch, device_count)

    x1, x2, init_fn, apply_fn, stax_kernel_fn, key = _get_inputs_and_model(
        1024, 256, xla_bridge.get_backend().platform == 'tpu')

    sample = monte_carlo.monte_carlo_kernel_fn(init_fn, apply_fn, key, 200,
                                             batch_size, device_count,
                                             store_on_device)

    ker_empirical = sample(x1, x2, 'nngp')
    ker_analytic = stax_kernel_fn(x1, x2, 'nngp')

    utils.assert_close_matrices(self, ker_analytic, ker_empirical, 2e-2)

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name': '[batch_size={}, '
                           'device_count={} '
                           'store_on_device={} '
                           ']'.format(batch_size, device_count, store_on_device
                                     ),
          'batch_size': batch_size,
          'device_count': device_count,
          'store_on_device': store_on_device,
      } for batch_size in BATCH_SIZES for device_count in DEVICE_COUNTS
                          for store_on_device in STORE_ON_DEVICE))
  def test_monte_carlo_vs_analytic_ntk(self, batch_size, device_count,
                                       store_on_device):
    utils.stub_out_pmap(batch, device_count)

    x1, x2, init_fn, apply_fn, stax_kernel_fn, key = _get_inputs_and_model(
        256, 2, xla_bridge.get_backend().platform == 'tpu')

    sample = monte_carlo.monte_carlo_kernel_fn(init_fn, apply_fn, key, 100,
                                             batch_size, device_count,
                                             store_on_device)

    ker_empirical = sample(x1, x2, 'ntk')
    ker_empirical = (
        np.sum(ker_empirical, axis=(-1, -2)) / ker_empirical.shape[-1])

    ker_analytic = stax_kernel_fn(x1, x2, 'ntk')

    utils.assert_close_matrices(self, ker_analytic, ker_empirical, 2e-2)

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name': '[batch_size={}, '
                           'device_count={} '
                           'store_on_device={} '
                           'get={}'
                           ']'.format(batch_size, device_count, store_on_device,
                                      get),
          'batch_size': batch_size,
          'device_count': device_count,
          'store_on_device': store_on_device,
          'get': get
      } for batch_size in BATCH_SIZES for device_count in DEVICE_COUNTS
                          for store_on_device in STORE_ON_DEVICE
                          for get in ALL_GET))
  def test_monte_carlo_generator(self, batch_size, device_count,
                                 store_on_device, get):
    utils.stub_out_pmap(batch, device_count)

    x1, x2, init_fn, apply_fn, stax_kernel_fn, key = _get_inputs_and_model(8, 1)
    x3, x4, _, _, _, _ = _get_inputs_and_model(8, 1)

    log_n_max = 4
    n_samples = [2**k for k in range(log_n_max)]
    sample_generator = monte_carlo.monte_carlo_kernel_fn(
        init_fn, apply_fn, key, n_samples, batch_size, device_count,
        store_on_device)

    if get is None:
      samples_12 = sample_generator(x1, x2)
      samples_34 = sample_generator(x3, x4)

      count = 0
      for n, s_12, s_34 in zip(n_samples, samples_12, samples_34):
        sample_fn = monte_carlo.monte_carlo_kernel_fn(init_fn, apply_fn, key,
                                                      n, batch_size,
                                                      device_count,
                                                      store_on_device)
        sample_12 = sample_fn(x1, x2)
        sample_34 = sample_fn(x3, x4)
        self.assertAllClose(s_12, sample_12, True)
        self.assertAllClose(s_12, s_34, True)
        self.assertAllClose(s_12, sample_34, True)
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
            device_count, store_on_device)
        sample_12 = sample_fn(x1, x2, get)
        sample_34 = sample_fn(x3, x4, get)
        self.assertAllClose(s_12, sample_12, True)
        self.assertAllClose(s_12, s_34, True)
        self.assertAllClose(s_12, sample_34, True)
        count += 1

      self.assertEqual(log_n_max, count)

      ker_analytic_12 = stax_kernel_fn(x1, x2, get)
      ker_analytic_34 = stax_kernel_fn(x3, x4, get)

    if get == 'ntk':
      s_12 = np.squeeze(s_12, (-1, -2))
    elif get is None or 'ntk' in get:
      s_12 = s_12._replace(ntk=np.squeeze(s_12.ntk, (-1, -2)))

    self.assertAllClose(ker_analytic_12, s_12, True, 2., 2.)
    self.assertAllClose(ker_analytic_12, ker_analytic_34, True)


if __name__ == '__main__':
  jtu.absltest.main()
