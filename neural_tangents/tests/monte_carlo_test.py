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


def _get_inputs_and_model(width=1, n_classes=2):
  key = random.PRNGKey(1)
  key, split = random.split(key)
  x1 = random.normal(key, (8, 4, 3, 2))
  x2 = random.normal(split, (4, 4, 3, 2))
  init_fun, apply_fun, ker_fun = stax.serial(
      stax.Conv(width, (3, 3)), stax.Relu(), stax.Flatten(),
      stax.Dense(n_classes, 2., 0.5))
  return x1, x2, init_fun, apply_fun, ker_fun, key


@jtu.parameterized.named_parameters(
    jtu.cases_from_list({
        'testcase_name': '[batch_size={}, '
                         'device_count={} '
                         'store_on_device={} '
                         ']'.format(batch_size, device_count, store_on_device),
        'batch_size': batch_size,
        'device_count': device_count,
        'store_on_device': store_on_device,
    } for batch_size in BATCH_SIZES for device_count in DEVICE_COUNTS
                        for store_on_device in STORE_ON_DEVICE))
class MonteCarloTest(jtu.JaxTestCase):

  def assertAllClose(self, x, y, check_dtypes, atol=None, rtol=None):
    if x is None and y is None:
      return
    super(MonteCarloTest, self).assertAllClose(x, y, check_dtypes, atol, rtol)

  def test_sample_once_batch(self, batch_size, device_count, store_on_device):
    utils.stub_out_pmap(batch, device_count)

    x1, x2, init_fun, apply_fun, _, key = _get_inputs_and_model()
    ker_fun = empirical.get_ker_fun_empirical(apply_fun)

    sample_once_fun = monte_carlo._get_ker_fun_sample_once(ker_fun, init_fun)
    one_sample = sample_once_fun(x1, x2, key)

    sample_once_batch_fun = monte_carlo._get_ker_fun_sample_once(
        ker_fun, init_fun, batch_size, device_count, store_on_device)
    one_sample_batch = sample_once_batch_fun(x1, x2, key)
    self.assertAllClose(one_sample, one_sample_batch, True)

  def test_batch_sample_once(self, batch_size, device_count, store_on_device):
    utils.stub_out_pmap(batch, device_count)

    x1, x2, init_fun, apply_fun, _, key = _get_inputs_and_model()
    ker_fun = empirical.get_ker_fun_empirical(apply_fun)

    sample_once_fun = monte_carlo._get_ker_fun_sample_once(ker_fun, init_fun)
    one_sample = sample_once_fun(x1, x2, key)

    batch_sample_once_fun = batch.batch(
        monte_carlo._get_ker_fun_sample_once(ker_fun, init_fun), batch_size,
        device_count, store_on_device)
    one_batch_sample = batch_sample_once_fun(x1, x2, key)
    self.assertAllClose(one_sample, one_batch_sample, True)

  def test_sample_many_batch(self, batch_size, device_count, store_on_device):
    utils.stub_out_pmap(batch, device_count)

    x1, x2, init_fun, apply_fun, _, key = _get_inputs_and_model()
    ker_fun = empirical.get_ker_fun_empirical(apply_fun)

    sample_once_fun = monte_carlo._get_ker_fun_sample_once(ker_fun, init_fun)
    sample_many_fun = monte_carlo._get_ker_fun_sample_many(sample_once_fun)
    sample_many_batch_fun = monte_carlo._get_ker_fun_sample_many(
        batch.batch(sample_once_fun, batch_size, device_count, store_on_device))

    many_samples = sample_many_fun(x1, x2, key, N_SAMPLES)
    many_samples_batch = sample_many_batch_fun(x1, x2, key, N_SAMPLES)
    self.assertAllClose(many_samples, many_samples_batch, True)

  def test_sample_vs_analytic_nngp(self, batch_size, device_count,
                                   store_on_device):
    utils.stub_out_pmap(batch, device_count)

    x1, x2, init_fun, apply_fun, stax_ker_fun, key = _get_inputs_and_model(
        512, 512)

    sample = monte_carlo.get_ker_fun_monte_carlo(init_fun, apply_fun, True,
                                                 False, batch_size,
                                                 device_count, store_on_device)

    ker_empirical = sample(x1, x2, key, 200).nngp
    ker_analytic = stax_ker_fun(x1, x2, compute_ntk=False, compute_nngp=True)
    ker_analytic = ker_analytic.nngp

    utils.assert_close_matrices(self, ker_analytic, ker_empirical, 1e-2)

  def test_monte_carlo_vs_analytic_ntk(self, batch_size, device_count,
                                       store_on_device):
    utils.stub_out_pmap(batch, device_count)

    x1, x2, init_fun, apply_fun, stax_ker_fun, key = _get_inputs_and_model(
        512, 2)

    sample = monte_carlo.get_ker_fun_monte_carlo(init_fun, apply_fun, False,
                                                 True, batch_size, device_count,
                                                 store_on_device)

    ker_empirical = sample(x1, x2, key, 100).ntk
    ker_empirical = (
        np.sum(ker_empirical, axis=(-1, -2)) / ker_empirical.shape[-1])

    ker_analytic = stax_ker_fun(x1, x2, compute_ntk=True, compute_nngp=True)
    ker_analytic = ker_analytic.ntk

    utils.assert_close_matrices(self, ker_analytic, ker_empirical, 1e-2)
#

if __name__ == '__main__':
  jtu.absltest.main()
