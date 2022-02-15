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

"""Tests for `neural_tangents/_src/utils/utils.py`."""

from absl.testing import absltest
import jax
from jax import device_get
from jax import jit
from jax import test_util as jtu
from jax.config import config
import jax.numpy as np
import jax.random as random
from neural_tangents._src.utils import utils


config.parse_flags_with_absl()
config.update('jax_numpy_rank_promotion', 'raise')


class UtilsTest(jtu.JaxTestCase):

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
          return device_get(random.normal(random.PRNGKey(1), (2, 3), dtype))

        x_jit = jit(x)
        # x_cpu_jit = jit(x_cpu)
        x_cpu_jit_cpu = jit(x_cpu, backend='cpu')

        self.assertTrue(utils.is_on_cpu(x_cpu()))
        # TODO(mattjj): re-enable this when device_put under jit works
        # self.assertTrue(utils.is_on_cpu(x_cpu_jit()))
        self.assertTrue(utils.is_on_cpu(x_cpu_jit_cpu()))

        if jax.default_backend() == 'cpu':
          self.assertTrue(utils.is_on_cpu(x()))
          self.assertTrue(utils.is_on_cpu(x_jit()))
        else:
          self.assertFalse(utils.is_on_cpu(x()))
          self.assertFalse(utils.is_on_cpu(x_jit()))


if __name__ == '__main__':
  absltest.main()
