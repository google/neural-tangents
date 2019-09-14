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

"""General-purpose internal utilities."""

from jax.api import vmap
from jax.lib import xla_bridge
import jax.numpy as np


def stub_out_pmap(batch, count):
  # If we are using GPU or CPU stub out pmap with vmap to simulate multi-core.
  if count > 1:
    class xla_bridge_stub(object):
      def device_count(self):
        return count

    platform = xla_bridge.get_backend().platform
    if platform == 'gpu' or platform == 'cpu':
      # TODO(romann): investigate why vmap is extremely slow in
      # `utils/monte_carlo_test.py`, `test_monte_carlo_vs_analytic`.
      # Example: http://sponge/e081c176-e77f-428c-846d-bafbfd86a46c
      batch.pmap = vmap
      batch.xla_bridge = xla_bridge_stub()


def assert_close_matrices(self, expected, actual, rtol):
  self.assertEqual(expected.shape, actual.shape)
  relative_error = (np.linalg.norm(actual - expected) /
                    np.maximum(np.linalg.norm(expected), 1e-12))
  if relative_error > rtol or np.isnan(relative_error):
    self.fail(self.failureException(float(relative_error), expected, actual))
  else:
    print('PASSED with %f relative error.' % relative_error)
