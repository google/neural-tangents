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

"""Tests for `utils/predict.py`."""

import itertools

from absl.testing import absltest
import jax
from jax import lax
from jax import test_util as jtu
from jax import device_get
from jax import jit
from jax.config import config
from jax.lib import xla_bridge
import jax.numpy as np
import jax.random as random
from neural_tangents.utils import utils


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

        if xla_bridge.get_backend().platform == 'cpu':
          self.assertTrue(utils.is_on_cpu(x()))
          self.assertTrue(utils.is_on_cpu(x_jit()))
        else:
          self.assertFalse(utils.is_on_cpu(x()))
          self.assertFalse(utils.is_on_cpu(x_jit()))

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              f' [n={n}_{padding}_dn={lhs_spec, rhs_spec, out_spec}]',
          'n': n,
          'padding': padding,
          'lhs_spec': lhs_spec,
          'rhs_spec': rhs_spec,
          'out_spec': out_spec
      }
                          for n in [0, 1, 2]
                          for padding in [
                              'SAME',
                              'VALID'
                          ]
                          for lhs_spec in [
                              ''.join(s)
                              for s in itertools.permutations('NCHWD'[:n + 2])]
                          for rhs_spec in [
                              ''.join(s)
                              for s in itertools.permutations('OIHWDX'[:n + 2])]
                          for out_spec in [
                              ''.join(s)
                              for s in itertools.permutations('NCHWDX'[:n + 2])]
                          ))
  def test_conv_local_general_dilated(self, n, padding, lhs_spec, rhs_spec,
                                      out_spec):
    """Make sure LCN with tiled CNN kernel matches CNN."""
    if xla_bridge.get_backend().platform == 'cpu' and n > 1:
      raise absltest.SkipTest('Skipping large tests on CPU.')

    lhs_spec_default = 'NCHWDX'[:n + 2]
    rhs_spec_default = 'OIHWDX'[:n + 2]

    lhs_default = random.normal(random.PRNGKey(1), (2, 4, 7, 6, 5, 8)[:n + 2])
    rhs_default = random.normal(random.PRNGKey(2), (3, 4, 2, 3, 1, 2)[:n + 2])

    window_strides = (1, 2, 3, 4)[:n]
    rhs_dilation = (2, 1, 3, 2)[:n]

    lhs_perm = [lhs_spec_default.index(c) for c in lhs_spec]
    lhs = np.transpose(lhs_default, lhs_perm)

    rhs_perm = [rhs_spec_default.index(c) for c in rhs_spec]
    rhs = np.transpose(rhs_default, rhs_perm)

    kwargs = dict(
        lhs=lhs,
        window_strides=window_strides,
        padding=padding,
        rhs_dilation=rhs_dilation,
        dimension_numbers=(lhs_spec, rhs_spec, out_spec)
    )

    out_conv = lax.conv_general_dilated(rhs=rhs, **kwargs)

    rhs_local = np.moveaxis(rhs, (rhs_spec.index('O'), rhs_spec.index('I')),
                            (0, 1))
    rhs_local = rhs_local.reshape((rhs_local.shape[0], -1) + (1,) * n)

    rhs_shape = (rhs_local.shape[:2] +
                 tuple(out_conv.shape[out_spec.index(c)]
                       for c in rhs_spec_default[2:]))

    rhs_local = np.broadcast_to(rhs_local, rhs_shape)
    rhs_local = np.transpose(rhs_local, rhs_perm)

    filter_shape = [rhs.shape[i]
                    for i in range(n + 2) if rhs_spec[i] not in ('O', 'I')]
    out_local = utils.conv_general_dilated_local(rhs=rhs_local,
                                                 filter_shape=filter_shape,
                                                 **kwargs)

    self.assertAllClose(out_conv, out_local, atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
  absltest.main()
