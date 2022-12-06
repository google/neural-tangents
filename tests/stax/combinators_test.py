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

"""Tests for `neural_tangents/_src/stax/combinators.py`."""

import random as prandom

from absl.testing import absltest
from jax import random
from jax.config import config
import jax.numpy as np
from neural_tangents import stax
from tests import test_utils


config.parse_flags_with_absl()
config.update('jax_numpy_rank_promotion', 'raise')


test_utils.update_test_tolerance()

prandom.seed(1)


class RepeatTest(test_utils.NeuralTangentsTestCase):

  def _test_repeat(self, x1, x2, layer, n, rng_params, **kwargs):
    init_fn, apply_fn, kernel_fn = (stax.Identity() if n == 0 else
                                    stax.serial(*([layer] * n)))
    init_fn_repeat, apply_fn_repeat, kernel_fn_repeat = stax.repeat(layer, n)

    out_shape, params = init_fn(rng_params, x1.shape)
    out_shape_repeat, params_repeat = init_fn_repeat(rng_params, x1.shape)

    self.assertEqual(out_shape, out_shape_repeat)

    kwargs1 = {k: kwargs[k][0] for k in kwargs}
    out = apply_fn(params, x1, **kwargs1)
    out_repeat = apply_fn_repeat(params_repeat, x1, **kwargs1)

    self.assertAllClose(out, out_repeat)

    for get in [None, 'ntk', 'nngp', 'cov1', ('nngp', 'cov1'), ('cov1', 'ntk')]:
      with self.subTest(get=get):
        k = kernel_fn(x1, x2, get, **kwargs)
        k_repeat = kernel_fn_repeat(x1, x2, get, **kwargs)
        self.assertAllClose(k, k_repeat)

  @test_utils.product(
      same_inputs=[
          False,
          True
      ],
      n=[
          0,
          1,
          2,
          3,
      ],
      layer=[
          stax.Identity(),
          stax.Dense(3),
          stax.serial(stax.Identity()),
          stax.serial(stax.Dense(3)),
          stax.GlobalAvgPool(),
          stax.serial(stax.Dense(3), stax.Relu()),
          stax.serial(stax.Dense(3), stax.Relu(), stax.Dense(3))
      ]
  )
  def test_repeat(
      self,
      same_inputs,
      n,
      layer
  ):
    rng_input, rng_params = random.split(random.PRNGKey(1), 2)
    x1 = np.cos(random.normal(rng_input, (2, 3)))
    x2 = None if same_inputs else random.normal(rng_input, (4, 3))

    self._test_repeat(x1, x2, layer, n, rng_params)

  @test_utils.product(
      same_inputs=[
          False,
          True
      ],
      n=[
          0,
          1,
          2,
          3,
      ],
      layer=[
          stax.serial(stax.Conv(3, (2, 2), padding='SAME'),
                      stax.Relu(),
                      stax.Conv(3, (2, 2), padding='SAME'),
                      stax.Gelu()
                      ),
      ]
  )
  def test_repeat_conv(
      self,
      same_inputs,
      n,
      layer
  ):
    rng_input, rng_params = random.split(random.PRNGKey(1), 2)
    x1 = np.cos(random.normal(rng_input, (2, 4, 4, 3)))
    x2 = None if same_inputs else random.normal(rng_input, (4, 4, 4, 3))

    self._test_repeat(x1, x2, layer, n, rng_params)

  @test_utils.product(
      same_inputs=[
          False,
          True
      ],
      n=[
          0,
          1,
          2,
          3,
      ],
      layer=[
          stax.Aggregate(),
          stax.serial(stax.Dense(3), stax.Aggregate(), stax.Abs()),
          stax.serial(stax.Conv(3, (2, 2), padding='SAME'),
                      stax.Aggregate(),
                      stax.Abs(),
                      stax.Conv(3, (1, 2), padding='SAME'),
                      )
      ]
  )
  def test_repeat_agg(
      self,
      same_inputs,
      n,
      layer
  ):
    rng_input, rng_params, rng_p1, rng_p2 = random.split(random.PRNGKey(1), 4)
    x1 = np.cos(random.normal(rng_input, (2, 4, 3, 3)))
    x2 = None if same_inputs else random.normal(rng_input, (4, 4, 3, 3))

    p1 = random.normal(rng_p1, x1.shape[:-1] + x1.shape[1:-1])
    p2 = p1 if x2 is None else random.normal(rng_p2,
                                             x2.shape[:-1] + x2.shape[1:-1])

    self._test_repeat(x1, x2, layer, n, rng_params, pattern=(p1, p2))
    self._test_repeat(x1, x2, layer, n, rng_params, pattern=(None, None))


if __name__ == '__main__':
  absltest.main()
