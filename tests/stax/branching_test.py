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

"""Tests for `neural_tangents/_src/stax/branching.py`."""

import random as prandom

from absl.testing import absltest
from jax import default_backend
from jax import random
from jax.config import config
import jax.numpy as np
import neural_tangents as nt
from neural_tangents import stax
from neural_tangents._src.empirical import _DEFAULT_TESTING_NTK_IMPLEMENTATION
from tests import test_utils


config.parse_flags_with_absl()
config.update('jax_numpy_rank_promotion', 'raise')


test_utils.update_test_tolerance()

prandom.seed(1)


class FanInTest(test_utils.NeuralTangentsTestCase):

  @classmethod
  def _get_phi(cls, i):
    return {
        0: stax.Relu(),
        1: stax.Erf(),
        2: stax.Abs()
    }[i % 3]

  @test_utils.product(
      same_inputs=[False],
      axis=[0, 1],
      n_branches=[3],
      get=['ntk'],
      branch_in=['dense_before_branch_in', 'dense_after_branch_in'],
      fan_in_mode=['FanInSum', 'FanInConcat', 'FanInProd']
  )
  def test_fan_in_fc(
      self,
      same_inputs,
      axis,
      n_branches,
      get,
      branch_in,
      fan_in_mode
  ):
    if fan_in_mode in ['FanInSum', 'FanInProd']:
      if axis != 0:
        raise absltest.SkipTest('`FanInSum` and `FanInProd` are skipped when '
                                'axis != 0.')
      axis = None
    if (fan_in_mode == 'FanInSum' or
        axis == 0) and branch_in == 'dense_after_branch_in':
      raise absltest.SkipTest('`FanInSum` and `FanInConcat(0)` '
                              'require `is_gaussian`.')

    if ((axis == 1 or fan_in_mode == 'FanInProd') and
        branch_in == 'dense_before_branch_in'):
      raise absltest.SkipTest(
          '`FanInConcat` or `FanInProd` on feature axis requires a dense layer '
          'after concatenation or Hadamard product.')
    if fan_in_mode == 'FanInSum':
      fan_in_layer = stax.FanInSum()
    elif fan_in_mode == 'FanInProd':
      fan_in_layer = stax.FanInProd()
    else:
      fan_in_layer = stax.FanInConcat(axis)

    if n_branches != 2:
      test_utils.skip_test(self)

    key = random.PRNGKey(1)
    X0_1 = np.cos(random.normal(key, (4, 3)))
    X0_2 = None if same_inputs else random.normal(key, (8, 3))

    width = 1024
    n_samples = 256 * 2

    if default_backend() == 'tpu':
      tol = 0.07
    else:
      tol = 0.02

    dense = stax.Dense(width, 1.25, 0.1)
    input_layers = [dense,
                    stax.FanOut(n_branches)]

    branches = []
    for b in range(n_branches):
      branch_layers = [FanInTest._get_phi(b)]
      for i in range(b):
        multiplier = 1 if axis not in (1, -1) else (1 + 0.25 * i)
        branch_layers += [
            stax.Dense(int(width * multiplier), 1. + 2 * i, 0.5 + i),
            FanInTest._get_phi(i)]

      if branch_in == 'dense_before_branch_in':
        branch_layers += [dense]
      branches += [stax.serial(*branch_layers)]

    output_layers = [
        fan_in_layer,
        stax.Relu()
    ]
    if branch_in == 'dense_after_branch_in':
      output_layers.insert(1, dense)

    nn = stax.serial(*(input_layers + [stax.parallel(*branches)] +
                       output_layers))

    if get == 'nngp':
      init_fn, apply_fn, kernel_fn = nn
    elif get == 'ntk':
      init_fn, apply_fn, kernel_fn = stax.serial(nn, stax.Dense(1, 1.25, 0.5))
    else:
      raise ValueError(get)

    kernel_fn_mc = nt.monte_carlo_kernel_fn(
        init_fn, apply_fn, key, n_samples,
        device_count=0 if axis in (0, -2) else -1,
        implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION,
        vmap_axes=None if axis in (0, -2) else 0,
    )

    exact = kernel_fn(X0_1, X0_2, get=get)
    empirical = kernel_fn_mc(X0_1, X0_2, get=get)
    test_utils.assert_close_matrices(self, empirical, exact, tol)

  @test_utils.product(
      same_inputs=[False],
      axis=[0, 1, 2, 3],
      n_branches=[2],
      get=['ntk'],
      branch_in=['dense_before_branch_in', 'dense_after_branch_in'],
      readout=['pool', 'flatten'],
      fan_in_mode=['FanInSum', 'FanInConcat', 'FanInProd']
  )
  def test_fan_in_conv(
      self,
      same_inputs,
      axis,
      n_branches,
      get,
      branch_in,
      readout,
      fan_in_mode
  ):
    test_utils.skip_test(self)
    if fan_in_mode in ['FanInSum', 'FanInProd']:
      if axis != 0:
        raise absltest.SkipTest('`FanInSum` and `FanInProd()` are skipped when '
                                'axis != 0.')
      axis = None
    if (fan_in_mode == 'FanInSum' or
        axis in [0, 1, 2]) and branch_in == 'dense_after_branch_in':
      raise absltest.SkipTest('`FanInSum` and `FanInConcat(0/1/2)` '
                              'require `is_gaussian`.')

    if ((axis == 3 or fan_in_mode == 'FanInProd') and
        branch_in == 'dense_before_branch_in'):
      raise absltest.SkipTest('`FanInConcat` or `FanInProd` on feature axis '
                              'requires a dense layer after concatenation '
                              'or Hadamard product.')

    if fan_in_mode == 'FanInSum':
      fan_in_layer = stax.FanInSum()
    elif fan_in_mode == 'FanInProd':
      fan_in_layer = stax.FanInProd()
    else:
      fan_in_layer = stax.FanInConcat(axis)

    key = random.PRNGKey(1)
    X0_1 = random.normal(key, (2, 5, 6, 3))
    X0_2 = None if same_inputs else random.normal(key, (3, 5, 6, 3))

    if default_backend() == 'tpu':
      width = 2048
      n_samples = 1024
      tol = 0.02
    else:
      width = 1024
      n_samples = 512
      tol = 0.01

    conv = stax.Conv(out_chan=width,
                     filter_shape=(3, 3),
                     padding='SAME',
                     W_std=1.25,
                     b_std=0.1)

    input_layers = [conv,
                    stax.FanOut(n_branches)]

    branches = []
    for b in range(n_branches):
      branch_layers = [FanInTest._get_phi(b)]
      for i in range(b):
        multiplier = 1 if axis not in (3, -1) else (1 + 0.25 * i)
        branch_layers += [
            stax.Conv(
                out_chan=int(width * multiplier),
                filter_shape=(i + 1, 4 - i),
                padding='SAME',
                W_std=1.25 + i,
                b_std=0.1 + i),
            FanInTest._get_phi(i)]

      if branch_in == 'dense_before_branch_in':
        branch_layers += [conv]
      branches += [stax.serial(*branch_layers)]

    output_layers = [
        fan_in_layer,
        stax.Relu(),
        stax.GlobalAvgPool() if readout == 'pool' else stax.Flatten()
    ]
    if branch_in == 'dense_after_branch_in':
      output_layers.insert(1, conv)

    nn = stax.serial(*(input_layers + [stax.parallel(*branches)] +
                       output_layers))

    init_fn, apply_fn, kernel_fn = stax.serial(
        nn, stax.Dense(1 if get == 'ntk' else width, 1.25, 0.5))

    kernel_fn_mc = nt.monte_carlo_kernel_fn(
        init_fn,
        apply_fn,
        key,
        n_samples,
        device_count=0 if axis in (0, -4) else -1,
        implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION,
        vmap_axes=None if axis in (0, -4) else 0,
    )

    exact = kernel_fn(X0_1, X0_2, get=get)
    empirical = kernel_fn_mc(X0_1, X0_2, get=get)
    test_utils.assert_close_matrices(self, empirical, exact, tol)


if __name__ == '__main__':
  absltest.main()
