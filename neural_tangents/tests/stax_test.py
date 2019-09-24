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

"""Tests for stax.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from functools import partial
from jax import test_util as jtu
from jax.config import config as jax_config
from jax.lib import xla_bridge
import jax.numpy as np
import jax.random as random
from neural_tangents import stax
from neural_tangents.utils import monte_carlo
from neural_tangents.utils import utils


jax_config.parse_flags_with_absl()


MODELS = [
    'fc',
    'conv'
]

INPUT_SHAPE = (2, 7, 6, 3)

WIDTHS = [2**11]

N_SAMPLES = 300

RTOL = 0.01

FILTER_SIZES = [
    (1, 1),
    (2, 1),
    (3, 2)
]

PADDINGS = [
    'SAME',
    'VALID',
    'CIRCULAR'
]

STRIDES = [
    None,
    (1, 2),
    (2, 1),
]

ACTIVATIONS = {
    # TODO(romann): investigate poor erf convergence.
    stax.Erf(): 'erf',
    stax.Relu(): 'Relu',
    stax.ABRelu(-0.5, 0.7): 'ABRelu(-0.5, 0.7)'
}

PROJECTIONS = [
    'FLAT',
    'POOL',
    'ATTN_FIXED',
    'ATTN_PARAM'
]


def _get_inputs(key, is_conv, same_inputs, input_shape, fn=np.cos):
  key, split = random.split(key)
  shape = input_shape if is_conv else (input_shape[0], np.prod(input_shape[1:]))
  x1 = fn(random.normal(key, shape))
  x2 = None if same_inputs else 2 * fn(random.normal(split, shape))
  return x1, x2


def _get_net(W_std, b_std, filter_shape, is_conv, use_pooling, is_res,
             padding, phi, strides, width, is_ntk, proj_into_2d):
  fc = partial(stax.Dense, W_std=W_std, b_std=b_std)
  conv = partial(
      stax.Conv,
      filter_shape=filter_shape,
      strides=strides,
      padding=padding,
      W_std=W_std,
      b_std=b_std)
  affine = conv(width) if is_conv else fc(width)

  res_unit = stax.serial((stax.AvgPool(
      (2, 3), None, 'SAME' if padding == 'SAME' else 'CIRCULAR')
                          if use_pooling else stax.Identity()), phi, affine)

  if is_res:
    block = stax.serial(affine, stax.FanOut(2),
                        stax.parallel(stax.Identity(), res_unit),
                        stax.FanInSum())
  else:
    block = stax.serial(affine, res_unit)

  if proj_into_2d == 'FLAT':
    proj_layer = stax.Flatten()
  elif proj_into_2d == 'POOL':
    proj_layer = stax.GlobalAvgPool()
  elif proj_into_2d.startswith('ATTN'):
    n_heads = int(np.sqrt(width))
    n_chan_val = int(np.round(float(width) / n_heads))
    fixed = proj_into_2d == 'ATTN_FIXED'
    proj_layer = stax.serial(
        stax.GlobalSelfAttention(
            width, n_chan_key=width, n_chan_val=n_chan_val, n_heads=n_heads,
            fixed=fixed, W_key_std=W_std, W_value_std=W_std, W_query_std=W_std,
            W_out_std=1.0, b_std=b_std),
        stax.Flatten())
  else:
    raise ValueError(proj_into_2d)
  readout = stax.serial(proj_layer, fc(1 if is_ntk else width))

  return stax.serial(block, readout)


class StaxTest(jtu.JaxTestCase):

  # pylint: disable=g-complex-comprehension
  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              '_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                  model, phi_name, width, 'same_inputs'
                  if same_inputs else 'different_inputs', 'filter_size=%s' %
                  str(filter_size), 'padding=%s' % padding, 'strides=%s' %
                  str(strides), 'pool' if use_pooling else 'flatten', 'NTK'
                  if is_ntk else 'NNGP', 'RESNET' if is_res else 'serial',
                  proj_into_2d),
          'model':
              model,
          'width':
              width,
          'strides':
              strides,
          'padding':
              padding,
          'phi':
              phi,
          'same_inputs':
              same_inputs,
          'filter_size':
              filter_size,
          'use_pooling':
              use_pooling,
          'is_ntk':
              is_ntk,
          'is_res':
              is_res,
          'proj_into_2d':
            proj_into_2d
      } for model in MODELS for width in WIDTHS
                          for phi, phi_name in ACTIVATIONS.items()
                          for same_inputs in [False, True]
                          for padding in PADDINGS
                          for strides in STRIDES
                          for filter_size in FILTER_SIZES
                          for use_pooling in [False, True]
                          for is_ntk in [False, True]
                          for is_res in [False, True]
                          for proj_into_2d in PROJECTIONS))
  def test_exact(self, model, width, strides, padding, phi, same_inputs,
                 filter_size, use_pooling, is_ntk, is_res, proj_into_2d):
    is_conv = 'conv' in model

    # Check for duplicate / incorrectly-shaped NN configs / wrong backend.
    if is_conv:
      if xla_bridge.get_backend().platform == 'cpu':
        raise jtu.SkipTest('Not running CNN models on CPU to save time.')

      if (is_res and is_conv and ((strides is not None and strides != (1, 1)) or
                                  (padding == 'VALID' and filter_size !=
                                   (1, 1)))):
        raise jtu.SkipTest('Different paths in a residual models need to return'
                           ' outputs of the same shape.')
    elif (filter_size != FILTER_SIZES[0] or padding != PADDINGS[0] or
          strides != STRIDES[0] or proj_into_2d != PROJECTIONS[0] or
          use_pooling):
      raise jtu.SkipTest('FC models do not have these parameters.')

    if (proj_into_2d.startswith('ATTN') and strides == (2, 1) and
        padding == 'VALID' and xla_bridge.get_backend().platform == 'tpu'):
      #TODO(jirihron): speed up the vmap alternative impl or fix the current one
      raise jtu.SkipTest('ATTN forward pass on TPU is broken if one of'
                         ' the spatial dimensions is singleton.')

    W_std, b_std = 2.**0.5, 0.5**0.5

    key = random.PRNGKey(1)
    x1, x2 = _get_inputs(key, is_conv, same_inputs, INPUT_SHAPE)

    init_fn, apply_fn, kernel_fn = _get_net(W_std, b_std, filter_size,
                                            is_conv, use_pooling, is_res,
                                            padding, phi, strides, width,
                                            is_ntk, proj_into_2d)

    def _get_empirical(n_samples, get):
      kernel_fn_empirical = monte_carlo.monte_carlo_kernel_fn(
          init_fn, apply_fn, key, n_samples)
      return kernel_fn_empirical(x1, x2, get)

    if proj_into_2d == 'ATTN_PARAM':
      # no analytic kernel available, just test forward/backward pass
      _get_empirical(1, 'ntk' if is_ntk else 'nngp')
    else:
      if is_ntk:
        exact = kernel_fn(x1, x2, 'ntk')
        empirical = np.reshape(_get_empirical(N_SAMPLES, 'ntk'), exact.shape)
      else:
        exact = kernel_fn(x1, x2, 'nngp')
        empirical = _get_empirical(N_SAMPLES, 'nngp')
      utils.assert_close_matrices(self, empirical, exact, RTOL)


@jtu.parameterized.parameters([
    {
        'same_inputs': True
    },
    {
        'same_inputs': False
    },
])
class ABReluTest(jtu.JaxTestCase):

  def test_ab_relu_relu(self, same_inputs):
    key = random.PRNGKey(1)
    X0_1 = random.normal(key, (5, 7))
    fc = stax.Dense(10, 1, 0)

    # Test that ABRelu(0, 1) == ReLU
    init_fn, apply_relu, kernel_fn_relu = stax.serial(fc, stax.Relu())
    params = init_fn(key, input_shape=(-1, 7))

    X0_2 = None if same_inputs else random.normal(key, (9, 7))

    for a, b in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
      with self.subTest(a=a, b=b):
        _, apply_ab_relu, kernel_fn_ab_relu = stax.serial(fc, stax.ABRelu(a, b))

        X1_1_relu = (b - a) * apply_relu(params, X0_1 * (-1 if a != 0 else 1))
        X1_1_ab_relu = apply_ab_relu(params, X0_1)
        self.assertAllClose(X1_1_relu, X1_1_ab_relu, True)

        kernels_relu = kernel_fn_relu(X0_1, X0_2, ('NNGP', 'NTK'))
        kernels_ab_relu = kernel_fn_ab_relu(X0_1, X0_2, ('NNGP', 'NTK'))
        self.assertAllClose(kernels_relu, kernels_ab_relu, True)

  def test_ab_relu_id(self, same_inputs):
    key = random.PRNGKey(1)
    X0_1 = random.normal(key, (5, 7))
    fc = stax.Dense(10, 1, 0)

    X0_2 = None if same_inputs else random.normal(key, (9, 7))

    # Test that ABRelu(a, a) == a * Identity
    init_fn, apply_id, kernel_fn_id = stax.serial(fc, stax.Identity())
    params = init_fn(key, input_shape=(-1, 7))

    for a in [-5, -1, -0.5, 0, 0.5, 1, 5]:
      with self.subTest(a=a):
        _, apply_ab_relu, kernel_fn_ab_relu = stax.serial(fc, stax.ABRelu(a, a))

        X1_1_id = a * apply_id(params, X0_1)
        X1_1_ab_relu = apply_ab_relu(params, X0_1)
        self.assertAllClose(X1_1_id, X1_1_ab_relu, True)

        kernels_id = kernel_fn_id(
            X0_1 * a, None if X0_2 is None else a * X0_2, ('NNGP', 'NTK'))
        kernels_ab_relu = kernel_fn_ab_relu(X0_1, X0_2, ('NNGP', 'NTK'))
        self.assertAllClose(kernels_id, kernels_ab_relu, True)

  def test_leaky_relu(self, same_inputs):
    key = random.PRNGKey(1)
    X0_1 = random.normal(key, (5, 7))
    fc = stax.Dense(10, 1, 0)

    X0_2 = None if same_inputs else random.normal(key, (9, 7))

    # Test that ABRelu(alpha, 1) == LeakyRelu(alpha)
    for a in [-2, -1, 0, 1, 2]:
      with self.subTest(alpha=a):
        init_fn, apply_leaky_relu, kernel_fn_leaky_relu = stax.serial(
            fc, stax.LeakyRelu(a))
        _, apply_ab_relu, kernel_fn_ab_relu = stax.serial(fc, stax.ABRelu(a, 1))

        params = init_fn(key, input_shape=(-1, 7))
        X1_1_leaky_relu = apply_leaky_relu(params, X0_1)
        X1_1_ab_relu = apply_ab_relu(params, X0_1)
        self.assertAllClose(X1_1_leaky_relu, X1_1_ab_relu, True)

        kernels_leaky_relu = kernel_fn_leaky_relu(X0_1, X0_2, ('NNGP', 'NTK'))
        kernels_ab_relu = kernel_fn_ab_relu(X0_1, X0_2, ('NNGP', 'NTK'))
        self.assertAllClose(kernels_leaky_relu, kernels_ab_relu, True)

  def test_abs(self, same_inputs):
    key = random.PRNGKey(1)
    X0_1 = random.normal(key, (5, 7))
    fc = stax.Dense(10, 1, 0)

    X0_2 = None if same_inputs else random.normal(key, (9, 7))

    # Test that Abs == ABRelu(-1, 1)
    init_fn, apply_leaky_relu, kernel_fn_abs = stax.serial(fc, stax.Abs())
    _, apply_ab_relu, kernel_fn_ab_relu = stax.serial(fc, stax.ABRelu(-1, 1))

    params = init_fn(key, input_shape=(-1, 7))
    X1_1_abs = apply_leaky_relu(params, X0_1)
    X1_1_ab_relu = apply_ab_relu(params, X0_1)
    self.assertAllClose(X1_1_abs, X1_1_ab_relu, True)

    kernels_abs = kernel_fn_abs(X0_1, X0_2, ('NNGP', 'NTK'))
    kernels_ab_relu = kernel_fn_ab_relu(X0_1, X0_2, ('NNGP', 'NTK'))
    self.assertAllClose(kernels_abs, kernels_ab_relu, True)


if __name__ == '__main__':
  jtu.absltest.main()
