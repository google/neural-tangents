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


import string
from functools import partial
from itertools import product

import random as prandom
import itertools
import logging
from absl.testing import absltest
from jax.api import jit
from jax import ops
from jax import test_util as jtu
from jax.config import config as jax_config
from jax.lib import xla_bridge
import jax.numpy as np
import jax.random as random
from neural_tangents import stax
from neural_tangents.utils import monte_carlo
from neural_tangents.utils import test_utils
import numpy as onp
import unittest
from typing import Tuple


jax_config.parse_flags_with_absl()


MODELS = [
    'fc',
    'conv'
]

BATCH_SIZE = 2

INPUT_SHAPE = (BATCH_SIZE, 8, 6, 4)

WIDTHS = [2**10]

N_SAMPLES = 100

RTOL = 0.025

FILTER_SHAPES = [
    (2, 1),
    (3, 2)
]

PADDINGS = [
    'SAME',
    'VALID',
    'CIRCULAR'
]

STRIDES = [
    (1, 2),
    (2, 1),
]

ACTIVATIONS = {
    stax.Erf(): 'erf',
    stax.Relu(): 'Relu',
}

PROJECTIONS = [
    'FLAT',
    'POOL',
    'ATTN_FIXED',
    'ATTN_PARAM'
]

LAYER_NORM = [
    'C',
    'HC',
    'CHW',
    'NC',
    'NWC',
    'NCHW'
]

POOL_TYPES = [
    'SUM',
    'AVG'
]

PARAMETERIZATIONS = [
    'NTK',
    'STANDARD'
]

test_utils.update_test_tolerance()


def _get_inputs(
    key,
    same_inputs,
    shape,
    fn=np.cos
) -> Tuple[np.ndarray, np.ndarray]:
  key, split = random.split(key)
  x1 = fn(random.normal(key, shape))
  batch_axis = shape.index(BATCH_SIZE)
  shape = shape[:batch_axis] + (2 * BATCH_SIZE,) + shape[batch_axis + 1:]
  x2 = None if same_inputs else fn(random.normal(split, shape)) * 2
  return x1, x2


def _get_net(W_std, b_std, filter_shape, is_conv, use_pooling, is_res, padding,
             phi, strides, width, is_ntk, proj_into_2d, pool_type, layer_norm,
             parameterization, use_dropout):

  if is_conv:
    # Select a random filter order.
    default_filter_spec = 'HW'
    filter_specs = [''.join(p) for p in itertools.permutations('HWIO')]
    filter_spec = prandom.choice(filter_specs)
    filter_shape = tuple(filter_shape[default_filter_spec.index(c)]
                         for c in filter_spec if c in default_filter_spec)
    strides = tuple(strides[default_filter_spec.index(c)]
                    for c in filter_spec if c in default_filter_spec)

    # Select the activation order.
    default_spec = 'NHWC'
    if xla_bridge.get_backend().platform == 'tpu':
      # Keep batch dimension leading for TPU for batching to work.
      specs = ['N' + ''.join(p) for p in itertools.permutations('CHW')]
    else:
      specs = [''.join(p) for p in itertools.permutations('NCHW')]
    spec = prandom.choice(specs)
    input_shape = tuple(INPUT_SHAPE[default_spec.index(c)] for c in spec)

  else:
    input_shape = (INPUT_SHAPE[0], onp.prod(INPUT_SHAPE[1:]))
    if xla_bridge.get_backend().platform == 'tpu':
      spec = 'NC'
    else:
      spec = prandom.choice(['NC', 'CN'])
      if spec.index('N') == 1:
        input_shape = input_shape[::-1]

    filter_spec = None

  dimension_numbers = (spec, filter_spec, spec)
  batch_axis, channel_axis = spec.index('N'), spec.index('C')

  spec_fc = ''.join(c for c in spec if c in ('N', 'C'))
  batch_axis_fc, channel_axis_fc = spec_fc.index('N'), spec_fc.index('C')

  if not is_conv:
    batch_axis = batch_axis_fc
    channel_axis = channel_axis_fc

  if layer_norm:
    layer_norm = tuple(spec.index(c) for c in layer_norm)

  logging.warning(f'DIMENSION NUMBERS: {dimension_numbers}')

  def fc(out_dim):
    return stax.Dense(
        out_dim=out_dim,
        W_std=W_std,
        b_std=b_std,
        parameterization=parameterization,
        batch_axis=batch_axis_fc,
        channel_axis=channel_axis_fc
    )

  def conv(out_chan):
    return stax.GeneralConv(
        dimension_numbers=dimension_numbers,
        out_chan=out_chan,
        filter_shape=filter_shape,
        strides=strides,
        padding=padding,
        W_std=W_std,
        b_std=b_std,
        parameterization=parameterization
    )

  affine = conv(width) if is_conv else fc(width)

  rate = onp.random.uniform(0.5, 0.9)
  dropout = stax.Dropout(rate, mode='train')

  if pool_type == 'AVG':
    pool_fn = stax.AvgPool
    global_pool_fn = stax.GlobalAvgPool
  elif pool_type == 'SUM':
    pool_fn = stax.SumPool
    global_pool_fn = stax.GlobalSumPool
  else:
    raise ValueError(pool_type)

  if use_pooling:
    pool_or_identity = pool_fn((2, 3),
                               None,
                               'SAME' if padding == 'SAME' else 'CIRCULAR',
                               batch_axis=batch_axis,
                               channel_axis=channel_axis)
  else:
    pool_or_identity = stax.Identity()
  dropout_or_identity = dropout if use_dropout else stax.Identity()
  layer_norm_or_identity = (stax.Identity() if layer_norm is None else
                            stax.LayerNorm(axis=layer_norm,
                                           batch_axis=batch_axis,
                                           channel_axis=channel_axis))
  res_unit = stax.serial(dropout_or_identity, affine, pool_or_identity)
  if is_res:
    block = stax.serial(
        affine,
        stax.FanOut(2),
        stax.parallel(stax.Identity(),
                      res_unit),
        stax.FanInSum(),
        layer_norm_or_identity,
        phi)
  else:
    block = stax.serial(
        affine,
        res_unit,
        layer_norm_or_identity,
        phi)

  if proj_into_2d == 'FLAT':
    proj_layer = stax.Flatten(batch_axis, batch_axis_fc)
  elif proj_into_2d == 'POOL':
    proj_layer = global_pool_fn(batch_axis, channel_axis)
  elif proj_into_2d.startswith('ATTN'):
    n_heads = int(np.sqrt(width))
    n_chan_val = int(np.round(float(width) / n_heads))
    fixed = proj_into_2d == 'ATTN_FIXED'
    proj_layer = stax.serial(
        stax.GlobalSelfAttention(
            n_chan_out=width,
            n_chan_key=width,
            n_chan_val=n_chan_val,
            n_heads=n_heads,
            fixed=fixed,
            W_key_std=W_std,
            W_value_std=W_std,
            W_query_std=W_std,
            W_out_std=1.0,
            b_std=b_std,
            batch_axis=batch_axis,
            channel_axis=channel_axis),
        stax.Flatten(batch_axis, batch_axis_fc))
  else:
    raise ValueError(proj_into_2d)
  readout = stax.serial(proj_layer, fc(1 if is_ntk else width))

  device_count = -1 if spec.index('N') == 0 else 0
  return stax.serial(block, readout), input_shape, device_count, channel_axis_fc


def _get_net_pool(width, is_ntk, pool_type, padding,
                  filter_shape, strides, normalize_edges):
  W_std, b_std = 2.**0.5, 0.5**0.5
  phi = stax.Relu()
  parameterization = 'ntk'

  fc = partial(
      stax.Dense, W_std=W_std, b_std=b_std, parameterization=parameterization)
  conv = partial(
      stax.Conv,
      filter_shape=(3, 2),
      strides=None,
      padding='SAME',
      W_std=W_std,
      b_std=b_std,
      parameterization=parameterization)

  if pool_type == 'AVG':
    pool_fn = partial(stax.AvgPool, normalize_edges=normalize_edges)
    global_pool_fn = stax.GlobalAvgPool
  elif pool_type == 'SUM':
    pool_fn = stax.SumPool
    global_pool_fn = stax.GlobalSumPool
  else:
    raise ValueError(pool_type)

  pool = pool_fn(filter_shape, strides, padding)

  return stax.serial(
      conv(width), phi, pool, conv(width), phi, global_pool_fn(),
      fc(1 if is_ntk else width)), INPUT_SHAPE, -1, -1


class StaxTest(test_utils.NeuralTangentsTestCase):

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              '_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                  model, phi_name, width, 'same_inputs'
                  if same_inputs else 'different_inputs', 'filter_shape=%s' %
                  str(filter_shape), 'padding=%s' % padding, 'strides=%s' %
                  str(strides), 'pool' if use_pooling else 'flatten',
                  'NTK' if is_ntk else 'NNGP', 'RESNET' if is_res else 'serial',
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
          'filter_shape':
              filter_shape,
          'use_pooling':
              use_pooling,
          'is_ntk':
              is_ntk,
          'is_res':
              is_res,
          'proj_into_2d':
              proj_into_2d
      }
                          for model in MODELS
                          for width in WIDTHS
                          for phi, phi_name in ACTIVATIONS.items()
                          for same_inputs in [False]
                          for padding in PADDINGS for strides in STRIDES
                          for filter_shape in FILTER_SHAPES
                          for use_pooling in [False, True]
                          for is_ntk in [False, True]
                          for is_res in [False, True]
                          for proj_into_2d in PROJECTIONS))
  def test_exact(self, model, width, strides, padding, phi, same_inputs,
                 filter_shape, use_pooling, is_ntk, is_res, proj_into_2d):
    is_conv = 'conv' in model

    # Check for duplicate / incorrectly-shaped NN configs / wrong backend.
    if is_conv:
      if xla_bridge.get_backend().platform == 'cpu':
        raise unittest.SkipTest('Not running CNN models on CPU to save time.')

      if (is_res and is_conv and ((strides is not None and strides != (1, 1)) or
                                  (padding == 'VALID' and filter_shape !=
                                   (1, 1)))):
        raise unittest.SkipTest('Different paths in a residual models need to '
                                'return outputs of the same shape.')
    elif (filter_shape != FILTER_SHAPES[0] or padding != PADDINGS[0] or
          strides != STRIDES[0] or proj_into_2d != PROJECTIONS[0] or
          use_pooling):
      raise unittest.SkipTest('FC models do not have these parameters.')

    pool_type = 'AVG'
    W_std, b_std = 2.**0.5, 0.5**0.5
    layer_norm = None
    parameterization = 'ntk'
    use_dropout = False

    net = _get_net(W_std, b_std, filter_shape, is_conv, use_pooling, is_res,
                   padding, phi, strides, width, is_ntk, proj_into_2d,
                   pool_type, layer_norm, parameterization, use_dropout)
    self._check_agreement_with_empirical(net, same_inputs, use_dropout, is_ntk,
                                         proj_into_2d)

  # pylint: disable=g-complex-comprehension
  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              '_{}_{}_{}_{}_{}_{}_{}'.format(
                  model, width, 'same_inputs'
                  if same_inputs else 'different_inputs', 'filter_shape=%s' %
                  str(filter_shape), proj_into_2d, 'NTK' if is_ntk else 'NNGP',
                  'parameterization=%s' % str(parameterization)),
          'model':
              model,
          'width':
              width,
          'same_inputs':
              same_inputs,
          'filter_shape':
              filter_shape,
          'proj_into_2d':
              proj_into_2d,
          'is_ntk':
              is_ntk,
          'parameterization':
              parameterization
      } for model in MODELS for width in WIDTHS
                          for same_inputs in [False]
                          for is_ntk in [False, True]
                          for filter_shape in FILTER_SHAPES
                          for proj_into_2d in PROJECTIONS[:2]
                          for parameterization in PARAMETERIZATIONS))
  def test_parameterizations(self, model, width, same_inputs, is_ntk,
                             filter_shape, proj_into_2d, parameterization):
    is_conv = 'conv' in model

    W_std, b_std = 2.**0.5, 0.5**0.5
    padding = PADDINGS[0]
    strides = STRIDES[0]
    phi = stax.Relu()
    use_pooling, is_res = False, False
    layer_norm = None
    pool_type = 'AVG'
    use_dropout = False

    # Check for duplicate / incorrectly-shaped NN configs / wrong backend.
    if is_conv:
      if xla_bridge.get_backend().platform == 'cpu':
        raise unittest.SkipTest('Not running CNN models on CPU to save time.')
    elif proj_into_2d != PROJECTIONS[0]:
      raise unittest.SkipTest('FC models do not have these parameters.')

    net = _get_net(W_std, b_std, filter_shape, is_conv, use_pooling, is_res,
                   padding, phi, strides, width, is_ntk, proj_into_2d,
                   pool_type, layer_norm, parameterization, use_dropout)
    self._check_agreement_with_empirical(net, same_inputs, use_dropout, is_ntk,
                                         proj_into_2d)

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              '_{}_{}_{}_{}_{}_{}'.format(
                  model,
                  width,
                  'same_inputs' if same_inputs else 'different_inputs',
                  'NTK' if is_ntk else 'NNGP',
                  proj_into_2d,
                  'layer_norm=%s' % str(layer_norm)),
          'model':
              model,
          'width':
              width,
          'same_inputs':
              same_inputs,
          'is_ntk':
              is_ntk,
          'proj_into_2d':
              proj_into_2d,
          'layer_norm':
              layer_norm
      }
                          for model in MODELS
                          for width in WIDTHS
                          for same_inputs in [False]
                          for is_ntk in [False, True]
                          for proj_into_2d in PROJECTIONS[:2]
                          for layer_norm in LAYER_NORM))
  def test_layernorm(self,
                     model,
                     width,
                     same_inputs,
                     is_ntk,
                     proj_into_2d,
                     layer_norm):
    is_conv = 'conv' in model
    # Check for duplicate / incorrectly-shaped NN configs / wrong backend.
    if is_conv:
      if xla_bridge.get_backend().platform == 'cpu':
        raise unittest.SkipTest('Not running CNN models on CPU to save time.')
    elif proj_into_2d != PROJECTIONS[0] or layer_norm not in ('C', 'NC'):
      raise unittest.SkipTest('FC models do not have these parameters.')

    W_std, b_std = 2.**0.5, 0.5**0.5
    filter_shape = FILTER_SHAPES[0]
    padding = PADDINGS[0]
    strides = STRIDES[0]
    phi = stax.Relu()
    use_pooling, is_res = False, False
    parameterization = 'ntk'
    pool_type = 'AVG'
    use_dropout = False

    net = _get_net(W_std, b_std, filter_shape, is_conv, use_pooling, is_res,
                   padding, phi, strides, width, is_ntk, proj_into_2d,
                   pool_type, layer_norm, parameterization, use_dropout)
    self._check_agreement_with_empirical(net, same_inputs, use_dropout, is_ntk,
                                         proj_into_2d, True)

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              '_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                  width, 'same_inputs' if same_inputs else 'different_inputs',
                  'filter_shape=%s' % str(filter_shape), 'padding=%s' %
                  padding, 'strides=%s' % str(strides),
                  'NTK' if is_ntk else 'NNGP', 'pool_type=%s' %
                  str(pool_type), 'normalize_edges=%s' % str(normalize_edges)),
          'width':
              width,
          'same_inputs':
              same_inputs,
          'is_ntk':
              is_ntk,
          'pool_type':
              pool_type,
          'padding':
              padding,
          'filter_shape':
              filter_shape,
          'strides':
              strides,
          'normalize_edges':
              normalize_edges
      } for width in WIDTHS for same_inputs in [False, True]
                          for is_ntk in [False, True]
                          for pool_type in POOL_TYPES for padding in PADDINGS
                          for filter_shape in FILTER_SHAPES
                          for strides in STRIDES
                          for normalize_edges in [True, False]))
  def test_pool(self, width, same_inputs, is_ntk, pool_type,
                padding, filter_shape, strides, normalize_edges):
    is_conv = True
    use_dropout = False
    proj_into_2d = 'POOL'
    # Check for duplicate / incorrectly-shaped NN configs / wrong backend.

    if xla_bridge.get_backend().platform == 'cpu':
      raise unittest.SkipTest('Not running CNN models on CPU to save time.')
    if pool_type == 'SUM' and normalize_edges:
      raise unittest.SkipTest('normalize_edges not applicable to SumPool.')

    net = _get_net_pool(width, is_ntk, pool_type,
                        padding, filter_shape, strides, normalize_edges)
    self._check_agreement_with_empirical(net, same_inputs, use_dropout, is_ntk,
                                         proj_into_2d)

  def test_avg_pool(self):
    X1 = np.ones((4, 2, 3, 2))
    X2 = np.ones((3, 2, 3, 2))

    _, apply_fn, kernel_fn = stax.AvgPool((2, 2), (1, 1), 'SAME',
                                          normalize_edges=False)
    _, apply_fn_norm, kernel_fn_norm = stax.AvgPool((2, 2), (1, 1), 'SAME',
                                                    normalize_edges=True)
    _, apply_fn_stax = stax.ostax.AvgPool((2, 2), (1, 1), 'SAME')

    out1 = apply_fn((), X1)
    out2 = apply_fn((), X2)

    out1_norm = apply_fn_norm((), X1)
    out2_norm = apply_fn_norm((), X2)

    out1_stax = apply_fn_stax((), X1)
    out2_stax = apply_fn_stax((), X2)

    self.assertAllClose((out1_stax, out2_stax), (out1_norm, out2_norm))

    out_unnorm = np.array([[1., 1., 0.5], [0.5, 0.5, 0.25]]).reshape(
        (1, 2, 3, 1))
    out1_unnormalized = np.broadcast_to(out_unnorm, X1.shape)
    out2_unnormalized = np.broadcast_to(out_unnorm, X2.shape)

    self.assertAllClose((out1_unnormalized, out2_unnormalized), (out1, out2))

    ker = kernel_fn(X1, X2)
    ker_norm = kernel_fn_norm(X1, X2)

    self.assertAllClose(np.ones_like(ker_norm.nngp), ker_norm.nngp)
    self.assertAllClose(np.ones_like(ker_norm.cov1), ker_norm.cov1)
    self.assertAllClose(np.ones_like(ker_norm.cov2), ker_norm.cov2)

    self.assertEqual(ker_norm.nngp.shape, ker.nngp.shape)
    self.assertEqual(ker_norm.cov1.shape, ker.cov1.shape)
    self.assertEqual(ker_norm.cov2.shape, ker.cov2.shape)

    ker_unnorm = np.outer(out_unnorm, out_unnorm).reshape((2, 3, 2, 3))
    ker_unnorm = np.transpose(ker_unnorm, axes=(0, 2, 1, 3))
    nngp = np.broadcast_to(
        ker_unnorm.reshape((1, 1) + ker_unnorm.shape), ker.nngp.shape)
    cov1 = np.broadcast_to(np.expand_dims(ker_unnorm, 0), ker.cov1.shape)
    cov2 = np.broadcast_to(np.expand_dims(ker_unnorm, 0), ker.cov2.shape)
    self.assertAllClose((nngp, cov1, cov2), (ker.nngp, ker.cov1, ker.cov2))

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              '_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                  model, phi_name, width, 'same_inputs'
                  if same_inputs else 'different_inputs', 'filter_shape=%s' %
                  str(filter_shape), 'padding=%s' % padding, 'strides=%s' %
                  str(strides), 'pool' if use_pooling else 'flatten',
                  'NTK' if is_ntk else 'NNGP', proj_into_2d),
          'model':
              model,
          'width':
              width,
          'same_inputs':
              same_inputs,
          'is_ntk':
              is_ntk,
          'padding':
              padding,
          'strides':
              strides,
          'filter_shape':
              filter_shape,
          'phi':
              phi,
          'use_pooling':
              use_pooling,
          'proj_into_2d':
              proj_into_2d
      } for model in MODELS for width in WIDTHS
                          for same_inputs in [True, False]
                          for phi, phi_name in ACTIVATIONS.items()
                          for padding in ['SAME'] for strides in STRIDES
                          for filter_shape in [(2, 1)]
                          for is_ntk in [True, False]
                          for use_pooling in [True, False]
                          for proj_into_2d in ['FLAT', 'POOL']))
  def test_dropout(self, model, width, same_inputs, is_ntk, padding, strides,
                   filter_shape, phi, use_pooling, proj_into_2d):
    if xla_bridge.get_backend().platform == 'tpu' and same_inputs:
      raise unittest.SkipTest(
          'Skip TPU test for `same_inputs`. Need to handle '
          'random keys carefully for dropout + empirical kernel.')

    pool_type = 'AVG'
    use_dropout = True
    is_conv = 'conv' in model
    is_res = False
    # Check for duplicate / incorrectly-shaped NN configs / wrong backend.
    W_std, b_std = 2.**0.5, 0.5**0.5
    layer_norm = None
    parameterization = 'ntk'
    if is_conv:
      if xla_bridge.get_backend().platform == 'cpu':
        raise unittest.SkipTest('Not running CNN models on CPU to save time.')

      if (is_res and is_conv and ((strides is not None and strides != (1, 1)) or
                                  (padding == 'VALID' and filter_shape !=
                                   (1, 1)))):
        raise unittest.SkipTest('Different paths in a residual models need to '
                                'return outputs of the same shape.')
    elif (filter_shape != FILTER_SHAPES[0] or padding != PADDINGS[0] or
          strides != STRIDES[0] or proj_into_2d != PROJECTIONS[0] or
          use_pooling):
      raise unittest.SkipTest('FC models do not have these parameters.')

    net = _get_net(W_std, b_std, filter_shape, is_conv, use_pooling, is_res,
                   padding, phi, strides, width, is_ntk, proj_into_2d,
                   pool_type, layer_norm, parameterization, use_dropout)
    self._check_agreement_with_empirical(net, same_inputs, use_dropout, is_ntk,
                                         proj_into_2d)

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name': '_act={}_kernel={}'.format(act, kern),
          'act': act,
          'kernel': kern
      }
                          for act in ['erf', 'relu']
                          for kern in ['nngp', 'ntk']))
  def test_sparse_inputs(self, act, kernel):
    key = random.PRNGKey(1)

    input_count = 4
    sparse_count = 2
    input_size = 128
    width = 4096

    # NOTE(schsam): It seems that convergence is slower when inputs are sparse.
    samples = N_SAMPLES

    if xla_bridge.get_backend().platform == 'gpu':
      jtu._default_tolerance[onp.dtype(onp.float64)] = 5e-4
      samples = 100 * N_SAMPLES
    else:
      jtu._default_tolerance[onp.dtype(onp.float32)] = 5e-2
      jtu._default_tolerance[onp.dtype(onp.float64)] = 5e-3

    # a batch of dense inputs
    x_dense = random.normal(key, (input_count, input_size))
    x_sparse = ops.index_update(x_dense, ops.index[:sparse_count, :], 0.)

    activation = stax.Relu() if act == 'relu' else stax.Erf()

    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(width),
        activation,
        stax.Dense(1 if kernel == 'ntk' else width))
    exact = kernel_fn(x_sparse, None, kernel)
    mc = monte_carlo.monte_carlo_kernel_fn(init_fn, apply_fn,
                                           random.split(key, 2)[0],
                                           samples)(x_sparse, None, kernel)
    mc = np.reshape(mc, exact.shape)

    assert not np.any(np.isnan(exact))
    self.assertAllClose(exact[sparse_count:, sparse_count:],
                        mc[sparse_count:, sparse_count:])

  def test_composition_dense(self):
    rng = random.PRNGKey(0)
    x1 = random.normal(rng, (10, 10))
    x2 = random.normal(rng, (10, 10))

    Block = stax.serial(stax.Dense(256), stax.Relu())

    _, _, ker_fn = Block
    _, _, composed_ker_fn = stax.serial(Block, Block)

    ker_out = ker_fn(ker_fn(x1))
    composed_ker_out = composed_ker_fn(x1)
    self.assertAllClose(ker_out, composed_ker_out)

    ker_out = ker_fn(ker_fn(x1, x2))
    composed_ker_out = composed_ker_fn(x1, x2)
    self.assertAllClose(ker_out, composed_ker_out)

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name': '_avg_pool={}_same_inputs={}'.format(avg_pool,
                                                                same_inputs),
          'avg_pool': avg_pool,
          'same_inputs': same_inputs
      }
                          for avg_pool in [True, False]
                          for same_inputs in [True, False]))
  def test_composition_conv(self, avg_pool, same_inputs):
    rng = random.PRNGKey(0)
    x1 = random.normal(rng, (5, 10, 10, 3))
    x2 = None if same_inputs else random.normal(rng, (5, 10, 10, 3))

    Block = stax.serial(stax.Conv(256, (3, 3)), stax.Relu())
    if avg_pool:
      Readout = stax.serial(stax.GlobalAvgPool(), stax.Dense(10))
    else:
      Readout = stax.serial(stax.Flatten(), stax.Dense(10))

    block_ker_fn, readout_ker_fn = Block[2], Readout[2]
    _, _, composed_ker_fn = stax.serial(Block, Readout)

    composed_ker_out = composed_ker_fn(x1, x2)
    ker_out_no_marg = readout_ker_fn(block_ker_fn(x1, x2,
                                                  diagonal_spatial=False))
    ker_out_default = readout_ker_fn(block_ker_fn(x1, x2))
    self.assertAllClose(composed_ker_out, ker_out_no_marg)
    self.assertAllClose(composed_ker_out, ker_out_default)

    if avg_pool:
      with self.assertRaises(ValueError):
        ker_out = readout_ker_fn(block_ker_fn(x1, x2, diagonal_spatial=True))
    else:
      ker_out_marg = readout_ker_fn(block_ker_fn(x1, x2,
                                                 diagonal_spatial=True))
      self.assertAllClose(composed_ker_out, ker_out_marg)

  def _check_agreement_with_empirical(
      self,
      net,
      same_inputs,
      use_dropout,
      is_ntk,
      proj_into_2d,
      use_layer_norm=False):
    ((init_fn, apply_fn, kernel_fn),
     input_shape, device_count, channel_axis) = net

    num_samples = N_SAMPLES * 5 if use_dropout else N_SAMPLES
    key = random.PRNGKey(1)
    x1, x2 = _get_inputs(key, same_inputs, input_shape)

    x1_out_shape, params = init_fn(key, x1.shape)
    if same_inputs:
      assert x2 is None
    if x2 is None:
      x2_out_shape = x1_out_shape
    else:
      x2_out_shape, params = init_fn(key, x2.shape)
    del params

    def _get_empirical(n_samples, get):
      kernel_fn_empirical = monte_carlo.monte_carlo_kernel_fn(
          init_fn, apply_fn, key, n_samples, device_count=device_count,
          trace_axes=(channel_axis,)
      )
      if same_inputs:
        assert x2 is None
      return kernel_fn_empirical(x1, x2, get)

    if proj_into_2d == 'ATTN_PARAM':
      # no analytic kernel available, just test forward/backward pass
      _get_empirical(1, 'ntk' if is_ntk else 'nngp')
    else:
      platform = xla_bridge.get_backend().platform
      if proj_into_2d == 'ATTN_FIXED':
        if platform == 'tpu':
          rtol = 0.08
        else:
          rtol = 0.04
      else:
        if use_layer_norm and platform == 'tpu':
          rtol = 0.05
        else:
          rtol = RTOL

      if is_ntk:
        exact, shape1, shape2 = kernel_fn(x1, x2, ('ntk', 'shape1', 'shape2'))
        empirical = np.reshape(_get_empirical(num_samples, 'ntk'), exact.shape)
      else:
        exact, shape1, shape2 = kernel_fn(x1, x2, ('nngp', 'shape1', 'shape2'))
        empirical = _get_empirical(num_samples, 'nngp')
      test_utils.assert_close_matrices(self, exact, empirical, rtol)
      self.assertEqual(shape1, x1_out_shape)
      self.assertEqual(shape2, x2_out_shape)



@jtu.parameterized.parameters([
    {
        'same_inputs': True
    },
    {
        'same_inputs': False
    },
])
class SinTest(test_utils.NeuralTangentsTestCase):

  def test_sin(self, same_inputs):
    key = random.PRNGKey(1)
    for a, b, c in product([5.],
                           [1.5],
                           [0., -np.pi/4.]):
      for get in ['nngp', 'ntk']:
        output_dim = 2048 if get == 'nngp' else 1
        key, split = random.split(key)
        for model in ['fc', 'conv-pool', 'conv-flatten']:
          with self.subTest(get=get, a=a, b=b, c=c, model=model):
            if model == 'fc':
              X0_1 = random.normal(key, (6, 7))
              X0_2 = None if same_inputs else random.normal(split, (10, 7))
              affine = stax.Dense(2048, 1., 0.)
              readout = stax.Dense(output_dim)
            else:
              if xla_bridge.get_backend().platform == 'cpu':
                raise unittest.SkipTest('Not running CNNs on CPU to save time.')
              X0_1 = random.normal(key, (4, 8, 8, 3))
              X0_2 = None if same_inputs else random.normal(split, (6, 8, 8, 3))
              affine = stax.Conv(1024, (3, 2), W_std=1., b_std=0.1,
                                 padding='SAME')
              readout = stax.serial(stax.GlobalAvgPool() if 'pool' in model else
                                    stax.Flatten(),
                                    stax.Dense(output_dim))
            init_fn, apply_sin, kernel_fn_sin = stax.serial(affine,
                                                            stax.Sin(a=a,
                                                                     b=b,
                                                                     c=c),
                                                            readout)
            analytic_kernel = kernel_fn_sin(X0_1, X0_2, get)
            key, split = random.split(key)
            mc_kernel_fn = monte_carlo.monte_carlo_kernel_fn(
                init_fn, apply_sin, key, 200)
            empirical_kernel = np.squeeze(mc_kernel_fn(X0_1, X0_2, get))
            test_utils.assert_close_matrices(self, analytic_kernel,
                                             empirical_kernel, 0.05)


@jtu.parameterized.parameters([
    {
        'same_inputs': True
    },
    {
        'same_inputs': False
    },
])
class ABReluTest(test_utils.NeuralTangentsTestCase):

  def test_ab_relu_relu(self, same_inputs):
    key = random.PRNGKey(1)
    X0_1 = random.normal(key, (5, 7))
    fc = stax.Dense(10, 1, 0)

    # Test that ABRelu(0, 1) == ReLU
    init_fn, apply_relu, kernel_fn_relu = stax.serial(fc, stax.Relu())
    _, params = init_fn(key, input_shape=(-1, 7))

    X0_2 = None if same_inputs else random.normal(key, (9, 7))

    for a, b in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
      with self.subTest(a=a, b=b):
        _, apply_ab_relu, kernel_fn_ab_relu = stax.serial(fc, stax.ABRelu(a, b))

        X1_1_relu = (b - a) * apply_relu(params, X0_1 * (-1 if a != 0 else 1))
        X1_1_ab_relu = apply_ab_relu(params, X0_1)
        self.assertAllClose(X1_1_relu, X1_1_ab_relu)

        kernels_relu = kernel_fn_relu(X0_1, X0_2)
        kernels_ab_relu = kernel_fn_ab_relu(X0_1, X0_2)
        self.assertAllClose(kernels_relu, kernels_ab_relu)

  def test_ab_relu_id(self, same_inputs):
    key = random.PRNGKey(1)
    X0_1 = random.normal(key, (5, 7))
    fc = stax.Dense(10, 1, 0)

    X0_2 = None if same_inputs else random.normal(key, (9, 7))

    # Test that ABRelu(a, a) == a * Identity
    init_fn, apply_id, kernel_fn_id = stax.serial(fc, stax.Identity())
    _, params = init_fn(key, input_shape=(-1, 7))

    for a in [-5, -1, -0.5, 0, 0.5, 1, 5]:
      with self.subTest(a=a):
        _, apply_ab_relu, kernel_fn_ab_relu = stax.serial(fc, stax.ABRelu(a, a))

        X1_1_id = a * apply_id(params, X0_1)
        X1_1_ab_relu = apply_ab_relu(params, X0_1)
        self.assertAllClose(X1_1_id, X1_1_ab_relu)

        kernels_id = kernel_fn_id(X0_1 * a, None if X0_2 is None else a * X0_2)
        kernels_ab_relu = kernel_fn_ab_relu(X0_1, X0_2)
        self.assertAllClose(kernels_id, kernels_ab_relu)

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

        _, params = init_fn(key, input_shape=(-1, 7))
        X1_1_leaky_relu = apply_leaky_relu(params, X0_1)
        X1_1_ab_relu = apply_ab_relu(params, X0_1)
        self.assertAllClose(X1_1_leaky_relu, X1_1_ab_relu)

        kernels_leaky_relu = kernel_fn_leaky_relu(X0_1, X0_2)
        kernels_ab_relu = kernel_fn_ab_relu(X0_1, X0_2)
        self.assertAllClose(kernels_leaky_relu, kernels_ab_relu)

  def test_abs(self, same_inputs):
    key = random.PRNGKey(1)
    X0_1 = random.normal(key, (5, 7))
    fc = stax.Dense(10, 1, 0)

    X0_2 = None if same_inputs else random.normal(key, (9, 7))

    # Test that Abs == ABRelu(-1, 1)
    init_fn, apply_leaky_relu, kernel_fn_abs = stax.serial(fc, stax.Abs())
    _, apply_ab_relu, kernel_fn_ab_relu = stax.serial(fc, stax.ABRelu(-1, 1))

    _, params = init_fn(key, input_shape=(-1, 7))
    X1_1_abs = apply_leaky_relu(params, X0_1)
    X1_1_ab_relu = apply_ab_relu(params, X0_1)
    self.assertAllClose(X1_1_abs, X1_1_ab_relu)

    kernels_abs = kernel_fn_abs(X0_1, X0_2, ('nngp', 'ntk'))
    kernels_ab_relu = kernel_fn_ab_relu(X0_1, X0_2, ('nngp', 'ntk'))
    self.assertAllClose(kernels_abs, kernels_ab_relu)


@jtu.parameterized.parameters([
    {
        'same_inputs': True
    },
    {
        'same_inputs': False
    },
])
class FlattenTest(test_utils.NeuralTangentsTestCase):

  def test_flatten(self, same_inputs):
    key = random.PRNGKey(1)
    X0_1 = random.normal(key, (8, 4, 3, 2))
    X0_2 = None if same_inputs else random.normal(key, (4, 4, 3, 2))

    X0_1_flat = np.reshape(X0_1, (X0_1.shape[0], -1))
    X0_2_flat = None if same_inputs else np.reshape(X0_2, (X0_2.shape[0], -1))

    init_fc, apply_fc, kernel_fc = stax.serial(stax.Dense(1024, 2., 0.5),
                                               stax.Relu(),
                                               stax.Dense(1024, 2., 0.5))
    init_top, apply_top, kernel_top = stax.serial(stax.Dense(1024, 2., 0.5),
                                                  stax.Relu(),
                                                  stax.Dense(1024, 2., 0.5),
                                                  stax.Flatten())
    init_mid, apply_mid, kernel_mid = stax.serial(stax.Dense(1024, 2., 0.5),
                                                  stax.Relu(),
                                                  stax.Flatten(),
                                                  stax.Dense(1024, 2., 0.5))
    init_bot, apply_bot, kernel_bot = stax.serial(stax.Flatten(),
                                                  stax.Dense(1024, 2., 0.5),
                                                  stax.Relu(),
                                                  stax.Dense(1024, 2., 0.5))

    kernel_fc_mc = monte_carlo.monte_carlo_kernel_fn(init_fc, apply_fc, key,
                                                     200)
    kernel_bot_mc = monte_carlo.monte_carlo_kernel_fn(init_bot, apply_bot, key,
                                                      200)
    kernel_mid_mc = monte_carlo.monte_carlo_kernel_fn(init_mid, apply_mid, key,
                                                      200)
    kernel_top_mc = monte_carlo.monte_carlo_kernel_fn(init_top, apply_top, key,
                                                      200)

    K = kernel_fc(X0_1_flat, X0_2_flat)

    K_bot = kernel_bot(X0_1, X0_2)
    K_bot_flat = kernel_bot(X0_1_flat, X0_2_flat)
    self.assertAllClose(K_bot, K)
    self.assertAllClose(K_bot_flat, K)

    def assert_close(a, b):
      self.assertAllClose(a, b, atol=0.05, rtol=0.02)

    K_fc_mc = kernel_fc_mc(X0_1_flat, X0_2_flat, get='nngp')
    K_bot_mc = kernel_bot_mc(X0_1, X0_2, get='nngp')
    K_bot_flat_mc = kernel_bot_mc(X0_1_flat, X0_2_flat, get='nngp')

    assert_close(K_fc_mc, K.nngp)
    assert_close(K_bot_mc, K_bot.nngp)
    assert_close(K_bot_flat_mc, K_bot_flat.nngp)

    K_mid = kernel_mid(X0_1, X0_2)
    K_mid_flat = kernel_mid(X0_1_flat, X0_2_flat)

    K_mid_mc = kernel_mid_mc(X0_1, X0_2, get='nngp')
    K_mid_flat_mc = kernel_mid_mc(X0_1_flat, X0_2_flat, get='nngp')

    assert_close(K_mid_mc, K_mid.nngp)
    assert_close(K_mid_flat, K)
    assert_close(K_mid_flat_mc, K_mid_flat.nngp)

    K_top = kernel_top(X0_1, X0_2).replace(is_gaussian=True,
                                           shape1=K_mid.shape1,
                                           shape2=K_mid.shape2)
    K_top_flat = kernel_top(X0_1_flat, X0_2_flat).replace(is_gaussian=True)

    K_top_mc = kernel_top_mc(X0_1, X0_2, get='nngp')
    K_top_flat_mc = kernel_top_mc(X0_1_flat, X0_2_flat, get='nngp')

    assert_close(K_top_flat, K)
    assert_close(K_top_mc, K_top.nngp)
    assert_close(K_top_flat_mc, K_top_flat.nngp)

    assert_close(K_top, K_mid)


class FanInTest(test_utils.NeuralTangentsTestCase):

  @classmethod
  def _get_phi(cls, i):
    return {
        0: stax.Relu(),
        1: stax.Erf(),
        2: stax.Abs()
    }[i % 3]

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list(
          {
              'testcase_name':
                  ' [{}_axis={}_n_branches={}_{}_{}]'.format(
                      'same_inputs' if same_inputs else 'different_inputs',
                      axis,
                      n_branches,
                      get,
                      branch_in),
              'same_inputs':
                  same_inputs,
              'axis':
                  axis,
              'n_branches':
                  n_branches,
              'get':
                  get,
              'branch_in':
                  branch_in
          }
          for same_inputs in [False, True]
          for axis in [None, 0, 1]
          for n_branches in [1, 2, 3] for get in ['nngp', 'ntk']
          for branch_in in ['dense_before_branch_in',
                            'dense_after_branch_in']))
  def test_fan_in_fc(self, same_inputs, axis, n_branches, get, branch_in):
    if axis in (None, 0) and branch_in == 'dense_after_branch_in':
      raise unittest.SkipTest('`FanInSum` and `FanInConcat(0)` '
                              'require `is_gaussian`.')

    if axis == 1 and branch_in == 'dense_before_branch_in':
      raise unittest.SkipTest('`FanInConcat` on feature axis requires a dense '
                              'layer after concatenation.')

    key = random.PRNGKey(1)
    X0_1 = random.normal(key, (4, 3))
    X0_2 = None if same_inputs else random.normal(key, (8, 3))

    width = 1024
    n_samples = 256

    if xla_bridge.get_backend().platform == 'tpu':
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
        stax.FanInSum() if axis is None else stax.FanInConcat(axis),
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

    kernel_fn_mc = monte_carlo.monte_carlo_kernel_fn(
        init_fn, apply_fn, key, n_samples,
        device_count=0 if axis in (0, -2) else -1)

    exact = kernel_fn(X0_1, X0_2, get=get)
    empirical = kernel_fn_mc(X0_1, X0_2, get=get)
    test_utils.assert_close_matrices(self, empirical, exact, tol)

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list(
          {
              'testcase_name':
                  ' [{}_axis={}_n_branches={}_{}_{}_{}]'.format(
                      'same_inputs' if same_inputs else 'different_inputs',
                      axis,
                      n_branches,
                      get,
                      branch_in,
                      readout),
              'same_inputs':
                  same_inputs,
              'axis':
                  axis,
              'n_branches':
                  n_branches,
              'get':
                  get,
              'branch_in':
                  branch_in,
              'readout':
                  readout
          }
          for same_inputs in [False, True]
          for axis in [None, 0, 1, 2, 3]
          for n_branches in [1, 2, 3] for get in ['nngp', 'ntk']
          for branch_in in ['dense_before_branch_in', 'dense_after_branch_in']
          for readout in ['pool', 'flatten']))
  def test_fan_in_conv(self,
                       same_inputs,
                       axis,
                       n_branches,
                       get,
                       branch_in,
                       readout):
    if xla_bridge.get_backend().platform == 'cpu':
      raise unittest.SkipTest('Not running CNNs on CPU to save time.')

    if axis in (None, 0, 1, 2) and branch_in == 'dense_after_branch_in':
      raise unittest.SkipTest('`FanInSum` and `FanInConcat(0/1/2)` '
                              'require `is_gaussian`.')

    if axis == 3 and branch_in == 'dense_before_branch_in':
      raise unittest.SkipTest('`FanInConcat` on feature axis requires a dense '
                              'layer after concatenation.')

    key = random.PRNGKey(1)
    X0_1 = random.normal(key, (2, 5, 6, 3))
    X0_2 = None if same_inputs else random.normal(key, (3, 5, 6, 3))

    if xla_bridge.get_backend().platform == 'tpu':
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
        stax.FanInSum() if axis is None else stax.FanInConcat(axis),
        stax.Relu(),
        stax.GlobalAvgPool() if readout == 'pool' else stax.Flatten()
    ]
    if branch_in == 'dense_after_branch_in':
      output_layers.insert(1, conv)

    nn = stax.serial(*(input_layers + [stax.parallel(*branches)] +
                       output_layers))

    init_fn, apply_fn, kernel_fn = stax.serial(
        nn, stax.Dense(1 if get == 'ntk' else width, 1.25, 0.5))

    kernel_fn_mc = monte_carlo.monte_carlo_kernel_fn(
        init_fn,
        apply_fn,
        key,
        n_samples,
        device_count=0 if axis in (0, -4) else -1)

    exact = kernel_fn(X0_1, X0_2, get=get)
    empirical = kernel_fn_mc(X0_1, X0_2, get=get)
    test_utils.assert_close_matrices(self, empirical, exact, tol)


class ConvNDTest(test_utils.NeuralTangentsTestCase):

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              ' [{}_n={}_{}_{}_{}_{}_{}_{}]'.format(
                  'same_inputs' if same_inputs else 'different_inputs', n, get,
                  proj,
                  'attn' if use_attn else '',
                  'channels_first' if channels_first else 'channels_last',
                  'dropout' if use_dropout else '',
                  'layernorm' if use_layernorm else ''
              ),
          'same_inputs':
              same_inputs,
          'n':
              n,
          'get':
              get,
          'proj':
              proj,
          'use_attn':
              use_attn,
          'channels_first':
              channels_first,
          'use_dropout':
              use_dropout,
          'use_layernorm':
              use_layernorm
      }
                          for same_inputs in [False]
                          for n in [0, 1, 2, 3]
                          for get in ['nngp', 'ntk']
                          for proj in ['flatten', 'pool']
                          for use_attn in [True]
                          for channels_first in [True, False]
                          for use_dropout in [True]
                          for use_layernorm in [True]))
  def test_conv_nd(self, same_inputs, n, get, proj, use_attn, channels_first,
                   use_dropout, use_layernorm):
    platform = xla_bridge.get_backend().platform
    if platform == 'cpu':
      raise unittest.SkipTest('Skipping CPU CNN tests for speed.')
    elif platform == 'gpu' and n not in (0, 1, 2, 3):
      raise unittest.SkipTest('>=4D CNN does not work on GPU.')
    elif platform == 'tpu' and use_dropout and same_inputs:
      raise unittest.SkipTest('Batched empirical kernel with dropout not '
                              'supported.')

    width = 1024
    n_samples = 512
    tol = 0.03 if platform == 'tpu' else 0.015
    key = random.PRNGKey(1)

    n_max = 5
    spatial_shape = (2, 3, 5, 4, 3)[:n] + (1,) * (n - n_max)
    filter_shape = (1, 2, 3, 1, 1)[:n] + (1,) * (n - n_max)
    strides = (1, 1, 2, 1, 2)[:n] + (1,) * (n - n_max)
    spatial_spec = ''.join(c for c in string.ascii_uppercase
                           if c not in ('N', 'C', 'I', 'O'))[:n]
    filter_spec = spatial_spec + 'IO'

    if channels_first:
      channel_axis = 1
      dimension_numbers = ('NC' + spatial_spec, filter_spec,
                           'NC' + spatial_spec)
      X0_1 = random.normal(key, (2, 3) + spatial_shape)
      X0_2 = None if same_inputs else random.normal(key, (4, 3) + spatial_shape)
    else:
      channel_axis = -1
      dimension_numbers = ('N' + spatial_spec + 'C', filter_spec,
                           'N' + spatial_spec + 'C')
      X0_1 = random.normal(key, (2,) + spatial_shape + (3,))
      X0_2 = None if same_inputs else random.normal(key,
                                                    (4,) + spatial_shape + (3,))

    layernorm_axes = (dimension_numbers[2].index('C'),)
    if 'H' in dimension_numbers[2]:
      layernorm_axes += (dimension_numbers[2].index('H'),)

    if proj == 'pool':
      proj = stax.GlobalAvgPool(channel_axis=channel_axis)
    elif proj == 'flatten':
      proj = stax.Flatten()
    else:
      raise ValueError(proj)

    if use_attn:
      n_heads = int(np.sqrt(width))
      n_chan_val = int(np.round(float(width) / n_heads))
      proj = stax.serial(stax.GlobalSelfAttention(
          n_chan_out=width,
          n_chan_key=width,
          n_chan_val=n_chan_val,
          n_heads=n_heads,
          fixed=True,
          W_key_std=2.,
          W_value_std=1.,
          W_query_std=1.,
          W_out_std=1.0,
          b_std=0.01,
          channel_axis=channel_axis), proj)

    nn = stax.serial(
        stax.GeneralConv(dimension_numbers, width, filter_shape, None, 'SAME'),
        (stax.LayerNorm(layernorm_axes,
                        channel_axis=channel_axis)
         if use_layernorm else stax.Identity()),
        stax.Relu(),
        (stax.Dropout(0.8) if use_dropout else stax.Identity()),
        stax.GeneralConv(dimension_numbers, width, filter_shape, strides,
                         'CIRCULAR'),
        stax.Abs(),
        proj
    )

    if get == 'nngp':
      init_fn, apply_fn, kernel_fn = stax.serial(nn, stax.Dense(width, 2., 0.5))
    elif get == 'ntk':
      init_fn, apply_fn, kernel_fn = stax.serial(nn, stax.Dense(1, 2., 0.5))
    else:
      raise ValueError(get)

    kernel_fn_mc = monte_carlo.monte_carlo_kernel_fn(
        init_fn, apply_fn, key, n_samples)

    exact = kernel_fn(X0_1, X0_2, get=get)
    empirical = kernel_fn_mc(X0_1, X0_2, get=get)
    test_utils.assert_close_matrices(self, empirical, exact, tol)


class DiagonalBatchTest(test_utils.NeuralTangentsTestCase):

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list(
          {
              'testcase_name':
                  ' [{}_{}]'.format(
                      'same_inputs' if same_inputs else 'different_inputs',
                      readout[0].__name__),
              'same_inputs':
                  same_inputs,
              'readout':
                  readout
          }
          for same_inputs in [False]
          for readout in [stax.Flatten(),
                          stax.GlobalAvgPool(),
                          stax.Identity()]))
  def test_diagonal_batch(self, same_inputs, readout):
    key = random.PRNGKey(1)
    x1 = random.normal(key, (2, 5, 6, 3))
    x2 = None if same_inputs else random.normal(key, (3, 5, 6, 3))

    if readout[0].__name__ == 'Identity':
      layers = [stax.Flatten()]
      filter_shape = ()
    else:
      layers = []
      filter_shape = (2, 3)

    layers += [stax.Conv(1, filter_shape, padding='SAME'),
               stax.Relu(),
               stax.Conv(1, filter_shape, padding='SAME'),
               stax.Erf(),
               readout]

    _, _, kernel_fn = stax.serial(*layers)

    K = kernel_fn(x1, x2)
    K_full = kernel_fn(x1, x2, diagonal_batch=False)

    if same_inputs:
      self.assertAllClose(K_full.cov1, K.nngp)
      self.assertAllClose(K_full.cov2, K.cov2)
    else:
      self.assertAllClose(K_full.cov1, kernel_fn(x1, None).nngp)
      self.assertAllClose(K_full.cov2, kernel_fn(x2, None).nngp)

    K_full = K_full.replace(cov1=K.cov1, cov2=K.cov2,
                            diagonal_batch=K.diagonal_batch)
    self.assertAllClose(K_full, K)


@jtu.parameterized.parameters([
    {
        'same_inputs': True
    },
    {
        'same_inputs': False
    },
])
class InputReqTest(test_utils.NeuralTangentsTestCase):

  def test_input_req(self, same_inputs):
    platform = xla_bridge.get_backend().platform
    if platform == 'cpu':
      raise unittest.SkipTest('Skipping CPU CNN tests for speed.')

    key = random.PRNGKey(1)
    x1 = random.normal(key, (2, 7, 8, 4, 3))
    x2 = None if same_inputs else random.normal(key, (4, 7, 8, 4, 3))

    _, _, wrong_conv_fn = stax.serial(
        stax.GeneralConv(dimension_numbers=('NDHWC', 'HDWIO', 'NCDWH'),
                         out_chan=1, filter_shape=(1, 2, 3)),
        stax.Relu(),
        stax.GeneralConv(dimension_numbers=('NHDWC', 'HWDIO', 'NCWHD'),
                         out_chan=1, filter_shape=(1, 2, 3))
    )
    with self.assertRaises(ValueError):
      wrong_conv_fn(x1, x2)

    init_fn, apply_fn, correct_conv_fn = stax.serial(
        stax.GeneralConv(dimension_numbers=('NHWDC', 'DHWIO', 'NCWDH'),
                         out_chan=2048, filter_shape=(1, 2, 3)),
        stax.Relu(),
        stax.GeneralConv(dimension_numbers=('NCHDW', 'WHDIO', 'NCDWH'),
                         out_chan=2048, filter_shape=(1, 2, 3)),
        stax.Flatten(),
        stax.Dense(2048)
    )

    correct_conv_fn_mc = monte_carlo.monte_carlo_kernel_fn(init_fn, apply_fn,
                                                           key, 400)
    K = correct_conv_fn(x1, x2, get='nngp')
    K_mc = correct_conv_fn_mc(x1, x2, get='nngp')
    self.assertAllClose(K, K_mc, atol=0.01, rtol=0.05)

    _, _, wrong_conv_fn = stax.serial(
        stax.GeneralConv(dimension_numbers=('NDHWC', 'HDWIO', 'NCDWH'),
                         out_chan=1, filter_shape=(1, 2, 3)),
        stax.GlobalAvgPool(channel_axis=2)
    )
    with self.assertRaises(ValueError):
      wrong_conv_fn(x1, x2)

    init_fn, apply_fn, correct_conv_fn = stax.serial(
        stax.GeneralConv(dimension_numbers=('NHDWC', 'DHWIO', 'NDWCH'),
                         out_chan=1024, filter_shape=(1, 2, 3)),
        stax.Relu(),
        stax.AvgPool((2, 1, 3), batch_axis=0, channel_axis=-2),
        stax.GeneralConv(dimension_numbers=('NDHCW', 'IHWDO', 'NDCHW'),
                         out_chan=1024, filter_shape=(1, 2, 3)),
        stax.Relu(),
        stax.GlobalAvgPool(channel_axis=2),
        stax.Dense(1024)
    )

    correct_conv_fn_mc = monte_carlo.monte_carlo_kernel_fn(init_fn, apply_fn,
                                                           key, 300)
    K = correct_conv_fn(x1, x2, get='nngp')
    K_mc = correct_conv_fn_mc(x1, x2, get='nngp')
    self.assertAllClose(K, K_mc, atol=0.01, rtol=0.05)

    _, _, wrong_conv_fn = stax.serial(
        stax.Flatten(),
        stax.Dense(1),
        stax.Erf(),
        stax.GeneralConv(dimension_numbers=('CN', 'IO', 'NC'),
                         out_chan=1, filter_shape=(1, 2)),
    )
    with self.assertRaises(ValueError):
      wrong_conv_fn(x1, x2)

    init_fn, apply_fn, correct_conv_fn = stax.serial(
        stax.Flatten(),
        stax.Conv(out_chan=1024, filter_shape=()),
        stax.Relu(),
        stax.Dense(1)
    )

    correct_conv_fn_mc = monte_carlo.monte_carlo_kernel_fn(init_fn, apply_fn,
                                                           key, 200)
    K = correct_conv_fn(x1, x2, get='ntk')
    K_mc = correct_conv_fn_mc(x1, x2, get='ntk')
    self.assertAllClose(K, K_mc, atol=0.01, rtol=0.05)


class MaskingTest(test_utils.NeuralTangentsTestCase):

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list(
          {
              'testcase_name':
                  ' [{}_get={}_axis={}_mask={}_concat={}_p={}]'.format(
                      'same_inputs' if same_inputs else 'different_inputs',
                      get,
                      mask_axis,
                      mask_constant,
                      concat,
                      p,
                  ),
              'same_inputs':
                  same_inputs,
              'get':
                  get,
              'mask_axis':
                  mask_axis,
              'mask_constant':
                  mask_constant,
              'concat':
                  concat,
              'p':
                  p,
          }
          for same_inputs in [False] for get in ['ntk', 'nngp']
          for concat in [None, 0, 1] for p in [0.5]
          for mask_axis in [(), (0,), (1,), (2,), (3,),
                            (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
                            (0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3),
                            (0, 1, 2, 3)]
          for mask_constant in [10.]))
  def test_mask_fc(self, same_inputs, get, concat, p, mask_axis, mask_constant):
    width = 1024
    n_samples = 128
    tol = 0.025
    key = random.PRNGKey(1)

    def apply_mask(x):
      if mask_constant is not None:
        mask_shape = [1 if i in mask_axis else s
                      for i, s in enumerate(x.shape)]
        mask = random.bernoulli(key, p=p, shape=mask_shape)
        x = np.where(mask, mask_constant, x)
      return x

    x1 = random.normal(key, (4, 6, 5, 7))
    x1 = apply_mask(x1)

    if same_inputs:
      x2 = None
    else:
      x2 = random.normal(key, (2, 6, 5, 7))
      x2 = apply_mask(x2)

    nn = stax.serial(
        stax.Flatten(),
        stax.FanOut(3),
        stax.parallel(
            stax.serial(
                stax.Dense(width, 1.5, 0.1),
                stax.Abs(),
                stax.Dense(width, 1.5, 0.1),
            ),
            stax.serial(
                stax.Dense(width, 1.5, 0.1),
                stax.Erf(),
                stax.Dense(width if concat != 1 else 512, 1.5, 0.1),
            ),
            stax.serial(
                stax.Dense(width, 1.5, 0.1),
                stax.ABRelu(-0.2, 0.4),
                stax.Dense(width if concat != 1 else 1024, 3, 0.5),
            )
        ),
        (stax.FanInSum() if concat is None else stax.FanInConcat(concat)),
        stax.Dense(width, 2., 0.01),
        stax.Relu()
    )

    if get == 'nngp':
      init_fn, apply_fn, kernel_fn = stax.serial(nn, stax.Dense(width, 2., 0.5))
    elif get == 'ntk':
      init_fn, apply_fn, kernel_fn = stax.serial(nn, stax.Dense(1, 2., 0.5))
    else:
      raise ValueError(get)

    kernel_fn_mc = monte_carlo.monte_carlo_kernel_fn(
        init_fn, apply_fn, key, n_samples,
        device_count=0 if concat in (0, -2) else -1)

    kernel_fn = jit(kernel_fn, static_argnums=(2,))
    exact = kernel_fn(x1, x2, get, mask_constant)
    empirical = kernel_fn_mc(x1, x2, get=get, mask_constant=mask_constant)
    test_utils.assert_close_matrices(self, empirical, exact, tol)

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
          ' [{}_get={}_axis={}_mask={}_concat={}_{}_p={}_attn={}_n={}]'.format(
              'same_inputs' if same_inputs else 'different_inputs',
              get,
              mask_axis,
              mask_constant,
              concat,
              proj,
              p,
              use_attn,
              n
          ),
          'same_inputs': same_inputs,
          'get': get,
          'mask_axis': mask_axis,
          'mask_constant': mask_constant,
          'concat': concat,
          'proj': proj,
          'p': p,
          'use_attn': use_attn,
          'n': n
      }
                          for proj in ['avg', 'flatten']
                          for use_attn in [True]
                          for same_inputs in [False]
                          for get in ['nngp', 'ntk']
                          for n in [0, 1, 2]
                          for concat in [None] + list(range(n + 1))
                          for mask_constant in [10.]
                          for p in [0.5]
                          for mask_axis in [(),
                                            (0,),
                                            (1,),
                                            (2,),
                                            (3,),
                                            (0, 1),
                                            (0, 2),
                                            (0, 3),
                                            (1, 2),
                                            (1, 3),
                                            (2, 3),
                                            (0, 1, 2),
                                            (0, 1, 3),
                                            (0, 2, 3),
                                            (1, 2, 3),
                                            (0, 1, 2, 3)]
                          ))
  def test_mask_conv(self, same_inputs, get, mask_axis, mask_constant, concat,
                     proj, p, use_attn, n):
    if xla_bridge.get_backend().platform == 'cpu':
      raise unittest.SkipTest('Skipping CNN tests on CPU for speed.')
    elif xla_bridge.get_backend().platform == 'gpu' and n > 3:
      raise unittest.SkipTest('>=4D-CNN is not supported on GPUs.')

    width = 1024
    n_samples = 128
    tol = 0.025
    key = random.PRNGKey(1)

    spatial_shape = (15, 8, 9)[:n]
    filter_shape = (7, 2, 3)[:n]
    strides = (2, 3, 1)[:n]
    spatial_spec = 'HWD'[:n]
    dimension_numbers = ('N' + spatial_spec + 'C',
                         'OI' + spatial_spec,
                         'N' + spatial_spec + 'C')

    def apply_mask(x):
      if mask_constant is not None:
        mask_shape = [1 if i in mask_axis else s
                      for i, s in enumerate(x.shape)]
        mask = random.bernoulli(key, p=p, shape=mask_shape)
        x = np.where(mask, mask_constant, x)
        x = np.sort(x, 1)
      return x

    x1 = random.normal(key, (4,) + spatial_shape + (3,))
    x1 = apply_mask(x1)

    if same_inputs:
      x2 = None
    else:
      x2 = random.normal(key, (2,) + spatial_shape + (3,))
      x2 = apply_mask(x2)

    def get_attn():
      return stax.GlobalSelfAttention(
          n_chan_out=width,
          n_chan_key=width,
          n_chan_val=int(np.round(float(width) / int(np.sqrt(width)))),
          n_heads=int(np.sqrt(width)),
      ) if use_attn else stax.Identity()

    nn = stax.serial(
        stax.FanOut(3),
        stax.parallel(
            stax.serial(
                stax.GeneralConv(
                    dimension_numbers=dimension_numbers,
                    out_chan=width,
                    strides=strides,
                    filter_shape=filter_shape,
                    padding='SAME',
                    W_std=1.1,
                    b_std=0.1),
                stax.LayerNorm(axis=(1, -1)),
                stax.Abs(),
                stax.GeneralConv(
                    dimension_numbers=dimension_numbers,
                    out_chan=width,
                    strides=strides,
                    filter_shape=filter_shape,
                    padding='VALID',
                    W_std=2.,
                    b_std=0.1),
            ),
            stax.serial(
                stax.GeneralConv(
                    dimension_numbers=dimension_numbers,
                    out_chan=width,
                    strides=strides,
                    filter_shape=filter_shape,
                    padding='SAME',
                    W_std=0.1,
                    b_std=0.3),
                stax.Relu(),
                stax.Dropout(0.7),
                stax.GeneralConv(
                    dimension_numbers=dimension_numbers,
                    out_chan=width,
                    strides=strides,
                    filter_shape=filter_shape,
                    padding='VALID',
                    W_std=1.5,
                    b_std=1.),
            ),
            stax.serial(
                get_attn(),
                stax.GeneralConv(
                    dimension_numbers=dimension_numbers,
                    out_chan=width,
                    strides=strides,
                    filter_shape=filter_shape,
                    padding='SAME',
                    W_std=1.,
                    b_std=0.1),
                stax.Erf(),
                stax.Dropout(0.2),
                stax.GeneralConv(
                    dimension_numbers=dimension_numbers,
                    out_chan=width,
                    strides=strides,
                    filter_shape=filter_shape,
                    padding='VALID',
                    W_std=1.,
                    b_std=0.1),
            )
        ),
        (stax.FanInSum() if concat is None else stax.FanInConcat(concat)),

        get_attn(),
        {
            'avg': stax.GlobalAvgPool(),
            'sum': stax.GlobalSumPool(),
            'flatten': stax.Flatten(),
        }[proj],
    )

    if get == 'nngp':
      init_fn, apply_fn, kernel_fn = stax.serial(nn, stax.Dense(width, 1., 0.))
    elif get == 'ntk':
      init_fn, apply_fn, kernel_fn = stax.serial(nn, stax.Dense(1, 1., 0.))
    else:
      raise ValueError(get)

    kernel_fn_mc = monte_carlo.monte_carlo_kernel_fn(
        init_fn, apply_fn, key, n_samples,
        device_count=0 if concat in (0, -n) else -1
    )

    kernel_fn = jit(kernel_fn, static_argnums=(2,))
    exact = kernel_fn(x1, x2, get, mask_constant)
    empirical = kernel_fn_mc(x1, x2, get=get, mask_constant=mask_constant)
    test_utils.assert_close_matrices(self, empirical, exact, tol)


if __name__ == '__main__':
  absltest.main()
