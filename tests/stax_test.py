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


import functools
import itertools
import random as prandom
import string
import time
from typing import Tuple

from absl.testing import absltest
from absl.testing import parameterized
from jax import lax
from jax import ops
from jax import test_util as jtu
from jax import jit, vjp, jvp, jacfwd, jacrev, value_and_grad
from jax.config import config
from jax.lib import xla_bridge
import jax.numpy as np
import jax.random as random
import more_itertools
from neural_tangents import stax
from neural_tangents.utils import monte_carlo, test_utils, utils, batch
import numpy as onp


config.parse_flags_with_absl()
config.update('jax_numpy_rank_promotion', 'raise')


MODELS = [
    'fc',
    'conv'
]

BATCH_SIZE = 4

INPUT_SHAPE = (BATCH_SIZE, 8, 6, 2)

WIDTHS = [2**10]

N_SAMPLES = 100

RTOL = 0.041

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
    stax.Relu(): 'Relu',
}

PROJECTIONS = [
    'FLAT',
    'POOL',
    'ATTN',
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


def _skip_test(msg='Skipping large tests for speed.', platforms=('cpu',)):
  if xla_bridge.get_backend().platform in platforms:
    raise absltest.SkipTest(msg)


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
    return stax.Conv(out_chan=out_chan, filter_shape=filter_shape,
                     strides=strides, padding=padding, W_std=W_std,
                     b_std=b_std, dimension_numbers=dimension_numbers,
                     parameterization=parameterization)

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
    proj_layer = stax.serial(
        stax.GlobalSelfAttention(
            n_chan_out=width,
            n_chan_key=width,
            n_chan_val=n_chan_val,
            n_heads=n_heads,
            linear_scaling=True,
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

  device_count = 0
  # TODO(http://b/160064607): uncomment when the bug is fixed.
  # device_count = -1 if spec.index('N') == 0 else 0

  return stax.serial(block, readout), input_shape, device_count, channel_axis_fc


def _get_net_pool(width, is_ntk, pool_type, padding,
                  filter_shape, strides, normalize_edges):
  W_std, b_std = 2.**0.5, 0.5**0.5
  phi = stax.Relu()
  parameterization = 'ntk'

  fc = functools.partial(
      stax.Dense, W_std=W_std, b_std=b_std, parameterization=parameterization)
  conv = functools.partial(
      stax.Conv,
      filter_shape=(3, 2),
      strides=None,
      padding='SAME',
      W_std=W_std,
      b_std=b_std,
      parameterization=parameterization)

  if pool_type == 'AVG':
    pool_fn = functools.partial(stax.AvgPool, normalize_edges=normalize_edges)
    global_pool_fn = stax.GlobalAvgPool
  elif pool_type == 'SUM':
    pool_fn = stax.SumPool
    global_pool_fn = stax.GlobalSumPool
  else:
    raise ValueError(pool_type)

  pool = pool_fn(filter_shape, strides, padding)

  # TODO(http://b/160064607): set to `-1` when the bug is fixed.
  device_count = 0

  return stax.serial(
      conv(width), phi, pool, conv(width), phi, global_pool_fn(),
      fc(1 if is_ntk else width)), INPUT_SHAPE, device_count, -1


def _mask(x, mask_constant, mask_axis, key, p):
  if mask_constant is not None:
    mask_shape = [1 if i in mask_axis else s
                  for i, s in enumerate(x.shape)]
    mask = random.bernoulli(key, p=p, shape=mask_shape)
    x = np.where(mask, mask_constant, x)
    x = np.sort(x, 1)
  return x


class StaxTest(test_utils.NeuralTangentsTestCase):

  def _skip_test(self, filter_shape, is_conv, is_res, padding, proj_into_2d,
                 strides, use_pooling):
    if is_conv:
      if xla_bridge.get_backend().platform == 'cpu':
        raise absltest.SkipTest('Not running CNN models on CPU to save time.')

      if (is_res and is_conv and ((strides is not None and strides != (1, 1)) or
                                  (padding == 'VALID' and filter_shape !=
                                   (1, 1)))):
        raise absltest.SkipTest('Different paths in a residual models need to '
                                'return outputs of the same shape.')
    elif (filter_shape != FILTER_SHAPES[0] or padding != PADDINGS[0] or
          strides != STRIDES[0] or proj_into_2d != PROJECTIONS[0] or
          use_pooling):
      raise absltest.SkipTest('FC models do not have these parameters.')

  @parameterized.named_parameters(
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
    self._skip_test(filter_shape, is_conv, is_res, padding, proj_into_2d,
                    strides, use_pooling)

    pool_type = 'AVG'
    W_std, b_std = 2.**0.5, 0.5**0.5
    layer_norm = None
    parameterization = 'ntk'
    use_dropout = False

    net = _get_net(W_std, b_std, filter_shape, is_conv, use_pooling, is_res,
                   padding, phi, strides, width, is_ntk, proj_into_2d,
                   pool_type, layer_norm, parameterization, use_dropout)
    self._check_agreement_with_empirical(
        net, same_inputs, use_dropout, is_ntk, RTOL)

  # pylint: disable=g-complex-comprehension
  @parameterized.named_parameters(
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
        raise absltest.SkipTest('Not running CNN models on CPU to save time.')
    elif proj_into_2d != PROJECTIONS[0]:
      raise absltest.SkipTest('FC models do not have these parameters.')

    net = _get_net(W_std, b_std, filter_shape, is_conv, use_pooling, is_res,
                   padding, phi, strides, width, is_ntk, proj_into_2d,
                   pool_type, layer_norm, parameterization, use_dropout)
    self._check_agreement_with_empirical(net, same_inputs, use_dropout, is_ntk)

  @parameterized.named_parameters(
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
        raise absltest.SkipTest('Not running CNN models on CPU to save time.')
    elif proj_into_2d != PROJECTIONS[0] or layer_norm not in ('C', 'NC'):
      raise absltest.SkipTest('FC models do not have these parameters.')

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
                                         0.07)

  @parameterized.named_parameters(
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
      } for width in WIDTHS for same_inputs in [False]
                          for is_ntk in [False, True]
                          for pool_type in POOL_TYPES for padding in PADDINGS
                          for filter_shape in FILTER_SHAPES
                          for strides in STRIDES
                          for normalize_edges in [True, False]))
  def test_pool(self, width, same_inputs, is_ntk, pool_type,
                padding, filter_shape, strides, normalize_edges):
    use_dropout = False
    # Check for duplicate / incorrectly-shaped NN configs / wrong backend.
    if xla_bridge.get_backend().platform == 'cpu':
      raise absltest.SkipTest('Not running CNN models on CPU to save time.')
    if pool_type == 'SUM' and normalize_edges:
      raise absltest.SkipTest('normalize_edges not applicable to SumPool.')

    net = _get_net_pool(width, is_ntk, pool_type,
                        padding, filter_shape, strides, normalize_edges)
    self._check_agreement_with_empirical(net, same_inputs, use_dropout, is_ntk)

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

  @parameterized.named_parameters(
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
    pool_type = 'AVG'
    use_dropout = True
    is_conv = 'conv' in model
    is_res = False
    W_std, b_std = 2.**0.5, 0.5**0.5
    layer_norm = None
    parameterization = 'ntk'
    # Check for duplicate / incorrectly-shaped NN configs / wrong backend.
    self._skip_test(filter_shape, is_conv, is_res, padding, proj_into_2d,
                    strides, use_pooling)

    net = _get_net(W_std, b_std, filter_shape, is_conv, use_pooling, is_res,
                   padding, phi, strides, width, is_ntk, proj_into_2d,
                   pool_type, layer_norm, parameterization, use_dropout)
    self._check_agreement_with_empirical(net, same_inputs, use_dropout, is_ntk)

  @parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              f'_act={act}_kernel={kern}_do_stabilize={do_stabilize}',
          'act': act,
          'kernel': kern,
          'do_stabilize': do_stabilize
      }
                          for act in ['erf', 'relu']
                          for do_stabilize in [True, False]
                          for kern in ['nngp', 'ntk']))
  def test_sparse_inputs(self, act, kernel, do_stabilize):
    if do_stabilize and act != 'relu':
      raise absltest.SkipTest('Stabilization possible only in Relu.')

    key = random.PRNGKey(1)

    input_count = 4
    sparse_count = 2
    input_size = 3
    width = 1024

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
    x_sparse = x_dense.at[:sparse_count, :].set(0.)

    activation = (stax.Relu(do_stabilize=do_stabilize) if act == 'relu'
                  else stax.Erf())

    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(width),
        activation,
        stax.Dense(1 if kernel == 'ntk' else width))
    exact = kernel_fn(x_sparse, None, kernel)

    # TODO(http://b/160064607): set to `device_count=-1` when the bug is fixed.
    mc = monte_carlo.monte_carlo_kernel_fn(init_fn, apply_fn,
                                           random.split(key, 2)[0],
                                           samples,
                                           vmap_axes=0,
                                           device_count=0,
                                           implementation=2)(x_sparse,
                                                             None,
                                                             kernel)
    mc = np.reshape(mc, exact.shape)

    assert not np.any(np.isnan(exact))
    self.assertAllClose(exact[sparse_count:, sparse_count:],
                        mc[sparse_count:, sparse_count:])

  def test_composition_dense(self):
    rng = random.PRNGKey(0)
    x1 = random.normal(rng, (2, 3))
    x2 = random.normal(rng, (4, 3))

    Block = stax.serial(stax.Dense(256), stax.Relu())

    _, _, ker_fn = Block
    _, _, composed_ker_fn = stax.serial(Block, Block)

    ker_out = ker_fn(ker_fn(x1))
    composed_ker_out = composed_ker_fn(x1)
    self.assertAllClose(ker_out, composed_ker_out)

    ker_out = ker_fn(ker_fn(x1, x2))
    composed_ker_out = composed_ker_fn(x1, x2)
    self.assertAllClose(ker_out, composed_ker_out)

  @parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name': '_avg_pool={}_same_inputs={}'.format(avg_pool,
                                                                same_inputs),
          'avg_pool': avg_pool,
          'same_inputs': same_inputs
      } for avg_pool in [True, False] for same_inputs in [True, False]))
  def test_composition_conv(self, avg_pool, same_inputs):
    rng = random.PRNGKey(0)
    x1 = random.normal(rng, (3, 5, 5, 3))
    x2 = None if same_inputs else random.normal(rng, (4, 5, 5, 3))

    Block = stax.serial(stax.Conv(256, (3, 3)), stax.Relu())
    if avg_pool:
      Readout = stax.serial(stax.Conv(256, (3, 3)),
                            stax.GlobalAvgPool(),
                            stax.Dense(10))
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
      rtol=RTOL
  ):
    ((init_fn, apply_fn, kernel_fn),
     input_shape, device_count, channel_axis) = net

    num_samples = N_SAMPLES * 5 if use_dropout else N_SAMPLES
    key = random.PRNGKey(1)
    x1, x2 = _get_inputs(key, same_inputs, input_shape)
    if xla_bridge.get_backend().platform == 'tpu' and use_dropout:
      # including a test case for tpu + dropout with (parallel + batching)
      batch_size = 2
    else:
      batch_size = 0
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
          trace_axes=(channel_axis,), batch_size=batch_size,
          implementation=2
      )
      if same_inputs:
        assert x2 is None
      return kernel_fn_empirical(x1, x2, get)

    if is_ntk:
      exact, shape1, shape2 = kernel_fn(x1, x2, ('ntk', 'shape1', 'shape2'))
      empirical = _get_empirical(num_samples, 'ntk')
    else:
      exact, shape1, shape2 = kernel_fn(x1, x2, ('nngp', 'shape1', 'shape2'))
      empirical = _get_empirical(num_samples, 'nngp')
    test_utils.assert_close_matrices(self, exact, empirical, rtol)
    self.assertEqual(shape1, x1_out_shape)
    self.assertEqual(shape2, x2_out_shape)


class ActivationTest(test_utils.NeuralTangentsTestCase):

  @stax.layer
  def _RBF(self, gamma):
    init_fn = lambda key, input_shape: (input_shape, ())
    def apply_fn(unused_params, unused_xs, **kwargs):
      raise NotImplementedError()
    def kernel_fn(kernels, **kwargs):
      if kernels.ntk is not None:
        raise ValueError('RBF Kernel does not have an associated NTK.')

      if kernels.nngp.ndim > 2:
        raise ValueError(
            ('RBF Kernel is not defined for covariance matrices with dimension'
             ' greater than two.'))

      input_dim = kernels.shape1[1]
      cov1 = kernels.cov1
      cov1 = np.reshape(cov1, (cov1.shape[0], 1))
      cov2 = cov1 if kernels.cov2 is None else kernels.cov2
      cov2 = np.reshape(cov2, (1, cov2.shape[0]))
      nngp = kernels.nngp

      # TODO(schsam): Update cov1 and cov2 if we want to compose this kernel
      # with other kernels.
      return kernels.replace(
          nngp=np.exp(-input_dim * gamma * (cov1 + cov2 - 2 * nngp)))
    return init_fn, apply_fn, kernel_fn

  def _test_activation(self, activation_fn, same_inputs, model, get,
                       rbf_gamma=None):
    platform = xla_bridge.get_backend().platform
    if platform == 'cpu' and 'conv' in model:
      raise absltest.SkipTest('Not running CNNs on CPU to save time.')

    key = random.PRNGKey(1)
    key, split = random.split(key)
    output_dim = 1024 if get == 'nngp' else 1
    b_std = 0.5
    W_std = 2.0
    if activation_fn[2].__name__ == 'Sin':
      W_std = 0.9
    if activation_fn[2].__name__ == 'Rbf':
      W_std = 1.0
      b_std = 0.0

    if model == 'fc':
      rtol = 0.04
      X0_1 = random.normal(key, (4, 2))
      X0_2 = None if same_inputs else random.normal(split, (2, 2))
      affine = stax.Dense(1024, W_std, b_std)
      readout = stax.Dense(output_dim)
      depth = 1

    else:
      rtol = 0.05
      X0_1 = random.normal(key, (2, 4, 4, 3))
      X0_2 = None if same_inputs else random.normal(split, (4, 4, 4, 3))
      affine = stax.Conv(512, (3, 2), W_std=W_std, b_std=b_std, padding='SAME')
      readout = stax.serial(stax.GlobalAvgPool() if 'pool' in model else
                            stax.Flatten(),
                            stax.Dense(output_dim))
      depth = 2

    if platform == 'cpu':
      num_samplings = 200
      rtol *= 2
    else:
      num_samplings = (500 if activation_fn[2].__name__ in ('Sin', 'Rbf')
                       else 300)

    init_fn, apply_fn, kernel_fn = stax.serial(
        *[affine, activation_fn]*depth, readout)
    analytic_kernel = kernel_fn(X0_1, X0_2, get)
    mc_kernel_fn = monte_carlo.monte_carlo_kernel_fn(
        init_fn, apply_fn, split, num_samplings, implementation=2,
        vmap_axes=0
    )
    empirical_kernel = mc_kernel_fn(X0_1, X0_2, get)
    test_utils.assert_close_matrices(self, analytic_kernel,
                                     empirical_kernel, rtol)

    # Check match with explicit RBF
    if rbf_gamma is not None and get == 'nngp' and model == 'fc':
      input_dim = X0_1.shape[1]
      _, _, kernel_fn = self._RBF(rbf_gamma / input_dim)
      direct_rbf_kernel = kernel_fn(X0_1, X0_2, get)
      test_utils.assert_close_matrices(self, analytic_kernel,
                                       direct_rbf_kernel, rtol)

  @parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              '_model={}_phi={}_{}_get={}_abc={}_approximate={}'.format(
                  model,
                  phi_name,
                  'Same_inputs' if same_inputs else 'Different_inputs',
                  get,
                  abc,
                  approximate),
          'model':
              model,
          'phi_name':
              phi_name,
          'same_inputs':
              same_inputs,
          'get': get,
          'abc': abc,
          'approximate': approximate
      }
                          for model in ['fc', 'conv-pool', 'conv-flatten']
                          for phi_name in ['Sin', 'Cos', 'Erf', 'Gelu', 'Sign']
                          for same_inputs in [False]
                          for get in ['nngp', 'ntk']
                          for approximate in [True, False]
                          for abc in itertools.product(
                              [2., 0.3],
                              [1.5, 0.3],
                              [0., -np.pi/4., np.pi/2.])))
  def test_activation(self, same_inputs, model, phi_name, get, abc,
                      approximate):
    platform = xla_bridge.get_backend().platform
    if platform == 'cpu':
      if abc != [0.3, 1.5, -np.pi/4]:
        raise absltest.SkipTest('Skipping Activation test on CPU to save time.')

    if approximate and phi_name != 'Gelu':
      raise absltest.SkipTest(
          f'{phi_name} does not have an `approximate parameter.')

    a, b, c = abc
    if phi_name == 'Sin':
      activation = stax.Sin(a=a, b=b, c=c)
    elif phi_name == 'Erf':
      activation = stax.Erf(a=a, b=b, c=c)
    elif phi_name in ['Gelu', 'Sign']:
      if a != 0.3 or b != 0.3 or c != 0.:
        raise absltest.SkipTest('Skip `Gelu/Sign` test if '
                                ' (a, b, c) != (.3, .3, 0.).')
      activation = stax.Gelu() if phi_name == 'Gelu' else stax.Sign()
    else:
      raise absltest.SkipTest(f'Activation {phi_name} is not implemented.')
    self._test_activation(activation, same_inputs, model, get)

  @parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              '_{}_Rbf_{}_{}_{}'.format(
                  model,
                  'Same_inputs' if same_inputs else 'Different_inputs',
                  get,
                  gamma),
          'model':
              model,
          'same_inputs':
              same_inputs,
          'get': get,
          'gamma': gamma,
      }
                          for model in ['fc', 'conv-pool', 'conv-flatten']
                          for same_inputs in [False, True]
                          for get in ['nngp', 'ntk']
                          for gamma in [1e-6, 1e-4, 1e-2, 1.0, 2.]
                          ))
  def test_rbf(self, same_inputs, model, get, gamma):
    activation = stax.Rbf(gamma)
    self._test_activation(activation, same_inputs, model, get,
                          rbf_gamma=gamma)



class ElementwiseTest(test_utils.NeuralTangentsTestCase):

  @parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              '_{}_{}_n={}_diag_batch={}_spatial={}'.format(
                  phi[0].__name__, same_inputs, n, diagonal_batch,
                  diagonal_spatial),
                              'phi':
                                  phi,
                              'same_inputs':
                                  same_inputs,
                              'n':
                                  n,
                              'diagonal_batch':
                                  diagonal_batch,
                              'diagonal_spatial':
                                  diagonal_spatial
                          } for phi in [
                              stax.Identity(),
                              stax.Erf(),
                              stax.Sin(),
                              stax.Relu(),
                          ]
                          for same_inputs in [False, True, None]
                          for n in [0, 1, 2]
                          for diagonal_batch in [True, False]
                          for diagonal_spatial in [True, False]))
  def test_elementwise(self, same_inputs, phi, n, diagonal_batch,
                       diagonal_spatial):
    fn = lambda x: phi[1]((), x)

    name = phi[0].__name__

    def nngp_fn(cov12, var1, var2):
      if 'Identity' in name:
        res = cov12

      elif 'Erf' in name:
        prod = (1 + 2 * var1) * (1 + 2 * var2)
        res = np.arcsin(2 * cov12 / np.sqrt(prod)) * 2 / np.pi

      elif 'Sin' in name:
        sum_ = (var1 + var2)
        s1 = np.exp((-0.5 * sum_ + cov12))
        s2 = np.exp((-0.5 * sum_ - cov12))
        res = (s1 - s2) / 2

      elif 'Relu' in name:
        prod = var1 * var2
        sqrt = stax._sqrt(prod - cov12 ** 2)
        angles = np.arctan2(sqrt, cov12)
        dot_sigma = (1 - angles / np.pi) / 2
        res = sqrt / (2 * np.pi) + dot_sigma * cov12

      else:
        raise NotImplementedError(name)

      return res

    _, _, kernel_fn = stax.serial(stax.Dense(1), stax.Elementwise(fn, nngp_fn),
                                  stax.Dense(1), stax.Elementwise(fn, nngp_fn))
    _, _, kernel_fn_manual = stax.serial(stax.Dense(1), phi,
                                         stax.Dense(1), phi)

    key = random.PRNGKey(1)
    shape = (4, 3, 2)[:n] + (1,)
    x1 = random.normal(key, (5,) + shape)
    if same_inputs is None:
      x2 = None
    elif same_inputs is True:
      x2 = x1
    else:
      x2 = random.normal(key, (6,) + shape)

    kwargs = dict(diagonal_batch=diagonal_batch,
                  diagonal_spatial=diagonal_spatial)

    k = kernel_fn(x1, x2, **kwargs)
    k_manual = kernel_fn_manual(x1, x2, **kwargs).replace(is_gaussian=False)
    self.assertAllClose(k_manual, k)


class ElementwiseNumericalTest(test_utils.NeuralTangentsTestCase):

  @parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              '_{}_{}_{}_{}'.format(
                  model,
                  phi[0].__name__,
                  'Same_inputs' if same_inputs else 'Different_inputs',
                  get),
          'model': model,
          'phi': phi,
          'same_inputs': same_inputs,
          'get': get,
      }
                          for model in ['fc', 'conv-pool', 'conv-flatten']
                          for phi in [
                              stax.Erf(),
                              stax.Gelu(),
                              stax.Sin(),
                          ]
                          for same_inputs in [False, True]
                          for get in ['nngp', 'ntk']))
  def test_elementwise_numerical(self, same_inputs, model, phi, get):
    platform = xla_bridge.get_backend().platform
    if platform == 'cpu' and 'conv' in model:
      raise absltest.SkipTest('Not running CNNs on CPU to save time.')

    key, split = random.split(random.PRNGKey(1))

    output_dim = 1
    b_std = 0.01
    W_std = 1.0
    rtol = 2e-3
    deg = 25
    if get == 'ntk':
      rtol *= 2
    if xla_bridge.get_backend().platform == 'tpu':
      rtol *= 2

    if model == 'fc':
      X0_1 = random.normal(key, (3, 7))
      X0_2 = None if same_inputs else random.normal(split, (5, 7))
      affine = stax.Dense(1024, W_std, b_std)
      readout = stax.Dense(output_dim)
      depth = 1
    else:
      X0_1 = random.normal(key, (2, 8, 8, 3))
      X0_2 = None if same_inputs else random.normal(split, (3, 8, 8, 3))
      affine = stax.Conv(1024, (3, 2), W_std=W_std, b_std=b_std, padding='SAME')
      readout = stax.serial(stax.GlobalAvgPool() if 'pool' in model else
                            stax.Flatten(),
                            stax.Dense(output_dim))
      depth = 2

    _, _, kernel_fn = stax.serial(*[affine, phi] * depth, readout)
    analytic_kernel = kernel_fn(X0_1, X0_2, get)

    fn = lambda x: phi[1]((), x)
    _, _, kernel_fn = stax.serial(
        *[affine, stax.ElementwiseNumerical(fn, deg=deg)] * depth, readout)
    numerical_activation_kernel = kernel_fn(X0_1, X0_2, get)

    test_utils.assert_close_matrices(self, analytic_kernel,
                                     numerical_activation_kernel, rtol)


@parameterized.parameters([
    {
        'same_inputs': True,
        'do_stabilize': True
    },
    {
        'same_inputs': False,
        'do_stabilize': True
    },
    {
        'same_inputs': True,
        'do_stabilize': False
    },
    {
        'same_inputs': False,
        'do_stabilize': False
    },
])
class ABReluTest(test_utils.NeuralTangentsTestCase):

  def test_ab_relu_relu(self, same_inputs, do_stabilize):
    key = random.PRNGKey(1)
    X0_1 = random.normal(key, (3, 2))
    fc = stax.Dense(5, 1, 0)

    # Test that ABRelu(0, 1) == ReLU
    init_fn, apply_relu, kernel_fn_relu = stax.serial(fc, stax.Relu())
    _, params = init_fn(key, input_shape=X0_1.shape)

    X0_2 = None if same_inputs else random.normal(key, (4, 2))

    for a, b in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
      with self.subTest(a=a, b=b):
        _, apply_ab_relu, kernel_fn_ab_relu = stax.serial(
            fc, stax.ABRelu(a, b, do_stabilize=do_stabilize))

        X1_1_relu = (b - a) * apply_relu(params, X0_1 * (-1 if a != 0 else 1))
        X1_1_ab_relu = apply_ab_relu(params, X0_1)
        self.assertAllClose(X1_1_relu, X1_1_ab_relu)

        kernels_relu = kernel_fn_relu(X0_1, X0_2)
        kernels_ab_relu = kernel_fn_ab_relu(X0_1, X0_2)
        self.assertAllClose(kernels_relu, kernels_ab_relu)

  def test_ab_relu_id(self, same_inputs, do_stabilize):
    key = random.PRNGKey(1)
    X0_1 = random.normal(key, (3, 2))
    fc = stax.Dense(5, 1, 0)

    X0_2 = None if same_inputs else random.normal(key, (4, 2))

    # Test that ABRelu(a, a) == a * Identity
    init_fn, apply_id, kernel_fn_id = stax.serial(fc, stax.Identity())
    _, params = init_fn(key, input_shape=X0_1.shape)

    for a in [-5, -1, -0.5, 0, 0.5, 1, 5]:
      with self.subTest(a=a):
        _, apply_ab_relu, kernel_fn_ab_relu = stax.serial(
            fc, stax.ABRelu(a, a, do_stabilize=do_stabilize))

        X1_1_id = a * apply_id(params, X0_1)
        X1_1_ab_relu = apply_ab_relu(params, X0_1)
        self.assertAllClose(X1_1_id, X1_1_ab_relu)

        kernels_id = kernel_fn_id(X0_1 * a, None if X0_2 is None else a * X0_2)
        kernels_ab_relu = kernel_fn_ab_relu(X0_1, X0_2)
        # Manually correct the value of `is_gaussian` because
        # `ab_relu` (incorrectly) sets `is_gaussian=False` when `a==b`.
        kernels_ab_relu = kernels_ab_relu.replace(is_gaussian=True)
        self.assertAllClose(kernels_id, kernels_ab_relu)

  def test_leaky_relu(self, same_inputs, do_stabilize):
    key = random.PRNGKey(1)
    X0_1 = random.normal(key, (3, 2))
    fc = stax.Dense(5, 1, 0)

    X0_2 = None if same_inputs else random.normal(key, (4, 2))

    # Test that ABRelu(alpha, 1) == LeakyRelu(alpha)
    for a in [-2, -1, 0, 1, 2]:
      with self.subTest(alpha=a):
        init_fn, apply_leaky_relu, kernel_fn_leaky_relu = stax.serial(
            fc, stax.LeakyRelu(a, do_stabilize=do_stabilize))
        _, apply_ab_relu, kernel_fn_ab_relu = stax.serial(fc, stax.ABRelu(a, 1))

        _, params = init_fn(key, input_shape=X0_1.shape)
        X1_1_leaky_relu = apply_leaky_relu(params, X0_1)
        X1_1_ab_relu = apply_ab_relu(params, X0_1)
        self.assertAllClose(X1_1_leaky_relu, X1_1_ab_relu)

        kernels_leaky_relu = kernel_fn_leaky_relu(X0_1, X0_2)
        kernels_ab_relu = kernel_fn_ab_relu(X0_1, X0_2)
        self.assertAllClose(kernels_leaky_relu, kernels_ab_relu)

  def test_abs(self, same_inputs, do_stabilize):
    key = random.PRNGKey(1)
    X0_1 = random.normal(key, (3, 2))
    fc = stax.Dense(5, 1, 0)

    X0_2 = None if same_inputs else random.normal(key, (4, 2))

    # Test that Abs == ABRelu(-1, 1)
    init_fn, apply_leaky_relu, kernel_fn_abs = stax.serial(
        fc, stax.Abs(do_stabilize=do_stabilize))
    _, apply_ab_relu, kernel_fn_ab_relu = stax.serial(fc, stax.ABRelu(-1, 1))

    _, params = init_fn(key, input_shape=X0_1.shape)
    X1_1_abs = apply_leaky_relu(params, X0_1)
    X1_1_ab_relu = apply_ab_relu(params, X0_1)
    self.assertAllClose(X1_1_abs, X1_1_ab_relu)

    kernels_abs = kernel_fn_abs(X0_1, X0_2, ('nngp', 'ntk'))
    kernels_ab_relu = kernel_fn_ab_relu(X0_1, X0_2, ('nngp', 'ntk'))
    self.assertAllClose(kernels_abs, kernels_ab_relu)


@parameterized.parameters([
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
    X0_1 = random.normal(key, (4, 4, 3, 2))
    X0_2 = None if same_inputs else random.normal(key, (2, 4, 3, 2))

    X0_1_flat = np.reshape(X0_1, (X0_1.shape[0], -1))
    X0_2_flat = None if same_inputs else np.reshape(X0_2, (X0_2.shape[0], -1))

    dense = stax.Dense(512, 1.7, 0.1)
    init_fc, apply_fc, kernel_fc = stax.serial(dense,
                                               stax.Erf(),
                                               dense)
    init_top, apply_top, kernel_top = stax.serial(dense,
                                                  stax.Erf(),
                                                  dense,
                                                  stax.Flatten())
    init_mid, apply_mid, kernel_mid = stax.serial(dense,
                                                  stax.Erf(),
                                                  stax.Flatten(),
                                                  dense)
    init_bot, apply_bot, kernel_bot = stax.serial(stax.Flatten(),
                                                  dense,
                                                  stax.Erf(),
                                                  dense)

    kernel_fc = jit(kernel_fc)
    kernel_top = jit(kernel_top)
    kernel_mid = jit(kernel_mid)
    kernel_bot = jit(kernel_bot)

    n = 100

    kernel_fc_mc = monte_carlo.monte_carlo_kernel_fn(init_fc, apply_fc, key,
                                                     n, vmap_axes=0,
                                                     implementation=2)
    kernel_bot_mc = monte_carlo.monte_carlo_kernel_fn(init_bot, apply_bot, key,
                                                      n, vmap_axes=0,
                                                      implementation=2)
    kernel_mid_mc = monte_carlo.monte_carlo_kernel_fn(init_mid, apply_mid, key,
                                                      n, vmap_axes=0,
                                                      implementation=2)
    kernel_top_mc = monte_carlo.monte_carlo_kernel_fn(init_top, apply_top, key,
                                                      n, vmap_axes=0,
                                                      implementation=2)

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

  @parameterized.named_parameters(
      jtu.cases_from_list(
          {
              'testcase_name':
                  ' [{}_axis={}_n_branches={}_{}_{}_{}]'.format(
                      'same_inputs' if same_inputs else 'different_inputs',
                      axis,
                      n_branches,
                      get,
                      branch_in,
                      fan_in_mode),
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
              'fan_in_mode':
                  fan_in_mode,
          }
          for same_inputs in [False]
          for axis in [0, 1]
          for n_branches in [3] for get in ['ntk']
          for branch_in in ['dense_before_branch_in',
                            'dense_after_branch_in']
          for fan_in_mode in ['FanInSum', 'FanInConcat', 'FanInProd']))
  def test_fan_in_fc(self, same_inputs, axis, n_branches, get, branch_in,
                     fan_in_mode):
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

    if xla_bridge.get_backend().platform == 'cpu':
      if n_branches != 2:
        raise absltest.SkipTest(
            'Skipping FanInFC test if n_branches != 2 on CPU.')

    key = random.PRNGKey(1)
    X0_1 = np.cos(random.normal(key, (4, 3)))
    X0_2 = None if same_inputs else random.normal(key, (8, 3))

    width = 1024
    n_samples = 256 * 2

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

    kernel_fn_mc = monte_carlo.monte_carlo_kernel_fn(
        init_fn, apply_fn, key, n_samples,
        device_count=0 if axis in (0, -2) else -1,
        implementation=2,
        vmap_axes=None if axis in (0, -2) else 0,
    )

    exact = kernel_fn(X0_1, X0_2, get=get)
    empirical = kernel_fn_mc(X0_1, X0_2, get=get)
    test_utils.assert_close_matrices(self, empirical, exact, tol)

  @parameterized.named_parameters(
      jtu.cases_from_list(
          {
              'testcase_name':
                  ' [{}_axis={}_n_branches={}_{}_{}_{}_{}]'.format(
                      'same_inputs' if same_inputs else 'different_inputs',
                      axis,
                      n_branches,
                      get,
                      branch_in,
                      readout,
                      fan_in_mode),
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
                  readout,
              'fan_in_mode':
                  fan_in_mode,
          }
          for same_inputs in [False]
          for axis in [0, 1, 2, 3]
          for n_branches in [2] for get in ['ntk']
          for branch_in in ['dense_before_branch_in', 'dense_after_branch_in']
          for readout in ['pool', 'flatten']
          for fan_in_mode in ['FanInSum', 'FanInConcat', 'FanInProd']))
  def test_fan_in_conv(self,
                       same_inputs,
                       axis,
                       n_branches,
                       get,
                       branch_in,
                       readout,
                       fan_in_mode):
    if xla_bridge.get_backend().platform == 'cpu':
      raise absltest.SkipTest('Not running CNNs on CPU to save time.')
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

    kernel_fn_mc = monte_carlo.monte_carlo_kernel_fn(
        init_fn,
        apply_fn,
        key,
        n_samples,
        device_count=0 if axis in (0, -4) else -1,
        implementation=2,
        vmap_axes=None if axis in (0, -4) else 0,
    )

    exact = kernel_fn(X0_1, X0_2, get=get)
    empirical = kernel_fn_mc(X0_1, X0_2, get=get)
    test_utils.assert_close_matrices(self, empirical, exact, tol)


class ConvNDTest(test_utils.NeuralTangentsTestCase):

  @parameterized.named_parameters(
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
                          for n in [0, 1, 2]
                          for get in ['ntk']
                          for proj in ['flatten', 'pool']
                          for use_attn in [True]
                          for channels_first in [True, False]
                          for use_dropout in [True]
                          for use_layernorm in [True]))
  def test_conv_nd(self, same_inputs, n, get, proj, use_attn, channels_first,
                   use_dropout, use_layernorm):
    platform = xla_bridge.get_backend().platform
    if platform == 'cpu':
      raise absltest.SkipTest('Skipping CPU CNN tests for speed.')
    elif platform == 'gpu' and n not in (0, 1, 2, 3):
      raise absltest.SkipTest('>=4D CNN does not work on GPU.')
    elif platform == 'tpu' and use_dropout and same_inputs:
      raise absltest.SkipTest('Batched empirical kernel with dropout not '
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
          linear_scaling=True,
          W_key_std=2.,
          W_value_std=1.,
          W_query_std=1.,
          W_out_std=1.0,
          b_std=0.1,
          channel_axis=channel_axis), proj)

    nn = stax.serial(
        stax.Conv(width, filter_shape, None, 'SAME',
                  dimension_numbers=dimension_numbers),
        (stax.LayerNorm(layernorm_axes,
                        channel_axis=channel_axis)
         if use_layernorm else stax.Identity()),
        stax.Relu(),
        (stax.Dropout(0.8) if use_dropout else stax.Identity()),
        stax.Conv(width, filter_shape, strides, 'CIRCULAR',
                  dimension_numbers=dimension_numbers),
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
        init_fn, apply_fn, key, n_samples, implementation=2, vmap_axes=0)

    exact = kernel_fn(X0_1, X0_2, get=get)
    empirical = kernel_fn_mc(X0_1, X0_2, get=get)
    test_utils.assert_close_matrices(self, empirical, exact, tol)


@parameterized.named_parameters(
    jtu.cases_from_list(
        {
            'testcase_name':
                ' [{}_out={}_in={}]'.format(
                    'same_inputs' if same_inputs else 'different_inputs',
                    readout[0].__name__,
                    readin[0].__name__
                ),
            'same_inputs':
                same_inputs,
            'readout':
                readout,
            'readin':
                readin
        }
        for same_inputs in [False, True]
        for readout in [stax.Flatten(),
                        stax.GlobalAvgPool(),
                        stax.Identity()]
        for readin in [stax.Flatten(),
                       stax.GlobalAvgPool(),
                       stax.Identity()]))
class DiagonalTest(test_utils.NeuralTangentsTestCase):

  def _get_kernel_fn(self, same_inputs, readin, readout):
    key = random.PRNGKey(1)
    x1 = random.normal(key, (2, 5, 6, 3))
    x2 = None if same_inputs else random.normal(key, (3, 5, 6, 3))
    layers = [readin]
    filter_shape = (2, 3) if readin[0].__name__ == 'Identity' else ()
    layers += [stax.Conv(1, filter_shape, padding='SAME'),
               stax.Relu(),
               stax.Conv(1, filter_shape, padding='SAME'),
               stax.Erf(),
               readout]
    _, _, kernel_fn = stax.serial(*layers)
    return kernel_fn, x1, x2

  def test_diagonal_batch(self, same_inputs, readin, readout):
    kernel_fn, x1, x2 = self._get_kernel_fn(same_inputs, readin, readout)
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

  def test_diagonal_spatial(self, same_inputs, readin, readout):
    kernel_fn, x1, x2 = self._get_kernel_fn(same_inputs, readin, readout)
    K = kernel_fn(x1, x2)
    K_full = kernel_fn(x1, x2, diagonal_spatial=False)
    batch_shape = x1.shape[0], (x1 if x2 is None else x2).shape[0]
    names = readout[0].__name__, readin[0].__name__

    if 'GlobalAvgPool' in names:
      if (readout[0].__name__ == 'GlobalAvgPool' and
          readin[0].__name__ == 'Identity'):
        self.assertRaises(ValueError, kernel_fn, x1, x2, diagonal_spatial=True)
      self.assertEqual(K_full.nngp.shape, batch_shape)
      self.assertAllClose(K_full, K)

    else:
      K_diag = kernel_fn(x1, x2, diagonal_spatial=True)
      if 'Flatten' in names:
        self.assertEqual(K_diag.nngp.shape, batch_shape)
        self.assertAllClose(K_diag, K)
        self.assertAllClose(K_diag, K_full)

      else:
        self.assertEqual(K_diag.nngp.shape, batch_shape + x1.shape[1:-1])
        self.assertAllClose(K_full, K)
        self.assertAllClose(K_diag.nngp, np.einsum('...iijj->...ij', K.nngp))


class DiagonalClassTest(test_utils.NeuralTangentsTestCase):

  def test_diagonal_compose_is_associative(self):
    for inp_a, inp_b, inp_c in itertools.product(stax._Bool, repeat=3):
      for out_a, out_b, out_c in itertools.product(stax._Bool, repeat=3):
        a = stax._Diagonal(inp_a, out_a)
        b = stax._Diagonal(inp_b, out_b)
        c = stax._Diagonal(inp_c, out_c)
        with self.subTest(a=a, b=b, c=c):
          ab_c = (a >> b) >> c
          a_bc = a >> (b >> c)
          self.assertEqual(ab_c, a_bc)

          _ab_c = c << (b << a)
          _a_bc = (c << b) << a
          self.assertEqual(_ab_c, _a_bc)

          self.assertEqual(ab_c, _ab_c)


@parameterized.parameters([
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
      raise absltest.SkipTest('Skipping CPU CNN tests for speed.')

    key = random.PRNGKey(1)
    x1 = random.normal(key, (2, 7, 8, 4, 3))
    x2 = None if same_inputs else random.normal(key, (4, 7, 8, 4, 3))

    _, _, wrong_conv_fn = stax.serial(
        stax.Conv(out_chan=1, filter_shape=(1, 2, 3),
                  dimension_numbers=('NDHWC', 'HDWIO', 'NCDWH')),
        stax.Relu(),
        stax.Conv(out_chan=1, filter_shape=(1, 2, 3),
                  dimension_numbers=('NHDWC', 'HWDIO', 'NCWHD'))
    )
    with self.assertRaises(ValueError):
      wrong_conv_fn(x1, x2)

    init_fn, apply_fn, correct_conv_fn = stax.serial(
        stax.Conv(out_chan=1024, filter_shape=(1, 2, 3),
                  dimension_numbers=('NHWDC', 'DHWIO', 'NCWDH')),
        stax.Relu(),
        stax.Conv(out_chan=1024, filter_shape=(1, 2, 3),
                  dimension_numbers=('NCHDW', 'WHDIO', 'NCDWH')),
        stax.Flatten(),
        stax.Dense(1024)
    )

    correct_conv_fn_mc = monte_carlo.monte_carlo_kernel_fn(init_fn, apply_fn,
                                                           key, 400,
                                                           implementation=2,
                                                           vmap_axes=0)
    K = correct_conv_fn(x1, x2, get='nngp')
    K_mc = correct_conv_fn_mc(x1, x2, get='nngp')
    self.assertAllClose(K, K_mc, atol=0.01, rtol=0.05)

    _, _, wrong_conv_fn = stax.serial(
        stax.Conv(out_chan=1, filter_shape=(1, 2, 3),
                  dimension_numbers=('NDHWC', 'HDWIO', 'NCDWH')),
        stax.GlobalAvgPool(channel_axis=2)
    )
    with self.assertRaises(ValueError):
      wrong_conv_fn(x1, x2)

    init_fn, apply_fn, correct_conv_fn = stax.serial(
        stax.Conv(out_chan=1024, filter_shape=(1, 2, 3),
                  dimension_numbers=('NHDWC', 'DHWIO', 'NDWCH')),
        stax.Relu(),
        stax.AvgPool((2, 1, 3), batch_axis=0, channel_axis=-2),
        stax.Conv(out_chan=1024, filter_shape=(1, 2, 3),
                  dimension_numbers=('NDHCW', 'IHWDO', 'NDCHW')),
        stax.Relu(),
        stax.GlobalAvgPool(channel_axis=2),
        stax.Dense(1024)
    )

    correct_conv_fn_mc = monte_carlo.monte_carlo_kernel_fn(init_fn, apply_fn,
                                                           key, 300,
                                                           implementation=2,
                                                           vmap_axes=0)
    K = correct_conv_fn(x1, x2, get='nngp')
    K_mc = correct_conv_fn_mc(x1, x2, get='nngp')
    self.assertAllClose(K, K_mc, atol=0.01, rtol=0.05)

    _, _, wrong_conv_fn = stax.serial(
        stax.Flatten(),
        stax.Dense(1),
        stax.Erf(),
        stax.Conv(out_chan=1, filter_shape=(1, 2),
                  dimension_numbers=('CN', 'IO', 'NC')),
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
                                                           key, 200,
                                                           implementation=2,
                                                           vmap_axes=0)
    K = correct_conv_fn(x1, x2, get='ntk')
    K_mc = correct_conv_fn_mc(x1, x2, get='ntk')
    self.assertAllClose(K, K_mc, atol=0.01, rtol=0.05)


class MaskingTest(test_utils.NeuralTangentsTestCase):

  @parameterized.named_parameters(
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
          for same_inputs in [False] for get in ['ntk']
          for concat in [None, 0, 1] for p in [0.5]
          for mask_axis in [(),
                            (0,),
                            (1, 3)]
          for mask_constant in [10.]))
  def test_mask_fc(self, same_inputs, get, concat, p, mask_axis, mask_constant):
    width = 512
    n_samples = 128
    tol = 0.04
    key = random.PRNGKey(1)

    x1 = random.normal(key, (4, 6, 5, 7))
    x1 = _mask(x1, mask_constant, mask_axis, key, p)

    if same_inputs:
      x2 = None
    else:
      x2 = random.normal(key, (2, 6, 5, 7))
      x2 = _mask(x2, mask_constant, mask_axis, key, p)

    nn = stax.serial(
        stax.Flatten(),
        stax.FanOut(3),
        stax.parallel(
            stax.serial(
                stax.Dense(width, 1.5, 0.1),
                stax.Abs(),
                stax.DotGeneral(lhs=2.),
                stax.Dense(width, 1.5, 0.1),
            ),
            stax.serial(
                stax.Dense(width, 1.5, 0.1),
                stax.DotGeneral(rhs=3.),
                stax.Erf(),
                stax.Dense(width if concat != 1 else 512, 1.5, 0.1),
            ),
            stax.serial(
                stax.DotGeneral(rhs=0.5),
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
        device_count=0 if concat in (0, -2) else -1,
        implementation=2,
        vmap_axes=None if concat in (0, -2) else 0,
    )

    kernel_fn = jit(kernel_fn, static_argnums=(2,))
    exact = kernel_fn(x1, x2, get, mask_constant=mask_constant)
    empirical = kernel_fn_mc(x1, x2, get=get, mask_constant=mask_constant)
    test_utils.assert_close_matrices(self, empirical, exact, tol)

  @parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
          ' [{}_get={}_axis={}_mask={}_concat={}_{}_p={}_n={}_{}]'
          ''.format(
              'same_inputs' if same_inputs else 'different_inputs',
              get,
              mask_axis,
              mask_constant,
              concat,
              proj,
              p,
              n,
              'transpose' if transpose else ''
          ),
          'same_inputs': same_inputs,
          'get': get,
          'mask_axis': mask_axis,
          'mask_constant': mask_constant,
          'concat': concat,
          'proj': proj,
          'p': p,
          'n': n,
          'transpose': transpose
      }
                          for proj in ['flatten', 'avg']
                          for same_inputs in [False]
                          for get in ['ntk']
                          for n in [0, 1, 2]
                          for concat in [None] + list(range(n + 1))
                          for mask_constant in [10.]
                          for p in [0.5]
                          for transpose in [True, False]
                          for mask_axis in [(),
                                            (0,),
                                            (0, 1, 2, 3)
                                            ]
                          ))
  def test_mask_conv(self, same_inputs, get, mask_axis, mask_constant, concat,
                     proj, p, n, transpose):
    if xla_bridge.get_backend().platform == 'cpu':
      raise absltest.SkipTest('Skipping CNN tests on CPU for speed.')
    elif xla_bridge.get_backend().platform == 'gpu' and n > 3:
      raise absltest.SkipTest('>=4D-CNN is not supported on GPUs.')

    width = 256
    n_samples = 256
    tol = 0.03
    key = random.PRNGKey(1)

    spatial_shape = ((1, 2, 3, 2, 1) if transpose else (15, 8, 9))[:n]
    filter_shape = ((2, 3, 1, 2, 1) if transpose else (7, 2, 3))[:n]
    strides = (2, 1, 3, 2, 3)[:n]
    spatial_spec = 'HWDZX'[:n]
    dimension_numbers = ('N' + spatial_spec + 'C',
                         'OI' + spatial_spec,
                         'N' + spatial_spec + 'C')

    x1 = np.cos(random.normal(key, (2,) + spatial_shape + (2,)))
    x1 = _mask(x1, mask_constant, mask_axis, key, p)

    if same_inputs:
      x2 = None
    else:
      x2 = np.cos(random.normal(key, (4,) + spatial_shape + (2,)))
      x2 = _mask(x2, mask_constant, mask_axis, key, p)

    def get_attn():
      return stax.GlobalSelfAttention(
          n_chan_out=width,
          n_chan_key=width,
          n_chan_val=int(np.round(float(width) / int(np.sqrt(width)))),
          n_heads=int(np.sqrt(width)),
      ) if proj == 'avg' else stax.Identity()

    conv = stax.ConvTranspose if transpose else stax.Conv

    nn = stax.serial(
        stax.FanOut(3),
        stax.parallel(
            stax.serial(
                conv(
                    dimension_numbers=dimension_numbers,
                    out_chan=width,
                    strides=strides,
                    filter_shape=filter_shape,
                    padding='CIRCULAR',
                    W_std=1.5,
                    b_std=0.2),
                stax.LayerNorm(axis=(1, -1)),
                stax.Abs(),
                stax.DotGeneral(rhs=1.5),
                conv(
                    dimension_numbers=dimension_numbers,
                    out_chan=width,
                    strides=strides,
                    filter_shape=filter_shape,
                    padding='VALID',
                    W_std=2.,
                    b_std=0.1),
            ),
            stax.serial(
                conv(
                    dimension_numbers=dimension_numbers,
                    out_chan=width,
                    strides=strides,
                    filter_shape=filter_shape,
                    padding='SAME',
                    W_std=0.1,
                    b_std=0.3),
                stax.Relu(),
                stax.Dropout(0.7),
                conv(
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
                conv(
                    dimension_numbers=dimension_numbers,
                    out_chan=width,
                    strides=strides,
                    filter_shape=filter_shape,
                    padding='CIRCULAR',
                    W_std=1.,
                    b_std=0.1),
                stax.Erf(),
                stax.Dropout(0.2),
                stax.DotGeneral(rhs=0.7),
                conv(
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
        device_count=0 if concat in (0, -n) else -1,
        implementation=2,
        vmap_axes=None if concat in (0, -n) else 0,
    )

    kernel_fn = jit(kernel_fn, static_argnums=(2,))
    exact = kernel_fn(x1, x2, get, mask_constant=mask_constant)
    empirical = kernel_fn_mc(x1, x2, get=get, mask_constant=mask_constant)
    test_utils.assert_close_matrices(self, empirical, exact, tol)


class ParallelInOutTest(test_utils.NeuralTangentsTestCase):

  @parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              f'_same_inputs={same_inputs}_kernel_type={kernel_type}',
          'same_inputs': same_inputs,
          'kernel_type': kernel_type
      }
                          for same_inputs in [True, False]
                          for kernel_type in ['ntk']))
  def test_parallel_in(self, same_inputs, kernel_type):
    platform = xla_bridge.get_backend().platform
    rtol = RTOL if platform != 'tpu' else 0.05

    rng = random.PRNGKey(0)
    input_key1, input_key2, mc_key = random.split(rng, 3)

    x1_1, x2_1 = _get_inputs(input_key1, same_inputs, (BATCH_SIZE, 2))
    x1_2, x2_2 = _get_inputs(input_key2, same_inputs, (BATCH_SIZE, 3))

    x1 = (x1_1, x1_2)
    x2 = (x2_1, x2_2)

    N = 2 ** 7

    def net(logits):
      return stax.serial(
          stax.parallel(stax.Dense(N), stax.Dense(N)),
          stax.serial(stax.FanInSum(), stax.Dense(logits)))

    init_fn, apply_fn, kernel_fn = net(N if kernel_type == 'nngp' else 1)

    kernel_fn_empirical = monte_carlo.monte_carlo_kernel_fn(
        init_fn, apply_fn, mc_key, N_SAMPLES, trace_axes=(-1,),
        implementation=2,
        vmap_axes=((0, 0), 0, {})
    )
    test_utils.assert_close_matrices(self,
                                     kernel_fn(x1, x2, kernel_type),
                                     kernel_fn_empirical(x1, x2, kernel_type),
                                     rtol)

  @parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              f'_same_inputs={same_inputs}_kernel_type={kernel_type}',
          'same_inputs': same_inputs,
          'kernel_type': kernel_type
      } for same_inputs in [True, False] for kernel_type in ['ntk']))
  def test_parallel_out(self, same_inputs, kernel_type):
    platform = xla_bridge.get_backend().platform
    rtol = RTOL if platform != 'tpu' else 0.05

    rng = random.PRNGKey(0)
    input_key1, mc_key = random.split(rng, 2)

    x1, x2 = _get_inputs(input_key1, same_inputs, (BATCH_SIZE, 1))

    N = 2 ** 10

    def net(logits):
      return stax.serial(
          stax.Dense(N),
          stax.FanOut(2),
          stax.parallel(stax.Dense(logits), stax.Dense(logits)))

    init_fn, apply_fn, kernel_fn = net(N if kernel_type == 'nngp' else 1)

    kernel_fn_empirical = monte_carlo.monte_carlo_kernel_fn(
        init_fn, apply_fn, mc_key, N_SAMPLES, trace_axes=(-1,),
        implementation=2,
        vmap_axes=(0, [0, 0], {}))

    test_utils.assert_close_matrices(self,
                                     kernel_fn(x1, x2, kernel_type),
                                     kernel_fn_empirical(x1, x2, kernel_type),
                                     rtol)

  @parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              f'_same_inputs={same_inputs}_kernel_type={kernel_type}',
          'same_inputs': same_inputs,
          'kernel_type': kernel_type,
      } for same_inputs in [True, False] for kernel_type in ['ntk']))
  def test_parallel_in_out(self, same_inputs, kernel_type):
    platform = xla_bridge.get_backend().platform
    rtol = RTOL if platform != 'tpu' else 0.05

    rng = random.PRNGKey(0)
    input_key1, input_key2, mc_key = random.split(rng, 3)

    x1_1, x2_1 = _get_inputs(input_key1, same_inputs, (BATCH_SIZE, 1))
    x1_2, x2_2 = _get_inputs(input_key2, same_inputs, (BATCH_SIZE, 2))

    x1 = (x1_1, x1_2)
    x2 = (x2_1, x2_2)

    N_in = 2 ** 10
    N_out = N_in if kernel_type == 'nngp' else 1

    readin = stax.serial(stax.parallel(stax.Dense(N_in), stax.Dense(N_in)),
                         stax.FanInSum())
    readout = stax.serial(stax.FanOut(3),
                          stax.parallel(stax.Dense(N_out),
                                        stax.Dense(N_out + 1),
                                        stax.Dense(N_out + 2)))
    init_fn, apply_fn, _ = stax.serial(readin, readout)

    K_readin_fn = jit(readin[2])
    K_readout_fn = jit(functools.partial(readout[2], get=kernel_type))

    kernel_fn_empirical = monte_carlo.monte_carlo_kernel_fn(
        init_fn, apply_fn, mc_key, N_SAMPLES, trace_axes=(-1,),
        implementation=2,
        vmap_axes=((0, 0), [0, 0, 0], {})
    )

    test_utils.assert_close_matrices(
        self,
        K_readout_fn(K_readin_fn(x1, x2)),
        kernel_fn_empirical(x1, x2, get=kernel_type),
        rtol)

    # Check Both (here we just want to make sure we _can_ compute the output).
    K_readin_fn = jit(readin[2])
    K_readout_fn = jit(functools.partial(readout[2], get=('nngp', 'ntk')))

    K_readout_fn(K_readin_fn(x1, x2))

  @parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              f'_same_inputs={same_inputs}_kernel_type={kernel_type}',
          'same_inputs': same_inputs,
          'kernel_type': kernel_type,
      } for same_inputs in [True, False] for kernel_type in ['ntk']))
  def test_nested_parallel(self, same_inputs, kernel_type):
    platform = xla_bridge.get_backend().platform
    rtol = RTOL if platform != 'tpu' else 0.05

    rng = random.PRNGKey(0)
    (input_key1,
     input_key2,
     input_key3,
     input_key4,
     mask_key,
     mc_key) = random.split(rng, 6)

    x1_1, x2_1 = _get_inputs(input_key1, same_inputs, (BATCH_SIZE, 5))
    x1_2, x2_2 = _get_inputs(input_key2, same_inputs, (BATCH_SIZE, 2, 2, 2))
    x1_3, x2_3 = _get_inputs(input_key3, same_inputs, (BATCH_SIZE, 2, 2, 3))
    x1_4, x2_4 = _get_inputs(input_key4, same_inputs, (BATCH_SIZE, 3, 4))

    m1_key, m2_key, m3_key, m4_key = random.split(mask_key, 4)

    x1_1 = _mask(x1_1, mask_constant=-1, mask_axis=(1,), key=m1_key, p=0.5)
    x1_2 = _mask(x1_2, mask_constant=-1, mask_axis=(2, 3,), key=m2_key, p=0.5)
    if not same_inputs:
      x2_3 = _mask(x2_3, mask_constant=-1, mask_axis=(1, 3,), key=m3_key, p=0.5)
      x2_4 = _mask(x2_4, mask_constant=-1, mask_axis=(2,), key=m4_key, p=0.5)

    x1 = (((x1_1, x1_2), x1_3), x1_4)
    x2 = (((x2_1, x2_2), x2_3), x2_4) if not same_inputs else None

    N_in = 2 ** 7

    # We only include dropout on non-TPU backends, because it takes large N to
    # converge on TPU.
    dropout_or_id = stax.Dropout(0.9) if platform != 'tpu' else stax.Identity()

    init_fn, apply_fn, kernel_fn = stax.parallel(
        stax.parallel(
            stax.parallel(stax.Dense(N_in),
                          stax.serial(stax.Conv(N_in + 1, (2, 2)),
                                      stax.Flatten())),
            stax.serial(stax.Conv(N_in + 2, (2, 2)),
                        dropout_or_id,
                        stax.GlobalAvgPool())),
        stax.Conv(N_in + 3, (2,)))

    kernel_fn_empirical = monte_carlo.monte_carlo_kernel_fn(
        init_fn, apply_fn, mc_key, N_SAMPLES, implementation=2,
        vmap_axes=(((((0, 0), 0), 0), (((0, 0), 0), 0), {})
                   if platform == 'tpu' else None)
    )

    test_utils.assert_close_matrices(
        self,
        kernel_fn(x1, x2, get=kernel_type, mask_constant=-1),
        kernel_fn_empirical(x1, x2, get=kernel_type, mask_constant=-1),
        rtol)


class AttentionTest(test_utils.NeuralTangentsTestCase):

  @parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              f'[same_inputs={same_inputs}_'
              f'get={get}_'
              f'axis={mask_axis}'
              f'_mask={mask_constant}_'
              f'p={p}_'
              f'linear_scaling={linear_scaling}_'
              f'n={n}_pos_emb_type={pos_emb_type}_'
              f'n_chan_pos_emb={n_chan_pos_emb}'
              f'_pos_emb_decay_fn={pos_emb_decay_fn}_'
              f'val_pos_emb={val_pos_emb}_'
              f'W_pos_emb_std={W_pos_emb_std}]',
          'same_inputs': same_inputs,
          'get': get,
          'n': n,
          'linear_scaling': linear_scaling,
          'mask_constant': mask_constant,
          'p': p,
          'mask_axis': mask_axis,
          'pos_emb_type': pos_emb_type,
          'n_chan_pos_emb': n_chan_pos_emb,
          'pos_emb_decay_fn': pos_emb_decay_fn,
          'val_pos_emb': val_pos_emb,
          'W_pos_emb_std': W_pos_emb_std
      }
                          for same_inputs in [
                              False
                          ]
                          for get in [
                              'ntk'
                          ]
                          for n in [
                              2,
                          ]
                          for linear_scaling in [
                              True,
                              False
                          ]
                          for mask_constant in [
                              10.
                          ]
                          for p in [0.5]
                          for mask_axis in [(-1,)]
                          for pos_emb_type in [
                              'CONCAT',
                              'SUM',
                              'NONE'
                          ]
                          for n_chan_pos_emb in ([None]
                                                 if pos_emb_type != 'CONCAT'
                                                 else [None, 512])
                          for pos_emb_decay_fn in [
                              None,
                              'linear'
                          ]
                          for val_pos_emb in ([
                              True,
                              False
                          ] if pos_emb_type != 'NONE' else [True])
                          for W_pos_emb_std in ([
                              2,
                          ] if pos_emb_type != 'NONE' else [0.])
                          ))
  def test_attention(
      self,
      same_inputs,
      get,
      n,
      linear_scaling,
      mask_constant,
      p,
      mask_axis,
      pos_emb_type,
      n_chan_pos_emb,
      pos_emb_decay_fn,
      val_pos_emb,
      W_pos_emb_std):
    if xla_bridge.get_backend().platform == 'cpu':
      raise absltest.SkipTest('Skipping attention tests on CPU for speed.')

    width = 1024
    n_samples = 1024
    tol = 0.05
    key = random.PRNGKey(1)
    n_chan_in = 2
    spatial_shape = (2, 3, 4, 3, 2, 1)[:n]
    mask_axis = [i % (n + 2) for i in mask_axis]

    def get_x0(batch_size):
      x0 = random.normal(key, (batch_size,) + spatial_shape + (n_chan_in,))
      x0 = _mask(x0, mask_constant, mask_axis, key, p)
      return x0

    X0_1 = get_x0(2)
    X0_2 = None if same_inputs else get_x0(4)

    pos_emb_fns = {
        None: None,
        'one_hot': lambda x: x == 0,
        'linear': lambda x: 1 / (1 + 4 * x)
    }

    def get_attn():
      return stax.GlobalSelfAttention(
          linear_scaling=linear_scaling,
          n_chan_out=width,
          n_chan_key=width,
          n_chan_val=int(np.round(float(width) / int(np.sqrt(width)))),
          n_heads=int(np.sqrt(width)),
          n_chan_pos_emb=n_chan_pos_emb,
          attention_mechanism='SOFTMAX' if linear_scaling else 'IDENTITY',
          pos_emb_type=pos_emb_type,
          W_pos_emb_std=W_pos_emb_std,
          pos_emb_decay_fn=pos_emb_fns[pos_emb_decay_fn],
          val_pos_emb=val_pos_emb,
          W_key_std=0.9,
          W_out_std=1.2,
          W_query_std=0.7,
          W_value_std=1.5,
          b_std=0.9
      )

    nn = stax.serial(
        stax.Conv(width, (1,) * n, padding='SAME'),
        get_attn(),
        stax.Relu(),
        stax.GlobalAvgPool()
    )

    if get == 'nngp':
      init_fn, apply_fn, kernel_fn = nn
    elif get == 'ntk':
      init_fn, apply_fn, kernel_fn = stax.serial(nn, stax.Dense(1, 1., 0.))
    else:
      raise ValueError(get)

    kernel_fn_mc = monte_carlo.monte_carlo_kernel_fn(
        init_fn, apply_fn, key, n_samples,
        device_count=-1,
        implementation=2,
        vmap_axes=0
    )

    kernel_fn = jit(kernel_fn, static_argnums=(2,))
    exact = kernel_fn(X0_1, X0_2, get, mask_constant=mask_constant)

    empirical = kernel_fn_mc(X0_1, X0_2, get=get, mask_constant=mask_constant)
    test_utils.assert_close_matrices(self, empirical, exact, tol)


class AggregateTest(test_utils.NeuralTangentsTestCase):

  @parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              f'{get}-{name}-same_inp={same_input}-{act_name}'
              f'-mask_constant={mask_constant}'
              f'-shape={shape}-batch_axis={batch_axis}'
              f'-channel_axis={channel_axis}-agg_axes={agg_axes}'
              f'-batch={do_batch}-to_dense={to_dense}'
              f'-implementation={implementation}',
          'get': get,
          'readout': readout,
          'same_input': same_input,
          'activation': activation,
          'mask_constant': mask_constant,
          'shape': shape,
          'batch_axis': batch_axis,
          'channel_axis': channel_axis,
          'agg_axes': agg_axes,
          'do_batch': do_batch,
          'implementation': implementation,
          'to_dense': to_dense
      }
                          for get in [
                              'ntk',
                          ]
                          for same_input in [
                              False,
                              True
                          ]
                          for act_name, activation in [
                              ('Relu', stax.Relu()),
                          ]
                          for mask_constant in [
                              10.
                          ]
                          for shape in [
                              (4,),
                              (3, 2),
                          ]
                          for batch_axis in range(len(shape) + 2)
                          for channel_axis in
                          [
                              c for c in range(len(shape) + 2)
                              if c != batch_axis
                          ]
                          for agg_axes in [None] +
                          list(more_itertools.powerset(
                              [p for p in range(len(shape) + 2)
                               if p not in (batch_axis, channel_axis)]))
                          for do_batch in ([
                              True
                          ] if batch_axis == 0 else [False])
                          for implementation in ['DENSE', 'SPARSE']
                          for to_dense in [
                              'identity',
                              'sparse_to_dense',
                          ]
                          for name, readout in [
                              ('Pooling',
                               stax.GlobalAvgPool(batch_axis=batch_axis,
                                                  channel_axis=channel_axis)),
                          ]
                          ))
  def test_aggregate(self, get, readout, same_input, activation, mask_constant,
                     shape, batch_axis, channel_axis, agg_axes, do_batch,
                     implementation, to_dense):
    if xla_bridge.get_backend().platform == 'cpu' and len(shape) > 1:
      raise absltest.SkipTest('Skipping large shapes on CPU.')

    if implementation == 'SPARSE' and to_dense != 'identity':
      raise absltest.SkipTest('`implementation="SPARSE"` ignores '
                              '`to_dense` argument.')

    if get == 'cov2' and same_input:
      raise absltest.SkipTest('`get="cov2"` only defined for different inputs.')

    if get in ('cov1', 'cov2') and do_batch:
      raise absltest.SkipTest('Batching of empirical kernel does not work for '
                              '`diagonal_axes != ()`.')

    prandom.seed(1)
    batch1, batch2 = 8, 4
    num_channels = 1
    output_dims = 1 if get == 'ntk' else 2**6

    key = random.PRNGKey(1)
    key, split1, split2 = random.split(key, 3)

    x1 = random.normal(split1, (batch1,) + shape + (num_channels,))
    x1 = np.moveaxis(x1, (0, -1), (batch_axis, channel_axis))

    if same_input:
      x2 = None
    else:
      x2 = random.normal(split2, (batch2,) + shape + (num_channels,))
      x2 = np.moveaxis(x2, (0, -1), (batch_axis, channel_axis))

    if mask_constant is not None:
      key, split1, split2 = random.split(key, 3)

      shape1 = list(x1.shape)
      shape1[channel_axis] = 1
      mask1 = random.bernoulli(split1, p=0.3, shape=shape1)
      x1 = np.where(mask1, mask_constant, x1)

      if not same_input:
        shape2 = list(x2.shape)
        shape2[channel_axis] = 1
        mask2 = random.bernoulli(split2, p=0.2, shape=shape2)
        x2 = np.where(mask2, mask_constant, x2)

    key, split1, split2 = random.split(key, 3)

    agg_shape = shape if agg_axes is None else tuple(x1.shape[a]
                                                     for a in agg_axes)
    agg_ndim = len(agg_shape)

    def sparse_to_dense(pattern):
      if pattern is None:
        return None

      pattern = pattern.reshape(pattern.shape[:2] + (pattern.shape[2] * 2,))

      bsz, n_edges, n_dims = pattern.shape
      batch_range = np.broadcast_to(
          np.arange(bsz).reshape((bsz, 1, 1)),
          (bsz, n_edges, 1))
      pattern = np.concatenate([batch_range, pattern], 2)
      pattern = pattern.reshape((bsz * n_edges, n_dims + 1))
      out = np.zeros((bsz,) + tuple(a for a in agg_shape for _ in (0, 1)))
      out = out.at[tuple(pattern.T)].add(1.)
      out = utils.unzip_axes(out, 1)
      return out

    if to_dense == 'sparse_to_dense' or implementation == 'SPARSE':
      def get_sparse_pattern(batch_size, rng):
        n_edges_max = onp.prod((1,) + agg_shape)**2
        n_edges = prandom.randint(0, n_edges_max)
        pattern = [np.zeros((batch_size, n_edges, 0, 2), np.int32)]

        for d in range(agg_ndim):
          rng, _ = random.split(rng)
          n_nodes = agg_shape[d]
          edges = random.randint(rng, (batch_size, n_edges, 1, 2), 0, n_nodes)
          pattern += [edges]

        pattern = np.concatenate(pattern, 2)

        # TODO(romann): make masking work for NTK after
        # https://github.com/google/jax/issues/7538.
        if get != 'ntk':
          mask = random.bernoulli(rng, p=0.2, shape=pattern.shape[:2])
          # Make sure the receivers are masked to large negative number.
          # The number needs to be larger than maximum size of `pattern` along
          # any of the shape axes, to make `jax.ops.at` ignore these entries in
          # `sparse_to_dense` above, otherwise they are treated as regular
          # negative indices.
          pattern = pattern.at[mask].set(-10000)

        return pattern

      pattern1 = get_sparse_pattern(batch1, split1)
      pattern2 = pattern1 if same_input else get_sparse_pattern(batch2, split2)

    else:
      pattern1 = random.uniform(split1, (batch1,) + agg_shape * 2)
      pattern2 = pattern1 if same_input else random.uniform(
          split2, (batch2,) + agg_shape * 2)

    # Build the infinite network.
    def get_nn(to_dense, implementation):
      return stax.serial(
          stax.Dense(2**6, batch_axis=batch_axis, channel_axis=channel_axis),
          activation,
          stax.Aggregate(aggregate_axis=agg_axes,
                         batch_axis=batch_axis,
                         channel_axis=channel_axis,
                         to_dense={
                             'identity': lambda p: p,
                             'sparse_to_dense': sparse_to_dense}[to_dense],
                         implementation=implementation
                         ),
          readout,
          stax.Dense(output_dims,
                     batch_axis=int(batch_axis > channel_axis),
                     channel_axis=int(batch_axis < channel_axis)))

    init_fn, apply_fn, kernel_fn = get_nn(to_dense, implementation)
    apply_fn = jit(apply_fn)
    kernel_fn = jit(kernel_fn, static_argnums=2)

    if do_batch:
      kernel_fn = batch.batch(kernel_fn, batch_size=2)

    exact = kernel_fn(x1, x2, get,
                      mask_constant=mask_constant,
                      pattern=(pattern1, pattern2))

    rtol = 0.08

    if to_dense == 'sparse_to_dense' or implementation == 'SPARSE':
      init_fn_dense, apply_fn_dense, kernel_fn_dense = get_nn('identity',
                                                              'DENSE')
      apply_fn_dense = jit(apply_fn_dense)
      kernel_fn_dense = jit(kernel_fn_dense, static_argnums=2)

      pattern1_dense = sparse_to_dense(pattern1)
      pattern2_dense = sparse_to_dense(pattern2)

      # Test parameters agreement
      key, old = random.split(key, 2)
      _, params_sparse = init_fn(key, x1.shape)
      _, params_dense = init_fn_dense(key, x1.shape)
      self.assertAllClose(params_dense, params_sparse)

      # Test forward-pass agreement
      fx1_dense = apply_fn_dense(params_dense, x1, pattern=pattern1_dense)
      fx1_sparse = apply_fn(params_sparse, x1, pattern=pattern1)
      test_utils.assert_close_matrices(self, fx1_dense, fx1_sparse, rtol)

      if not same_input:
        fx2_dense = apply_fn_dense(params_dense, x2, pattern=pattern2_dense)
        fx2_sparse = apply_fn(params_sparse, x2, pattern=pattern2)
        test_utils.assert_close_matrices(self, fx2_dense, fx2_sparse, rtol)

      # Test agreement with analytic dense kernel
      exact_dense = kernel_fn_dense(x1, x2, get,
                                    mask_constant=mask_constant,
                                    pattern=(pattern1_dense, pattern2_dense))

      self.assertAllClose(exact_dense, exact)

    # Test agreement with empirical kernel
    kernel_mc_fn = monte_carlo.monte_carlo_kernel_fn(
        init_fn=init_fn,
        apply_fn=apply_fn,
        key=random.PRNGKey(10),
        n_samples=2**6,
        batch_size=2 if (xla_bridge.get_backend().platform == 'tpu'
                         and batch_axis == 0) else 0,
        device_count=-1 if batch_axis == 0 else 0,
        implementation=2,
        trace_axes=(int(batch_axis < channel_axis),)
    )

    if get in ('nngp', 'ntk'):
      empirical = kernel_mc_fn(x1, x2, get,
                               mask_constant=mask_constant,
                               pattern=(pattern1, pattern2))

    elif get in ('cov1', 'cov2'):
      if get == 'cov1':
        empirical = kernel_mc_fn(x1, None, 'nngp',
                                 mask_constant=mask_constant,
                                 pattern=(pattern1, pattern1))

      elif get == 'cov2':
        empirical = kernel_mc_fn(x2, None, 'nngp',
                                 mask_constant=mask_constant,
                                 pattern=(pattern2, pattern2))

      empirical = np.moveaxis(np.diagonal(empirical), -1, 0)

    else:
      raise ValueError(get)

    test_utils.assert_close_matrices(self, exact, empirical, rtol)


class ConvTransposeTest(test_utils.NeuralTangentsTestCase):

  @parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              f'_same_inputs={same_inputs}_{padding}_size={size}_'
              f'strides={strides}_filter={filter_shape}_'
              f'diag_batch={diagonal_batch}_diag_spatial={diagonal_spatial}',
          'padding': padding,
          'size': size,
          'same_inputs': same_inputs,
          'filter_shape': filter_shape,
          'strides': strides,
          'diagonal_batch': diagonal_batch,
          'diagonal_spatial': diagonal_spatial
      }
                          for padding in ['CIRCULAR', 'SAME', 'VALID']
                          for same_inputs in [False]
                          for filter_shape in range(2, 5)
                          for strides in range(2, 5)
                          for size in range(2, 5)
                          for diagonal_batch in [True]
                          for diagonal_spatial in [True, False]))
  def test_conv_transpose(self, same_inputs, padding, filter_shape, strides,
                          size, diagonal_batch, diagonal_spatial):
    if size > 2:
      _skip_test()

    width = 512
    tol = 0.01
    n_samples = 512
    filter_shape = (filter_shape,)
    strides = (strides,)

    init_fn, apply_fn, kernel_fn = stax.ConvTranspose(width, filter_shape,
                                                      strides, padding,
                                                      b_std=0.1)

    key = random.PRNGKey(1)
    shape = (size, 1)
    x1 = random.normal(key, (2,) + shape)
    x2 = random.normal(key, (3,) + shape) if not same_inputs else None

    k = kernel_fn(x1, x2,
                  diagonal_batch=diagonal_batch,
                  diagonal_spatial=diagonal_spatial,
                  get='cov1' if diagonal_batch else 'nngp')

    diagonal_axes = ()
    if diagonal_batch:
      diagonal_axes += (0,)
    if diagonal_spatial:
      diagonal_axes += (1,)

    kernel_fn_mc = monte_carlo.monte_carlo_kernel_fn(
        init_fn, apply_fn, key, n_samples, diagonal_axes=diagonal_axes,
        device_count=0, implementation=2, vmap_axes=0)
    k_mc = kernel_fn_mc(x1, None if diagonal_batch else x2, 'nngp')

    test_utils.assert_close_matrices(self, k_mc, k, tol)

  @classmethod
  def _conv_transpose_circular_via_grad(cls,
                                        lhs,
                                        params,
                                        strides,
                                        padding,
                                        dimension_numbers):
    """Helper method: calculates conv transpose via grad for testing.

    Adapted from `jax.tests.lax_test`.
    """
    rhs = params[0]
    rhs = np.swapaxes(rhs, dimension_numbers[1].index('O'),
                      dimension_numbers[1].index('I'))
    rhs = np.flip(rhs, dimension_numbers[1].index('H'))
    assert len(lhs.shape) == len(rhs.shape)
    nspatial = len(lhs.shape) - 2
    dn = lax.conv_dimension_numbers(lhs.shape, rhs.shape, dimension_numbers)
    in_shape = onp.take(lhs.shape, dn.lhs_spec)
    in_sdims = in_shape[2:]
    k_shape = onp.take(rhs.shape, dn.rhs_spec)
    o_sdims = [in_sdims[i]*strides[i] for i in range(nspatial)]
    o_shape = [in_shape[0], k_shape[1]] + o_sdims
    out_spec_inv = [x[0] for x in
                    sorted(enumerate(dn.out_spec), key=lambda x: x[1])]
    o_layout = onp.take(onp.array(o_shape), out_spec_inv)
    placeholder = np.ones(o_layout, lhs.dtype)

    _, apply_fn, _ = stax.Conv(
        out_chan=rhs.shape[dimension_numbers[1].index('I')],
        filter_shape=(rhs.shape[dimension_numbers[1].index('H')],),
        strides=strides,
        padding=padding,
        dimension_numbers=dimension_numbers,
        parameterization='standard'
    )
    conv = lambda x: apply_fn((rhs, 0.), x)
    _, g = vjp(conv, placeholder)
    return g(lhs)[0]

  @classmethod
  def _conv_transpose_circular(cls,
                               lhs,
                               params,
                               strides,
                               padding,
                               dimension_numbers):
    """Helper method: calculates conv transpose."""
    _, apply_fn, _ = stax.ConvTranspose(
        out_chan=params[0].shape[dimension_numbers[1].index('O')],
        filter_shape=(params[0].shape[dimension_numbers[1].index('H')],),
        strides=strides,
        padding=padding,
        dimension_numbers=dimension_numbers,
        parameterization='standard'
    )
    return apply_fn((params[0], 0.), lhs)

  @parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              f'size={size}_strides={strides}_filter={filter_shape}',
          'size': size,
          'filter_shape': filter_shape,
          'strides': strides,
      }
                          for filter_shape in range(1, 5)
                          for strides in range(1, 5)
                          for size in range(1, 5)))
  def test_conv_transpose_circular(self, size, filter_shape, strides):
    if size > 2:
      _skip_test()

    x = random.normal(random.PRNGKey(1), (2, size, 3))
    dn = ('NHC', 'HIO', 'NHC')
    padding = 'CIRCULAR'
    filter_shape = (filter_shape,)
    strides = (strides,)

    init_fn, _, _ = stax.ConvTranspose(4, filter_shape, strides, padding)
    _, params = init_fn(random.PRNGKey(2), x.shape)
    f_conv = self._conv_transpose_circular(x, params, strides, padding, dn)
    f_adj = self._conv_transpose_circular_via_grad(x, params, strides, padding,
                                                   dn)
    self.assertAllClose(f_adj, f_conv)


class DotGeneralTest(test_utils.NeuralTangentsTestCase):

  @parameterized.named_parameters(
      jtu.cases_from_list(
          {
              'testcase_name':
                  ' [{}_n={}_dn=(({}, {}), ({}, {}))_channel_axis={}_'
                  'batch_axis={}_{}_{}_batch={}_spatial={}]'.format(
                      'same_inputs' if same_inputs else 'different_inputs',
                      n,
                      contracting_dims,
                      c_dims,
                      batch_dims,
                      b_dims,
                      channel_axis,
                      batch_axis,
                      'rhs' if is_rhs else 'lhs',
                      r_permutation,
                      diagonal_batch,
                      diagonal_spatial
                  ),
              'same_inputs': same_inputs,
              'n': n,
              'batch_dims': batch_dims,
              'contracting_dims': contracting_dims,
              'b_dims': b_dims,
              'c_dims': c_dims,
              'r_permutation': r_permutation,
              'channel_axis': channel_axis,
              'batch_axis': batch_axis,
              'is_rhs': is_rhs,
              'diagonal_spatial': diagonal_spatial,
              'diagonal_batch': diagonal_batch
          }
          for same_inputs in [True, False]
          for n in [2, 3]
          for is_rhs in [False, True]
          for batch_axis in range(n)
          for channel_axis in [i for i in range(n) if i != batch_axis]
          for diagonal_spatial in [True, False]
          for diagonal_batch in [True, False]
          for batch_dims in more_itertools.powerset(
              i for i in range(n)
              if i != channel_axis)
          for contracting_dims in more_itertools.powerset(
              i for i in range(n)
              if i not in batch_dims + (channel_axis,))
          for c_dims in itertools.permutations(contracting_dims)
          for b_dims in itertools.permutations(batch_dims)
          for r_permutation in itertools.permutations(range(n))
      )
  )
  def test_dot_general(self, same_inputs, n, batch_dims, contracting_dims,
                       c_dims, b_dims, r_permutation, channel_axis, is_rhs,
                       diagonal_spatial, diagonal_batch, batch_axis):
    if xla_bridge.get_backend().platform == 'cpu' and n != 2:
      raise absltest.SkipTest(f'Skipping n = {n} on CPU.')

    n_b = 2
    n_c = 1
    key1, key2, key3 = random.split(random.PRNGKey(1), 3)

    x_shape_n_c = [2, 4, 6, 8, 10, 12, 14][:n - 2]
    x_shape = list(x_shape_n_c)

    for a in sorted((batch_axis, channel_axis)):
      x_shape.insert(a, n_b if a == batch_axis else n_c)

    mask_constant = 10.

    x1 = np.cos(random.normal(key1, x_shape))
    mask1 = random.bernoulli(key1, p=0.8, shape=x1.shape)
    x1 = np.where(mask1, mask_constant, x1)

    if same_inputs:
      x2 = None
    else:
      x2_shape = (x_shape[:batch_axis] +
                  [4 if (batch_axis not in contracting_dims + batch_dims)
                   else x_shape[batch_axis]] +
                  x_shape[batch_axis + 1:])
      x2 = np.cos(random.normal(key2, x2_shape))
      mask2 = random.bernoulli(key2, p=0.4, shape=x2.shape)
      x2 = np.where(mask2, mask_constant, x2)

    other_shape = [1, 3, 5, 7, 9, 11, 13, 15][:n]
    for i in contracting_dims + batch_dims:
      other_shape[i] = x_shape[i]
    other = random.normal(key3, other_shape)
    other = np.arange(np.size(other)).reshape(other_shape)

    other_t = np.transpose(other, r_permutation)
    r_c_dims = tuple(r_permutation.index(c) for c in c_dims)
    r_b_dims = tuple(r_permutation.index(b) for b in b_dims)

    if is_rhs:
      lhs, rhs = None, other_t
      dn = ((c_dims, r_c_dims), (b_dims, r_b_dims))
    else:
      lhs, rhs = other_t, None
      dn = ((r_c_dims, c_dims), (r_b_dims, b_dims))

    lhs_ndim = None if lhs is None else lhs.ndim
    init_fn, apply_fn, kernel_fn = stax.DotGeneral(lhs=lhs,
                                                   rhs=rhs,
                                                   dimension_numbers=dn,
                                                   batch_axis=batch_axis)
    def get_exact():
      return kernel_fn(x1, x2,
                       diagonal_spatial=diagonal_spatial,
                       diagonal_batch=diagonal_batch,
                       batch_axis=batch_axis,
                       channel_axis=channel_axis,
                       mask_constant=mask_constant)

    if (([i for i in c_dims if i not in (batch_axis, channel_axis)] and
         diagonal_spatial) or
        (batch_axis in c_dims and diagonal_batch)):
      self.assertRaises(ValueError, get_exact)

    else:
      exact = get_exact()

      out_c_axis = utils.axis_after_dot(channel_axis, c_dims, b_dims, lhs_ndim)
      out_b_axis = utils.axis_after_dot(batch_axis, c_dims, b_dims, lhs_ndim)

      def get_empirical(get):
        def get_diagonal_axes():
          axes = ()
          if (get in ('cov1', 'cov2') and
              diagonal_batch and
              batch_axis not in c_dims):
            axes += (out_b_axis,)

          if diagonal_spatial:
            axes += tuple(
                utils.axis_after_dot(i, c_dims, b_dims, lhs_ndim)
                for i in range(n)
                if i not in c_dims + (batch_axis, channel_axis))
            rhs_ndim = None if rhs is None else rhs.ndim
            axes += tuple(
                utils.axis_after_dot(i, r_c_dims, r_b_dims, rhs_ndim)
                for i in range(n)
                if i not in r_c_dims and
                not (i in r_b_dims and b_dims[r_b_dims.index(i)] == batch_axis))
          return axes

        def batch_axes():
          if batch_axis in contracting_dims:
            return (), ()

          axis = out_b_axis
          if out_c_axis < axis:
            axis -= 1
          if not diagonal_spatial:
            axis *= 2

          if get in ('cov1', 'cov2') and diagonal_batch:
            return (axis,), (0,)
          return (axis, axis + 1), (0, 1)

        kernel_fn_mc = monte_carlo.monte_carlo_kernel_fn(
            init_fn=init_fn,
            apply_fn=apply_fn,
            key=key1,
            n_samples=1,
            trace_axes=(out_c_axis,),
            diagonal_axes=get_diagonal_axes(),
            device_count=-1 if (get == 'nngp' and
                                batch_axis == out_b_axis == 0 and
                                0 not in c_dims + b_dims) else 0,
            implementation=2,
        )

        empirical = kernel_fn_mc(x1=x2 if get == 'cov2' else x1,
                                 x2=x2 if get == 'nngp' else None,
                                 get='nngp',
                                 batch_axis=batch_axis,
                                 channel_axis=channel_axis,
                                 mask_constant=mask_constant)
        empirical = np.moveaxis(empirical, *batch_axes())
        return empirical

      for get in ('nngp', 'cov1', 'cov2'):
        if get == 'cov2' and same_inputs:
          continue

        with self.subTest(get=get):
          test_utils.assert_close_matrices(
              self, get_empirical(get), getattr(exact, get), 0.01)

  @parameterized.named_parameters(
      jtu.cases_from_list(
          {
              'testcase_name': ' [{}_get={}_n={}_{}_{}_{}]'.format(
                  'same_inputs' if same_inputs else 'different_inputs',
                  get,
                  n,
                  'pool' if do_pool else 'flat',
                  'rhs' if is_rhs else 'lhs',
                  'dot_first' if dot_first else 'conv_first'
              ),
              'same_inputs': same_inputs,
              'get': get,
              'n': n,
              'do_pool': do_pool,
              'is_rhs': is_rhs,
              'dot_first': dot_first
          }
          for same_inputs in [False, True]
          for get in ['ntk']
          for do_pool in [True, False]
          for n in [3, 4]
          for is_rhs in [False, True]
          for dot_first in [True, False]
      )
  )
  def test_dot_general_nn(self, same_inputs, get, n, is_rhs, do_pool,
                          dot_first):
    if xla_bridge.get_backend().platform == 'cpu' and n != 2:
      raise absltest.SkipTest(f'Skipping n = {n} on CPU.')

    width = 2**8
    n_samples = 2**8
    tol = 0.03
    key1, key2, key3 = random.split(random.PRNGKey(1), 3)

    mask_constant = 10.

    x_shape = [6, 3, 4, 5][:n - 1] + [1]
    x1 = np.cos(random.normal(key1, x_shape))
    mask1 = random.bernoulli(key1, p=0.8, shape=x1.shape)
    x1 = np.where(mask1, mask_constant, x1)

    if same_inputs:
      x2 = None
    else:
      x2 = np.cos(random.normal(key2, x_shape))
      mask2 = random.bernoulli(key2, p=0.4, shape=x2.shape)
      x2 = np.where(mask2, mask_constant, x2)

    other = random.normal(key3, [3, 4, 6, 2])

    c_dims, b_dims = (1,), (0,)
    o_c_dims, o_b_dims = (0,), (2,)

    if is_rhs:
      lhs, rhs = None, other
      dn = ((c_dims, o_c_dims), (b_dims, o_b_dims))
    else:
      lhs, rhs = other, None
      dn = ((o_c_dims, c_dims), (o_b_dims, b_dims))

    lhs_ndim = None if lhs is None else lhs.ndim
    out_c_axis = utils.axis_after_dot(n - 1, c_dims, b_dims, lhs_ndim)
    out_b_axis = utils.axis_after_dot(0, c_dims, b_dims, lhs_ndim)

    top_b_axis = int(out_b_axis > out_c_axis and do_pool)
    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Identity() if dot_first else stax.Conv(
            width, (3,) * (n - 2), padding='SAME'),
        stax.DotGeneral(lhs=lhs,
                        rhs=rhs,
                        dimension_numbers=dn),
        stax.Dense(width, batch_axis=out_b_axis, channel_axis=out_c_axis),
        stax.Relu(),
        (stax.GlobalAvgPool(channel_axis=out_c_axis,
                            batch_axis=out_b_axis) if do_pool else
         stax.Flatten(batch_axis=out_b_axis)),
        stax.Dense(
            width if get == 'nngp' else 1, 0.9, 0.1,
            batch_axis=top_b_axis,
            channel_axis=int(out_c_axis > out_b_axis or not do_pool))
    )

    kernel_fn_mc = monte_carlo.monte_carlo_kernel_fn(
        init_fn, apply_fn, key1, n_samples,
        trace_axes=(int(out_c_axis > out_b_axis) if do_pool else 1,),
        device_count=0,
        implementation=2
    )

    empirical = kernel_fn_mc(x1, x2, get, mask_constant=mask_constant)
    exact = kernel_fn(x1, x2, get, mask_constant=mask_constant)
    test_utils.assert_close_matrices(self, empirical, exact, tol)

  def test_dot_general_mask(self):
    x1, x2 = np.ones((4, 2, 3, 1)), np.ones((4, 2, 3, 1))

    mask_constant = 10.

    def get_k(x1, x2, m1, m2):
      x1, x2 = np.where(m1, mask_constant, x1), np.where(m2, mask_constant, x2)
      k_fn = stax.DotGeneral(
          rhs=np.ones(x1.shape[:-1]),
          dimension_numbers=(((1,), (1,)), ((2,), (2,))))[2]
      k = k_fn(x1, x2, 'nngp', mask_constant=mask_constant)
      return k

    m1, m2 = np.zeros_like(x1, np.bool_), np.zeros_like(x2, np.bool_)
    k = get_k(x1, x2, m1, m2)
    self.assertAllClose(np.ones_like(k) * 4, k)

    m1, m2 = np.ones_like(x1, np.bool_), np.zeros_like(x2, np.bool_)
    k = get_k(x1, x2, m1, m2)
    self.assertAllClose(np.zeros_like(k), k)

    m1, m2 = np.ones_like(x1, np.bool_), np.ones_like(x2, np.bool_)
    k = get_k(x1, x2, m1, m2)
    self.assertAllClose(np.zeros_like(k), k)

    m1 = np.concatenate([np.ones_like(x1[:2], np.bool_),
                         np.zeros_like(x1[2:], np.bool_)])
    m2 = np.zeros_like(x2, np.bool_)
    k = get_k(x1, x2, m1, m2)
    self.assertAllClose(np.zeros_like(k[:2]), k[:2])
    self.assertAllClose(np.full_like(k[2:], 4.), k[2:])


class ImageResizeTest(test_utils.NeuralTangentsTestCase):

  @parameterized.named_parameters(
      jtu.cases_from_list(
          {
              'testcase_name':
                  ' [{}_n={}_channel_axis={}_'
                  'batch_axis={}_batch={}_spatial={}_method={}_antialias={}_'
                  'precison={}_shape={}]'.format(
                      'same_inputs' if same_inputs else 'different_inputs',
                      n,
                      channel_axis,
                      batch_axis,
                      diagonal_batch,
                      diagonal_spatial,
                      method,
                      antialias,
                      precision,
                      shape
                  ),
              'same_inputs': same_inputs,
              'n': n,
              'channel_axis': channel_axis,
              'batch_axis': batch_axis,
              'diagonal_spatial': diagonal_spatial,
              'diagonal_batch': diagonal_batch,
              'method': method,
              'antialias': antialias,
              'precision': precision,
              'shape': shape
          }
          for same_inputs in [
              True,
              False
          ]
          for n in [
              2,
              3,
              4
          ]
          for batch_axis in range(n)
          for channel_axis in [i for i in range(n) if i != batch_axis]
          for diagonal_spatial in [
              True,
              False
          ]
          for diagonal_batch in [
              True,
              False
          ]
          for method in [
              'linear',
              'nearest'
          ]
          for antialias in [
              True,
              False
          ]
          for precision in [
              lax.Precision.DEFAULT
          ]
          for shape in [s[:n] for s in [
              (-1, 2, 3, 4),
              (-1, 3, -1, 4),
              (10, 5, 1, 8),
              (5, -1, 2, 3)
          ]]
      )
  )
  def test_image_resize(self, same_inputs, n, channel_axis, diagonal_spatial,
                        diagonal_batch, batch_axis, method, antialias,
                        precision, shape):
    if xla_bridge.get_backend().platform == 'cpu' and n > 2:
      raise absltest.SkipTest(f'Skipping n = {n} on CPU.')

    n_b1, n_b2 = 2, 4
    n_c = 1
    key1, key2, key3 = random.split(random.PRNGKey(1), 3)

    shape = shape[:channel_axis] + (-1,) + shape[channel_axis + 1:]

    x_shape_n_c = [2, 4, 6, 8, 10, 12, 14][:n - 2]
    x_shape = list(x_shape_n_c)

    for a in sorted((batch_axis, channel_axis)):
      x_shape.insert(a, n_b1 if a == batch_axis else n_c)

    mask_constant = 10.

    x1 = np.cos(random.normal(key1, x_shape))
    mask1 = random.bernoulli(key1, p=0.3, shape=x1.shape)
    x1 = np.where(mask1, mask_constant, x1)

    if same_inputs:
      x2 = None
    else:
      x2_shape = (x_shape[:batch_axis] +
                  [n_b2] +
                  x_shape[batch_axis + 1:])
      x2 = np.cos(random.normal(key2, x2_shape))
      mask2 = random.bernoulli(key2, p=0.2, shape=x2.shape)
      x2 = np.where(mask2, mask_constant, x2)

    init_fn, apply_fn, kernel_fn = stax.ImageResize(method=method,
                                                    antialias=antialias,
                                                    precision=precision,
                                                    batch_axis=batch_axis,
                                                    channel_axis=channel_axis,
                                                    shape=shape
                                                    )
    def get_exact():
      return kernel_fn(x1, x2,
                       diagonal_spatial=diagonal_spatial,
                       diagonal_batch=diagonal_batch,
                       batch_axis=batch_axis,
                       channel_axis=channel_axis,
                       mask_constant=mask_constant
                       )

    if ((shape[batch_axis] != -1 and diagonal_batch) or
        (any(shape[i] != -1 for i in range(len(shape))
            if i not in (batch_axis, channel_axis)) and diagonal_spatial)):
      self.assertRaises(ValueError, get_exact)

    else:
      exact = get_exact()

      def get_empirical(get):
        def get_diagonal_axes():
          axes = ()
          if get in ('cov1', 'cov2') and diagonal_batch:
            axes += (batch_axis,)

          if diagonal_spatial:
            axes += tuple(i for i in range(n)
                          if i not in (batch_axis, channel_axis))

          return axes

        kernel_fn_mc = monte_carlo.monte_carlo_kernel_fn(
            init_fn=init_fn,
            apply_fn=apply_fn,
            key=key1,
            n_samples=1,
            trace_axes=(channel_axis,),
            diagonal_axes=get_diagonal_axes(),
            device_count=-1 if (get == 'nngp' and
                                batch_axis == 0 and
                                shape[batch_axis] == -1) else 0,
            implementation=2,
        )

        empirical = kernel_fn_mc(x1=x2 if get == 'cov2' else x1,
                                 x2=x2 if get == 'nngp' else None,
                                 get='nngp',
                                 batch_axis=batch_axis,
                                 channel_axis=channel_axis,
                                 mask_constant=mask_constant
                                 )

        def batch_axes():
          axis = batch_axis
          if channel_axis < batch_axis:
            axis -= 1
          if not diagonal_spatial:
            axis *= 2

          if get in ('cov1', 'cov2') and diagonal_batch:
            return (axis,), (0,)
          return (axis, axis + 1), (0, 1)

        empirical = np.moveaxis(empirical, *batch_axes())
        return empirical

      for get in ('nngp', 'cov1', 'cov2'):
        if get == 'cov2' and same_inputs:
          continue

        with self.subTest(get=get):
          tol = 1e-2 if xla_bridge.get_backend().platform == 'tpu' else 1e-5
          test_utils.assert_close_matrices(
              self, get_empirical(get), getattr(exact, get), tol)

  @parameterized.named_parameters(
      jtu.cases_from_list(
          {
              'testcase_name': ' [{}_get={}_n={}_{}_{}_{}_shape={}]'.format(
                  'same_inputs' if same_inputs else 'different_inputs',
                  get,
                  n,
                  'pool' if do_pool else 'flat',
                  method,
                  bottom_layer,
                  shape
              ),
              'same_inputs': same_inputs,
              'get': get,
              'n': n,
              'do_pool': do_pool,
              'method': method,
              'bottom_layer': bottom_layer,
              'shape': shape
          }
          for same_inputs in [False, True]
          for get in ['ntk']
          for do_pool in [True, False]
          for n in [3]
          for bottom_layer in ['resize', 'conv', 'relu']
          for method in ['linear', 'nearest']
          for shape in [
              (1, 2, 4),
              (2, 1, 1),
              (-1, 2, -1),
              (2, 4, -1),
              (9, -1, -1),
              (-1, -1, -1),
              (3, 4, -1),
              (1, 1, -1),
          ]
      )
  )
  def test_image_resize_nn(self, same_inputs, get, n, do_pool, bottom_layer,
      method, shape):
    # if xla_bridge.get_backend().platform == 'cpu' and n != 2:
    #   raise absltest.SkipTest(f'Skipping n = {n} on CPU.')

    width = 2**7
    n_samples = 2**7
    tol = 0.03
    key1, key2, key3 = random.split(random.PRNGKey(1), 3)

    mask_constant = 10.

    x_shape = [6, 3, 4, 5][:n - 1] + [1]
    x1 = np.cos(random.normal(key1, x_shape))
    mask1 = random.bernoulli(key1, p=0.2, shape=x1.shape)
    x1 = np.where(mask1, mask_constant, x1)

    if same_inputs:
      x2 = None
    else:
      x2 = np.cos(random.normal(key2, x_shape))
      mask2 = random.bernoulli(key2, p=0.1, shape=x2.shape)
      x2 = np.where(mask2, mask_constant, x2)

    bottom = {'conv': stax.Conv(width, (3,) * (n - 2), padding='SAME'),
              'relu': stax.serial(
                  stax.Conv(width, (3,) * (n - 2), padding='SAME'),
                  stax.Relu()),
              'resize': stax.Identity()}[bottom_layer]

    init_fn, apply_fn, kernel_fn = stax.serial(
        bottom,
        stax.ImageResize(method=method,
                         shape=shape),
        stax.Conv(width, (2,), padding='SAME'),
        stax.Relu(),
        (stax.GlobalAvgPool() if do_pool else stax.Flatten()),
        stax.Dense(width if get == 'nngp' else 1, 0.9, 0.1)
    )

    kernel_fn_mc = monte_carlo.monte_carlo_kernel_fn(
        init_fn, apply_fn, key1, n_samples,
        device_count=0,
        implementation=2
    )

    empirical = kernel_fn_mc(x1, x2, get, mask_constant=mask_constant)

    def get_exact():
      return kernel_fn(x1, x2, get, mask_constant=mask_constant)

    if shape[-1] != -1:
      # Make sure an error is thrown if resizing a channel axis is requested.
      self.assertRaises(ValueError, get_exact)
    else:
      exact = get_exact()
      test_utils.assert_close_matrices(self, empirical, exact, tol)


class ConvLocalTest(test_utils.NeuralTangentsTestCase):

  @parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name': f'_diag_spatial={diagonal_spatial}_',
          'diagonal_spatial': diagonal_spatial,
      }
                          for diagonal_spatial in [True, False]))
  def test_whitened_inputs(self, diagonal_spatial):
    _skip_test()

    x = np.cos(random.normal(random.PRNGKey(1), (4 * 8 * 8, 512)))
    cov = x @ x.T
    whiten = np.linalg.cholesky(np.linalg.inv(cov))
    x_white = whiten.T @ x
    cov_white = x_white @ x_white.T
    self.assertAllClose(np.eye(x.shape[0]), cov_white)

    width = 256

    scales = random.normal(random.PRNGKey(2), (4, 8, 8, 1))

    x_white = x_white.reshape((4, 8, 8, 512)) * scales
    x = x.reshape(x_white.shape) * scales

    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.AvgPool((2, 3)),
        stax.ConvLocal(width, (3, 1), padding='SAME', W_std=4.2, b_std=0.09),
        stax.Relu(),
        stax.Conv(width, (2, 3), padding='SAME', W_std=3.8, b_std=0.04),
        stax.Relu(),
        stax.ConvLocal(width, (2, 2), padding='SAME', W_std=6.4, b_std=0.1),
        stax.GlobalAvgPool())

    k_white = kernel_fn(x_white, None, diagonal_spatial=diagonal_spatial)
    self._test_against_mc(apply_fn, init_fn, k_white.nngp, x_white, None)

    k = kernel_fn(x, None, diagonal_spatial=diagonal_spatial)

    if diagonal_spatial:
      with self.assertRaises(AssertionError):
        self._test_against_mc(apply_fn, init_fn, k.nngp, x, None)
    else:
      self._test_against_mc(apply_fn, init_fn, k.nngp, x, None)

  @parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              f'_same_inputs={same_inputs}_{padding}_size={size}_'
              f'strides={strides}_filter={filter_shape}_'
              f'diag_batch={diagonal_batch}_diag_spatial={diagonal_spatial}_'
              f'get={get}_parameterization={parameterization}',
          'padding': padding,
          'size': size,
          'same_inputs': same_inputs,
          'filter_shape': filter_shape,
          'strides': strides,
          'diagonal_batch': diagonal_batch,
          'diagonal_spatial': diagonal_spatial,
          'get': get,
          'parameterization': parameterization
      }
                          for padding in ['SAME', 'VALID', 'CIRCULAR']
                          for same_inputs in [False]
                          for filter_shape in range(2, 4)
                          for strides in range(1, 3)
                          for size in range(2, 4)
                          for diagonal_batch in [True]
                          for diagonal_spatial in [True, False]
                          for get in ['cov1', 'nngp', 'ntk']
                          for parameterization in ['standard', 'ntk']))
  def test_conv_local(self, same_inputs, padding, filter_shape, strides,
                      size, diagonal_batch, diagonal_spatial, get,
                      parameterization):
    _skip_test()

    if diagonal_batch and get != 'cov1':
      raise absltest.SkipTest('Checking `diagonal_batch` only on `cov1`.')

    key1, key2, key_mc = random.split(random.PRNGKey(1), 3)
    shape = (size, 1)
    x1 = random.normal(key1, (2,) + shape)
    x2 = random.normal(key2, (3,) + shape) if not same_inputs else None

    kernel_kwargs = dict(diagonal_batch=diagonal_batch,
                         diagonal_spatial=diagonal_spatial)

    conv_kwargs = dict(out_chan=512,
                       filter_shape=(filter_shape,),
                       strides=(strides,),
                       padding=padding,
                       b_std=0.2,
                       W_std=1.5,
                       parameterization=parameterization)

    init_fn, apply_fn, kernel_fn = stax.ConvLocal(**conv_kwargs)
    k = kernel_fn(x1, x2, **kernel_kwargs)

    # Compared to MC estimate
    diagonal_axes = ()
    if diagonal_batch:
      diagonal_axes += (0,)
    if diagonal_spatial:
      diagonal_axes += (1,)

    kernel_fn_mc = monte_carlo.monte_carlo_kernel_fn(
        init_fn, apply_fn, key_mc, n_samples=512, diagonal_axes=diagonal_axes,
        device_count=0,
        implementation=2,
        vmap_axes=0
    )
    k_mc = kernel_fn_mc(x1, None if get == 'cov1' else x2,
                        'nngp' if get == 'cov1' else get)
    test_utils.assert_close_matrices(self, k_mc, getattr(k, get), 0.011)

    # Compared diagonal entries to CNN
    _, _, kernel_fn_conv = stax.Conv(**conv_kwargs)
    k_conv = kernel_fn_conv(x1, x2, **kernel_kwargs)

    if not diagonal_spatial:
      def get_diag(k):
        k = getattr(k, get)
        k = np.diagonal(k, axis1=-1, axis2=-2)
        return k
      k_conv = get_diag(k_conv)
      k = get_diag(k)

    tol = 0.005 if xla_bridge.get_backend().platform == 'tpu' else 0.001
    self.assertAllClose(k_conv, k, atol=tol, rtol=tol)

  @parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              f'_get={get}'
              f'_same_inputs={same_inputs}_'
              f'readout={readout[0].__name__}_'
              f'parameterization={parameterization}_'
              f'pool={pool[0].__name__}',
          'same_inputs': same_inputs,
          'parameterization': parameterization,
          'readout': readout,
          'pool': pool,
          'get': get
      }
                          for pool in [stax.Identity(),
                                       stax.AvgPool((2, 3), (2, 1), 'VALID')]
                          for readout in [stax.Flatten(),
                                          stax.GlobalAvgPool()]
                          for same_inputs in [False]
                          for get in ['ntk']
                          for parameterization in ['ntk', 'standard']))
  def test_conv_local_deep(self, get, pool, same_inputs, readout,
                           parameterization):
    _skip_test()

    key1, key2, key_mc = random.split(random.PRNGKey(1), 3)
    x1 = random.normal(key1, (2, 7, 8, 3))
    x2 = random.normal(key2, (3, 7, 8, 3)) if not same_inputs else None

    def get_nn(conv):
      width = 256
      return stax.serial(
          conv(width, (2, 3), (2, 1), padding='CIRCULAR', W_std=1.5, b_std=0.2,
               parameterization=parameterization),
          pool,
          stax.Erf(),
          conv(width, (3, 1), (1, 2), padding='SAME'),
          stax.Relu(),
          conv(width, (2, 3), (2, 1), padding='VALID', W_std=1.2, b_std=0.3,
               parameterization=parameterization),
          readout,
          stax.Dense(1 if get == 'ntk' else width)
      )

    init_fn, apply_fn, kernel_fn_local = get_nn(stax.ConvLocal)

    k_local = kernel_fn_local(x1, x2, get)

    # Test results for consistency with different diagonalizations.
    for diagonal_batch in [True]:
      for diagonal_spatial in [True, False]:
        kwargs = dict(get=get,
                      diagonal_batch=diagonal_batch,
                      diagonal_spatial=diagonal_spatial)
        with self.subTest(**kwargs):
          k_local_d = kernel_fn_local(x1, x2, **kwargs)
          test_utils.assert_close_matrices(self, k_local, k_local_d, 0.01)

    # Test against CNN-GP diagonal if only flattening is used.
    if pool[0].__name__ == 'Identity' and readout[0].__name__ == 'Flatten':
      _, _, kernel_fn_conv = get_nn(stax.Conv)
      k_conv = kernel_fn_conv(x1, x2, get)
      self.assertAllClose(k_conv, k_local)

    # Test against MC.
    kernel_fn_mc = monte_carlo.monte_carlo_kernel_fn(
        init_fn, apply_fn, key_mc, n_samples=512, device_count=0,
        implementation=2,
        vmap_axes=0
    )
    k_mc = kernel_fn_mc(x1, x2, get)
    test_utils.assert_close_matrices(self, k_mc, k_local, 0.015)

  def test_conv_local_conv(self):
    _skip_test(platforms=('cpu', 'tpu'))

    key1, key2 = random.split(random.PRNGKey(1), 2)
    x1 = np.cos(random.normal(key1, (5, 32, 32, 1)))
    x2 = np.sin(random.normal(key2, (5, 32, 32, 1)))

    width = 128
    local_conv = stax.serial(stax.ConvLocal(width, (3, 2)),
                             stax.AvgPool((2, 3), padding='SAME'),
                             stax.Relu(),
                             stax.ConvLocal(width, (1, 2), padding='SAME'),
                             stax.AvgPool((2, 1), padding='SAME'),
                             stax.Relu(),
                             stax.Conv(width, (3, 3), padding='SAME'),
                             stax.Relu(),
                             stax.Conv(width, (3, 3), padding='SAME'))

    init_fn, apply_fn, kernel_fn = local_conv

    # No projection layer
    k = kernel_fn(x1, x2)
    self.assertEqual(k.diagonal_spatial, False)
    self._test_against_mc(apply_fn, init_fn, k.nngp, x1, x2, 0.03)

    # Top layer flat
    init_fn, apply_fn, kernel_fn = stax.serial(local_conv, stax.Flatten())
    k_jit = jit(lambda x1, x2: kernel_fn(x1, x2))
    k_jit(x2, x1).nngp.block_until_ready()
    time_flat = time.time()
    k = k_jit(x1, x2).nngp.block_until_ready()
    time_flat = time.time() - time_flat
    self._test_against_mc(apply_fn, init_fn, k, x1, x2, 0.03)

    # Top layer pooling
    init_fn, apply_fn, kernel_fn = stax.serial(local_conv, stax.GlobalAvgPool())
    k_jit = jit(lambda x1, x2: kernel_fn(x1, x2))
    k_jit(x2, x1).nngp.block_until_ready()
    time_pool = time.time()
    k = k_jit(x1, x2).nngp.block_until_ready()
    time_pool = time.time() - time_pool
    self.assertLess(time_flat * 5, time_pool)
    self._test_against_mc(apply_fn, init_fn, k, x1, x2, 0.03)

    # Top layer LCN + pooling
    init_fn, apply_fn, kernel_fn = stax.serial(local_conv,
                                               stax.ConvLocal(width, (2, 2),
                                                              padding='SAME'),
                                               stax.GlobalAvgPool())
    k_jit = jit(lambda x1, x2: kernel_fn(x1, x2))
    k_jit(x2, x1).nngp.block_until_ready()
    time_lcn_pool = time.time()
    k = k_jit(x1, x2).nngp.block_until_ready()
    time_lcn_pool = time.time() - time_lcn_pool
    self.assertLess(time_lcn_pool * 5, time_pool)
    self._test_against_mc(apply_fn, init_fn, k, x1, x2, 0.03)

  def test_double_pool(self):
    _skip_test()

    key1, key2 = random.split(random.PRNGKey(1), 2)
    x1 = np.cos(random.normal(key1, (2, 4, 6, 3)))
    x2 = np.sin(random.normal(key2, (3, 4, 6, 3)))

    width = 256
    single_pool = stax.serial(stax.ConvLocal(width, (2, 3),
                                             W_std=2., b_std=0.01),
                              stax.AvgPool((3, 2)))
    init_fn, apply_fn, kernel_fn = stax.serial(single_pool,
                                               stax.Flatten())
    k_single = kernel_fn(x1, x2)
    self._test_against_mc(apply_fn, init_fn, k_single.nngp, x1, x2, 0.05)

    init_fn, apply_fn, kernel_fn = stax.serial(single_pool,
                                               stax.AvgPool((1, 2)),
                                               stax.Flatten())
    k_double = kernel_fn(x1, x2)
    self._test_against_mc(apply_fn, init_fn, k_double.nngp, x1, x2, 0.05)

  def _test_against_mc(self, apply_fn, init_fn, k, x1, x2, tol=0.01, n=256):
    kernel_fn_mc = monte_carlo.monte_carlo_kernel_fn(
        init_fn, apply_fn, random.PRNGKey(2), n_samples=n, device_count=0,
        implementation=2,
        vmap_axes=0
    )
    k_mc = kernel_fn_mc(x1, x2, 'nngp')
    test_utils.assert_close_matrices(self, k_mc, k, tol)


class AutodiffTest(test_utils.NeuralTangentsTestCase):

  @parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name':
              f'{get}-{same_inputs}-{phi.__name__}',
          'get': get,
          'same_inputs': same_inputs,
          'phi': phi,
      }
                          for get in [
                              'ntk',
                              'nngp'
                          ]
                          for same_inputs in [True, False, None]
                          for phi in [
                              stax.Erf,
                              stax.Sin,
                              stax.Gelu,
                              stax.Relu,
                              stax.ElementwiseNumerical
                          ]))
  def test_autodiff(self, get, same_inputs, phi):
    x1 = np.cos(random.normal(random.PRNGKey(1), (3, 1, 2, 3)))
    if same_inputs is None:
      x2 = None
    elif same_inputs is True:
      x2 = x1
    else:
      x2 = np.cos(random.normal(random.PRNGKey(2), (4, 1, 2, 3)))

    name = phi.__name__
    if name == 'LeakyRelu':
      phi = phi(0.1)
    elif name == 'ElementwiseNumerical':
      phi = phi(fn=np.cos, deg=25)
    else:
      phi = phi()

    _, _, kernel_fn = stax.serial(stax.Dense(1, 2., 0.01), phi,
                                  stax.Dense(1, 2., 0.01), phi)

    def k(x1, x2):
      return kernel_fn(x1, x2, get)

    dx1 = random.normal(random.PRNGKey(3), x1.shape) * 0.01
    if same_inputs is None:
      dx2 = None
    else:
      dx2 = random.normal(random.PRNGKey(4), x2.shape) * 0.01

    def dk(x1, x2):
      return jvp(k, (x1, x2), (dx1, dx2))[1]

    def d2k(x1, x2):
      return jvp(dk, (x1, x2), (dx1, dx2))[1]

    _dk = dk(x1, x2)

    if (same_inputs is not False and
        get == 'ntk' and
        ('Relu' in name or 'Abs' in name)):
      # TODO(romann): revisit numerical issues of second derivative of `Relu`
      _d2k = 0
      tol = 0.02
    else:
      _d2k = d2k(x1, x2)
      tol = 2e-3 if name == 'ElementwiseNumerical' else 1e-4

    def assert_close(x, y, tol=3e-5):
      if xla_bridge.get_backend().platform == 'tpu':
        # TODO(romann): understand why TPUs have high errors.
        tol = 0.21
      self.assertLess(
          np.max(np.abs(x - y)) / (np.mean(np.abs(x)) + np.mean(np.abs(y))),
          tol)

    # k(x + dx) ~ k(x) + dk(x) dx + dx^T d2k(x) dx
    assert_close(k(x1 + dx1, None if same_inputs is None else x2 + dx2),
                 k(x1, x2) + _dk + _d2k / 2,
                 tol=tol)

    # d/dx1
    k_fwd_0 = jacfwd(k)(x1, x2)
    k_rev_0 = jacrev(k)(x1, x2)
    assert_close(k_fwd_0, k_rev_0)

    if same_inputs is not None:
      # d/dx2
      k_fwd_1 = jacfwd(k, 1)(x1, x2)
      k_rev_1 = jacrev(k, 1)(x1, x2)
      assert_close(k_fwd_1, k_rev_1)

      # dk(x2, x1)/dx2 = dk(x1, x2)/dx1
      k_fwd_01 = jacfwd(k, 1)(x2, x1)
      k_rev_01 = jacrev(k, 1)(x2, x1)
      assert_close(np.moveaxis(k_fwd_0, (0, 2, 4), (1, 3, 5)), k_fwd_01)
      assert_close(np.moveaxis(k_rev_0, (0, 2, 4), (1, 3, 5)), k_rev_01)

      # dk(x2, x1)/dx1 = dk(x1, x2)/dx2
      k_fwd_10 = jacfwd(k)(x2, x1)
      k_rev_10 = jacrev(k)(x2, x1)
      assert_close(np.moveaxis(k_fwd_1, (0, 2, 4), (1, 3, 5)), k_fwd_10)
      assert_close(np.moveaxis(k_rev_1, (0, 2, 4), (1, 3, 5)), k_rev_10)

  @parameterized.named_parameters(
    jtu.cases_from_list({
        'testcase_name':
            f'{get}-{same_inputs}-{input_type}-{phi.__name__}-{do_jit}',
        'get': get,
        'same_inputs': same_inputs,
        'phi': phi,
        'input_type': input_type,
        'do_jit': do_jit
    }
                        for get in [
                            'ntk',
                            'nngp'
                        ]
                        for do_jit in [True, False]
                        for input_type in ['zeros', 'ones', 'random']
                        for same_inputs in [True, False, None]
                        for phi in [
                            stax.Erf,
                            stax.Abs,
                            stax.Gelu,
                            stax.Relu,
                        ]))
  def test_issue_123(self, get, input_type, same_inputs, phi, do_jit):
    """Tests https://github.com/google/neural-tangents/issues/123."""
    def WideResnetBlock(channels, strides=(1, 1), channel_mismatch=False):
      main = stax.serial(
          phi(),
          stax.Conv(
              channels, (3, 3), strides, padding='SAME',
              parameterization='standard'
          ),
          phi(),
          stax.Conv(channels, (3, 3), padding='SAME',
                    parameterization='standard'),
      )
      shortcut = (
          stax.Identity()
          if not channel_mismatch
          else stax.Conv(
              channels, (3, 3), strides, padding='SAME',
              parameterization='standard'
          )
      )
      return stax.serial(stax.FanOut(2), stax.parallel(main, shortcut),
                         stax.FanInSum())

    def WideResnetGroup(n, channels, strides=(1, 1)):
      blocks = []
      blocks += [WideResnetBlock(channels, strides, channel_mismatch=True)]
      for _ in range(n - 1):
          blocks += [WideResnetBlock(channels, (1, 1))]
      return stax.serial(*blocks)

    def WideResnet(block_size, k, num_classes):
      return stax.serial(
          stax.Conv(16, (3, 3), padding='SAME', parameterization='standard'),
          WideResnetGroup(block_size, int(16 * k)),
          stax.GlobalAvgPool(),
          stax.Dense(num_classes, 1.0, 0.0, parameterization='standard'),
      )

    _, _, kernel_fn = WideResnet(block_size=1, k=1, num_classes=1)

    def get_x(key):
      shape = (1, 8, 8, 3)
      if input_type == 'zeros':
        x = np.zeros(shape)
      elif input_type == 'ones':
        x = np.ones(shape)
      elif input_type == 'random':
        x = random.normal(random.PRNGKey(key), shape)
      else:
        raise ValueError(input_type)
      return x

    x1 = get_x(1)
    if same_inputs is None:
      x2 = None
    elif same_inputs:
      x2 = x1
    else:
      x2 = get_x(2)

    def kernel_scalar(x1, x2):
      return kernel_fn(x1, x2, get)[0, 0]

    if do_jit:
      kernel_scalar = jit(kernel_scalar)

    k1 = kernel_scalar(x1, x2)
    k2 = value_and_grad(kernel_scalar)(x1, x2)[0]
    self.assertAllClose(k1, k2)


if __name__ == '__main__':
  absltest.main()
