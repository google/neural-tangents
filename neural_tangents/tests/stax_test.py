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

from functools import partial
import random as prandom
import itertools
import logging
from jax import ops
from jax import test_util as jtu
from jax.config import config as jax_config
from jax.lib import xla_bridge
import jax.numpy as np
import jax.random as random
from neural_tangents import stax
from neural_tangents.utils import monte_carlo
from neural_tangents.utils import test_utils

jax_config.parse_flags_with_absl()


MODELS = [
    'fc',
    'conv'
]

INPUT_SHAPE = (2, 7, 6, 3)

WIDTHS = [2**11]

N_SAMPLES = 100

RTOL = 0.02

FILTER_SHAPES = [
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
    (1, 1),
    (1, 2),
    (2, 1),
]

ACTIVATIONS = {
    # TODO(romann): investigate poor erf convergence.
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


def _get_inputs(key, is_conv, same_inputs, input_shape, fn=np.cos):
  key, split = random.split(key)
  shape = input_shape if is_conv else (input_shape[0], np.prod(input_shape[1:]))
  x1 = fn(random.normal(key, shape))
  x2 = None if same_inputs else 2 * fn(random.normal(split, shape))

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

    # Select a activation order.
    default_spec = 'NHWC'
    if xla_bridge.get_backend().platform == 'tpu':
      # Keep batch dimension leading for TPU for batching to work.
      specs = ['N' + ''.join(p) for p in itertools.permutations('CHW')]
    else:
      # Keep batch dimension before channel dimension for empirical kernel.
      specs = [''.join(p) for p in itertools.permutations('NCHW')
               if p.index('N') < p.index('C')]
    spec = prandom.choice(specs)
    input_shape = tuple(INPUT_SHAPE[default_spec.index(c)] for c in spec)

    if layer_norm:
      layer_norm = tuple(spec.index(c) for c in layer_norm)

  else:
    # Only `NC` dimension order is supported by empirical kernel.
    spec = 'NC'
    filter_spec = None
    input_shape = INPUT_SHAPE
    if layer_norm:
      layer_norm = prandom.choice([(1,), (-1,)])

  dimension_numbers = (spec, filter_spec, spec)
  logging.warning(f'DIMENSION NUMBERS: {dimension_numbers}')

  fc = partial(
      stax.Dense, W_std=W_std, b_std=b_std, parameterization=parameterization)

  def conv(out_chan): return stax.GeneralConv(
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

  spec = dimension_numbers[-1]

  rate = np.onp.random.uniform(0.5, 0.9)
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
                               spec=spec)
  else:
    pool_or_identity = stax.Identity()
  dropout_or_identity = dropout if use_dropout else stax.Identity()
  layer_norm_or_identity = (stax.Identity() if layer_norm is None
                            else stax.LayerNorm(axis=layer_norm, spec=spec))
  res_unit = stax.serial(pool_or_identity, phi, dropout_or_identity, affine)
  if is_res:
    block = stax.serial(
        affine,
        stax.FanOut(2),
        stax.parallel(stax.Identity(),
                      res_unit),
        stax.FanInSum(),
        layer_norm_or_identity)
  else:
    block = stax.serial(
        affine,
        res_unit,
        layer_norm_or_identity)

  if proj_into_2d == 'FLAT':
    proj_layer = stax.Flatten(spec=spec)
  elif proj_into_2d == 'POOL':
    proj_layer = global_pool_fn(spec=spec)
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
            spec=spec), stax.Flatten(spec=spec))
  else:
    raise ValueError(proj_into_2d)
  readout = stax.serial(proj_layer, fc(1 if is_ntk else width))

  return stax.serial(block, readout), input_shape


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

  pool = pool_fn(filter_shape, strides, padding)

  return stax.serial(
      conv(width), phi, pool, conv(width), phi, global_pool_fn(),
      fc(1 if is_ntk else width)), INPUT_SHAPE


class StaxTest(jtu.JaxTestCase):

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
      } for model in MODELS for width in WIDTHS
                          for phi, phi_name in ACTIVATIONS.items()
                          for same_inputs in [False, True]
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
        raise jtu.SkipTest('Not running CNN models on CPU to save time.')

      if (is_res and is_conv and ((strides is not None and strides != (1, 1)) or
                                  (padding == 'VALID' and filter_shape !=
                                   (1, 1)))):
        raise jtu.SkipTest('Different paths in a residual models need to return'
                           ' outputs of the same shape.')
    elif (filter_shape != FILTER_SHAPES[0] or padding != PADDINGS[0] or
          strides != STRIDES[0] or proj_into_2d != PROJECTIONS[0] or
          use_pooling):
      raise jtu.SkipTest('FC models do not have these parameters.')

    pool_type = 'AVG'
    W_std, b_std = 2.**0.5, 0.5**0.5
    layer_norm = None
    parameterization = 'ntk'
    use_dropout = False

    net = _get_net(W_std, b_std, filter_shape, is_conv, use_pooling, is_res,
                   padding, phi, strides, width, is_ntk, proj_into_2d,
                   pool_type, layer_norm, parameterization, use_dropout)
    self._check_agreement_with_empirical(net, same_inputs, is_conv, use_dropout,
                                         is_ntk, proj_into_2d)

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
                          for same_inputs in [False, True]
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
        raise jtu.SkipTest('Not running CNN models on CPU to save time.')
    elif proj_into_2d != PROJECTIONS[0]:
      raise jtu.SkipTest('FC models do not have these parameters.')

    net = _get_net(W_std, b_std, filter_shape, is_conv, use_pooling, is_res,
                   padding, phi, strides, width, is_ntk, proj_into_2d,
                   pool_type, layer_norm, parameterization, use_dropout)
    self._check_agreement_with_empirical(net, same_inputs, is_conv, use_dropout,
                                         is_ntk, proj_into_2d)

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
      } for model in MODELS
        for width in WIDTHS
        for same_inputs in [False, True]
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
        raise jtu.SkipTest('Not running CNN models on CPU to save time.')
    elif proj_into_2d != PROJECTIONS[0] or layer_norm != LAYER_NORM[0]:
      raise jtu.SkipTest('FC models do not have these parameters.')

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
    self._check_agreement_with_empirical(net, same_inputs, is_conv, use_dropout,
                                         is_ntk, proj_into_2d)

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
      raise jtu.SkipTest('Not running CNN models on CPU to save time.')
    if pool_type == 'SUM' and normalize_edges:
      raise jtu.SkipTest('normalize_edges not applicable to SumPool.')

    net = _get_net_pool(width, is_ntk, pool_type,
                        padding, filter_shape, strides, normalize_edges)
    self._check_agreement_with_empirical(net, same_inputs, is_conv, use_dropout,
                                         is_ntk, proj_into_2d)

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

    self.assertAllClose((out1_stax, out2_stax), (out1_norm, out2_norm), True)

    out_unnorm = np.array([[1., 1., 0.5], [0.5, 0.5, 0.25]]).reshape(
        (1, 2, 3, 1))
    out1_unnormalized = np.broadcast_to(out_unnorm, X1.shape)
    out2_unnormalized = np.broadcast_to(out_unnorm, X2.shape)

    self.assertAllClose((out1_unnormalized, out2_unnormalized), (out1, out2),
                        True)

    ker = kernel_fn(X1, X2)
    ker_norm = kernel_fn_norm(X1, X2)

    self.assertAllClose(np.ones_like(ker_norm.nngp), ker_norm.nngp, True)
    self.assertAllClose(np.ones_like(ker_norm.var1), ker_norm.var1, True)
    self.assertAllClose(np.ones_like(ker_norm.var2), ker_norm.var2, True)

    self.assertEqual(ker_norm.nngp.shape, ker.nngp.shape)
    self.assertEqual(ker_norm.var1.shape, ker.var1.shape)
    self.assertEqual(ker_norm.var2.shape, ker.var2.shape)

    ker_unnorm = np.outer(out_unnorm, out_unnorm).reshape((2, 3, 2, 3))
    ker_unnorm = np.transpose(ker_unnorm, axes=(0, 2, 1, 3))
    nngp = np.broadcast_to(
        ker_unnorm.reshape((1, 1) + ker_unnorm.shape), ker.nngp.shape)
    var1 = np.broadcast_to(np.expand_dims(ker_unnorm, 0), ker.var1.shape)
    var2 = np.broadcast_to(np.expand_dims(ker_unnorm, 0), ker.var2.shape)
    self.assertAllClose((nngp, var1, var2), (ker.nngp, ker.var1, ker.var2),
                        True)

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
                          for padding in PADDINGS for strides in STRIDES
                          for filter_shape in FILTER_SHAPES
                          for is_ntk in [True, False]
                          for use_pooling in [True, False]
                          for proj_into_2d in ['FLAT', 'POOL']))
  def test_dropout(self, model, width, same_inputs, is_ntk, padding, strides,
                   filter_shape, phi, use_pooling, proj_into_2d):
    if xla_bridge.get_backend().platform == 'tpu' and same_inputs:
      raise jtu.SkipTest(
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
        raise jtu.SkipTest('Not running CNN models on CPU to save time.')

      if (is_res and is_conv and ((strides is not None and strides != (1, 1)) or
                                  (padding == 'VALID' and filter_shape !=
                                   (1, 1)))):
        raise jtu.SkipTest('Different paths in a residual models need to return'
                           ' outputs of the same shape.')
    elif (filter_shape != FILTER_SHAPES[0] or padding != PADDINGS[0] or
          strides != STRIDES[0] or proj_into_2d != PROJECTIONS[0] or
          use_pooling):
      raise jtu.SkipTest('FC models do not have these parameters.')

    net = _get_net(W_std, b_std, filter_shape, is_conv, use_pooling, is_res,
                   padding, phi, strides, width, is_ntk, proj_into_2d,
                   pool_type, layer_norm, parameterization, use_dropout)
    self._check_agreement_with_empirical(net, same_inputs, is_conv, use_dropout,
                                         is_ntk, proj_into_2d)

  def _check_agreement_with_empirical(self, net, same_inputs, is_conv,
                                      use_dropout, is_ntk, proj_into_2d):

    (init_fn, apply_fn, kernel_fn), input_shape = net


    num_samples = N_SAMPLES * 5 if use_dropout else N_SAMPLES
    key = random.PRNGKey(1)
    x1, x2 = _get_inputs(key, is_conv, same_inputs, input_shape)

    x1_out_shape, params = init_fn(key, x1.shape)
    if same_inputs:
      assert (x2 is None)
    if x2 is None:
      x2_out_shape = x1_out_shape
    else:
      x2_out_shape, params = init_fn(key, x2.shape)
    del (params)

    def _get_empirical(n_samples, get):
      kernel_fn_empirical = monte_carlo.monte_carlo_kernel_fn(
          init_fn, apply_fn, key, n_samples)
      if same_inputs:
        assert (x2 is None)
      return kernel_fn_empirical(x1, x2, get)

    if proj_into_2d == 'ATTN_PARAM':
      # no analytic kernel available, just test forward/backward pass
      _get_empirical(1, 'ntk' if is_ntk else 'nngp')
    else:
      if is_ntk:
        exact, shape1, shape2 = kernel_fn(x1, x2, ('ntk', 'shape1', 'shape2'))
        empirical = np.reshape(_get_empirical(num_samples, 'ntk'), exact.shape)
      else:
        exact, shape1, shape2 = kernel_fn(x1, x2, ('nngp', 'shape1', 'shape2'))
        empirical = _get_empirical(num_samples, 'nngp')
      test_utils.assert_close_matrices(self, exact, empirical, RTOL)
      self.assertEqual(shape1, x1_out_shape)
      self.assertEqual(shape2, x2_out_shape)

  def test_composition_dense(self):
    rng = random.PRNGKey(0)
    x1 = random.normal(rng, (10, 10))
    x2 = random.normal(rng, (10, 10))

    Block = stax.serial(stax.Dense(256), stax.Relu())

    _, _, ker_fn = Block
    _, _, composed_ker_fn = stax.serial(Block, Block)

    ker_out = ker_fn(ker_fn(x1))
    composed_ker_out = composed_ker_fn(x1)
    self.assertAllClose(ker_out, composed_ker_out, True)

    ker_out = ker_fn(ker_fn(x1, x2))
    composed_ker_out = composed_ker_fn(x1, x2)
    self.assertAllClose(ker_out, composed_ker_out, True)

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name': '_act={}_kernel={}'.format(act, kern),
          'act': act,
          'kernel': kern
      } for act in ['erf', 'relu']
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
      jtu._default_tolerance[np.onp.dtype(np.onp.float64)] = 5e-4
      samples = 100 * N_SAMPLES
    else:
      jtu._default_tolerance[np.onp.dtype(np.onp.float32)] = 5e-2
      jtu._default_tolerance[np.onp.dtype(np.onp.float64)] = 5e-3

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
                        mc[sparse_count:, sparse_count:], True)

  @jtu.parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name': '_avg_pool={}'.format(avg_pool),
          'avg_pool': avg_pool
      } for avg_pool in [True, False]))
  def test_composition_conv(self, avg_pool):
    rng = random.PRNGKey(0)
    x1 = random.normal(rng, (5, 10, 10, 3))
    x2 = random.normal(rng, (5, 10, 10, 3))

    Block = stax.serial(stax.Conv(256, (3, 3)), stax.Relu())
    if avg_pool:
      Readout = stax.serial(stax.GlobalAvgPool(), stax.Dense(10))
      marginalization = 'none'
    else:
      Readout = stax.serial(stax.Flatten(), stax.Dense(10))
      marginalization = 'auto'

    block_ker_fn, readout_ker_fn = Block[2], Readout[2]
    _, _, composed_ker_fn = stax.serial(Block, Readout)

    ker_out = readout_ker_fn(block_ker_fn(x1, marginalization=marginalization))
    composed_ker_out = composed_ker_fn(x1)
    self.assertAllClose(ker_out, composed_ker_out, True)

    if avg_pool:
      with self.assertRaises(ValueError):
        ker_out = readout_ker_fn(block_ker_fn(x1))

    ker_out = readout_ker_fn(
        block_ker_fn(x1, x2, marginalization=marginalization))
    composed_ker_out = composed_ker_fn(x1, x2)
    self.assertAllClose(ker_out, composed_ker_out, True)


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
    _, params = init_fn(key, input_shape=(-1, 7))

    X0_2 = None if same_inputs else random.normal(key, (9, 7))

    for a, b in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
      with self.subTest(a=a, b=b):
        _, apply_ab_relu, kernel_fn_ab_relu = stax.serial(fc, stax.ABRelu(a, b))

        X1_1_relu = (b - a) * apply_relu(params, X0_1 * (-1 if a != 0 else 1))
        X1_1_ab_relu = apply_ab_relu(params, X0_1)
        self.assertAllClose(X1_1_relu, X1_1_ab_relu, True)

        kernels_relu = kernel_fn_relu(X0_1, X0_2)
        kernels_ab_relu = kernel_fn_ab_relu(X0_1, X0_2)
        self.assertAllClose(kernels_relu, kernels_ab_relu, True)

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
        self.assertAllClose(X1_1_id, X1_1_ab_relu, True)

        kernels_id = kernel_fn_id(X0_1 * a, None if X0_2 is None else a * X0_2)
        kernels_ab_relu = kernel_fn_ab_relu(X0_1, X0_2)
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

        _, params = init_fn(key, input_shape=(-1, 7))
        X1_1_leaky_relu = apply_leaky_relu(params, X0_1)
        X1_1_ab_relu = apply_ab_relu(params, X0_1)
        self.assertAllClose(X1_1_leaky_relu, X1_1_ab_relu, True)

        kernels_leaky_relu = kernel_fn_leaky_relu(X0_1, X0_2)
        kernels_ab_relu = kernel_fn_ab_relu(X0_1, X0_2)
        self.assertAllClose(kernels_leaky_relu, kernels_ab_relu, True)

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
    self.assertAllClose(X1_1_abs, X1_1_ab_relu, True)

    kernels_abs = kernel_fn_abs(X0_1, X0_2, ('nngp', 'ntk'))
    kernels_ab_relu = kernel_fn_ab_relu(X0_1, X0_2, ('nngp', 'ntk'))
    self.assertAllClose(kernels_abs, kernels_ab_relu, True)


@jtu.parameterized.parameters([
    {
        'same_inputs': True
    },
    {
        'same_inputs': False
    },
])
class FlattenTest(jtu.JaxTestCase):

  def test_flatten_first(self, same_inputs):
    key = random.PRNGKey(1)
    X0_1 = random.normal(key, (5, 4, 3, 2))
    X0_2 = None if same_inputs else random.normal(key, (3, 4, 3, 2))

    X0_1_flat = np.reshape(X0_1, (X0_1.shape[0], -1))
    X0_2_flat = None if same_inputs else np.reshape(X0_2, (X0_2.shape[0], -1))

    _, _, fc_flat = stax.serial(stax.Dense(10, 2., 0.5),
                                stax.Erf())
    _, _, fc = stax.serial(stax.Flatten(),
                           stax.Dense(10, 2., 0.5),
                           stax.Erf())

    K_flat = fc_flat(X0_1_flat, X0_2_flat)
    K = fc(X0_1, X0_2)
    self.assertAllClose(K_flat, K, True)


class FanInTest(jtu.JaxTestCase):

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
          } for same_inputs in [False, True]
            for axis in [None, 0, 1]
            for n_branches in [1, 2, 3] for get in ['nngp', 'ntk']
            for branch_in in ['dense_before_branch_in',
                              'dense_after_branch_in']))
  def test_fan_in_fc(self, same_inputs, axis, n_branches, get, branch_in):
    if axis in (None, 0) and branch_in == 'dense_after_branch_in':
      raise jtu.SkipTest('`FanInSum` and `FanInConcat(0)` '
                         'require `is_gaussian`.')

    if axis == 1 and branch_in == 'dense_before_branch_in':
      raise jtu.SkipTest('`FanInConcat` on feature axis requires a dense layer'
                         'after concatenation.')

    key = random.PRNGKey(1)
    X0_1 = random.normal(key, (10, 20))
    X0_2 = None if same_inputs else random.normal(key, (8, 20))

    if xla_bridge.get_backend().platform == 'tpu':
      width = 2048
      n_samples = 1024
      tol = 0.02
    else:
      width = 1024
      n_samples = 256
      tol = 0.01

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
    empirical = empirical.reshape(exact.shape)
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
          } for same_inputs in [False, True]
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
      raise jtu.SkipTest('Not running CNNs on CPU to save time.')

    if axis in (None, 0, 1, 2) and branch_in == 'dense_after_branch_in':
      raise jtu.SkipTest('`FanInSum` and `FanInConcat(0/1/2)` '
                         'require `is_gaussian`.')

    if axis == 3 and branch_in == 'dense_before_branch_in':
      raise jtu.SkipTest('`FanInConcat` on feature axis requires a dense layer '
                         'after concatenation.')

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
    empirical = empirical.reshape(exact.shape)
    test_utils.assert_close_matrices(self, empirical, exact, tol)


if __name__ == '__main__':
  jtu.absltest.main()
