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

"""Tests for `neural_tangents/_src/stax/linear.py."""

import itertools
import random as prandom
import string
import time
from absl.testing import absltest
from jax import lax
from jax import jit, vjp
from jax.config import config
from jax import default_backend
import jax.numpy as np
from jax import random
import more_itertools
import neural_tangents as nt
from neural_tangents import stax
from tests import test_utils
from neural_tangents._src.utils import utils
import numpy as onp
from neural_tangents._src.empirical import _DEFAULT_TESTING_NTK_IMPLEMENTATION


config.parse_flags_with_absl()
config.update('jax_numpy_rank_promotion', 'raise')


test_utils.update_test_tolerance()

prandom.seed(1)


@test_utils.product(
    same_inputs=[True, False]
)
class FlattenTest(test_utils.NeuralTangentsTestCase):

  def test_flatten(self, same_inputs):
    key = random.PRNGKey(1)
    X0_1 = random.normal(key, (4, 4, 3, 2))
    X0_2 = None if same_inputs else random.normal(key, (2, 4, 3, 2))

    X0_1_flat = np.reshape(X0_1, (X0_1.shape[0], -1))
    X0_2_flat = None if X0_2 is None else np.reshape(X0_2, (X0_2.shape[0], -1))

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

    kernel_fc_mc = nt.monte_carlo_kernel_fn(
        init_fc, apply_fc, key, n, vmap_axes=0,
        implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION
    )
    kernel_bot_mc = nt.monte_carlo_kernel_fn(
        init_bot, apply_bot, key, n, vmap_axes=0,
        implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION
    )
    kernel_mid_mc = nt.monte_carlo_kernel_fn(
        init_mid, apply_mid, key, n, vmap_axes=0,
        implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION
    )
    kernel_top_mc = nt.monte_carlo_kernel_fn(
        init_top, apply_top, key, n, vmap_axes=0,
        implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION
    )

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


class ConvNDTest(test_utils.NeuralTangentsTestCase):

  @test_utils.product(
      same_inputs=[False],
      n=[0, 1, 2],
      get=['ntk'],
      proj=['flatten', 'pool'],
      use_attn=[True],
      channels_first=[True, False],
      use_dropout=[True],
      use_layernorm=[True],
  )
  def test_conv_nd(
      self,
      same_inputs,
      n,
      get,
      proj,
      use_attn,
      channels_first,
      use_dropout,
      use_layernorm
  ):
    platform = default_backend()
    if platform == 'cpu':
      test_utils.skip_test(self)
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

    kernel_fn_mc = nt.monte_carlo_kernel_fn(
        init_fn, apply_fn, key, n_samples,
        implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION,
        vmap_axes=0
    )

    exact = kernel_fn(X0_1, X0_2, get=get)
    empirical = kernel_fn_mc(X0_1, X0_2, get=get)
    test_utils.assert_close_matrices(self, empirical, exact, tol)


class AttentionTest(test_utils.NeuralTangentsTestCase):

  @test_utils.parameters(
      dict(
          same_inputs=same_inputs,
          get=get,
          n=n,
          linear_scaling=linear_scaling,
          mask_constant=mask_constant,
          p=p,
          mask_axis=mask_axis,
          pos_emb_type=pos_emb_type,
          n_chan_pos_emb=n_chan_pos_emb,
          pos_emb_decay_fn=pos_emb_decay_fn,
          val_pos_emb=val_pos_emb,
          W_pos_emb_std=W_pos_emb_std
      )
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
      for n_chan_pos_emb in (
          [None] if pos_emb_type != 'CONCAT'
          else [None, 512]
      )
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
  )
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
      W_pos_emb_std
  ):
    test_utils.skip_test(self)

    width = 1024
    n_samples = 1024
    tol = 0.05
    key = random.PRNGKey(1)
    n_chan_in = 2
    spatial_shape = (2, 3, 4, 3, 2, 1)[:n]
    mask_axis = [i % (n + 2) for i in mask_axis]

    def get_x0(batch_size):
      x0 = random.normal(key, (batch_size,) + spatial_shape + (n_chan_in,))
      x0 = test_utils.mask(x0, mask_constant, mask_axis, key, p)
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
          W_out_std=0.8,
          W_query_std=0.7,
          W_value_std=1.2,
          b_std=0.5
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

    kernel_fn_mc = nt.monte_carlo_kernel_fn(
        init_fn=init_fn,
        apply_fn=stax.unmask_fn(apply_fn),
        key=key,
        n_samples=n_samples,
        device_count=-1,
        implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION,
        vmap_axes=0
    )

    kernel_fn = jit(kernel_fn, static_argnames='get')
    exact = kernel_fn(X0_1, X0_2, get, mask_constant=mask_constant)

    empirical = kernel_fn_mc(X0_1, X0_2, get=get, mask_constant=mask_constant)
    test_utils.assert_close_matrices(self, empirical, exact, tol, 2.)


class AggregateTest(test_utils.NeuralTangentsTestCase):

  @test_utils.parameters(
      dict(
          get=get,
          readout=readout,
          same_input=same_input,
          activation=activation,
          mask_constant=mask_constant,
          shape=shape,
          batch_axis=batch_axis,
          channel_axis=channel_axis,
          agg_axes=agg_axes,
          do_batch=do_batch,
          implementation=implementation,
          to_dense=to_dense
      )
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
           stax.GlobalAvgPool(
               batch_axis=batch_axis,
               channel_axis=channel_axis)),
      ]
  )
  def test_aggregate(
      self,
      get,
      readout,
      same_input,
      activation,
      mask_constant,
      shape,
      batch_axis,
      channel_axis,
      agg_axes,
      do_batch,
      implementation,
      to_dense
  ):
    if len(shape) > 1:
      test_utils.skip_test(self)

    if implementation == 'SPARSE' and to_dense != 'identity':
      raise absltest.SkipTest('`implementation="SPARSE"` ignores '
                              '`to_dense` argument.')

    if get == 'cov2' and same_input:
      raise absltest.SkipTest('`get="cov2"` only defined for different inputs.')

    if get in ('cov1', 'cov2') and do_batch:
      raise absltest.SkipTest('Batching of empirical kernel does not work for '
                              '`diagonal_axes != ()`.')

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

      if x2 is not None:
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
                         to_dense=dict(
                             identity=lambda p: p,
                             sparse_to_dense=sparse_to_dense
                         )[to_dense],
                         implementation=implementation
                         ),
          readout,
          stax.Dense(output_dims,
                     batch_axis=int(batch_axis > channel_axis),
                     channel_axis=int(batch_axis < channel_axis)))

    init_fn, apply_fn, kernel_fn = get_nn(to_dense, implementation)
    apply_fn = jit(apply_fn)
    kernel_fn = jit(kernel_fn, static_argnames='get')

    if do_batch:
      kernel_fn = nt.batch(kernel_fn, batch_size=2)

    exact = kernel_fn(x1, x2, get,
                      mask_constant=mask_constant,
                      pattern=(pattern1, pattern2))

    rtol = 0.08

    if to_dense == 'sparse_to_dense' or implementation == 'SPARSE':
      init_fn_dense, apply_fn_dense, kernel_fn_dense = get_nn('identity',
                                                              'DENSE')
      apply_fn_dense = jit(apply_fn_dense)
      kernel_fn_dense = jit(kernel_fn_dense, static_argnames='get')

      pattern1_dense = sparse_to_dense(pattern1)
      pattern2_dense = sparse_to_dense(pattern2)

      # Test parameters agreement
      key, _ = random.split(key, 2)
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
    kernel_mc_fn = nt.monte_carlo_kernel_fn(
        init_fn=init_fn,
        apply_fn=stax.unmask_fn(apply_fn),
        key=random.PRNGKey(10),
        n_samples=2**6,
        batch_size=2 if (default_backend() == 'tpu' and batch_axis == 0) else 0,
        device_count=-1 if batch_axis == 0 else 0,
        implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION,
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

    test_utils.assert_close_matrices(self, exact, empirical, rtol, 0.2)


class ConvTransposeTest(test_utils.NeuralTangentsTestCase):

  @test_utils.product(
      padding=['CIRCULAR', 'SAME', 'VALID'],
      same_inputs=[False],
      filter_shape=[2, 3, 4],
      strides=[2, 3, 4],
      size=[2, 3, 4],
      diagonal_batch=[True],
      diagonal_spatial=[True, False],
  )
  def test_conv_transpose(
      self,
      same_inputs,
      padding,
      filter_shape,
      strides,
      size,
      diagonal_batch,
      diagonal_spatial
  ):
    if size > 2:
      test_utils.skip_test(self)

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

    kernel_fn_mc = nt.monte_carlo_kernel_fn(
        init_fn, apply_fn, key, n_samples, diagonal_axes=diagonal_axes,
        device_count=0,
        implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION,
        vmap_axes=0
    )
    k_mc = kernel_fn_mc(x1, None if diagonal_batch else x2, 'nngp')

    test_utils.assert_close_matrices(self, k_mc, k, tol)

  @classmethod
  def _conv_transpose_circular_via_grad(
      cls,
      lhs,
      params,
      strides,
      padding,
      dimension_numbers
  ):
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
  def _conv_transpose_circular(
      cls,
      lhs,
      params,
      strides,
      padding,
      dimension_numbers
  ):
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

  @test_utils.product(
      filter_shape=[1, 2, 3, 4],
      strides=[1, 2, 3, 4],
      size=[1, 2, 3, 4]
  )
  def test_conv_transpose_circular(self, size, filter_shape, strides):
    if size > 2:
      test_utils.skip_test(self)

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

  @test_utils.parameters(
      dict(
          same_inputs=same_inputs,
          n=n,
          batch_dims=batch_dims,
          contracting_dims=contracting_dims,
          b_dims=b_dims,
          c_dims=c_dims,
          r_permutation=r_permutation,
          channel_axis=channel_axis,
          batch_axis=batch_axis,
          is_rhs=is_rhs,
          diagonal_spatial=diagonal_spatial,
          diagonal_batch=diagonal_batch
      )
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
  def test_dot_general(
      self,
      same_inputs,
      n,
      batch_dims,
      contracting_dims,
      c_dims,
      b_dims,
      r_permutation,
      channel_axis,
      is_rhs,
      diagonal_spatial,
      diagonal_batch,
      batch_axis
  ):
    if n != 2:
      test_utils.skip_test(self)

    if default_backend() == 'tpu':
      atol = 1.
    else:
      atol = 0.1

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

        kernel_fn_mc = nt.monte_carlo_kernel_fn(
            init_fn=init_fn,
            apply_fn=stax.unmask_fn(apply_fn),
            key=key1,
            n_samples=1,
            trace_axes=(out_c_axis,),
            diagonal_axes=get_diagonal_axes(),
            device_count=-1 if (get == 'nngp' and
                                batch_axis == out_b_axis == 0 and
                                0 not in c_dims + b_dims) else 0,
            implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION,
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
              self, get_empirical(get), getattr(exact, get), 0.01, atol)

  @test_utils.product(
      same_inputs=[False, True],
      get=['ntk'],
      do_pool=[True, False],
      n=[3, 4],
      is_rhs=[False, True],
      dot_first=[True, False]
  )
  def test_dot_general_nn(
      self,
      same_inputs,
      get,
      n,
      is_rhs,
      do_pool,
      dot_first
  ):
    if n != 2:
      test_utils.skip_test(self)

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

    kernel_fn_mc = nt.monte_carlo_kernel_fn(
        init_fn=init_fn,
        apply_fn=stax.unmask_fn(apply_fn),
        key=key1,
        n_samples=n_samples,
        trace_axes=(int(out_c_axis > out_b_axis) if do_pool else 1,),
        device_count=0,
        implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION,
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

  @test_utils.parameters(
      dict(
          same_inputs=same_inputs,
          n=n,
          channel_axis=channel_axis,
          batch_axis=batch_axis,
          diagonal_spatial=diagonal_spatial,
          diagonal_batch=diagonal_batch,
          method=method,
          antialias=antialias,
          precision=precision,
          shape=shape
      )
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
  def test_image_resize(
      self,
      same_inputs,
      n,
      channel_axis,
      diagonal_spatial,
      diagonal_batch,
      batch_axis,
      method,
      antialias,
      precision,
      shape
  ):
    if n > 2:
      test_utils.skip_test(self)

    n_b1, n_b2 = 2, 4
    n_c = 1
    key1, key2, _ = random.split(random.PRNGKey(1), 3)

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

        kernel_fn_mc = nt.monte_carlo_kernel_fn(
            init_fn=init_fn,
            apply_fn=stax.unmask_fn(apply_fn),
            key=key1,
            n_samples=1,
            trace_axes=(channel_axis,),
            diagonal_axes=get_diagonal_axes(),
            device_count=-1 if (get == 'nngp' and
                                batch_axis == 0 and
                                shape[batch_axis] == -1) else 0,
            implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION,
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
          tol = 1e-2 if default_backend() == 'tpu' else 1e-5
          test_utils.assert_close_matrices(
              self, get_empirical(get), getattr(exact, get), tol)

  @test_utils.product(
      same_inputs=[False, True],
      get=['ntk'],
      do_pool=[True, False],
      n=[3],
      bottom_layer=['resize', 'conv', 'relu'],
      method=['linear', 'nearest'],
      shape=[
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
  def test_image_resize_nn(
      self,
      same_inputs,
      get,
      n,
      do_pool,
      bottom_layer,
      method,
      shape
  ):
    if n != 2:
      test_utils.skip_test(self)

    width = 2**7
    n_samples = 2**7
    tol = 0.03
    key1, key2, _ = random.split(random.PRNGKey(1), 3)

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

    kernel_fn_mc = nt.monte_carlo_kernel_fn(
        init_fn=init_fn,
        apply_fn=stax.unmask_fn(apply_fn),
        key=key1,
        n_samples=n_samples,
        device_count=0,
        implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION,
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

  @test_utils.product(
      diagonal_spatial=[True, False]
  )
  def test_whitened_inputs(self, diagonal_spatial):
    test_utils.skip_test(self)

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

  @test_utils.product(
      padding=['SAME', 'VALID', 'CIRCULAR'],
      same_inputs=[False],
      filter_shape=[2, 3],
      strides=[1, 2],
      size=[2, 3],
      diagonal_batch=[True],
      diagonal_spatial=[True, False],
      get=['cov1', 'nngp', 'ntk'],
      parameterization=['standard', 'ntk']
  )
  def test_conv_local(
      self,
      same_inputs,
      padding,
      filter_shape,
      strides,
      size,
      diagonal_batch,
      diagonal_spatial,
      get,
      parameterization
  ):
    test_utils.skip_test(self)

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

    kernel_fn_mc = nt.monte_carlo_kernel_fn(
        init_fn, apply_fn, key_mc, n_samples=512, diagonal_axes=diagonal_axes,
        device_count=0,
        implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION,
        vmap_axes=0
    )
    k_mc = kernel_fn_mc(x1, None if get == 'cov1' else x2,
                        'nngp' if get == 'cov1' else get)
    test_utils.assert_close_matrices(self, k_mc, getattr(k, get), 0.011, 1.)

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

    tol = 0.005 if default_backend() == 'tpu' else 0.001
    self.assertAllClose(k_conv, k, atol=tol, rtol=tol)

  @test_utils.product(
      pool=[
          stax.Identity(),
          stax.AvgPool((2, 3), (2, 1), 'VALID')
      ],
      readout=[
          stax.Flatten(),
          stax.GlobalAvgPool()
      ],
      same_inputs=[False],
      get=['ntk'],
      parameterization=['ntk', 'standard']
  )
  def test_conv_local_deep(
      self,
      get,
      pool,
      same_inputs,
      readout,
      parameterization
  ):
    test_utils.skip_test(self)

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
    kernel_fn_mc = nt.monte_carlo_kernel_fn(
        init_fn, apply_fn, key_mc, n_samples=512, device_count=0,
        implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION,
        vmap_axes=0
    )
    k_mc = kernel_fn_mc(x1, x2, get)
    test_utils.assert_close_matrices(self, k_mc, k_local, 0.015, 1.)

  def test_conv_local_conv(self):
    test_utils.skip_test(self, platforms=('cpu', 'tpu'))

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
    test_utils.skip_test(self)

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
    kernel_fn_mc = nt.monte_carlo_kernel_fn(
        init_fn, apply_fn, random.PRNGKey(2), n_samples=n, device_count=0,
        implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION,
        vmap_axes=0
    )
    k_mc = kernel_fn_mc(x1, x2, 'nngp')
    test_utils.assert_close_matrices(self, k_mc, k, tol)


class IndexTest(test_utils.NeuralTangentsTestCase):

  @test_utils.product(
      same_inputs=[
          True,
          False
      ],
      get=[
          'nngp',
          'ntk',
          'cov1',
          'cov2',
      ],
      index_layer=[
          0,
          1,
          2,
          3
      ],
      mask_constant=[
          None,
          10.
      ],
      idx=[
          stax.Slice[0],
          stax.Slice[-1],
          stax.Slice[:],
          stax.Slice[:, 0],
          stax.Slice[:, -1],
          stax.Slice[:, -3:],
          stax.Slice[:, :],
          stax.Slice[::2],
          stax.Slice[...],
          stax.Slice[0, ...],
          stax.Slice[1:2, ...],
          stax.Slice[0:2, ...],
          stax.Slice[:, ::-2, ...],
          stax.Slice[::2, ::-2, 0, ...],
          stax.Slice[..., 1],
          stax.Slice[..., :2],
          stax.Slice[::2, 1, ...],
          stax.Slice[:, 1, -1, :],
          stax.Slice[..., 1::2],
          stax.Slice[:3, 1, 2],
          stax.Slice[:2, :2, :2],
          stax.Slice[..., ::2],
          stax.Slice[1:2:-1, 1, 2],
          stax.Slice[:, 0, :],
      ],
      readout=[
          stax.GlobalAvgPool,
          stax.Flatten,
      ]
  )
  def test_index(
      self,
      same_inputs,
      get,
      index_layer,
      mask_constant,
      idx,
      readout,
  ):
    if index_layer == 3 and isinstance(idx, tuple) and len(idx) > 2:
      raise absltest.SkipTest(f'Readout outputs have only 2 dimensions, but '
                              f'the index has {len(idx)}.')

    if get == 'cov2' and same_inputs:
      raise absltest.SkipTest('cov2 is None when x2 is None.')

    width = 2**7
    n_samples = 2**7
    tol = 0.05
    key1, key2, key_mc = random.split(random.PRNGKey(1), 3)

    x1 = np.cos(random.normal(key1, [6, 3, 4, 5]))
    if mask_constant is not None:
      mask1 = random.bernoulli(key1, p=0.2, shape=x1.shape)
      x1 = np.where(mask1, mask_constant, x1)

    if same_inputs:
      x2 = None
    else:
      x2 = np.cos(random.normal(key2, [7, 3, 4, 5]))
      if mask_constant is not None:
        mask2 = random.bernoulli(key2, p=0.1, shape=x2.shape)
        x2 = np.where(mask2, mask_constant, x2)

    canonical_idx = utils.canonicalize_idx(
        idx=idx,
        ndim=x1.ndim if index_layer != 3 else 2
    )

    filter_shape = (2, 3)
    if index_layer == 0:
      for i, s in enumerate(canonical_idx):
        if isinstance(s, int) and i in (1, 2):
          filter_shape = filter_shape[:-1]

    layers = [
        stax.Conv(width, filter_shape, padding='SAME'),
        stax.Relu(),
        readout(),
        stax.Dense(1 if get == 'ntk' else width)
    ]

    layers.insert(index_layer, stax.Index(idx=idx))
    init_fn, apply_fn, kernel_fn = stax.serial(*layers)

    def get_exact():
      return kernel_fn(x1, x2, get, mask_constant=mask_constant)

    if isinstance(canonical_idx[0], int) or canonical_idx[-1] != slice(None):
      # Unsupported integer indexing into batch axis, or any indexing into
      # the channel axis.
      self.assertRaises(NotImplementedError, get_exact)

    else:
      exact = get_exact()

      if get in ('cov1', 'cov2'):
        diagonal_axes = (0,)
        get_e = 'nngp'
        if get == 'cov1':
          x1_e, x2_e = x1, None
        elif get == 'cov2':
          x1_e, x2_e = x2, None
      else:
        diagonal_axes = ()
        x1_e, x2_e = x1, x2
        get_e = get

      kernel_fn_mc = nt.monte_carlo_kernel_fn(
          init_fn=init_fn,
          apply_fn=stax.unmask_fn(apply_fn),
          key=key_mc,
          n_samples=n_samples,
          device_count=0,
          diagonal_axes=diagonal_axes,
          implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION,
      )
      empirical = kernel_fn_mc(x1_e, x2_e, get_e, mask_constant=mask_constant)

      test_utils.assert_close_matrices(self, empirical, exact, tol)


if __name__ == '__main__':
  absltest.main()
