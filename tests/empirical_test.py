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

"""Tests for `neural_tangents/_src/empirical.py`."""

from functools import partial
import logging
import operator
from typing import Any, Callable, Sequence, Tuple, Optional, Dict, List
from absl.testing import absltest
from flax import linen as nn
import jax
from jax import jacobian, lax, remat
from jax import jit, tree_map
from jax import random
from jax.config import config
import jax.numpy as np
from jax.tree_util import tree_reduce
import neural_tangents as nt
from neural_tangents import stax
from neural_tangents._src.utils import utils
from tests import test_utils
import numpy as onp


config.parse_flags_with_absl()
config.update('jax_numpy_rank_promotion', 'raise')


TAYLOR_MATRIX_SHAPES = [(3, 3), (4, 4)]
TAYLOR_RANDOM_SAMPLES = 10

FLAT = 'FLAT'
POOLING = 'POOLING'
CONV = 'CONV'

TRAIN_SHAPES = [(4, 4), (4, 8), (8, 8), (6, 4, 4, 3), (4, 4, 4, 3),
                (4, 4, 4, 3)]
TEST_SHAPES = [(2, 4), (6, 8), (16, 8), (2, 4, 4, 3), (2, 4, 4, 3),
               (2, 4, 4, 3)]
NETWORK = [FLAT, FLAT, FLAT, FLAT, POOLING, CONV]
OUTPUT_LOGITS = [1, 2, 3]

CONVOLUTION_CHANNELS = 2
test_utils.update_test_tolerance()


def _build_network(input_shape, network, out_logits):
  if len(input_shape) == 1:
    assert network == FLAT
    return stax.Dense(out_logits, W_std=2.0, b_std=0.5)
  elif len(input_shape) == 3:
    if network == POOLING:
      return stax.serial(
          stax.Conv(CONVOLUTION_CHANNELS, (3, 3), W_std=2.0, b_std=0.05),
          stax.GlobalAvgPool(), stax.Dense(out_logits, W_std=2.0, b_std=0.5))
    elif network == CONV:
      return stax.serial(
          stax.Conv(CONVOLUTION_CHANNELS, (1, 2), W_std=1.5, b_std=0.1),
          stax.Relu(),
          stax.Conv(CONVOLUTION_CHANNELS, (3, 2), W_std=2.0, b_std=0.05),
      )
    elif network == FLAT:
      return stax.serial(
          stax.Conv(CONVOLUTION_CHANNELS, (3, 3), W_std=2.0, b_std=0.05),
          stax.Flatten(), stax.Dense(out_logits, W_std=2.0, b_std=0.5))
    else:
      raise ValueError('Unexpected network type found: {}'.format(network))
  else:
    raise ValueError('Expected flat or image test input.')


def _kernel_fns(key,
                input_shape,
                network,
                out_logits,
                diagonal_axes,
                trace_axes,
                vmap_axes=None):
  init_fn, f, _ = _build_network(input_shape, network, out_logits)
  _, params = init_fn(key, (-1,) + input_shape)
  kwargs = dict(
      f=f,
      trace_axes=trace_axes,
      diagonal_axes=diagonal_axes,
  )

  ntk_fns = {
      i: partial(jit(nt.empirical_ntk_fn(
          **kwargs,
          vmap_axes=vmap_axes,
          implementation=i)),
                 params=params)
      for i in nt.NtkImplementation
  }

  nngp_kernel_fn = partial(jit(nt.empirical_nngp_fn(**kwargs)),
                           params=params)

  return nngp_kernel_fn, ntk_fns


KERNELS = {}
for o in OUTPUT_LOGITS:
  KERNELS['empirical_logits_{}'.format(o)] = partial(_kernel_fns, out_logits=o)


class EmpiricalTest(test_utils.NeuralTangentsTestCase):

  # We use a three layer deep linear network for testing.
  @classmethod
  def _f(cls, x, params, do_alter, do_shift_x=True):
    w1, w2, b = params
    if do_alter:
      b *= 2.
      w1 += 5.
      w2 /= 0.9
    if do_shift_x:
      x = x * 2 + 1.
    return ({'list': [
        {
            'quadratic': 0.5 * np.dot(np.dot(x.T, w1), x) + np.dot(w2, x) + b,
            'linear': np.dot(w1, x)
        },
        w2
    ]},)

  @classmethod
  def _f_lin_exact(cls, x0, x, params, do_alter, do_shift_x=True):
    w1, w2, b = params
    f0 = EmpiricalTest._f(x0, params, do_alter, do_shift_x)
    if do_shift_x:
      x0 = x0 * 2 + 1.
      x = x * 2 + 1.
    dx = x - x0
    if do_alter:
      b *= 2.
      w1 += 5.
      w2 /= 0.9
    return tree_map(
        operator.add,
        f0,
        ({'list': [
            {
                'quadratic': np.dot(np.dot(x0.T, w1) + w2, dx),
                'linear': np.dot(w1, dx)
            },
            0.
        ]},)
    )

  @classmethod
  def _get_init_data(cls, shape):
    key = random.PRNGKey(0)
    key, s1, s2, s3, = random.split(key, 4)
    w1 = random.normal(s1, shape)
    w1 = 0.5 * (w1 + w1.T)
    w2 = random.normal(s2, shape)
    b = random.normal(s3, (1,) * (len(shape) - 1) + (shape[-1],))
    params = (w1, w2, b)
    key, split = random.split(key)
    x0 = random.normal(split, (shape[-1], 1))
    return key, params, x0

  @test_utils.product(
      shape=TAYLOR_MATRIX_SHAPES
  )
  def testLinearization(self, shape):
    key, params, x0 = self._get_init_data(shape)

    f_lin = nt.linearize(EmpiricalTest._f, x0)

    for _ in range(TAYLOR_RANDOM_SAMPLES):
      for do_alter in [True, False]:
        for do_shift_x in [True, False]:
          key, split = random.split(key)
          x = random.normal(split, (shape[-1], 1))
          self.assertAllClose(
              EmpiricalTest._f_lin_exact(
                  x0, x, params, do_alter, do_shift_x=do_shift_x),
              f_lin(x, params, do_alter, do_shift_x=do_shift_x))

  @test_utils.product(
      shape=TAYLOR_MATRIX_SHAPES
  )
  def testTaylorExpansion(self, shape):

    def f_2_exact(x0, x, params, do_alter, do_shift_x=True):
      w1, w2, b = params
      f_lin = EmpiricalTest._f_lin_exact(x0, x, params, do_alter, do_shift_x)
      if do_shift_x:
        x0 = x0 * 2 + 1.
        x = x * 2 + 1.
      if do_alter:
        b *= 2.
        w1 += 5.
        w2 /= 0.9
      dx = x - x0
      return tree_map(
          operator.add,
          f_lin,
          ({'list': [
              {
                  'quadratic': 0.5 * np.dot(np.dot(dx.T, w1), dx),
                  'linear': 0.
              },
              0.
          ]},)
      )

    key, params, x0 = self._get_init_data(shape)

    f_lin = nt.taylor_expand(EmpiricalTest._f, x0, 1)
    f_2 = nt.taylor_expand(EmpiricalTest._f, x0, 2)

    for _ in range(TAYLOR_RANDOM_SAMPLES):
      for do_alter in [True, False]:
        for do_shift_x in [True, False]:
          key, split = random.split(key)
          x = random.normal(split, (shape[-1], 1))
          self.assertAllClose(
              EmpiricalTest._f_lin_exact(x0, x, params, do_alter,
                                         do_shift_x=do_shift_x),
              f_lin(x, params, do_alter, do_shift_x=do_shift_x))
          self.assertAllClose(f_2_exact(x0, x, params, do_alter,
                                        do_shift_x=do_shift_x),
                              f_2(x, params, do_alter, do_shift_x=do_shift_x))

  def _compare_kernels(self, x1, x2, ntk_fns, ntk_fns_vmapped, nngp_fn):
    nngp = nngp_fn(x1, x2)

    ntks = {i: ntk_fns[i](x1, x2) for i in ntk_fns}
    ntks_vmapped = {i: ntk_fns_vmapped[i](x1, x2) for i in ntk_fns_vmapped}

    ntk_ref = ntks[nt.NtkImplementation.JACOBIAN_CONTRACTION]

    tree_map(lambda x, y: self.assertEqual(x.shape, y.shape), nngp, ntk_ref)

    for i, ntk in ntks.items():
      self.assertAllClose(ntk_ref, ntk, err_msg=f'{i} impl. fails.')

    for i, ntk in ntks_vmapped.items():
      self.assertAllClose(ntk_ref, ntk, err_msg=f'{i} vmapped impl. fails.')

  @test_utils.product(
      train_test_network=list(zip(TRAIN_SHAPES, TEST_SHAPES, NETWORK)),
      kernel_type=list(KERNELS.keys())
  )
  def testNTKAgainstDirect(self, train_test_network, kernel_type):
    kernel_fn = KERNELS[kernel_type]
    train_shape, test_shape, network = train_test_network
    key = random.PRNGKey(0)
    key, self_split, other_split = random.split(key, 3)
    x1 = random.normal(self_split, train_shape)
    x2 = random.normal(other_split, test_shape)

    nngp_fn, ntk_fns = kernel_fn(
        key,
        train_shape[1:],
        network,
        diagonal_axes=(),
        trace_axes=()
    )

    _, ntk_fns_vmapped = kernel_fn(
        key,
        train_shape[1:],
        network,
        diagonal_axes=(),
        trace_axes=(),
        vmap_axes=0
    )

    self._compare_kernels(x1, None, ntk_fns, ntk_fns_vmapped, nngp_fn)
    self._compare_kernels(x1, x2, ntk_fns, ntk_fns_vmapped, nngp_fn)

  @test_utils.product(
      diagonal_axes=[
          (),
          (0,),
          (0, 1),
          (0, 1, 2),
          (0, 1, 2, 3),
          (-1,),
          (-2,),
          (0, -1),
          (1, -2),
          (2, 3),
          (3, 0, 2)
      ],
      trace_axes=[
          (),
          (0,),
          (0, 1),
          (-1,),
          (1,),
          (0, -1),
          (-1, -2),
          (0, 1, 2, 3),
          (3, 1, 2, 0),
          (1, 2, 3),
          (-3, -2),
          (-3, -1),
          (-2, -4),
          (2, 0, -1)
      ]
  )
  def testAxes(self, diagonal_axes, trace_axes):
    key = random.PRNGKey(0)
    key, self_split, other_split = random.split(key, 3)
    x1 = random.normal(self_split, (4, 5, 6, 3))
    x2 = random.normal(other_split, (2, 5, 6, 3))

    _diagonal_axes = tuple(d % x1.ndim for d in diagonal_axes)
    _trace_axes = tuple(t % x1.ndim for t in trace_axes)

    if any(d == c for d in _diagonal_axes for c in _trace_axes):
      raise absltest.SkipTest(
          'diagonal axes must be different from channel axes.')

    get_kernel_fns = KERNELS['empirical_logits_3']
    kwargs = dict(
        key=key,
        input_shape=(5, 6, 3),
        network=CONV,
        diagonal_axes=diagonal_axes,
        trace_axes=trace_axes
    )

    nngp_fn, ntk_fns = get_kernel_fns(**kwargs)
    _, ntk_fns_vmapped = get_kernel_fns(**kwargs, vmap_axes=0)

    self._compare_kernels(x1, None, ntk_fns, ntk_fns_vmapped, nngp_fn)
    if 0 not in _trace_axes and 0 not in _diagonal_axes:
      self._compare_kernels(x1, x2, ntk_fns, ntk_fns_vmapped, nngp_fn)

  @test_utils.product(
      same_inputs=[True, False]
  )
  def test_parallel_in_out(self, same_inputs):
    rng = random.PRNGKey(0)
    input_key1, input_key2, net_key = random.split(rng, 3)

    x1_1, x1_2 = np.split(random.normal(input_key1, (3, 21)), (10,), axis=1)
    x2_1, x2_2 = np.split(random.normal(input_key2, (4, 21)), (10,), axis=1)

    x1 = (x1_1, x1_2)
    x2 = (x2_1, x2_2) if not same_inputs else None

    def layer(N_out):
      return stax.parallel(stax.Dense(N_out), stax.Dense(N_out + 1))

    init_fn, apply_fn, _ = stax.serial(layer(1024), layer(1))

    _, params = init_fn(net_key, (x1_1.shape, x1_2.shape))

    ntk_fns = {
        i: jit(
            partial(
                nt.empirical_ntk_fn(
                    apply_fn,
                    implementation=i),
                params=params))
        for i in nt.NtkImplementation
    }

    ntk_fns_vmapped = {
        i: jit(
            partial(
                nt.empirical_ntk_fn(
                    apply_fn,
                    implementation=i,
                    vmap_axes=(0, 0)),
                params=params))
        for i in nt.NtkImplementation
    }

    nngp_fn = jit(partial(nt.empirical_nngp_fn(apply_fn), params=params))
    nngp = nngp_fn(x1, x2)
    self.assertEqual(len(nngp), 2)
    self.assertEqual(nngp[0].shape, (3, 3 if same_inputs else 4))
    self.assertEqual(nngp[1].shape, (3, 3 if same_inputs else 4))
    self._compare_kernels(x1, x2, ntk_fns, ntk_fns_vmapped, nngp_fn)

  @test_utils.product(
      same_inputs=[True, False]
  )
  def test_parallel_nested(self, same_inputs):
    rng = random.PRNGKey(0)
    input_key1, input_key2, net_key = random.split(rng, 3)

    x1_1, x1_2, x1_3 = np.split(random.normal(input_key1, (3, 33)),
                                (10, 21), axis=1)
    x2_1, x2_2, x2_3 = np.split(random.normal(input_key2, (4, 33)),
                                (10, 21), axis=1)

    x1 = ([x1_1, x1_2], x1_3)
    x2 = ([x2_1, x2_2], x2_3) if not same_inputs else None

    def layer(N_out):
      return stax.parallel(stax.parallel(stax.Dense(N_out),
                                         stax.Dense(N_out + 1)),
                           stax.Dense(N_out + 2))

    init_fn, apply_fn, _ = stax.serial(layer(1024), layer(1))

    _, params = init_fn(net_key, tree_map(np.shape, x1))

    ntk_fns = {
        i: jit(nt.empirical_ntk_fn(apply_fn, implementation=i))
        for i in nt.NtkImplementation
    }

    ntk_fns_vmapped = {
        i: jit(nt.empirical_ntk_fn(
            apply_fn,
            implementation=i,
            vmap_axes=([0, 0], 0)
        ))
        for i in nt.NtkImplementation
    }

    ntks = {i: ntk_fns[i](x1, x2, params) for i in ntk_fns}
    ntks_vmapped = {i: ntk_fns_vmapped[i](x1, x2, params)
                    for i in ntk_fns_vmapped}

    ntk_ref = ntks[nt.NtkImplementation.JACOBIAN_CONTRACTION]

    for i, ntk in ntks.items():
      self.assertAllClose(ntk_ref, ntk, err_msg=f'{i} impl. fails.')

    for i, ntk in ntks_vmapped.items():
      self.assertAllClose(ntk_ref, ntk, err_msg=f'{i} vmapped impl. fails.')

    nngp_kernel_fn = jit(nt.empirical_nngp_fn(apply_fn))
    nngp = nngp_kernel_fn(x1, x2, params)

    self.assertEqual(len(nngp), 2)
    nngp_shape = (3, 3 if same_inputs else 4)
    self.assertEqual(nngp[0][0].shape, nngp_shape)
    self.assertEqual(nngp[0][1].shape, nngp_shape)
    self.assertEqual(nngp[1].shape, nngp_shape)

  @test_utils.product(
      same_inputs=[True, False, None],
      in_dict=[True, False],
      out_dict=[True, False]
  )
  def test_vmap_axes(self, same_inputs, out_dict, in_dict):
    n1, n2 = 3, 4
    c1, c2, c3 = 9, 5, 7
    h2, h3, w3 = 6, 8, 2

    def get_x(n, k):
      k1, k2, k3 = random.split(k, 3)
      x1 = random.normal(k1, (n, c1))
      x2 = random.normal(k2, (h2, n, c2))
      x3 = random.normal(k3, (c3, w3, n, h3))
      x = [(x1, x2), x3]
      return x

    x1 = get_x(n1, random.PRNGKey(1))
    p1 = random.normal(random.PRNGKey(5), (n1, h2, h2))

    if same_inputs is None:
      x2 = None
      p2 = p1

    elif same_inputs is False:
      x2 = get_x(n2, random.PRNGKey(2))
      p2 = random.normal(random.PRNGKey(6), (n2, h2, h2))

    elif same_inputs is True:
      x2 = [(None, None), None]
      p2 = p1

    else:
      raise ValueError(same_inputs)

    init_fn, apply_fn_, _ = stax.serial(
        stax.parallel(
            stax.parallel(
                stax.serial(stax.Dense(4, 2., 0.1),
                            stax.Relu(),
                            stax.Dense(3, 1., 0.15)),  # 1
                stax.serial(stax.Conv(7, (2,), padding='SAME',
                                      dimension_numbers=('HNC', 'OIH', 'NHC')),
                            stax.Erf(),
                            stax.Aggregate(1, 0, -1),
                            stax.GlobalAvgPool(),
                            stax.Dense(3, 0.5, 0.2)),  # 2
            ),
            stax.serial(
                stax.Conv(5, (2, 3), padding='SAME',
                          dimension_numbers=('CWNH', 'IOHW', 'HWCN')),
                stax.Sin(),
            )  # 3
        ),
        stax.parallel(
            stax.FanInSum(),
            stax.Conv(2, (2, 1), dimension_numbers=('HWCN', 'OIHW', 'HNWC'))
        )
    )

    _, params = init_fn(random.PRNGKey(3), tree_map(np.shape, x1))

    in_axes = [(0, 1), 2]
    out_axes = [-2, -3]

    def nttree_to_pytree_in(x):
      if x is None:
        return x
      return {'x1_x2': (x[0][0], x[0][1]), 'x3': (None, x[1],)}

    def pytree_to_nttree_in(x):
      if x is None:
        return x
      return [(x['x1_x2'][0], x['x1_x2'][1]), x['x3'][1]]

    def nttree_to_pytree_out(x):
      if x is None:
        return None
      return {'outs': [{'out_1': x[0]}, (x[1], None)]}

    if in_dict:
      x1 = nttree_to_pytree_in(x1)
      x2 = nttree_to_pytree_in(x2)
      in_axes = nttree_to_pytree_in(in_axes)

    if out_dict:
      out_axes = nttree_to_pytree_out(out_axes)

    def apply_fn(params, x, **kwargs):
      if in_dict:
        x = pytree_to_nttree_in(x)

      out = apply_fn_(params, x, **kwargs)
      if out_dict:
        out = nttree_to_pytree_out(out)
      return out

    ntk_fns = {
        i: jit(nt.empirical_ntk_fn(apply_fn, implementation=i))
        for i in nt.NtkImplementation
    }

    ntk_fns_vmapped = {
        i: jit(nt.empirical_ntk_fn(
            apply_fn,
            implementation=i,
            vmap_axes=(in_axes, out_axes, dict(pattern=0))
        ))
        for i in nt.NtkImplementation
    }

    ntks = {i: ntk_fns[i](x1, x2, params, pattern=(p1, p2))
            for i in ntk_fns}
    ntks_vmapped = {i: ntk_fns_vmapped[i](x1, x2, params, pattern=(p1, p2))
                    for i in ntk_fns_vmapped}

    ntk_ref = ntks[nt.NtkImplementation.JACOBIAN_CONTRACTION]

    for i, ntk in ntks.items():
      self.assertAllClose(ntk_ref, ntk, err_msg=f'{i} impl. fails.')

    for i, ntk in ntks_vmapped.items():
      self.assertAllClose(ntk_ref, ntk, err_msg=f'{i} vmapped impl. fails.')


_functions = {
    '[p[0]**(p[1] + x), p[2] * x + p[0]]':
        lambda p, x: [np.abs(p[0])**(p[1] + x), p[2] * x + p[0]],
    '[p[0]**(p[1] + x), p[2] / x + p[0]]':
        lambda p, x: [np.abs(p[0])**(p[1] + x), p[2] / x + p[0]],

    '[p[0] * p[1] * p[2] + (p[0] @ p[1].T) @ (p[2].T @ p[1]) @ x]':
        lambda p, x: [p[0] * p[1] * p[2] + (p[0] @ p[1].T) @ (p[2].T @ p[1]) @ x],
    '[p[0] / (p[1] * p[2]) + (p[0] @ p[1].T) @ (p[2].T @ p[1]) @ x]':
        lambda p, x: [p[0] / (p[1] * p[2]) + (p[0] @ p[1].T) @ (p[2].T @ p[1]) @ x],

    'x': lambda p, x: x,
    '(x, x)': lambda p, x: (x, x),
    '(x, (x, p))': lambda p, x: (x, (x, p)),
    '[np.eye(1)]': lambda p, x: [np.eye(1)],
    'x**2': lambda p, x: x**2,
    'x @ x.T': lambda p, x: x @ x.T,
    'p': lambda p, x: p,
    'p[0] * p[1]': lambda p, x: p[0] * p[1],
    'p[0] + p[1]': lambda p, x: p[0] + p[1],
    'p[0] + p[1].T': lambda p, x: p[0] + p[1].T,
    'p[0] + p[1] + p[2]': lambda p, x: p[0] + p[1] + p[2],
    'p[0] + p[0].T': lambda p, x: p[0] + p[0].T,
    'p[0] + p[0]': lambda p, x: p[0] + p[0],
    'p[2] * x + p[2] / x': lambda p, x: p[2] * x + p[2] / x,
    '-p[0] + 2 * p[1] - p[2] * 3': lambda p, x: -p[0] + p[1],
    '-p[0] + 2 / p[1] - p[2] / 3': lambda p, x: -p[0] + 2 / p[1] - p[2] / 3,

    'np.prod(p[2])': lambda p, x: np.prod(p[2]),
    'sum(p)': lambda p, x: tree_reduce(lambda x, y: x + np.sum(y), 0.),
    'prod(p)': lambda p, x: tree_reduce(lambda x, y: x * np.prod(y), 1.),
    'sum(p)_typed': lambda p, x: tree_reduce(lambda x, y: x + np.sum(y), np.zeros((), x.dtype)),
    'prod(p)_typed': lambda p, x: tree_reduce(lambda x, y: x * np.prod(y), np.ones((), x.dtype)),
    'x + p[0]': lambda p, x: x + p[0],
    'x - p[1]': lambda p, x: x - p[1],
    '-p[0]': lambda p, x: -p[0],
    'np.squeeze(np.expand_dims(p[0], 0))': lambda p, x: np.squeeze(np.expand_dims(p[0], 0)),
    'p[1]**2': lambda p, x: p[1]**2,
    'p[1] * p[1]': lambda p, x: p[1] * p[1],
    'p[1] / p[1]': lambda p, x: p[1] / p[1],
    'p[1] * p[0]': lambda p, x: p[1] * p[0],
    'p[1] / p[0]': lambda p, x: p[1] / p[0],

    'p[1] * np.expand_dims(np.arange(p[1].shape[1]))': lambda p, x: p[1] * np.expand_dims(np.arange(p[1].shape[1])),
    'p[1] * np.expand_dims(p[0][0])': lambda p, x: p[1] * np.expand_dims(p[0][0]),
    'p[1] / np.expand_dims(np.arange(p[1].shape[1]))': lambda p, x: p[1] / np.expand_dims(np.arange(p[1].shape[1])),
    'p[1] / np.expand_dims(p[0][0])': lambda p, x: p[1] / np.expand_dims(p[0][0]),

    '[p[0], p[1], p[0] / p[1], 2 * p[0], -p[1] + p[0]]': lambda p, x: [p[0], p[1], p[0] / p[1], 2 * p[0], -p[1] + p[0]],
    '[np.sum(p[0], axis=0), np.sum(p[0], axis=1)]': lambda p, x: [np.sum(p[0], axis=0), np.sum(p[0], axis=1)],
    '[np.sum(p[0], axis=0, keepdims=True), np.sum(p[0], axis=1, keepdims=True)]': lambda p, x: [np.sum(p[0], axis=0, keepdims=True), np.sum(p[0], axis=1, keepdims=True)],
    '[p[0], np.sum(p[0], axis=0, keepdims=True), np.sum(p[0], axis=1, keepdims=True)]': lambda p, x: [p[0], np.sum(p[0], axis=0, keepdims=True), np.sum(p[0], axis=1, keepdims=True)],
    '[p[0], p[0].T]': lambda p, x: [p[0], p[0].T],
    '[p[0], p[0]]': lambda p, x: [p[0], p[0]],

    '[p[0].reshape((-1,), p[0].reshape((-1,))]': lambda p, x: [p[0].reshape((-1,)), p[0].reshape((-1,))],
    '[p[0].reshape((2, -1)), p[0].reshape((-1, 2))]': lambda p, x: [p[0].reshape((2, -1)), p[0].reshape((-1, 2))],
    '[p[0].reshape((2, -1)), p[0].reshape((-1, 2)), p[0].T.reshape((2, -1)), p[0].T.reshape((-1, 2))]': lambda p, x: [p[0].reshape((2, -1)), p[0].reshape((-1, 2)), p[0].T.reshape((2, -1)), p[0].T.reshape((-1, 2))],
    '[p[0], p[0].T, p[0].reshape((-1,))]': lambda p, x: [p[0], p[0].T, p[0].reshape((-1,))],
    '[p[0], p[0].T, p[0].reshape((-1,)), p[0].reshape((-1, 1))': lambda p, x: [p[0], p[0].T, p[0].reshape((-1,)), p[0].reshape((-1, 1))],
    '[p[0], p[0].T, p[0].reshape((-1,)), 2 * p[0].reshape((-1, 1)), -p[0].reshape((1, -1))': lambda p, x: [p[0], p[0].T, p[0].reshape((-1,)), 2 * p[0].reshape((-1, 1)), -p[0].reshape((1, -1))],

    'p[0] @ p[0]': lambda p, x: p[0] @ p[0],
    'p[0] @ p[1]': lambda p, x: p[0] @ p[1],
    'p[0] @ p[1] @ p[2]': lambda p, x: p[0] @ p[1] @ p[2],
    'p[0].T @ p[0]': lambda p, x: p[0].T @ p[0],
    'p[1].T @ p[0]': lambda p, x: p[1].T @ p[0],
    'p[2] @ p[0] @ p[1]': lambda p, x: p[2] @ p[0] @ p[1],
    '(p[0] @ p[1], p[0])': lambda p, x: (p[0] @ p[1], p[0]),
    '(p[0] @ p[1], p[1])': lambda p, x: (p[0] @ p[1], p[1]),
    '(p[0] @ p[1], p[1].T)': lambda p, x: (p[0] @ p[1], p[1].T),
    '(p[0] @ p[1], p[0].T)': lambda p, x: (p[0] @ p[1], p[0].T),

    'np.sum(p[0])': lambda p, x: np.sum(p[0]),
    'np.sum(p[0], axis=1, keepdims=True)': lambda p, x: np.sum(p[0], axis=1, keepdims=True),
    'np.sum(p[1], axis=0, keepdims=False)': lambda p, x: np.sum(p[1], axis=0, keepdims=False),
    'np.sum(p[0] @ p[0])': lambda p, x: np.sum(p[0] @ p[0]),
    'np.sum(p[2] * p[1])': lambda p, x: np.sum(p[2] * p[1]),
    'np.sum(p[1] * p[1])': lambda p, x: np.sum(p[1] * p[1]),
    'np.sum(p[2] / p[1])': lambda p, x: np.sum(p[2] / p[1]),
    'np.sum(p[1] / p[1])': lambda p, x: np.sum(p[1] / p[1]),

    'np.zeros((2, 4))': lambda p, x: np.zeros((2, 4)),
    'np.zeros((2, 4))_typed': lambda p, x: np.zeros((2, 4), x.dtype),
    'np.ones((1, 2))': lambda p, x: np.ones((1, 2)),

    'p[2]': lambda p, x: p[2],
    '[p[1], p[0], p[2]]': lambda p, x: [p[1], p[0], p[2]],

    'np.real(p[2])': lambda p, x: np.real(p[2]),
    'np.real(x)': lambda p, x: np.real(x),
    'np.imag(p[2])': lambda p, x: np.imag(p[2]),
    'np.imag(x)': lambda p, x: np.imag(x),
    'np.abs(np.real(p[2]) + np.imag(p[2])) @ np.imag(p[0])': lambda p, x: np.abs(np.real(p[2]) + np.imag(p[2])) @ np.imag(p[0]),
    '[np.real(p[1]), np.imag(p[0]), np.abs(-p[2])],': lambda p, x: [np.real(p[1]), np.imag(p[0]), np.abs(-p[2])],
    'lax.complex(p[0], p[1])': lambda p, x: lax.complex(p[0], p[1]),
    'lax.conj(p[0])': lambda p, x: lax.conj(p[0]),
    'lax.conj(p[0]) @ lax.conj(p[1])': lambda p, x: lax.conj(p[0]) @ lax.conj(p[1]),
    'lax.complex(x, p[1]) * lax.complex(p[0], p[2])': lambda p, x: lax.complex(x, p[1]) * lax.complex(p[0], p[2]),

    'long': lambda p, x: [
        p[0] @ (p[1] + p[0] @ (p[1] / p[0])),
        (p[2], x[1]),
        x[1:2, :] * (x - 1.),
        x[1] * p[2][:1, 0],
        np.array(1., dtype=x.dtype)
    ],

    'reshape': lambda p, x: [p[0].reshape((1, -1,)) @ p[1].reshape((-1, 1)),
                             p[2][:2, :3].reshape((3, 2))],
    'p[0].reshape((-1,)).reshape((3, 3)).T': lambda p, x: p[0].reshape((-1,)).reshape((3, 3)).T,
    'lax_reshape_all': lambda p, x: [
        lax.reshape(p[0], (1, onp.prod(p[0].shape, dtype=int)), tuple(reversed(range(p[0].ndim)))),
        lax.reshape(p[0], (onp.prod(p[0].shape, dtype=int), 1, 1)),
        lax.reshape(p[1], (onp.prod(p[1].shape, dtype=int),), tuple(range(p[1].ndim))),
        lax.reshape(p[1], (1, onp.prod(p[1].shape, dtype=int), 1), tuple(reversed(range(p[1].ndim)))),
        lax.reshape(p[2], tuple(reversed(p[2].shape)), tuple(range(p[2].ndim))),
        lax.reshape(p[2], tuple(reversed(p[2].shape)), tuple(reversed(range(p[2].ndim)))),
        lax.reshape(p[2], utils.zip_flat(reversed(p[2].shape), [1] * p[2].ndim), tuple(range(p[2].ndim))),
        lax.reshape(p[2], (1,) + tuple(reversed(p[2].shape)) + (1,), tuple(reversed(range(p[2].ndim)))),
        lax.reshape(p[2], p[2].shape, tuple(reversed(range(p[2].ndim)))),
        lax.reshape(p[2], p[2].shape, tuple(range(p[2].ndim))),
        lax.reshape(p[2], p[2].shape + (1,), tuple(range(p[2].ndim))),
        lax.reshape(p[2], (1, 1) + p[2].shape, tuple(range(p[2].ndim))),
        lax.reshape(p[2], p[2].shape),
    ],
    'lax_reshape_1_2': lambda p, x: [
        lax.reshape(p[0], (1, onp.prod(p[0].shape, dtype=int)), tuple(reversed(range(p[0].ndim)))) * np.prod(p[0]),
        lax.reshape(p[0], (onp.prod(p[0].shape, dtype=int), 1, 1)) * np.sum(p[0]),
    ],
    'lax_reshape_3_4': lambda p, x: [
        lax.reshape(p[1], (onp.prod(p[1].shape, dtype=int),), tuple(range(p[1].ndim))) + np.sum(p[1]),
        lax.reshape(p[1], (1, onp.prod(p[1].shape, dtype=int), 1), tuple(reversed(range(p[1].ndim)))) - np.sum(p[1]),
    ],
    'lax_reshape_12_13': lambda p, x: [
        lax.reshape(p[2], (1, 1) + p[2].shape, tuple(range(p[2].ndim))) * np.sum(p[1]) - 3,
        lax.reshape(p[2], p[2].shape) + np.sum(p[0]) + 1,
    ],
    'lax_reshape_1_10_11': lambda p, x: [
        lax.reshape(p[0], (1, onp.prod(p[0].shape, dtype=int)), tuple(reversed(range(p[0].ndim)))) + np.sum(p[2]),
        lax.reshape(p[2], p[2].shape, tuple(range(p[2].ndim))) * np.sum(p[0]),
        lax.reshape(p[2], p[2].shape + (1,), tuple(range(p[2].ndim))) + np.sum(p[1]) - 1,
    ],
    'lax_reshape_4_5_6': lambda p, x: [
        lax.reshape(p[1], (1, onp.prod(p[1].shape, dtype=int), 1), tuple(reversed(range(p[1].ndim)))),
        lax.reshape(p[2], tuple(reversed(p[2].shape)), tuple(range(p[2].ndim))),
        lax.reshape(p[2], tuple(reversed(p[2].shape)), tuple(reversed(range(p[2].ndim)))),
    ],
    'lax_reshape_5_6': lambda p, x: [
        lax.reshape(p[2], tuple(reversed(p[2].shape)), tuple(range(p[2].ndim))),
        lax.reshape(p[2], tuple(reversed(p[2].shape)), tuple(reversed(range(p[2].ndim)))),
    ],
    'lax_reshape_1': lambda p, x: lax.reshape(p[0], (1, onp.prod(p[0].shape, dtype=int)), tuple(reversed(range(p[0].ndim)))),
    'lax_reshape_2': lambda p, x: lax.reshape(p[0], (onp.prod(p[0].shape, dtype=int), 1, 1)),
    'lax_reshape_3': lambda p, x: lax.reshape(p[1], (onp.prod(p[1].shape, dtype=int),), tuple(range(p[1].ndim))),
    'lax_reshape_4': lambda p, x: lax.reshape(p[1], (1, onp.prod(p[1].shape, dtype=int), 1), tuple(reversed(range(p[1].ndim)))),
    'lax_reshape_5': lambda p, x: lax.reshape(p[2], tuple(reversed(p[2].shape)), tuple(range(p[2].ndim))),
    'lax_reshape_6': lambda p, x: lax.reshape(p[2], tuple(reversed(p[2].shape)), tuple(reversed(range(p[2].ndim)))),
    'lax_reshape_7': lambda p, x: lax.reshape(p[2], utils.zip_flat(reversed(p[2].shape), [1] * p[2].ndim), tuple(range(p[2].ndim))),
    'lax_reshape_8': lambda p, x: lax.reshape(p[2], (1,) + tuple(reversed(p[2].shape)) + (1,), tuple(reversed(range(p[2].ndim)))),
    'lax_reshape_9': lambda p, x: lax.reshape(p[2], p[2].shape, tuple(reversed(range(p[2].ndim)))),
    'lax_reshape_10': lambda p, x: lax.reshape(p[2], p[2].shape, tuple(range(p[2].ndim))),
    'lax_reshape_11': lambda p, x: lax.reshape(p[2], p[2].shape + (1,), tuple(range(p[2].ndim))),
    'lax_reshape_12': lambda p, x: lax.reshape(p[2], (1, 1) + p[2].shape, tuple(range(p[2].ndim))),
    'lax_reshape_13': lambda p, x: lax.reshape(p[2], p[2].shape),

    'rev': lambda p, x: (lax.rev(p[0], (0,)), lax.rev(p[1], (1,)), lax.rev(p[2], [0, 1])),
    'rev2': lambda p, x: lax.rev(p[0], (0,)) * lax.rev(p[1], (1,)) - lax.rev(p[2], [0, 1])**2,

    'np.squeeze(p[0]) * np.squeeze(p[1])': lambda p, x: np.squeeze(p[0]) * np.squeeze(p[1]),

    'pad_1': lambda p, x: lax.pad(p[0], np.ones((), p[0].dtype), [(0, 0, 0), (0, 1, 2)]),
    'pad_np_const': lambda p, x: np.pad(p[0], [(0, 0), (1, 2)]),
    'pad_np_wrap': lambda p, x: np.pad(p[0], [(2, 3), (0, 0)], 'wrap'),
    'pad_np_max': lambda p, x: np.pad(p[0], [(0, 0), (1, 0)], 'maximum'),
    'pad_2': lambda p, x: lax.pad(p[1], np.ones((), p[1].dtype), [(0, 0, 0), (0, 0, 0)]),
    'pad_3': lambda p, x: lax.pad(p[2], np.ones((), p[2].dtype), [(1, 0, 2), (0, 1, 0)]),
    'pad_4': lambda p, x: lax.pad(p[0], np.ones((), p[0].dtype), [(0, 0, 0), (0, 1, 2)]) + lax.pad(p[0], np.ones((), p[0].dtype), [(0, 1, 2), (0, 0, 0)]).T,

    'lax.concatenate([p[0], p[0]], 0)': lambda p, x: lax.concatenate([p[0], p[0]], 0),
    'lax.concatenate([p[0], p[0]], 1)': lambda p, x: lax.concatenate([p[0], p[0]], 1),
    'lax.concatenate([p[0], p[1]], 0)': lambda p, x: lax.concatenate([p[0], p[1]], 0),
    'lax.concatenate([p[0], p[1]], 1)': lambda p, x: lax.concatenate([p[0], p[1]], 1),
    'lax.concatenate([p[0], p[0].T], 0)': lambda p, x: lax.concatenate([p[0], p[0].T], 0),
    'lax.concatenate([p[0], p[0].T], 1)': lambda p, x: lax.concatenate([p[0], p[0].T], 1),
    'lax.concatenate([p[0], p[1], p[0].T], 1)': lambda p, x: lax.concatenate([p[0], p[1], p[0].T], 1),
    'lax.concatenate([p[0], p[1], p[0].T], 0)': lambda p, x: lax.concatenate([p[0], p[1], p[0].T], 0),
    'lax.concatenate(p, 0)': lambda p, x: lax.concatenate(p, 0),
    'lax.concatenate(p, 1)': lambda p, x: lax.concatenate(p, 1),
    '(lax.concatenate([p[0], x], 1) @ lax.concatenate([p[1], p[2]], 0))**2': lambda p, x: (lax.concatenate([p[0], x], 1) @ lax.concatenate([p[1], p[2]], 0))**2,

    'np.transpose(np.stack(p), (0, 1, 2))': lambda p, x: np.transpose(np.stack(p), (0, 1, 2)),
    'np.transpose(np.stack(p), (0, 2, 1))': lambda p, x: np.transpose(np.stack(p), (0, 2, 1)),
    'np.transpose(np.stack(p), (1, 0, 2))': lambda p, x: np.transpose(np.stack(p), (1, 0, 2)),
    'np.transpose(np.stack(p), (1, 2, 0))': lambda p, x: np.transpose(np.stack(p), (1, 2, 0)),
    'np.transpose(np.stack(p), (2, 1, 0))': lambda p, x: np.transpose(np.stack(p), (2, 1, 0)),
    'np.transpose(np.stack(p), (2, 0, 1))': lambda p, x: np.transpose(np.stack(p), (2, 0, 1)),

    'transpose_3': lambda p, x: np.transpose(np.expand_dims(np.stack(p, 1), 0), (2, 0, 3, 1)),
    'transpose_4': lambda p, x: np.transpose(np.expand_dims(np.stack(p, 1), 1), (0, 2, 1, 3)),
    'transpose_5': lambda p, x: np.transpose(np.expand_dims(np.stack(p, 2), 2), (0, 1, 2, 3)),
    'transpose_6': lambda p, x: np.transpose(np.expand_dims(np.stack(p, 2), 0), (1, 0, 3, 2)),

    # pytype: disable=module-attr
    'lax._reduce_window_sum_1': lambda p, x: lax._reduce_window_sum(p[0], (1, 2), (1, 1), [(0, 0), (0, 1)]),
    'lax._reduce_window_sum_2': lambda p, x: lax._reduce_window_sum(p[0], (1, 1), (1, 1), [(0, 0), (0, 0)]),
    'lax._reduce_window_sum_3': lambda p, x: lax._reduce_window_sum(p[0], (2, 1), (1, 2), [(0, 0), (0, 2)]),
    'lax._reduce_window_sum_4': lambda p, x: lax._reduce_window_sum(p[0], (2, 2), (1, 1), [(2, 3), (0, 0)]),
    'lax._reduce_window_sum_5': lambda p, x: lax._reduce_window_sum(p[0], (1, 1), (2, 1), [(0, 0), (1, 0)]),
    # pytype: enable=module-attr

    'dg1-l': lambda p, x: lax.dot_general(p[0], x, (((), ()), ((), ()))),
    'dg2-l': lambda p, x: lax.dot_general(p[0], x, (((1,), (0,)), ((), ()))),
    'dg3-l': lambda p, x: lax.dot_general(p[0], x, (((0,), (0,)), ((), ()))),
    'dg4-l': lambda p, x: lax.dot_general(p[0], x, (((0, 1), (0, 1)), ((), ()))),
    'dg5-l': lambda p, x: lax.dot_general(p[0], x, (((1,), (1,)), ((0,), (0,)))),
    'dg6-l': lambda p, x: lax.dot_general(p[0], x, (((), ()), ((0, 1), (0, 1)))),
    'dg7-l': lambda p, x: lax.dot_general(p[0], x, (((), ()), ((1,), (0,)))),
    'dg8-l': lambda p, x: lax.dot_general(p[0], x, (((0,), (1,)), ((1,), (0,)))),

    'dg1-r': lambda p, x: lax.dot_general(x, p[0], (((), ()), ((), ()))),
    'dg2-r': lambda p, x: lax.dot_general(x, p[0], (((1,), (0,)), ((), ()))),
    'dg3-r': lambda p, x: lax.dot_general(x, p[0], (((0,), (0,)), ((), ()))),
    'dg4-r': lambda p, x: lax.dot_general(x, p[0], (((0, 1), (0, 1)), ((), ()))),
    'dg5-r': lambda p, x: lax.dot_general(x, p[0], (((1,), (1,)), ((0,), (0,)))),
    'dg6-r': lambda p, x: lax.dot_general(x, p[0], (((), ()), ((0, 1), (0, 1)))),
    'dg7-r': lambda p, x: lax.dot_general(x, p[0], (((), ()), ((1,), (0,)))),
    'dg8-r': lambda p, x: lax.dot_general(x, p[0], (((0,), (1,)), ((1,), (0,)))),

    'dg1-p': lambda p, x: lax.dot_general(p[0], p[1], (((), ()), ((), ()))),
    'dg2-p': lambda p, x: lax.dot_general(p[0], p[1], (((1,), (0,)), ((), ()))),
    'dg3-p': lambda p, x: lax.dot_general(p[0], p[1], (((0,), (0,)), ((), ()))),
    'dg4-p': lambda p, x: lax.dot_general(p[0], p[1], (((0, 1), (0, 1)), ((), ()))),
    'dg5-p': lambda p, x: lax.dot_general(p[0], p[1], (((1,), (1,)), ((0,), (0,)))),
    'dg6-p': lambda p, x: lax.dot_general(p[0], p[1], (((), ()), ((0, 1), (0, 1)))),
    'dg7-p': lambda p, x: lax.dot_general(p[0], p[1], (((), ()), ((1,), (0,)))),
    'dg8-p': lambda p, x: lax.dot_general(p[0], p[1], (((0,), (1,)), ((1,), (0,)))),

    'p[1] * p[0][1, 0]': lambda p, x: p[1] * p[0][1, 0],
    'p[1] / p[0][0, -1]': lambda p, x: p[1] / p[0][1, -1],

    # TODO(romann): investigate full support for compiled loops.
    'lax.map_1': lambda p, x: lax.map(lambda s: 2 * s, p[0]) * np.sum(p[1]),
    'lax.map_2': lambda p, x: lax.map(lambda s: 2 * s + 1, p[0]) * np.sum(p[0]),
    'lax.map_3': lambda p, x: np.sum(lax.map(lambda s: -s / 2., p[0])) * p[0],
    'lax.map_4': lambda p, x: lax.map(lambda s: -s / 2., p[0]) * lax.map(lambda s: 2 * s, p[0]),
    'lax.map_5': lambda p, x: (lax.map(lambda s: lax.map(lambda p: 2 * p, s) + 1., p[0]), p[1]),
    'lax.map_6': lambda p, x: [lax.map(lambda s: lax.map(lambda p: 2 * p, s) + 1., p[0]), p[0]],

    # TODO(romann): revisit if JAX figures out AD for out-of-bounds indexing.
    # 'p[0][1, 0] * p[2].T': lambda p, x: p[0][1, 0] * p[2].T,
}


def _compare_ntks(
    self,
    do_jit,
    do_remat,
    f,
    p,
    x1,
    x2,
    _j_rules,
    _s_rules,
    _fwd,
    vmap_axes=None,
    allow_forward_pass_fail=False,
    rtol=None,
    atol=None,
):
  if do_remat:
    f = remat(f)

  try:
    f1 = f(p, x1)
    jacobian(f)(p, x1)
    if x2 is not None:
      f2 = f(p, x2)
      jacobian(f)(p, x2)

  except Exception as e:
    logging.exception(e)
    if allow_forward_pass_fail:
      raise absltest.SkipTest('Forward/Jacobian pass fails!')
    else:
      raise e

  k_fns = {
      i: nt.empirical_ntk_fn(
          f=f,
          trace_axes=(),
          implementation=i,
          vmap_axes=vmap_axes,
          _j_rules=_j_rules,
          _s_rules=_s_rules,
          _fwd=_fwd
      )
      for i in nt.NtkImplementation
      if i not in (nt.NtkImplementation.AUTO,)
  }

  if do_jit:
    for i in k_fns:
      k_fns[i] = jit(k_fns[i])

  kernels = {
      i: k_fns[i](x1, x2, p)
      for i in k_fns
  }

  kernels = list(enumerate(kernels.items()))

  for idx_1, (i_1, k_1) in kernels:
    for idx2_2, (i_2, k_2) in kernels[idx_1 + 1:]:
      msg = f'Mismatch between implementations {i_1} and {i_2}'
      self.assertAllClose(
          k_1,
          k_2,
          rtol=rtol,
          atol=atol,
          check_dtypes=False,  # TODO(romann): revisit.
          check_finite=False,
          err_msg=msg)


class StructuredDerivativesTest(test_utils.NeuralTangentsTestCase):

  @test_utils.product(
      _j_rules=[
          True,
          False
      ],
      _s_rules=[
          True,
          False
      ],
      _fwd=[
          True,
          False,
          None,
      ],
      same_inputs=[
          # True,
          False
      ],
      shapes=[
          # [[p_i.shape for i in range(num_params)],
          #  [x1.shape, x2.shape]]
          [[(3, 3), (3, 3), (3, 3)],
           [(3, 3), (3, 3)]],
          [[(5, 1, 2), (2, 1, 3), (4, 3, 1)],
           [(2, 3), (3, 2)]],
          [[(2, 3), (3, 2, 1), (2, 3, 5)],
           [(2, 3), (3, 2)]],
          [[(2, 2), (2, 2), (2, 2)],
           [(3, 3), (3, 3)]],
          [[(3, 3), (3, 3), (3, 3)],
           [(3, 3), (2, 3)]],
          [[(3, 2), (2, 3), (3, 1)],
           [(1,), (1,)]],
          [[(3, 2), (2, 3), (3, 1)],
           [(2,), (1,)]],
          [[(2, 1), (2, 4), (4, 1)],
           [(2, 2), (2, 2)]],
          [[(2, 1), (2, 4), (4, 1)],
           [(1, 2), (2, 2)]],
          [[(5,), (1, 5), (5, 1)],
           [(5, 5), (5, 5)]],
          [[(5,), (1, 5), (5, 1)],
           [(4, 5), (5, 5)]],
          [[(1, 1), (0, 0), (0, 1)],
           [(1, 0), (1, 0)]],
          [[(1, 1), (0, 0), (0, 1)],
           [(2, 0), (2, 0)]],
          [[(1, 2), (2, 0), (3, 1)],
           [(1, 4), (1, 4)]],
          [[(1, 2), (2, 0), (3, 1)],
           [(1, 4), (2, 4)]],
          [[(3, 2), (2, 1), (3, 1)],
           [(1, 4), (1, 3)]],
          [[(3, 2), (2, 1), (3, 1)],
           [(1, 4), (2, 4)]],
          [[(), (2, 1), (3, 1)],
           [(1,), (2,)]],
          [[(1,), (1,), (1,)],
           [(2,), (2,)]],
          [[(0,), (0,), (0,)],
           [(0,), (0,)]],
          [[(), (), ()],
           [(), ()]],
          [[(), (), ()],
           [(2,), (1,)]],
          [[(2,), (), (1,)],
           [(0,), (2,)]],
          [[(2,), (0, 3), (1,)],
           [(3, 2), (2, 1)]]
      ],
      p_list=[
          True,
          # False
      ],
      x_list=[
          # True,
          False
      ],
      dtype=[
          np.float32,
          # np.float64,
          # np.float16,
      ],
      do_jit=[
          True,
          # False
      ],
      do_remat=[
          # TODO(romann): support remat
          # True,
          False
      ],
      f_name=list(_functions.keys())
  )
  def test_function(
      self,
      same_inputs,
      f_name,
      shapes,
      p_list,
      x_list,
      do_jit,
      do_remat,
      dtype,
      _j_rules,
      _s_rules,
      _fwd
  ):
    if f_name == 'lax_reshape_all':
      # TODO(romann): investigate slow CPU execution.
      test_utils.skip_test('Skipping large non-structured reshapes on CPU.')

    if 'lax.map' in f_name and shapes[0][0] and shapes[0][0][0] == 0:
      # TODO(romann): fix.
      raise absltest.SkipTest('Zero-length scans not supported without JIT.')

    p = [random.normal(random.PRNGKey(i), s, dtype) for i, s in
         enumerate(shapes[0])]

    k1, k2 = random.split(random.PRNGKey(len(shapes)))
    x1 = random.normal(k1, shapes[1][0], dtype)
    x2 = None if same_inputs else random.normal(k2, shapes[1][1], dtype)

    if not p_list:
      p = p[0]

    if x_list:
      x1 = [x1]
      x2 = [x2]

    if dtype == np.float16:
      atol = 0.1
      rtol = 0.01
    else:
      atol = None
      rtol = None

    _compare_ntks(
        self,
        do_jit=do_jit,
        do_remat=do_remat,
        f=_functions[f_name],
        p=p,
        x1=x1,
        x2=x2,
        atol=atol,
        rtol=rtol,
        allow_forward_pass_fail=True,
        _j_rules=_j_rules,
        _s_rules=_s_rules,
        _fwd=_fwd
    )


# FLAX examples forked from https://github.com/google/flax.


class _MLP(nn.Module):
  features: Sequence[int]

  @nn.compact
  def __call__(self, x):
    for feat in self.features[:-1]:
      x = nn.relu(nn.Dense(feat)(x))
    x = nn.Dense(self.features[-1])(x)
    return x


class _CNN(nn.Module):

  features: int
  feature_group_counts: List[int]

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=self.features, kernel_size=(3, 3),
                feature_group_count=self.feature_group_counts[0])(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=self.features, kernel_size=(3, 3),
                feature_group_count=self.feature_group_counts[1])(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=128)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    x = nn.log_softmax(x)
    return x


class _AutoEncoder(nn.Module):
  encoder_widths: Sequence[int]
  decoder_widths: Sequence[int]
  input_shape: Sequence[int]

  def setup(self):
    input_dim = onp.prod(self.input_shape)
    self.encoder = _MLP(self.encoder_widths)
    self.decoder = _MLP(tuple(self.decoder_widths) + (input_dim,))

  def __call__(self, x):
    return self.decode(self.encode(x))

  def encode(self, x):
    assert x.shape[1:] == self.input_shape
    return self.encoder(np.reshape(x, (x.shape[0], -1)))

  def decode(self, z):
    z = self.decoder(z)
    x = nn.sigmoid(z)
    x = np.reshape(x, (x.shape[0],) + tuple(self.input_shape))
    return x


class _Encoder(nn.Module):
  latents: int

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(32, name='fc1')(x)
    x = nn.relu(x)
    mean_x = nn.Dense(self.latents, name='fc2_mean')(x)
    logvar_x = nn.Dense(self.latents, name='fc2_logvar')(x)
    return mean_x, logvar_x


class _Decoder(nn.Module):

  @nn.compact
  def __call__(self, z):
    z = nn.Dense(16, name='fc1')(z)
    z = nn.relu(z)
    z = nn.Dense(32, name='fc2')(z)
    return z


class _VAE(nn.Module):
  latents: int = 20

  def setup(self):
    self.encoder = _Encoder(self.latents)
    self.decoder = _Decoder()

  def __call__(self, x, z_rng):
    mean, logvar = self.encoder(x)
    z = _reparameterize(z_rng, mean, logvar)
    recon_x = self.decoder(z)
    return recon_x, mean, logvar

  def generate(self, z):
    return nn.sigmoid(self.decoder(z))


_ModuleDef = Any


class _ResNetBlock(nn.Module):
  """ResNet block."""
  filters: int
  conv: _ModuleDef
  norm: _ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x,):
    residual = x
    y = self.conv(self.filters, (3, 3), self.strides)(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3))(y)
    y = self.norm(scale_init=nn.initializers.zeros)(y)

    if residual.shape != y.shape:
      residual = self.conv(self.filters, (1, 1),
                           self.strides, name='conv_proj')(residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)


class _BottleneckResNetBlock(nn.Module):
  """Bottleneck ResNet block."""
  filters: int
  conv: _ModuleDef
  norm: _ModuleDef
  act: Callable
  strides: Tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x):
    residual = x
    y = self.conv(self.filters, (1, 1))(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3), self.strides)(y)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters * 4, (1, 1))(y)
    y = self.norm(scale_init=nn.initializers.zeros)(y)

    if residual.shape != y.shape:
      residual = self.conv(self.filters * 4, (1, 1),
                           self.strides, name='conv_proj')(residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)


class _ResNet(nn.Module):
  """A narrow ResNetV1."""
  stage_sizes: Sequence[int]
  block_cls: _ModuleDef
  num_classes: int
  num_filters: int = 4
  dtype: Any = np.float32
  act: Callable = nn.relu
  conv: _ModuleDef = nn.Conv

  @nn.compact
  def __call__(self, x, train: bool = True):
    conv = partial(self.conv, use_bias=False, dtype=self.dtype)
    norm = partial(nn.BatchNorm,
                   use_running_average=not train,
                   momentum=0.9,
                   epsilon=1e-5,
                   dtype=self.dtype)

    x = conv(self.num_filters, (7, 7), (2, 2),
             padding=[(3, 3), (3, 3)],
             name='conv_init')(x)
    x = norm(name='bn_init')(x)
    x = nn.relu(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = self.block_cls(self.num_filters * 2 ** i,
                           strides=strides,
                           conv=conv,
                           norm=norm,
                           act=self.act)(x)
    x = np.mean(x, axis=(1, 2))
    x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
    x = np.asarray(x, self.dtype)
    return x


_ResNet18 = partial(_ResNet, stage_sizes=[2, 2, 2, 2],
                    block_cls=_ResNetBlock)


def _reparameterize(rng, mean, logvar):
  std = np.exp(0.5 * logvar)
  eps = random.normal(rng, logvar.shape)
  return mean + eps * std


# MLP Mixer forked from https://github.com/google-research/vision_transformer.


class _MlpBlock(nn.Module):
  mlp_dim: int

  @nn.compact
  def __call__(self, x):
    y = nn.Dense(self.mlp_dim)(x)
    y = nn.gelu(y)
    return nn.Dense(x.shape[-1])(y)


class _MixerBlock(nn.Module):
  """Mixer block layer."""
  tokens_mlp_dim: int
  channels_mlp_dim: int

  @nn.compact
  def __call__(self, x):
    y = nn.LayerNorm()(x)
    y = np.swapaxes(y, 1, 2)
    y = _MlpBlock(self.tokens_mlp_dim, name='token_mixing')(y)
    y = np.swapaxes(y, 1, 2)
    x = x + y
    y = nn.LayerNorm()(x)
    return x + _MlpBlock(self.channels_mlp_dim, name='channel_mixing')(y)


class _MlpMixer(nn.Module):
  """Mixer architecture."""
  patches: Any
  num_classes: int
  num_blocks: int
  hidden_dim: int
  tokens_mlp_dim: int
  channels_mlp_dim: int
  model_name: Optional[str] = None

  @nn.compact
  def __call__(self, inputs, *, train):
    del train
    x = nn.Conv(self.hidden_dim, self.patches['size'],
                strides=self.patches['size'], name='stem')(inputs)
    x = x.reshape((x.shape[0], -1, x.shape[-1]))
    for _ in range(self.num_blocks):
      x = _MixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim)(x)
    x = nn.LayerNorm(name='pre_head_layer_norm')(x)
    x = np.mean(x, axis=1)
    if self.num_classes:
      x = nn.Dense(self.num_classes, kernel_init=nn.initializers.zeros,
                   name='head')(x)
    return x


def _get_mixer_b16_config() -> Dict[str, Any]:
  """Returns a narrow Mixer-B/16 configuration."""
  return dict(
      model_name='Mixer-B_16',
      patches={'size': (16, 16)},
      hidden_dim=16,
      num_blocks=2,
      tokens_mlp_dim=4,
      channels_mlp_dim=8,
  )


@test_utils.product(
    j_rules=[
        True,
        False
    ],
    s_rules=[
        True,
        # False
    ],
    fwd=[
        True,
        False,
        None,
    ],
    same_inputs=[
        # True,
        False
    ],
    do_jit=[
        True,
        # False
    ],
    do_remat=[
        # True,
        False
    ],
    dtype=[
        jax.dtypes.canonicalize_dtype(np.float64),
    ]
)
class FlaxOtherTest(test_utils.NeuralTangentsTestCase):

  def test_mlp(self, same_inputs, do_jit, do_remat, dtype, j_rules,
               s_rules, fwd):
    model = _MLP([12, 8, 4])

    k1, k2, ki = random.split(random.PRNGKey(1), 3)

    x1 = random.normal(k1, (4, 10), dtype)
    x2 = None if same_inputs else random.normal(k2, (3, 10), dtype)

    p = model.init(ki, x1)
    _compare_ntks(self, do_jit, do_remat, model.apply, p, x1, x2, j_rules,
                  s_rules, fwd, vmap_axes=0)

  def test_autoencoder(self, same_inputs, do_jit, do_remat, dtype, j_rules,
                       s_rules, fwd):
    test_utils.skip_test(self)

    model = _AutoEncoder(encoder_widths=[20, 10, 5],
                         decoder_widths=[5, 10, 20],
                         input_shape=(12,))

    k1, k2, ki = random.split(random.PRNGKey(1), 3)

    x1 = random.normal(k1, (5, 12), dtype)
    x2 = None if same_inputs else random.normal(k2, (2, 12), dtype)
    p = model.init(ki, x1)

    # Test encoding-decoding.
    _compare_ntks(self, do_jit, do_remat, model.apply, p, x1, x2, j_rules,
                  s_rules, fwd, vmap_axes=0)

    # Test encoding.
    def encode(p, x):
      return model.apply(p, x, method=model.encode)

    _compare_ntks(self, do_jit, do_remat, encode, p, x1, x2, j_rules,
                  s_rules, fwd, vmap_axes=0)

    # Test decoding.
    x1d = model.apply(p, x1, method=model.encode)
    x2d = None if x2 is None else model.apply(p, x2, method=model.encode)

    def decode(p, x):
      return model.apply(p, x, method=model.decode)

    _compare_ntks(self, do_jit, do_remat, decode, p, x1d, x2d, j_rules,
                  s_rules, fwd, vmap_axes=0)

    # Test manual encoding-decoding
    def encode_decode(p, x):
      encoded = model.apply(p, x, method=model.encode)
      decoded = model.apply(p, encoded, method=model.decode)
      return decoded

    # Test encoding-decoding.
    _compare_ntks(self, do_jit, do_remat, encode_decode, p, x1, x2, j_rules,
                  s_rules, fwd, vmap_axes=0)

  def test_vae(self, same_inputs, do_jit, do_remat, dtype, j_rules,
               s_rules, fwd):
    test_utils.skip_test(self)

    model = _VAE(latents=2)
    k1, k2, ki, kzi, kza = random.split(random.PRNGKey(1), 5)
    x1 = random.normal(k1, (1, 1), dtype)
    x2 = None if same_inputs else random.normal(k2, (1, 1), dtype)
    p = model.init(ki, x1, z_rng=kzi)

    _compare_ntks(self, do_jit, do_remat, partial(model.apply, z_rng=kza),
                  p, x1, x2, j_rules, s_rules, fwd)

  def test_resnet18(self, same_inputs, do_jit, do_remat, dtype, j_rules,
                    s_rules, fwd):
    test_utils.skip_test(self)

    model = _ResNet18(num_classes=1)
    k1, k2, ki = random.split(random.PRNGKey(1), 3)
    x1 = random.normal(k1, (1, 224, 224, 1), dtype)
    x2 = None if same_inputs else random.normal(k2, (1, 224, 224, 1), dtype)
    p = model.init(ki, x1)

    def apply_fn(params, x):
      return model.apply(params, x, mutable=['batch_stats'])[0]

    _compare_ntks(self, do_jit, do_remat, apply_fn, p, x1, x2, j_rules,
                  s_rules, fwd)

  def test_mixer_b16(self, same_inputs, do_jit, do_remat, dtype, j_rules,
                     s_rules, fwd):
    test_utils.skip_test(self)

    model = _MlpMixer(num_classes=1, **_get_mixer_b16_config())
    k1, k2, ki = random.split(random.PRNGKey(1), 3)
    x1 = random.normal(k1, (1, 224, 224, 1), dtype)
    x2 = None if same_inputs else random.normal(k2, (1, 224, 224, 1), dtype)
    p = model.init(ki, x1, train=True)

    def apply_fn(params, x):
      return model.apply(params, x, mutable=['batch_stats'], train=True)[0]

    _compare_ntks(self, do_jit, do_remat, apply_fn, p, x1, x2, j_rules,
                  s_rules, fwd)


@test_utils.product(
    j_rules=[
        True,
        False
    ],
    s_rules=[
        True,
        False
    ],
    fwd=[
        True,
        False,
        None,
    ],
    same_inputs=[
        # True,
        False
    ],
    do_jit=[
        True,
        # False
    ],
    do_remat=[
        # True,
        False
    ],
    dtype=[
        jax.dtypes.canonicalize_dtype(np.float64),
    ],
    feature_group_counts=[
        [1, 1],
        [1, 5],
        [5, 1],
        [5, 5]
    ],
)
class FlaxCnnTest(test_utils.NeuralTangentsTestCase):

  def test_flax_cnn(self, same_inputs, do_jit, do_remat, dtype, j_rules,
                    s_rules, fwd, feature_group_counts):
    test_utils.skip_test(self)
    n_chan = 5
    x1 = random.normal(random.PRNGKey(1), (2, 8, 8, n_chan), dtype)
    x2 = None if same_inputs else random.normal(random.PRNGKey(2),
                                                (3, 8, 8, n_chan),
                                                dtype)
    model = _CNN(n_chan, feature_group_counts)
    p = model.init(random.PRNGKey(0), x1)
    _compare_ntks(self, do_jit, do_remat, model.apply, p, x1, x2, j_rules,
                  s_rules, fwd, vmap_axes=0)


@test_utils.product(
    j_rules=[
        True,
        False
    ],
    s_rules=[
        True,
        False
    ],
    fwd=[
        True,
        False,
        None,
    ],
    same_inputs=[
        # True,
        False
    ],
    do_jit=[
        True,
        # False
    ],
    do_remat=[
        # True,
        False
    ],
    dtype=[
        jax.dtypes.canonicalize_dtype(np.float64),
    ],
    n_chan_in=[
        1,
        2,
        3,
        4
    ],
    batch_size=[
        1,
        2,
        3,
        4
    ],
    group_count=[
        1,
        2,
        4,
        8,
        16,
    ],
    group_mode=[
        'batch',
        'feature'
    ],
    vmap_axes=[
        0,
        None
    ]
)
class ConvTest(test_utils.NeuralTangentsTestCase):

  def test_conv(
      self,
      same_inputs,
      do_jit,
      do_remat,
      dtype,
      j_rules,
      s_rules,
      fwd,
      n_chan_in,
      batch_size,
      group_count,
      group_mode,
      vmap_axes
  ):
    # TODO(b/235167364): unskip when the bug is fixed.
    test_utils.skip_test(self, platforms=('cpu', 'tpu',))

    n_chan_out = 16

    if group_mode == 'batch':
      batch_group_count = group_count
      feature_group_count = 1
      if vmap_axes == 0 and group_count > 1:
        raise absltest.SkipTest('Batch grouped convolution not vmap-able.')

    elif group_mode == 'feature':
      batch_group_count = 1
      feature_group_count = group_count

    else:
      raise ValueError(group_mode)

    n_chan_in *= feature_group_count
    batch_size *= batch_group_count

    x1 = random.normal(random.PRNGKey(1), (batch_size, n_chan_in, 5, 4), dtype)
    x2 = None if same_inputs else random.normal(random.PRNGKey(2),
                                                (batch_size, n_chan_in, 5, 4),
                                                dtype)
    p = random.normal(random.PRNGKey(2),
                      (n_chan_out, n_chan_in // feature_group_count, 3, 2))
    def f(p, x):
      return lax.conv_general_dilated(x, p, (1, 1), 'SAME',
                                      feature_group_count=feature_group_count,
                                      batch_group_count=batch_group_count)

    _compare_ntks(self, do_jit, do_remat, f, p, x1, x2, j_rules, s_rules, fwd,
                  vmap_axes=vmap_axes)


class EmpiricalNtkVpTest(test_utils.NeuralTangentsTestCase):

  @test_utils.product(
      same_inputs=[
          True,
          False
      ],
      do_jit=[
          True,
          False
      ],
  )
  def test_ntk_vp_fn(
      self,
      same_inputs,
      do_jit,
  ):
    N1 = 4
    N2 = N1 if same_inputs else 6
    O = 3

    init_fn, f, _ = stax.serial(
        stax.Dense(8),
        stax.Relu(),
        stax.Dense(O)
    )

    k1, k2, k3, k4 = random.split(random.PRNGKey(1), 4)
    x1 = random.normal(k1, (N1, 7))
    x2 = None if same_inputs else random.normal(k2, (N2, 7))
    _, params = init_fn(k3, x1.shape)

    ntk_ref = nt.empirical_ntk_fn(f, (), vmap_axes=0)(x1, x2, params)
    ntk_ref = np.moveaxis(ntk_ref, 1, 2)

    # Compute an NTK via NTK-vps and compare to the reference
    ntk_vp_fn = nt.empirical_ntk_vp_fn(f, x1, x2, params)
    if do_jit:
      ntk_vp_fn = jit(ntk_vp_fn)

    eye = np.eye(N2 * O).reshape((N2 * O, N2, O))
    ntk_vps = jit(jax.vmap(ntk_vp_fn))(eye)
    ntk_vps = np.moveaxis(ntk_vps, (0,), (2,))
    ntk_vps = ntk_vps.reshape((N1, O, N2, O))
    self.assertAllClose(ntk_ref, ntk_vps)

    # Compute a single NTK-vp via reference NTK, and compare to the NTK-vp.
    cotangents = random.normal(k4, f(params, x1 if same_inputs else x2).shape)
    ntk_vp_ref = np.tensordot(ntk_ref, cotangents, ((2, 3), (0, 1)))
    ntk_vp = ntk_vp_fn(cotangents)
    self.assertAllClose(ntk_vp_ref, ntk_vp)


if __name__ == '__main__':
  absltest.main()
