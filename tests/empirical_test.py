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
import operator
from absl.testing import absltest
from absl.testing import parameterized
from jax import jit, tree_map, tree_multimap
from jax import test_util as jtu
from jax.config import config
import jax.numpy as np
import jax.random as random
import neural_tangents as nt
from neural_tangents import stax
from tests import test_utils


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

CONVOLUTION_CHANNELS = 8
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
  implicit_kernel_fn = jit(nt.empirical_ntk_fn(
      f,
      trace_axes,
      diagonal_axes,
      vmap_axes,
      implementation=2
  ))
  direct_kernel_fn = jit(nt.empirical_ntk_fn(
      f,
      trace_axes,
      diagonal_axes,
      vmap_axes,
      implementation=1
  ))

  nngp_kernel_fn = jit(nt.empirical_nngp_fn(f, trace_axes, diagonal_axes))

  return (partial(implicit_kernel_fn, params=params),
          partial(direct_kernel_fn, params=params),
          partial(nngp_kernel_fn, params=params))


KERNELS = {}
for o in OUTPUT_LOGITS:
  KERNELS['empirical_logits_{}'.format(o)] = partial(_kernel_fns, out_logits=o)


class EmpiricalTest(jtu.JaxTestCase):

  # We use a three layer deep linear network for testing.
  @classmethod
  def f(cls, x, params, do_alter, do_shift_x=True):
    w1, w2, b = params
    if do_alter:
      b *= 2.
      w1 += 5.
      w2 /= 0.9
    if do_shift_x:
      x = x * 2 + 1.
    return [0.5 * np.dot(np.dot(x.T, w1), x) + np.dot(w2, x) + b,
            (np.dot(w1, x),
             w2)
            ]

  @classmethod
  def f_lin_exact(cls, x0, x, params, do_alter, do_shift_x=True):
    w1, w2, b = params
    f0 = EmpiricalTest.f(x0, params, do_alter, do_shift_x)
    if do_shift_x:
      x0 = x0 * 2 + 1.
      x = x * 2 + 1.
    dx = x - x0
    if do_alter:
      b *= 2.
      w1 += 5.
      w2 /= 0.9
    return tree_multimap(operator.add,
                         f0,
                         [np.dot(np.dot(x0.T, w1) + w2, dx),
                          (np.dot(w1, dx),
                           0.)
                          ])

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

  @parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name': '_{}'.format(shape),
          'shape': shape
      } for shape in TAYLOR_MATRIX_SHAPES))
  def testLinearization(self, shape):
    key, params, x0 = self._get_init_data(shape)

    f_lin = nt.linearize(EmpiricalTest.f, x0)

    for _ in range(TAYLOR_RANDOM_SAMPLES):
      for do_alter in [True, False]:
        for do_shift_x in [True, False]:
          key, split = random.split(key)
          x = random.normal(split, (shape[-1], 1))
          self.assertAllClose(EmpiricalTest.f_lin_exact(x0, x, params, do_alter,
                                                        do_shift_x=do_shift_x),
                              f_lin(x, params, do_alter, do_shift_x=do_shift_x))

  @parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name': '_{}'.format(shape),
          'shape': shape
      } for shape in TAYLOR_MATRIX_SHAPES))
  def testTaylorExpansion(self, shape):

    def f_2_exact(x0, x, params, do_alter, do_shift_x=True):
      w1, w2, b = params
      f_lin = EmpiricalTest.f_lin_exact(x0, x, params, do_alter, do_shift_x)
      if do_shift_x:
        x0 = x0 * 2 + 1.
        x = x * 2 + 1.
      if do_alter:
        b *= 2.
        w1 += 5.
        w2 /= 0.9
      dx = x - x0
      return tree_multimap(operator.add,
                           f_lin,
                           [0.5 * np.dot(np.dot(dx.T, w1), dx),
                            (0.,
                             0.)
                            ])

    key, params, x0 = self._get_init_data(shape)

    f_lin = nt.taylor_expand(EmpiricalTest.f, x0, 1)
    f_2 = nt.taylor_expand(EmpiricalTest.f, x0, 2)

    for _ in range(TAYLOR_RANDOM_SAMPLES):
      for do_alter in [True, False]:
        for do_shift_x in [True, False]:
          key, split = random.split(key)
          x = random.normal(split, (shape[-1], 1))
          self.assertAllClose(EmpiricalTest.f_lin_exact(x0, x, params, do_alter,
                                                        do_shift_x=do_shift_x),
                              f_lin(x, params, do_alter, do_shift_x=do_shift_x))
          self.assertAllClose(f_2_exact(x0, x, params, do_alter,
                                        do_shift_x=do_shift_x),
                              f_2(x, params, do_alter, do_shift_x=do_shift_x))

  @parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name': '_train_shape={}_test_shape={}_network={}_{}'.format(
              train, test, network, name),
          'train_shape': train,
          'test_shape': test,
          'network': network,
          'name': name,
          'kernel_fn': kernel_fn
      } for train, test, network in zip(TRAIN_SHAPES, TEST_SHAPES, NETWORK)
                          for name, kernel_fn in KERNELS.items()))
  def testNTKAgainstDirect(
      self, train_shape, test_shape, network, name, kernel_fn):
    key = random.PRNGKey(0)
    key, self_split, other_split = random.split(key, 3)
    data_self = random.normal(self_split, train_shape)
    data_other = random.normal(other_split, test_shape)

    implicit, direct, _ = kernel_fn(key, train_shape[1:], network,
                                    diagonal_axes=(), trace_axes=())

    implicit_batched, direct_batched, _ = kernel_fn(key, train_shape[1:],
                                                    network,
                                                    diagonal_axes=(),
                                                    trace_axes=(),
                                                    vmap_axes=0)

    g = implicit(data_self, None)
    g_direct = direct(data_self, None)
    g_batched = implicit_batched(data_self, None)
    g_direct_batched = direct_batched(data_self, None)
    self.assertAllClose(g, g_direct)
    self.assertAllClose(g, g_batched)
    self.assertAllClose(g, g_direct_batched)

    g = implicit(data_other, data_self)
    g_direct = direct(data_other, data_self)
    g_batched = implicit_batched(data_other, data_self)
    g_direct_batched = direct_batched(data_other, data_self)
    self.assertAllClose(g, g_direct)
    self.assertAllClose(g, g_batched)
    self.assertAllClose(g, g_direct_batched)

  @parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name': '_diagonal_axes={}_trace_axes={}'.format(
              diagonal_axes, trace_axes),
          'diagonal_axes': diagonal_axes,
          'trace_axes': trace_axes,
      }
                          for diagonal_axes in [(),
                                                (0,),
                                                (0, 1),
                                                (0, 1, 2),
                                                (0, 1, 2, 3),
                                                (-1,),
                                                (-2,),
                                                (0, -1),
                                                (1, -2),
                                                (2, 3),
                                                (3, 0, 2)]
                          for trace_axes in [(),
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
                                             (2, 0, -1)]))
  def testAxes(self, diagonal_axes, trace_axes):
    key = random.PRNGKey(0)
    key, self_split, other_split = random.split(key, 3)
    data_self = random.normal(self_split, (4, 5, 6, 3))
    data_other = random.normal(other_split, (2, 5, 6, 3))

    _diagonal_axes = tuple(d % data_self.ndim for d in diagonal_axes)
    _trace_axes = tuple(t % data_self.ndim for t in trace_axes)

    if any(d == c for d in _diagonal_axes for c in _trace_axes):
      raise absltest.SkipTest(
          'diagonal axes must be different from channel axes.')

    get_kernel = KERNELS['empirical_logits_3']
    kwargs = dict(
        key=key,
        input_shape=(5, 6, 3),
        network=CONV,
        diagonal_axes=diagonal_axes,
        trace_axes=trace_axes
    )

    implicit, direct, nngp = get_kernel(**kwargs)
    implicit_batched, direct_batched, _ = get_kernel(**kwargs, vmap_axes=0)

    n_marg = len(_diagonal_axes)
    n_chan = len(_trace_axes)

    g_nngp = nngp(data_self, None)
    self.assertEqual(2 * (data_self.ndim - n_chan) - n_marg, g_nngp.ndim)

    g_direct = direct(data_self, None)
    self.assertEqual(g_nngp.shape, g_direct.shape)

    g_direct_batched = direct_batched(data_self, None)
    g = implicit(data_self, None)
    g_batched = implicit_batched(data_self, None)

    self.assertAllClose(g_direct, g)
    self.assertAllClose(g_direct, g_direct_batched)
    self.assertAllClose(g_direct, g_batched)

    if 0 not in _trace_axes and 0 not in _diagonal_axes:
      g_nngp = nngp(data_other, data_self)
      self.assertEqual(2 * (data_self.ndim - n_chan) - n_marg, g_nngp.ndim)

      g_direct = direct(data_other, data_self)
      self.assertEqual(g_nngp.shape, g_direct.shape)

      g_direct_batched = direct_batched(data_other, data_self)
      g = implicit(data_other, data_self)
      g_batched = implicit_batched(data_other, data_self)

      self.assertAllClose(g_direct, g)
      self.assertAllClose(g_direct, g_direct_batched)
      self.assertAllClose(g_direct, g_batched)

  @parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name': '_same_inputs={}'.format(same_inputs),
          'same_inputs': same_inputs
      } for same_inputs in [True, False]))
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

    implicit_kernel_fn = jit(nt.empirical_ntk_fn(apply_fn, implementation=2))
    direct_kernel_fn = jit(nt.empirical_ntk_fn(apply_fn, implementation=1))
    implicit_batched_kernel_fn = jit(nt.empirical_ntk_fn(
        apply_fn, vmap_axes=(0, 0), implementation=2))
    direct_batched_kernel_fn = jit(nt.empirical_ntk_fn(
        apply_fn, vmap_axes=(0, 0), implementation=1))

    k_direct = direct_kernel_fn(x1, x2, params)

    self.assertAllClose(k_direct, implicit_kernel_fn(x1, x2, params))
    self.assertAllClose(k_direct, direct_batched_kernel_fn(x1, x2, params))
    self.assertAllClose(k_direct, implicit_batched_kernel_fn(x1, x2, params))

    nngp_kernel_fn = jit(nt.empirical_nngp_fn(apply_fn))
    nngp = nngp_kernel_fn(x1, x2, params)
    self.assertEqual(len(nngp), 2)
    self.assertEqual(nngp[0].shape, (3, 3 if same_inputs else 4))
    self.assertEqual(nngp[1].shape, (3, 3 if same_inputs else 4))

  @parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name': '_same_inputs={}'.format(same_inputs),
          'same_inputs': same_inputs
      } for same_inputs in [True, False]))
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
    implicit_kernel_fn = jit(nt.empirical_ntk_fn(apply_fn, implementation=2))
    direct_kernel_fn = jit(nt.empirical_ntk_fn(apply_fn, implementation=1))

    implicit_batched_kernel_fn = jit(nt.empirical_ntk_fn(
        apply_fn, vmap_axes=([0, 0], 0), implementation=2))
    direct_batched_kernel_fn = jit(nt.empirical_ntk_fn(
        apply_fn, vmap_axes=([0, 0], 0), implementation=1))

    k_direct = direct_kernel_fn(x1, x2, params)

    self.assertAllClose(k_direct, implicit_kernel_fn(x1, x2, params))
    self.assertAllClose(k_direct, direct_batched_kernel_fn(x1, x2, params))
    self.assertAllClose(k_direct, implicit_batched_kernel_fn(x1, x2, params))

    nngp_kernel_fn = jit(nt.empirical_nngp_fn(apply_fn))
    nngp = nngp_kernel_fn(x1, x2, params)

    self.assertEqual(len(nngp), 2)
    nngp_shape = (3, 3 if same_inputs else 4)
    self.assertEqual(nngp[0][0].shape, nngp_shape)
    self.assertEqual(nngp[0][1].shape, nngp_shape)
    self.assertEqual(nngp[1].shape, nngp_shape)

  @parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name': '_same_inputs={}'.format(same_inputs),
          'same_inputs': same_inputs
      } for same_inputs in [True, False]))
  def test_vmap_axes(self, same_inputs):
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
    x2 = get_x(n2, random.PRNGKey(2)) if not same_inputs else None

    p1 = random.normal(random.PRNGKey(5), (n1, h2, h2))
    p2 = None if same_inputs else random.normal(random.PRNGKey(6), (n2, h2, h2))

    init_fn, apply_fn, _ = stax.serial(
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
    implicit = jit(nt.empirical_ntk_fn(apply_fn, implementation=2))
    direct = jit(nt.empirical_ntk_fn(apply_fn, implementation=1))

    implicit_batched = jit(nt.empirical_ntk_fn(
        apply_fn, vmap_axes=([(0, 1), 2], [-2, -3], dict(pattern=0)),
        implementation=2))
    direct_batched = jit(nt.empirical_ntk_fn(
        apply_fn, vmap_axes=([(-2, -2), -2], [0, 1], dict(pattern=-3)),
        implementation=1))

    k = direct(x1, x2, params, pattern=(p1, p2))

    self.assertAllClose(k, implicit(x1, x2, params, pattern=(p1, p2)))
    self.assertAllClose(k, direct_batched(x1, x2, params, pattern=(p1, p2)))
    self.assertAllClose(k, implicit_batched(x1, x2, params, pattern=(p1, p2)))


if __name__ == '__main__':
  absltest.main()
