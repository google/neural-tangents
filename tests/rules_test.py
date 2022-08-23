# Copyright 2022 Google LLC
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

"""Tests for `neural_tangents/_src/utils/rules.py`."""

import itertools
import logging
import random
from typing import Optional, Sequence
from typing import Tuple
import warnings

from absl.testing import absltest
import jax
from jax import lax
from jax.config import config
from jax.core import Primitive, ShapedArray
from jax.interpreters import ad
from jax._src import dispatch as jax_dispatch
import jax.numpy as np
import more_itertools
from neural_tangents._src.utils import rules
from tests import test_utils
import numpy as onp


config.parse_flags_with_absl()
config.update('jax_numpy_rank_promotion', 'raise')


random.seed(1)


_PRECISIONS = [
    None,
    # lax.Precision.HIGHEST,
    # lax.Precision.HIGH,
    # lax.Precision.DEFAULT
]


_DTYPES = [
    # np.bfloat16,
    # np.float16,
    np.float32
] + ([np.float64] if jax.dtypes.canonicalize_dtype(np.float64) == np.float64
     else [])


_SHAPES = [
    (),
    # (0,),
    (2,),
    (0, 0),
    (1, 0),
    (0, 1),
    # (1, 1),
    # (2, 1),
    (1, 2),
    (2, 3),
    # (3, 2),
    (0, 1, 0),
    (1, 2, 3),
    (6, 3, 2),
    # (2, 1, 1),
    (3, 2, 1),
    (1, 2, 1, 3),
    # (2, 2, 2, 2),
    (2, 1, 3, 4),
    # (1, 2, 4, 3, 2),
    # (2, 2, 2, 1, 3)
]


def _hypercube(ndim: int, start: int = 1, end: int = 3):
  end = (end,) * ndim
  start = (start,) * ndim
  return tuple(itertools.product(
      *[tuple(range(s, e)) for s, e in zip(start, end)]
  ))


def _prod(x: Sequence[int]) -> int:
  out = 1
  for i in x:
    out *= i
  return out


def _is_broadcastable(s1, s2) -> bool:
  if not (len(s1) == 0 or len(s2) == 0 or len(s1) == len(s2)):
    return False

  for a, b in zip(s1, s2):
    if not (a == 1 or b == 1 or a == b):
      return False

  return True


def _dot_dim_nums(s1, s2):
  pairs = []
  for i in range(len(s1)):
    for j in range(len(s2)):
      if s1[i] == s2[j]:
        pairs += [(i, j)]

  def get_dn(pairs, dn):
    if len(pairs) == 0:
      yield dn

    for p in pairs:
      new_pairs = [_p for _p in pairs if (_p[0] != p[0] and _p[1] != p[1])]
      yield from get_dn(new_pairs, dn)

      dn_c = (
          (dn[0][0] + (p[0],), dn[0][1] + (p[1],)),
          dn[1]
      )
      yield from get_dn(new_pairs, dn_c)

      dn_b = (
          dn[0],
          (dn[1][0] + (p[0],), dn[1][1] + (p[1],)),
      )
      yield from get_dn(new_pairs, dn_b)

  yield from get_dn(pairs, (((), ()), ((), ())))


def _conv_dim_nums(n: int, s2: Tuple[int, ...]):
  dims = itertools.permutations(range(n))
  dns = []
  for i in itertools.product(dims, repeat=3):
    dn = lax.ConvDimensionNumbers(*i)
    if all(s2[s] != 0 for s in dn[1][2:]):
      dns += [dn]
  return random.sample(dns, min(50, len(dns)))


def _paddings(n: int):
  pads = [
      # (0, 0),
      (0, 1),
      (1, 0)
  ]
  return list(itertools.product(pads, repeat=n))


def _strides(n: int, strides=(1, 2)):
  return list(itertools.product(strides, repeat=n))


def _feature_group_counts(lhs_in: int, rhs_in: int, rhs_out: int):
  if rhs_in == 0:
    return []

  feature_group_count, rem = divmod(lhs_in, rhs_in)
  if rem != 0:
    return []

  if feature_group_count == 0 or rhs_out % feature_group_count != 0:
    return []

  return [feature_group_count]


def _batch_group_counts(lhs_in: int, rhs_out: int, feature_group_count: int):
  batch_group_counts = []

  if feature_group_count == 1:
    for i in range(1, lhs_in + 1):
      if lhs_in % i == 0 and rhs_out % i == 0:
        batch_group_counts += [i]

  return batch_group_counts


def _get_inputs(shapes, dtype):
  n = len(shapes)
  keys = jax.random.split(jax.random.PRNGKey(1), n)
  return [jax.random.normal(k, s, dtype) for k, s in zip(keys, shapes)]


def _get_invals(idx, *xs):
  return [ShapedArray(x.shape, x.dtype) if idx == i else
          x for i, x in enumerate(xs)]


def _get_f_and_eqn(params, primitive, *inputs):
  if primitive is None:
    f = lambda x: x
    eqn = None

  else:
    if primitive is lax.pad_p:
      # TODO(romann): find a way to call primitive.bind directly.
      f = lambda *inputs: lax.pad(*inputs, **params)

    elif primitive is lax.conv_general_dilated_p:
      # TODO(romann): find a way to call primitive.bind directly.
      f = lambda *inputs: lax.conv_general_dilated(*inputs, **params)

    else:
      f = lambda *inputs: primitive.bind(*inputs, **params)

    eqn = jax.make_jaxpr(f)(*inputs).eqns[0]

  return eqn, f


def _concat_dims(*shapes):
  dims = []
  if len(shapes) == 0:
    return dims

  s0 = shapes[0]
  n = len(s0)
  if any(len(s) != n for s in shapes):
    return dims

  for i in range(n):
    if all(s[j] == s0[j] for s in shapes for j in range(n) if j != i):
      dims += [i]

  return dims


def _concat_shapes(max_n_args: int = 4, *shapes):
  sets = []
  if len(shapes) == 0:
    return sets

  bins = {}

  for s in shapes:
    n = len(s)
    if n in bins:
      bins[n] += [s]
    else:
      bins[n] = [s]

  for n in bins:
    for n_args in range(max_n_args):
      sets += list(itertools.combinations_with_replacement(bins[n], n_args))

  return sets


_UNARY_PRIMITIVES = {
    None: lambda s, _: [{}],

    jax.lax.copy_p: lambda s, _: [{}],

    ad.zeros_like_p: lambda s, _: [{}],

    lax.neg_p: lambda s, _: [{}],

    lax.transpose_p: lambda s, _: [
        {'permutation': p}
        for p in itertools.permutations(range(len(s)))
    ],

    lax.reduce_sum_p: lambda s, _: [
        {'axes': p}
        for p in more_itertools.powerset(range(len(s)))
    ],

    lax.reduce_window_sum_p: lambda s, _: [
        {
            'base_dilation': b_d,
            'padding': p,
            'window_dilation': w_dl,
            'window_dimensions': w_dd,
            'window_strides': w_s
        } for b_d in _hypercube(len(s))[:3]
        for p in map(tuple, ([[]] if len(s) == 0 else [
            [(0, 0) for _ in range(len(s))],
            [(i, i // 2 + 1) for i in range(len(s))],
            [(i // 2 + 1, i) for i in range(len(s))],
        ])) for w_dl in _hypercube(len(s))[:3]
        for w_dd in _hypercube(len(s))[:3]
        for w_s in _hypercube(len(s))[:3]],

    lax.broadcast_in_dim_p:
        lambda s, _: [
            {
                'shape': sd,
                'broadcast_dimensions': bd
            } for sd, bd in [  # inserting 1 dimension
                (s[:i] + (3,) + s[i:], tuple(range(i)) + tuple(
                    range(i + 1,
                          len(s) + 1))) for i in range(len(s) + 1)
            ] + [  # inserting 2 dimensions
                (s[:i] + (3,) + s[i:j] + (4,) + s[j:], tuple(range(i)) + tuple(
                    range(i + 1, j + 1)) + tuple(range(j + 2,
                                                       len(s) + 2)))
                for i in range(len(s))
                for j in range(i,
                               len(s) + 1)
            ]
        ],

    lax.squeeze_p:
        lambda s, _: [{
            'dimensions': d
        } for d in more_itertools.powerset(
            [idx for idx, i in enumerate(s) if i == 1])],

    lax.convert_element_type_p:
        lambda s, dtype: [{
            'new_dtype': d,
            'weak_type': w
        } for d in set(
            jax.dtypes.canonicalize_dtype(t)
            for t in [np.bfloat16, np.float16, np.float32, np.float64]
            if (jax.dtypes.canonicalize_dtype(t) != jax.dtypes.
                canonicalize_dtype(dtype))) for w in [False] +
                          ([True] if d != np.bfloat16 else [])],

    lax.rev_p:
        lambda s, _: [{
            'dimensions': d
        } for d in more_itertools.powerset(range(len(s)))],

    jax_dispatch.device_put_p:
        lambda s, _: [{}],  # Test cases generated elsewhere.

    lax.pad_p:
        lambda s, dtype: [{
            'padding_value': v,
            'padding_config': c
        }
                          for v in [onp.array(f, dtype) for f in [-0.1, 1.5]]
                          for c in map(tuple, ([[]] if len(s) == 0 else [[
                              (1, 0, k) for k in range(len(s))
                          ], [(k, 1, 0) for k in range(len(s))
                             ], [(3 - k, k // 2, 1) for k in range(len(s))]]))],

    lax.reshape_p:
        lambda s, _: [{
            'new_sizes': n_s,
            'dimensions': d
        }
                      for n_s in {
                          (_prod(s),),
                          (_prod(s), 1),
                          (1, _prod(s), 1),
                      } | ({(2, _prod(s) // 2, 1), (_prod(s) // 2, 1, 2)}
                           if _prod(s) % 2 == 0 else set()
                          ) | ({(1, _prod(s) // 3, 3), (3, _prod(s) // 3)}
                               if _prod(s) % 3 == 0 else set()) | ({
                                   (2, _prod(s) // 6, 3),
                                   (3, _prod(s) // 6, 2),
                                   (3, _prod(s) // 6, 2),
                                   (_prod(s) // 6, 2, 3),
                                   (2, 3, _prod(s) // 6),
                                   (3, 2, _prod(s) // 6),
                               } if _prod(s) % 6 == 0 else set())
                      for d in itertools.permutations(range(len(s)))]
}


_BINARY_PRIMITIVES = {
    # TODO(romann): what is the purpose of this primitive?
    ad.add_jaxvals_p:
        lambda s1, s2: ([{}] if s1 == s2 else []),

    lax.mul_p:
        lambda s1, s2: ([{}] if _is_broadcastable(s1, s2) else []),

    lax.div_p:
        lambda s1, s2: ([{}] if _is_broadcastable(s1, s2) else []),

    lax.add_p:
        lambda s1, s2: ([{}] if _is_broadcastable(s1, s2) else []),

    lax.sub_p:
        lambda s1, s2: ([{}] if _is_broadcastable(s1, s2) else []),

    lax.dot_general_p:
        lambda s1, s2: [
            {
                'dimension_numbers': dn,
                'precision': precision,
                'preferred_element_type': dtype
            } for dn in set(_dot_dim_nums(s1, s2))
            for precision in _PRECISIONS
            for dtype in [None]
        ],

    lax.conv_general_dilated_p:
        lambda s1, s2: [
            {
                'window_strides': window_strides,
                'padding': padding,
                'lhs_dilation': lhs_dilation,
                'rhs_dilation': rhs_dilation,
                'dimension_numbers': dn,
                'feature_group_count': feature_group_count,
                'batch_group_count': batch_group_count,
                'precision': precision,
                'preferred_element_type': dtype
            } for dn in _conv_dim_nums(len(s1), s2)
            for padding in _paddings(len(s1) - 2)
            for window_strides in _strides(len(s1) - 2)
            for feature_group_count in _feature_group_counts(
                lhs_in=s1[dn[0][1]], rhs_in=s2[dn[1][1]], rhs_out=s2[dn[1][0]])
            for batch_group_count in _batch_group_counts(
                s1[dn[0][0]], s2[dn[1][0]], feature_group_count)
            for lhs_dilation in _strides(len(s1) - 2)
            for rhs_dilation in _strides(len(s2) - 2)
            for precision in _PRECISIONS
            for dtype in [None]
        ] if (
            len(s1) == len(s2) and
            len(s1) >= 2
        ) else [],
}


_N_ARY_PRIMITIVES = {
    lax.concatenate_p: lambda *shapes: [{'dimension': d}
                                        for d in _concat_dims(*shapes)]
}


class JacobianRulesTest(test_utils.NeuralTangentsTestCase):

  def _assert_is_diagonal(self, j, axis1, axis2, constant_diagonal: bool):
    c = j.shape[axis1]
    self.assertEqual(c, j.shape[axis2])
    mask_shape = [c if i in (axis1, axis2) else 1 for i in range(j.ndim)]
    mask = np.eye(c, dtype=np.bool_).reshape(mask_shape)

    # Check that removing the diagonal makes the array all 0.
    j_masked = np.where(mask, np.zeros((), j.dtype), j)
    self.assertAllClose(np.zeros_like(j, j.dtype), j_masked)

    if constant_diagonal:
      # Check that diagonal is constant.
      if j.size != 0:
        j_diagonals = np.diagonal(j, axis1=axis1, axis2=axis2)
        self.assertAllClose(np.min(j_diagonals, -1), np.max(j_diagonals, -1))

  def _assert_constant(self, j, axis):
    if axis is not None:
      j = np.moveaxis(j, axis, 0)
      j = list(j)
      for ji in j:
        self.assertAllClose(j[0], ji)

  def _compare_jacobians(self, j_fwd, j_rev, j_rule, primitive):
    if primitive == lax.convert_element_type_p:
      # Check that only one of fwd/red Jacobians matches the rule.
      e_fwd, e_rev = None, None
      try:
        self.assertAllClose(j_fwd, j_rule)
      except Exception as e:
        logging.exception('Forward-mode Jacobian does not match the rule.')
        e_fwd = e

      try:
        self.assertAllClose(j_rev, j_rule)
      except Exception as e:
        logging.exception('Reverse-mode Jacobian does not match the rule.')
        e_rev = e

      if e_fwd is not None and e_rev is not None:
        raise ValueError(e_fwd, e_rev)

    else:
      if primitive == lax.reshape_p:
        # Reshape Jacobian is special-case defined as identity.
        j_rule = j_rule.reshape(j_fwd.shape)

      self.assertAllClose(j_fwd, j_rev)
      if j_rule is not None:
        self.assertAllClose(j_fwd, j_rule)
        self.assertAllClose(j_rev, j_rule)

  def _test_primitive(
      self,
      primitive: Optional[Primitive],
      shapes,
      dtype,
      params
  ):
    xs = _get_inputs(shapes, dtype)
    n = len(xs)
    eqn, f = _get_f_and_eqn(params, primitive, *xs)

    out = f(*xs)
    cts_in = ShapedArray(out.shape, out.dtype)

    argnums = tuple(range(n))
    js_fwd = jax.jacfwd(f, argnums)(*xs)
    js_rev = jax.jacrev(f, argnums)(*xs)

    for idx in range(n):
      if primitive == lax.conv_general_dilated_p and idx == 0:
        raise absltest.SkipTest('Jacobian of CNN wrt inputs not implemented.')

      if primitive == lax.div_p and idx == 1:
        raise absltest.SkipTest('Division is linear only in the first arg.')

      invals = _get_invals(idx, *xs)
      j_fwd, j_rev = js_fwd[idx], js_rev[idx]

      if primitive in rules.JACOBIAN_RULES:
        j_rule = rules.JACOBIAN_RULES[primitive](eqn, idx, invals, cts_in)
      else:
        warnings.warn(f'Jacobian rule for {primitive} at position {idx} not '
                      f'found.')
        j_rule = None

      with self.subTest(f'Jacobian ({idx})'):
        self._compare_jacobians(j_fwd, j_rev, j_rule, primitive)

      structure = rules.STRUCTURE_RULES[primitive](eqn, idx, invals, cts_in)

      j = j_fwd if j_rule is None else j_rule

      if primitive == lax.reshape_p:
        out_ndim = xs[0].ndim
        j = j.transpose(tuple(xs[0].ndim + i
                              for i in onp.argsort(structure.in_trace)) +
                        tuple(i for i in onp.argsort(structure.in_trace)))
        j = j.reshape(
            xs[0].shape +
            tuple(xs[0].shape[i] for i in onp.argsort(structure.in_trace)))

      else:
        out_ndim = out.ndim

      with self.subTest(f'Diagonal axes ({idx})'):
        for i, o in zip(structure.in_diagonal, structure.out_diagonal):
          self._assert_is_diagonal(
              j=j,
              axis1=out_ndim + i[idx],
              axis2=o,
              constant_diagonal=False)

      with self.subTest(f'Constant diagonal axes ({idx})'):
        for i, o in zip(structure.in_trace, structure.out_trace):
          self._assert_is_diagonal(
              j=j,
              axis1=out_ndim + i,
              axis2=o,
              constant_diagonal=True)

      with self.subTest(f'Input broadcast axes ({idx})'):
        for i in structure.in_broadcast:
          self._assert_constant(j=j, axis=i)

      with self.subTest(f'Output broadcast axes ({idx})'):
        for i in structure.out_broadcast:
          self._assert_constant(j=j, axis=i)

  @test_utils.parameters(
      dict(
          primitive=primitive,
          shape=shape,
          dtype=dtype,
          params=params,
      )
      for shape in _SHAPES for dtype in _DTYPES
      for primitive in _UNARY_PRIMITIVES.keys()
      for params in _UNARY_PRIMITIVES[primitive](shape, dtype)
  )
  def test_unary(self, primitive: Optional[Primitive], shape, dtype, params):
    if primitive == jax_dispatch.device_put_p:
      # Can't instantiate devices at test generation time; using subtests.
      for device in [None] + jax.devices() + jax.devices('cpu'):
        with self.subTest(device=device):
          params = {'device': device}
          self._test_primitive(primitive, [shape], dtype, params)

    else:
      self._test_primitive(primitive, [shape], dtype, params)

  @test_utils.parameters(
      dict(
          primitive=primitive,
          shape1=shape1,
          shape2=shape2,
          dtype=dtype,
          params=params
      )
      for shape1 in _SHAPES
      for shape2 in _SHAPES
      for dtype in _DTYPES
      for primitive in _BINARY_PRIMITIVES.keys()
      for params in _BINARY_PRIMITIVES[primitive](shape1, shape2)
  )
  def test_binary(
      self,
      primitive: Optional[Primitive],
      shape1,
      shape2,
      dtype,
      params
  ):
    # TODO(romann): revisit when bugs below are fixed.
    if primitive == lax.conv_general_dilated_p:
      if jax.default_backend() == 'tpu':
        raise absltest.SkipTest('http://b/235167364')

      elif jax.default_backend() == 'gpu' and params['batch_group_count'] != 1:
        raise absltest.SkipTest('http://b/235485533')

    if len(shape1) > 3 or len(shape2) > 3:
      test_utils.skip_test(self)

    self._test_primitive(primitive, [shape1, shape2], dtype, params)

  @test_utils.parameters(
      dict(
          primitive=primitive,
          shapes=shapes,
          dtype=dtype,
          params=params
      )
      for shapes in _concat_shapes(4, *_SHAPES)
      for dtype in _DTYPES
      for primitive in _N_ARY_PRIMITIVES.keys()
      for params in _N_ARY_PRIMITIVES[primitive](*shapes)
  )
  def test_n_ary(self, primitive: Optional[Primitive], shapes, dtype, params):
    self._test_primitive(primitive, shapes, dtype, params)


if __name__ == '__main__':
  absltest.main()
