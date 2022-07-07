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

"""Tests for `neural_tangents/_src/stax/requirements.py`."""


import itertools
import random as prandom

from absl.testing import absltest
from jax import default_backend
from jax import jit
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


@test_utils.product(
    same_inputs=[False, True],
    readout=[
        stax.Flatten(),
        stax.GlobalAvgPool(),
        stax.Identity()
    ],
    readin=[
        stax.Flatten(),
        stax.GlobalAvgPool(),
        stax.Identity()
    ]
)
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
    for inp_a, inp_b, inp_c in itertools.product(stax.Bool, repeat=3):
      for out_a, out_b, out_c in itertools.product(stax.Bool, repeat=3):
        a = stax.Diagonal(inp_a, out_a)
        b = stax.Diagonal(inp_b, out_b)
        c = stax.Diagonal(inp_c, out_c)
        with self.subTest(a=a, b=b, c=c):
          ab_c = (a >> b) >> c
          a_bc = a >> (b >> c)
          self.assertEqual(ab_c, a_bc)

          _ab_c = c << (b << a)
          _a_bc = (c << b) << a
          self.assertEqual(_ab_c, _a_bc)

          self.assertEqual(ab_c, _ab_c)


@test_utils.product(
    same_inputs=[True, False]
)
class InputReqTest(test_utils.NeuralTangentsTestCase):

  def test_input_req(self, same_inputs):
    test_utils.skip_test(self)

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

    correct_conv_fn_mc = nt.monte_carlo_kernel_fn(
        init_fn=init_fn,
        apply_fn=stax.unmask_fn(apply_fn),
        key=key,
        n_samples=400,
        implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION,
        vmap_axes=0
    )
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

    correct_conv_fn_mc = nt.monte_carlo_kernel_fn(
        init_fn=init_fn,
        apply_fn=stax.unmask_fn(apply_fn),
        key=key,
        n_samples=300,
        implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION,
        vmap_axes=0
    )
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

    correct_conv_fn_mc = nt.monte_carlo_kernel_fn(
        init_fn=init_fn,
        apply_fn=stax.unmask_fn(apply_fn),
        key=key,
        n_samples=200,
        implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION,
        vmap_axes=0
    )
    K = correct_conv_fn(x1, x2, get='ntk')
    K_mc = correct_conv_fn_mc(x1, x2, get='ntk')
    self.assertAllClose(K, K_mc, atol=0.01, rtol=0.05)


class MaskingTest(test_utils.NeuralTangentsTestCase):

  @test_utils.product(
      same_inputs=[False],
      get=['ntk'],
      concat=[None, 0, 1],
      p=[0.5],
      mask_axis=[
          (),
          (0,),
          (1, 3)
      ],
      mask_constant=[10.]
  )
  def test_mask_fc(self, same_inputs, get, concat, p, mask_axis, mask_constant):
    width = 512
    n_samples = 128
    tol = 0.04
    key = random.PRNGKey(1)

    x1 = random.normal(key, (4, 6, 5, 7))
    x1 = test_utils.mask(x1, mask_constant, mask_axis, key, p)

    if same_inputs:
      x2 = None
    else:
      x2 = random.normal(key, (2, 6, 5, 7))
      x2 = test_utils.mask(x2, mask_constant, mask_axis, key, p)

    nn = stax.serial(
        stax.Flatten(),
        stax.FanOut(3),
        stax.parallel(
            stax.serial(
                stax.Dense(width, 1., 0.1),
                stax.Abs(),
                stax.DotGeneral(lhs=-0.2),
                stax.Dense(width, 1.5, 0.01),
            ),
            stax.serial(
                stax.Dense(width, 1.1, 0.1),
                stax.DotGeneral(rhs=0.7),
                stax.Erf(),
                stax.Dense(width if concat != 1 else 512, 1.5, 0.1),
            ),
            stax.serial(
                stax.DotGeneral(rhs=0.5),
                stax.Dense(width, 1.2),
                stax.ABRelu(-0.2, 0.4),
                stax.Dense(width if concat != 1 else 1024, 1.3, 0.2),
            )
        ),
        (stax.FanInSum() if concat is None else stax.FanInConcat(concat)),
        stax.Dense(width, 2., 0.01),
        stax.Relu()
    )

    if get == 'nngp':
      init_fn, apply_fn, kernel_fn = stax.serial(nn, stax.Dense(width, 1., 0.1))
    elif get == 'ntk':
      init_fn, apply_fn, kernel_fn = stax.serial(nn, stax.Dense(1, 1., 0.1))
    else:
      raise ValueError(get)

    kernel_fn_mc = nt.monte_carlo_kernel_fn(
        init_fn=init_fn,
        apply_fn=stax.unmask_fn(apply_fn),
        key=key,
        n_samples=n_samples,
        device_count=0 if concat in (0, -2) else -1,
        implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION,
        vmap_axes=None if concat in (0, -2) else 0,
    )

    kernel_fn = jit(kernel_fn, static_argnames='get')
    exact = kernel_fn(x1, x2, get, mask_constant=mask_constant)
    empirical = kernel_fn_mc(x1, x2, get=get, mask_constant=mask_constant)
    test_utils.assert_close_matrices(self, empirical, exact, tol)

  @test_utils.product(
      proj=['flatten', 'avg'],
      same_inputs=[False],
      get=['ntk'],
      n=[0, 1],
      concat=[None, 0, 1],
      mask_constant=[10.],
      p=[0.5],
      transpose=[True, False],
      mask_axis=[(), (0,), (0, 1, 2, 3)]
  )
  def test_mask_conv(
      self,
      same_inputs,
      get,
      mask_axis,
      mask_constant,
      concat,
      proj,
      p,
      n,
      transpose
  ):
    if isinstance(concat, int) and concat > n:
      raise absltest.SkipTest('Concatenation axis out of bounds.')

    test_utils.skip_test(self)
    if default_backend() == 'gpu' and n > 3:
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
    x1 = test_utils.mask(x1, mask_constant, mask_axis, key, p)

    if same_inputs:
      x2 = None
    else:
      x2 = np.cos(random.normal(key, (4,) + spatial_shape + (2,)))
      x2 = test_utils.mask(x2, mask_constant, mask_axis, key, p)

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
                stax.DotGeneral(rhs=0.9),
                conv(
                    dimension_numbers=dimension_numbers,
                    out_chan=width,
                    strides=strides,
                    filter_shape=filter_shape,
                    padding='VALID',
                    W_std=1.2,
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
                    W_std=0.9,
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

    kernel_fn_mc = nt.monte_carlo_kernel_fn(
        init_fn=init_fn,
        apply_fn=stax.unmask_fn(apply_fn),
        key=key,
        n_samples=n_samples,
        device_count=0 if concat in (0, -n) else -1,
        implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION,
        vmap_axes=None if concat in (0, -n) else 0,
    )

    kernel_fn = jit(kernel_fn, static_argnames='get')
    exact = kernel_fn(x1, x2, get, mask_constant=mask_constant)
    empirical = kernel_fn_mc(x1, x2, get=get, mask_constant=mask_constant)
    test_utils.assert_close_matrices(self, empirical, exact, tol)


if __name__ == '__main__':
  absltest.main()
