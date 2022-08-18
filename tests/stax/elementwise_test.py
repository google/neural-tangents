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

"""Tests for `neural_tangents/_src/stax/elementwise.py`."""

import itertools
import random as prandom

from absl.testing import absltest
from jax import default_backend
from jax import grad, jacfwd, jacrev, jit, jvp, value_and_grad
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


class ActivationTest(test_utils.NeuralTangentsTestCase):

  def _test_activation_fc(self, phi, get):
    key1, key2, key_mc = random.split(random.PRNGKey(1), 3)
    x1 = np.cos(random.normal(key1, (3, 2)))
    x2 = np.cos(random.normal(key2, (2, 2)))

    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(1024),
        phi,
        stax.Dense(1 if get == 'ntk' else 1024)
    )

    analytic_kernel = kernel_fn(x1, x2, get, diagonal_spatial=True)
    mc_kernel_fn = nt.monte_carlo_kernel_fn(
        init_fn=init_fn,
        apply_fn=apply_fn,
        key=key_mc,
        n_samples=800,
        implementation=2,
        vmap_axes=0,
        device_count=0,
    )

    if get == 'cov1':
      empirical_kernel = np.diag(mc_kernel_fn(x1, None, 'nngp'))
    else:
      empirical_kernel = mc_kernel_fn(x1, x2, get)

    self.assertAllClose(analytic_kernel, empirical_kernel, atol=0.01, rtol=0.05)

  @test_utils.product(
      phi=[
          stax.Gabor,
          stax.Sigmoid_like
      ],
      get=['cov1', 'nngp', 'ntk'],
  )
  def test_nonparametric(
      self,
      phi,
      get,
  ):
    self._test_activation_fc(phi(), get)

  @test_utils.product(
      phi=[stax.Monomial, stax.RectifiedMonomial],
      get=['cov1', 'nngp', 'ntk'],
      degree=[0, 1, 2, 3, 4, 5],
  )
  def test_monomial(
      self,
      phi,
      get,
      degree
  ):
    if phi == stax.RectifiedMonomial and default_backend() == 'tpu':
      raise absltest.SkipTest('`NaN` issues in Rectified Monomials on TPU.')
    self._test_activation_fc(phi(degree=degree), get)

  @test_utils.product(
      phi=[stax.Polynomial],
      get=['cov1', 'nngp', 'ntk'],
      coef=[
          [],
          [0],
          [-2],
          [0, 0],
          [0, 1],
          [1, 0],
          [1, -1],
          [-0.5, 1.2],
          [1.3, 0, -1.2, -0.5],
          [-0.1, 2.1, 0, 0, -1.2, -0.5, 0, 0]
      ],
  )
  def test_polynomial(
      self,
      phi,
      get,
      coef
  ):
    self._test_activation_fc(phi(coef=coef), get)

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

  def _test_activation(
      self,
      activation_fn,
      same_inputs,
      model,
      get,
      rbf_gamma=None
  ):
    if 'conv' in model:
      test_utils.skip_test(self)

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

    if default_backend() == 'cpu':
      num_samplings = 200
      rtol *= 2
    else:
      num_samplings = (500 if activation_fn[2].__name__ in ('Sin', 'Rbf')
                       else 300)

    init_fn, apply_fn, kernel_fn = stax.serial(
        *[affine, activation_fn]*depth, readout)
    analytic_kernel = kernel_fn(X0_1, X0_2, get)
    mc_kernel_fn = nt.monte_carlo_kernel_fn(
        init_fn, apply_fn, split, num_samplings,
        implementation=_DEFAULT_TESTING_NTK_IMPLEMENTATION,
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

  @test_utils.product(
      model=[
          'fc',
          'conv-pool',
          'conv-flatten'
      ],
      phi_name=[
          'Sin',
          'Cos',
          'Erf',
          'Gelu',
          'Sign',
      ],
      same_inputs=[False],
      get=['nngp', 'ntk'],
      approximate=[True, False],
      abc=list(itertools.product(
          [2., 0.3],
          [1.5, 0.3],
          [0., -np.pi/4., np.pi/2.]
      ))
  )
  def test_activation(
      self,
      same_inputs,
      model,
      phi_name,
      get,
      abc,
      approximate
  ):
    if abc != [0.3, 1.5, -np.pi/4]:
      test_utils.skip_test(self)

    if approximate and phi_name != 'Gelu':
      raise absltest.SkipTest(
          f'{phi_name} does not have an `approximate parameter.')

    a, b, c = abc
    if phi_name == 'Sin':
      activation = stax.Sin(a=a, b=b, c=c)
    elif phi_name == 'Erf':
      activation = stax.Erf(a=a, b=b, c=c)
    elif phi_name in ['Gelu', 'Sign', 'Cos']:
      if a != 0.3 or b != 0.3 or c != 0.:
        raise absltest.SkipTest('Skip `Gelu/Sign/Cos` test if '
                                ' (a, b, c) != (.3, .3, 0.).')
      activation = stax.Gelu() if phi_name == 'Gelu' else stax.Sign()
    else:
      raise NotImplementedError(f'Activation {phi_name} is not implemented.')
    self._test_activation(activation, same_inputs, model, get)

  @test_utils.product(
      model=[
          'fc',
          'conv-pool',
          'conv-flatten'
      ],
      same_inputs=[False, True],
      get=['nngp', 'ntk'],
      gamma=[1e-6, 1e-4, 1e-2, 1.0, 2.]
  )
  def test_rbf(self, same_inputs, model, get, gamma):
    activation = stax.Rbf(gamma)
    self._test_activation(activation, same_inputs, model, get,
                          rbf_gamma=gamma)

  @test_utils.product(
      a=[-0.5, 0.25],
      b=[-0.5, -0.1, 0.1],
      phi=[stax.Gaussian, stax.Exp],
      same_inputs=[False, True, None],
      n=[0]
  )
  def test_nonlineariy(self, phi, same_inputs, a, b, n):
    width = 2**10
    n_samples = 2**9
    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(width),
        phi(a=a, b=b),
        stax.Dense(width),
        phi(a=a, b=b),
        stax.Dense(1))

    key1, key2, key_mc = random.split(random.PRNGKey(1), 3)
    shape = (4, 3, 2)[:n] + (1,)
    x1 = np.cos(random.normal(key1, (2,) + shape))
    if same_inputs is None:
      x2 = None
    elif same_inputs is True:
      x2 = x1
    else:
      x2 = np.cos(random.normal(key2, (3,) + shape))

    k = kernel_fn(x1, x2)
    mc_kernel_fn = nt.monte_carlo_kernel_fn(init_fn, apply_fn, key_mc,
                                            n_samples)
    k_mc = mc_kernel_fn(x1, x2, ('nngp', 'ntk'))
    test_utils.assert_close_matrices(self, k_mc.nngp, k.nngp, 6e-2)
    test_utils.assert_close_matrices(self, k_mc.ntk, k.ntk, 6e-2)

  def test_exp_normalized(self):
    key = random.PRNGKey(0)
    x1 = random.normal(key, (2, 6, 7, 1))
    x2 = random.normal(key, (4, 6, 7, 1))

    for do_clip in [True, False]:
      for gamma in [1., 2., 0.5]:
        for get in ['nngp', 'ntk']:
          with self.subTest(do_clip=do_clip, gamma=gamma, get=get):
            _, _, kernel_fn = stax.serial(
                stax.Conv(1, (3, 3)),
                stax.ExpNormalized(gamma, do_clip),
                stax.Conv(1, (3, 3)),
                stax.ExpNormalized(gamma, do_clip),
                stax.GlobalAvgPool(),
                stax.Dense(1)
            )
            k_12 = kernel_fn(x1, x2, get=get)
            self.assertEqual(k_12.shape, (x1.shape[0], x2.shape[0]))

            k_11 = kernel_fn(x1, None, get=get)
            self.assertEqual(k_11.shape, (x1.shape[0],) * 2)
            self.assertGreater(np.min(np.linalg.eigvalsh(k_11)), 0)

            k_22 = kernel_fn(x2, None, get=get)
            self.assertEqual(k_22.shape, (x2.shape[0],) * 2)
            self.assertGreater(np.min(np.linalg.eigvalsh(k_22)), 0)

  def test_exp_normalized_ntk(self):
    def nngp_fn(cov12, var1, var2):
      prod = np.sqrt(var1 * var2)
      return prod * np.exp(cov12 / prod - 1)

    _, _, kernel_fn = stax.serial(stax.Dense(1),
                                  stax.Elementwise(nngp_fn=nngp_fn))

    _, _, kernel_fn_manual = stax.serial(stax.Dense(1),
                                         stax.ExpNormalized())

    key = random.PRNGKey(1)
    x1 = random.normal(key, (5, 4, 3, 1))
    x2 = random.normal(key, (6, 4, 3, 1))

    k = kernel_fn(x1, x2)
    k_manual = kernel_fn_manual(x1, x2)
    self.assertAllClose(k_manual, k)

  @test_utils.product(
      same_inputs=[False, True],
      degree=[1, 2, 3, 4, 5, 6],
      get=['ntk', 'nngp'],
      readout=['pool', 'flatten']
  )
  def test_hermite(self, same_inputs, degree, get, readout):
    key = random.PRNGKey(1)
    key1, key2, key = random.split(key, 3)

    if degree > 2:
      width = 10000
      n_samples = 5000
      test_utils.skip_test(self)
    else:
      width = 10000
      n_samples = 100

    x1 = np.cos(random.normal(key1, [2, 6, 6, 3]))
    x2 = x1 if same_inputs else np.cos(random.normal(key2, [3, 6, 6, 3]))

    conv_layers = [
        stax.Conv(width, (3, 3), W_std=2., b_std=0.5),
        stax.LayerNorm(),
        stax.Hermite(degree),
        stax.GlobalAvgPool() if readout == 'pool' else stax.Flatten(),
        stax.Dense(1) if get == 'ntk' else stax.Identity()]

    init_fn, apply_fn, kernel_fn = stax.serial(*conv_layers)
    analytic_kernel = kernel_fn(x1, x2, get)
    mc_kernel_fn = nt.monte_carlo_kernel_fn(init_fn, apply_fn, key, n_samples)
    mc_kernel = mc_kernel_fn(x1, x2, get)
    rot = degree / 2. * 1e-2
    test_utils.assert_close_matrices(self, mc_kernel, analytic_kernel, rot)


class ElementwiseTest(test_utils.NeuralTangentsTestCase):

  @test_utils.product(
      phi=[
          stax.Identity(),
          stax.Erf(),
          stax.Sin(),
          stax.Relu(),
      ],
      same_inputs=[False, True, None],
      n=[0, 1, 2],
      diagonal_batch=[True, False],
      diagonal_spatial=[True, False]
  )
  def test_elementwise(
      self,
      same_inputs,
      phi,
      n,
      diagonal_batch,
      diagonal_spatial
  ):
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
        sqrt = np.sqrt(np.maximum(prod - cov12 ** 2, 1e-30))
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

  @test_utils.product(
      model=[
          'fc',
          'conv-pool',
          'conv-flatten'
      ],
      phi=[
          stax.Erf(),
          stax.Gelu(),
          stax.Sin(),
      ],
      same_inputs=[False, True],
      get=['nngp', 'ntk']
  )
  def test_elementwise_numerical(self, same_inputs, model, phi, get):
    if 'conv' in model:
      test_utils.skip_test(self)

    key, split = random.split(random.PRNGKey(1))

    output_dim = 1
    b_std = 0.01
    W_std = 1.0
    rtol = 2e-3
    deg = 25
    if get == 'ntk':
      rtol *= 2
    if default_backend() == 'tpu':
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


@test_utils.product(
    same_inputs=[True, False],
    do_stabilize=[True, False],
)
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


class AutodiffTest(test_utils.NeuralTangentsTestCase):

  @test_utils.product(
      get=[
          'ntk',
          'nngp'
      ],
      same_inputs=[True, False, None],
      phi=[
          stax.Erf,
          stax.Sin,
          stax.Gelu,
          stax.Relu,
          stax.ElementwiseNumerical
      ]
  )
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
    if x2 is None:
      dx2 = None
    else:
      dx2 = random.normal(random.PRNGKey(4), x2.shape) * 0.01

    def dk(x1, x2):
      return jvp(k, (x1, x2), (dx1, dx2))[1]

    def d2k(x1, x2):
      return jvp(dk, (x1, x2), (dx1, dx2))[1]

    _dk = dk(x1, x2)
    _d2k = d2k(x1, x2)

    if same_inputs is not False and get == 'ntk' and 'Relu' in name:
      tol = 8e-3
    else:
      tol = 2e-3 if name == 'ElementwiseNumerical' else 1e-4

    def assert_close(x, y, tol=3e-5):
      if default_backend() == 'tpu':
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

  @test_utils.product(
      get=[
          'ntk',
          'nngp'
      ],
      parameterization=[
          'standard',
          'ntk'
      ],
      parameterization_out=[
          'ntk'
      ],
      do_jit=[
          True,
      ],
      x1_type=[
          'zeros',
          'ones',
          'random',
      ],
      x2_type=[
          'zeros',
          'ones',
          'random',
          'x1',
          'none',
      ],
      b_std=[
          None,
          0.1,
      ],
      phi=[
          stax.Identity,
          stax.Erf,
          stax.Abs,
          stax.Gelu,
          stax.Relu,
          stax.Sigmoid_like,
          stax.ABRelu,
          stax.Exp,
          stax.ExpNormalized,
          stax.Gaussian,
          stax.Sign,
          stax.Rbf,
          stax.Cos,
          stax.Sin
      ]
  )
  def test_activations(
      self,
      get,
      parameterization,
      parameterization_out,
      x1_type,
      x2_type,
      b_std,
      phi,
      do_jit
  ):
    """Tests forward- and reverse-mode autodiff for nonlinearities."""
    if phi == stax.ABRelu:
      phi_ = phi(0.25, 0.5)
    else:
      phi_ = phi()

    if phi not in [stax.Relu]:
      test_utils.skip_test(self)

    n_out = 1 if get == 'ntk' else 1024
    width = 832

    W_std_in = width**(-0.5) if parameterization_out == 'standard' else 1.
    if phi == stax.Exp:
      W_std_in /= 10.

    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(
            width,
            W_std=W_std_in,
            b_std=b_std,
            parameterization=parameterization),
        phi_,
        stax.Dense(
            n_out,
            b_std=b_std,
            parameterization=parameterization_out
        ),
    )

    def get_x(x_type, key):
      shape = (1, 2)
      if x_type == 'zeros':
        x = np.zeros(shape)
      elif x_type == 'ones':
        x = np.ones(shape)
      elif x_type == 'random':
        x = random.normal(random.PRNGKey(key), shape)
      elif x_type == 'sin':
        x = np.sin(random.normal(random.PRNGKey(key), shape))
      elif x_type == 'none':
        return None
      else:
        raise ValueError(x_type)
      return x

    x1 = get_x(x1_type, 1)
    if x2_type == 'x1':
      x2 = x1
    else:
      x2 = get_x(x2_type, 2)

    def kernel_scalar(x1, x2):
      return kernel_fn(x1, x2, get)[0, 0]

    if do_jit:
      kernel_scalar = jit(kernel_scalar)

    k1 = kernel_scalar(x1, x2)
    k2, k2_grad = value_and_grad(kernel_scalar)(x1, x2)
    self.assertAllClose(k1, k2)

    # Compare to forward-mode.
    k2_fwd, _ = jvp(kernel_scalar, (x1, x2), (x1, x2))
    k2_grad_fwd = jacfwd(kernel_scalar)(x1, x2)
    self.assertAllClose(k1, k2_fwd)
    self.assertAllClose(k2_grad, k2_grad_fwd)

    # `stax.ExpNormalized` has no forward pass.
    # `stax.Sign` is discontinuous at `0`, so NTK MC kernel does not converge to
    # infinite-width kernel.
    if phi == stax.ExpNormalized or (get == 'ntk' and phi == stax.Sign):
      raise absltest.SkipTest('Not comparing against MC kernels.')

    _kernel_scalar_mc = nt.monte_carlo_kernel_fn(
        init_fn,
        apply_fn,
        key=random.PRNGKey(3),
        n_samples=1,
        device_count=0,
    )

    def kernel_scalar_mc(x1, x2):
      return _kernel_scalar_mc(x1, x2, get)[0, 0]

    k_mc = kernel_scalar_mc(x1, x2)
    k_mc2, k_mc2_grad = value_and_grad(kernel_scalar_mc)(x1, x2)
    self.assertAllClose(k_mc, k_mc2)

    # Compare MC to forward-mode.
    k_mc2_fwd, _ = jvp(kernel_scalar_mc, (x1, x2), (x1, x2))
    k_mc2_grad_fwd = jacfwd(kernel_scalar_mc)(x1, x2)
    self.assertAllClose(k_mc, k_mc2_fwd)
    self.assertAllClose(k_mc2_grad, k_mc2_grad_fwd)

    def kernel_fn_emp(x1, x2, get, params):
      return nt.empirical_kernel_fn(apply_fn)(x1, x2, get, params)[0, 0]

    kernel_fn_emp_g = jit(value_and_grad(kernel_fn_emp), static_argnames='get')

    def kernel_scalar_mc_grad_mean(x1, x2):
      key = random.PRNGKey(4)
      n_samples = 2**9
      k, k_grad = 0., 0.

      for _ in range(n_samples):
        _, params = init_fn(key, x1.shape)
        k_mc2, k_mc2_grad = kernel_fn_emp_g(x1, x2, get, params)
        k += k_mc2
        k_grad += k_mc2_grad
        key, _ = random.split(key)

      k /= n_samples
      k_grad /= n_samples
      return k, k_grad

    k_mc2_mean, k_mc2_grad_mean = kernel_scalar_mc_grad_mean(x1, x2)

    # Compare kernels.
    self.assertAllClose(k1, k_mc2_mean, atol=4e-3, rtol=4e-2)

    if phi == stax.Sign and get == 'nngp':
      raise absltest.SkipTest('Derivative of the empirical NNGP of a '
                              'discontinuous function does not converge '
                              'to the derivative of the infinite width NNGP.')

    if (phi in [stax.Abs, stax.Relu, stax.LeakyRelu, stax.ABRelu] and
        get == 'ntk'):
      raise absltest.SkipTest('Derivative of the empirical NTK of a '
                              'non-differentiable function does not converge '
                              'to the derivative of the infinite width NTK.')

    atol = 1e-2

    # Compare gradient of the analytic kernel to empirical kernel.
    if np.max(np.abs(k2_grad - k_mc2_grad_mean)) > atol:
      test_utils.assert_close_matrices(self,
                                       k_mc2_grad_mean,
                                       k2_grad,
                                       rtol=0.05,
                                       atol=10.)

  @test_utils.product(
      architecture=[
          'conv',
          'wrn'
      ],
      get=[
          'ntk',
          'nngp'
      ],
      do_jit=[
          True,
      ]
  )
  def test_issue_123(
      self,
      get,
      architecture,
      do_jit
  ):
    """Tests https://github.com/google/neural-tangents/issues/123."""
    if architecture == 'wrn':
      # https://github.com/google/neural-tangents/issues/123#issue-992927376
      def WideResnetBlock(channels, strides=(1, 1), channel_mismatch=False):
        main = stax.serial(
            stax.Relu(),
            stax.Conv(
                channels, (3, 3), strides, padding='SAME',
                parameterization='standard'
            ),
            stax.Relu(),
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
            WideResnetGroup(block_size, int(32 * k), (2, 2)),
            WideResnetGroup(block_size, int(64 * k), (2, 2)),
            stax.AvgPool((8, 8), padding='SAME'),
            stax.Flatten(),
            stax.Dense(num_classes, 1.0, 0.0, parameterization='standard'),
        )

      init_fn, apply_fn, kernel_fn = WideResnet(block_size=1,
                                                k=1,
                                                num_classes=1)

    elif architecture == 'conv':
      # https://github.com/google/neural-tangents/issues/123#issuecomment-932809224
      init_fn, apply_fn, kernel_fn = stax.serial(
          stax.Conv(
              1,
              (3, 3)
          ),
          stax.Relu(),
          stax.Flatten(),
      )

    else:
      raise ValueError(architecture)

    x1 = x2 = np.zeros((1, 8, 8, 3))

    def kernel_scalar(x1, x2):
      return kernel_fn(x1, x2, get)[0, 0]

    if do_jit:
      kernel_scalar = jit(kernel_scalar)

    # Compare forward pass to `value_and_grad`.
    k1 = kernel_scalar(x1, x2)
    k2, k2_grad = value_and_grad(kernel_scalar)(x1, x2)
    self.assertAllClose(k1, k2)

    # Compare to forward-mode.
    k2_fwd, _ = jvp(kernel_scalar, (x1, x2), (x1, x2))
    k2_grad_fwd = jacfwd(kernel_scalar)(x1, x2)
    self.assertAllClose(k1, k2_fwd)
    self.assertAllClose(k2_grad, k2_grad_fwd)

    # Compare to 0.
    self.assertAllClose(grad(kernel_scalar)(x1, x2), np.zeros_like(x1))


if __name__ == '__main__':
  absltest.main()
