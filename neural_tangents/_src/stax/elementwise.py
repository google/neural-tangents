# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License');
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

"""Elementwise nonlinearities / activation functions.

For details, please see "`Fast Neural Kernel Embeddings for General Activations
<https://arxiv.org/abs/2209.04121>`_".
"""

import functools
import operator as op
from typing import Callable, Optional, Sequence, Tuple
import warnings

import jax
from jax import custom_jvp, grad, vmap
from jax import numpy as np
from jax.scipy.special import erf
import numpy as onp
from .requirements import Diagonal, get_diagonal, get_diagonal_outer_prods, layer, requires, supports_masking
import scipy as osp
from ..utils import utils
from ..utils.kernel import Kernel
from ..utils.typing import InternalLayer, LayerKernelFn


@layer
@supports_masking(remask_kernel=True)
def Erf(
    a: float = 1.,
    b: float = 1.,
    c: float = 0.) -> InternalLayer:
  """Affine transform of `Erf` nonlinearity, i.e. `a * Erf(b * x) + c`.

  Args:
    a: output scale.
    b: input scale.
    c: output shift.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  def fn(x):
    return a * erf(b * x) + c

  def kernel_fn(k: Kernel) -> Kernel:
    k *= b

    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    cov1_denom = 1 + 2 * cov1
    cov2_denom = None if cov2 is None else 1 + 2 * cov2

    prod11, prod12, prod22 = get_diagonal_outer_prods(cov1_denom,
                                                      cov2_denom,
                                                      k.diagonal_batch,
                                                      k.diagonal_spatial,
                                                      op.mul)

    factor = 2 / np.pi

    def nngp_ntk_fn(
        nngp: np.ndarray,
        prod: np.ndarray,
        ntk: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
      square_root = _sqrt(prod - 4 * nngp**2)
      nngp = factor * np.arctan2(2 * nngp, square_root)

      if ntk is not None:
        dot_sigma = 2 * factor / square_root
        ntk *= dot_sigma

      return nngp, ntk

    def nngp_fn_diag(nngp: np.ndarray) -> np.ndarray:
      return factor * np.arctan2(nngp, np.sqrt(nngp + 1. / 4))

    nngp, ntk = nngp_ntk_fn(nngp, prod12, ntk)

    if k.diagonal_batch and k.diagonal_spatial:
      cov1 = nngp_fn_diag(cov1)
      if cov2 is not None:
        cov2 = nngp_fn_diag(cov2)
    else:
      cov1, _ = nngp_ntk_fn(cov1, prod11)
      if cov2 is not None:
        cov2, _ = nngp_ntk_fn(cov2, prod22)

    k = k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)
    return a * k + c

  return _elementwise(fn, f'Erf({a}, {b}, {c})', kernel_fn)


def Sigmoid_like():
  """A sigmoid like function `f(x) = .5 * erf(x / 2.4020563531719796) + .5`.

  The constant `2.4020563531719796` is chosen so that the squared loss between
  this function and the ground truth sigmoid is minimized on the interval
  `[-5, 5]`; see
  https://gist.github.com/SiuMath/679e8bb4bce13d5f2383a27eca649575.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  return Erf(a=0.5, b=1/2.4020563531719796, c=0.5)


@layer
@supports_masking(remask_kernel=False)
def Gabor() -> InternalLayer:
  """Gabor function `exp(-x^2) * sin(x)`.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """

  def fn(x):
    return np.exp(-x**2) * np.sin(x)

  def kernel_fn(k: Kernel) -> Kernel:
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    prod11, prod12, prod22 = get_diagonal_outer_prods(
        cov1, cov2, k.diagonal_batch, k.diagonal_spatial, op.mul)

    sum11, sum12, sum22 = get_diagonal_outer_prods(
        cov1, cov2, k.diagonal_batch, k.diagonal_spatial, op.add)

    def nngp_ntk_fn(
        nngp: np.ndarray,
        prod: np.ndarray,
        sum_: np.ndarray,
        ntk: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
      diff = 4 * (prod - nngp**2)
      denom = 2 * sum_ + diff + 1
      num = sum_ + diff + 2 * nngp
      exp_left = np.exp(-num / (2 * denom))
      exp_right = np.exp(2 * nngp / denom)

      if ntk is not None:
        shared_term = 1 + 2 * sum_ + 4 * (nngp**2 + prod)
        diff_term = 4 * nngp * (diff + 3 * sum_ + 2)
        lhs = shared_term - diff_term
        rhs = shared_term + diff_term
        t_dot = exp_left * (lhs + exp_right * rhs) / denom**(5. / 2)
        ntk *= t_dot / 2

      nngp = exp_left * (exp_right - 1) / (2 * _sqrt(denom))
      return nngp, ntk

    def nngp_fn_diag(nngp: np.ndarray) -> np.ndarray:
      denom = 1 + 4 * nngp
      return (1 - np.exp(-2 * nngp / denom)) / (2 * _sqrt(denom))

    nngp, ntk = nngp_ntk_fn(nngp, prod12, sum12, ntk)

    if k.diagonal_batch and k.diagonal_spatial:
      cov1 = nngp_fn_diag(cov1)
      if cov2 is not None:
        cov2 = nngp_fn_diag(cov2)
    else:
      cov1, _ = nngp_ntk_fn(cov1, prod11, sum11)
      if cov2 is not None:
        cov2, _ = nngp_ntk_fn(cov2, prod22, sum22)

    return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

  return _elementwise(fn, 'Gabor', kernel_fn)


@layer
@supports_masking(remask_kernel=False)
def Gelu(
    approximate: bool = False) -> InternalLayer:
  """Gelu function.

  Args:
    approximate:
      only relevant for finite-width network, `apply_fn`. If `True`, computes
      an approximation via `tanh`, see "`Gaussian Error Linear Units (GELUs)
      <https://arxiv.org/abs/1606.08415>`_" and :obj:`jax.nn.gelu` for details.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  def fn(x):
    return jax.nn.gelu(x, approximate=approximate)

  def kernel_fn(k: Kernel) -> Kernel:
    """Compute kernels after a `Gelu` layer.

    For NNGP see "`Avoiding Kernel Fixed Points: Computing with ELU and GELU
    Infinite Networks <https://arxiv.org/abs/2002.08517>`_".
    """
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    cov1_plus_1 = cov1 + 1
    cov2_plus_1 = None if cov2 is None else cov2 + 1

    prod11_plus_1, prod12_plus_1, prod22_plus_1 = get_diagonal_outer_prods(
        cov1_plus_1, cov2_plus_1, k.diagonal_batch, k.diagonal_spatial, op.mul)
    prod11, prod12, prod22 = get_diagonal_outer_prods(
        cov1, cov2, k.diagonal_batch, k.diagonal_spatial, op.mul)

    def nngp_ntk_fn(
        nngp: np.ndarray,
        prod: np.ndarray,
        prod_plus_1: np.ndarray,
        ntk: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
      delta_squared = prod_plus_1 - nngp**2
      delta = _sqrt(delta_squared)
      angles = np.arctan2(nngp, delta)
      new_nngp = (nngp**2 + prod * delta_squared) / (prod_plus_1 * delta)
      new_nngp += nngp * angles
      new_nngp /= 2 * np.pi
      new_nngp += 0.25 * nngp

      if ntk is not None:
        second_term = 0.25 + angles / (2 * np.pi)
        first_term = 1 / delta_squared + (1 - prod) / prod_plus_1 + 1
        first_term *= nngp / delta / (2. * np.pi)
        dot_sigma = first_term + second_term
        ntk *= dot_sigma
      return new_nngp, ntk

    def nngp_fn_diag(nngp: np.ndarray) -> np.ndarray:
      square_root = np.sqrt(1. + 2. * nngp)
      new_nngp = nngp / ((nngp + 1.) * np.sqrt(1. + 2. * nngp))
      new_nngp += np.arctan2(nngp, square_root) / 2
      new_nngp /= np.pi
      new_nngp += 0.25
      new_nngp *= nngp
      return new_nngp

    nngp, ntk = nngp_ntk_fn(nngp, prod12, prod12_plus_1, ntk)

    if k.diagonal_batch and k.diagonal_spatial:
      cov1 = nngp_fn_diag(cov1)
      if cov2 is not None:
        cov2 = nngp_fn_diag(cov2)
    else:
      cov1, _ = nngp_ntk_fn(cov1, prod11, prod11_plus_1)
      if cov2 is not None:
        cov2, _ = nngp_ntk_fn(cov2, prod22, prod22_plus_1)

    return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

  return _elementwise(fn, 'Gelu', kernel_fn)


@layer
@supports_masking(remask_kernel=True)
def Sin(
    a: float = 1.,
    b: float = 1.,
    c: float = 0.) -> InternalLayer:
  """Affine transform of `Sin` nonlinearity, i.e. `a sin(b*x + c)`.

  Args:
    a: output scale.
    b: input scale.
    c: input phase shift.
  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  def fn(x):
    return a * np.sin(b * x + c)

  def kernel_fn(k: Kernel) -> Kernel:
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    sum11, sum12, sum22 = get_diagonal_outer_prods(cov1,
                                                   cov2,
                                                   k.diagonal_batch,
                                                   k.diagonal_spatial,
                                                   op.add)
    half_a_square = a**2 / 2.

    def nngp_ntk_fn(nngp, sum_, ntk=None):
      s1 = np.exp(b ** 2 * (-0.5 * sum_ + nngp))
      s2 = np.exp(b ** 2 * (-0.5 * sum_ - nngp)) * np.cos(2 * c)
      nngp = half_a_square * (s1 - s2)
      if ntk is not None:
        ntk *= half_a_square * b**2 * (s1 + s2)
      return nngp, ntk

    def nngp_fn_diag(nngp):
      return half_a_square * (1. - np.exp(-2 * b**2 * nngp) * np.cos(2 * c))

    nngp, ntk = nngp_ntk_fn(nngp, sum12, ntk)

    if k.diagonal_batch and k.diagonal_spatial:
      cov1 = nngp_fn_diag(cov1)
      if cov2 is not None:
        cov2 = nngp_fn_diag(cov2)
    else:
      cov1, _ = nngp_ntk_fn(cov1, sum11)
      if cov2 is not None:
        cov2, _ = nngp_ntk_fn(cov2, sum22)

    return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

  return _elementwise(fn, f'Sin({a}, {b}, {c})', kernel_fn)


def Cos(
    a: float = 1.,
    b: float = 1.,
    c: float = 0.) -> InternalLayer:
  """Affine transform of `Cos` nonlinearity, i.e. `a cos(b*x + c)`.

  Args:
    a: output scale.
    b: input scale.
    c: input phase shift.
  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  return Sin(a=a, b=b, c=c + np.pi / 2)


@layer
@supports_masking(remask_kernel=True)
def Rbf(
    gamma: float = 1.0) -> InternalLayer:
  """Dual activation function for normalized RBF or squared exponential kernel.

  Dual activation function is `f(x) = sqrt(2)*sin(sqrt(2*gamma) x + pi/4)`.
  NNGP kernel transformation correspond to (with input dimension `d`)
  `k = exp(- gamma / d * ||x - x'||^2) = exp(- gamma*(q11 + q22 - 2 * q12))`.

  Args:
    gamma:
      related to characteristic length-scale (l) that controls width of the
      kernel, where `gamma = 1 / (2 l^2)`.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  def fn(x):
    return np.sqrt(2) * np.sin(np.sqrt(2 * gamma) * x + np.pi/4)

  def kernel_fn(k: Kernel) -> Kernel:
    """Compute new kernels after an `Rbf` layer."""
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    sum11, sum12, sum22 = get_diagonal_outer_prods(cov1,
                                                   cov2,
                                                   k.diagonal_batch,
                                                   k.diagonal_spatial,
                                                   op.add)

    def nngp_ntk_fn(nngp, sum_, ntk):
      nngp = np.exp(gamma * (-sum_ + 2 * nngp))
      if ntk is not None:
        ntk *= 2 * gamma * nngp
      return nngp, ntk

    def nngp_fn_diag(nngp):
      return np.ones_like(nngp)

    nngp, ntk = nngp_ntk_fn(nngp, sum12, ntk)

    if k.diagonal_batch and k.diagonal_spatial:
      cov1 = nngp_fn_diag(cov1)
      if cov2 is not None:
        cov2 = nngp_fn_diag(cov2)
    else:
      cov1, _ = nngp_ntk_fn(cov1, sum11, None)
      if cov2 is not None:
        cov2, _ = nngp_ntk_fn(cov2, sum22, None)

    return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

  return _elementwise(fn, f'Rbf({gamma})', kernel_fn)


@layer
@supports_masking(remask_kernel=False)
def ABRelu(
    a: float,
    b: float,
    do_stabilize: bool = False) -> InternalLayer:
  """ABReLU nonlinearity, i.e. `a * min(x, 0) + b * max(x, 0)`.

  Args:
    a: slope for `x < 0`.
    b: slope for `x > 0`.
    do_stabilize: set to `True` for very deep networks.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  def fn(x):
    return a * np.minimum(x, 0) + b * np.maximum(x, 0)

  def kernel_fn(k: Kernel) -> Kernel:
    """Compute new kernels after an `ABRelu` layer.

    See "`Invariance of Weight Distributions in Rectified MLPs
    <https://arxiv.org/abs/1711.09090>`_" for the leaky ReLU derivation.
    """
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    if do_stabilize:
      factor = np.maximum(np.max(np.abs(nngp)), 1e-12)
      nngp /= factor
      cov1 /= factor
      if cov2 is not None:
        cov2 /= factor

    prod11, prod12, prod22 = get_diagonal_outer_prods(cov1,
                                                      cov2,
                                                      k.diagonal_batch,
                                                      k.diagonal_spatial,
                                                      op.mul)

    def nngp_ntk_fn(nngp, prod, ntk=None):
      square_root = _sqrt(prod - nngp**2)
      angles = _arctan2(square_root, nngp, fill_zero=np.pi / 2)

      factor = (a - b)**2 / (2 * np.pi)
      dot_sigma = (a**2 + b**2) / 2 - factor * angles
      nngp = factor * square_root + dot_sigma * nngp

      if ntk is not None:
        ntk *= dot_sigma

      return nngp, ntk

    def nngp_fn_diag(nngp):
      return (a**2 + b**2) / 2 * nngp

    nngp, ntk = nngp_ntk_fn(nngp, prod12, ntk=ntk)

    if k.diagonal_batch and k.diagonal_spatial:
      cov1 = nngp_fn_diag(cov1)
      if cov2 is not None:
        cov2 = nngp_fn_diag(cov2)
    else:
      cov1, _ = nngp_ntk_fn(cov1, prod11)
      if cov2 is not None:
        cov2, _ = nngp_ntk_fn(cov2, prod22)

    if do_stabilize:
      nngp *= factor
      cov1 *= factor
      if cov2 is not None:
        cov2 *= factor

    return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

  return _elementwise(fn, f'ABReLU({a}, {b})', kernel_fn)


def Relu(
    do_stabilize: bool = False) -> InternalLayer:
  """ReLU nonlinearity.

  Args:
    do_stabilize: set to `True` for very deep networks.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  return ABRelu(0, 1, do_stabilize)


def LeakyRelu(
    alpha: float,
    do_stabilize: bool = False) -> InternalLayer:
  """Leaky ReLU nonlinearity, i.e. `alpha * min(x, 0) + max(x, 0)`.

  Args:
    alpha: slope for `x < 0`.
    do_stabilize: set to `True` for very deep networks.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  return ABRelu(alpha, 1, do_stabilize)


def Abs(
    do_stabilize: bool = False) -> InternalLayer:
  """Absolute value nonlinearity.

  Args:
    do_stabilize: set to `True` for very deep networks.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  return ABRelu(-1, 1, do_stabilize)


@layer
@supports_masking(remask_kernel=False)
def Sign() -> InternalLayer:
  """Sign function.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  def fn(x):
    return np.sign(x)

  def kernel_fn(k: Kernel) -> Kernel:
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk
    if ntk is not None:
      ntk = np.zeros_like(ntk)
    _, prod12, _ = get_diagonal_outer_prods(cov1,
                                            cov2,
                                            k.diagonal_batch,
                                            k.diagonal_spatial,
                                            op.mul)
    angles = _arctan2(_sqrt(prod12 - nngp**2), nngp, fill_zero=np.pi / 2)
    nngp = 1 -  angles * 2 / np.pi
    cov1 = np.where(cov1 == 0., 0., 1.)
    cov2 = cov2 if cov2 is None else np.where(cov2 == 0, 0., 1.)
    k = k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)
    return k

  return _elementwise(fn, 'Sign', kernel_fn)


@layer
@supports_masking(remask_kernel=True)
def Exp(a: float = 1, b: float = 1) -> InternalLayer:
  """Elementwise natural exponent function `a * np.exp(b * x)`.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """

  def fn(x):
    return a * np.exp(b * x)

  def kernel_fn(k: Kernel) -> Kernel:
    """Compute new kernels after an `Exp` layer."""
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    sum11, sum12, sum22 = get_diagonal_outer_prods(
        cov1, cov2, k.diagonal_batch, k.diagonal_spatial, op.add)

    def nngp_ntk_fn(nngp, sum_, ntk):
      nngp = np.exp(b**2 * (sum_ / 2 + nngp))
      if ntk is not None:
        ntk *= b**2 * nngp
      return nngp, ntk

    def nngp_fn_diag(nngp):
      return np.exp(2 * b**2 * nngp)

    nngp, ntk = nngp_ntk_fn(nngp, sum12, ntk)

    if k.diagonal_batch and k.diagonal_spatial:
      cov1 = nngp_fn_diag(cov1)
      if cov2 is not None:
        cov2 = nngp_fn_diag(cov2)
    else:
      cov1, _ = nngp_ntk_fn(cov1, sum11, None)
      if cov2 is not None:
        cov2, _ = nngp_ntk_fn(cov2, sum22, None)

    return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk) * a

  return _elementwise(fn, f'Exp({a}, {b})', kernel_fn)


@layer
@supports_masking(remask_kernel=True)
def Gaussian(a: float = 1, b: float = -1) -> InternalLayer:
  """Elementwise Gaussian function `a * np.exp(b * x**2)`.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  def fn(x):
    return a * np.exp(b * x**2)

  def kernel_fn(k: Kernel) -> Kernel:
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    cov1_denom = 1 - 2 * b * cov1
    cov2_denom = None if cov2 is None else 1 - 2 * b * cov2

    prod11, prod12, prod22 = get_diagonal_outer_prods(cov1_denom,
                                                      cov2_denom,
                                                      k.diagonal_batch,
                                                      k.diagonal_spatial,
                                                      op.mul)

    factor = 4 * b**2

    def nngp_ntk_fn(
        nngp: np.ndarray,
        prod: np.ndarray,
        ntk: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
      det = _sqrt((prod - factor * nngp**2))

      if ntk is not None:
        ntk *= factor * nngp / det**3

      nngp = 1 / det
      return nngp, ntk

    def nngp_fn_diag(nngp: np.ndarray) -> np.ndarray:
      return 1 / _sqrt(1 - 4 * b * nngp)

    nngp, ntk = nngp_ntk_fn(nngp, prod12, ntk)

    if k.diagonal_batch and k.diagonal_spatial:
      cov1 = nngp_fn_diag(cov1)
      if cov2 is not None:
        cov2 = nngp_fn_diag(cov2)
    else:
      cov1, _ = nngp_ntk_fn(cov1, prod11)
      if cov2 is not None:
        cov2, _ = nngp_ntk_fn(cov2, prod22)

    return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk) * a

  return _elementwise(fn, f'Gaussian({a}, {b})', kernel_fn)


@layer
@supports_masking(remask_kernel=True)
def ExpNormalized(
    gamma: float = 1,
    shift: float = -1,
    do_clip: bool = False) -> InternalLayer:
  """Simulates the "Gaussian normalized kernel".

  See page 6 in
  "`Neural Kernels Without Tangents <https://arxiv.org/abs/2003.02237>`_".

  Args:
    gamma: exponent scalar coefficient.
    shift: shift exponentiated normalized covariance by this much.
    do_clip: True to clip normalized covariance, potentially improving accuracy.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.

  Raises:
    NotImplementedError: if finite width `apply_fn` is called.
  """

  def kernel_fn(k: Kernel) -> Kernel:
    cov1, cov2, nngp, ntk = k.cov1, k.cov2, k.nngp, k.ntk
    prod11, prod12, prod22 = get_diagonal_outer_prods(cov1,
                                                      cov2,
                                                      k.diagonal_batch,
                                                      k.diagonal_spatial,
                                                      op.mul)
    tol = 1e-30
    prod11 = _sqrt(prod11, tol)
    prod12 = _sqrt(prod12, tol)
    prod22 = _sqrt(prod22, tol) if prod22 is not None else None

    def exp(cov, prod):
      if cov is not None:
        cov /= prod
        if do_clip:
          cov = np.clip(cov, -1, 1)
        cov = np.exp(gamma * (cov + shift))
      return cov

    exp12 = exp(nngp, prod12)

    return k.replace(
        nngp=prod12 * exp12,
        cov1=prod11 * exp(cov1, prod11),
        cov2=None if cov2 is None else prod22 * exp(cov2, prod22),
        ntk=ntk if ntk is None else gamma * ntk * exp12)

  return _elementwise(None, 'ExpNormalized', kernel_fn)


@layer
@supports_masking(remask_kernel=True)
def Hermite(degree: int) -> InternalLayer:
  """Hermite polynomials.

  Inputs to this layer are assumed to have unit norm, i.e.
  `np.std(x, axis=channel_axis) == 1`. The Hermite polynomials are normalized
  so that the L2 norm w.r.t. standard Gaussian is 1.

  Args:
    degree: a non-negative integer.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  if degree < 0:
    raise NotImplementedError('`degree` must be a non-negative integer.')

  p = onp.polynomial.hermite_e.herme2poly([0] * degree + [1])[::-1]
  coeff = functools.reduce(op.mul, range(1, degree + 1), 1)**0.5

  def fn(x):
    return np.polyval(p, x) / coeff

  def kernel_fn(k: Kernel) -> Kernel:
    warnings.warn(
        'Inputs to this layer are assumed to have unit norm across '
        ' channels/features, i.e. np.std(x, axis=channel_axis) == 1.')

    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    if ntk is not None:
      if degree == 0:
        ntk = np.zeros_like(ntk)
      else:
        ntk = degree * nngp**(degree - 1) * ntk

    def _power(mat):
      return mat**degree if mat is not None else None

    nngp, cov1, cov2 = map(_power, (nngp, cov1, cov2))
    k = k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)
    return k

  return _elementwise(fn, f'{degree}-Hermite polynomial', kernel_fn)


@layer
@supports_masking(remask_kernel=False)
def Monomial(degree: int) -> InternalLayer:
  """Monomials, i.e. `x^degree`.

  Args:
    degree: an integer between 0 and 5.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  if degree not in [0, 1, 2, 3, 4, 5]:
    raise NotImplementedError('The `degree` must be an integer between '
                              '`0` and `5`.')

  def fn(x):
    return x**degree

  def kernel_fn(k: Kernel) -> Kernel:
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    prod11, prod12, prod22 = get_diagonal_outer_prods(cov1,
                                                      cov2,
                                                      k.diagonal_batch,
                                                      k.diagonal_spatial,
                                                      op.mul)

    def nngp_ntk_fn(
        nngp: np.ndarray,
        prod: np.ndarray,
        ntk: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:

      def nngp_fn(nngp: np.ndarray, degree: int) -> np.ndarray:
        if degree == -1:
          nngp = np.zeros_like(nngp)

        elif degree == 0:
          nngp = np.ones_like(nngp)

        elif degree == 1:
          pass

        elif degree == 2:
          nngp = 2 * nngp ** 2 + prod

        elif degree == 3:
          nngp = 6 * nngp ** 3 + 9 * nngp * prod

        elif degree == 4:
          nngp = 3 * (8 * nngp ** 4 + 3 * prod * (8 * nngp ** 2 + prod))

        elif degree == 5:
          nngp = 15 * nngp * (
              8 * nngp ** 4 + 5 * prod * (8 * nngp ** 2 + 3 * prod))

        else:
          raise NotImplementedError(degree)

        return nngp

      if ntk is not None:
        ntk *= degree**2 * nngp_fn(nngp, degree - 1)

      nngp = nngp_fn(nngp, degree)
      return nngp, ntk

    def nngp_fn_diag(nngp: np.ndarray) -> np.ndarray:
      return _double_factorial(2 * degree - 1) * nngp**degree

    nngp, ntk = nngp_ntk_fn(nngp, prod12, ntk)

    if k.diagonal_batch and k.diagonal_spatial:
      cov1 = nngp_fn_diag(cov1)
      if cov2 is not None:
        cov2 = nngp_fn_diag(cov2)
    else:
      cov1, _ = nngp_ntk_fn(cov1, prod11)
      if cov2 is not None:
        cov2, _ = nngp_ntk_fn(cov2, prod22)

    k = k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)
    return k

  return _elementwise(fn, f'{degree}-monomial', kernel_fn)


@layer
@supports_masking(remask_kernel=False)
def RectifiedMonomial(degree: int) -> InternalLayer:
  """Rectified monomials, i.e. `(x >= 0) * x^degree`.

  Args:
    degree: a non-negative integer power.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  if degree < 0:
    raise NotImplementedError('`degree` must be a non-negative integer.')

  def fn(x):
    return (x >= 0) * x**degree

  def kernel_fn(k: Kernel) -> Kernel:
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    prod11, prod12, prod22 = get_diagonal_outer_prods(cov1,
                                                      cov2,
                                                      k.diagonal_batch,
                                                      k.diagonal_spatial,
                                                      op.mul)

    def j(nngp: np.ndarray, sqrt_prod: np.ndarray) -> np.ndarray:
      theta = np.arccos(nngp / sqrt_prod)

      def f0(theta: np.ndarray) -> np.ndarray:
        return (np.pi - theta) / np.sin(theta)

      def diff(f: Callable[[np.ndarray], np.ndarray]
               ) -> Callable[[np.ndarray], np.ndarray]:

        def df(theta: np.ndarray) -> np.ndarray:
          return np.vectorize(grad(f))(theta) / np.sin(theta)

        return df

      f = f0
      for _ in range(degree):
        f = diff(f)

      return (-1)**degree * (np.sin(theta))**(2 * degree + 1) * f(theta)

    def nngp_ntk_fn(
        nngp: np.ndarray,
        prod: np.ndarray,
        ntk: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:

      sqrt_prod = _sqrt(prod)
      coeff = sqrt_prod**degree / (2 * np.pi)

      if ntk is not None:
        if degree == 0:
          ntk = np.zeros_like(ntk)
        else:
          j_dot = np.vectorize(grad(j))(nngp, sqrt_prod)
          ntk *= coeff * j_dot

      nngp = coeff * j(nngp, sqrt_prod)
      return nngp, ntk

    def nngp_fn_diag(nngp: np.ndarray) -> np.ndarray:
      return _double_factorial(2 * degree - 1) * nngp**degree / 2

    nngp, ntk = nngp_ntk_fn(nngp, prod12, ntk)

    if k.diagonal_batch and k.diagonal_spatial:
      cov1 = nngp_fn_diag(cov1)
      if cov2 is not None:
        cov2 = nngp_fn_diag(cov2)
    else:
      cov1, _ = nngp_ntk_fn(cov1, prod11)
      if cov2 is not None:
        cov2, _ = nngp_ntk_fn(cov2, prod22)

    k = k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)
    return k

  return _elementwise(fn, f'{degree}-rectified-monomial', kernel_fn)


@layer
@supports_masking(remask_kernel=False)
def Polynomial(coef: Sequence[float]) -> InternalLayer:
  """Polynomials, i.e. `coef[0] + coef[1] * x + … + coef[n] * x**n`.

  Args:
    coef:
      a sequence of coefficients. Follows
      :class:`numpy.polynomial.polynomial.Polynomial` API.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  coef = onp.array(coef)

  def fn(x):
    return np.polyval(coef[::-1], x)

  degree = len(coef)

  def kernel_fn(k: Kernel) -> Kernel:
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    def r(n: Optional[np.ndarray], l: int) -> Optional[np.ndarray]:
      if n is None:
        return None

      coef_dict = {
          2 * i + l: coef[2 * i + l] * _factorial(2 * i + l) / (
              2**i * _factorial(i) * _factorial(l)**0.5)
          for i in range(0, (degree - 1 - l) // 2 + 1)
      }
      coef_l = onp.array(
          [coef_dict[i] if i in coef_dict else 0 for i in range(degree)])
      return np.polyval(coef_l[::-1], n**0.5)

    if degree == 0:
      rs11, rs12, rs22 = [], [], []
    else:
      rs11, rs12, rs22 = list(zip(*[
          get_diagonal_outer_prods(r(cov1, l),
                                   r(cov2, l),
                                   k.diagonal_batch,
                                   k.diagonal_spatial,
                                   op.mul)
          for l in range(degree)
      ]))

    prod11, prod12, prod22 = get_diagonal_outer_prods(cov1,
                                                      cov2,
                                                      k.diagonal_batch,
                                                      k.diagonal_spatial,
                                                      op.mul)

    def nngp_ntk_fn(
        nngp: np.ndarray,
        prod: np.ndarray,
        r_prods: Sequence[np.ndarray],
        ntk: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
      ratio = nngp / _sqrt(prod)

      if ntk is not None:
        t_dot = np.zeros_like(ntk)
        for l in range(1, degree):
          t_dot += l * r_prods[l] * ratio**(l - 1)
        ntk *= t_dot / _sqrt(prod)

      nngp = np.zeros_like(nngp)
      for l in range(degree):
        nngp += r_prods[l] * ratio ** l

      return nngp, ntk

    def nngp_fn_diag(nngp: np.ndarray,
                     r_prods: Sequence[np.ndarray]) -> np.ndarray:
      out = np.zeros_like(nngp)
      for l in range(degree):
        out += r_prods[l]
      return out

    nngp, ntk = nngp_ntk_fn(nngp, prod12, rs12, ntk)

    if k.diagonal_batch and k.diagonal_spatial:
      cov1 = nngp_fn_diag(cov1, rs11)
      if cov2 is not None:
        cov2 = nngp_fn_diag(cov2, rs22)
    else:
      cov1, _ = nngp_ntk_fn(cov1, prod11, rs11)
      if cov2 is not None:
        cov2, _ = nngp_ntk_fn(cov2, prod22, rs22)

    k = k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)
    return k

  return _elementwise(fn, f'{coef}-polynomial', kernel_fn)


@layer
@supports_masking(remask_kernel=True)
def Elementwise(
    fn: Optional[Callable[[float], float]] = None,
    nngp_fn: Optional[Callable[[float, float, float], float]] = None,
    d_nngp_fn: Optional[Callable[[float, float, float], float]] = None
) -> InternalLayer:
  """Elementwise application of `fn` using provided `nngp_fn`.

  Constructs a layer given only scalar-valued nonlinearity / activation
  `fn` and the 2D integral `nngp_fn`. NTK function is derived automatically in
  closed form from `nngp_fn`.

  If you cannot provide the `nngp_fn`, see :obj:`ElementwiseNumerical` to use
  numerical integration or `nt.monte_carlo.monte_carlo_kernel_fn` to use Monte
  Carlo sampling.

  If your function is implemented separately (e.g. `nt.stax.Relu` etc) it's best
  to use the custom implementation, since it uses symbolically simplified
  expressions that are more precise and numerically stable.

  For details, please see "`Fast Neural Kernel Embeddings for General
  Activations <https://arxiv.org/abs/2209.04121>`_".

  See Also:
    `examples/elementwise.py`.

  Example:
    >>> fn = jax.scipy.special.erf  # type: Callable[[float], float]
    >>> #
    >>> def nngp_fn(cov12: float, var1: float, var2: float) -> float:
    >>>   prod = (1 + 2 * var1) * (1 + 2 * var2)
    >>>   return np.arcsin(2 * cov12 / np.sqrt(prod)) * 2 / np.pi
    >>> #
    >>> # Use autodiff and vectorization to construct the layer:
    >>> _, _, kernel_fn_auto = stax.Elementwise(fn, nngp_fn)
    >>> #
    >>> # Use custom pre-derived expressions
    >>> # (should be faster and more numerically stable):
    >>> _, _, kernel_fn_stax = stax.Erf()
    >>> #
    >>> kernel_fn_auto(x1, x2) == kernel_fn_stax(x1, x2)  # usually `True`.

  Args:
    fn:
      a scalar-input/valued function `fn : R -> R`, the activation /
      nonlinearity. If `None`, invoking the finite width `apply_fn` will raise
      an exception.

    nngp_fn:
      a scalar-valued function
      `nngp_fn : (cov12, var1, var2) |-> E[fn(x_1) * fn(x_2)]`, where the
      expectation is over bivariate normal `x1, x2` with variances `var1`,
      `var2` and covarianve `cov12`. Needed for both NNGP and NTK calculation.
      If `None`, invoking infinite width `kernel_fn` will raise an exception.

    d_nngp_fn:
      an optional scalar-valued function
      `d_nngp_fn : (cov12, var1, var2) |-> E[fn'(x_1) * fn'(x_2)]` with the same
      `x1, x2` distribution as in `nngp_fn`. If `None`, will be computed using
      automatic differentiation as `d_nngp_fn = d(nngp_fn)/d(cov12)`, which may
      lead to worse precision or numerical stability. `nngp_fn` and `d_nngp_fn`
      are used to derive the closed-form expression for the NTK.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.

  Raises:
    NotImplementedError: if a `fn`/`nngp_fn` is not provided, but `apply_fn`/
      `kernel_fn` is called respectively.
  """
  if fn is not None:
    name = fn.__name__
  elif nngp_fn is not None:
    name = nngp_fn.__name__
  else:
    raise ValueError('No finite (`fn`) or infinite (`nngp_fn`) functions '
                     'provided, the layer will not do anything.')

  if nngp_fn is None:
    kernel_fn = None

  else:
    if d_nngp_fn is None:
      url = 'https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where'
      warnings.warn(
          f'Using JAX autodiff to compute the `fn` derivative for NTK. Beware '
          f'of {url}.')
      d_nngp_fn = np.vectorize(grad(nngp_fn))

    def kernel_fn(k: Kernel) -> Kernel:
      cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

      var1 = get_diagonal(cov1, k.diagonal_batch, k.diagonal_spatial)
      var2 = get_diagonal(cov2, k.diagonal_batch, k.diagonal_spatial)

      if ntk is not None:
        ntk *= _vmap_2d(d_nngp_fn, nngp, var1, var2, False, k.diagonal_spatial)

      nngp = _vmap_2d(nngp_fn, nngp, var1, var2, False, k.diagonal_spatial)
      cov1 = _vmap_2d(
          nngp_fn, cov1, var1, None, k.diagonal_batch, k.diagonal_spatial)
      if cov2 is not None:
        cov2 = _vmap_2d(
            nngp_fn, cov2, var2, None, k.diagonal_batch, k.diagonal_spatial)
      return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

  return _elementwise(fn, name, kernel_fn)


@layer
@supports_masking(remask_kernel=True)
def ElementwiseNumerical(
    fn: Callable[[float], float],
    deg: int,
    df: Optional[Callable[[float], float]] = None) -> InternalLayer:
  """Activation function using numerical integration.

  Supports general activation functions using Gauss-Hermite quadrature.

  For details, please see "`Fast Neural Kernel Embeddings for General
  Activations <https://arxiv.org/abs/2209.04121>`_".

  See Also:
    `examples/elementwise_numerical.py`.

  Args:
    fn:
      activation function.

    deg:
      number of sample points and weights for quadrature. It must be >= 1.
      We observe for smooth activations `deg=25` is a good place to start.
      For non-smooth activation functions (e.g. ReLU, Abs) quadrature is not
      recommended (for now use `nt.monte_carlo_kernel_fn`). Due to bivariate
      integration, compute time and memory scale as O(deg**2) for more
      precision. See eq (13) in
      https://mathworld.wolfram.com/Hermite-GaussQuadrature.html
      for error estimates in the case of 1d Gauss-Hermite quadrature.

    df:
      optional, derivative of the activation function (`fn`). If not provided,
      it is computed by `jax.grad`. Providing analytic derivative can speed up
      the NTK computations.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  warnings.warn(
      f'Numerical Activation Layer with fn={fn}, deg={deg} used!'
      'Note that numerical error is controlled by `deg` and for a given'
      'tolerance level, required `deg` will highly be dependent on the choice'
      'of `fn`.')

  quad_points = osp.special.roots_hermite(deg)

  if df is None:
    url = 'https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where'
    warnings.warn(
        f'Using JAX autodiff to compute the `fn` derivative for NTK. Beware of '
        f'{url}.')
    df = np.vectorize(grad(fn))

  def kernel_fn(k: Kernel) -> Kernel:
    """Kernel transformation of activation function using quadrature."""
    cov1, nngp, cov2, ntk = k.cov1, k.nngp, k.cov2, k.ntk

    d1 = get_diagonal(cov1, k.diagonal_batch, k.diagonal_spatial)
    d2 = get_diagonal(cov2, k.diagonal_batch, k.diagonal_spatial)

    end_axis = 1 if k.diagonal_spatial else cov1.ndim
    q11 = utils.interleave_ones(d1, 0, end_axis, True)
    q22 = utils.interleave_ones(d1 if d2 is None else d2, 0, end_axis, False)

    def nngp_ntk_fn(nngp, q11, q22, ntk=None):
      """Simple Gauss-Hermite quadrature routine."""
      xs, ws = quad_points
      grid = np.outer(ws, ws)
      x = xs.reshape((xs.shape[0],) + (1,) * (nngp.ndim + 1))
      y = xs.reshape((1, xs.shape[0]) + (1,) * nngp.ndim)
      xy_axes = (0, 1)

      nngp = np.expand_dims(nngp, xy_axes)
      q11, q22 = np.expand_dims(q11, xy_axes), np.expand_dims(q22, xy_axes)

      def integrate(f):
        fvals = f(_sqrt(2 * q11) * x) * f(
            nngp / _sqrt(q11 / 2, 1e-30) * x + _sqrt(
                2*(q22 - nngp**2/q11)) * y)
        return np.tensordot(grid, fvals, (xy_axes, xy_axes)) / np.pi

      if ntk is not None:
        ntk *= integrate(df)
      nngp = integrate(fn)
      return nngp, ntk

    def nngp_fn_diag(nngp):
      xs, ws = quad_points
      x = xs.reshape((xs.shape[0],) + (1,) * nngp.ndim)
      x_axes = (0,)
      nngp = np.expand_dims(nngp, x_axes)
      fval = fn(_sqrt(2 * nngp) * x) ** 2
      return np.tensordot(ws, fval, (x_axes, x_axes)) / np.sqrt(np.pi)

    nngp, ntk = nngp_ntk_fn(nngp, q11, q22, ntk)

    if k.diagonal_batch and k.diagonal_spatial:
      cov1 = nngp_fn_diag(cov1)
      if cov2 is not None:
        cov2 = nngp_fn_diag(cov2)

    else:
      start_axis = 1 if k.diagonal_batch else 0
      q11 = utils.interleave_ones(d1, start_axis, end_axis, True)
      q22 = utils.interleave_ones(d1, start_axis, end_axis, False)
      cov1, _ = nngp_ntk_fn(cov1, q11, q22)

      if cov2 is not None:
        q11 = utils.interleave_ones(d2, start_axis, end_axis, True)
        q22 = utils.interleave_ones(d2, start_axis, end_axis, False)
        cov2, _ = nngp_ntk_fn(cov2, q11, q22)

    return k.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

  return _elementwise(fn, f'ElementwiseNumerical({fn},deg={deg})', kernel_fn)


def _elementwise(
    fn: Optional[Callable[[float], float]],
    name: str,
    kernel_fn: Optional[LayerKernelFn],
) -> InternalLayer:
  init_fn = lambda rng, input_shape: (input_shape, ())

  def apply_fn(params, inputs, **kwargs):
    if fn is None:
      raise NotImplementedError(fn)
    return fn(inputs)

  @requires(diagonal_spatial=Diagonal())
  def new_kernel_fn(k: Kernel, **kwargs) -> Kernel:
    if kernel_fn is None:
      raise NotImplementedError(kernel_fn)

    if not k.is_gaussian:
      raise ValueError('The input to the activation function must be Gaussian, '
                       'i.e. a random affine transform is required before the '
                       'activation function.')
    k = kernel_fn(k)
    return k.replace(is_gaussian=False)

  init_fn.__name__ = apply_fn.__name__ = new_kernel_fn.__name__ = name
  return init_fn, apply_fn, new_kernel_fn


@functools.partial(custom_jvp, nondiff_argnums=(1,))
def _sqrt(x, tol=0.):
  return np.sqrt(np.maximum(x, tol))


@getattr(_sqrt, 'defjvp', lambda f: f)  # ReadTheDocs-friendly `@_sqrt.defjvp`.
def _sqrt_jvp(tol, primals, tangents):
  x, = primals
  x_dot, = tangents
  safe_tol = max(tol, 1e-30)
  square_root = _sqrt(x, safe_tol)
  square_root_out = _sqrt(x, tol)
  return square_root_out, np.where(x > safe_tol, x_dot / (2 * square_root), 0.)


@functools.partial(custom_jvp, nondiff_argnums=(2,))
def _arctan2(x, y, fill_zero: Optional[float] = None):
  if fill_zero is not None:
    return np.where(np.bitwise_and(x == 0., y == 0.),
                    fill_zero,
                    np.arctan2(x, y))
  return np.arctan2(x, y)


@getattr(_arctan2, 'defjvp', lambda f: f)  # Equivalent to `@_arctan2.defjvp`.
def _arctan2_jvp(fill_zero, primals, tangents):
  x, y = primals
  x_dot, y_dot = tangents
  primal_out = _arctan2(x, y, fill_zero)
  safe_tol = 1e-30
  denom = np.maximum(x**2 + y**2, safe_tol)
  tangent_out = x_dot * (y / denom) - y_dot * (x / denom)
  return primal_out, tangent_out


def _vmap_2d(fn: Callable[[float, float, float], float],
             cov12: np.ndarray,
             var1: np.ndarray,
             var2: Optional[np.ndarray],
             diagonal_batch: bool,
             diagonal_spatial: bool) -> np.ndarray:
  """Effectively a "2D vmap" of `fn(cov12, var1, var2)`.

  Applicable for all possible kernel layouts.

  Args:
    fn:
      scalar-valued, elementwise `fn(cov12, var1, var2)` function to apply.

    cov12:
      covariance tensor (`q12`), `nngp`/`ntk`/`cov1`/`cov2`, of shape
      `(N1[, N2])`, `(N1[, N2], X, Y, ...)`, `(N1[, N2], X, X, Y, Y, ...)`
      depending on `diagonal_batch`, `diagonal_spatial`, and the number of
      spatial dimensions.

    var1:
      variance tensor (`q11`), has shape `(N1[, X, Y, ...])`.

    var2:
      variance tensor (`q22`), has shape `(N1[, X, Y, ...])`.

    diagonal_batch:
      `True` if `cov12` has only one batch dimension.

    diagonal_spatial:
      `True` if `cov12` has spatial dimensions appearing once (vs twice).

  Returns:
    Resulting array `[fn(cov12[i, j], var1[i], var2[j])]_{i j}`. Has the same
    shape as `cov12`.
  """
  batch_ndim = 1 if diagonal_batch else 2
  start = 2 - batch_ndim
  cov_end = batch_ndim if diagonal_spatial else cov12.ndim
  _cov12 = utils.make_2d(cov12, start, cov_end)

  var_end = 1 if diagonal_spatial else var1.ndim
  var1 = var1.reshape(var1.shape[:start] + (-1,) + var1.shape[var_end:])
  var2 = var1 if var2 is None else var2.reshape(var2.shape[:start] + (-1,) +
                                                var2.shape[var_end:])

  fn = vmap(
      vmap(
          np.vectorize(fn),
          in_axes=(start, None, start),
          out_axes=start
      ),
      in_axes=(start, start, None),
      out_axes=start
  )
  out = fn(_cov12, var1, var2)  # type: np.ndarray
  out_shape = (cov12.shape[:start] +
               cov12.shape[start:cov_end:2] +
               cov12.shape[start + 1:cov_end:2] +
               cov12.shape[cov_end:])
  out = out.reshape(out_shape)
  out = utils.zip_axes(out, start, cov_end)
  return out


def _factorial(n: int) -> int:
  return functools.reduce(op.mul, range(1, n + 1), 1)


def _double_factorial(n: int) -> int:
  return functools.reduce(op.mul, range(n, 0, -2), 1)
