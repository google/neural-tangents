from jax import random
from jax import numpy as np
from neural_tangents._src.utils import dataclasses
from typing import Optional, Callable


# TensorSRHT of degree 2. This version allows different input vectors.
@dataclasses.dataclass
class TensorSRHT:

  input_dim1: int
  input_dim2: int
  sketch_dim: int

  rng: random.KeyArray
  shape: Optional[np.ndarray] = None

  rand_signs1: Optional[np.ndarray] = None
  rand_signs2: Optional[np.ndarray] = None
  rand_inds1: Optional[np.ndarray] = None
  rand_inds2: Optional[np.ndarray] = None

  replace = ...  # type: Callable[..., 'TensorSRHT']

  def init_sketches(self) -> 'TensorSRHT':
    rng1, rng2, rng3, rng4 = random.split(self.rng, 4)
    rand_signs1 = random.choice(rng1, 2, shape=(self.input_dim1,)) * 2 - 1
    rand_signs2 = random.choice(rng2, 2, shape=(self.input_dim2,)) * 2 - 1
    rand_inds1 = random.choice(rng3,
                               self.input_dim1,
                               shape=(self.sketch_dim // 2,))
    rand_inds2 = random.choice(rng4,
                               self.input_dim2,
                               shape=(self.sketch_dim // 2,))
    shape = (self.input_dim1, self.input_dim2, self.sketch_dim)
    return self.replace(shape=shape,
                        rand_signs1=rand_signs1,
                        rand_signs2=rand_signs2,
                        rand_inds1=rand_inds1,
                        rand_inds2=rand_inds2)

  def sketch(self, x1, x2, real_output=False):
    x1fft = np.fft.fftn(x1 * self.rand_signs1, axes=(-1,))[:, self.rand_inds1]
    x2fft = np.fft.fftn(x2 * self.rand_signs2, axes=(-1,))[:, self.rand_inds2]
    out = np.sqrt(1 / self.rand_inds1.shape[-1]) * (x1fft * x2fft)
    return np.concatenate((out.real, out.imag), 1) if real_output else out


# pytype: disable=attribute-error
@dataclasses.dataclass
class PolyTensorSketch:

  input_dim: int
  sketch_dim: int
  degree: int

  rng: random.KeyArray

  tree_rand_signs: Optional[list] = None
  tree_rand_inds: Optional[list] = None
  rand_signs: Optional[np.ndarray] = None
  rand_inds: Optional[np.ndarray] = None

  replace = ...  # type: Callable[..., 'PolyTensorSketch']

  def init_sketches(self) -> 'PolyTensorSketch':

    tree_rand_signs = [0 for i in range((self.degree - 1).bit_length())]
    tree_rand_inds = [0 for i in range((self.degree - 1).bit_length())]
    rng1, rng3 = random.split(self.rng, 2)

    ske_dim_ = self.sketch_dim // 4 - 1
    deg_ = self.degree // 2

    for i in range((self.degree - 1).bit_length()):
      rng1, rng2 = random.split(rng1)

      if i == 0:
        tree_rand_signs[i] = random.choice(
            rng1, 2, shape=(deg_, 2, self.input_dim)) * 2 - 1
        tree_rand_inds[i] = random.choice(rng2,
                                          self.input_dim,
                                          shape=(deg_, 2, ske_dim_))
      else:
        tree_rand_signs[i] = random.choice(rng1, 2,
                                           shape=(deg_, 2, ske_dim_)) * 2 - 1
        tree_rand_inds[i] = random.choice(rng2,
                                          ske_dim_,
                                          shape=(deg_, 2, ske_dim_))
      deg_ = deg_ // 2

    rng1, rng2 = random.split(rng3, 2)
    rand_signs = random.choice(rng1, 2,
                               shape=(1 + self.degree * ske_dim_,)) * 2 - 1
    rand_inds = random.choice(rng2,
                              1 + self.degree * ske_dim_,
                              shape=(self.sketch_dim // 2,))

    return self.replace(tree_rand_signs=tree_rand_signs,
                        tree_rand_inds=tree_rand_inds,
                        rand_signs=rand_signs,
                        rand_inds=rand_inds)

  # TensorSRHT of degree 2
  def tensorsrht(self, x1, x2, rand_inds, rand_signs):
    x1fft = np.fft.fftn(x1 * rand_signs[0, :], axes=(-1,))[:, rand_inds[0, :]]
    x2fft = np.fft.fftn(x2 * rand_signs[1, :], axes=(-1,))[:, rand_inds[1, :]]
    return np.sqrt(1 / rand_inds.shape[1]) * (x1fft * x2fft)

  # Standard SRHT
  def standardsrht(self, x, rand_inds=None, rand_signs=None):
    rand_inds = self.rand_inds if rand_inds is None else rand_inds
    rand_signs = self.rand_signs if rand_signs is None else rand_signs
    xfft = np.fft.fftn(x * rand_signs, axes=(-1,))[:, rand_inds]
    return np.sqrt(1 / rand_inds.shape[0]) * xfft

  def sketch(self, x):
    n = x.shape[0]
    log_degree = len(self.tree_rand_signs)
    V = [0 for i in range(log_degree)]

    for i in range(log_degree):
      deg = self.tree_rand_signs[i].shape[0]
      V[i] = np.zeros((deg, n, self.tree_rand_inds[i].shape[2]),
                      dtype=np.complex64)
      for j in range(deg):
        if i == 0:
          V[i] = V[i].at[j, :, :].set(
              self.tensorsrht(x, x, self.tree_rand_inds[i][j, :, :],
                              self.tree_rand_signs[i][j, :, :]))

        else:
          V[i] = V[i].at[j, :, :].set(
              self.tensorsrht(V[i - 1][2 * j, :, :], V[i - 1][2 * j + 1, :, :],
                              self.tree_rand_inds[i][j, :, :],
                              self.tree_rand_signs[i][j, :, :]))

    U = [0 for i in range(2**log_degree)]
    U[0] = V[log_degree - 1][0, :, :]

    SetE1 = set()

    for j in range(1, len(U)):
      p = (j - 1) // 2
      for i in range(log_degree):
        if j % (2**(i + 1)) == 0:
          SetE1.add((i, p))
        else:
          if i == 0:
            V[i] = V[i].at[p, :, :].set(
                self.standardsrht(x, self.tree_rand_inds[i][p, 0, :],
                                  self.tree_rand_signs[i][p, 0, :]))
          else:
            if (i - 1, 2 * p) in SetE1:
              V[i] = V[i].at[p, :, :].set(V[i - 1][2 * p + 1, :, :])
            else:
              V[i] = V[i].at[p, :, :].set(
                  self.tensorsrht(V[i - 1][2 * p, :, :],
                                  V[i - 1][2 * p + 1, :, :],
                                  self.tree_rand_inds[i][p, :, :],
                                  self.tree_rand_signs[i][p, :, :]))
        p = p // 2
      U[j] = V[log_degree - 1][0, :, :]

    return U

  def expand_feats(self, polysketch_feats, coeffs):
    n, sktch_dim = polysketch_feats[0].shape
    Z = np.zeros((len(self.rand_signs), n), dtype=np.complex64)
    Z = Z.at[0, :].set(np.sqrt(coeffs[0]) * np.ones(n))
    degree = len(polysketch_feats)
    for i in range(degree):
      Z = Z.at[sktch_dim * i + 1:sktch_dim * (i + 1) + 1, :].set(
          np.sqrt(coeffs[i + 1]) * polysketch_feats[degree - i - 1].T)

    return Z.T
# pytype: enable=attribute-error