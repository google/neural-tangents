from jax import random
from jax import numpy as np
from jax.numpy.fft import fftn
from neural_tangents._src.utils import dataclasses
from typing import Optional, Callable


def _random_signs_indices(rngs, input_dim, output_dim, shape=()):
  rand_signs = random.bernoulli(rngs[0], shape=shape + (input_dim,)) * 2 - 1.
  rand_inds = random.choice(rngs[1], input_dim, shape=shape + (output_dim,))
  return rand_signs, rand_inds


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
    rand_signs1, rand_inds1 = _random_signs_indices(
        (rng1, rng3), self.input_dim1, self.sketch_dim // 2)
    rand_signs2, rand_inds2 = _random_signs_indices(
        (rng2, rng4), self.input_dim2, self.sketch_dim // 2)
    shape = (self.input_dim1, self.input_dim2, self.sketch_dim)
    return self.replace(shape=shape,
                        rand_signs1=rand_signs1,
                        rand_signs2=rand_signs2,
                        rand_inds1=rand_inds1,
                        rand_inds2=rand_inds2)

  def sketch(self, x1, x2, real_output=False):
    x1fft = fftn(x1 * self.rand_signs1[None, :], axes=(-1,))[:, self.rand_inds1]
    x2fft = fftn(x2 * self.rand_signs2[None, :], axes=(-1,))[:, self.rand_inds2]
    out = (x1fft * x2fft) / self.rand_inds1.shape[-1]**0.5
    return np.concatenate((out.real, out.imag), 1) if real_output else out


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
    height = (self.degree - 1).bit_length()
    tree_rand_signs = [0] * height
    tree_rand_inds = [0] * height
    rng1, rng3 = random.split(self.rng, 2)

    internal_sketch_dim = self.sketch_dim // 4 - 1
    degree = self.degree // 2

    for lvl in range(height):
      rng1, rng2 = random.split(rng1)

      input_dim = self.input_dim if lvl == 0 else internal_sketch_dim
      tree_rand_signs[lvl], tree_rand_inds[lvl] = _random_signs_indices(
          (rng1, rng2), input_dim, internal_sketch_dim, (degree, 2))

      degree = degree // 2

    rng1, rng2 = random.split(rng3, 2)
    rand_signs, rand_inds = _random_signs_indices(
        (rng1, rng2), 1 + self.degree * internal_sketch_dim,
        self.sketch_dim // 2)

    return self.replace(tree_rand_signs=tree_rand_signs,
                        tree_rand_inds=tree_rand_inds,
                        rand_signs=rand_signs,
                        rand_inds=rand_inds)

  # TensorSRHT of degree 2
  def tensorsrht(self, x1, x2, rand_inds, rand_signs):
    x1fft = fftn(x1 * rand_signs[:1, :], axes=(-1,))[:, rand_inds[0, :]]
    x2fft = fftn(x2 * rand_signs[1:, :], axes=(-1,))[:, rand_inds[1, :]]
    return rand_inds.shape[1]**(-0.5) * (x1fft * x2fft)

  # Standard SRHT
  def standardsrht(self, x, rand_inds=None, rand_signs=None):
    rand_inds = self.rand_inds if rand_inds is None else rand_inds
    rand_signs = self.rand_signs if rand_signs is None else rand_signs
    xfft = fftn(x * rand_signs[None, :], axes=(-1,))[:, rand_inds]
    return rand_inds.shape[0]**(-0.5) * xfft

  def sketch(self, x):
    n = x.shape[0]
    dtype = np.complex64 if x.real.dtype == np.float32 else np.complex128

    height = len(self.tree_rand_signs)
    V = [np.zeros(())] * height

    for lvl in range(height):
      deg = self.tree_rand_signs[lvl].shape[0]
      output_dim = self.tree_rand_inds[lvl].shape[2]
      V[lvl] = np.zeros((deg, n, output_dim), dtype=dtype)
      for j in range(deg):
        if lvl == 0:
          x1, x2 = x, x
        else:
          x1, x2 = V[lvl - 1][2 * j, :, :], V[lvl - 1][2 * j + 1, :, :]

        V[lvl] = V[lvl].at[j, :, :].set(
            self.tensorsrht(x1, x2, self.tree_rand_inds[lvl][j, :, :],
                            self.tree_rand_signs[lvl][j, :, :]))

    U = [np.zeros(())] * 2**height
    U[0] = V[-1][0, :, :]

    SetE1 = set()

    for j in range(1, 2**height):
      p = (j - 1) // 2
      for lvl in range(height):
        if j % (2**(lvl + 1)) == 0:
          SetE1.add((lvl, p))
        else:
          if lvl == 0:
            V[lvl] = V[lvl].at[p, :, :].set(
                self.standardsrht(x, self.tree_rand_inds[lvl][p, 0, :],
                                  self.tree_rand_signs[lvl][p, 0, :]))
          else:
            if (lvl - 1, 2 * p) in SetE1:
              V[lvl] = V[lvl].at[p, :, :].set(V[lvl - 1][2 * p + 1, :, :])
            else:
              V[lvl] = V[lvl].at[p, :, :].set(
                  self.tensorsrht(V[lvl - 1][2 * p, :, :],
                                  V[lvl - 1][2 * p + 1, :, :],
                                  self.tree_rand_inds[lvl][p, :, :],
                                  self.tree_rand_signs[lvl][p, :, :]))
        p = p // 2
      U[j] = V[-1][0, :, :]

    return U

  def expand_feats(self, sketches, coeffs):
    n = sketches[0].shape[0]
    degree = len(sketches)
    return np.concatenate(
        [coeffs[0]**0.5 * np.ones((n, 1))] +
        [coeffs[i + 1]**0.5 * sketches[-i - 1] for i in range(degree)],
        axis=-1)