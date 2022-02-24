from jax import random
from jax import numpy as np
from neural_tangents._src.utils import utils, dataclasses
from neural_tangents._src.utils.typing import Optional


# TensorSRHT of degree 2. This version allows different input vectors.
@dataclasses.dataclass
class TensorSRHT2:

  input_dim1: int
  input_dim2: int
  sketch_dim: int

  rng: np.ndarray
  shape: Optional[np.ndarray] = None

  rand_signs1: Optional[np.ndarray] = None
  rand_signs2: Optional[np.ndarray] = None
  rand_inds1: Optional[np.ndarray] = None
  rand_inds2: Optional[np.ndarray] = None

  replace = ...

  def init_sketches(self):
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

  def sketch(self, x1, x2):
    x1fft = np.fft.fftn(x1 * self.rand_signs1, axes=(-1,))[:, self.rand_inds1]
    x2fft = np.fft.fftn(x2 * self.rand_signs2, axes=(-1,))[:, self.rand_inds2]
    out = np.sqrt(1 / self.rand_inds1.shape[-1]) * (x1fft * x2fft)
    return np.concatenate((out.real, out.imag), 1)


# Function implementation of TensorSRHT of degree 2 (duplicated)
def tensorsrht(x1, x2, rand_inds, rand_signs):
  x1fft = np.fft.fftn(x1 * rand_signs[0, :], axes=(-1,))[:, rand_inds[0, :]]
  x2fft = np.fft.fftn(x2 * rand_signs[1, :], axes=(-1,))[:, rand_inds[1, :]]
  return np.sqrt(1 / rand_inds.shape[1]) * (x1fft * x2fft)


# TensorSRHT of degree p. This operates the same input vectors.
class PolyTensorSRHT:

  def __init__(self, rng, input_dim, sketch_dim, coeffs):
    self.coeffs = coeffs
    degree = len(coeffs) - 1
    self.degree = degree

    self.tree_rand_signs = [0 for i in range((self.degree - 1).bit_length())]
    self.tree_rand_inds = [0 for i in range((self.degree - 1).bit_length())]
    rng1, rng2, rng3 = random.split(rng, 3)

    ske_dim_ = sketch_dim // 4
    deg_ = degree // 2
    for i in range((degree - 1).bit_length()):
      rng1, rng2 = random.split(rng1)
      if i == 0:
        self.tree_rand_signs[i] = random.choice(
            rng1, 2, shape=(deg_, 2, input_dim)) * 2 - 1
        self.tree_rand_inds[i] = random.choice(rng2,
                                               input_dim,
                                               shape=(deg_, 2, ske_dim_))
      else:
        self.tree_rand_signs[i] = random.choice(
            rng1, 2, shape=(deg_, 2, ske_dim_)) * 2 - 1
        self.tree_rand_inds[i] = random.choice(rng2,
                                               ske_dim_,
                                               shape=(deg_, 2, ske_dim_))
      deg_ = deg_ // 2

    rng1, rng2 = random.split(rng3)
    self.rand_signs = random.choice(rng1, 2, shape=(degree * ske_dim_,)) * 2 - 1
    self.rand_inds = random.choice(rng2,
                                   degree * ske_dim_,
                                   shape=(sketch_dim // 2,))

  def sketch(self, x):
    n = x.shape[0]
    log_degree = len(self.tree_rand_signs)
    V = [0 for i in range(log_degree)]
    E1 = np.concatenate((np.ones(
        (n, 1), dtype=x.dtype), np.zeros((n, x.shape[-1] - 1), dtype=x.dtype)),
                        1)
    for i in range(log_degree):
      deg = self.tree_rand_signs[i].shape[0]
      V[i] = np.zeros((deg, n, self.tree_rand_inds[i].shape[2]),
                      dtype=np.complex64)
      for j in range(deg):
        if i == 0:
          V[i] = V[i].at[j, :, :].set(
              tensorsrht(x, x, self.tree_rand_inds[i][j, :, :],
                         self.tree_rand_signs[i][j, :, :]))
        else:
          V[i] = V[i].at[j, :, :].set(
              tensorsrht(V[i - 1][2 * j, :, :], V[i - 1][2 * j + 1, :, :],
                         self.tree_rand_inds[i][j, :, :],
                         self.tree_rand_signs[i][j, :, :]))
    U = [0 for i in range(2**log_degree)]
    U[0] = V[log_degree - 1][0, :, :].clone()

    for j in range(1, len(U)):
      p = (j - 1) // 2
      for i in range(log_degree):
        if j % (2**(i + 1)) == 0:
          V[i] = V[i].at[p, :, :].set(
              np.concatenate((np.ones((n, 1)), np.zeros(
                  (n, V[i].shape[-1] - 1))), 1))
        else:
          if i == 0:
            V[i] = V[i].at[p, :, :].set(
                tensorsrht(x, E1, self.tree_rand_inds[i][p, :, :],
                           self.tree_rand_signs[i][p, :, :]))
          else:
            V[i] = V[i].at[p, :, :].set(
                tensorsrht(V[i - 1][2 * p, :, :], V[i - 1][2 * p + 1, :, :],
                           self.tree_rand_inds[i][p, :, :],
                           self.tree_rand_signs[i][p, :, :]))
          p = p // 2
      U[j] = V[log_degree - 1][0, :, :].clone()
    return U
