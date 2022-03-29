import sys

sys.path.append("./")
import scipy
from jax import random, jit
from jax import numpy as jnp
from experimental.sketching import PolyTensorSketch

# Coefficients of Taylor series of exp(x)
degree = 8
coeffs = jnp.asarray([1 / scipy.special.factorial(i) for i in range(degree)])

n = 4
d = 32
sketch_dim = 256

rng = random.PRNGKey(1)
x = random.normal(rng, shape=(n, d))
norm_x = jnp.linalg.norm(x, axis=-1)
x_normalized = x / norm_x[:, None]

rng2 = random.PRNGKey(2)
pts = PolyTensorSketch(rng=rng2,
                       input_dim=d,
                       sketch_dim=sketch_dim,
                       degree=degree).init_sketches()  # pytype:disable=wrong-keyword-args
x_sketches = pts.sketch(x_normalized)

z = pts.expand_feats(x_sketches, coeffs)  # z.shape[1] is not the desired.
z = pts.standardsrht(z)  # z is complex ndarray.
z = jnp.concatenate((z.real, z.imag), axis=-1)

K = jnp.polyval(coeffs[::-1], x_normalized @ x_normalized.T)
K_approx = z @ z.T

print("Exact kernel matrix:")
print(K)
print()

print(f"Approximate kernel matrix (sketch_dim: {z.shape[1]}):")
print(K_approx)
