from absl.testing import absltest
from absl.testing import parameterized

import jax.numpy as np
from math import factorial
import jax.random as random
from experimental.sketching import PolyTensorSketch
from tests import test_utils

NUM_POINTS = [10, 100, 1000]
NUM_DIMS = [64, 256, 1024]


class SketchingTest(test_utils.NeuralTangentsTestCase):

  @classmethod
  def _get_init_data(cls, rng, shape, normalized_output=True):
    x = random.normal(rng, shape)
    if normalized_output:
      return x / np.linalg.norm(x, axis=-1, keepdims=True)
    else:
      return x

  @parameterized.parameters({
      'n': n,
      'd': d,
      'sketch_dim': 1024,
      'degree': 16
  } for n in NUM_POINTS for d in NUM_DIMS)
  def test_exponential_kernel(self, n, d, sketch_dim, degree):
    rng = random.PRNGKey(1)
    x = self._get_init_data(rng, (n, d), True)

    coeffs = np.asarray([1 / factorial(i) for i in range(degree)])

    rng2 = random.PRNGKey(2)
    pts = PolyTensorSketch(rng=rng2,
                           input_dim=d,
                           sketch_dim=sketch_dim,
                           degree=degree).init_sketches()  # pytype:disable=wrong-keyword-args

    x_sketches = pts.sketch(x)

    z = pts.expand_feats(x_sketches, coeffs)
    z = pts.standardsrht(z)
    z = np.concatenate((z.real, z.imag), axis=-1)

    k_exact = np.polyval(coeffs[::-1], x @ x.T)
    k_approx = z @ z.T

    test_utils.assert_close_matrices(self, k_exact, k_approx, 0.15, 1.)


if __name__ == "__main__":
  absltest.main()
