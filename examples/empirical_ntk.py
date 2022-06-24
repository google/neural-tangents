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

"""Example of using different implementations for empirical NTK computation.

All implementations apply to any differentiable functions, (not necessarily ones
constructed with Neural Tangents).

For details about the empirical (finite width) NTK computation, please see
"`Fast Finite Width Neural Tangent Kernel <https://arxiv.org/abs/2206.08720>`_".
"""

from absl import app
import jax
from jax import numpy as np
from jax import random
import neural_tangents as nt
from neural_tangents import stax


def main(unused_argv):
  key1, key2, key3 = random.split(random.PRNGKey(1), 3)
  x1 = random.normal(key1, (2, 8, 8, 3))
  x2 = random.normal(key2, (3, 8, 8, 3))

  # A vanilla CNN.
  init_fn, f, _ = stax.serial(
      stax.Conv(8, (3, 3)),
      stax.Relu(),
      stax.Conv(8, (3, 3)),
      stax.Relu(),
      stax.Conv(8, (3, 3)),
      stax.Flatten(),
      stax.Dense(10)
  )

  _, params = init_fn(key3, x1.shape)
  kwargs = dict(
      f=f,
      trace_axes=(),
      vmap_axes=0,
  )

  # Default, baseline Jacobian contraction.
  jacobian_contraction = nt.empirical_ntk_fn(
      **kwargs,
      implementation=nt.NtkImplementation.JACOBIAN_CONTRACTION)

  # (6, 3, 10, 10) full `np.ndarray` test-train NTK
  ntk_jc = jacobian_contraction(x2, x1, params)

  # NTK-vector products-based implementation.
  ntk_vector_products = nt.empirical_ntk_fn(
      **kwargs,
      implementation=nt.NtkImplementation.NTK_VECTOR_PRODUCTS)

  ntk_vp = ntk_vector_products(x2, x1, params)

  # Structured derivatives-based implementation.
  structured_derivatives = nt.empirical_ntk_fn(
      **kwargs,
      implementation=nt.NtkImplementation.STRUCTURED_DERIVATIVES)

  ntk_sd = structured_derivatives(x2, x1, params)

  # Auto-FLOPs-selecting implementation. Doesn't work correctly on CPU/GPU.
  auto = nt.empirical_ntk_fn(
      **kwargs,
      implementation=nt.NtkImplementation.AUTO)

  ntk_auto = auto(x2, x1, params)

  # Check that implementations match
  for ntk1 in [ntk_jc, ntk_vp, ntk_sd, ntk_auto]:
    for ntk2 in [ntk_jc, ntk_vp, ntk_sd, ntk_auto]:
      diff = np.max(np.abs(ntk1 - ntk2))
      print(f'NTK implementation diff {diff}.')
      assert diff < (1e-4 if jax.default_backend() != 'tpu' else 0.1), diff

  print('All NTK implementations match.')


if __name__ == '__main__':
  app.run(main)
