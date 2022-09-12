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

"""Example of approximating the NNGP and NTK using quadrature and autodiff.

For details, see :obj:`~neural_tangents.stax.ElementwiseNumerical` and "`Fast
Neural Kernel Embeddings for General Activations
<https://arxiv.org/abs/2209.04121>`_".
"""

from absl import app
from jax import numpy as np
from jax import random
import jax.nn
from neural_tangents import stax


def main(unused_argv):
  key1, key2 = random.split(random.PRNGKey(1))

  x1 = random.normal(key1, (10, 3))
  x2 = random.normal(key2, (20, 3))

  # Consider a nonlinearity for which we know the closed-form expression (GeLU).
  _, _, kernel_fn_closed_form = stax.serial(
      stax.Dense(1),
      stax.Gelu(),  # Contains the closed-form GeLU NNGP/NTK expression.
      stax.Dense(1)
  )
  kernel_closed_form = kernel_fn_closed_form(x1, x2)

  # Construct the layer from only the elementwise forward-pass GeLU.
  _, _, kernel_fn_numerical = stax.serial(
      stax.Dense(1),
      # Approximation using Gaussian quadrature and autodiff.
      stax.ElementwiseNumerical(jax.nn.gelu, deg=25),
      stax.Dense(1)
  )
  kernel_numerical = kernel_fn_numerical(x1, x2)

  # The two kernels are close!
  assert np.max(np.abs(kernel_closed_form.nngp - kernel_numerical.nngp)) < 1e-3
  assert np.max(np.abs(kernel_closed_form.ntk - kernel_numerical.ntk)) < 1e-3
  print('Gaussian quadrature approximation of the kernel is accurate!')


if __name__ == '__main__':
  app.run(main)
