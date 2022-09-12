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

"""Example of automatically deriving the closed-form NTK from NNGP.

For details, see :obj:`~neural_tangents.stax.Elementwise` and "`Fast Neural
Kernel Embeddings for General Activations <https://arxiv.org/abs/2209.04121>`_".
"""

from absl import app
from jax import numpy as np
from jax import random
from neural_tangents import stax


def main(unused_argv):
  # Consider the normalized exponential kernel from
  # https://arxiv.org/abs/2003.02237 (page 6).
  def nngp_fn(cov12, var1, var2):
    prod = np.sqrt(var1 * var2)
    return prod * np.exp(cov12 / prod - 1)

  # This kernel has no known corresponding elementwise nonlinearity.
  # `stax.Elementwise` derives the NTK kernel automatically under the hood using
  # automatic differentiation, without the need to know the respective
  # nonlinearity or computing the integrals by hand.
  _, _, kernel_fn = stax.serial(stax.Dense(1),
                                stax.Elementwise(nngp_fn=nngp_fn))

  # Below we construct the kernel using the manually-derived NTK expression.
  _, _, kernel_fn_manual = stax.serial(stax.Dense(1),
                                       stax.ExpNormalized())

  key = random.PRNGKey(1)
  x1 = random.normal(key, (10, 2))
  x2 = random.normal(key, (20, 2))

  k_auto = kernel_fn(x1, x2, 'ntk')
  k_manual = kernel_fn_manual(x1, x2, 'ntk')

  # The two kernels match!
  assert np.max(np.abs(k_manual - k_auto)) < 1e-6
  print('NTK derived via autodiff matches the hand-derived NTK!')


if __name__ == '__main__':
  app.run(main)
