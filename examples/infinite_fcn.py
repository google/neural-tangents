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

"""An example doing inference with an infinitely wide fully-connected network.

By default, this example does inference on a small CIFAR10 subset.
"""

import time
from absl import app
from absl import flags
import jax.numpy as np
import neural_tangents as nt
from neural_tangents import stax
from jax import random


flags.DEFINE_integer('train_size', 1000,
                     'Dataset size to use for training.')
flags.DEFINE_integer('test_size', 1000,
                     'Dataset size to use for testing.')
flags.DEFINE_integer('batch_size', 0,
                     'Batch size for kernel computation. 0 for no batching.')


FLAGS = flags.FLAGS

import pdb

def main(unused_argv):
  # Build data pipelines.
  print('Loading data.')
  key = random.PRNGKey(0)
  key, split = random.split(key)
  x_train = random.normal(key=key, shape=[10, 30])
  x_train2 = random.normal(key=split, shape=[10, 30])

  # Build the infinite network.
  init_fn, apply_fn, kernel_fn = stax.serial(
      stax.Dense(1000, 2., 0.05),
      stax.BatchNormRelu(0),
      stax.Dense(1000, 2., 0.05)
  )

  mc_kernel_fn = nt.monte_carlo_kernel_fn(init_fn, apply_fn, key, 1000)
  kerobj = kernel_fn(x_train, x_train2)
  theory_ker = kerobj.nngp
  diff = theory_ker - mc_kernel_fn(x_train, x_train2, get='nngp')
  print(diff)
  # print(kerobj.cov1 - kerobj.nngp)
  print(np.linalg.norm(diff) / np.linalg.norm(theory_ker))
  return


if __name__ == '__main__':
  app.run(main)
