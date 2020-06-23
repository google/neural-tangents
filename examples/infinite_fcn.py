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
from examples import datasets
from examples import util


flags.DEFINE_integer('train_size', 1000,
                     'Dataset size to use for training.')
flags.DEFINE_integer('test_size', 1000,
                     'Dataset size to use for testing.')
flags.DEFINE_integer('batch_size', 0,
                     'Batch size for kernel computation. 0 for no batching.')


FLAGS = flags.FLAGS


def main(unused_argv):
  # Build data pipelines.
  print('Loading data.')
  x_train, y_train, x_test, y_test = \
    datasets.get_dataset('cifar10', FLAGS.train_size, FLAGS.test_size)

  # Build the infinite network.
  _, _, kernel_fn = stax.serial(
      stax.Dense(1, 2., 0.05),
      stax.Relu(),
      stax.Dense(1, 2., 0.05)
  )

  # Optionally, compute the kernel in batches, in parallel.
  kernel_fn = nt.batch(kernel_fn,
                       device_count=0,
                       batch_size=FLAGS.batch_size)

  start = time.time()
  # Bayesian and infinite-time gradient descent inference with infinite network.
  predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, x_train,
                                                        y_train, diag_reg=1e-3)
  fx_test_nngp, fx_test_ntk = predict_fn(x_test=x_test)
  fx_test_nngp.block_until_ready()
  fx_test_ntk.block_until_ready()

  duration = time.time() - start
  print('Kernel construction and inference done in %s seconds.' % duration)

  # Print out accuracy and loss for infinite network predictions.
  loss = lambda fx, y_hat: 0.5 * np.mean((fx - y_hat) ** 2)
  util.print_summary('NNGP test', y_test, fx_test_nngp, None, loss)
  util.print_summary('NTK test', y_test, fx_test_ntk, None, loss)


if __name__ == '__main__':
  app.run(main)
