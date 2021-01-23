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

"""An example comparing training a neural network with the NTK dynamics.

In this example, we train a neural network on a small subset of MNIST using an
MSE loss and SGD. We compare this training with the analytic function space
prediction using the NTK. Data is loaded using tensorflow datasets.
"""

from absl import app
from absl import flags
from jax import random
from jax.api import grad
from jax.api import jit
from jax.experimental import optimizers
import jax.numpy as np
import neural_tangents as nt
from neural_tangents import stax
from examples import datasets
from examples import util


flags.DEFINE_float('learning_rate', 1.0,
                   'Learning rate to use during training.')
flags.DEFINE_integer('train_size', 128,
                     'Dataset size to use for training.')
flags.DEFINE_integer('test_size', 128,
                     'Dataset size to use for testing.')
flags.DEFINE_float('train_time', 1000.0,
                   'Continuous time denoting duration of training.')


FLAGS = flags.FLAGS


def main(unused_argv):
  # Build data pipelines.
  print('Loading data.')
  x_train, y_train, x_test, y_test = \
      datasets.get_dataset('mnist', FLAGS.train_size, FLAGS.test_size)

  # Build the network
  init_fn, apply_fn, _ = stax.serial(
      stax.Dense(512, 1., 0.05),
      stax.Erf(),
      stax.Dense(10, 1., 0.05))

  key = random.PRNGKey(0)
  _, params = init_fn(key, (-1, 784))

  # Create and initialize an optimizer.
  opt_init, opt_apply, get_params = optimizers.sgd(FLAGS.learning_rate)
  state = opt_init(params)

  # Create an mse loss function and a gradient function.
  loss = lambda fx, y_hat: 0.5 * np.mean((fx - y_hat) ** 2)
  grad_loss = jit(grad(lambda params, x, y: loss(apply_fn(params, x), y)))

  # Create an MSE predictor to solve the NTK equation in function space.
  ntk = nt.batch(nt.empirical_ntk_fn(apply_fn, vmap_axes=0),
                 batch_size=64, device_count=0)
  g_dd = ntk(x_train, None, params)
  g_td = ntk(x_test, x_train, params)
  predictor = nt.predict.gradient_descent_mse(g_dd, y_train)

  # Get initial values of the network in function space.
  fx_train = apply_fn(params, x_train)
  fx_test = apply_fn(params, x_test)

  # Train the network.
  train_steps = int(FLAGS.train_time // FLAGS.learning_rate)
  print('Training for {} steps'.format(train_steps))

  for i in range(train_steps):
    params = get_params(state)
    state = opt_apply(i, grad_loss(params, x_train, y_train), state)

  # Get predictions from analytic computation.
  print('Computing analytic prediction.')
  fx_train, fx_test = predictor(FLAGS.train_time, fx_train, fx_test, g_td)

  # Print out summary data comparing the linear / nonlinear model.
  util.print_summary('train', y_train, apply_fn(params, x_train), fx_train,
                     loss)
  util.print_summary('test', y_test, apply_fn(params, x_test), fx_test,
                     loss)

if __name__ == '__main__':
  app.run(main)
