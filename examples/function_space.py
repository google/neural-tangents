"""An example comparing training a neural network with the NTK dynamics.

In this example, we train a neural network on a small subset of MNIST using an
MSE loss and SGD. We compare this training with the analytic function space
prediction using the NTK. Data is loaded using tensorflow datasets.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

from jax import random

from jax.api import grad
from jax.api import jit

from jax.experimental import optimizers
from jax.experimental import stax

import jax.numpy as np

from neural_tangents import layers
from neural_tangents import tangents

import datasets
import util

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
      datasets.mnist(FLAGS.train_size, FLAGS.test_size)

  # Build the network
  init_fn, f = stax.serial(
      layers.Dense(4096),
      stax.Tanh,
      layers.Dense(10))

  key = random.PRNGKey(0)
  _, params = init_fn(key, (-1, 784))

  # Create and initialize an optimizer.
  opt_init, opt_apply, get_params = optimizers.sgd(FLAGS.learning_rate)
  state = opt_init(params)

  # Create an mse loss function and a gradient function.
  loss = lambda fx, y_hat: 0.5 * np.mean((fx - y_hat) ** 2)
  grad_loss = jit(grad(lambda params, x, y: loss(f(params, x), y)))

  # Create an MSE predictor to solve the NTK equation in function space.
  theta = tangents.ntk(f, batch_size=32)
  g_dd = theta(params, x_train)
  g_td = theta(params, x_test, x_train)
  predictor = tangents.analytic_mse_predictor(g_dd, y_train, g_td)

  # Get initial values of the network in function space.
  fx_train = f(params, x_train)
  fx_test = f(params, x_test)

  # Train the network.
  train_steps = int(FLAGS.train_time // FLAGS.learning_rate)
  print('Training for {} steps'.format(train_steps))

  for i in range(train_steps):
    params = get_params(state)
    state = opt_apply(i, grad_loss(params, x_train, y_train), state)

  # Get predictions from analytic computation.
  print('Computing analytic prediction.')
  fx_train, fx_test = predictor(fx_train, fx_test, FLAGS.train_time)

  # Print out summary data comparing the linear / nonlinear model.
  util.print_summary('train', y_train, f(params, x_train), fx_train, loss)
  util.print_summary('test', y_test, f(params, x_test), fx_test, loss)

if __name__ == '__main__':
  app.run(main)
