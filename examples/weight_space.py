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

"""An example comparing training a neural network with its linearization.

In this example we train a neural network and a linear model corresponding to
the first order Taylor seres of the network about its initial parameters. The
network is a fully-connected network with one hidden layer. We use momentum and
minibatching on the full MNIST dataset. Data is loaded using tensorflow
datasets.
"""


from absl import app
from absl import flags
from jax import random
from jax.api import grad
from jax.api import jit
from jax.experimental import optimizers
from jax.experimental.stax import logsoftmax
import jax.numpy as np
import neural_tangents as nt
from neural_tangents import stax
from examples import datasets
from examples import util


flags.DEFINE_float('learning_rate', 1.0,
                   'Learning rate to use during training.')
flags.DEFINE_integer('batch_size', 128,
                     'Batch size to use during training.')
flags.DEFINE_integer('train_epochs', 10,
                     'Number of epochs to train for.')


FLAGS = flags.FLAGS


def main(unused_argv):
  # Build data and .
  print('Loading data.')
  x_train, y_train, x_test, y_test = datasets.get_dataset('mnist',
                                                          permute_train=True)

  # Build the network
  init_fn, f, _ = stax.serial(
      stax.Dense(512, 1., 0.05),
      stax.Erf(),
      stax.Dense(10, 1., 0.05))

  key = random.PRNGKey(0)
  _, params = init_fn(key, (-1, 784))

  # Linearize the network about its initial parameters.
  f_lin = nt.linearize(f, params)

  # Create and initialize an optimizer for both f and f_lin.
  opt_init, opt_apply, get_params = optimizers.momentum(FLAGS.learning_rate,
                                                        0.9)
  opt_apply = jit(opt_apply)

  state = opt_init(params)
  state_lin = opt_init(params)

  # Create a cross-entropy loss function.
  loss = lambda fx, y_hat: -np.mean(logsoftmax(fx) * y_hat)

  # Specialize the loss function to compute gradients for both linearized and
  # full networks.
  grad_loss = jit(grad(lambda params, x, y: loss(f(params, x), y)))
  grad_loss_lin = jit(grad(lambda params, x, y: loss(f_lin(params, x), y)))

  # Train the network.
  print('Training.')
  print('Epoch\tLoss\tLinearized Loss')
  print('------------------------------------------')

  epoch = 0
  steps_per_epoch = 50000 // FLAGS.batch_size

  for i, (x, y) in enumerate(datasets.minibatch(
      x_train, y_train, FLAGS.batch_size, FLAGS.train_epochs)):

    params = get_params(state)
    state = opt_apply(i, grad_loss(params, x, y), state)

    params_lin = get_params(state_lin)
    state_lin = opt_apply(i, grad_loss_lin(params_lin, x, y), state_lin)

    if i % steps_per_epoch == 0:
      print('{}\t{:.4f}\t{:.4f}'.format(
          epoch, loss(f(params, x), y), loss(f_lin(params_lin, x), y)))
      epoch += 1

  # Print out summary data comparing the linear / nonlinear model.
  x, y = x_train[:10000], y_train[:10000]
  util.print_summary('train', y, f(params, x), f_lin(params_lin, x), loss)
  util.print_summary(
      'test', y_test, f(params, x_test), f_lin(params_lin, x_test), loss)

if __name__ == '__main__':
  app.run(main)
