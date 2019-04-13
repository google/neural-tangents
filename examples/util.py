"""A set of utility operations for running examples.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax.numpy as np


def _accuracy(y, y_hat):
  """Compute the accuracy of the predictions with respect to one-hot labels."""
  return np.mean(np.argmax(y, axis=1) == np.argmax(y_hat, axis=1))


def print_summary(name, labels, net_p, lin_p, loss):
  """Print summary information comparing a network with its linearization."""
  print('\nEvaluating Network on {} data.'.format(name))
  print('---------------------------------------')
  print('RMSE of predictions: {}'.format(
      np.sqrt(np.mean((net_p - lin_p) ** 2))))
  print('Network Accuracy = {}'.format(_accuracy(net_p, labels)))
  print('Linearization Accuracy = {}'.format(_accuracy(lin_p, labels)))
  print('Network Loss = {}'.format(loss(net_p, labels)))
  print('Linearization Loss = {}'.format(loss(lin_p, labels)))
  print('---------------------------------------')
