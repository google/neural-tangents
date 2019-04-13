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

"""Datasets used in examples.

This code was adapted from JAX, with permission.
https://github.com/google/jax/examples/datasets.py [Visited on 04/10/2019]
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import array
import gzip
import os
from os import path
import struct
from six.moves.urllib.request import urlretrieve

import numpy as np
from jax import random


# NOTE(schsam): We could change this, but is there really a point? This
# directory will basically only ever contain MNIST data downloaded by JAX users.
_DATA = "/tmp/jax_example_data/"


def _download(url, filename):
  """Download a url to a file in the JAX data temp directory."""
  if not path.exists(_DATA):
    os.makedirs(_DATA)
  out_file = path.join(_DATA, filename)
  if not path.isfile(out_file):
    urlretrieve(url, out_file)
    print("downloaded {} to {}".format(url, _DATA))

def _partial_flatten_and_normalize(x):
  """Flatten all but the first dimension of an ndarray."""
  x = np.reshape(x, (x.shape[0], -1))
  return (x - np.mean(x)) / np.std(x)

def _one_hot(x, k, dtype=np.float32):
  """Create a one-hot encoding of x of size k."""
  return np.array(x[:, None] == np.arange(k), dtype)

def mnist_raw():
  """Download and parse the raw MNIST dataset."""
  # CVDF mirror of http://yann.lecun.com/exdb/mnist/
  base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

  def parse_labels(filename):
    with gzip.open(filename, "rb") as fh:
      _ = struct.unpack(">II", fh.read(8))
      return np.array(array.array("B", fh.read()), dtype=np.uint8)

  def parse_images(filename):
    with gzip.open(filename, "rb") as fh:
      _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
      return np.array(array.array("B", fh.read()),
                      dtype=np.uint8).reshape(num_data, rows, cols)

  for filename in ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                   "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]:
    _download(base_url + filename, filename)

  train_images = parse_images(path.join(_DATA, "train-images-idx3-ubyte.gz"))
  train_labels = parse_labels(path.join(_DATA, "train-labels-idx1-ubyte.gz"))
  test_images = parse_images(path.join(_DATA, "t10k-images-idx3-ubyte.gz"))
  test_labels = parse_labels(path.join(_DATA, "t10k-labels-idx1-ubyte.gz"))

  return train_images, train_labels, test_images, test_labels

def mnist(n_train=None, n_test=None, permute_train=False):
  """Download, parse and process MNIST data to unit scale and one-hot labels."""
  train_images, train_labels, test_images, test_labels = mnist_raw()

  train_images = _partial_flatten_and_normalize(train_images)
  test_images = _partial_flatten_and_normalize(test_images)
  train_labels = _one_hot(train_labels, 10)
  test_labels = _one_hot(test_labels, 10)

  if n_train is not None:
    train_images = train_images[:n_train]
    train_labels = train_labels[:n_train]
  if n_test is not None:
    test_images = test_images[:n_test]
    test_labels = test_labels[:n_test]

  if permute_train:
    perm = np.random.RandomState(0).permutation(train_images.shape[0])
    train_images = train_images[perm]
    train_labels = train_labels[perm]

  return train_images, train_labels, test_images, test_labels

def minibatch(x_train, y_train, batch_size, train_epochs):
  """Generate minibatches of data for a set number of epochs."""
  epoch = 0
  start = 0
  key = random.PRNGKey(0)

  while epoch < train_epochs:
    end = start + batch_size

    if end > x_train.shape[0]:
      key, split = random.split(key)
      permutation = random.shuffle(
          split, np.arange(x_train.shape[0], dtype=np.int64))
      x_train = x_train[permutation]
      y_train = y_train[permutation]
      epoch = epoch + 1
      start = 0
      continue

    yield x_train[start:end], y_train[start:end]
    start = start + batch_size
