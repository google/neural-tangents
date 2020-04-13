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

"""Datasets used in examples."""


from jax import random
import numpy as np
import tensorflow_datasets as tfds


def _partial_flatten_and_normalize(x):
  """Flatten all but the first dimension of an `np.ndarray`."""
  x = np.reshape(x, (x.shape[0], -1))
  return (x - np.mean(x)) / np.std(x)


def _one_hot(x, k, dtype=np.float32):
  """Create a one-hot encoding of x of size k."""
  return np.array(x[:, None] == np.arange(k), dtype)


def get_dataset(name, n_train=None, n_test=None, permute_train=False,
                do_flatten_and_normalize=True):
  """Download, parse and process a dataset to unit scale and one-hot labels."""
  ds_builder = tfds.builder(name)

  ds_train, ds_test = tfds.as_numpy(
      tfds.load(
          name + ':3.*.*',
          split=['train' + ('[:%d]' % n_train if n_train is not None else ''),
                 'test' + ('[:%d]' % n_test if n_test is not None else '')],
          batch_size=-1,
          as_dataset_kwargs={'shuffle_files': False}))

  train_images, train_labels, test_images, test_labels = (ds_train['image'],
                                                          ds_train['label'],
                                                          ds_test['image'],
                                                          ds_test['label'])

  if do_flatten_and_normalize:
    train_images = _partial_flatten_and_normalize(train_images)
    test_images = _partial_flatten_and_normalize(test_images)

  num_classes = ds_builder.info.features['label'].num_classes
  train_labels = _one_hot(train_labels, num_classes)
  test_labels = _one_hot(test_labels, num_classes)

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
      permutation = random.shuffle(split,
                                   np.arange(x_train.shape[0], dtype=np.int64))
      x_train = x_train[permutation]
      y_train = y_train[permutation]
      epoch += 1
      start = 0
      continue

    yield x_train[start:end], y_train[start:end]
    start = start + batch_size
