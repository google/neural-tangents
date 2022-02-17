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


import gzip
import os
import shutil
import urllib.request

from jax import random
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def _partial_flatten_and_normalize(x):
  """Flatten all but the first dimension of an `np.ndarray`."""
  x = np.reshape(x, (x.shape[0], -1))
  return (x - np.mean(x)) / np.std(x)


def _one_hot(x, k, dtype=np.float32):
  """Create a one-hot encoding of x of size k."""
  return np.array(x[:, None] == np.arange(k), dtype)


def get_dataset(name,
                n_train=None,
                n_test=None,
                permute_train=False,
                do_flatten_and_normalize=True,
                data_dir=None,
                input_key='image'):
  """Download, parse and process a dataset to unit scale and one-hot labels."""
  # Need this following http://cl/378185881 to prevent GPU test breakages.
  tf.config.set_visible_devices([], 'GPU')

  ds_builder = tfds.builder(name)

  ds_train, ds_test = tfds.as_numpy(
      tfds.load(
          name + (':3.*.*' if name != 'imdb_reviews' else ''),
          split=['train' + ('[:%d]' % n_train if n_train is not None else ''),
                 'test' + ('[:%d]' % n_test if n_test is not None else '')],
          batch_size=-1,
          as_dataset_kwargs={'shuffle_files': False},
          data_dir=data_dir))

  train_images, train_labels, test_images, test_labels = (ds_train[input_key],
                                                          ds_train['label'],
                                                          ds_test[input_key],
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
      permutation = random.permutation(
          split,
          np.arange(x_train.shape[0], dtype=np.int64),
          independent=True
      )
      x_train = x_train[permutation]
      y_train = y_train[permutation]
      epoch += 1
      start = 0
      continue

    yield x_train[start:end], y_train[start:end]
    start = start + batch_size


def embed_glove(xs, glove_path, max_sentence_length=1000, mask_constant=1000.):
  """Embed a list of string arrays into GloVe word embeddings.

  Adapted from https://keras.io/examples/pretrained_word_embeddings/.

  Args:
    xs: list of string numpy arrays to embed.
    glove_path: path to the GloVe embedding file.
    max_sentence_length: pad/truncate embeddings to this length.
    mask_constant: mask padding with this constant.

  Returns:
    xs with words replaced by word embeddings, padded/truncated to a fixed
      length, with padding masked with the given constant.
  """
  xs = list(map(_decode, xs))
  tokenizer = tf.keras.preprocessing.text.Tokenizer()
  tokenizer.fit_on_texts(np.concatenate(xs))
  glove_embedding_layer = _get_glove_embedding_layer(tokenizer,
                                                     glove_path,
                                                     max_sentence_length)

  def embed(x):
    # Replace strings with sequences of integer tokens.
    x_tok = tokenizer.texts_to_sequences(x)
    lenghts = np.array([len(s) for s in x_tok])

    # Pad all sentences to a fixed max sentence length.
    x_tok = tf.keras.preprocessing.sequence.pad_sequences(
        x_tok,
        max_sentence_length,
        padding='post',
        truncating='post')

    # Replace integer tokens with word embeddings.
    x_emb = glove_embedding_layer(x_tok).numpy()

    # Mask padding tokens.
    mask = np.arange(max_sentence_length)[None, :] >= lenghts[:, None]
    x_emb[mask, ...] = mask_constant
    return x_emb

  return map(embed, xs)


def _get_glove_embedding_layer(tokenizer, glove_path, max_sentence_length):
  """Get a Keras embedding layer for a given GloVe embeddings.

  Adapted from https://keras.io/examples/pretrained_word_embeddings/.

  Args:
    tokenizer: the `keras.preprocessing.text.Tokenizer` used to tokenize inputs.
    glove_path: path to the GloVe embedding file.
    max_sentence_length: pad/truncate embeddings to this length.

  Returns:
    Keras embedding layer for a given GloVe embeddings.
  """
  embedding_dim = 50
  word_index = tokenizer.word_index
  print('Loading the embedding model')
  embeddings_index = {}

  if not os.path.exists(glove_path):
    if not os.path.exists(f'{glove_path}.gz'):
      print(f'Did not find {glove_path} word embeddings, downloading...')
      url = 'https://github.com/icml2020-attention/glove/raw/main/glove.6B.50d.txt.gz'
      urllib.request.urlretrieve(url, f'{glove_path}.gz')

    with gzip.open(f'{glove_path}.gz', 'rt') as f_in:
      with open(glove_path, 'wt') as f_out:
        shutil.copyfileobj(f_in, f_out)

  with open(glove_path) as f:
    for line in f:
      word, coefs = line.split(sep=' ', maxsplit=1)
      coefs = np.fromstring(coefs, 'f', sep=' ')
      embeddings_index[word] = coefs

  print(f'Found {len(embeddings_index)} word vectors.')
  print(f'Found {len(word_index)} unique tokens.')
  num_words = len(word_index) + 1

  emb_mat = np.zeros((num_words, embedding_dim))
  for word, i in word_index.items():
    emb_vector = embeddings_index.get(word)
    if emb_vector is not None:
      # words not found in embedding index will be all-zeros.
      emb_mat[i] = emb_vector
    embedding_layer = tf.keras.layers.Embedding(
        num_words, embedding_dim,
        embeddings_initializer=tf.keras.initializers.Constant(emb_mat),
        input_length=max_sentence_length,
        trainable=False)

  return embedding_layer


def _decode(x):
  return np.array([s.decode() for s in x])
