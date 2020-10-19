"""An example doing inference with an infinitely wide attention network on IMDb.

Adapted from
https://github.com/google/neural-tangents/blob/master/examples/infinite_fcn.py

By default, this example does inference on a very small subset, and uses small
 word embeddings for performance. A 300/300 train/test split takes 30 seconds
 on a machine with 2 Titan X Pascal GPUs, please adjust settings accordingly.
"""

import time
from typing import Tuple

from absl import app
from absl import flags
from jax import random
import jax.numpy as np
import neural_tangents as nt
from neural_tangents import stax
from examples import datasets
from examples import util


flags.DEFINE_integer('n_train', 300,
                     'Dataset size to use for training.')
flags.DEFINE_integer('n_test', 300,
                     'Dataset size to use for testing.')
flags.DEFINE_integer('batch_size', 15,
                     'Batch size for kernel computation. 0 for no batching.')
flags.DEFINE_integer('max_sentence_length', 500,
                     'Pad/truncate sentences to this length.')
flags.DEFINE_string('glove_path',
                    '/tmp/glove.6B.50d.txt',
                    'Path to GloVe word embeddings.')
flags.DEFINE_string('imdb_path',
                    '/tmp/imdb_reviews',
                    'Path to imdb sentences.')


FLAGS = flags.FLAGS


def main(*args, use_dummy_data: bool = False, **kwargs) -> None:
  # Mask all padding with this value.
  mask_constant = 100.

  if use_dummy_data:
    x_train, y_train, x_test, y_test = _get_dummy_data(mask_constant)
  else:
    # Build data pipelines.
    print('Loading IMDb data.')
    x_train, y_train, x_test, y_test = datasets.get_dataset(
        name='imdb_reviews',
        n_train=FLAGS.n_train,
        n_test=FLAGS.n_test,
        do_flatten_and_normalize=False,
        data_dir=FLAGS.imdb_path,
        input_key='text')

    # Embed words and pad / truncate sentences to a fixed size.
    x_train, x_test = datasets.embed_glove(
        xs=[x_train, x_test],
        glove_path=FLAGS.glove_path,
        max_sentence_length=FLAGS.max_sentence_length,
        mask_constant=mask_constant)

  # Build the infinite network.
  # Not using the finite model, hence width is set to 1 everywhere.
  _, _, kernel_fn = stax.serial(
      stax.Conv(out_chan=1, filter_shape=(9,), strides=(1,), padding='VALID'),
      stax.Relu(),
      stax.GlobalSelfAttention(
          n_chan_out=1,
          n_chan_key=1,
          n_chan_val=1,
          pos_emb_type='SUM',
          W_pos_emb_std=1.,
          pos_emb_decay_fn=lambda d: 1 / (1 + d**2),
          n_heads=1),
      stax.Relu(),
      stax.GlobalAvgPool(),
      stax.Dense(out_dim=1)
  )

  # Optionally, compute the kernel in batches, in parallel.
  kernel_fn = nt.batch(kernel_fn, device_count=-1, batch_size=FLAGS.batch_size)

  start = time.time()
  # Bayesian and infinite-time gradient descent inference with infinite network.
  predict = nt.predict.gradient_descent_mse_ensemble(
      kernel_fn=kernel_fn,
      x_train=x_train,
      y_train=y_train,
      diag_reg=1e-6,
      mask_constant=mask_constant)

  fx_test_nngp, fx_test_ntk = predict(x_test=x_test, get=('nngp', 'ntk'))

  fx_test_nngp.block_until_ready()
  fx_test_ntk.block_until_ready()

  duration = time.time() - start
  print(f'Kernel construction and inference done in {duration} seconds.')

  # Print out accuracy and loss for infinite network predictions.
  loss = lambda fx, y_hat: 0.5 * np.mean((fx - y_hat) ** 2)
  util.print_summary('NNGP test', y_test, fx_test_nngp, None, loss)
  util.print_summary('NTK test', y_test, fx_test_ntk, None, loss)


def _get_dummy_data(mask_constant: float
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Return dummy data for when downloading embeddings is not feasible."""
  n_train, n_test = 6, 6

  def get_x(shape, key):
    key_x, key_mask = random.split(key)
    x = random.normal(key_x, shape)
    mask = random.bernoulli(key_mask, 0.6, shape)
    x = np.where(mask, mask_constant, x)
    return x

  def get_y(x):
    x = np.where(x == mask_constant, 0., x)

    def weighted_sum(x, start, end):
      return np.sum(x[..., start:end] *
                    np.arange(x.shape[1])[None, ..., None],
                    axis=(1, 2))

    y_label = np.stack([weighted_sum(x, 0, x.shape[-1] // 2),
                        weighted_sum(x, x.shape[-1] // 2, x.shape[-1])],
                       axis=-1) > 0
    y = np.where(y_label, 0.5, -0.5)
    return y

  rng_train, rng_test = random.split(random.PRNGKey(1), 2)
  x_train = get_x((n_train, FLAGS.max_sentence_length, 50), rng_train)
  x_test = get_x((n_test, FLAGS.max_sentence_length, 50), rng_test)

  y_train, y_test = get_y(x_train), get_y(x_test)
  return x_train, y_train, x_test, y_test


if __name__ == '__main__':
  app.run(main)
