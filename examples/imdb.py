"""An example doing inference with an infinitely wide attention network on IMDb.

Adapted from
https://github.com/google/neural-tangents/blob/master/examples/infinite_fcn.py

By default, this example does inference on a very small subset, and uses small
 word embeddings for performance. The below example takes 2-3 minutes on a
 machine with 2 Titan X Pascal GPUs, please adjust settings accordingly.
"""

import time
from absl import app
from absl import flags
import jax.numpy as np
import neural_tangents as nt
from neural_tangents import stax
from examples import datasets
from examples import util


flags.DEFINE_integer('train_size', 500,
                     'Dataset size to use for training.')
flags.DEFINE_integer('test_size', 500,
                     'Dataset size to use for testing.')
flags.DEFINE_integer('batch_size', 10,
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


def main(unused_argv):
  # Build data pipelines.
  print('Loading IMDb data.')
  x_train, y_train, x_test, y_test = \
    datasets.get_dataset('imdb_reviews', FLAGS.train_size, FLAGS.test_size,
                         do_flatten_and_normalize=False,
                         data_dir=FLAGS.imdb_path,
                         input_key='text')

  # Mask all padding with this value.
  mask_constant = 100.

  # Embed words and pad / truncate sentences to a fixed size.
  x_train, x_test = datasets.embed_glove([x_train, x_test],
      FLAGS.glove_path, FLAGS.max_sentence_length, mask_constant)

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
          W_pos_emb_std=2.,
          n_heads=1),
      stax.LayerNorm(),
      stax.Relu(),
      stax.GlobalAvgPool(),
      stax.Dense(out_dim=1, W_std=1., b_std=0.)
  )

  # Optionally, compute the kernel in batches, in parallel.
  kernel_fn = nt.batch(kernel_fn, device_count=-1, batch_size=FLAGS.batch_size)

  start = time.time()
  # Bayesian and infinite-time gradient descent inference with infinite network.
  fx_test_nngp, fx_test_ntk = nt.predict.gp_inference(
      kernel_fn,
      x_train,
      y_train,
      x_test,
      get=('nngp', 'ntk'),
      diag_reg=1e-2,
      mask_constant=mask_constant
  )
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
