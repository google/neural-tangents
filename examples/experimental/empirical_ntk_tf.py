# Copyright 2022 Google LLC
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

"""Minimal highly-experimental Tensorflow NTK example.

Specifically, Tensorflow NTK appears to have very long compile times (but OK
runtime), is prone to triggering XLA errors, and does not distinguish between
trainable and non-trainable parameters of the model.

For details about the empirical (finite width) NTK computation, please see
"`Fast Finite Width Neural Tangent Kernel <https://arxiv.org/abs/2206.08720>`_".
"""

from absl import app
import neural_tangents as nt
import tensorflow as tf


tf.random.set_seed(1)


def _get_ntks(f, x1, x2, params, vmap_axes):
  """Returns a list of NTKs computed using different implementations."""
  kwargs = dict(
      f=f,
      trace_axes=(),
      vmap_axes=vmap_axes,
  )

  # Default, baseline Jacobian contraction.
  jacobian_contraction = nt.experimental.empirical_ntk_fn_tf(
      **kwargs,
      implementation=nt.NtkImplementation.JACOBIAN_CONTRACTION)
  # (6, 3, 10, 10) full `np.ndarray` test-train NTK
  ntk_jc = jacobian_contraction(x2, x1, params)

  # NTK-vector products-based implementation.
  ntk_vector_products = nt.experimental.empirical_ntk_fn_tf(
      **kwargs,
      implementation=nt.NtkImplementation.NTK_VECTOR_PRODUCTS)
  ntk_vp = ntk_vector_products(x2, x1, params)

  # Structured derivatives-based implementation.
  structured_derivatives = nt.experimental.empirical_ntk_fn_tf(
      **kwargs,
      implementation=nt.NtkImplementation.STRUCTURED_DERIVATIVES)
  ntk_sd = structured_derivatives(x2, x1, params)

  # Auto-FLOPs-selecting implementation. Doesn't work correctly on CPU/GPU.
  auto = nt.experimental.empirical_ntk_fn_tf(
      **kwargs,
      implementation=nt.NtkImplementation.AUTO)
  ntk_auto = auto(x2, x1, params)

  return [ntk_jc, ntk_vp, ntk_sd, ntk_auto]


def _check_ntks(ntks):
  # Check that implementations match
  for ntk1 in ntks:
    for ntk2 in ntks:
      diff = tf.reduce_max(tf.abs(ntk1 - ntk2))
      print(f'NTK implementation diff {diff}.')
      assert diff < 1e-4, diff

  print('All NTK implementations match.')


def _compute_and_check_ntks(f, x1, x2, params):
  ntks = _get_ntks(f, x1, x2, params, vmap_axes=None)
  ntks_vmap = _get_ntks(f, x1, x2, params, vmap_axes=0)
  _check_ntks(ntks + ntks_vmap)


def main(unused_argv):
  x1 = tf.random.normal((6, 8, 8, 3), seed=1)
  x2 = tf.random.normal((3, 8, 8, 3), seed=2)

  # A vanilla CNN `tf.keras.Model` example.
  print('A Keras CNN example.')

  f = tf.keras.Sequential()
  f.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu'))
  f.add(tf.keras.layers.Conv2D(16, (3, 3), activation='relu'))
  f.add(tf.keras.layers.Conv2D(16, (3, 3)))
  f.add(tf.keras.layers.Flatten())
  f.add(tf.keras.layers.Dense(10))

  f.build((None, *x1.shape[1:]))

  _, params = nt.experimental.get_apply_fn_and_params(f)
  _compute_and_check_ntks(f, x1, x2, params)

  # A `tf.function` example.
  print('A `tf.function` example.')

  params_tf = tf.random.normal((1, 2, 3, 4), seed=3)

  @tf.function(input_signature=[tf.TensorSpec(None),
                                tf.TensorSpec((None, *x1.shape[1:]))])
  def f_tf(params, x):
    return tf.transpose(x, (0, 3, 1, 2)) * tf.reduce_mean(params**2) + 1.

  _compute_and_check_ntks(f_tf, x1, x2, params_tf)


if __name__ == '__main__':
  app.run(main)
