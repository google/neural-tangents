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

"""Tests for `experimental/empirical_tf/empirical.py`."""

from absl.testing import absltest
from absl.testing import parameterized
import neural_tangents as nt
from neural_tangents import experimental
import numpy as onp
import tensorflow as tf


tf.random.set_seed(1)


_input_signature = [tf.TensorSpec((1, 2, 1, 4)),
                    tf.TensorSpec((None, 2, 3, 1))]


def _f1(params, x):
  return x * tf.reduce_mean(params**2) + 1.


def _f2(params, x):
  return tf.reduce_mean(x) * params**2 + 1.


def _f3(params, x):
  return _f1(params, _f1(params, x)) + tf.reduce_sum(_f2(params, x))


def _f4(params, x):
  return _f1(params, x) + tf.reduce_sum(_f2(params, _f3(params, x)))


# TF module copied from https://www.tensorflow.org/api_docs/python/tf/Module


class _Dense(tf.Module):

  def __init__(self, input_dim, output_size, name=None):
    super(_Dense, self).__init__(name=name)
    self.w = tf.Variable(
        tf.random.normal([input_dim, output_size]), name='w')
    self.b = tf.Variable(tf.zeros([1, output_size]), name='b')

  def __call__(self, x):
    y = tf.matmul(x, self.w) + self.b
    return tf.nn.relu(y)


class _MLP(tf.Module):

  def __init__(self, input_size, sizes, name=None):
    super(_MLP, self).__init__(name=name)
    self.input_shape = (None, input_size)
    self.layers = []
    with self.name_scope:
      for size in sizes:
        self.layers.append(_Dense(input_dim=input_size, output_size=size))
        input_size = size

  @tf.Module.with_name_scope
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x


class EmpiricalTfTest(parameterized.TestCase):

  def _compare_ntks(
      self,
      f,
      params,
      trace_axes,
      diagonal_axes,
      vmap_axes
  ):
    if any(i == j for i in trace_axes for j in diagonal_axes):
      raise absltest.SkipTest('Overlapping trace and diagonal axes.')

    kwargs = dict(
        f=f,
        trace_axes=trace_axes,
        diagonal_axes=diagonal_axes,
    )

    ntk_fns = [
        experimental.empirical_ntk_fn_tf(**kwargs,
                                         implementation=i,
                                         vmap_axes=v)
        for i in nt.NtkImplementation
        for v in vmap_axes if v not in trace_axes + diagonal_axes
    ]

    x_shape = (f.input_shape[1:] if isinstance(f, tf.Module) else
               f.input_signature[1].shape[1:])

    x1 = tf.random.normal((2,) + x_shape, seed=2)
    x2 = tf.random.normal((3,) + x_shape, seed=3)

    ntks = list(enumerate([ntk_fn_i(x1, x2, params)
                           for ntk_fn_i in ntk_fns]))

    for i1, ntk1 in ntks:
      for i2, ntk2 in ntks[i1 + 1:]:
        onp.testing.assert_allclose(ntk1, ntk2, rtol=3e-5, atol=1e-5)

  @parameterized.product(
      f=[
          tf.keras.applications.MobileNet,
      ],
      input_shape=[
          (32, 32, 3)
      ],
      trace_axes=[
          (),
          (1,)
      ],
      diagonal_axes=[
          (),
          (1,)
      ],
      vmap_axes=[
          (0, None)
      ]
  )
  def test_keras_functional(
      self,
      f,
      input_shape,
      trace_axes,
      diagonal_axes,
      vmap_axes,
  ):
    if len(tf.config.list_physical_devices()) != 2:
      # TODO(romann): file bugs on enormous compile time on CPU and TPU.
      raise absltest.SkipTest('Skipping CPU and TPU keras tests.')

    f = f(classes=1, input_shape=input_shape, weights=None)
    f.build((None, *input_shape))
    _, params = experimental.get_apply_fn_and_params(f)
    self._compare_ntks(f, params, trace_axes, diagonal_axes, vmap_axes)

  @parameterized.product(
      input_shape=[
          (16, 16, 3)
      ],
      trace_axes=[
          (),
          (1,)
      ],
      diagonal_axes=[
          (),
          (1,)
      ],
      vmap_axes=[
          (0, None)
      ]
  )
  def test_keras_sequential(
      self,
      input_shape,
      trace_axes,
      diagonal_axes,
      vmap_axes,
  ):
    if tf.config.list_physical_devices('GPU') and diagonal_axes:
      # TODO(romann): figure out the `XlaRuntimeError`.
      raise absltest.SkipTest('RET_CHECK failure')

    f = tf.keras.Sequential()
    f.add(tf.keras.layers.Conv2D(4, (3, 3), activation='relu'))
    f.add(tf.keras.layers.Conv2D(2, (2, 2), activation='relu'))
    f.add(tf.keras.layers.Flatten())
    f.add(tf.keras.layers.Dense(2))

    f.build((None, *input_shape))
    _, params = experimental.get_apply_fn_and_params(f)
    self._compare_ntks(f, params, trace_axes, diagonal_axes, vmap_axes)

  @parameterized.product(
      f=[
          _f1,
          _f2,
          _f3,
          _f4,
      ],
      params_shape=[
          _input_signature[0].shape
      ],
      trace_axes=[
          (),
          (1,)
      ],
      diagonal_axes=[
          (),
          (1,)
      ],
      vmap_axes=[
          (None,)
      ]
  )
  def test_tf_function(
      self,
      f,
      params_shape,
      trace_axes,
      diagonal_axes,
      vmap_axes,
  ):
    f = tf.function(f, input_signature=_input_signature)
    params = tf.random.normal(params_shape, seed=4)
    self._compare_ntks(f, params, trace_axes, diagonal_axes, vmap_axes)

  @parameterized.product(
      trace_axes=[
          (),
          (1,)
      ],
      diagonal_axes=[
          (),
          (1,)
      ],
      vmap_axes=[
          (0, None)
      ]
  )
  def test_module(
      self,
      trace_axes,
      diagonal_axes,
      vmap_axes,
  ):
    f = _MLP(input_size=5, sizes=[4, 6, 3], name='MLP')
    _, params = experimental.get_apply_fn_and_params(f)
    self._compare_ntks(f, params, trace_axes, diagonal_axes, vmap_axes)


if __name__ == '__main__':
  absltest.main()
