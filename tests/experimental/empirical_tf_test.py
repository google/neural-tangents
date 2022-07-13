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
import jax
from jax import numpy as np
import neural_tangents as nt
from neural_tangents import experimental
import numpy as onp
import tensorflow as tf


tf.random.set_seed(1)


# TF module copied from https://www.tensorflow.org/api_docs/python/tf/Module


class _Dense(tf.Module):

  def __init__(self, input_dim, output_size, name=None):
    super(_Dense, self).__init__(name=name)
    self.w = tf.Variable(
        tf.random.normal([input_dim, output_size]), name='w')
    self.b = tf.Variable(tf.zeros([1, output_size]), name='b')

  def __call__(self, x):
    y = tf.matmul(x, self.w) / x.shape[-1]**0.5 + self.b
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


# Functions to compare TF/JAX manually.


_input_signature = [tf.TensorSpec((1, 2, 1, 4)),
                    tf.TensorSpec((None, 2, 3, 1))]


def _f1(params, x):
  return x * tf.reduce_mean(params**2) + 1.


def _f1_jax(params, x):
  return x * np.mean(params**2) + 1.


def _f2(params, x):
  return tf.reduce_mean(x) * params**2 + 1.


def _f2_jax(params, x):
  return np.mean(x) * params**2 + 1.


def _f3(params, x):
  return _f1(params, _f1(params, x)) + tf.reduce_mean(_f2(params, x))


def _f3_jax(params, x):
  return _f1_jax(params, _f1_jax(params, x)) + np.mean(_f2_jax(params, x))


def _f4(params, x):
  return _f1(params, x) + tf.reduce_mean(_f2(params, _f3(params, x)))


def _f4_jax(params, x):
  return _f1_jax(params, x) + np.mean(_f2_jax(params, _f3_jax(params, x)))


# ResNet18 adapted from
# https://github.com/jimmyyhwu/resnet18-tf2/blob/master/resnet.py


_kaiming_normal = tf.keras.initializers.VarianceScaling(
    scale=2.0, mode='fan_out', distribution='untruncated_normal')


def _conv3x3(x, out_planes, stride=1, name=None):
  x = tf.keras.layers.ZeroPadding2D(padding=1, name=f'{name}_pad')(x)
  return tf.keras.layers.Conv2D(
      filters=out_planes, kernel_size=3, strides=stride, use_bias=False,
      kernel_initializer=_kaiming_normal, name=name)(x)


def _basic_block(x, planes, stride=1, downsample=None, name=None):
  identity = x

  out = _conv3x3(x, planes, stride=stride, name=f'{name}.conv1')
  out = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5,
                                           name=f'{name}.bn1')(out)
  out = tf.keras.layers.ReLU(name=f'{name}.relu1')(out)

  out = _conv3x3(out, planes, name=f'{name}.conv2')
  out = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5,
                                           name=f'{name}.bn2')(out)

  if downsample is not None:
    for layer in downsample:
      identity = layer(identity)

  out = tf.keras.layers.Add(name=f'{name}.add')([identity, out])
  out = tf.keras.layers.ReLU(name=f'{name}.relu2')(out)

  return out


def _make_layer(x, planes, blocks, stride=1, name=None):
  downsample = None
  inplanes = x.shape[3]
  if stride != 1 or inplanes != planes:
    downsample = [
        tf.keras.layers.Conv2D(
            filters=planes, kernel_size=1, strides=stride,
            use_bias=False, kernel_initializer=_kaiming_normal,
            name=f'{name}.0.downsample.0'),
        tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5,
                                           name=f'{name}.0.downsample.1'),
    ]

  x = _basic_block(x, planes, stride, downsample, name=f'{name}.0')
  for i in range(1, blocks):
    x = _basic_block(x, planes, name=f'{name}.{i}')

  return x


def _resnet(x, blocks_per_layer, classes, filters):
  x = tf.keras.layers.ZeroPadding2D(padding=3, name='conv1_pad')(x)
  x = tf.keras.layers.Conv2D(
      filters=filters, kernel_size=7, strides=2, use_bias=False,
      kernel_initializer=_kaiming_normal, name='conv1')(x)
  x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5,
                                         name='bn1')(x)
  x = tf.keras.layers.ReLU(name='relu1')(x)
  x = tf.keras.layers.ZeroPadding2D(padding=1, name='maxpool_pad')(x)
  x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, name='maxpool')(x)

  x = _make_layer(x, filters, blocks_per_layer[0], name='layer1')

  x = tf.keras.layers.GlobalAveragePooling2D(name='avgpool')(x)
  initializer = tf.keras.initializers.RandomUniform(-1.0 / (2 * filters)**0.5,
                                                    1.0 / (2 * filters)**0.5)
  x = tf.keras.layers.Dense(units=classes, kernel_initializer=initializer,
                            bias_initializer=initializer, name='fc')(x)

  return x


def _MiniResNet(classes, input_shape, weights):
  inputs = tf.keras.Input(shape=input_shape)
  outputs = _resnet(inputs, [1, 1, 1, 1], classes=classes, filters=4)
  return tf.keras.Model(inputs=inputs, outputs=outputs)


class EmpiricalTfTest(parameterized.TestCase):

  def _compare_ntks(
      self,
      f,
      f_jax,
      params,
      trace_axes,
      diagonal_axes,
      vmap_axes
  ):
    if any(i == j for i in trace_axes for j in diagonal_axes):
      raise absltest.SkipTest('Overlapping trace and diagonal axes.')

    kwargs = dict(
        trace_axes=trace_axes,
        diagonal_axes=diagonal_axes,
    )

    jax_ntk_fns = [
        jax.jit(nt.empirical_ntk_fn(
            **kwargs, f=f_jax, implementation=i, vmap_axes=v))
        for i in nt.NtkImplementation
        for v in vmap_axes if v not in trace_axes + diagonal_axes
    ]

    ntk_fns = [
        experimental.empirical_ntk_fn_tf(**kwargs,
                                         f=f,
                                         implementation=i,
                                         vmap_axes=v)
        for i in nt.NtkImplementation
        for v in vmap_axes if v not in trace_axes + diagonal_axes
    ]

    x_shape = (f.input_shape[1:] if isinstance(f, tf.Module) else
               f.input_signature[1].shape[1:])

    x1 = tf.random.normal((2,) + x_shape, seed=2) / onp.prod(x_shape)**0.5
    x2 = tf.random.normal((3,) + x_shape, seed=3) / onp.prod(x_shape)**0.5

    x1_jax = np.array(x1)
    x2_jax = np.array(x2)
    params_jax = jax.tree_map(lambda x: np.array(x), params)

    jax_ntks = [ntk_fn_i(x1_jax, x2_jax, params_jax)
                for ntk_fn_i in jax_ntk_fns]

    ntks = list(enumerate([ntk_fn_i(x1, x2, params)
                           for ntk_fn_i in ntk_fns]))

    if len(tf.config.list_physical_devices()) > 1:  # TPU
      atol = 0.
      rtol = 5e-3
      atol_jax = 0.4
      rtol_jax = 0.15  # TODO(romann): revisit poor TPU agreement.
    else:
      atol = 1e-5
      rtol = 1e-4
      atol_jax = 0.
      rtol_jax = 5e-5

    for i1, ntk1 in ntks:
      for i2, ntk2 in ntks[i1 + 1:]:
        # Compare different implementation
        onp.testing.assert_allclose(ntk1, ntk2, rtol=rtol, atol=atol)
        # Compare against the JAX version (without calling `jax2tf`).
        onp.testing.assert_allclose(ntk1, jax_ntks[i1], rtol=rtol_jax,
                                    atol=atol_jax)

  @parameterized.product(
      f=[
          _MiniResNet,
          # # TODO(romann): MobileNet works, but takes too long to compile.
          # tf.keras.applications.MobileNet,
      ],
      input_shape=[
          (64, 64, 3)
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
    f = f(classes=1, input_shape=input_shape, weights=None)
    f.build((None, *input_shape))
    f_jax, params = experimental.get_apply_fn_and_params(f)
    self._compare_ntks(f, f_jax, params, trace_axes, diagonal_axes, vmap_axes)

  @parameterized.product(
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
  def test_keras_sequential(
      self,
      input_shape,
      trace_axes,
      diagonal_axes,
      vmap_axes,
  ):
    f = tf.keras.Sequential()
    f.add(tf.keras.layers.Conv2D(4, (3, 3), activation='relu'))
    f.add(tf.keras.layers.Conv2D(2, (2, 2), activation='relu'))
    f.add(tf.keras.layers.Flatten())
    f.add(tf.keras.layers.Dense(2))

    f.build((None, *input_shape))
    f_jax, params = experimental.get_apply_fn_and_params(f)
    self._compare_ntks(f, f_jax, params, trace_axes, diagonal_axes, vmap_axes)

  @parameterized.product(
      f_f_jax=[
          (_f1, _f1_jax),
          (_f2, _f2_jax),
          (_f3, _f3_jax),
          (_f4, _f4_jax)
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
      f_f_jax,
      params_shape,
      trace_axes,
      diagonal_axes,
      vmap_axes,
  ):
    f, f_jax = f_f_jax
    f = tf.function(f, input_signature=_input_signature)
    params = tf.random.normal(params_shape, seed=4)
    self._compare_ntks(f, f_jax, params, trace_axes, diagonal_axes, vmap_axes)

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
  def test_tf_module(
      self,
      trace_axes,
      diagonal_axes,
      vmap_axes,
  ):
    f = _MLP(input_size=5, sizes=[4, 6, 3], name='MLP')
    f_jax, params = experimental.get_apply_fn_and_params(f)
    self._compare_ntks(f, f_jax, params, trace_axes, diagonal_axes, vmap_axes)


if __name__ == '__main__':
  absltest.main()
