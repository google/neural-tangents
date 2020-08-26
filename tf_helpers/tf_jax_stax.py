# Copyright 2018 Google LLC
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

"""Stax is a small but flexible neural net specification library from scratch.

For an example of its use, see examples/resnet50.py.
"""

import functools
import itertools
import operator as op

import sys
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.ops import numpy_ops as tfnp
from tf_helpers import lax
from tf_shape_conversion import shape_conversion
import numpy as onp
from stateless_random_ops import split
from stateless_random_ops import stateless_random_normal as rn
from tensorflow.random import stateless_uniform

from tensorflow.nn import (relu, log_softmax, softmax, softplus, sigmoid, elu,
                    leaky_relu, selu)
from tensorflow import zeros_initializer as zi
from tensorflow import ones_initializer as oi

# Following the convention used in Keras and tf.layers, we use CamelCase for the
# names of layer constructors, like Conv and Relu, while using snake_case for
# other functions, like tfnp.conv and relu.

# Each layer constructor function returns an (init_fun, apply_fun) pair, where
#   init_fun: takes an rng key and an input shape and returns an
#     (output_shape, params) pair,
#   apply_fun: takes params, inputs, and an rng key and applies the layer.


def Dense(out_dim, W_init=rn, b_init=rn):
  """Layer constructor function for a dense (fully-connected) layer."""
  def init_fun(rng, input_shape):
    output_shape = input_shape[:-1] + (out_dim,)
    keys = split(seed=tf.convert_to_tensor(rng, dtype=tf.int32), num=2)
    k1 = keys[0]
    k2 = keys[1]
    # convert the two keys from shape (2,) into a scalar
    k1 = stateless_uniform(shape=[], seed=k1, minval=None, maxval=None, dtype=tf.int32)
    k2 = stateless_uniform(shape=[], seed=k2, minval=None, maxval=None, dtype=tf.int32)
    W = W_init(seed=k1, shape=(input_shape[-1], out_dim))
    b = b_init(seed=k2, shape=(out_dim,))
    return tfnp.zeros(output_shape), (W.numpy(), b.numpy())
  def apply_fun(params, inputs, **kwargs):
    W, b = params
    return tfnp.dot(inputs, W) + b
  return init_fun, apply_fun


def GeneralConv(dimension_numbers, out_chan, filter_shape,
                strides=None, padding='VALID', W_init=rn,
                b_init=rn):
  """Layer construction function for a general convolution layer."""
  lhs_spec, rhs_spec, out_spec = dimension_numbers
  one = (1,) * len(filter_shape)
  strides = strides or one
  def init_fun(rng, input_shape):
    input_shape = shape_conversion(input_shape)
    filter_shape_iter = iter(filter_shape)
    kernel_shape = [out_chan if c == 'O' else
                    input_shape[lhs_spec.index('C')] if c == 'I' else
                    next(filter_shape_iter) for c in rhs_spec]
    output_shape = lax.conv_general_shape_tuple(
        input_shape, kernel_shape, strides, padding, dimension_numbers)
    bias_shape = [out_chan if c == 'C' else 1 for c in out_spec]
    bias_shape = tuple(itertools.dropwhile(lambda x: x == 1, bias_shape))
    keys = split(seed=tf.convert_to_tensor(rng, dtype=tf.int32), num=2)
    k1 = keys[0]
    k2 = keys[1]
    W = W_init(seed=k1, shape=kernel_shape)
    b = b_init(stddev=1e-6, seed=k2, shape=bias_shape)
    return tfnp.zeros(output_shape), (W, b)
  def apply_fun(params, inputs, **kwargs):
    W, b = params
    return lax.conv_general_dilated(inputs, W, strides, padding, one, one,
                                    dimension_numbers=dimension_numbers) + b
  return init_fun, apply_fun
Conv = functools.partial(GeneralConv, ('NHWC', 'HWIO', 'NHWC'))


def elementwise(fun, **fun_kwargs):
  """Layer that applies a scalar function elementwise on its inputs."""
  def init_fun(rng, input_shape):
    return (tfnp.zeros(input_shape), ())
  # init_fun = lambda rng, input_shape: (tfnp.zeros(input_shape), ())
  apply_fun = lambda params, inputs, **kwargs: fun(inputs, **fun_kwargs)
  return init_fun, apply_fun
Tanh = elementwise(tfnp.tanh)
Relu = elementwise(relu)
Exp = elementwise(tfnp.exp)
LogSoftmax = elementwise(log_softmax, axis=-1)
Softmax = elementwise(softmax, axis=-1)
Softplus = elementwise(softplus)
Sigmoid = elementwise(sigmoid)
Elu = elementwise(elu)
LeakyRelu = elementwise(leaky_relu)
Selu = elementwise(selu)


def _pooling_layer(reducer, init_val, rescaler=None):
  def PoolingLayer(window_shape, strides=None, padding='VALID', spec=None):
    """Layer construction function for a pooling layer."""
    strides = strides or (1,) * len(window_shape)
    rescale = rescaler(window_shape, strides, padding) if rescaler else None

    dim = len(window_shape)
    batch_dim, channel_dim = None, None
    if spec is None:
      batch_dim, channel_dim = 0, len(window_shape) + 1
    else:
      batch_dim, channel_dim = spec.index('N'), spec.index('C')
    window_shape = window_shape
    strides = strides

    def init_fun(rng, input_shape):
      # Move the batch and channel dimension of the input shape such
      # that it is of data format "NHWC"
      shape = [input_shape[batch_dim]]
      for i in range(len(input_shape)):
        if i not in [batch_dim, channel_dim]:
          shape.append(input_shape[i])
      shape.append(input_shape[channel_dim])
      out_shape = lax.reduce_window_shape_tuple(shape, window_shape,
                                                strides, padding)
      return tfnp.zeros(out_shape), ()
    def apply_fun(params, inputs, **kwargs):
      inputs = onp.moveaxis(inputs, (batch_dim, channel_dim), \
                          (0, dim + 1))
      output = lax.reduce_window(inputs, init_val, reducer, window_shape,
                              strides, padding)
      return rescale(out, inputs, spec) if rescale else out
      # return output
      return tfnp.array(output)
    return init_fun, apply_fun
  return PoolingLayer
MaxPool = _pooling_layer(tfnp.max, -tfnp.inf)


def _normalize_by_window_size(dims, strides, padding):
  def rescale(outputs, inputs, spec):
    if spec is None:
      non_spatial_axes = 0, inputs.ndim - 1
    else:
      non_spatial_axes = spec.index('N'), spec.index('C')

    spatial_shape = tuple(inputs.shape[i]
                          for i in range(inputs.ndim)
                          if i not in non_spatial_axes)
    one = tfnp.ones(spatial_shape, dtype=inputs.dtype)
    window_sizes = lax.reduce_window(one, 0., tfnp.add, dims, strides, padding)
    for i in sorted(non_spatial_axes):
      window_sizes = tfnp.expand_dims(window_sizes, i)

    return outputs * window_sizes
  return rescale
SumPool = _pooling_layer(tfnp.add, 0., _normalize_by_window_size)
AvgPool = _pooling_layer(tfnp.add, 0.)


def Flatten():
  """Layer construction function for flattening all but the leading dim."""
  def init_fun(rng, input_shape):
    output_shape = input_shape[0], functools.reduce(op.mul, input_shape[1:], 1)
    return tfnp.zeros(output_shape), ()
  def apply_fun(params, inputs, **kwargs):
    return tfnp.reshape(inputs, (inputs.shape[0], -1))
  return init_fun, apply_fun
Flatten = Flatten()


def Identity():
  """Layer construction function for an identity layer."""
  init_fun = lambda rng, input_shape: (tfnp.zeros(input_shape), ())
  apply_fun = lambda params, inputs, **kwargs: inputs
  return init_fun, apply_fun
Identity = Identity()


def FanOut(num):
  """Layer construction function for a fan-out layer."""
  def init_fun(rng, input_shape):
    return ([tfnp.zeros(input_shape)] * num, ())
  apply_fun = lambda params, inputs, **kwargs: [inputs] * num
  return init_fun, apply_fun


def FanInSum():
  """Layer construction function for a fan-in sum layer."""
  init_fun = lambda rng, input_shape: (tfnp.zeros(input_shape[0]), ())
  apply_fun = lambda params, inputs, **kwargs: sum(inputs)
  return init_fun, apply_fun
FanInSum = FanInSum()


def FanInConcat(axis=-1):
  """Layer construction function for a fan-in concatenation layer."""
  def init_fun(rng, input_shape):
    ax = axis % len(input_shape[0])
    concat_size = sum(shape[ax] for shape in input_shape)
    out_shape = input_shape[0][:ax] + (concat_size,) + input_shape[0][ax+1:]
    return tfnp.zeros(out_shape), ()
  def apply_fun(params, inputs, **kwargs):
    return tfnp.concatenate(inputs, axis)
  return init_fun, apply_fun


def Dropout(rate, mode='train'):
  """Layer construction function for a dropout layer with given rate."""
  def init_fun(rng, input_shape):
    return tfnp.zeros(input_shape), ()
  def apply_fun(params, inputs, **kwargs):
    rng = kwargs.get('rng', None)
    if rng is None:
      msg = ("Dropout layer requires apply_fun to be called with a PRNG key "
             "argument. That is, instead of `apply_fun(params, inputs)`, call "
             "it like `apply_fun(params, inputs, rng)` where `rng` is a "
             "jax.random.PRNGKey value.")
      raise ValueError(msg)
    if mode == 'train':
      prob = tf.ones(inputs.shape) * rate
      keep = stateless_uniform(shape=inputs.shape, seed=rng, minval=0, maxval=1) < prob
      return tfnp.where(keep, inputs / rate, 0)
    else:
      return inputs
  return init_fun, apply_fun


# Composing layers via combinators
def serial(*layers):
  """Combinator for composing layers in serial.

  Args:
    *layers: a sequence of layers, each an (init_fun, apply_fun) pair.

  Returns:
    A new layer, meaning an (init_fun, apply_fun) pair, representing the serial
    composition of the given sequence of layers.
  """
  nlayers = len(layers)
  init_funs, apply_funs = zip(*layers)
  def init_fun(rng, input_shape):
    params = []
    i = 0
    for init_fun in init_funs:
      i += 1
      keys = split(seed=tf.convert_to_tensor(rng, dtype=tf.int32), num=2)
      rng = keys[0]
      layer_rng = keys[1]
      input_shape = shape_conversion(input_shape)
      input_shape, param = init_fun(layer_rng, input_shape)
      params.append(param)
    return input_shape, params
  def apply_fun(params, inputs, **kwargs):
    rng = kwargs.pop('rng', None)
    rngs = None
    if rng is not None:
      rngs = split(seed=tf.convert_to_tensor(rng, dtype=tf.int32), num=nlayers)
    else:
      rngs = (None,) * nlayers
    for i in range(nlayers):
      inputs = apply_funs[i](params[i], inputs, rng=rngs[i], **kwargs)
    return inputs
  return init_fun, apply_fun


def parallel(*layers):
  """Combinator for composing layers in parallel.

  The layer resulting from this combinator is often used with the FanOut and
  FanInSum layers.

  Args:
    *layers: a sequence of layers, each an (init_fun, apply_fun) pair.

  Returns:
    A new layer, meaning an (init_fun, apply_fun) pair, representing the
    parallel composition of the given sequence of layers. In particular, the
    returned layer takes a sequence of inputs and returns a sequence of outputs
    with the same length as the argument `layers`.
  """
  nlayers = len(layers)
  init_funs, apply_funs = zip(*layers)
  def init_fun(rng, input_shape):
    rngs = split(seed=tf.convert_to_tensor(rng, dtype=tf.int32), num=nlayers)
    result = []
    for i in range(nlayers):
      result.append(init_funs[i](rngs[i], input_shape[i]))
    return zip(*result)
  def apply_fun(params, inputs, **kwargs):
    rng = kwargs.pop('rng', None)
    rngs = None
    if rng is not None:
      rngs = split(seed=tf.convert_to_tensor(rng, dtype=tf.int32), num=nlayers)
    else:
      rngs = (None,) * nlayers
    result = []
    for i in range(len(apply_funs)):
      result.append(apply_funs[i](params[i], inputs[i], rng=rngs[i], **kwargs))
    return result
  return init_fun, apply_fun


def shape_dependent(make_layer):
  """Combinator to delay layer constructor pair until input shapes are known.

  Args:
    make_layer: a one-argument function that takes an input shape as an argument
      (a tuple of positive integers) and returns an (init_fun, apply_fun) pair.

  Returns:
    A new layer, meaning an (init_fun, apply_fun) pair, representing the same
    layer as returned by `make_layer` but with its construction delayed until
    input shapes are known.
  """
  def init_fun(rng, input_shape):
    return make_layer(input_shape)[0](rng, input_shape)
  def apply_fun(params, inputs, **kwargs):
    return make_layer(inputs.shape)[1](params, inputs, **kwargs)
  return init_fun, apply_fun
