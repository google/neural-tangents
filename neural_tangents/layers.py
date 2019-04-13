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

"""Includes Dense and Conv layers in the NTK parameterization for stax.

See jax.experimental.stax for general information about the implementation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import operator as op

from jax import lax
from jax.experimental import stax

import jax.numpy as np


def Dense(
    out_dim,
    W_gain=1.0, W_init=stax.randn(1.0),
    b_gain=0.0, b_init=stax.randn(1.0)):
  """Layer constructor function for a dense (fully-connected) layer.

  Uses jax.experimental.stax.Dense as a base.
  """
  init_fun, _ = stax.Dense(out_dim, W_init, b_init)
  def apply_fun(params, inputs, **kwargs):
    W, b = params
    norm = W_gain / np.sqrt(inputs.shape[-1])
    return norm * np.dot(inputs, W) + b_gain * b
  return init_fun, apply_fun


def GeneralConv(dimension_numbers, out_chan, filter_shape,
                strides=None, padding='VALID',
                W_gain=1.0, W_init=stax.randn(1.0),
                b_gain=0.0, b_init=stax.randn(1.0)):
  """Layer construction function for a general convolution layer.

  Uses jax.experimental.stax.GeneralConv as a base.
  """
  lhs_spec, rhs_spec, out_spec = dimension_numbers
  one = (1,) * len(filter_shape)
  strides = strides or one
  init_fun, _ = stax.GeneralConv(dimension_numbers, out_chan, filter_shape,
                                 strides, padding, W_init, b_init)
  def apply_fun(params, inputs, **kwargs):
    W, b = params
    norm = inputs.shape[lhs_spec.index('C')]
    norm *= functools.reduce(op.mul, filter_shape)
    norm = W_gain / np.sqrt(norm)
    return norm * lax.conv_general_dilated(
        inputs, W, strides, padding, one, one, dimension_numbers) + b_gain * b
  return init_fun, apply_fun
Conv = functools.partial(GeneralConv, ('NHWC', 'HWIO', 'NHWC'))
