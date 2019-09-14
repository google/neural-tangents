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

"""The `Kernel` class containing NTK and and NNGP `np.ndarray`s as fields."""

import collections


class Kernel(
    collections.namedtuple(
        'Kernel',
        ['var1', 'nngp', 'var2', 'ntk', 'is_gaussian', 'is_height_width'])):
  """A tuple containing information about the analytic NTK and NNGP of a model.

  Attributes:
    var1: variances of the first batch of inputs. A `np.ndarray` with shape
      `[batch_size_1]`  for fully-connected networks or `[batch_size_1, height,
      width]` for CNNs.
    nngp: covariance between the first and second input (NNGP). A `np.ndarray`
      of shape `[batch_size_1, batch_size_2]` for fully-connected networks, or
      `[batch_size_1, batch_size_2, height, height, width, width]` or
      `[batch_size_1, batch_size_2, height, width]` for CNNs if pooling is used
      or not respectively.
    var2: optional variances of the second batch of inputs. A `np.ndarray` with
      shape `[batch_size_2]` for fully-connected networks or `[batch_size_2,
      height, width]` for CNNs.
    ntk: the neural tangent kernel (NTK). `np.ndarray` of same shape as `nngp`.
    is_gaussian: a boolean, specifying whether the output features or channels
      of the layer / NN function (returning this `Kernel` as the `ker_fun`) are
      i.i.d. Gaussian with covariance `nngp`, conditioned on fixed inputs to the
      layer and i.i.d. Gaussian weights and biases of the layer. For example,
      passing an input through a CNN layer with i.i.d. Gaussian weights and
      biases produces i.i.d. Gaussian random variables along the channel
      dimension, while passing an input through a nonlinearity does not.
    is_height_width: a boolean specifying whether the covariance matrices
      `nngp`, `var1`, `var2`, and `ntk` have `height` dimensions preceding
      `width` or the other way around. Is always set to `True` if `nngp` and
      `ntk` are less than 6-dimensional and alternates between consecutive CNN
      layers otherwise to avoid self-cancelling transpositions.
  """

  def __new__(cls, var1, nngp, var2, ntk, is_gaussian, is_height_width):
    """Returns a `Kernel`.

    Args:
      var1: variances of the first batch of inputs. A `np.ndarray` with shape
        `[batch_size_1]`  for fully-connected networks or `[batch_size_1,
        height, width]` for CNNs.
      nngp: covariance between the first and second input (NNGP). A `np.ndarray`
        of shape `[batch_size_1, batch_size_2]` for fully-connected networks, or
        `[batch_size_1, batch_size_2, height, height, width, width]` or
        `[batch_size_1, batch_size_2, height, width]` for CNNs if pooling is
        used or not respectively.
      var2: optional variances of the second batch of inputs. A `np.ndarray`
        with shape `[batch_size_2]` for fully-connected networks or
        `[batch_size_2, height, width]` for CNNs.
      ntk: the neural tangent kernel (NTK). `np.ndarray` of same shape as
        `nngp`.
      is_gaussian: a boolean, specifying whether the output features or channels
        of the layer / NN function (returning this `Kernel` as the `ker_fun`)
        are i.i.d. Gaussian with covariance `nngp`, conditioned on fixed inputs
        to the layer and i.i.d. Gaussian weights and biases of the layer. For
        example, passing an input through a CNN layer with i.i.d. Gaussian
        weights and biases produces i.i.d. Gaussian random variables along the
        channel dimension, while passing an input through a nonlinearity does
        not.
      is_height_width: a boolean specifying whether the covariance matrices
        `nngp`, `var1`, `var2`, and `ntk` have `height` dimensions preceding
        `width` or the other way around. Is always set to `True` if `nngp` and
        `ntk` are less than 6-dimensional and alternates between consecutive CNN
        layers otherwise to avoid self-cancelling transpositions.

    Returns:
      A `Kernel`.
    """
    return super(Kernel, cls).__new__(cls, var1, nngp, var2, ntk, is_gaussian,
                                      is_height_width)

  def __add__(self, other):
    return Kernel(_add(self.var1, other.var1),
                  _add(self.nngp, other.nngp),
                  _add(self.var2, other.var2),
                  _add(self.ntk, other.ntk),
                  _add(self.is_gaussian, other.is_gaussian),
                  _add(self.is_height_width, other.is_height_width))

  def __truediv__(self, other):
    return Kernel(_div(self.var1, other),
                  _div(self.nngp, other),
                  _div(self.var2, other),
                  _div(self.ntk, other),
                  self.is_gaussian,
                  self.is_height_width)

  def __mul__(self, other):
    return Kernel(_mul(self.var1, other),
                  _mul(self.nngp, other),
                  _mul(self.var2, other),
                  _mul(self.ntk, other),
                  self.is_gaussian,
                  self.is_height_width)


def _add(x, y):
  error = ValueError('x and y must be of the same type, got %s and %s.'
                     % (str(x), str(y)))

  if x is None:
    if y is None:
      return None
    else:
      raise error
  if y is None:
    raise error

  if isinstance(x, bool):
    if isinstance(y, bool):
      if x == y:
        return x
      else:
        raise NotImplementedError('bool x and y must have matching values.')
    else:
      raise error

  if isinstance(y, bool):
    raise error

  return x + y


def _div(x, y):
  if x is None:
    return x

  return x / y


def _mul(x, y):
  if x is None:
    return x

  return x * y
