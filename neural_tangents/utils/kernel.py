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

"""The `Kernel` class containing NTK and NNGP `np.ndarray`s as fields."""

import collections
import aenum


class Marginalisation(aenum.OrderedEnum):
  """Types of marginal distributions for which covariances can be computed.

  Let k_{ij}(x, y) represent a covariance between the spatial dimensions i and j
  for the inputs x and y (for multiple spatial dimensions, imagine i and j have
  multiple entries). Then and instance of `Kernel` with `marginal`/`cross`:

  `OVER_ALL`: (used for architectures with no spatial dimensions)
    `var1`/`var2`: k(x, x), shape is (batch_size,)
    `nnpg`/`ntk`: k(x, y), shape is (batch_size_1, batch_size_2)
  `OVER_PIXELS`:
    `var1`/`var2`: k_{ii}(x, x), shape is
     (batch_size, spatial_dim_1, spatial_dim_2)
    `nngp`/`ntk`: k_{ii}(x, y), shape is
     (batch_size_1, batch_size_2, spatial_dim_1, spatial_dim_2)
  `OVER_POINTS`:
    `var1`/`var2`: k_{ij}(x, x), shape is
     (batch_size, spatial_dim_1, spatial_dim_1, spatial_dim_2, spatial_dim_2)
    `nngp`/`ntk`: not allowed; please use `NO` instead
  `NO`:
    `var1`/`var2`: k_{ij}(x, y), shape is
     (batch_size, batch_size,
      spatial_dim_1, spatial_dim_1, spatial_dim_2, spatial_dim_2)
    `nngp`/`ntk`: k_{ij}(x, y)
     (batch_size_1, batch_size_2,
      spatial_dim_1, spatial_dim_1, spatial_dim_2, spatial_dim_2)

  The number associated with each instance of this enum represents the relative
  amount of information being tracked compared to the other options, i.e.,
  the information tracked by `OVER_PIXELS` is a strict subset of that tracked
  by `OVER_POINTS`, which itself tracks a strict subset of information
  compared to the `NO` option. Note that this is an `OrderedEnum`
  meaning that comparison operators `<`, `==` work as set inclusion and equality
  (and `>`, `<=`, `>=` as would be expected given this definition).
  """
  NONE = None
  OVER_ALL = 0
  OVER_PIXELS = 1
  OVER_POINTS = 2
  NO = 3


class Kernel(
    collections.namedtuple(
        'Kernel',
        ['var1', 'nngp', 'var2', 'ntk', 'is_gaussian', 'is_height_width',
         'marginal', 'cross'])):
  """A tuple containing information about the analytic NTK and NNGP of a model.

  Attributes:
    var1: variances of the first batch of inputs. A `np.ndarray` with shape
      `[batch_size_1]` for fully-connected networks, or matching the one
      specified by the `marginal` argument for CNNs with `batch_size_1` for
      data dimension(s).
    nngp: covariance between the first and second input (NNGP). A `np.ndarray`
      of shape `[batch_size_1, batch_size_2]` for fully-connected networks, or
      matching the one specifed by the `cross` argument for CNNs with
      `[batch_size_1, batch_size_2]` for data dimensions.
    var2: optional variances of the second batch of inputs. A `np.ndarray`
      with shape `[batch_size_2]` for fully-connected networks or matching
      the one  specified by the `marginal` argument for CNNs with
      `batch_size_2` for data dimension(s).
    ntk: the neural tangent kernel (NTK). `np.ndarray` of same shape as
      `nngp`.
    is_gaussian: a boolean, specifying whether the output features or channels
      of the layer / NN function (returning this `Kernel` as the `kernel_fn`)
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
    marginal: an instance of `Marginalisation` enum or its ID, specifying
      types of covariances between spatial dimensions tracked in
      `var1`/`var2`.
    cross: an instance of `Marginalisation` enum or its ID, specifying types
      of covariances between spatial dimensions tracked in `nngp`/`ntk`.
  """

  @staticmethod
  def _cov_match_check(first, second):
    if isinstance(second, Kernel):
      marginal1 = Marginalisation(first.marginal)
      marginal2 = Marginalisation(second.marginal)
      cross1 = Marginalisation(first.cross)
      cross2 = Marginalisation(second.cross)

      if not (marginal1 == marginal2 and cross1 == cross2):
        raise ValueError('The types covariances stored do not match. '
                         'nngp/ntk covariance type stored: {} vs. {} '
                         'var1/var2 covariance type stored: {} vs {} '.format(
            marginal1, marginal2, cross1, cross2))


  def __new__(cls, var1, nngp, var2, ntk, is_gaussian, is_height_width,
              marginal, cross):
    """Returns a `Kernel`.

    Args:
      var1: variances of the first batch of inputs. A `np.ndarray` with shape
        `[batch_size_1]` for fully-connected networks, or matching the one
        specified by the `marginal` argument for CNNs with `batch_size_1` for
        data dimension(s).
      nngp: covariance between the first and second input (NNGP). A `np.ndarray`
        of shape `[batch_size_1, batch_size_2]` for fully-connected networks, or
        matching the one specifed by the `cross` argument for CNNs with
        `[batch_size_1, batch_size_2]` for data dimensions.
      var2: optional variances of the second batch of inputs. A `np.ndarray`
        with shape `[batch_size_2]` for fully-connected networks or matching
        the one  specified by the `marginal` argument for CNNs with
        `batch_size_2` for data dimension(s).
      ntk: the neural tangent kernel (NTK). `np.ndarray` of same shape as
        `nngp`.
      is_gaussian: a boolean, specifying whether the output features or channels
        of the layer / NN function (returning this `Kernel` as the `kernel_fn`)
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
      marginal: an instance of `Marginalisation` enum or its ID, specifying
        types of covariances between spatial dimensions tracked in
        `var1`/`var2`.
      cross: an instance of `Marginalisation` enum or its ID, specifying types
        of covariances between spatial dimensions tracked in `nngp`/`ntk`.

    Returns:
      A `Kernel`.
    """
    def _convert(marg):
      if marg is None or isinstance(marg, (int, Marginalisation)):
        return Marginalisation(marg).value
      else:
        return marg
      #TODO(jirihron): parallel + tree_map pass `object` for `marginal`, `cross`
      # and other arguments which does not go well with Marginalisation(marg)

    return super(Kernel, cls).__new__(
        cls, var1, nngp, var2, ntk, is_gaussian, is_height_width,
        _convert(marginal), _convert(cross))

  def __add__(self, other):
    Kernel._cov_match_check(self, other)
    return Kernel(_add(self.var1, other.var1),
                  _add(self.nngp, other.nngp),
                  _add(self.var2, other.var2),
                  _add(self.ntk, other.ntk),
                  _add(self.is_gaussian, other.is_gaussian),
                  _add(self.is_height_width, other.is_height_width),
                  marginal=self.marginal,
                  cross=self.cross)

  def __truediv__(self, other):
    Kernel._cov_match_check(self, other)
    return Kernel(_div(self.var1, other),
                  _div(self.nngp, other),
                  _div(self.var2, other),
                  _div(self.ntk, other),
                  self.is_gaussian,
                  self.is_height_width,
                  marginal=self.marginal,
                  cross=self.cross)

  def __mul__(self, other):
    Kernel._cov_match_check(self, other)
    return Kernel(_mul(self.var1, other),
                  _mul(self.nngp, other),
                  _mul(self.var2, other),
                  _mul(self.ntk, other),
                  self.is_gaussian,
                  self.is_height_width,
                  marginal=self.marginal,
                  cross=self.cross)

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
