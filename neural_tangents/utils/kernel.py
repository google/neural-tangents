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
import jax.numpy as np
from neural_tangents.utils import utils


class Kernel(
    collections.namedtuple('Kernel', [
        'cov1', 'nngp', 'cov2', 'ntk', 'is_gaussian', 'is_reversed',
        'diagonal_batch', 'diagonal_spatial', 'shape1', 'shape2',
        'x1_is_x2', 'is_input', 'batch_axis', 'channel_axis'
    ])):
  """A tuple containing information about the analytic NTK and NNGP of a model.

  Attributes:
    cov1: covariance of the first batch of inputs. A `np.ndarray` with shape
      `(batch_size_1, [batch_size_1,] height, [height,], width, [width,], ...)`
      where the exact shape depends on `diagonal_batch` and `diagonal_spatial`.
    nngp: covariance between the first and second batches (NNGP). A `np.ndarray`
      of shape
      `(batch_size_1, batch_size_2, height, [height,], width, [width,], ...))`,
      where the exact shape depends on `diagonal_spatial`.
    cov2: optional covariance of the second batch of inputs. A `np.ndarray` with
      shape
      `(batch_size_2, [batch_size_2,] height, [height,], width, [width,], ...)`
      where the exact shape depends on `diagonal_batch` and `diagonal_spatial`.
    ntk: the neural tangent kernel (NTK). `np.ndarray` of same shape as `nngp`.
    is_gaussian: a boolean, specifying whether the output features or channels
      of the layer / NN function (returning this `Kernel` as the `kernel_fn`)
      are i.i.d. Gaussian with covariance `nngp`, conditioned on fixed inputs to
      the layer and i.i.d. Gaussian weights and biases of the layer. For
      example, passing an input through a CNN layer with i.i.d. Gaussian weights
      and biases produces i.i.d. Gaussian random variables along the channel
      dimension, while passing an input through a nonlinearity does not.
    is_reversed: a boolean specifying whether the covariance matrices `nngp`,
      `cov1`, `cov2`, and `ntk` have the ordering of spatial dimensions
      reversed. Ignored unless `diagonal_spatial` is `False`. Used internally
      to avoid self-cancelling transpositions in a sequence of CNN layers that
      flip the order of kernel spatial dimensions.
    diagonal_batch: a boolean specifying whether `cov1` and `cov2` store only
      the diagonal of the sample-sample covariance
      (`diagonal_batch == True`,
       `cov1.shape == (batch_size_1, ...)`),
      or the full covariance
      (`diagonal_batch == False`,
       `cov1.shape == (batch_size_1, batch_size_1, ...)`).
      Defaults to `True` as no current layers require the full covariance.
    diagonal_spatial: a boolean specifying whether all (`cov1`, `ntk`, etc.)
      covariance matrices store only the diagonals of the location-location
      covariances
      (`diagonal_spatial == True`,
       `nngp.shape == (batch_size_1, batch_size_2, height, width, depth, ...)`),
      or the full covariance
      (`diagonal_spatial == False`,
       `nngp.shape == (batch_size_1, batch_size_2, height, height,
                       width, width, depth, depth, ...)`).
      Defaults to `False`, but is set to `True` if the output top-layer
      covariance depends only on the diagonals (e.g. when a CNN network has no
      pooling layers and `Flatten` on top).
    shape1: a tuple specifying the shape of the random variable in the first
      batch of inputs. These have covariance `cov1` and covariance with the
      second batch of inputs given by `nngp`.
    shape2: a tuple specifying the shape of the random variable in the second
      batch of inputs. These have covariance `cov2` and covariance with the
      first batch of inputs given by `nngp`.
    x1_is_x2: a boolean specifying whether `x1` and `x2` are the same.
    is_input: a boolean specifying whether the current layer is the input
      layer and it is used to avoid applying dropout to the input layer.
    batch_axis: integer, the batch axis of the activations.
    channel_axis: integer, channel axis of the activations (taken to infinity).
  """

  def __new__(cls, cov1, nngp, cov2, ntk, is_gaussian, is_reversed,
              diagonal_batch, diagonal_spatial, shape1, shape2,
              x1_is_x2, is_input, batch_axis, channel_axis):
    """Returns a `Kernel`.

    Args:
      cov1: covariance of the first batch of inputs. A `np.ndarray` with shape
        `(batch_size_1, [batch_size_1,] height, [height,], width, [width,], ..)`
        where exact shape depends on `diagonal_batch` and `diagonal_spatial`.
      nngp: covariance between the first and second batches (NNGP). `np.ndarray`
        of shape
        `(batch_size_1, batch_size_2, height, [height,], width, [width,], ..))`,
        where the exact shape depends on `diagonal_spatial`.
      cov2: optional covariance of the second batch of inputs. `np.ndarray` with
        shape
        `(batch_size_2, [batch_size_2,] height, [height,], width, [width,], ..)`
        where exact shape depends on `diagonal_batch` and `diagonal_spatial`.
      ntk: neural tangent kernel (NTK). `np.ndarray` of same shape as `nngp`.
      is_gaussian: a boolean, specifying whether the output features or channels
        of the layer / NN function (returning this `Kernel` as the `kernel_fn`)
        are i.i.d. Gaussian with covariance `nngp`, conditioned on fixed inputs
        to the layer and i.i.d. Gaussian weights and biases of the layer. For
        example, passing an input through a CNN layer with i.i.d. Gaussian
        weights and biases produces i.i.d. Gaussian random variables along the
        channel dimension, while passing input through a nonlinearity does not.
      is_reversed: a boolean specifying whether the covariance matrices `nngp`,
        `cov1`, `cov2`, and `ntk` have the ordering of spatial dimensions
        reversed. Ignored unless `diagonal_spatial` is `False`. Used
        internally to avoid self-cancelling transpositions in a sequence of CNN
        layers that flip the order of kernel spatial dimensions.
      diagonal_batch: a boolean specifying whether `cov1` and `cov2` store
        only the diagonal of the sample-sample covariance
        (`diagonal_batch == True`,
         `cov1.shape == (batch_size_1, ...)`),
        or the full covariance
        (`diagonal_batch == False`,
         `cov1.shape == (batch_size_1, batch_size_1, ...)`).
        Defaults to `True` as no current layers require the full covariance.
      diagonal_spatial: a boolean specifying if all (`cov1`, `ntk`, etc.)
        covariance matrices store only the diagonals of the location-location
        covariances
        (`diagonal_spatial == True`,
         `nngp.shape == (batch_size_1, batch_size_2, height, width, depth, ..)`)
        or the full covariance
        (`diagonal_spatial == False`,
         `nngp.shape == (batch_size_1, batch_size_2, height, height,
                         width, width, depth, depth, ...)`).
        Defaults to `False`, but is set to `True` if the output top-layer
        covariance depends only on the diagonals (e.g. when a CNN network has no
        pooling layers and `Flatten` on top).
      shape1: a tuple specifying the shape of the random variable in the first
        batch of inputs. These have covariance `cov1` and covariance with the
        second batch of inputs given by `nngp`.
      shape2: a tuple specifying the shape of the random variable in the second
        batch of inputs. These have covariance `cov2` and covariance with the
        first batch of inputs given by `nngp`.
      x1_is_x2: a boolean specifying whether `x1` and `x2` are the same.
      is_input: a boolean specifying whether the current layer is the input
        layer and it is used to avoid applying dropout to the input layer.
      batch_axis: integer, the batch axis of the activations.
      channel_axis: integer, channel axis of the activations
        (taken to infinity).
    Returns:
      A `Kernel`.
    """
    return super(Kernel, cls).__new__(
        cls, cov1, nngp, cov2, ntk, is_gaussian,
        is_reversed, diagonal_batch, diagonal_spatial, shape1,
        shape2, x1_is_x2, is_input, batch_axis, channel_axis)

  def reverse(self):
    """Reverse the order of spatial axes in the covariance matrices.

    Args:
      self: a `Kernel` object.

    Returns:
      A `Kernel` object with spatial axes order flipped in
      all covariance matrices. For example, if `kernels.nngp` has shape
      `(batch_size_1, batch_size_2, H, H, W, W, D, D, ...)`, then
      `reverse(kernels).nngp` has shape
      `(batch_size_1, batch_size_2, ..., D, D, W, W, H, H)`.
    """
    # Number of spatial dimensions = total - (1 for batch + 1 for channels)
    ndim = len(self.shape1) - 2

    # ndim == 3: (-5, -6, -3, -4, -1, -2)
    source_axes = tuple(j for i in range(-ndim * 2, 0, 2) for j in (i + 1, i))

    # ndim == 3: (-1, -2, -3, -4, -5, -6)
    target_axes = tuple(range(-1, -ndim * 2 - 1, -1))

    def reverse(mat):
      if utils.is_array(mat):
        return np.moveaxis(mat, source_axes, target_axes)
      return mat

    cov1, nngp, cov2, ntk = map(reverse,
                                (self.cov1, self.nngp, self.cov2, self.ntk))
    return self._replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk,
                         is_reversed=not self.is_reversed)

  def permute_spatial(self, permutation):
    """Permute spatial dimensions of the `Kernel` according to `permutation`."""
    def permute(mat, batch_ndim):
      if utils.is_array(mat):
        _permutation = tuple(batch_ndim + p for p in permutation)
        if not self.diagonal_spatial:
          _permutation = tuple(j for p in _permutation
                               for j in (2 * p - batch_ndim,
                                         2 * p - batch_ndim + 1))
        _permutation = tuple(range(batch_ndim)) + _permutation
        return np.transpose(mat, _permutation)
      return mat

    cov1 = permute(self.cov1, 1 if self.diagonal_batch else 2)
    cov2 = permute(self.cov2, 1 if self.diagonal_batch else 2)
    nngp = permute(self.nngp, 2)
    ntk = permute(self.ntk, 2)
    return self._replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)
