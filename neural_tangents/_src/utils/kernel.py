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

"""Class with infinite-width NTK and NNGP :class:`jax.numpy.ndarray` fields."""

import operator as op
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import dataclasses
from . import utils
from jax import lax
import jax.numpy as np


@dataclasses.dataclass
class Kernel:
  """Dataclass containing information about the NTK and NNGP of a model.

  Attributes:
    nngp:
      covariance between the first and second batches (NNGP). A `np.ndarray` of
      shape
      `(batch_size_1, batch_size_2, height, [height,], width, [width,], ...))`,
      where exact shape depends on `diagonal_spatial`.

    ntk:
      the neural tangent kernel (NTK). `np.ndarray` of same shape as `nngp`.

    cov1:
      covariance of the first batch of inputs. A `np.ndarray` with shape
      `(batch_size_1, [batch_size_1,] height, [height,], width, [width,], ...)`
      where exact shape depends on `diagonal_batch` and `diagonal_spatial`.

    cov2:
      optional covariance of the second batch of inputs. A `np.ndarray` with
      shape
      `(batch_size_2, [batch_size_2,] height, [height,], width, [width,], ...)`
      where the exact shape depends on `diagonal_batch` and `diagonal_spatial`.

    x1_is_x2:
      a boolean specifying whether `x1` and `x2` are the same.

    is_gaussian:
      a boolean, specifying whether the output features or channels of the layer
      / NN function (returning this `Kernel` as the `kernel_fn`) are i.i.d.
      Gaussian with covariance `nngp`, conditioned on fixed inputs to the layer
      and i.i.d. Gaussian weights and biases of the layer. For example, passing
      an input through a CNN layer with i.i.d. Gaussian weights and biases
      produces i.i.d. Gaussian random variables along the channel dimension,
      while passing an input through a nonlinearity does not.

    is_reversed:
      a boolean specifying whether the covariance matrices `nngp`, `cov1`,
      `cov2`, and `ntk` have the ordering of spatial dimensions reversed.
      Ignored unless `diagonal_spatial` is `False`. Used internally to avoid
      self-cancelling transpositions in a sequence of CNN layers that flip the
      order of kernel spatial dimensions.

    is_input:
      a boolean specifying whether the current layer is the input layer and it
      is used to avoid applying dropout to the input layer.

    diagonal_batch:
      a boolean specifying whether `cov1` and `cov2` store only the diagonal of
      the sample-sample covariance (`diagonal_batch == True`,
      `cov1.shape == (batch_size_1, ...)`), or the full covariance
      (`diagonal_batch == False`,
      `cov1.shape == (batch_size_1, batch_size_1, ...)`). Defaults to `True` as
      no current layers require the full covariance.

    diagonal_spatial:
      a boolean specifying whether all (`cov1`, `ntk`, etc.) covariance matrices
      store only the diagonals of the location-location covariances
      (`diagonal_spatial == True`,
      `nngp.shape == (batch_size_1, batch_size_2, height, width, depth, ...)`),
      or the full covariance (`diagonal_spatial == False`, `nngp.shape ==
      (batch_size_1, batch_size_2, height, height, width, width, depth, depth,
      ...)`).
      Defaults to `False`, but is set to `True` if the
      output top-layer covariance depends only on the diagonals (e.g. when a CNN
      network has no pooling layers and `Flatten` on top).

    shape1:
      a tuple specifying the shape of the random variable in the first batch of
      inputs. These have covariance `cov1` and covariance with the second batch
      of inputs given by `nngp`.

    shape2:
      a tuple specifying the shape of the random variable in the second batch of
      inputs. These have covariance `cov2` and covariance with the first batch
      of inputs given by `nngp`.

    batch_axis:
      the batch axis of the activations.

    channel_axis:
      channel axis of the activations (taken to infinity).

    mask1:
      an optional boolean `np.ndarray` with a shape broadcastable to `shape1`
      (and the same number of dimensions). `True` stands for the input being
      masked at that position, while `False` means the input is visible. For
      example, if `shape1 == (5, 32, 32, 3)` (a batch of 5 `NHWC` CIFAR10
      images), a `mask1` of shape `(5, 1, 32, 1)` means different images can
      have different blocked columns (`H` and `C` dimensions are always either
      both blocked or unblocked). `None` means no masking.

    mask2:
      same as `mask1`, but for the second batch of inputs.
  """

  nngp: np.ndarray
  ntk: Optional[np.ndarray]

  cov1: np.ndarray
  cov2: Optional[np.ndarray]
  x1_is_x2: np.ndarray

  is_gaussian: bool = dataclasses.field(pytree_node=False)
  is_reversed: bool = dataclasses.field(pytree_node=False)
  is_input: bool = dataclasses.field(pytree_node=False)

  diagonal_batch: bool = dataclasses.field(pytree_node=False)
  diagonal_spatial: bool = dataclasses.field(pytree_node=False)

  shape1: Tuple[int, ...] = dataclasses.field(pytree_node=False)
  shape2: Tuple[int, ...] = dataclasses.field(pytree_node=False)

  batch_axis: int = dataclasses.field(pytree_node=False)
  channel_axis: int = dataclasses.field(pytree_node=False)

  mask1: Optional[np.ndarray] = None
  mask2: Optional[np.ndarray] = None

  replace = ...  # type: Callable[..., 'Kernel']
  asdict = ...  # type: Callable[[], Dict[str, Any]]
  astuple = ...  # type: Callable[[], Tuple[Any, ...]]

  def slice(self, n1_slice: slice, n2_slice: slice) -> 'Kernel':
    cov1 = self.cov1[n1_slice]
    cov2 = self.cov1[n2_slice] if self.cov2 is None else self.cov2[n2_slice]
    ntk = self.ntk

    mask1 = None if self.mask1 is None else self.mask1[n1_slice]
    mask2 = None if self.mask2 is None else self.mask2[n2_slice]

    return self.replace(
        cov1=cov1,
        nngp=self.nngp[n1_slice, n2_slice],
        cov2=cov2,
        ntk=ntk if ntk is None or ntk.ndim == 0 else ntk[n1_slice, n2_slice],
        shape1=(cov1.shape[0],) + self.shape1[1:],
        shape2=(cov2.shape[0],) + self.shape2[1:],
        mask1=mask1,
        mask2=mask2)

  def reverse(self) -> 'Kernel':
    """Reverse the order of spatial axes in the covariance matrices.

    Returns:
      A `Kernel` object with spatial axes order flipped in
      all covariance matrices. For example, if `kernel.nngp` has shape
      `(batch_size_1, batch_size_2, H, H, W, W, D, D, ...)`, then
      `reverse(kernels).nngp` has shape
      `(batch_size_1, batch_size_2, ..., D, D, W, W, H, H)`.
    """
    batch_ndim = 1 if self.diagonal_batch else 2
    cov1 = utils.reverse_zipped(self.cov1, batch_ndim)
    cov2 = utils.reverse_zipped(self.cov2, batch_ndim)
    nngp = utils.reverse_zipped(self.nngp, 2)
    ntk = utils.reverse_zipped(self.ntk, 2)

    return self.replace(cov1=cov1,
                        nngp=nngp,
                        cov2=cov2,
                        ntk=ntk,
                        is_reversed=not self.is_reversed)

  def transpose(self, axes: Optional[Sequence[int]] = None) -> 'Kernel':
    """Permute spatial dimensions of the `Kernel` according to `axes`.

    Follows
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.transpose.html

    Note that `axes` apply only to spatial axes, batch axes are ignored and
    remain leading in all covariance arrays, and channel axes are not present
    in a `Kernel` object. If the covariance array is of shape
    `(batch_size, X, X, Y, Y)`, and `axes == (0, 1)`, resulting array is of
    shape `(batch_size, Y, Y, X, X)`.
    """
    if axes is None:
      axes = tuple(range(len(self.shape1) - 2))

    def permute(mat: Optional[np.ndarray],
                batch_ndim: int) -> Optional[np.ndarray]:
      if mat is not None:
        _axes = tuple(batch_ndim + a for a in axes)
        if not self.diagonal_spatial:
          _axes = tuple(j for a in _axes
                        for j in (2 * a - batch_ndim,
                                  2 * a - batch_ndim + 1))
        _axes = tuple(range(batch_ndim)) + _axes
        return np.transpose(mat, _axes)
      return mat

    cov1 = permute(self.cov1, 1 if self.diagonal_batch else 2)
    cov2 = permute(self.cov2, 1 if self.diagonal_batch else 2)
    nngp = permute(self.nngp, 2)
    ntk = permute(self.ntk, 2)
    return self.replace(cov1=cov1, nngp=nngp, cov2=cov2, ntk=ntk)

  def mask(self,
           mask1: Optional[np.ndarray],
           mask2: Optional[np.ndarray]) -> 'Kernel':
    """Mask all covariance matrices according to `mask1`, `mask2`."""
    mask11, mask12, mask22 = self._get_mask_prods(mask1, mask2)

    cov1 = utils.mask(self.cov1, mask11)
    cov2 = utils.mask(self.cov2, mask22)
    nngp = utils.mask(self.nngp, mask12)
    ntk = utils.mask(self.ntk, mask12)

    return self.replace(cov1=cov1,
                        nngp=nngp,
                        cov2=cov2,
                        ntk=ntk,
                        mask1=mask1,
                        mask2=mask2)

  def _get_mask_prods(
      self,
      mask1: Optional[np.ndarray],
      mask2: Optional[np.ndarray]
  ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Gets outer products of `mask1, mask1`, `mask1, mask2`, `mask2, mask2`."""
    def get_mask_prod(m1, m2, batch_ndim):
      if m1 is None and m2 is None:
        return None

      def reshape(m):
        if m is not None:
          if m.shape[self.channel_axis] != 1:
            raise NotImplementedError(
                f'Different channel-wise masks are not supported for '
                f'infinite-width layers now (got `mask.shape == {m.shape}). '
                f'Please describe your use case at '
                f'https://github.com/google/neural-tangents/issues/new')

          m = np.squeeze(np.moveaxis(m, (self.batch_axis, self.channel_axis),
                                     (0, -1)), -1)
          if self.is_reversed:
            m = np.moveaxis(m, range(1, m.ndim), range(m.ndim - 1, 0, -1))
        return m

      m1, m2 = reshape(m1), reshape(m2)

      start_axis = 2 - batch_ndim
      end_axis = 1 if self.diagonal_spatial else m1.ndim

      mask = utils.outer_prod(m1, m2, start_axis, end_axis, op.or_)
      return mask

    mask11 = get_mask_prod(mask1, mask1, 1 if self.diagonal_batch else 2)
    mask22 = (get_mask_prod(mask2, mask2, 1 if self.diagonal_batch else 2)
              if mask2 is not None else mask11)
    mask12 = get_mask_prod(mask1, mask2, 2)
    return mask11, mask12, mask22

  def dot_general(
      self,
      other1: Optional[np.ndarray],
      other2: Optional[np.ndarray],
      is_lhs: bool,
      dimension_numbers: lax.DotDimensionNumbers
  ) -> 'Kernel':
    """Covariances of :obj:`jax.lax.dot_general` of `x1/2` with `other1/2`."""
    if other1 is None and other2 is None:
      return self

    if other1 is not None and other2 is not None:
      if other1.ndim != other2.ndim:
        raise NotImplementedError(
            f'Factors 1/2 with different dimensionality not implemented, got '
            f'{other1.ndim} and {other2.ndim}.')

    if is_lhs:
      (other_cs, input_cs), (other_bs, input_bs) = dimension_numbers
    else:
      (input_cs, other_cs), (input_bs, other_bs) = dimension_numbers

    n_input = len(self.shape1)
    if other1 is not None:
      n_other = other1.ndim
    elif other2 is not None:
      n_other = other2.ndim
    else:
      raise ValueError(other1, other2)

    input_cs = utils.mod(input_cs, n_input)
    input_bs = utils.mod(input_bs, n_input)

    other_cs = utils.mod(other_cs, n_other)
    other_bs = utils.mod(other_bs, n_other)

    other_dims = other_bs + other_cs
    input_dims = input_bs + input_cs

    def to_kernel_dim(i: int, batch_ndim: int, is_left: bool) -> int:
      if i == self.batch_axis:
        i = 0 if (is_left or batch_ndim == 1) else 1
      elif i == self.channel_axis:
        raise ValueError(f'Batch or contracting dimension {i} cannot be equal '
                         f'to `channel_axis`.')
      else:
        i -= int(i > self.batch_axis) + int(i > self.channel_axis)
        i = batch_ndim + (1 if self.diagonal_spatial else 2) * i
        i += not is_left and not self.diagonal_spatial
      return i

    def get_other_dims(batch_ndim: int, is_left: bool) -> List[int]:
      dims = [-i - 1 - (0 if is_left or self.diagonal_spatial else n_other)
              for i in range(n_other)]
      for i_inputs, i_other in zip(input_dims, other_dims):
        dims[i_other] = to_kernel_dim(i_inputs, batch_ndim, is_left)
      return dims

    def get_mat_non_c_dims(batch_ndim: int) -> List[int]:
      input_non_c_dims = input_bs + [
          i for i in range(n_input)
          if i not in input_cs + input_bs + [self.channel_axis]
      ]

      # Batch axes are always leading in `mat`.
      if self.batch_axis in input_non_c_dims:
        input_non_c_dims.remove(self.batch_axis)
        input_non_c_dims.insert(0, self.batch_axis)

      mat_non_c_dims = []
      for i in input_non_c_dims:
        left = to_kernel_dim(i, batch_ndim, True)
        right = to_kernel_dim(i, batch_ndim, False)
        mat_non_c_dims += [left] if left == right else [left, right]
      return mat_non_c_dims

    def get_other_non_c_dims() -> List[int]:
      other_non_c_dims = [-i - 1 for i in range(n_other) if i not in other_dims]
      if not self.diagonal_spatial:
        other_non_c_dims = list(utils.zip_flat(
            other_non_c_dims,
            [-i - 1 - n_other for i in range(n_other) if i not in other_dims]))
      return other_non_c_dims

    def get_out_dims(batch_ndim: int) -> List[int]:
      mat_non_c_dims = get_mat_non_c_dims(batch_ndim)
      other_non_c_dims = get_other_non_c_dims()

      n_b_spatial = len(input_bs) - (1 if self.batch_axis in input_bs else 0)
      n_b = (len(mat_non_c_dims) if not is_lhs else
             (((0 if self.batch_axis in input_cs else batch_ndim) +
               (1 if self.diagonal_spatial else 2) * n_b_spatial)))

      return mat_non_c_dims[:n_b] + other_non_c_dims + mat_non_c_dims[n_b:]

    def dot(mat: Optional[np.ndarray],
            batch_ndim: int,
            other1: Optional[np.ndarray] = None,
            other2: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
      if mat is None or mat.ndim == 0 or other1 is None and other2 is None:
        return mat

      operands = ()

      if other1 is not None:
        other1_dims = get_other_dims(batch_ndim, True)
        operands += (other1, other1_dims)

      mat_dims = list(range(mat.ndim))
      if self.is_reversed:
        mat_dims = utils.reverse_zipped(mat_dims, batch_ndim)
      operands += (mat, mat_dims)

      if other2 is not None:
        other2_dims = get_other_dims(batch_ndim, False)
        operands += (other2, other2_dims)

      return np.einsum(*operands, get_out_dims(batch_ndim), optimize=True)

    cov1 = dot(self.cov1, 1 if self.diagonal_batch else 2, other1, other1)
    cov2 = dot(self.cov2, 1 if self.diagonal_batch else 2, other2, other2)
    nngp = dot(self.nngp, 2, other1, other2)
    ntk = dot(self.ntk, 2, other1, other2)

    lhs_ndim = n_other if is_lhs else None
    return self.replace(
        cov1=cov1,
        nngp=nngp,
        cov2=cov2,
        ntk=ntk,
        is_reversed=False,
        batch_axis=utils.axis_after_dot(self.batch_axis, input_cs,
                                        input_bs, lhs_ndim),
        channel_axis=utils.axis_after_dot(self.channel_axis, input_cs,
                                          input_bs, lhs_ndim)
    )

  def __mul__(self, other: float) -> 'Kernel':
    var = other**2
    return self.replace(cov1=var * self.cov1,
                        nngp=var * self.nngp,
                        cov2=None if self.cov2 is None else var * self.cov2,
                        ntk=None if self.ntk is None else var * self.ntk)

  __rmul__ = __mul__

  def __add__(self, other: float) -> 'Kernel':
    var = other**2
    return self.replace(cov1=var + self.cov1,
                        nngp=var + self.nngp,
                        cov2=None if self.cov2 is None else var + self.cov2)

  __sub__ = __add__

  def __truediv__(self, other: float) -> 'Kernel':
    return self.__mul__(1. / other)

  def __neg__(self) -> 'Kernel':
    return self

  __pos__ = __neg__

  def __getitem__(self, idx: utils.SliceType) -> 'Kernel':
    idx = utils.canonicalize_idx(idx, len(self.shape1))

    channel_idx = idx[self.channel_axis]
    batch_idx = idx[self.batch_axis]

    # Not allowing to index the channel axis.
    if channel_idx != slice(None):
      raise NotImplementedError(
          f'Indexing into the (infinite) channel axis {self.channel_axis} not '
          f'supported.'
      )

    # Removing the batch.
    if isinstance(batch_idx, int):
      raise NotImplementedError(
          f'Indexing an axis with an integer index (e.g. `0` vs `(0,)` removes '
          f'the respective axis. Neural Tangents requires there to always be a '
          f'batch axis ({self.batch_axis}), so it cannot be indexed with '
          f'integers (please use tuples or `slice` instead).'
      )

    spatial_idx = tuple(s for i, s in enumerate(idx) if i not in
                        (self.batch_axis, self.channel_axis))

    if self.is_reversed:
      spatial_idx = spatial_idx[::-1]

    if not self.diagonal_spatial:
      spatial_idx = utils.double_tuple(spatial_idx)

    nngp_batch_slice = (batch_idx, batch_idx)
    cov_batch_slice = (batch_idx,) if self.diagonal_batch else (batch_idx,) * 2

    nngp_slice = nngp_batch_slice + spatial_idx
    cov_slice = cov_batch_slice + spatial_idx

    nngp = self.nngp[nngp_slice]
    ntk = (self.ntk if (self.ntk is None or self.ntk.ndim == 0) else  # pytype: disable=attribute-error
           self.ntk[nngp_slice])

    cov1 = self.cov1[cov_slice]
    cov2 = None if self.cov2 is None else self.cov2[cov_slice]

    # Axes may shift if some indices are integers (and not tuples / slices).
    channel_axis = self.channel_axis
    batch_axis = self.batch_axis

    for i, s in reversed(list(enumerate(idx))):
      if isinstance(s, int):
        if i < channel_axis:
          channel_axis -= 1
        if i < batch_axis:
          batch_axis -= 1

    return self.replace(
        nngp=nngp,
        ntk=ntk,
        cov1=cov1,
        cov2=cov2,
        channel_axis=channel_axis,
        batch_axis=batch_axis,
        shape1=utils.slice_shape(self.shape1, idx),
        shape2=utils.slice_shape(self.shape2, idx),
        mask1=None if self.mask1 is None else self.mask1[idx],
        mask2=None if self.mask2 is None else self.mask2[idx],
    )
