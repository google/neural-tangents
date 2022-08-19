# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License');
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

"""Branching functions.

These layers split an input into multiple branches or fuse multiple inputs from
several branches into one.
"""


import functools
from typing import Callable, Iterable, List, Optional, Sequence, Tuple
import warnings

from jax import numpy as np
import jax.example_libraries.stax as ostax
from .requirements import layer, supports_masking
from ..utils.kernel import Kernel
from ..utils.typing import InternalLayer, InternalLayerMasked, Kernels


@layer
def FanOut(num: int) -> InternalLayer:
  """Fan-out.

  This layer takes an input and produces `num` copies that can be fed into
  different branches of a neural network (for example with residual
  connections).

  Args:
    num: The number of going edges to fan out into.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  init_fn, apply_fn = ostax.FanOut(num)
  kernel_fn = lambda k, **kwargs: [k] * num
  return init_fn, apply_fn, kernel_fn


@layer
@supports_masking(remask_kernel=False)
def FanInSum() -> InternalLayerMasked:
  """Fan-in sum.

  This layer takes a number of inputs (e.g. produced by
  :obj:`~neural_tangents.stax.FanOut`) and sums the inputs to produce a single
  output. Based on :obj:`jax.example_libraries.stax.FanInSum`.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  init_fn, apply_fn = ostax.FanInSum

  def kernel_fn(ks: Kernels, **kwargs) -> Kernel:
    ks, is_reversed = _preprocess_kernels_for_fan_in(ks)
    if not all([k.shape1 == ks[0].shape1 and
                k.shape2 == ks[0].shape2 for k in ks[1:]]):
      raise ValueError('All shapes should be equal in `FanInSum/FanInProd`, '
                       f'got `x1.shape`s of {[k.shape1 for k in ks]}, '
                       f'`x2.shape`s of {[k.shape2 for k in ks]}.')

    is_gaussian = all(k.is_gaussian for k in ks)
    if not is_gaussian and len(ks) != 1:
      # TODO(xlc): FanInSum/FanInConcat could allow non-Gaussian inputs, but
      # we need to propagate the mean of the random variables as well.
      raise NotImplementedError('`FanInSum` is only implemented for the '
                                'case where all input layers guaranteed to be '
                                'mean-zero Gaussian, i.e. having all '
                                '`is_gaussian` set to `True`, got '
                                f'{[k.is_gaussian for k in ks]}.')

    _mats_sum = lambda mats: None if mats[0] is None else sum(mats)

    cov1s = [k.cov1 for k in ks]
    cov2s = [k.cov2 for k in ks]
    nngps = [k.nngp for k in ks]
    ntks = [k.ntk for k in ks]
    cov1, cov2, nngp, ntk = map(_mats_sum, (cov1s, cov2s, nngps, ntks))

    return Kernel(cov1=cov1,
                  cov2=cov2,
                  nngp=nngp,
                  ntk=ntk,
                  x1_is_x2=ks[0].x1_is_x2,
                  is_gaussian=is_gaussian,
                  is_reversed=is_reversed,
                  is_input=ks[0].is_input,
                  diagonal_batch=ks[0].diagonal_batch,
                  diagonal_spatial=ks[0].diagonal_spatial,
                  shape1=ks[0].shape1,
                  shape2=ks[0].shape2,
                  batch_axis=ks[0].batch_axis,
                  channel_axis=ks[0].channel_axis,
                  mask1=None,
                  mask2=None)  # pytype:disable=wrong-keyword-args

  def mask_fn(mask, input_shape):
    return _sum_masks(mask)

  return init_fn, apply_fn, kernel_fn, mask_fn


@layer
@supports_masking(remask_kernel=False)
def FanInProd() -> InternalLayerMasked:
  """Fan-in product.

  This layer takes a number of inputs (e.g. produced by
  :obj:`~neural_tangents.stax.FanOut`) and elementwise-multiplies the inputs to
  produce a single output.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  init_fn, _ = ostax.FanInSum

  def apply_fn(params, inputs, **kwargs):
    return functools.reduce(np.multiply, inputs)

  def kernel_fn(ks: Kernels, **kwargs) -> Kernel:
    ks, is_reversed = _preprocess_kernels_for_fan_in(ks)
    if not all([k.shape1 == ks[0].shape1 and
                k.shape2 == ks[0].shape2 for k in ks[1:]]):
      raise ValueError('All shapes should be equal in `FanInProd`.')

    is_gaussian = len(ks) == 1 and ks[0].is_gaussian

    def _mats_prod(nngps, ntks):
      if None in ntks:
        return functools.reduce(np.multiply, nngps), None

      nngp_prod, ntk_prod = 1., 0.
      for nngp, ntk in zip(nngps, ntks):
        ntk_prod = ntk_prod * nngp + nngp_prod * ntk
        nngp_prod *= nngp
      return nngp_prod, ntk_prod

    cov1s = [k.cov1 for k in ks]
    cov2s = [k.cov2 for k in ks]
    nngps = [k.nngp for k in ks]
    ntks = [k.ntk for k in ks]

    cov1 = functools.reduce(np.multiply, cov1s)
    cov2 = None if None in cov2s else functools.reduce(np.multiply, cov2s)
    nngp, ntk = _mats_prod(nngps, ntks)

    return Kernel(cov1=cov1,
                  cov2=cov2,
                  nngp=nngp,
                  ntk=ntk,
                  x1_is_x2=ks[0].x1_is_x2,
                  is_gaussian=is_gaussian,
                  is_reversed=is_reversed,
                  is_input=ks[0].is_input,
                  diagonal_batch=ks[0].diagonal_batch,
                  diagonal_spatial=ks[0].diagonal_spatial,
                  shape1=None,
                  shape2=None,
                  batch_axis=ks[0].batch_axis,
                  channel_axis=ks[0].channel_axis,
                  mask1=None,
                  mask2=None)  # pytype:disable=wrong-keyword-args

  def mask_fn(mask, input_shape):
    return _sum_masks(mask)

  return init_fn, apply_fn, kernel_fn, mask_fn


@layer
@supports_masking(remask_kernel=False)
def FanInConcat(axis: int = -1) -> InternalLayerMasked:
  """Fan-in concatenation.

  This layer takes a number of inputs (e.g. produced by
  :obj:`~neural_tangents.stax.FanOut`) and concatenates the inputs to produce a
  single output. Based on :obj:`jax.example_libraries.stax.FanInConcat`.

  Args:
    axis: Specifies the axis along which input tensors should be concatenated.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  init_fn, apply_fn = ostax.FanInConcat(axis)

  def kernel_fn(ks: Kernels, **kwargs) -> Kernel:
    ks, is_reversed = _preprocess_kernels_for_fan_in(ks)

    diagonal_batch = ks[0].diagonal_batch
    diagonal_spatial = ks[0].diagonal_spatial

    shape1, shape2 = ks[0].shape1, ks[0].shape2

    ndim = len(shape1)
    _axis = axis % ndim
    batch_axis = ks[0].batch_axis
    channel_axis = ks[0].channel_axis

    new_shape1 = shape1[:_axis] + shape1[_axis + 1:]
    new_shape2 = shape2[:_axis] + shape2[_axis + 1:]
    for k in ks:
      k_shape1 = k.shape1[:_axis] + k.shape1[_axis + 1:]
      k_shape2 = k.shape2[:_axis] + k.shape2[_axis + 1:]
      if k_shape1 != new_shape1 or k_shape2 != new_shape2:
        raise ValueError('Non-`axis` shapes should be equal in `FanInConcat`.')

    # Check if inputs are independent Gaussians.
    if _axis != channel_axis:
      is_gaussian = all(k.is_gaussian for k in ks)
      if not is_gaussian:
        # TODO(xlc): FanInSum/FanInConcat could allow non-Gaussian inputs, but
        # we need to propagate the mean of the random variables as well.
        raise NotImplementedError(
            '`FanInConcat` layer along the non-channel axis is only implemented'
            'for the case if all input layers guaranteed to be mean-zero '
            'Gaussian, i.e. having all `is_gaussian` set to `True`.')
    else:
      # TODO(romann): allow nonlinearity after channelwise concatenation.
      # TODO(romann): support concatenating different channelwise masks.
      is_gaussian = False

    if _axis == batch_axis:
      warnings.warn(f'Concatenation along the batch axis ({_axis}) gives '
                    f'inconsistent covariances when batching - '
                    f'proceed with caution.')

    spatial_axes = tuple(i for i in range(ndim)
                         if i not in (channel_axis, batch_axis))
    # Change spatial axis according to the kernel `is_reversed`.
    if _axis in spatial_axes and is_reversed:
      _axis = spatial_axes[::-1][spatial_axes.index(_axis)]

    # Map activation tensor axis to the covariance tensor axis.
    tensor_axis_to_kernel_axis = {
        **{
            batch_axis: 0,
            channel_axis: -1,
        },
        **{
            spatial_axis: idx + 1
            for idx, spatial_axis in enumerate(spatial_axes)
        }
    }

    _axis = tensor_axis_to_kernel_axis[_axis]
    widths = [k.shape1[channel_axis] for k in ks]

    cov1 = _concat_kernels([k.cov1 for k in ks], _axis,
                           diagonal_batch, diagonal_spatial, widths)
    cov2 = _concat_kernels([k.cov2 for k in ks], _axis,
                           diagonal_batch, diagonal_spatial, widths)
    nngp = _concat_kernels([k.nngp for k in ks], _axis,
                           False, diagonal_spatial, widths)
    ntk = _concat_kernels([k.ntk for k in ks], _axis,
                          False, diagonal_spatial, widths)

    return Kernel(cov1=cov1,
                  cov2=cov2,
                  nngp=nngp,
                  ntk=ntk,
                  x1_is_x2=ks[0].x1_is_x2,
                  is_gaussian=is_gaussian,
                  is_reversed=is_reversed,
                  is_input=ks[0].is_input,
                  diagonal_batch=diagonal_batch,
                  diagonal_spatial=diagonal_spatial,
                  shape1=None,
                  shape2=None,
                  batch_axis=batch_axis,
                  channel_axis=channel_axis,
                  mask1=None,
                  mask2=None)  # pytype:disable=wrong-keyword-args

  def mask_fn(mask, input_shape):
    return _concat_masks(mask, input_shape, axis)

  return init_fn, apply_fn, kernel_fn, mask_fn


# INTERNAL UTILITIES


def _map_tuples(fn: Callable, tuples: Iterable[Tuple]) -> Tuple:
  return tuple(map(fn, zip(*(t for t in tuples))))


def _sum_masks(masks: List[Optional[np.ndarray]]) -> Optional[np.ndarray]:
  def add_two_masks(mask1, mask2):
    if mask1 is None:
      return mask2

    if mask2 is None:
      return mask1

    return mask1 & mask2

  mask = functools.reduce(add_two_masks, masks, None)
  return mask


def _concat_masks(
    masks: List[Optional[np.ndarray]],
    input_shapes: Sequence[Sequence[int]],
    axis: int) -> Optional[np.ndarray]:
  """Returns a mask which is a concatenation of `masks`.

  Since elements of `masks` can have any shapes broadcastable to respective
  elements of `input_shapes`, their concatenation may require broadcasting and
  cannot be done with a single `np.concatenate` call.

  Args:
    masks: list of masks to concatenate.
    input_shapes: list of input shapes to which the masks are applied.
    axis: concatenation axis.

  Returns:
    A single `np.ndarray` mask applicable to the concatenated inputs.
  """
  if len(masks) != len(input_shapes):
    raise ValueError(f'Number of masks ({len(masks)}) and inputs '
                     f'({len(input_shapes)}) don\'t match, please file a bug at'
                     f' https://github.com/google/neural-tangents/issues/new.')

  if all(m is None for m in masks):
    return None

  axis %= len(input_shapes[0])

  # Expand the concatenation dimension of each mask.
  masks = [m if m is None else np.broadcast_to(
      m,
      (m.shape[:axis] +
       tuple(input_shapes[i][axis: axis + 1]) +
       m.shape[axis + 1:]))
           for i, m in enumerate(masks)]

  # Max shape to broadcast all masks to along non-concat dimension.
  max_shape = _map_tuples(max, (m.shape for m in masks if m is not None))

  # Shape of the mask to replace `None` masks with.
  max_shapes = [tuple(map(min, max_shape, i)) for i in input_shapes]

  masks = [
      (np.broadcast_to(
          m,
          max_shape[:axis] + m.shape[axis: axis + 1] + max_shape[axis + 1:])
       if m is not None
       else np.zeros_like(max_shapes[i], dtype=np.bool_))
      for i, m in enumerate(masks)
  ]

  return np.concatenate(masks, axis)


def _preprocess_kernels_for_fan_in(ks: Kernels) -> Tuple[List[Kernel], bool]:
  # Check diagonal requirements.
  if not all(k.diagonal_batch == ks[0].diagonal_batch and
             k.diagonal_spatial == ks[0].diagonal_spatial and
             k.batch_axis == ks[0].batch_axis and
             k.channel_axis == ks[0].channel_axis
             for k in ks[1:]):
    raise NotImplementedError('`FanIn` layers are only implemented for the '
                              'case if all input layers output the same layout '
                              'of covariance matrices, i.e. having all '
                              'matching `diagonal_batch` and '
                              '`diagonal_spatial` and other attributes.')

  # If kernels have different spatial axes order, transpose some of them.
  n_kernels = len(ks)
  n_reversed = sum(ker.is_reversed for ker in ks)
  ks = list(ks)

  if n_reversed > n_kernels / 2:
    is_reversed = True
    for i in range(n_kernels):
      if not ks[i].is_reversed:
        ks[i] = ks[i].reverse()

  else:
    is_reversed = False
    for i in range(n_kernels):
      if ks[i].is_reversed:
        ks[i] = ks[i].reverse()

  # Warnings.
  warnings.warn('`FanIn` layers assume independent inputs which is not verified'
                ' in the code. Please make sure to have at least one `Dense` / '
                '`Conv` / `GlobalSelfAttention` etc. layer in each branch.')

  return ks, is_reversed


def _concat_kernels(
    mats: Sequence[Optional[np.ndarray]],
    axis: int,
    diagonal_batch: bool,
    diagonal_spatial: bool,
    widths: Sequence[int]) -> Optional[np.ndarray]:
  """Compute the covariance of concatenated activations with given covariances.

  Args:
    mats: Covariance tensors of the same shape.

    axis: Specifies the axis along which the covariances (not activations) are
      concatenated. `-1` corresponds to averaging.

    diagonal_batch: Specifies whether `cov1` and `cov2` store only
      the diagonal of the sample-sample covariance
      (`diagonal_batch == True`,
       `cov1.shape == (batch_size_1, ...)`),
      or the full covariance
      (`diagonal_batch == False`,
       `cov1.shape == (batch_size_1, batch_size_1, ...)`).

    diagonal_spatial: Specifies whether only the diagonals of the
      location-location covariances will be computed,
      (`diagonal_spatial == True`,
       `nngp.shape == (batch_size_1, batch_size_2, height, width, depth, ...)`),
      or the full covariance
      (`diagonal_spatial == False`,
       `nngp.shape == (batch_size_1, batch_size_2, height, height,
                       width, width, depth, depth, ...)`).

    widths: list of integer channel widths of the finite model inputs.

  Returns:
    A new `np.ndarray` representing covariance between concatenated activations.
  """
  if mats[0] is None:
    return None

  n_mats = len(mats)
  mat_ndim = mats[0].ndim

  # Averaging if concatenating along features or diagonalized dimension.
  if axis == -1:
    if all(w == widths[0] for w in widths):
      widths = [1] * len(widths)
    mat = sum(mats[i] * widths[i] for i in range(n_mats)) / sum(widths)

  # Simple concatenation along the axis if the axis is not duplicated.
  elif ((axis == 0 and diagonal_batch) or
        (axis != 0 and diagonal_spatial)):
    concat_axis = axis + (0 if diagonal_batch else 1)
    mat = np.concatenate(mats, concat_axis)

  # 2D concatenation with insertion of 0-blocks if the axis is present twice.
  else:
    rows = []
    pad_axis = max(0, 2 * axis - (1 if diagonal_batch else 0))
    for i, mat in enumerate(mats):
      pads = [(0, 0)] * mat_ndim
      pads[pad_axis] = (
          sum(mats[j].shape[pad_axis] for j in range(i)),
          sum(mats[j].shape[pad_axis] for j in range(i + 1, n_mats))
      )
      rows.append(np.pad(mat, pads))
    mat = np.concatenate(rows, pad_axis + 1)

  return mat

