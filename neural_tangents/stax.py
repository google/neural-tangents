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

"""Analytic NNGP and NTK library.

This library contains layer constructors mimicking those in
`jax.experimental.stax` with similar API apart apart from:

1) Instead of `(init_fn, apply_fn)` tuple, layer constructors return a triple
  `(init_fn, apply_fn, kernel_fn)`, where the added `kernel_fn` maps an
  `Kernel` to a new `Kernel`, and represents the change in the
  analytic NTK and NNGP kernels (fields of `Kernel`). These functions
  are chained / stacked together within the `serial` or `parallel` combinators,
  similarly to `init_fn` and `apply_fn`.

2) In layers with random weights, NTK parameterization is used by default
  (https://arxiv.org/abs/1806.07572, page 3). Standard parameterization
  (https://arxiv.org/abs/2001.07301) can be specified for Conv and Dense layers
  by a keyword argument.

3) Some functionality may be missing (e.g. `BatchNorm`), and some may be present
  only in our library (e.g. `CIRCULAR` padding, `LayerNorm`, `GlobalAvgPool`,
  `GlobalSelfAttention` etc.).

Example:
  ```python
  >>> from jax import random
  >>> import neural_tangents as nt
  >>> from neural_tangents import stax
  >>>
  >>> key1, key2 = random.split(random.PRNGKey(1), 2)
  >>> x_train = random.normal(key1, (20, 32, 32, 3))
  >>> y_train = random.uniform(key1, (20, 10))
  >>> x_test = random.normal(key2, (5, 32, 32, 3))
  >>>
  >>> init_fn, apply_fn, kernel_fn = stax.serial(
  >>>     stax.Conv(128, (3, 3)),
  >>>     stax.Relu(),
  >>>     stax.Conv(256, (3, 3)),
  >>>     stax.Relu(),
  >>>     stax.Conv(512, (3, 3)),
  >>>     stax.Flatten(),
  >>>     stax.Dense(10)
  >>> )
  >>>
  >>> # (5, 10) np.ndarray NNGP test prediction
  >>> y_test_nngp = nt.predict.gp_inference(kernel_fn, x_train, y_train, x_test,
  >>>                                       get='nngp')
  >>>
  >>> # (5, 10) np.ndarray NTK prediction
  >>> y_test_ntk = nt.predict.gp_inference(kernel_fn, x_train, y_train, x_test,
  >>>                                      get='ntk')
  ```
"""

import functools
import operator as op
import warnings
import enum
from jax import lax
from jax import linear_util as lu
from jax import ops
from jax.abstract_arrays import ShapedArray
from jax.api_util import flatten_fun
import jax.experimental.stax as ostax
import jax.numpy as np
from jax.scipy.special import erf
from neural_tangents.utils.kernel import Kernel
from neural_tangents.utils.kernel import Marginalisation as M
import frozendict
from jax import random
import jax.interpreters.partial_eval as pe
from jax.tree_util import tree_map, tree_flatten, tree_unflatten
from neural_tangents.utils import utils


_CONV_QAB_DIMENSION_NUMBERS = ('NCHW', 'OIHW', 'NCHW')
_INPUT_REQ = 'covariances_req'
_DEFAULT_INPUT_REQ = frozendict.frozendict({'marginal': M.OVER_ALL,
                                            'cross': M.OVER_ALL})
_NEG_INF = -1e20  # softmax raises an error if all entries are -np.inf


class Padding(enum.Enum):
  CIRCULAR = 'CIRCULAR'
  SAME = 'SAME'
  VALID = 'VALID'


class Pooling(enum.Enum):
  AVG = 'AVG'
  SUM = 'SUM'


def _set_input_req_attr(combinator_kernel_fn, kernel_fns):
  """Labels which covariances are required by the individual layers
  combined in `combinator_kernel_fn` based on `kernel_fns`.

  Specifically, sets `combinator_kernel_fn`'s attribute `covariances_req`
  to a dictionary with keys `marginal` and `cross` which respectively correspond
  to the types of covariances tracked in `var1`/`var2` and `nngp`/`ntk`.
  The current combinations for `marginal` and `cross` are:
    (`Marginalisation.OVER_ALL`, `Marginalisation.OVER_ALL`) if only Dense
      layers are use
    (`Marginalisation.OVER_PIXELS`, `Marginalisation.OVER_PIXELS`) if CNN but no
      average pooling is used
    (`Marginalisation.OVER_POINTS`, `Marginalisation.NO`) if CNN and average
      pooling or attention are used
  #TODO: make `NO` marginalisation the default

  Args:
    combinator_kernel_fn: a 'kernel_fn` of a `serial` or `parallel` combinator.
    kernel_fns: list of 'kernel_fn`s fed to the `kernel_fns` (e.g. a list of
      convolutional layers and nonlinearities to be chained together with the
      `serial` combinator).

  Returns:
    `kernel_fns` with the `_COVARIANCES_REQ` attribute set accordingly to
      the needs of their corresponding layer
  """
  def _get_maximal_element(input_req, comparison_op):
    for f in reversed(kernel_fns):
      if hasattr(f, _INPUT_REQ):
        reqs = getattr(f, _INPUT_REQ)

        marginal = reqs['marginal']
        if comparison_op(marginal, input_req['marginal']):
          input_req['marginal'] = marginal

        cross = reqs['cross']
        if comparison_op(cross, input_req['cross']):
          input_req['cross'] = cross

        if 'spec' in reqs:
          input_req['spec'] = reqs['spec']

    return input_req

  # `_get_maximal_element` sets up the code for `NO` marginalisation by default
  input_req = _get_maximal_element(
      dict(_DEFAULT_INPUT_REQ), lambda x, y: x > y)

  setattr(combinator_kernel_fn, _INPUT_REQ, input_req)
  return combinator_kernel_fn


def _randn(stddev=1e-2):
  """`jax.experimental.stax.randn` for implicitly-typed results."""
  def init(rng, shape):
    return stddev * random.normal(rng, shape)
  return init


def _double_tuple(x):
  return tuple(v for v in x for _ in range(2))


def _zip_flat(x, y):
  return tuple(c for xy in zip(x, y) for c in xy)


def _interleave_ones(x, start_axis, x_first):
  x_axes = x.shape[start_axis:]
  ones = (1,) * (x.ndim - start_axis)
  shape = x.shape[:start_axis]
  if x_first:
    shape += _zip_flat(x_axes, ones)
  else:
    shape += _zip_flat(ones, x_axes)
  return x.reshape(shape)


def _outer_prod(x, y, start_axis, prod_op):
  x = _interleave_ones(x, start_axis, True)
  y = _interleave_ones(y, start_axis, False)
  return prod_op(x, y)


def _zip_axes(mat, start_axis=0, unzip=False):
  """Zip (interleave) axes starting from `start_axis`.

  Changes the shape as follows:
    If `unzip == True`:
    [..., X, X, ..., Y, Y, ..., Z, Z, ...] -> [..., X, Y, Z, ..., X, Y, Z, ...]
    If `unzip == False`:
    [..., X, Y, Z, ..., X, Y, Z, ...] -> [..., X, X, ..., Y, Y, ..., Z, Z, ...]

  Args:
    mat: `np.ndarray` with an even number of dimensions following `start_axis`.
    start_axis: `int`, number of axis from which to zip (interleave).
    unzip: `bool`, set to `True` to do the opposite.

  Returns:
    A `np.ndarray` with a new shape.
  """
  n_axes, ragged = divmod(mat.ndim - start_axis, 2)
  if ragged:
    raise ValueError('Need even number of axes to zip, got %d.'
                     % (mat.ndim - start_axis))

  odd_axes = range(start_axis + 1, start_axis + 1 + n_axes * 2, 2)
  last_axes = range(-n_axes, 0)

  if unzip:
    mat = np.moveaxis(mat, odd_axes, last_axes)
  else:
    mat = np.moveaxis(mat, last_axes, odd_axes)
  return mat


def _size_at(x, axis):
  if isinstance(x, np.ndarray):
    x = x.shape
  elif not isinstance(x, tuple):
    raise TypeError(x, type(x))

  return functools.reduce(op.mul, (x[i] for i in axis), 1)


def _parse_axes(spec, x):
  if isinstance(x, np.ndarray):
    ndim = x.ndim
  elif isinstance(x, tuple):
    ndim = len(x)
  else:
    raise TypeError(x, type(x))

  if spec is None:
    batch_axis, channel_axis = 0, ndim - 1
  else:
    batch_axis, channel_axis = spec.index('N'), spec.index('C')

  spatial_axes = tuple(a for a in range(ndim)
                       if a not in (batch_axis, channel_axis))
  return batch_axis, channel_axis, spatial_axes


def _canonicalize_axis(axis, x):
  axis = (axis,) if isinstance(axis, int) else tuple(axis)
  if isinstance(x, np.ndarray):
    ndim = x.ndim
  elif isinstance(x, tuple):
    ndim = len(x)
  else:
    raise TypeError(x, type(x))
  return list(set(np.arange(ndim)[list(axis)]))


def _variance_over_all_over_pixels(x, batch_axis, channel_axis):
  ret = np.sum(x ** 2, axis=channel_axis)
  new_batch_axis = batch_axis - (1 if batch_axis > channel_axis else 0)
  ret = np.moveaxis(ret, new_batch_axis, 0)
  return ret


def _variance_over_points(x, batch_axis, channel_axis):
  x = np.moveaxis(x, (batch_axis, channel_axis), (0, -1))
  ret = lax.dot_general(x, x, (((x.ndim - 1,), (x.ndim - 1,)),
                               ((0,), (0,))))
  ret = _zip_axes(ret, 1)
  return ret


def _covariance_no_marg(x1, x2, batch_axis, channel_axis):
  ret = np.tensordot(x1, x2, (channel_axis, channel_axis))
  new_batch_axis = batch_axis - (1 if batch_axis > channel_axis else 0)
  ret = np.moveaxis(ret, (new_batch_axis, x1.ndim - 1 + new_batch_axis), (0, 1))
  ret = _zip_axes(ret, 2)
  return ret


def _covariance_over_all_over_pixels(x1, x2, batch_axis, channel_axis):
  ret = np.matmul(np.moveaxis(x1, (batch_axis, channel_axis), (-2, -1)),
                  np.moveaxis(x2, (batch_axis, channel_axis), (-1, -2)))
  ret = np.moveaxis(ret, (-2, -1), (0, 1))
  return ret


def _get_variance(x, marginal_type, batch_axis, channel_axis):
  if marginal_type in (M.OVER_ALL, M.OVER_PIXELS):
    ret = _variance_over_all_over_pixels(x, batch_axis, channel_axis)

  elif marginal_type == M.OVER_POINTS:
    ret = _variance_over_points(x, batch_axis, channel_axis)

  elif marginal_type == M.NO:
    ret = _covariance_no_marg(x, x, batch_axis, channel_axis)

  else:
    raise NotImplementedError(
        "Only implemented for `OVER_ALL`, `OVER_PIXELS`, `OVER_POINTS` "
        "and `NO`; supplied {}".format(marginal_type))

  return ret / x.shape[channel_axis]


def _get_covariance(x1, x2, marginal_type, batch_axis, channel_axis):
  """Computes uncentred covariance (nngp) between two sets of inputs

  Args:
    x1: a (2+k)D (k = 0, 2) `np.ndarray` of shape
      `[n1, <k inner dimensions>, n_features]`.`n1`, `n_features` may be in a
      different position based on `batch_axis` and `channel_axis`.
    x2: an optional `np.ndarray` that has the same shape as `a` apart from
      possibly different batch (`n2`) dimension. `None` means `x2 == x1`.
    marginal_type: an instance of `Marginalisation` specifying between which
      dimensions should the covariances be computed.
    batch_axis: integer, specifying which axis is the batch axis.
    channel_axis: integer, specifying which axis is the channel / feature axis.

  Returns:
    an `np.ndarray` with uncentred batch covariance with shape
    `[n1, n2]`
    `+ [<k inner dimensions>]` (if `covar_type` is `OVER_PIXELS`)
    `+ [<k inner dimensions>, <k spatial dimensions>]` (if `covar_type` is
    `OVER_POINTS` or `NO`).
  """
  x2 = x1 if x2 is None else x2

  if marginal_type in (M.OVER_ALL, M.OVER_PIXELS):
    ret = _covariance_over_all_over_pixels(x1, x2, batch_axis, channel_axis)

  elif marginal_type == M.NO:
    ret = _covariance_no_marg(x1, x2, batch_axis, channel_axis)

  else:
    raise NotImplementedError(
        "Only implemented for `OVER_ALL`, `OVER_PIXELS`, `OVER_POINTS` "
        "and `NO`; supplied {}".format(marginal_type))

  return ret / x1.shape[channel_axis]


def _diag_mul_over_points(A, mul):
  ndims, ragged = divmod(A.ndim, 2)
  if ragged:
    raise ValueError(f'Expected an even-dimensional kernel, got {A.ndim}.')

  diag = A
  idx = ()
  for i in range(ndims):
    if A.shape[2 * i] != A.shape[2 * i + 1]:
      raise ValueError(f'Expected a kernel with the same even and odd axes '
                       f'sizes, got {A.shape}.')
    shape = [1] * ndims
    size = A.shape[2 * i]
    shape[i] = size
    idx += (np.arange(size).reshape(shape),) * 2

    diag = np.diagonal(diag)

  A = ops.index_update(A, idx, diag * mul)
  return A


def _diag_mul_over_pixels(A, mul):
  idx = np.diag_indices(A.shape[0]) + (Ellipsis,)
  diag = np.moveaxis(np.diagonal(A), -1, 0)
  A = ops.index_update(A, idx, diag * mul)
  return A


def _diag_mul_over_all(A, mul):
  idx = np.diag_indices(A.shape[0])
  diag = np.diag(A)
  A = ops.index_update(A, idx, diag * mul)
  return A


def _diag_mul(A, mul, cross):
  if A.shape[0] != A.shape[1]:
    return A

  if cross == M.OVER_ALL:
    return _diag_mul_over_all(A, mul)
  elif cross == M.OVER_PIXELS:
    return _diag_mul_over_pixels(A, mul)
  elif cross == M.NO:
    return _diag_mul_over_points(A, mul)

  raise NotImplementedError(cross)


def _inputs_to_kernel(x1,
                      x2,
                      marginal,
                      cross,
                      compute_ntk,
                      spec=None,
                      eps=1e-12,
                      mask_constant=None):
  """Transforms (batches of) inputs to a `Kernel`.

  This is a private method. Docstring and example are for internal reference.

   The kernel contains the empirical covariances between different inputs and
     their entries (pixels) necessary to compute the covariance of the Gaussian
     Process corresponding to an infinite Bayesian or continuous gradient
     descent trained neural network.

   The smallest necessary number of covariance entries is tracked. For example,
     all networks are assumed to have i.i.d. weights along the channel / feature
     / logits dimensions, hence covariance between different entries along these
     dimensions is known to be 0 and is not tracked.

  Args:
    x1: an (N+2)D `np.ndarray` of shape
      `[batch_size_1, height, width, depth, ..., n_features]` with `N` spatial
      dimensions (`N >= 0`).
    x2: an optional `np.ndarray` with the same shape as `x1` apart
      from possibly different leading batch size. `None` means
      `x2 == x1`.
    marginal: an instance of `Marginalisation` specifying for which spatial
      dimensions should the covariances be tracked in `var1`/`var2`.
    cross: an instance of `Marginalisation` specifying for which spatial
      dimensions should the covariances be tracked in `nngp`/`ntk`.
    compute_ntk: a boolean, `True` to compute both NTK and NNGP kernels,
      `False` to only compute NNGP.
    spec: an optional `string`, specifying the dimension order of
      the input, e.g. `NCHW` or `NHWC` or `NCHWDXYZ`.
    eps: a small number used to check whether x1 and x2 are the same up to
        `eps`.

    Example:
      ```python
          >>> x = np.ones((10, 32, 16, 3))
          >>> o = _inputs_to_kernel(x, None,
          >>>                       marginal=M.OVER_POINTS,
          >>>                       cross=M.NO,
          >>>                       compute_ntk=True)
          >>> o.var1.shape, o.ntk.shape
          (10, 32, 32, 16, 16), (10, 10, 32, 32, 16, 16)
          >>> o = _inputs_to_kernel(x, None,
          >>>                       marginal=M.OVER_PIXELS,
          >>>                       cross=M.OVER_PIXELS,
          >>>                       compute_ntk=True)
          >>> o.var1.shape, o.ntk.shape
          (10, 32, 16), (10, 10, 32, 16)
          >>> x1 = np.ones((10, 128))
          >>> x2 = np.ones((20, 128))
          >>> o = _inputs_to_kernel(x1, x2,
          >>>                       marginal=M.OVER_ALL,
          >>>                       cross=M.OVER_ALL,
          >>>                       compute_ntk=False)
          >>> o.var1.shape, o.nngp.shape
          (10,), (10, 20)
          >>> o.ntk
          None
      ```

  Returns:
    a `Kernel` object.
  """
  if x1.ndim < 2:
    raise ValueError('Inputs must have be least 2D (one batch dimension and one '
                     'feature dimension), got %d.' % x1.ndim)

  if cross == M.OVER_POINTS:
    raise ValueError('Required `OVER_POINTS` to be computed for `nngp`/`ntk`. '
                     '`OVER_POINTS` is only meant for `var1`/`var2`. '
                     'Use `NO` instead to compute all covariances.')

  # Batch axis first, channel / feature axis last by default.
  batch_axis, channel_axis, _ = _parse_axes(spec, x1)

  batch_axis = spec.index('N') if spec else 0
  if batch_axis != 0:
    # TODO: add support or clear error for batching.
    warnings.warn('!!! Non-leading (!= 0) batch dimension (`N`) in the '
                  'input layer is not supported for batching and empirical '
                  'kernels, got spec = %s. !!!' % spec)

  # Flatten inputs if marginalizing over everything.
  if cross == marginal == M.OVER_ALL:
    n1 = x1.shape[batch_axis]
    x1 = np.reshape(np.moveaxis(x1, batch_axis, 0), (n1, -1))
    if x2 is not None:
      n2 = x2.shape[batch_axis]
      x2 = np.reshape(np.moveaxis(x2, batch_axis, 0), (n2, -1))

    # Update batch and channel axes if inputs are flattened to `NC`.
    batch_axis, channel_axis = 0, 1

  # Generate masks and zero-out masked values if needed.
  x1, mask1 = _get_masked_inputs_and_mask(x1, mask_constant)
  x2, mask2 = _get_masked_inputs_and_mask(x2, mask_constant)

  # TODO: Think more about dtype automatic vs manual dtype promotion.
  x1 = x1.astype(np.float64)
  var1 = _get_variance(x1, marginal, batch_axis, channel_axis)
  x1_is_x2 = utils.x1_is_x2(x1, x2, eps=eps)

  if x2 is None:
    x2 = x1
    var2 = None

  else:
    x1_non_batch_shape = list(x1.shape)
    x1_non_batch_shape.pop(batch_axis)

    x2_non_batch_shape = list(x2.shape)
    x2_non_batch_shape.pop(batch_axis)

    if x1_non_batch_shape != x2_non_batch_shape:
      raise ValueError('`x1` and `x2` are expected to be batches of inputs'
                       ' with the same shape (apart from the batch size),'
                       ' got %s and %s.' %
                       (str(x1.shape), str(x2.shape)))

    x2 = x2.astype(np.float64)
    var2 = _get_variance(x2, marginal, batch_axis, channel_axis)

  nngp = _get_covariance(x1, x2, cross, batch_axis, channel_axis)
  ntk = 0. if compute_ntk else None

  is_reversed = False
  is_gaussian = False
  is_input = True

  return Kernel(var1, nngp, var2, ntk, is_gaussian, is_reversed,
                marginal, cross, x1.shape, x2.shape, x1_is_x2, is_input,
                mask1, mask2)


def _propagate_shape(init_fn, shape):
  """Statically, abstractly, evaluate the init_fn to get shape information."""
  akey = ShapedArray((2,), np.uint32)
  closed_init_fn = functools.partial(init_fn, input_shape=shape)
  args_flat, in_tree = tree_flatten(((akey,), {}))
  fun, out_tree = flatten_fun(lu.wrap_init(closed_init_fn), in_tree)
  out = pe.abstract_eval_fun(fun.call_wrapped, akey)
  out_shape = tree_unflatten(out_tree(), out)[0]
  out_shape = tree_map(lambda x: int(x.val), out_shape)
  return out_shape


def _apply_kernel(init_fn, kernel_fn, in_kernel):
  """Apply a kernel_fn to a Kernel propagating side information."""
  out_kernel = kernel_fn(in_kernel)
  if isinstance(in_kernel, Kernel):
    shape1 = _propagate_shape(init_fn, in_kernel.shape1)
    shape2 = _propagate_shape(init_fn, in_kernel.shape2)
  elif isinstance(in_kernel, list):
    shape1 = _propagate_shape(init_fn, [k.shape1 for k in in_kernel])
    shape2 = _propagate_shape(init_fn, [k.shape2 for k in in_kernel])
  else:
    raise TypeError((
        'Expected input kernel to be a Kernel or a list of kernels.'
        ' Found {}.'.format(type(out_kernel))))

  if isinstance(out_kernel, Kernel):
    return out_kernel._replace(shape1=shape1, shape2=shape2)
  elif isinstance(out_kernel, list):
    return [k._replace(shape1=s1, shape2=s2) for
            k, s1, s2 in zip(out_kernel, shape1, shape2)]
  else:
    raise TypeError((
        'Expected output kernel to be a Kernel or a list of kernels.'
        ' Found {}.'.format(type(out_kernel))))


def _check_marginalization(kernel_fn, kernel):
  if isinstance(kernel, list):
    for k in kernel:
      _check_marginalization(kernel_fn, k)
    return

  deps = getattr(kernel_fn, _INPUT_REQ, _DEFAULT_INPUT_REQ)
  kernel_deps = {'marginal': kernel.marginal, 'cross': kernel.cross}
  if kernel.marginal < deps['marginal'] or kernel.cross < deps['cross']:
    raise ValueError((
        'Attempted to apply a {} layer to a kernel that contains '
        'insufficient information due to marginalization. Found '
        'marginalization level {} expected at least {}. To fix this specify '
        '`marginalization=\'none\' in the kernel_fn or specify a '
        'marginalization level manually.'
        ).format(kernel_fn.__name__, kernel_deps, deps))


def _preprocess_kernel_fn(init_fn, kernel_fn):
  def new_kernel_fn(x1_or_kernel,
                    x2=None,
                    get=None,
                    marginalization='auto',
                    mask_constant=None):
    """Returns the `Kernel` resulting from applying `ker_fun` to given inputs.

    Args:
      x1_or_kernel: either a `np.ndarray` with shape
        `[batch_size_1] + input_shape`, or a `Kernel`.
      x2: an optional `np.ndarray` with shape `[batch_size_2] + input_shape`.
        `None` means `x2 == x1` or `x1_or_kernel is Kernel`.
      get: either `None`, a string, or a tuple of strings specifying which data
        should be returned by the kernel function. Can be "nngp", "ntk", "var1",
        "var2", "is_gaussian", "is_reversed", "marginal", "cross".
      marginalization: Either a string with value "auto" or "none" or a dict.
        If "auto" then stax attempts to automatically identify which
        dimensions are most appropriate to marginalize over. If "none" then no
        marginalization is performed. If a dict then the user can manually
        specify the marginalization for the self- and cross- correlations.
      mask_constant: TODO(romann).
    Returns:
      If `get` is a string, returns the requested `np.ndarray`. If `get` is a
      tuple, returns an `AnalyticKernel` namedtuple containing only the
      requested information. If `get` is None then a Kernel object is returned
      containing all the data.
    """
    if (isinstance(x1_or_kernel, Kernel) or
        (isinstance(x1_or_kernel, list) and
         all(isinstance(k, Kernel) for k in x1_or_kernel))):

      if mask_constant is not None:
        raise ValueError('`mask_constant` parameters only apply to '
                         'array inputs, and would have no effect on a '
                         '`Kernel`.')

      _check_marginalization(kernel_fn, x1_or_kernel)
      return _apply_kernel(init_fn, kernel_fn, x1_or_kernel)

    return outer_kernel_fn(x1_or_kernel,
                           x2,
                           get,
                           marginalization,
                           mask_constant)

  @utils.get_namedtuple('AnalyticKernel')
  def outer_kernel_fn(x1, x2, get, marginalization, mask_constant):
    if isinstance(x1, tuple) or isinstance(x2, tuple):
      raise NotImplementedError('Only `mask_constant` masking mode is supported'
                                ' for `kernel_fn` currently')

    if not isinstance(x1, np.ndarray):
      raise TypeError('Inputs to a kernel propagation function should be '
                      'a `Kernel`, '
                      'a `list` of `Kernel`s, '
                      'or a (tuple of) `np.ndarray`(s), got %s.' % type(x1))

    if not (x2 is None or isinstance(x2, np.ndarray)):
      raise TypeError('`x2` to a kernel propagation function '
                      'should be `None` or a `np.ndarray`, got %s.'
                      % type(x2))

    input_req = getattr(
        kernel_fn, _INPUT_REQ, _DEFAULT_INPUT_REQ)
    if marginalization != 'auto':
      if isinstance(marginalization, dict):
        for k in input_req:
          if marginalization[k] > input_req[k]:
            input_req[k] = marginalization[k]
      elif marginalization == 'none' and x1.ndim > 2:
        input_req = {'marginal': M.OVER_POINTS, 'cross': M.NO}
      else:
        raise NotImplementedError(
            ('marginalization should be set to one of "auto", "none", or a dict'
             'specifying the marginalization levels of the variance and the '
             'covariance respectively. Found {}.'.format(marginalization)))

    compute_ntk = (get is None) or ('ntk' in get)

    kernel = _inputs_to_kernel(x1, x2,
                               compute_ntk=compute_ntk,
                               mask_constant=mask_constant,
                               **input_req)
    return _apply_kernel(init_fn, kernel_fn, kernel)

  if hasattr(kernel_fn, _INPUT_REQ):
    setattr(new_kernel_fn,
            _INPUT_REQ,
            getattr(kernel_fn, _INPUT_REQ))

  return new_kernel_fn


def _layer(layer):
  """A convenience decorator to be added to all public layers like `Relu` etc.

  Makes the `kernel_fn` of the layer work with both input `np.ndarray`s (when
    the layer is the first one applied to inputs), and with `Kernel` for
    intermediary layers. Also adds optional arguments to make the `kernel_fn`
    call the empirical Monte Carlo kernel estimation instead of computing it
    analytically (by default), as well as specifying the batching strategy.

  Args:
    layer: A layer function returning a triple `(init_fn, apply_fn, kernel_fn)`.

  Returns:
    A function with the same signature as `layer` with `kernel_fn` now
    accepting `np.ndarray`s as inputs if needed, and optional `n_samples=0`,
    `key=None`, `compute_ntk=True` arguments to let the user indicate that
    they want the kernel to be computed by Monte Carlo sampling.
  """
  @utils.wraps(layer)
  def layer_fn(*args, **kwargs):
    init_fn, apply_fn, kernel_fn = layer(*args, **kwargs)
    kernel_fn = _preprocess_kernel_fn(init_fn, kernel_fn)
    init_fn.__name__ = apply_fn.__name__ = kernel_fn.__name__ = layer.__name__
    return init_fn, apply_fn, kernel_fn
  return layer_fn


def _elementwise(fn, **fn_kwargs):
  init_fn, apply_fn_old = ostax.elementwise(fn, **fn_kwargs)

  def apply_fn(params, inputs, mask_constant=None, **kwargs):
    inputs, mask = _get_masked_inputs_and_mask(inputs, mask_constant)
    outputs = apply_fn_old(params, inputs, **kwargs)
    return _drop_mask(outputs, mask)

  kernel_fn = lambda kernels: _transform_kernels(kernels, fn, **fn_kwargs)
  return init_fn, apply_fn, kernel_fn


def _ab_relu(x, a, b, **kwargs):
  return a * np.minimum(x, 0) + b * np.maximum(x, 0)


def _erf(x, **kwargs):
  return erf(x)


@_layer
def Erf(do_backprop=False):
  return _elementwise(_erf, do_backprop=do_backprop)


@_layer
def Relu(do_backprop=False, do_stabilize=False):
  return _elementwise(_ab_relu, a=0, b=1, do_backprop=do_backprop,
                      do_stabilize=do_stabilize)


@_layer
def ABRelu(a, b, do_backprop=False, do_stabilize=False):
  return _elementwise(_ab_relu, a=a, b=b, do_backprop=do_backprop,
                      do_stabilize=do_stabilize)


@_layer
def LeakyRelu(alpha, do_backprop=False, do_stabilize=False):
  return _elementwise(_ab_relu, a=alpha, b=1, do_backprop=do_backprop,
                      do_stabilize=do_stabilize)


@_layer
def Abs(do_backprop=False, do_stabilize=False):
  return _elementwise(_ab_relu, a=-1, b=1, do_backprop=do_backprop,
                      do_stabilize=do_stabilize)


def _arccos(x, do_backprop):
  if do_backprop:
    # https://github.com/google/jax/issues/654
    x = np.where(np.abs(x) >= 1, np.sign(x), x)
  else:
    x = np.clip(x, -1, 1)
  return np.arccos(x)


def _sqrt(x, do_backprop):
  if do_backprop:
    # https://github.com/google/jax/issues/654
    x = np.where(x <= 0, 0, x)
  else:
    x = np.maximum(x, 0)
  return np.sqrt(x)


def _safe_sqrt(x):
  return np.sqrt(np.maximum(x, 1e-20))


def _arcsin(x, do_backprop):
  if do_backprop:
    # https://github.com/google/jax/issues/654
    x = np.where(np.abs(x) >= 1, np.sign(x), x)
  else:
    x = np.clip(x, -1, 1)
  return np.arcsin(x)


def _get_dimensionwise_marg_var(var, marginal):
  """Extracts `OVER_ALL`/`OVER_PIXELS` marginal covariance from `var1`/`var2` of
  either of the `OVER_POINTS` or `NO` types.
  """
  if marginal in (M.OVER_ALL, M.OVER_PIXELS):
    return var
  elif marginal == M.NO:
    var = np.moveaxis(np.diagonal(var, axis1=0, axis2=1), -1, 0)

  # [..., X, X, Y, Y, Z, Z, ...] -> [..., X, Y, Z, ..., X, Y, Z, ...]
  var = _zip_axes(var, 1, unzip=True)
  spatial_shape = var.shape[1 + var.ndim // 2:]
  spatial_shape_prod = functools.reduce(op.mul, spatial_shape, 1)

  sqnorms = np.diagonal(
      var.reshape((-1, spatial_shape_prod, spatial_shape_prod)),
      axis1=-2,
      axis2=-1)
  sqnorms = sqnorms.reshape((-1,) + spatial_shape)
  return sqnorms


def _get_normalising_prod(var1, var2, marginal, axis=()):
  """Returns three tensors, `prod11`, `prod12` and `prod22` which contain
  products of marginal variances of `var1`, `nngp` and `var2` respectively.

  `prod12` is an (2*N+2)D tensor where an entry [x1, x2, h, h, w, w, ...] equals
  k_{ab}(x1, x1) * k_{cd}(x2, x2), if `marginal` is `OVER_POINTS` or `NO`,
  or a (N+2)D tensor k_{aa}(x1, x1) k_{cc}(x2, x2) if `marginal` is
  `OVER_PIXELS`, or a 2D tensor k(x1, x1) k(x2, x2) if `marginal` is `OVER_ALL`.
  In the last two cases, both `prod11` and `prod22` will be `None`. Otherwise
  they will be (2*N+1)D tensors k_{ab}(x1, x1) k_{cd}(x1, x1) in the
  `marginal == OVER_POINTS` case, or (2*N+2)D tensors akin to the one for
  `prod12` if `marginal == NO`.
  """
  axis = (axis,) if isinstance(axis, int) else tuple(axis)
  same_input = var2 is None
  if same_input:
    var2 = var1
  else:
    if var1.shape[1:] != var2.shape[1:]:
      raise ValueError(var1.shape, var2.shape)

  if marginal in (M.OVER_ALL, M.OVER_PIXELS):
    if marginal == M.OVER_ALL and len(axis) > 0:
      raise ValueError("Required normalisation over axis={} is impossible when"
                       " {}. Maybe axis=()?".format(axis, marginal))
    sqnorms1, sqnorms2 = var1, var2
    sqnorms1 = np.mean(sqnorms1, axis=axis, keepdims=True)
    if same_input:
      sqnorms2 = sqnorms1
    else:
      sqnorms2 = np.mean(sqnorms2, axis=axis, keepdims=True)

    prod12 = np.expand_dims(sqnorms1, 1) * np.expand_dims(sqnorms2, 0)
    prod11 = sqnorms1**2.0
    prod22 = sqnorms2**2.0 if not same_input else prod11

  elif marginal in (M.OVER_POINTS, M.NO):
    sqnorms1 = _get_dimensionwise_marg_var(var1, marginal)
    sqnorms1 = np.mean(sqnorms1, axis=axis, keepdims=True)
    if same_input:
      sqnorms2 = sqnorms1
    else:
      sqnorms2 = _get_dimensionwise_marg_var(var2, marginal)
      sqnorms2 = np.mean(sqnorms2, axis=axis, keepdims=True)

    prod12 = _outer_prod(sqnorms1, sqnorms2, 0, op.mul)

    if marginal == M.OVER_POINTS:
      prod11 = _outer_prod(sqnorms1, sqnorms1, 1, op.mul)
      prod22 = (_outer_prod(sqnorms2, sqnorms2, 1, op.mul)
                if not same_input else prod11)
    else:
      prod11 = _outer_prod(sqnorms1, sqnorms1, 0, op.mul)
      prod22 = (_outer_prod(sqnorms2, sqnorms2, 0, op.mul)
                if not same_input else prod11)

  else:
    raise NotImplementedError(
        "Only implemented for `OVER_ALL`, `OVER_PIXELS`, `OVER_POINTS` "
        "and `NO`; supplied {}".format(marginal))

  return prod11, prod12, prod22


def _get_ab_relu_kernel(ker_mat, prod, a, b, do_backprop, ntk=None):
  cosines = ker_mat / _safe_sqrt(prod)
  angles = _arccos(cosines, do_backprop)

  dot_sigma = (a**2 + b**2 - (a - b)**2 * angles / np.pi) / 2
  ker_mat = ((a - b)**2 * _sqrt(prod - ker_mat**2, do_backprop) / (2 * np.pi) +
             dot_sigma * ker_mat)

  if ntk is not None:
    ntk *= dot_sigma

  return ker_mat, ntk


def _transform_kernels_ab_relu(kernels, a, b, do_backprop, do_stabilize):
  """Compute new kernels after an `ABRelu` layer.

  See https://arxiv.org/pdf/1711.09090.pdf for the leaky ReLU derivation.
  """
  var1, nngp, var2, ntk, marginal = \
      kernels.var1, kernels.nngp, kernels.var2, kernels.ntk, kernels.marginal

  if do_stabilize:
    factor = np.max([np.max(np.abs(nngp)), 1e-12])
    nngp /= factor
    var1 /= factor
    if var2 is not None:
      var2 /= factor

  prod11, prod12, prod22 = _get_normalising_prod(var1, var2, marginal)
  nngp, ntk = _get_ab_relu_kernel(nngp, prod12, a, b, do_backprop, ntk=ntk)
  if do_stabilize:
    nngp *= factor

  if marginal in (M.OVER_ALL, M.OVER_PIXELS):
    var1 *= (a**2 + b**2) / 2
    if var2 is not None:
      var2 *= (a**2 + b**2) / 2
  elif marginal in (M.OVER_POINTS, M.NO):
    var1, _ = _get_ab_relu_kernel(var1, prod11, a, b, do_backprop)
    if var2 is not None:
      var2, _ = _get_ab_relu_kernel(var2, prod22, a, b, do_backprop)
  else:
    raise NotImplementedError(
        "Only implemented for `OVER_ALL`, `OVER_PIXELS`, `OVER_POINTS` "
        "and `NO`; supplied {}".format(marginal))

  if do_stabilize:
    var1 *= factor
    if var2 is not None:
      var2 *= factor

  return kernels._replace(
      var1=var1, nngp=nngp, var2=var2, ntk=ntk,
      is_gaussian=(a == b), marginal=marginal)


def _get_erf_kernel(ker_mat, prod, do_backprop, ntk=None):
  dot_sigma = 4 / (np.pi * np.sqrt(prod - 4 * ker_mat**2))
  ker_mat = _arcsin(2 * ker_mat / _safe_sqrt(prod), do_backprop) * 2 / np.pi

  if ntk is not None:
    ntk *= dot_sigma

  return ker_mat, ntk


def _transform_kernels_erf(kernels, do_backprop):
  """Compute new kernels after an `Erf` layer."""
  var1, nngp, var2, ntk, marginal = \
      kernels.var1, kernels.nngp, kernels.var2, kernels.ntk, kernels.marginal

  _var1_denom = 1 + 2 * var1
  _var2_denom = None if var2 is None else 1 + 2 * var2

  prod11, prod12, prod22 = _get_normalising_prod(
      _var1_denom, _var2_denom, marginal)
  nngp, ntk = _get_erf_kernel(nngp, prod12, do_backprop, ntk=ntk)

  if marginal in (M.OVER_ALL, M.OVER_PIXELS):
    var1 = np.arcsin(2 * var1 / _var1_denom) * 2 / np.pi
    if var2 is not None:
      var2 = np.arcsin(2 * var2 / _var2_denom) * 2 / np.pi
  elif marginal in [M.OVER_POINTS, M.NO]:
    var1, _ = _get_erf_kernel(var1, prod11, do_backprop)
    if var2 is not None:
      var2, _ = _get_erf_kernel(var2, prod22, do_backprop)
  else:
    raise NotImplementedError(
        "Only implemented for `OVER_ALL`, `OVER_PIXELS`, `OVER_POINTS` "
        "and `NO`; supplied {}".format(marginal))

  return kernels._replace(
      var1=var1, nngp=nngp, var2=var2, ntk=ntk,
      is_gaussian=False, marginal=marginal)


def _transform_kernels(kernels, fn, **fn_kwargs):
  """Apply transformation to kernels.

  Args:
    kernels: a `Kernel` object.
    fn: nonlinearity function, can only be Relu, Erf or Identity.
  Returns:
    The transformed kernel.
  """
  is_gaussian = kernels.is_gaussian
  if not is_gaussian:
    raise ValueError('An affine layer (i.e. dense or convolution) '
                     'has to be applied before a nonlinearity layer.')
  if fn is _ab_relu:
    return _transform_kernels_ab_relu(kernels, **fn_kwargs)
  if fn is _erf:
    return _transform_kernels_erf(kernels, **fn_kwargs)
  # TODO: Monte Carlo approximation to the integral (suggested by schsam.)
  raise NotImplementedError('Analaytic kernel for activiation {} is not '
                            'implmented'.format(fn))


def _affine(nngp, W_std, b_std):
  """Get [co]variances of affine outputs if inputs have [co]variances `nngp`.

  The output is assumed to be `xW + b`, where `x` is the input, `W` is a matrix
    of i.i.d. Gaussian weights with std `W_std`, `b` is a vector of i.i.d.
    Gaussian biases with std `b_std`.

  Args:
    nngp: a 2D or 1D `np.ndarray` containing either
      a) sample-sample covariances of shape `[batch_size_1, batch_size_2]` or
      b) sample variances of shape `[batch_size]`.
    W_std: a float, standard deviation of a fully-connected layer weights.
    b_std: a float, standard deviation of a fully-connected layer biases.

  Returns:
    a 2D or 1D `np.ndarray` containing sample[-sample] [co]variances of FC
      outputs. Has the same shape as `nngp`.
  """
  if nngp is None:
    return nngp

  return  W_std**2 * nngp + b_std**2


@_layer
def Dense(out_dim,
          W_std=1.,
          b_std=0.,
          W_init=_randn(1.0),
          b_init=_randn(1.0),
          parameterization='ntk',
          spec=None):
  r"""Layer constructor function for a dense (fully-connected) layer.

  Based on `jax.experimental.stax.Dense`. Has a similar API, apart from:

  `W_init` and `b_init` only change the behavior of the finite width network,
    and are not used by `kernel_fn`. In most cases, `W_std` and `b_std` should
    be used instead.

  Args:
    parameterization: Either 'ntk' or 'standard'.
      Under ntk parameterization (https://arxiv.org/abs/1806.07572, page 3),
        weights and biases are initialized as W_ij ~ N(0,1), b_i ~ N(0,1), and
        the finite width layer equation is z_i = W_std / sqrt([width]) sum_j
        W_ij x_j + b_std b_i Under standard parameterization
        (https://arxiv.org/abs/2001.07301), weights and biases are initialized
        as W_ij ~ N(0,W_std^2/[width]), b_i ~ N(0,b_std^2), and
        the finite width layer equation is z_i = \sum_j W_ij x_j + b_i.
    spec: an optional `string`, specifying the dimension order of the input,
      e.g. `NC` or `CN`. `NC` is more efficient.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  # TODO: after experimentation, evaluate whether to change default
  # parameterization from "ntk" to "standard"

  parameterization = parameterization.lower()

  def ntk_init_fn(rng, input_shape):
    _, channel_axis, _ = _parse_axes(spec, input_shape)
    output_shape = (input_shape[:channel_axis] + (out_dim,)
                    + input_shape[channel_axis + 1:])
    k1, k2 = random.split(rng)
    W = W_init(k1, (input_shape[channel_axis], out_dim))
    b = b_init(k2, (out_dim,))
    return output_shape, (W, b)

  def standard_init_fn(rng, input_shape):
    _, channel_axis, _ = _parse_axes(spec, input_shape)
    output_shape, (W, b) = ntk_init_fn(rng, input_shape)
    return output_shape, (W * W_std / np.sqrt(input_shape[channel_axis]),
                          b * b_std)

  if parameterization == 'ntk':
    init_fn = ntk_init_fn
  elif parameterization == 'standard':
    init_fn = standard_init_fn
  else:
    raise ValueError('Parameterization not supported: %s' % parameterization)

  def apply_fn(params, inputs, mask_constant=None, **kwargs):
    inputs, mask = _get_masked_inputs_and_mask(inputs, mask_constant)
    _, channel_axis, _ = _parse_axes(spec, inputs)

    # Collapse channel dimension the mask, since an FC layer is applied at each
    # non-channel location.
    if mask is not None:
      mask = np.all(mask, axis=channel_axis, keepdims=True)

    W, b = params

    prod = np.moveaxis(np.tensordot(W, inputs, (0, channel_axis)),
                       0, channel_axis)

    if parameterization == 'ntk':
      norm = W_std / np.sqrt(inputs.shape[channel_axis])
      outputs = norm * prod + b_std * b

    elif parameterization == 'standard':
      outputs = prod  + b

    # TODO: for convenience with the empirical kernel evaluation, assuming no
    #  mask is needed after an FC layer. This may be false when an FC layer is
    #  applied to a structured input. Please use a 1x1[x1...] convolution as an
    #  alternative.
    return outputs  # _drop_mask(outputs, mask)

  def kernel_fn(kernels):
    """Compute the transformed kernels after a dense layer."""
    var1, nngp, var2, ntk = \
        kernels.var1, kernels.nngp, kernels.var2, kernels.ntk

    def fc(x):
      return _affine(x, W_std, b_std)

    if parameterization == 'ntk':
      var1, nngp, var2 = map(fc, (var1, nngp, var2))
      if ntk is not None:
        ntk = nngp + W_std**2 * ntk
    elif parameterization == 'standard':
      input_width = kernels.shape1[1]
      if ntk is not None:
        ntk = input_width * nngp + 1. + W_std**2 * ntk
      var1, nngp, var2 = map(fc, (var1, nngp, var2))

    return kernels._replace(
        var1=var1, nngp=nngp, var2=var2, ntk=ntk, is_gaussian=True,
        is_input=False)

  setattr(kernel_fn, _INPUT_REQ, {'marginal': M.OVER_ALL,
                                  'cross': M.OVER_ALL,
                                  'spec': spec})
  return init_fn, apply_fn, kernel_fn


@_layer
def Identity():
  """Layer construction function for an identity layer.

  Based on `jax.experimental.stax.Identity`.
  """
  init_fn, apply_fn = ostax.Identity
  kernel_fn = lambda kernels: kernels
  return init_fn, apply_fn, kernel_fn


@_layer
def FanOut(num):
  """Layer construction function for a fan-out layer.

  Based on `jax.experimental.stax.FanOut`.
  """
  init_fn, apply_fn = ostax.FanOut(num)
  kernel_fn = lambda kernels: [kernels] * num
  return init_fn, apply_fn, kernel_fn


@_layer
def FanInSum():
  """Layer construction function for a fan-in sum layer.

  Based on `jax.experimental.stax.FanInSum`.
  """
  init_fn, apply_fn_old = ostax.FanInSum

  def apply_fn(params, inputs, mask_constant=None, **kwargs):
    inputs, masks = zip(*(_get_masked_inputs_and_mask(i, mask_constant)
                          for i in inputs))
    inputs_sum = apply_fn_old(params, inputs, **kwargs)
    masks_sum = _add_masks(masks)
    return _drop_mask(inputs_sum, masks_sum)

  kernel_fn = lambda kernels: _fan_in_kernel_fn(kernels, axis=None, spec=None)
  return init_fn, apply_fn, kernel_fn


@_layer
def FanInConcat(axis=-1, spec=None):
  """Layer construction function for a fan-in concatenation layer.

  Based on `jax.experimental.stax.FanInConcat`.
  """
  init_fn, apply_fn_old = ostax.FanInConcat(axis)

  def apply_fn(params, inputs, mask_constant=None, **kwargs):
    inputs, masks = zip(*(_get_masked_inputs_and_mask(i, mask_constant)
                          for i in inputs))
    outputs = apply_fn_old(params, inputs, **kwargs)
    mask = _concat_masks(masks, inputs, axis % inputs[0].ndim)
    return _drop_mask(outputs, mask)

  kernel_fn = lambda kernels: _fan_in_kernel_fn(kernels, axis=axis, spec=spec)
  return init_fn, apply_fn, kernel_fn


def _fan_in_kernel_fn(kernels, axis, spec):
  marginal, cross = kernels[0].marginal, kernels[0].cross
  shape1, shape2 = kernels[0].shape1, kernels[0].shape2

  # Check marginalization
  if not all(k.marginal == marginal and
             k.cross == cross
             for k in kernels):
    raise NotImplementedError('`FanIn` layers are only implemented for the '
                              'case if all input layers output the same type'
                              'of covariance matrices, i.e. having all '
                              'matching `marginal` and `cross` attributes.')

  # If kernels have different height/width order, transpose some of them.
  n_kernels = len(kernels)
  n_reversed = sum(ker.is_reversed for ker in kernels)

  if n_reversed > n_kernels / 2:
    is_reversed = True
    for i in range(n_kernels):
      if not kernels[i].is_reversed:
        kernels[i] = kernels[i].reverse()

  else:
    is_reversed = False
    for i in range(n_kernels):
      if kernels[i].is_reversed:
        kernels[i] = kernels[i].reverse()

  axis = None if axis is None else range(len(shape1))[axis]
  batch_axis, channel_axis, _ = _parse_axes(spec, shape1)

  # Check shapes.
  if axis is None:
    if not all([k.shape1 == shape1 and k.shape2 == shape2 for k in kernels]):
      raise ValueError('All shapes should be equal in FanInSum.')

  else:
    new_shape1 = shape1[:axis] + shape1[axis + 1:]
    new_shape2 = shape2[:axis] + shape2[axis + 1:]
    for k in kernels:
      k_shape1 = k.shape1[:axis] + k.shape1[axis + 1:]
      k_shape2 = k.shape2[:axis] + k.shape2[axis + 1:]
      if k_shape1 != new_shape1 or k_shape2 != new_shape2:
        raise ValueError('All non-axis shapes should be equal in FanInConcat.')

  # Check if inputs are independent Gaussians.
  if axis is None or axis != channel_axis:
    is_gaussian = all(k.is_gaussian for k in kernels)
    if not is_gaussian and n_kernels > 1:
      raise NotImplementedError('`FanInSum` or `FanInConcat` layer along the '
                                'non-channel axis is only implemented for the '
                                'case if all input layers are guaranteed to be '
                                'mean-zero Gaussian, i.e. having all '
                                '`is_gaussian` set to `True`.')
  else:
    # TODO: allow to apply nonlinearity after channelwise concatenation.
    is_gaussian = False

  # Warnings.
  warnings.warn('`FanIn` layers assume independent inputs which is not verified '
                'in the code. Please make sure to have at least one Dense or '
                'CNN layer in each branch.')
  if axis == batch_axis:
    warnings.warn('Concatenation along the batch axis (%d) gives inconsistent'
                  ' covariances when batching - proceed with caution.' % axis)

  # Concatenate masks.
  mask1 = _fan_in_masks([k.mask1 for k in kernels], None, axis)
  mask2 = _fan_in_masks([k.mask2 for k in kernels], None, axis)

  spatial_axes = tuple(i for i in range(len(shape1))
                  if i not in (channel_axis, batch_axis))
  # Change spatial axis according to `is_reversed`.
  if axis in spatial_axes and is_reversed:
    axis = spatial_axes[::-1][spatial_axes.index(axis)]

  # Map activation tensor axis to the covariance tensor axis.
  axis = {
      **{
          None: None,
          batch_axis: 0,
          channel_axis: -1,
      },
      **{
          spatial_axis: idx + 1 for idx, spatial_axis in enumerate(spatial_axes)
      }
  }[axis]

  widths = [k.shape1[channel_axis] for k in kernels]
  var1 = _concat_kernels([k.var1 for k in kernels], axis, marginal, 1, widths)
  var2 = _concat_kernels([k.var2 for k in kernels], axis, marginal, 1, widths)
  nngp = _concat_kernels([k.nngp for k in kernels], axis, cross, 2, widths)
  ntk = _concat_kernels([k.ntk for k in kernels], axis, cross, 2, widths)
  kers = (var1, nngp, var2, ntk)

  return Kernel(*(
      kers + (is_gaussian, is_reversed, marginal, cross, None, None,
              kernels[0].x1_is_x2, kernels[0].is_input, mask1, mask2)))


def _concat_kernels(mats, axis, marginalisation, batch_ndim, widths):
  """Compute the covariance of concatenated activations with given covariances.

  Args:
    mats: a list of `np.ndarrray` covariance tensors of the same shape.
    axis: an `int` along which the covariances (not activations) are
      concatenated. `None` corresponds to sum, `-1` to averaging.
    marginalisation: a single `Kernel.Marginalisation` of all covariance
      matrices.
    widths: list of integer channel widths of the finite model inputs.

  Returns:
    A new `np.ndarray` representing covariance between concatenated activations.
  """
  if mats[0] is None:
    return None

  n_mats = len(mats)
  mat_ndim = mats[0].ndim

  # Sum if `axis == None` i.e. called from `FanInSum`.
  if axis is None:
    mat = sum(mats)

  # Averaging if concatenating along features or marginalized dimension.
  elif (axis == -1 or
        (marginalisation == M.OVER_ALL and axis != 0)):
    if all(w == widths[0] for w in widths):
      widths = [1] * len(widths)
    mat = sum(mats[i] * widths[i] for i in range(n_mats)) / sum(widths)

  # Simple concatenation along the axis if the axis is not duplicated.
  elif ((axis != 0 and marginalisation == M.OVER_PIXELS) or
        (axis == 0 and batch_ndim == 1)):
    concat_axis = axis + batch_ndim - 1
    mat = np.concatenate(mats, concat_axis)

  # 2D concatenation with insertion of 0-blocks if the axis is present twice.
  elif axis == 0 or marginalisation in (M.NO, M.OVER_POINTS):
    rows = []
    pad_axis = max(0, 2 * axis + batch_ndim - 2)
    for i, mat in enumerate(mats):
      pads = [(0, 0)] * mat_ndim
      pads[pad_axis] = (
          sum(mats[j].shape[pad_axis] for j in range(i)),
          sum(mats[j].shape[pad_axis] for j in range(i + 1, n_mats))
      )
      rows.append(np.pad(mat, pads))
    mat = np.concatenate(rows, pad_axis + 1)

  else:
    raise NotImplementedError(
        'Asked to concatenate along axis %d given '
        'covariance tensors of marginalization %s, which is not implemented. '
        'Please file a bug at '
        'https://github.com/google/neural-tangents/issues/new'
        % (axis, M(marginalisation)))

  return mat


@_layer
def serial(*layers):
  """Combinator for composing layers in serial.

  Based on `jax.experimental.stax.serial`.

  Args:
    *layers: a sequence of layers, each an (init_fn, apply_fn, kernel_fn) tuple.

  Returns:
    A new layer, meaning an `(init_fn, apply_fn, kernel_fn)` tuple, representing
      the serial composition of the given sequence of layers.
  """
  init_fns, apply_fns, kernel_fns = zip(*layers)
  init_fn, apply_fn = ostax.serial(*zip(init_fns, apply_fns))

  def kernel_fn(kernels):
    for f in kernel_fns:
      kernels = f(kernels)
    return kernels

  _set_input_req_attr(kernel_fn, kernel_fns)
  return init_fn, apply_fn, kernel_fn


@_layer
def parallel(*layers):
  """Combinator for composing layers in parallel.

  The layer resulting from this combinator is often used with the `FanOut` and
    `FanInSum` layers. Based on `jax.experimental.stax.parallel`.

  Args:
    *layers: a sequence of layers, each an `(init_fn, apply_fn, kernel_fn)`
      triple.

  Returns:
    A new layer, meaning an `(init_fn, apply_fn, kernel_fn)` triples,
      representing the parallel composition of the given sequence of layers. In
      particular, the returned layer takes a sequence of inputs and returns a
      sequence of outputs with the same length as the argument `layers`.
  """
  init_fns, apply_fns, kernel_fns = zip(*layers)
  init_fn_stax, apply_fn = ostax.parallel(*zip(init_fns, apply_fns))

  def init_fn(rng, input_shape):
    return list(init_fn_stax(rng, input_shape))

  def kernel_fn(kernels):
    return [f(ker) for ker, f in zip(kernels, kernel_fns)]

  _set_input_req_attr(kernel_fn, kernel_fns)
  return init_fn, apply_fn, kernel_fn


def _same_pad_for_filter_shape(x, filter_shape, strides, axes, mode):
  """Pad an array to imitate `SAME` padding with `VALID`.

  See `Returns` section for details. This method is usually needed to implement
    `CIRCULAR` padding using `VALID` padding.

  Args:
    x: `np.ndarray` to pad, e.g. a 4D `NHWC` image.
    filter_shape: tuple of positive integers, the convolutional filters spatial
      shape (e.g. `(3, 3)` for a 2D convolution).
    strides: tuple of positive integers, the convolutional spatial strides, e.g.
      e.g. `(1, 1)` for a 2D convolution.
    axes: tuple of non-negative integers, the spatial axes to apply
      convolution over (e.g. `(1, 2)` for an `NHWC` image).
    mode: a string, padding mode, for all options see
      https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html.

  Returns:
    A `np.ndarray` of the same dimensionality as `x` padded to a potentially
      larger shape such that a `VALID` convolution with `filter_shape` applied
      to `x` over `axes` outputs an array of the same shape as `x`.
  """
  if not utils.is_array(x):
    return x

  axes_shape = tuple(np.size(x, axis) for axis in axes)
  axes_pads = lax.padtype_to_pads(axes_shape, filter_shape, strides,
                                  Padding.SAME.name)

  pads = [(0, 0),] * x.ndim
  for i, axis in enumerate(axes):
    pads[axis] = axes_pads[i]

  x = np.pad(x, pads, mode)
  return x


def _pad_one_side(x, pads, axes, mode):
  """Pad an array on one side. See `Returns` section for details.

  Args:
    x: `np.ndarray` to pad, e.g. a 4D `NHWC` image.
    pads: tuple of integers, the convolutional filters spatial
      shape (e.g. `(3, 3)` for a 2D convolution).
    axes: tuple of non-negative integers, the axes to apply padding of sizes
      `pads` to.
    mode: a string, padding mode, for all options see
      https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html.

  Returns:
    A `np.ndarray` of the same dimensionality as `x` padded to a potentially
      larger shape with `pads` applied at `axes`, where positive values in
      `pads` are applied on the left (start), and negative on the right (end).
  """
  axis_pads = [(p, 0) if p >= 0 else (0, -p) for p in pads]
  pads = [(0, 0),] * x.ndim
  for i in range(len(axes)):
    pads[axes[i]] = axis_pads[i]
  x = np.pad(x, pads, mode)
  return x


def _conv_kernel(mat, filter_shape, strides, padding, batch_ndim):
  """Compute covariance of the CNN outputs given inputs with covariance `nngp`.

  Uses 2D convolution and works on any hardware platform.

  Args:
    mat: an (N+batch_ndim)D `np.ndarray` containing sample-(sample-)pixel-pixel
      covariances. Has shape
      `[batch_size_1, (batch_size_2,)
        height, height, width, width, depth, depth, ...]`.
    filter_shape: tuple of positive integers, the convolutional filters spatial
      shape (e.g. `(3, 3)` for a 2D convolution).
    strides: tuple of positive integers, the CNN strides (e.g. `(1, 1)` for a
      2D convolution).
    padding: a `Padding` enum, e.g. `Padding.CIRCULAR`.
    batch_ndim: integer, number of batch dimensions, 1 or 2.

  Returns:
    an (N+batch_ndim)D `np.ndarray` containing sample-(sample-)pixel-pixel
    covariances of CNN outputs. Has shape
    `[batch_size_1, (batch_size_2,) new_width, new_width,
      new_height, new_height, new_depth, new_depth, ...]`.
  """
  if not utils.is_array(mat):
    return mat

  if padding == Padding.CIRCULAR:
    pixel_axes = tuple(range(batch_ndim, mat.ndim))
    mat = _same_pad_for_filter_shape(
        mat,
        _double_tuple(filter_shape),
        _double_tuple(strides),
        pixel_axes,
        'wrap'
    )
    padding = Padding.VALID

  for i in range(mat.ndim - 1, batch_ndim, -2):
    spatial_i = (i - batch_ndim) // 2
    filter_i = filter_shape[spatial_i]
    stride_i = strides[spatial_i]

    ker = np.diag(np.full((filter_i,), 1. / filter_i, mat.dtype))
    for c in _CONV_QAB_DIMENSION_NUMBERS[1]:
      if c in ('I', 'O'):
        ker = np.expand_dims(ker, _CONV_QAB_DIMENSION_NUMBERS[1].index(c))

    size_i = mat.shape[i]
    mat = np.moveaxis(mat, (i - 1, i), (-2, -1))
    mat_preshape = mat.shape[:-2]
    mat = np.expand_dims(mat.reshape((-1, size_i, size_i)),
                         _CONV_QAB_DIMENSION_NUMBERS[0].index('C'))
    mat = lax.conv_general_dilated(
        lhs=mat,
        rhs=ker,
        window_strides=(stride_i, stride_i),
        padding=padding.name,
        dimension_numbers=_CONV_QAB_DIMENSION_NUMBERS)
    mat = np.squeeze(mat,
                     _CONV_QAB_DIMENSION_NUMBERS[2].index('C'))
    mat = mat.reshape(mat_preshape + mat.shape[-2:])

  return mat


def _conv_kernel_over_pixels(mat, filter_shape, strides, padding, batch_ndim):
  """Compute covariance of the CNN outputs given inputs with covariance `nngp`.

  Uses 2D convolution and works on any platform, but only works with
    sample-sample-(same pixel) covariances.

  Args:
    mat: an (N+batch_ndim) `np.ndarray` containing sample-sample-(same pixel)
      covariances. Has 2 batch and `N` spatial dimensions with the shape of
      `[batch_size_1, (batch_size_2,) height, width, depth, ...]`.
    filter_shape: tuple of positive integers, the convolutional filters spatial
      shape (e.g. `(3, 3)` for a 2D convolution).
    strides: tuple of positive integers, the CNN strides (e.g. `(1, 1)` for a
      2D convolution).
    padding: a `Padding` enum, e.g. `Padding.CIRCULAR`.
    batch_ndim: integer, number of leading batch dimensions, 1 or 2.

  Returns:
    a (N+batch_ndim)D `np.ndarray` containing sample-sample-pixel-pixel
      covariances of CNN outputs. Has shape
      `[batch_size_1, (batch_size_2,) new_height, new_width, new_depth, ...]`.
  """
  if not utils.is_array(mat):
    return mat

  spatial_axes = tuple(range(mat.ndim)[batch_ndim:])

  if padding == Padding.CIRCULAR:
    mat = _same_pad_for_filter_shape(mat, filter_shape, strides,
                                     spatial_axes, 'wrap')
    padding = Padding.VALID

  ker = np.full((1, 1) + filter_shape, 1. / np.prod(filter_shape), mat.dtype)

  batch_shape, spatial_shape = mat.shape[:batch_ndim], mat.shape[batch_ndim:]
  mat = np.reshape(mat, (-1,) + spatial_shape)
  mat = np.expand_dims(mat, 1)
  mat = lax.conv_general_dilated(mat, ker, strides, padding.name)
  mat = mat.reshape(batch_shape + mat.shape[2:])
  return mat


@_layer
def GeneralConv(dimension_numbers,
                out_chan,
                filter_shape,
                strides=None,
                padding=Padding.VALID.name,
                W_std=1.0,
                W_init=_randn(1.0),
                b_std=0.0,
                b_init=_randn(1.0),
                parameterization='ntk'):
  """Layer construction function for a general convolution layer.

  Based on `jax.experimental.stax.GeneralConv`. Has a similar API apart from:

  Args:
    padding: in addition to `VALID` and `SAME' padding, supports `CIRCULAR`, not
      available in `jax.experimental.stax.GeneralConv`.
  """

  parameterization = parameterization.lower()

  if dimension_numbers is None:
    spatial_dims = 'HWDXYZEABMJUPLTKQRS'[:len(filter_shape)]  # TODO allow more.
    lhs_spec = 'N' + spatial_dims + 'C'
    dimension_numbers = (lhs_spec, spatial_dims + 'IO', lhs_spec)

  lhs_spec = dimension_numbers[0]

  one = (1,) * len(filter_shape)
  strides = strides or one

  padding = Padding(padding)
  init_padding = padding
  if padding == Padding.CIRCULAR:
    init_padding = Padding.SAME

  def input_total_dim(input_shape):
    return input_shape[lhs_spec.index('C')] * np.prod(filter_shape)

  ntk_init_fn, _ = ostax.GeneralConv(dimension_numbers, out_chan, filter_shape,
                                     strides, init_padding.name, W_init, b_init)

  def standard_init_fn(rng, input_shape):
    output_shape, (W, b) = ntk_init_fn(rng, input_shape)
    norm = W_std / np.sqrt(input_total_dim(input_shape))
    return output_shape, (W * norm, b * b_std)

  if parameterization == 'ntk':
    init_fn = ntk_init_fn
  elif parameterization == 'standard':
    init_fn = standard_init_fn
  else:
    raise ValueError('Parameterization not supported: %s' % parameterization)

  def apply_fn(params, inputs, mask_constant=None, **kwargs):
    W, b = params

    inputs, mask = _get_masked_inputs_and_mask(inputs, mask_constant)
    mask = _conv_mask(mask, filter_shape, strides, padding, dimension_numbers)

    if parameterization == 'ntk':
      norm = W_std / np.sqrt(input_total_dim(inputs.shape))
      b_rescale = b_std
    elif parameterization == 'standard':
      norm = 1.
      b_rescale = 1.

    apply_padding = padding
    if padding == Padding.CIRCULAR:
      if mask is not None:
        raise NotImplementedError('`CIRCULAR` padding is not implemented for '
                                  'masked inputs')

      apply_padding = Padding.VALID
      spatial_axes = tuple(dimension_numbers[0].index(c)
                           for c in dimension_numbers[1]
                           if c not in ('I', 'O'))
      inputs = _same_pad_for_filter_shape(inputs, filter_shape, strides,
                                          spatial_axes, 'wrap')

    outputs = norm * lax.conv_general_dilated(
        inputs,
        W,
        strides,
        apply_padding.name,
        dimension_numbers=dimension_numbers) + b_rescale * b

    return _drop_mask(outputs, mask)

  def kernel_fn(kernels):
    """Compute the transformed kernels after a conv layer."""
    var1, nngp, var2, ntk, is_reversed, marginal, cross, mask1, mask2 = (
        kernels.var1, kernels.nngp, kernels.var2, kernels.ntk,
        kernels.is_reversed, kernels.marginal, kernels.cross, kernels.mask1,
        kernels.mask2
    )

    input_spec = tuple(c for c in dimension_numbers[0] if c not in ('N', 'C'))
    conv_spec = tuple(c for c in dimension_numbers[1] if c not in ('I', 'O'))
    input_to_filter_permutation = tuple(input_spec.index(c) for c in conv_spec)

    filter_shape_kernel = tuple(filter_shape[p] for p in
                                input_to_filter_permutation)
    strides_kernel = tuple(strides[p] for p in
                           input_to_filter_permutation)

    mask1 = _conv_mask(mask1, filter_shape_kernel, strides_kernel, padding,
                       dimension_numbers)
    mask2 = _conv_mask(mask2, filter_shape_kernel, strides_kernel, padding,
                       dimension_numbers)

    if cross == M.OVER_PIXELS:
      def conv_unscaled(x, batch_ndim):
        return _conv_kernel_over_pixels(x, filter_shape_kernel, strides_kernel,
                                        padding, batch_ndim)

    elif cross in [M.OVER_POINTS, M.NO]:
      if is_reversed:
        filter_shape_kernel = filter_shape_kernel[::-1]
        strides_kernel = strides_kernel[::-1]

      is_reversed = not is_reversed

      def conv_unscaled(x, batch_ndim):
        return _conv_kernel(x, filter_shape_kernel, strides_kernel, padding,
                            batch_ndim)

    else:
      raise NotImplementedError(
          "Only implemented for `OVER_PIXELS`, `OVER_POINTS` and `NO`;"
          " supplied {}".format(cross))

    def conv(x, batch_ndim):
      x = conv_unscaled(x, batch_ndim)
      return _affine(x, W_std, b_std)

    var1 = conv(var1, 1)
    var2 = conv(var2, 1)

    if parameterization == 'ntk':
      nngp = conv(nngp, 2)
      ntk = conv(ntk, 2) + nngp - b_std**2 if ntk is not None else ntk

    elif parameterization == 'standard':
      nngp_unscaled = conv_unscaled(nngp, 2)
      if ntk is not None:
        ntk = (
            input_total_dim(kernels.shape1) * nngp_unscaled + 1. +
            W_std**2 * conv_unscaled(ntk, 2))
      nngp = _affine(nngp_unscaled, W_std, b_std)

    var1, nngp, var2, ntk = _mask_kernels(var1, nngp, var2, ntk, cross,
                                          marginal, is_reversed, mask1, mask2,
                                          dimension_numbers[0])

    return kernels._replace(
        var1=var1, nngp=nngp, var2=var2, ntk=ntk, is_gaussian=True,
        is_reversed=is_reversed, marginal=marginal, cross=cross,
        is_input=False, mask1=mask1, mask2=mask2)

  setattr(kernel_fn, _INPUT_REQ, {'marginal': M.OVER_PIXELS,
                                  'cross': M.OVER_PIXELS,
                                  'spec': dimension_numbers[0]})
  return init_fn, apply_fn, kernel_fn


def Conv(out_chan,
         filter_shape,
         strides=None,
         padding=Padding.VALID.name,
         W_std=1.0,
         W_init=_randn(1.0),
         b_std=0.0,
         b_init=_randn(1.0),
         parameterization='ntk'):
  """Layer construction function for a convolution layer.

  Based on `jax.experimental.stax.Conv`. Has a similar API apart from:

  W_init and b_init only change the behavior of the finite width network, and
    are not used by kernel_fn. In most cases, W_std and b_std should be used
    instead

  Args:
    padding: in addition to `VALID` and `SAME' padding, supports `CIRCULAR`,
      not available in `jax.experimental.stax.GeneralConv`.
    parameterization: Either 'ntk' or 'standard'. These parameterizations are
      the direct analogues for convolution of the corresponding
      parameterizations for Dense layers.
  """
  return GeneralConv(None,
                     out_chan,
                     filter_shape,
                     strides,
                     padding,
                     W_std,
                     W_init,
                     b_std,
                     b_init,
                     parameterization)


def _pool_kernel(mat,
                 pool_type,
                 window_shape,
                 strides,
                 padding,
                 normalize_edges,
                 batch_ndim):
  """Get covariances of pooling outputs given inputs covariances `mat`.

  Args:
    mat: an (N+batch_ndim)D `np.ndarray` containing sample-(sample-)pixel-pixel
      covariances. Has shape
      `[batch_size_1, (batch_size_2,) height, height,
        width, width, depth, depth, ...]`.
    pool_type: a `Pooling` enum, e.g. `Pooling.AVG`.
    window_shape: tuple of two positive integers, the pooling spatial shape
      (e.g. `(3, 3)`).
    strides: tuple of two positive integers, the pooling strides, e.g. `(1, 1)`.
    padding: a `Padding` enum, e.g. `Padding.CIRCULAR`.
    normalize_edges: `True` to normalize output by the effective receptive
      field, `False` to normalize by the window size. Only has effect at the
      edges when `SAME` padding is used. Set to `True` to retain correspondence
      to `ostax.AvgPool`.
    batch_ndim: integer, number of leading batch dimensions, 1 or 2.

  Returns:
    an (N+batch_ndim)D `np.ndarray` containing sample-(sample-)pixel-pixel
    covariances of the average pooling outputs. Has shape
    `[batch_size_1, (batch_size_2,) new_height, new_height,
      new_width, new_width, new_depth, new_depth, ...]`.
  """
  if not utils.is_array(mat):
    return mat

  if padding == Padding.CIRCULAR:
    pixel_axes = tuple(range(batch_ndim, mat.ndim))
    mat = _same_pad_for_filter_shape(mat, _double_tuple(window_shape),
                                     _double_tuple(strides), pixel_axes, 'wrap')
    padding = Padding.VALID

  window_shape = (1,) * batch_ndim + _double_tuple(window_shape)
  strides = (1,) * batch_ndim + _double_tuple(strides)

  nngp_out = lax.reduce_window(mat, 0., lax.add, window_shape, strides,
                               padding.name)

  if pool_type == Pooling.AVG:
    if padding == Padding.SAME and normalize_edges:
      # `SAME` padding in `jax.experimental.stax.AvgPool` normalizes by actual
      # window size, which is smaller at the edges.
      one = np.ones(mat.shape, mat.dtype)
      window_sizes = lax.reduce_window(one, 0., lax.add, window_shape, strides,
                                       padding.name)
      nngp_out /= window_sizes
    else:
      nngp_out /= np.prod(window_shape)

  return nngp_out

@_layer
def _Pool(pool_type,
          window_shape,
          strides,
          padding,
          spec,
          normalize_edges):
  """Layer construction function for a 2D pooling layer.

  Based on `jax.experimental.stax.AvgPool` and `jax.experimental.stax.SumPool`.
  Has a similar API apart from:

  Args:
    pool_type: specifies whether average or sum pooling should be performed.
      (`Pooling.AVG` or `Pooling.SUM`)
    padding: in addition to `VALID` and `SAME` padding, supports `CIRCULAR`, not
      available in `jax.experimental.stax.GeneralConv`.
    spec: an optional `string`, specifying the dimension order of the input,
      e.g. `NCHW` or `NHWC`.
    normalize_edges: `True` to normalize output by the effective receptive
      field, `False` to normalize by the window size. Only has effect at the
      edges when `SAME` padding is used. Set to `True` to retain correspondence
      to `ostax.AvgPool`.
  """

  strides = strides or (1,) * len(window_shape)
  padding = Padding(padding)

  if pool_type == Pooling.AVG:
    pool_fn = ostax.AvgPool
  elif pool_type == Pooling.SUM:
    pool_fn = ostax.SumPool
  else:
    raise ValueError('Invalid pooling type {}'.format(pool_type))

  if padding == Padding.CIRCULAR:
    init_fn, _ = pool_fn(window_shape, strides, Padding.SAME.name, spec)
    _, apply_fn_0 = pool_fn(window_shape, strides, Padding.VALID.name, spec)

    def apply_fn(params, inputs, **kwargs):
      _, _, spatial_axes = _parse_axes(spec, inputs)
      inputs = _same_pad_for_filter_shape(inputs, window_shape, strides,
                                          spatial_axes, 'wrap')
      res = apply_fn_0(params, inputs, **kwargs)
      return res

  elif normalize_edges or pool_type == Pooling.SUM:
    init_fn, apply_fn = pool_fn(window_shape, strides, padding.name, spec)

  else:
    def rescaler(dims, strides, padding):
      del dims, strides, padding  # Unused.
      return lambda outputs, inputs, spec: outputs / np.prod(window_shape)

    pool_fn = ostax._pooling_layer(lax.add, 0., rescaler)
    init_fn, apply_fn = pool_fn(window_shape, strides, padding.name, spec)

  def apply_fn_mask(params, inputs, mask_constant=None, **kwargs):
    inputs, mask = _get_masked_inputs_and_mask(inputs, mask_constant)

    if padding == Padding.CIRCULAR and mask is not None:
      raise NotImplementedError('`CIRCULAR` padding is not implemented for '
                                  'masked inputs')

    outputs = apply_fn(params, inputs, **kwargs)
    mask = _pool_mask(mask, window_shape, strides, padding, spec)
    return _drop_mask(outputs, mask)

  def kernel_fn(kernels):
    """Kernel transformation."""
    (var1, nngp, var2, ntk, is_gaussian, is_reversed, marginal, cross, mask1,
     mask2) = (
        kernels.var1, kernels.nngp, kernels.var2, kernels.ntk,
        kernels.is_gaussian, kernels.is_reversed, kernels.marginal,
        kernels.cross, kernels.mask1, kernels.mask2)

    _, channel_axis, _ = _parse_axes(spec, kernels.shape1)
    if (mask1 is not None and
        mask1.shape[channel_axis] > 1 and
        pool_type == Pooling.AVG):
      # TODO: implement.
      warnings.warn('Average pooling over different per-channel '
                    'masks is not implemented.')

    mask1 = _pool_mask(mask1, window_shape, strides, padding, spec)
    mask2 = _pool_mask(mask2, window_shape, strides, padding, spec)

    window_shape_kernel = window_shape[::(-1 if is_reversed else 1)]
    strides_kernel = strides[::(-1 if is_reversed else 1)]

    nngp = _pool_kernel(nngp, pool_type, window_shape_kernel, strides_kernel,
                        padding, normalize_edges, 2)
    ntk = _pool_kernel(ntk, pool_type, window_shape_kernel, strides_kernel,
                       padding, normalize_edges, 2)
    var1 = _pool_kernel(var1, pool_type, window_shape_kernel, strides_kernel,
                        padding, normalize_edges, 1)
    var2 = _pool_kernel(var2, pool_type, window_shape_kernel, strides_kernel,
                        padding, normalize_edges, 1)

    var1, nngp, var2, ntk = _mask_kernels(var1, nngp, var2, ntk, cross,
                                          marginal, is_reversed, mask1, mask2,
                                          spec)

    return kernels._replace(
        var1=var1, nngp=nngp, var2=var2, ntk=ntk, is_gaussian=is_gaussian,
        is_reversed=is_reversed, marginal=marginal, cross=cross, mask1=mask1,
        mask2=mask2
    )

  setattr(kernel_fn, _INPUT_REQ, {'marginal': M.OVER_POINTS,
                                  'cross': M.NO,
                                  'spec': spec})
  return init_fn, apply_fn_mask, kernel_fn


def AvgPool(window_shape,
            strides=None,
            padding=Padding.VALID.name,
            spec=None,
            normalize_edges=True):
  """Layer construction function for a 2D average pooling layer.

  Based on `jax.experimental.stax.AvgPool`. Has a similar API apart from:

  Args:
    padding: in addition to `VALID` and `SAME` padding, supports `CIRCULAR`, not
      available in `jax.experimental.stax.AvgPool`.
    spec: an optional `string`, specifying the dimension order of the input,
      e.g. `NCHW` or `NHWC`.
    normalize_edges: `True` to normalize output by the effective receptive
      field, `False` to normalize by the window size. Only has effect at the
      edges when `SAME` padding is used. Set to `True` to retain correspondence
      to `ostax.AvgPool`.
  """
  return _Pool(Pooling.AVG, window_shape, strides, padding, spec,
               normalize_edges)


def SumPool(window_shape,
            strides=None,
            padding=Padding.VALID.name,
            spec=None):
  """Layer construction function for a 2D sum pooling layer.

  Based on `jax.experimental.stax.SumPool`. Has a similar API apart from:

  Args:
    padding: in addition to `VALID` and `SAME` padding, supports `CIRCULAR`, not
      available in `jax.experimental.stax.SumPool`.
    spec: an optional `string`, specifying the dimension order of the input,
      e.g. `NCHW` or `NHWC`.
  """
  return _Pool(Pooling.SUM, window_shape, strides, padding, spec, False)


def GlobalSumPool(spec=None):
  return _GlobalPool(Pooling.SUM, spec=spec)


def GlobalAvgPool(spec=None):
  return _GlobalPool(Pooling.AVG, spec=spec)


@_layer
def _GlobalPool(pool_type, spec):
  """Layer construction function for a global average pooling layer.

  Pools over and removes (`keepdims=False`) all spatial dimensions, making the
    batch dimension (`N`) leading.

  Args:
    pool_type: specifies whether average or sum pooling should be performed.
      (Pooling.AVG or Pooling.SUM)
    spec: an optional `string`, specifying the dimension order of
      the input, e.g. `NCHW` or `NHWC`.
  """

  if pool_type == Pooling.AVG:
    pool_fn = np.mean
  elif pool_type == Pooling.SUM:
    pool_fn = np.sum
  else:
    raise ValueError(f'Invalid pooling type {pool_type}')

  def init_fn(rng, input_shape):
    batch_axis, channel_axis, spatial_axes = _parse_axes(spec, input_shape)
    output_shape = tuple(input_shape[i] for i in sorted((batch_axis,
                                                         channel_axis)))
    return output_shape, ()

  def apply_fn(params, inputs, mask_constant=None, **kwargs):
    inputs, mask = _get_masked_inputs_and_mask(inputs, mask_constant)
    _, _, spatial_axes = _parse_axes(spec, inputs)

    if mask is None or pool_type == Pooling.SUM:
      outputs = pool_fn(inputs, axis=spatial_axes)
    elif pool_type == Pooling.AVG:
      total_size = _size_at(inputs, spatial_axes)
      size = total_size - np.count_nonzero(mask, axis=spatial_axes)
      outputs = np.sum(inputs, spatial_axes) / np.maximum(size, 1)
    else:
      raise NotImplementedError(
          f'Unpexpected {pool_type} and {mask}, please file a bug at '
          f'https://github.com/google/neural-tangents/issues/new.')

    # Collapse spatial dimensions of the mask, since they are all averaged into
    # a single per-channel entry.
    mask = np.all(mask, axis=spatial_axes) if mask is not None else None
    return _drop_mask(outputs, mask)

  def kernel_fn(kernels):
    (var1, nngp, var2, ntk, marginal, cross, mask1, mask2, is_reversed,
     is_gaussian, shape1) = (
        kernels.var1, kernels.nngp, kernels.var2, kernels.ntk,
        kernels.marginal, kernels.cross,
        kernels.mask1, kernels.mask2, kernels.is_reversed, kernels.is_gaussian,
        kernels.shape1
    )

    _, channel_axis, _ = _parse_axes(spec, shape1)
    if (mask1 is not None and
        mask1.shape[channel_axis] > 1 and
        pool_type == Pooling.AVG):
      # TODO: implement.
      warnings.warn('Average pooling over different per-channel '
                    'masks is not implemented.')

    def _pool(ker_mat, mask1, mask2, marginal, batch_ndim):
      if not utils.is_array(ker_mat):
        return ker_mat

      spatial_axes = tuple(range(ker_mat.ndim)[batch_ndim:])
      if mask1 is None or pool_type == Pooling.SUM:
        return pool_fn(ker_mat, axis=spatial_axes)
      elif pool_type == Pooling.AVG:
        total_size = _size_at(ker_mat, spatial_axes)
        mask_prod = _get_mask_prod(mask1, mask2, batch_ndim, spec, marginal,
                                   is_reversed)
        size = total_size - np.count_nonzero(mask_prod, spatial_axes)
        ker = np.sum(ker_mat, spatial_axes) / np.maximum(size, 1)
        return ker
      else:
        raise ValueError(f'Unrecognized pool_type {pool_type}.')

    nngp = _pool(nngp, mask1, mask2, cross, 2)
    ntk = _pool(ntk, mask1, mask2, cross, 2)
    var1 = _pool(var1, mask1, None, marginal, 1)
    var2 = _pool(var2, mask2, None, marginal, 1)

    _, _, spatial_axes = _parse_axes(spec, shape1)
    mask1 = np.all(mask1, spatial_axes) if mask1 is not None else None
    mask2 = np.all(mask2, spatial_axes) if mask2 is not None else None

    return kernels._replace(
        var1=var1, nngp=nngp, var2=var2, ntk=ntk,
        is_reversed=False, marginal=M.OVER_ALL, cross=M.OVER_ALL,
        mask1=mask1, mask2=mask2
    )

  setattr(kernel_fn, _INPUT_REQ, {'marginal': M.OVER_POINTS,
                                  'cross': M.NO,
                                  'spec': spec})
  return init_fn, apply_fn, kernel_fn


@_layer
def Flatten(spec=None):
  """Layer construction function for flattening all but the batch (`N`) dim.

  Based on `jax.experimental.stax.Flatten`. Has a similar API.

  Args:
    spec: an optional string specifying ordering of the input dimensions, e.g.,
      `'NHWC'` for `[batch_size, height, width, channels] or `'NCHW'` for
      `[batch_size, channels, height, width]`.
  """

  def get_output_shape(input_shape):
    batch_axis, _, _ = _parse_axes(spec, input_shape)
    return (
        input_shape[batch_axis],
        functools.reduce(op.mul,
            input_shape[:batch_axis] + input_shape[batch_axis + 1:],
            1))

  def init_fn(rng, input_shape):
    output_shape = get_output_shape(input_shape)
    return output_shape, ()

  def apply_fn(params, inputs, mask_constant=None, **kwargs):
    batch_axis, _, _ = _parse_axes(spec, inputs)
    inputs, mask = _get_masked_inputs_and_mask(inputs, mask_constant)
    output_shape = get_output_shape(inputs.shape)
    outputs = np.moveaxis(inputs, batch_axis, 0).reshape(output_shape)

    if mask is not None:
      mask = np.broadcast_to(mask, inputs.shape)
      mask = np.moveaxis(mask, batch_axis, 0).reshape(output_shape)

    return _drop_mask(outputs, mask)

  def kernel_fn(kernels):
    """Compute kernels."""
    var1, nngp, var2, ntk, is_gaussian, marginal, cross = (
        kernels.var1, kernels.nngp, kernels.var2, kernels.ntk,
        kernels.is_gaussian, kernels.marginal, kernels.cross)

    if marginal == M.OVER_ALL and cross == M.OVER_ALL:
      return kernels

    def trace(x):
      data_ndim = 2 - x.ndim % 2
      while x.ndim > data_ndim:
        x = np.trace(x, axis1=-2, axis2=-1) / x.shape[-1]
      return x

    if marginal == M.OVER_PIXELS:
      spatial_axes = tuple(range(var1.ndim)[1:])
      var1 = np.mean(var1, axis=spatial_axes)
      var2 = var2 if var2 is None else np.mean(var2, axis=spatial_axes)

    elif marginal in [M.OVER_POINTS, M.NO]:
      if marginal == M.NO:
        var1 = np.moveaxis(np.diagonal(var1, axis1=0, axis2=1), -1, 0)
        if var2 is not None:
          var2 = np.moveaxis(np.diagonal(var2, axis1=0, axis2=1), -1, 0)

      var1 = trace(var1)
      var2 = var2 if var2 is None else trace(var2)

    elif marginal != M.OVER_ALL:
      raise NotImplementedError(
          "Only implemented for , `OVER_ALL`, `OVER_PIXELS`, `OVER_POINTS` and "
          " `NO`;  supplied {}".format(marginal))

    if cross == M.OVER_PIXELS:
      spatial_axes = tuple(range(nngp.ndim))[2:]
      nngp = np.mean(nngp, axis=spatial_axes)
      ntk =  np.mean(ntk, axis=spatial_axes) if utils.is_array(ntk) else ntk

    elif cross in [M.OVER_POINTS, M.NO]:
      nngp = trace(nngp)
      ntk = trace(ntk) if utils.is_array(ntk) else ntk

    elif cross != M.OVER_ALL:
      raise NotImplementedError(
          "Only implemented for , `OVER_ALL`, `OVER_PIXELS`, `OVER_POINTS` and "
          "`NO`; supplied {}".format(cross))

    return kernels._replace(
        var1=var1, nngp=nngp, var2=var2, ntk=ntk, is_gaussian=is_gaussian,
        is_reversed=False, marginal=M.OVER_ALL, cross=M.OVER_ALL)

  setattr(kernel_fn, _INPUT_REQ, {'marginal': M.OVER_ALL,
                                  'cross': M.OVER_ALL,
                                  'spec': spec})
  return init_fn, apply_fn, kernel_fn


class PositionalEmbedding(enum.Enum):
  NONE = 'NONE'
  SUM = 'SUM'
  CONCAT = 'CONCAT'
  DECAYING = 'DECAYING'


def _attention_mechanism(name):
  return {
      'softmax': ostax.softmax,
      'identity': lambda x: x,
      'abs': np.abs,
      'relu': lambda x: np.maximum(x, 0.)
  }[name.lower()]


@_layer
def GlobalSelfAttention(n_chan_out,
                        n_chan_key,
                        n_chan_val,
                        n_heads,
                        fixed=True,
                        W_key_std=1.0,
                        W_value_std=1.0,
                        W_query_std=1.0,
                        W_out_std=1.0,
                        b_std=0.0,
                        W_key_init=_randn(1.0),
                        W_value_init=_randn(1.0),
                        W_query_init=_randn(1.0),
                        W_out_init=_randn(1.0),
                        b_init=_randn(1.0),
                        attention_mechanism='SOFTMAX',
                        pos_emb_type=PositionalEmbedding.NONE.name,
                        n_chan_pos_emb=None,  # None for same as n_chan_in
                        W_pos_emb_std=1.0,  # sqrt(rho)
                        pos_emb_decay_rate=5.0,  # phi
                        val_pos_emb=False,
                        spec=None):
  """Layer construction function for (global) scaled dot-product self-attention
  with multiple attention heads.

  Two versions of attention are available (the version to be used is
  determined by the argument `fixed`):

  1) Parametric: this is the standard scaled dot-product attention, i.e.,
   the dot product between keys and queries is scaled by the squared root
   of their dimension. The expression for `nngp`/`ntk` involves an integral
   with no known closed form and thus call to `kernel_fn` results in an error.

  2) Fixed: same as Parametric except for scaling the dot products
   between keys and queries by their dimension instead of the square root
   of the same quantity, and tying the key and query weight matrices.
   This makes the `nngp`/`ntk` analytically tractable but for the price
   that, unlike in the parametric case, the dot products of keys and queries
   converge to a constant. Because this constant would be zero
   if the key and query weights are independent, the variant where these
   two weight matrices are tied was implemented resulting in non-constant
   attention weights.

  The final computation for single head is then
   ```f_h (x) + softmax(<scaling> Q(x) K(x)^T) V(x)```
  and the output of this layer is computed as
   ```f(x) = concat[f_1(x) , ... , f_<n_heads> (x)] W_out + b```
  where the shape of of `b` is (n_chan_out,), i.e., single bias per channel

  The `kernel_fn` computes the limiting kernel of the outputs of this layer
  as the number of heads and the number of feature dimensions of keys/queries
  goes to infinity.

  Args:
    n_chan_out: number of feature dimensions of outputs
    n_chan_key: number of feature dimensions of keys/queries
    n_chan_val: number of feature dimensions of values
    n_heads: number of attention heads
    fixed: if `True`, the dot products between keys and queries are
      scaled by `1 / n_chan_key` and the key and query weight matrices are tied;
      if `False`, the dot products are scaled by `1 / sqrt(n_chan_key)` and
      the key and query matrices are independent
    W_out_std: init standard deviation of the output weights values
    b_std: init standard deviation of the bias values
    W_value_std: init standard deviation of the key weights values
    W_key_std: init standard deviation of the key weights values
    W_query_std: init standard deviation of the query weights values; if `fixed`
      is `True` (and thus key and query weights are tied---see above) then keys
      are computed with `WK = WK_std * W / sqrt(n_chan_in)` and the queries are
      computed with `WQ = W_query_std * W / sqrt(n_chan_in)` weight matrices
    W_out_init: function used to sample the initial (unscaled) output weights
    b_init:  function used to sample the initial (unscaled) biases
    W_value_init: function used to sample the initial (unscaled) value weights
    W_key_init: function used to sample the initial (unscaled) key weights
    W_query_init: function used to sample the initial (unscaled) query weights
      unless `fixed` is `True` in which case the argument is ignored (see above)
    attention_mechanism: a string, `"SOFTMAX"`, `"IDENTITY"`, `"ABS"`,
      or `"RELU"`, the transformation applied to dot product attention weights.
    pos_emb_type: a string, `"NONE"`, `"SUM"`, `"CONCAT"` or `"DECAYING"`, the
      type of positional embeddings to use.
    n_chan_pos_emb: int, the number of channels in positional embeddings. `None`
      means use the same number of channels as in the layer inputs.
    W_pos_emb_std: float, init standard deviation of the random positional
      embeddings.
    pos_emb_decay_rate: float, the rate at which correlations at different
      positions in the positional embeddings decay as positions get more
      distant. Used only if `pos_emb_type == "DECAYING"`.
    val_pos_emb: a boolean, `True` indicates using positional embeddings when
      computing all of the keys/queries/values matrices, `False` makes them
      only used for keys and queries, but not values.
    spec: a string specifying ordering of the input dimensions, e.g.,
      `'NHWC'` for `[batch_size, height, width, channels] or `'NCHW'` for
      `[batch_size, channels, height, width]`.

  Raises:
    NotImplementedError: If `fixed` is `False`, call to `kernel_fn` will result
      in an error as there is no knwn analytic expression for the kernel.
    NotImplementedError: if a finite-width model forward pass is called with
      `pos_emb_type == "DECAYING"`, which is not currently implemented in the
      finite width. TODO: implement.
  """
  QK_gain = W_query_std * W_key_std
  QK_prod_scaling = float(n_chan_key if fixed else n_chan_key**0.5)

  pos_emb_type = PositionalEmbedding(pos_emb_type)
  attn_fn = _attention_mechanism(attention_mechanism)

  def init_fn(rng, input_shape):
    batch_axis, channel_axis, _ = _parse_axes(spec, input_shape)
    output_shape = (input_shape[:channel_axis] + (n_chan_out,) +
                    input_shape[channel_axis + 1:])

    rng_Q, rng_K, rng_V, rng_O, rng_b, rng_pe = random.split(rng, 6)
    n_chan_in_keys = n_chan_in_vals = input_shape[channel_axis]

    # Generate and add / append positional embeddings.
    if pos_emb_type == PositionalEmbedding.NONE:
      pos_emb = None
    else:
      # `None` means positional embeddings have the same number of channels
      # as inputs.
      _n_chan_pos_emb = (n_chan_in_keys if n_chan_pos_emb is None
                         else n_chan_pos_emb)

      pos_emb_shape = list(input_shape)
      pos_emb_shape[channel_axis] = _n_chan_pos_emb
      pos_emb_shape[batch_axis] = 1
      pos_emb = random.normal(rng_pe, pos_emb_shape) * W_pos_emb_std

      if pos_emb_type == PositionalEmbedding.CONCAT:
        n_chan_in_keys += _n_chan_pos_emb
        if val_pos_emb:
          n_chan_in_vals += _n_chan_pos_emb

      elif pos_emb_type != PositionalEmbedding.SUM:
        warnings.warn(f'pos_emb_type={pos_emb_type} not supported in finite-'
                      f'width networks. Exception will be raise in the forward '
                      f'pass.')

    key_matrices = W_key_init(rng_K, shape=(n_heads, n_chan_in_keys,
                                            n_chan_key))
    val_matrices = W_value_init(rng_V, shape=(n_heads, n_chan_in_vals,
                                              n_chan_val))
    W_out = W_out_init(rng_O, shape=(n_chan_val * n_heads, n_chan_out))
    b = b_init(rng_b, shape=(n_chan_out,))

    if fixed:
      query_matrices = None
      warnings.warn("Fixed attention used -> `W_query_init` ignored, tying"
                    " the weights (see docstring for more details).")
    else:
      query_matrices = W_query_init(rng_Q, (n_heads, n_chan_in_keys,
                                            n_chan_key))

    return (output_shape,
           (query_matrices, key_matrices, val_matrices, W_out, b, pos_emb))

  def apply_fn(params, inputs, mask_constant=None, **kwargs):
    query_matrices, key_matrices, val_matrices, W_out, b, pos_emb = params
    inputs, mask = _get_masked_inputs_and_mask(inputs, mask_constant)
    batch_axis, channel_axis, spatial_axes = _parse_axes(spec, inputs)

    # Collapse channel dimension of masks, since an FC layer is applied at each
    # spatial location.
    if mask is not None:
      mask = np.all(mask, channel_axis, keepdims=True)

    # Mask embeddings.
    if mask is not None and pos_emb is not None:
      pos_emb = np.where(mask, np.zeros((), inputs.dtype), pos_emb)

    # Add / concat positional embeddings.
    if pos_emb_type == PositionalEmbedding.SUM:
      inputs_val = inputs if not val_pos_emb else None
      inputs = pos_emb + inputs

    elif pos_emb_type == PositionalEmbedding.CONCAT:
      inputs_val = inputs if not val_pos_emb else None
      _n_chan_pos_emb = (inputs.shape[channel_axis] if n_chan_pos_emb is None
                         else n_chan_pos_emb)
      pos_emb = np.broadcast_to(pos_emb,
          inputs.shape[:channel_axis] + (_n_chan_pos_emb,) +
          inputs.shape[channel_axis + 1:])
      inputs = np.concatenate([inputs, pos_emb], axis=channel_axis)

    elif pos_emb_type == PositionalEmbedding.NONE:
      inputs_val = None

    else:
      raise NotImplementedError(f'Positional embeddings of type {pos_emb_type} '
                                f'not implemented for finite-width networks.')

    n = inputs.shape[batch_axis]
    n_chan_in = inputs.shape[channel_axis]
    spatial_shape = tuple(inputs.shape[i] for i in spatial_axes)

    # Prepare separate inputs for values if asked to not add positional
    # embeddings to values.
    if inputs_val is not None:
      inputs_val = np.moveaxis(inputs_val, (batch_axis, channel_axis), (0, -1))
      inputs_val = inputs_val.reshape((n, -1, inputs_val.shape[-1]))

    # Flatten all spatial dimensions and make input of shape
    # `[batch_size, total_spatial_size, n_channels]`.
    inputs = np.moveaxis(inputs, (batch_axis, channel_axis), (0, -1))
    inputs = inputs.reshape((n, -1, n_chan_in))

    def _inputs_dot(matrices, std, _inputs=inputs):
      ret = np.dot(_inputs, std * matrices / np.sqrt(n_chan_in))
      return np.moveaxis(ret, 2, 0)

    # Drop positional embedding information for value matrices if requested.
    if inputs_val is not None:
      values = _inputs_dot(val_matrices, W_value_std, inputs_val)
    else:
      values = _inputs_dot(val_matrices, W_value_std)

    keys = _inputs_dot(key_matrices, W_key_std)
    if fixed:
      queries = keys * W_query_std / W_key_std
    else:
      queries = _inputs_dot(query_matrices, W_query_std)

    G_mat = np.matmul(queries, np.moveaxis(keys, -1, -2)) / QK_prod_scaling

    if mask is not None:
      # Flatten all spatial dimensions and make mask of shape
      # `[1, batch_size, 1, total_spatial_size]`.
      mask_flat = np.moveaxis(mask, (batch_axis, channel_axis), (0, -1))
      mask_flat = mask_flat.reshape((1, mask.shape[0], 1,
                                     _size_at(mask, spatial_axes)))
      if attention_mechanism.lower() == 'softmax':
          G_mat = np.where(mask_flat, _NEG_INF, G_mat)
      elif attention_mechanism.lower() in ('identity', 'relu', 'abs'):
          G_mat = np.where(mask_flat, np.zeros((), G_mat.dtype), G_mat)
      else:
        raise NotImplementedError(attention_mechanism, mask)

    G_mat = attn_fn(G_mat)
    heads = np.matmul(G_mat, values)
    heads = np.moveaxis(heads, 0, -1)
    heads = np.reshape(heads, heads.shape[:-2] + (-1,))

    outputs = np.matmul(heads,
                        W_out_std * W_out / np.sqrt(n_chan_val * n_heads))
    outputs = np.reshape(outputs,
                         (n,) + spatial_shape + (n_chan_out,)) + b_std * b
    outputs = np.moveaxis(outputs, (0, -1), (batch_axis, channel_axis))
    return _drop_mask(outputs, mask)

  def kernel_fn(kernels):
    (var1, nngp, var2, ntk, cross, marginal, shape1, shape2, mask1, mask2,
    is_reversed) = (
        kernels.var1, kernels.nngp, kernels.var2, kernels.ntk, kernels.cross,
        kernels.marginal, kernels.shape1, kernels.shape2, kernels.mask1,
        kernels.mask2, kernels.is_reversed)

    batch_axis, channel_axis, _ = _parse_axes(spec, kernels.shape1)

    if not fixed:
      raise NotImplementedError("No known closed form expression.")

    def _get_weighting(mat, mask):
      if mat is None:
        return None

      if marginal == M.NO:
        mat = np.moveaxis(np.diagonal(mat, axis1=0, axis2=1), -1, 0)
      axes = tuple(range(mat.ndim))

      if mask is not None:
        mask = np.moveaxis(mask, (batch_axis, channel_axis), (0, -1))
        mask = np.all(mask, axis=channel_axis)
        if is_reversed:
          mask = np.moveaxis(mask, range(1, mask.ndim),
                                   range(mask.ndim -1, 0, -1))
        mask = _interleave_ones(mask, 1, False)
        if attention_mechanism.lower() == 'softmax':
          mat = np.where(mask, _NEG_INF, mat)
        else:
          mat = np.where(mask, np.zeros((), mat.dtype), mat)

      if attention_mechanism.lower() == 'softmax':
        return attn_fn(QK_gain * mat, axis=axes[2::2])
      else:
        return attn_fn(QK_gain * mat)

    def _weigh_kernel(mat, G1, G2=None):
      if not utils.is_array(mat):
        return mat

      G2 = G1 if G2 is None else G2

      # Spatial axes
      G1_dims = tuple(range(1, G1.ndim))
      G2_dims = tuple(range(G1.ndim, G1.ndim + G2.ndim - 1))
      mat_dims = _zip_flat(G1_dims[1::2], G2_dims[1::2])
      res_dims = _zip_flat(G1_dims[::2], G2_dims[::2])

      if G1.shape[0] == 1:
        G1 = np.squeeze(G1, 0)
      else:
        G1_dims = (0,) + G1_dims

      # Batch axes
      if mat.ndim % 2:
        if G2.shape[0] == 1:
          G2 = np.squeeze(G2, 0)
        else:
          G2_dims = (0,) + G2_dims

        if mat.shape[0] == 1:
          mat = np.squeeze(mat, 0)
        else:
          mat_dims = (0,) + mat_dims
        res_dims = (0,) + res_dims

      else:
        if G2.shape[0] == 1:
          G2 = np.squeeze(G2, 0)
        else:
          G2_dims = (-1,) + G2_dims

        if mat.shape[0] == mat.shape[1] == 1:
          mat = np.squeeze(mat, (0, 1))
        else:
          mat_dims = (0, -1) + mat_dims
        res_dims = (0, -1) + res_dims

      res = np.einsum(G1, G1_dims, mat, mat_dims, G2, G2_dims, res_dims,
                      optimize=True)

      OV_gain = W_out_std * W_value_std
      if pos_emb_type == PositionalEmbedding.CONCAT and not val_pos_emb:
        inputs_weight, _ = _get_pos_emb_coeffs(shape1[channel_axis],
                                               n_chan_pos_emb)
        OV_gain *= inputs_weight ** 0.5

      return _affine(res, OV_gain, b_std)

    # Collapse channel dimension of masks, since an FC layer is applied at each
    # spatial location.
    if mask1 is not None:
      mask1 = np.all(mask1, channel_axis, keepdims=True)
    if mask2 is not None:
      mask2 = np.all(mask2, channel_axis, keepdims=True)

    # Generate (optional) positional embedding covariances.
    R1, R12, R2 = _get_all_pos_emb(shape1, shape2, mask1, mask2, spec, cross,
                                   marginal, is_reversed, pos_emb_type,
                                   pos_emb_decay_rate)

    def _get_interpolation_coefficients():
      mat_weight, pos_emb_weight = 1, W_pos_emb_std**2

      if pos_emb_type == PositionalEmbedding.CONCAT:
        # Reweight based on relative widths of inputs and channels.
        inputs_weight, emb_weight = _get_pos_emb_coeffs(shape1[channel_axis],
                                                        n_chan_pos_emb)
        mat_weight *= inputs_weight
        pos_emb_weight *= emb_weight

      return mat_weight, pos_emb_weight

    # Generate kernel interpolations.
    kern_weight, pos_emb_weight = _get_interpolation_coefficients()
    var1_interp = _weighted_sum(var1, R1, kern_weight, pos_emb_weight)
    var2_interp = _weighted_sum(var2, R2, kern_weight, pos_emb_weight)
    if val_pos_emb:
      nngp = _weighted_sum(nngp, R12, kern_weight, pos_emb_weight)
      ntk = _weighted_sum(ntk, R12, kern_weight, pos_emb_weight)

    G1 = _get_weighting(var1_interp, mask1)
    G2 = _get_weighting(var2_interp, mask2)

    var1 = _weigh_kernel(var1_interp if val_pos_emb else var1, G1)
    var2 = _weigh_kernel(var2_interp if val_pos_emb else var2, G2)

    nngp = _weigh_kernel(nngp, G1, G2)
    if ntk is not None:
      ntk = _weigh_kernel(ntk, G1, G2) + 2 * (nngp - b_std**2)

    var1, nngp, var2, ntk = _mask_kernels(var1, nngp, var2, ntk, cross,
                                          marginal, is_reversed, mask1, mask2,
                                          spec)

    return kernels._replace(var1=var1, nngp=nngp, var2=var2, ntk=ntk,
                            is_gaussian=True, mask1=mask1, mask2=mask2)

  setattr(kernel_fn, _INPUT_REQ, {'marginal': M.OVER_POINTS,
                                  'cross': M.NO,
                                  'spec': spec})
  return init_fn, apply_fn, kernel_fn


def _get_pos_emb_coeffs(n_chan_input, n_chan_pos_emb):
  n_chan_inputs = n_chan_input
  _n_chan_pos_emb = (n_chan_inputs if n_chan_pos_emb is None
                     else n_chan_pos_emb)
  n_chan_total = n_chan_inputs + _n_chan_pos_emb
  inputs_weight = n_chan_inputs / n_chan_total
  pos_emb_weight = _n_chan_pos_emb / n_chan_total
  return inputs_weight, pos_emb_weight


def _weighted_sum(x, y, x_weight, y_weight):
  if x is None or y is None:
    return x
  return x_weight * x + y_weight * y


def _pos_emb_identity(shape, mask1, mask2, batch_ndim, spec, marginal,
                      is_reversed):
  if shape is None:
    return None

  _, _, spatial_axes = _parse_axes(spec, shape)
  spatial_size = _size_at(shape, spatial_axes)
  spatial_shape = tuple(shape[i] for i in spatial_axes)

  R = np.eye(spatial_size)
  R = np.reshape(R, (1,) * batch_ndim + spatial_shape * 2)
  R = _zip_axes(R, batch_ndim)
  R = _mask_kernel(R, mask1, mask2, batch_ndim, spec, marginal, False)
  if is_reversed:
    R = utils.revert_zipped(R, shape)
  return R


def _pos_emb_decay(shape1, shape2, mask1, mask2, batch_ndim, spec, marginal,
                   is_reversed, pos_emb_decay_rate):
  if shape1 is None:
    return None

  _, channel_axis, spatial_axes = _parse_axes(spec, shape1)

  def get_dist_arr(shape, mask, axis):
    if shape is None:
      return None

    arange = np.arange(shape[axis])
    size = len(arange)
    s = (1,) * axis + (size,) +  (1,) * (len(spatial_axes) - axis + 1)
    arange = np.reshape(arange, s)
    if mask is not None:
      size -= np.expand_dims(np.count_nonzero(mask, axis), axis)
    return np.squeeze(arange / np.maximum(size, 1), channel_axis)

  def get_pdist_arr(d1, d2):
    d2 = d1 if d2 is None else d2
    return _outer_prod(d1, d2, 2 - batch_ndim, op.sub)

  R = np.zeros((1,) * (batch_ndim + len(spatial_axes) * 2))
  for axis in spatial_axes:
    d1 = get_dist_arr(shape1, mask1, axis)
    d2 = get_dist_arr(shape2, mask2, axis)
    pd = get_pdist_arr(d1, d2)
    R += pd**2

  R /= max(len(spatial_axes), 1)
  R = np.exp(-pos_emb_decay_rate * R)

  R = _mask_kernel(R, mask1, mask2, batch_ndim, spec, marginal, False)
  if is_reversed:
    R = utils.revert_zipped(R, shape1)
  return R


def _get_all_pos_emb(shape1, shape2, mask1, mask2, spec, cross, marginal,
                     is_reversed, pos_emb_type, pos_emb_decay_rate):
  if pos_emb_type in (PositionalEmbedding.SUM, PositionalEmbedding.CONCAT):
    R1 = _pos_emb_identity(shape1, mask1, None, 1, spec, marginal, is_reversed)
    R2 = _pos_emb_identity(shape2, mask2, None, 1, spec, marginal, is_reversed)
    R12 = _pos_emb_identity(shape1, mask1, mask2, 2, spec, cross, is_reversed)

  elif pos_emb_type == PositionalEmbedding.DECAYING:
    R1 = _pos_emb_decay(shape1, None, mask1, None, 1, spec, marginal,
                        is_reversed, pos_emb_decay_rate)
    R2 = _pos_emb_decay(shape2, None, mask2, None, 1, spec, marginal,
                        is_reversed, pos_emb_decay_rate)
    R12 = _pos_emb_decay(shape1, shape2, mask1, mask2, 2, spec, cross,
                         is_reversed, pos_emb_decay_rate)

  elif pos_emb_type == PositionalEmbedding.NONE:
    R1, R12, R2 = None, None, None

  else:
    raise NotImplementedError(f'Positional embeddings of type {pos_emb_type} '
                              f'not implemented for infinite-width networks.')
  return R1, R12, R2



@_layer
def LayerNorm(axis=-1, eps=1e-12, spec=None):
  """Layer normalisation.

  Args:
    axis: int or a tuple, specifies dimensions over which to normalise
    eps: float, specifies (small) positive constant to be added to the variance
      estimates in order to prevent division by zero.
    spec: an optional `string`, specifying the dimension order of the input,
      e.g. `NCHW` or `NHWC`.
  """

  def init_fn(rng, input_shape):
    return input_shape, ()

  def apply_fn(params, inputs, mask_constant=None, **kwargs):
    inputs, mask = _get_masked_inputs_and_mask(inputs, mask_constant)
    _axis = _canonicalize_axis(axis, inputs)

    if mask is not None:
      size = _size_at(inputs, _axis)
      mask_size = np.count_nonzero(mask, _axis)
      for i in sorted(a % inputs.ndim for a in _axis):
        mask_size = np.expand_dims(mask_size, i)
      size -= mask_size
      mean = np.sum(inputs, axis=_axis, keepdims=True) / size
      var = np.sum((inputs - mean)**2, axis=_axis, keepdims=True) / size

    else:
      mean = np.mean(inputs, axis=_axis, keepdims=True)
      var = np.var(inputs, axis=_axis, keepdims=True)

    outputs = (inputs - mean) / np.sqrt(eps + var)
    return _drop_mask(outputs, mask)

  def kernel_fn(kernels):
    var1, nngp, var2, ntk, is_reversed, marginal, cross, shape1 = (
        kernels.var1, kernels.nngp, kernels.var2, kernels.ntk,
        kernels.is_reversed, kernels.marginal, kernels.cross, kernels.shape1)

    batch_axis, channel_axis, spatial_axes = _parse_axes(spec, shape1)
    _axis = _canonicalize_axis(axis, shape1)

    if marginal != M.OVER_ALL or cross != M.OVER_ALL:
      if channel_axis not in _axis:
        raise ValueError("Normalisation over channels (axis %d) necessary for "
                         "convergence to an asymptotic kernel; "
                         "got axis=%s" % (channel_axis, _axis))

      if batch_axis in _axis:
        raise ValueError("Normalisation over batch (axis %d) not supported for "
                         "convergence to an asymptotic kernel; "
                         "got axis=%s" % (batch_axis, _axis))

      _axis.remove(channel_axis)
      kernel_axis = tuple(1 +
                          spatial_axes[::(-1 if is_reversed else 1)].index(i)
                          for i in _axis)

    else:
      if _axis != [channel_axis]:
        raise ValueError("Normalisation over features necessary for convergence"
                         " to an asymptotic kernel; axis={}".format(_axis))
      kernel_axis = ()

    prod11, prod12, prod22 = _get_normalising_prod(
        eps + var1, var2 if var2 is None else eps + var2,
        marginal, axis=kernel_axis)

    nngp /= np.sqrt(prod12)

    if utils.is_array(ntk):
      ntk /= np.sqrt(prod12)

    var1 /= np.sqrt(prod11)
    if var2 is not None:
      var2 /= np.sqrt(prod22)

    return kernels._replace(var1=var1, nngp=nngp, var2=var2, ntk=ntk)

  if isinstance(axis, tuple) and len(axis) > 1:
    setattr(kernel_fn, _INPUT_REQ, {'marginal': M.OVER_PIXELS,
                                    'cross': M.OVER_PIXELS,
                                    'spec': spec})
  return init_fn, apply_fn, kernel_fn


@_layer
def Dropout(rate, mode='train'):
  """Dropout layer.

  Args:
    rate: A float specifying the keep `rate`, e.g. `rate=1` is equivalent to
      keeping all neurons.
    mode: either `train` or `test`.
  """
  if mode not in ['test', 'train']:
    raise ValueError('The `mode` must be either `test`  or `train`.')
  if rate <= 0. or rate > 1.:
    raise ValueError('The `rate` must be > 0. and <= 1.')

  init_fn, apply_fn_old = ostax.Dropout(rate, mode=mode)

  def apply_fn(params, inputs, mask_constant=None, **kwargs):
    inputs, mask = _get_masked_inputs_and_mask(inputs, mask_constant)
    outputs = apply_fn_old(params, inputs, **kwargs)
    return _drop_mask(outputs, mask)

  kernel_fn_test = lambda kernels: kernels

  def kernel_fn_train(kernels):
    """kernel_fn for `train` mode. """
    var1, nngp, var2, ntk, cross, x1_is_x2, is_input = (
        kernels.var1, kernels.nngp, kernels.var2, kernels.ntk,
        kernels.cross, kernels.x1_is_x2, kernels.is_input)

    if is_input:
      raise ValueError('Dropout cannot be applied to the input layer.')
    factor = 1./rate
    var1 *= factor
    if var2 is not None:
      var2 *= factor
    new_factor = np.where(x1_is_x2, factor, 1.)
    nngp = _diag_mul(nngp, new_factor, cross)
    ntk = _diag_mul(ntk, new_factor, cross) if utils.is_array(ntk) else ntk

    # TODO: under which condition could we leave `is_gaussian` unchanged?
    return kernels._replace(var1=var1, nngp=nngp, var2=var2, ntk=ntk,
                            is_gaussian=False)

  kernel_fn = kernel_fn_test if mode == 'test' else kernel_fn_train

  return init_fn, apply_fn, kernel_fn


# MASKING


def _get_masked_inputs_and_mask(x, mask_constant):
  if x is None:
    mask = None

  elif isinstance(x, tuple):
    x, mask = x

  elif isinstance(x, np.ndarray):
    if mask_constant is None:
      mask = None
    elif np.isnan(mask_constant):
      mask = np.isnan(x)
    else:
      mask = x == mask_constant
  else:
    raise TypeError(x, type(x))

  if mask is not None:
    x = np.where(mask, np.zeros((), x.dtype), x)

  return x, mask


def _add_two_masks(mask1, mask2):
  if mask1 is None:
    return mask2

  if mask2 is None:
    return mask1

  return mask1 & mask2


def _add_masks(masks):
  return functools.reduce(_add_two_masks, masks, None)


def _map_tuples(fn, tuples):
  return tuple(map(fn, zip(*(t for t in tuples))))


def _concat_masks(masks, inputs, axis):
  if all(m is None for m in masks):
    return None

  if inputs is not None:
    masks = [m if m is None else np.broadcast_to(
        m,
        m.shape[:axis] + inputs[i].shape[axis: axis + 1] + m.shape[axis + 1:])
             for i, m in enumerate(masks)]

  max_shape = _map_tuples(max, (m.shape for m in masks if m is not None))

  if inputs is not None:
    max_shapes = [tuple(map(min, max_shape, i.shape)) for i in inputs]

  masks = [
      (np.broadcast_to(
          m,
          max_shape[:axis] + m.shape[axis: axis + 1] + max_shape[axis + 1:])
       if m is not None
       else np.zeros_like(max_shapes[i], dtype=np.bool_))
      for i, m in enumerate(masks)
  ]

  return np.concatenate(masks, axis)


def _fan_in_masks(masks, inputs, axis):
  if axis is None:
    return _add_masks(masks)

  return _concat_masks(masks, inputs, axis)


def _drop_mask(outputs, mask):
  if mask is None:
    return outputs

  return outputs, mask


def _pool_mask(mask, window_shape, strides, padding, spec):
  if mask is None:
    return mask

  window_shape = list(window_shape)
  strides = list(strides)
  batch_axis, channel_axis, _ = _parse_axes(spec, mask)

  for i in sorted((batch_axis, channel_axis)):
    if i == batch_axis:
      window_shape.insert(i, 1)
      strides.insert(i, 1)
    else:
      window_shape.insert(i, mask.shape[i])
      strides.insert(i, mask.shape[i])

  # Get the output shape.
  out_shape = lax.reduce_window_shape_tuple(
      mask.shape,
      window_shape,
      strides,
      padding.name
  )

  # If shapes match, return mask without change.
  if out_shape == mask.shape:
    return mask

  # If not, stride through the mask.
  else:
    pads = lax.padtype_to_pads(mask.shape, window_shape, strides, padding.name)
    slices = ()
    for i in range(mask.ndim):
      start = - pads[i][0] + (window_shape[i] - 1) // 2
      end = start + 1 + (out_shape[i] - 1) * strides[i]
      slices += (slice(start, end, strides[i]),)

    mask = mask[slices]
    if mask.shape != out_shape:
      raise ValueError('This should not happen.')
    return mask


def _conv_mask(mask, filter_shape, strides, padding, dimension_numbers):
  if mask is None:
    return None

  # Collapse channel dimension of masks, since an FC layer is applied at each
  # spatial location.
  mask = np.all(mask, axis=dimension_numbers[0].index('C'), keepdims=True)
  return _pool_mask(mask, filter_shape, strides, padding,
                    dimension_numbers[0])


def _mask_kernel(mat, mask1, mask2, batch_ndim, spec, marginal, is_reversed):
  if not utils.is_array(mat) or mask1 is None:
    return mat

  mask = _get_mask_prod(mask1, mask2, batch_ndim, spec, marginal, is_reversed)
  mat = np.where(mask, np.zeros((), mat.dtype), mat)
  return mat


def _mask_kernels(var1, nngp, var2, ntk, cross, marginal, is_reversed, mask1,
                  mask2, spec):
  var1 = _mask_kernel(var1, mask1, None, 1, spec, marginal, is_reversed)
  var2 = _mask_kernel(var2, mask2, None, 1, spec, marginal, is_reversed)
  nngp = _mask_kernel(nngp, mask1, mask2, 2, spec, cross, is_reversed)
  ntk = _mask_kernel(ntk, mask1, mask2, 2, spec, cross, is_reversed)
  return var1, nngp, var2, ntk


def _get_mask_prod(mask1, mask2, batch_ndim, spec, marginal, is_reversed):
  if mask1 is None:
    return False

  batch_axis, channel_axis, _ = _parse_axes(spec, mask1)

  def reshape(m):
    if m is not None:
      m = np.all(m, axis=channel_axis, keepdims=True)
      m = np.squeeze(np.moveaxis(m, (batch_axis, channel_axis), (0, -1)), -1)
      if is_reversed:
        m = np.moveaxis(m, range(1, m.ndim), range(m.ndim - 1, 0, -1))
    return m

  mask1, mask2 = reshape(mask1), reshape(mask2)
  if mask2 is None:
    mask2 = mask1

  if marginal in (M.OVER_POINTS, M.NO):
    mask = _outer_prod(mask1, mask2, 2 - batch_ndim, op.or_)

  elif marginal in (M.OVER_ALL, M.OVER_PIXELS):
    if batch_ndim == 2:
      if mask2 is None:
        mask2 = mask1

      mask1 = np.expand_dims(mask1, batch_ndim - 1)
      mask2 = np.expand_dims(mask2, batch_ndim - 2)

    mask = mask1 | mask2

  else:
    raise ValueError(f'Marginalisation {marginal} not recognized.')

  return mask
