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
  (https://arxiv.org/abs/1806.07572, page 3). Standard parameterization can
  be specified for Conv and Dense layers by a keyword argument.

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
import frozendict
from jax import lax
from jax import linear_util as lu
from jax import ops
from jax import random
from jax.abstract_arrays import ShapedArray
from jax.api_util import flatten_fun
import jax.experimental.stax as ostax
import jax.interpreters.partial_eval as pe
import jax.numpy as np
from jax.scipy.special import erf
from jax.tree_util import tree_map, tree_flatten, tree_unflatten
from neural_tangents.utils import utils
from neural_tangents.utils.kernel import Kernel
from neural_tangents.utils.kernel import Marginalisation as M


_CONV_QAB_DIMENSION_NUMBERS = ('NCHW', 'HWIO', 'NCHW')
_COVARIANCES_REQ = 'covariances_req'
_DEFAULT_MARGINALIZATION = frozendict.frozendict({'marginal': M.OVER_ALL,
                                                  'cross': M.OVER_ALL})


class Padding(enum.Enum):
  CIRCULAR = 'CIRCULAR'
  SAME = 'SAME'
  VALID = 'VALID'


def _is_array(x):
  return isinstance(x, np.ndarray) and hasattr(x, 'shape') and x.shape


def _set_covariances_req_attr(combinator_kernel_fn, kernel_fns):
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
  #TODO(jirihron): make `NO` marginalisation the default

  Args:
    combinator_kernel_fn: a 'kernel_fn` of a `serial` or `parallel` combinator.
    kernel_fns: list of 'kernel_fn`s fed to the `kernel_fns` (e.g. a list of
      convolutional layers and nonlinearities to be chained together with the
      `serial` combinator).

  Returns:
    `kernel_fns` with the `_COVARIANCES_REQ` attribute set accordingly to
      the needs of their corresponding layer
  """
  def _get_maximal_element(covs_req, comparison_op):
    for f in reversed(kernel_fns):
      if hasattr(f, _COVARIANCES_REQ):
        reqs = getattr(f, _COVARIANCES_REQ)

        marginal = reqs['marginal']
        if comparison_op(marginal, covs_req['marginal']):
          covs_req['marginal'] = marginal

        cross = reqs['cross']
        if comparison_op(cross, covs_req['cross']):
          covs_req['cross'] = cross

        if 'spec' in reqs:
          covs_req['spec'] = reqs['spec']

    return covs_req

  # `_get_maximal_element` sets up the code for `NO` marginalisation by default
  covs_req = _get_maximal_element(
      dict(_DEFAULT_MARGINALIZATION), lambda x, y: x > y)

  setattr(combinator_kernel_fn, _COVARIANCES_REQ, covs_req)
  return combinator_kernel_fn


def _randn(stddev=1e-2):
  """`jax.experimental.stax.randn` for implicitly-typed results."""
  def init(rng, shape):
    return stddev * random.normal(rng, shape)
  return init


def _double_tuple(x):
  return tuple(v for v in x for _ in range(2))


def _interleave_axes(mat, start_axis=0):
  """Interleave axes starting from `start_axis`.

  Changes the shape as follows:
  [..., X, Y, Z, ..., X, Y, Z, ...] -> [..., X, X, ..., Y, Y, ..., Z, Z, ...]

  Args:
    mat: `np.ndarray` with an even number of dimensions following `start_axis`.
    start_axis: `int`, number of axis from which to interleave.

  Returns:
    A `np.ndarray` with a new shape.
  """
  n_axes, ragged = divmod(mat.ndim - start_axis, 2)
  if ragged:
    raise ValueError('Need even number of axes to interleave, got %d.'
                     % (mat.ndim - start_axis))

  mat = np.moveaxis(mat,
                    range(mat.ndim)[-n_axes:],
                    range(start_axis + 1, start_axis + 1 + n_axes * 2, 2))
  return mat


def _point_marg(x, batch_axis, channel_axis):
  n, channels = x.shape[batch_axis], x.shape[channel_axis]
  spatial_shape = tuple(s for i, s in enumerate(x.shape)
                        if i not in (batch_axis, channel_axis))
  x_flat = np.reshape(np.moveaxis(x, (batch_axis, channel_axis), (0, -1)),
                      (n, -1, channels))
  x_flat_t = np.reshape(np.moveaxis(x, (batch_axis, channel_axis), (0, 1)),
                        (n, channels, -1))
  ret = np.matmul(x_flat, x_flat_t)
  ret = np.reshape(ret, (n,) + spatial_shape * 2)

  ret = _interleave_axes(ret, start_axis=1)
  return ret


def _get_variance(x, marginal_type, batch_axis, channel_axis):
  if marginal_type in (M.OVER_ALL, M.OVER_PIXELS):
    ret = np.sum(x**2, axis=channel_axis, keepdims=True)
    ret = np.moveaxis(ret, (batch_axis, channel_axis), (0, -1))
    ret = np.squeeze(ret, axis=-1)

  elif marginal_type == M.OVER_POINTS:
    ret = _point_marg(x, batch_axis, channel_axis)

  elif marginal_type == M.NO:
    x_c = np.moveaxis(x, (batch_axis, channel_axis), (0, -1))
    ret = np.squeeze(np.dot(x_c, x_c[..., None]), -1)
    ret = _interleave_axes(ret)

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
    ret = np.matmul(np.moveaxis(x1, (batch_axis, channel_axis), (-2, -1)),
                    np.moveaxis(x2, (batch_axis, channel_axis), (-1, -2)))
    ret = np.moveaxis(ret, (-2, -1), (0, 1))

  elif marginal_type in (M.OVER_POINTS, M.NO):
    # OVER_POINTS and NO coincide for the cross term
    ret = np.squeeze(np.dot(
        np.moveaxis(x1, (batch_axis, channel_axis), (0, -1)),
        np.moveaxis(x2, (batch_axis, channel_axis), (0, -1))[..., None]),
        -1)
    ret = _interleave_axes(ret)

  else:
    raise NotImplementedError(
        "Only implemented for `OVER_ALL`, `OVER_PIXELS`, `OVER_POINTS` "
        "and `NO`; supplied {}".format(marginal_type))

  return ret / x1.shape[channel_axis]


def _diag_mul_over_points(A, mul):
  if (A.shape[0] != A.shape[1] or
      A.shape[2] != A.shape[3] or
      A.shape[4] != A.shape[5]):
    return A

  batch_idx = np.arange(A.shape[0]).reshape((A.shape[0], 1, 1))
  x_idx = np.arange(A.shape[2]).reshape((1, A.shape[2], 1))
  y_idx = np.arange(A.shape[4]).reshape((1, 1, A.shape[4]))
  idx = (batch_idx,) * 2 + (x_idx,) * 2 + (y_idx,) * 2

  diag = np.diagonal(np.diagonal(np.diagonal(A)))
  A = ops.index_update(A, idx, diag * mul)
  return A


def _diag_mul_over_pixels(A, mul):
  if A.shape[0] != A.shape[1]:
    return A

  idx = np.diag_indices(A.shape[0]) + (Ellipsis,)
  diag = np.moveaxis(np.diagonal(A), -1, 0)
  A = ops.index_update(A, idx, diag * mul)
  return A


def _diag_mul_over_all(A, mul):
  if A.shape[0] != A.shape[1]:
    return A
  idx = np.diag_indices(A.shape[0])
  diag = np.diag(A)
  A = ops.index_update(A, idx, diag * mul)
  return A


def _diag_mul(A, mul):
  if A.ndim not in [2, 4, 6]:
    raise ValueError('The array must be a 2, 4 or 6d tensor. A {}d tensor is '
                     'provided.'.format(A.ndim))
  if A.ndim == 2:
    return _diag_mul_over_all(A, mul)
  elif A.ndim == 4:
    return _diag_mul_over_pixels(A, mul)
  return _diag_mul_over_points(A, mul)


def _inputs_to_kernel(x1,
                      x2,
                      marginal,
                      cross,
                      compute_ntk,
                      spec=None,
                      eps=1e-12):
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
    x1: a 2D `np.ndarray` of shape `[batch_size_1, n_features]` (dense
      network) or 4D of shape `[batch_size_1, height, width, channels]`
      (conv-nets).
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
      the input, e.g. `NCHW` or `NHWC`.
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
  if x1.ndim not in (2, 4):
    raise ValueError('Inputs must be 2D or 4D `np.ndarray`s of shape '
                     '`[batch_size, n_features]` or '
                     '`[batch_size, height, width, channels]`, '
                     'got %s.' % str(x1.shape))

  if x1.ndim == 2 and not marginal == cross == M.OVER_ALL:
    raise ValueError('`OVER_ALL` marginalisation should be used for 2D inputs; '
                     'was: `marginal`={}, `cross`={}'.format(marginal, cross))

  if cross == M.OVER_POINTS:
    raise ValueError('Required `OVER_POINTS` to be computed for `nngp`/`ntk`. '
                     '`OVER_POINTS` is only meant for `var1`/`var2`. '
                     'Use `NO` instead to compute all covariances.')

  if spec and 'N' not in spec:
    raise NotImplementedError('A separate batch dimension `N` is required, got'
                              ' spec = %s.' % spec)
  batch_axis = spec.index('N') if spec else 0
  if batch_axis != 0:
    # TODO(romann): add support or clear error for batching.
    warnings.warn('!!! Non-leading (!= 0) batch dimension (`N`) in the '
                  'input layer is not supported for batching and on '
                  'TPUs, got spec = %s. !!!' % spec)

  if spec == 'CN':
    raise NotImplementedError('Fully-connected models only support `NC` (batch '
                              '- features) dimension order, '
                              'got spec = %s.' % spec)

  # Flatten inputs if marginalizing over everything.
  if cross == marginal == M.OVER_ALL:
    n1 = x1.shape[batch_axis]
    x1 = np.reshape(np.moveaxis(x1, batch_axis, 0), (n1, -1))
    if x2 is not None:
      n2 = x2.shape[batch_axis]
      x2 = np.reshape(np.moveaxis(x2, batch_axis, 0), (n2, -1))

    batch_axis, channel_axis = 0, 1

  else:
    if spec and 'C' not in spec:
      raise NotImplementedError('A separate channel dimension `C` is required, '
                                'got spec = %s.' % spec)
    channel_axis = spec.index('C') if spec else x1.ndim - 1

  # TODO(schsam, romann): Think more about dtype automatic vs manual dtype
  # promotion.
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
  is_gaussian = False
  is_height_width = True
  is_input = True

  return Kernel(var1, nngp, var2, ntk, is_gaussian, is_height_width,
                marginal, cross, x1.shape, x2.shape, x1_is_x2, is_input)


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

  deps = getattr(kernel_fn, _COVARIANCES_REQ, _DEFAULT_MARGINALIZATION)
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
                    marginalization='auto'):
    """Returns the `Kernel` resulting from applying `ker_fun` to given inputs.

    Args:
      x1_or_kernel: either a `np.ndarray` with shape
        `[batch_size_1] + input_shape`, or a `Kernel`.
      x2: an optional `np.ndarray` with shape `[batch_size_2] + input_shape`.
        `None` means `x2 == x1` or `x1_or_kernel is Kernel`.
      get: either `None`, a string, or a tuple of strings specifying which data
        should be returned by the kernel function. Can be "nngp", "ntk", "var1",
        "var2", "is_gaussian", "is_height_width", "marginal", "cross".
      marginalization: Either a string with value "auto" or "none" or a dict.
        If "auto" then stax attempts to automatically identify which
        dimensions are most appropriate to marginalize over. If "none" then no
        marginalization is performed. If a dict then the user can manually
        specify the marginalization for the self- and cross- correlations.
    Returns:
      If `get` is a string, returns the requested `np.ndarray`. If `get` is a
      tuple, returns an `AnalyticKernel` namedtuple containing only the
      requested information. If `get` is None then a Kernel object is returned
      containing all the data.
    """

    if (isinstance(x1_or_kernel, Kernel) or
        (isinstance(x1_or_kernel, list) and
         all(isinstance(k, Kernel) for k in x1_or_kernel))):
      _check_marginalization(kernel_fn, x1_or_kernel)
      return _apply_kernel(init_fn, kernel_fn, x1_or_kernel)
    return outer_kernel_fn(x1_or_kernel, x2, get, marginalization)

  @utils.get_namedtuple('AnalyticKernel')
  def outer_kernel_fn(x1, x2, get, marginalization):
    if not isinstance(x1, np.ndarray):
      raise TypeError('Inputs to a kernel propagation function should be '
                      'a `Kernel`, '
                      'a `list` of `Kernel`s, '
                      'or a (tuple of) `np.ndarray`(s), got %s.' % type(x1))

    if not (x2 is None or isinstance(x2, np.ndarray)):
      raise TypeError('`x2` to a kernel propagation function '
                      'should be `None` or a `np.ndarray`, got %s.'
                      % type(x2))

    covs_req = getattr(
        kernel_fn, _COVARIANCES_REQ, _DEFAULT_MARGINALIZATION)
    if marginalization != 'auto':
      if isinstance(marginalization, dict):
        for k in covs_req:
          if marginalization[k] > covs_req[k]:
            covs_req[k] = marginalization[k]
      elif marginalization == 'none' and x1.ndim > 2:
        covs_req = {'marginal': M.OVER_POINTS, 'cross': M.NO}
      else:
        raise NotImplementedError(
            ('marginalization should be set to one of "auto", "none", or a dict'
             'specifying the marginalization levels of the variance and the '
             'covariance respectively. Found {}.'.format(marginalization)))

    compute_ntk = (get is None) or ('ntk' in get)
    kernel = _inputs_to_kernel(x1, x2, compute_ntk=compute_ntk, **covs_req)
    return _apply_kernel(init_fn, kernel_fn, kernel)

  if hasattr(kernel_fn, _COVARIANCES_REQ):
    setattr(new_kernel_fn,
            _COVARIANCES_REQ,
            getattr(kernel_fn, _COVARIANCES_REQ))

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
  init_fn, apply_fn = ostax.elementwise(fn, **fn_kwargs)
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

  var = np.swapaxes(var, -3, -2)  # [..., X, X, Y, Y] -> [..., X, Y, X, Y]
  X, Y = var.shape[-2:]

  sqnorms = np.diagonal(var.reshape((-1, X * Y, X * Y)), axis1=-2, axis2=-1)
  sqnorms = sqnorms.reshape((-1,) + (X, Y))

  return sqnorms


def _get_normalising_prod(var1, var2, marginal, axis=()):
  """Returns three tensors, `prod11`, `prod12` and `prod22` which contain
  products of marginal variances of `var1`, `nngp` and `var2` respectively.

  `prod12` is a 6D tensor where an entry [x1, x2, a, b, c, d] equals
  k_{ab}(x1, x1) * k_{cd}(x2, x2), if `marginal` is `OVER_POINTS` or `NO`,
  or a 4D tensor k_{aa}(x1, x1) k_{cc}(x2, x2) if `marginal` is `OVER_PIXELS`,
  or a 2D tensor k(x1, x1) k(x2, x2) if `marginal` is `OVER_ALL`. In the last
  two cases, both `prod11` and `prod22` will be None. Otherwise they will be
  5D tensors k_{ab}(x1, x1) k_{cd}(x1, x1) in the `marginal == OVER_POINTS`
  case, or 6D tensors akin to the one for `prod12` if `marginal == NO`.
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
    def outer_prod_full(sqnorms1, sqnorms2):
      sqnorms1 = sqnorms1[:, None, :, None, :, None]
      sqnorms2 = sqnorms2[None, :, None, :, None, :]
      return sqnorms1 * sqnorms2

    sqnorms1 = _get_dimensionwise_marg_var(var1, marginal)
    sqnorms1 = np.mean(sqnorms1, axis=axis, keepdims=True)
    if same_input:
      sqnorms2 = sqnorms1
    else:
      sqnorms2 = _get_dimensionwise_marg_var(var2, marginal)
      sqnorms2 = np.mean(sqnorms2, axis=axis, keepdims=True)

    prod12 = outer_prod_full(sqnorms1, sqnorms2)

    if marginal == M.OVER_POINTS:
      def outer_prod_pix(sqnorms1, sqnorms2):
        sqnorms1 = sqnorms1[:, :, None, :, None]
        sqnorms2 = sqnorms2[:, None, :, None, :]
        return sqnorms1 * sqnorms2

      prod11 = outer_prod_pix(sqnorms1, sqnorms1)
      prod22 = outer_prod_pix(sqnorms2, sqnorms2) if not same_input else prod11
    else:
      prod11 = outer_prod_full(sqnorms1, sqnorms1)
      prod22 = outer_prod_full(sqnorms2, sqnorms2) if not same_input else prod11
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
  # TODO(xlc): Monte Carlo approximation to the integral (suggested by schsam.)
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
          parameterization='ntk'):
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
        W_ij x_j + b_std b_i Under standard parameterization, weights and biases
        are initialized as W_ij ~ N(0,W_std^2/[width]), b_i ~ N(0,b_std^2), and
        the finite width layer equation is z_i = \sum_j W_ij x_j + b_i.


  Returns:
    `(init_fn, apply_fn, kernel_fn)`.
  """
  # TODO(jaschasd) after experimentation, evaluate whether to change default
  # parameterization from "ntk" to "standard"

  parameterization = parameterization.lower()

  ntk_init_fn, _ = ostax.Dense(out_dim, W_init, b_init)

  def standard_init_fn(rng, input_shape):
    output_shape, (W, b) = ntk_init_fn(rng, input_shape)
    return output_shape, (W * W_std / np.sqrt(input_shape[-1]), b * b_std)

  if parameterization == 'ntk':
    init_fn = ntk_init_fn
  elif parameterization == 'standard':
    init_fn = standard_init_fn
  else:
    raise ValueError('Parameterization not supported: %s' % parameterization)

  def apply_fn(params, inputs, **kwargs):
    W, b = params
    if parameterization == 'ntk':
      norm = W_std / np.sqrt(inputs.shape[-1])
      return norm * np.dot(inputs, W) + b_std * b
    elif parameterization == 'standard':
      return np.dot(inputs, W) + b

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

  setattr(kernel_fn, _COVARIANCES_REQ, {'marginal': M.OVER_ALL,
                                        'cross': M.OVER_ALL})
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
  init_fn, apply_fn = ostax.FanInSum
  kernel_fn = lambda kernels: _fan_in_kernel_fn(kernels, axis=None, spec=None)
  return init_fn, apply_fn, kernel_fn


@_layer
def FanInConcat(axis=-1, spec=None):
  """Layer construction function for a fan-in concatenation layer.

  Based on `jax.experimental.stax.FanInConcat`.
  """
  init_fn, apply_fn = ostax.FanInConcat(axis)
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
  n_height_width = sum(ker.is_height_width for ker in kernels)

  if n_height_width == n_kernels:
    is_height_width = True

  elif n_height_width == 0:
    is_height_width = False

  elif n_height_width >= n_kernels / 2:
    is_height_width = True
    for i in range(n_kernels):
      if not kernels[i].is_height_width:
        kernels[i] = _flip_height_width(kernels[i])

  else:
    is_height_width = False
    for i in range(n_kernels):
      if kernels[i].is_height_width:
        kernels[i] = _flip_height_width(kernels[i])

  axis = None if axis is None else range(len(shape1))[axis]
  batch_axis = 0 if spec is None else spec.index('N')
  channel_axis = (len(shape1) - 1) if spec is None else spec.index('C')

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
    if not is_gaussian:
      raise NotImplementedError('`FanInSum` or `FanInConcat` layer along the '
                                'non-channel axis is only implemented for the '
                                'case if all input layers guaranteed to be mean'
                                '-zero Gaussian, i.e. having all `is_gaussian '
                                'set to `True`.')
  else:
    # TODO(romann): allow to apply nonlinearity after channelwise concatenation.
    is_gaussian = False

  # Warnings.
  warnings.warn('`FanIn` layers assume independent inputs which is not verified '
                'in the code. Please make sure to have at least one Dense or '
                'CNN layer in each branch.')
  if axis == batch_axis:
    warnings.warn('Concatenation along the batch axis (%d) gives inconsistent'
                  ' covariances when batching - proceed with caution.' % axis)

  spatial_axes = [i for i in range(len(shape1))
                  if i not in (channel_axis, batch_axis)]
  # Flip spatial axis if `is_height_width` is `False`.
  if axis in spatial_axes:
    axis = spatial_axes[(spatial_axes.index(axis) + ~is_height_width)
                        % len(spatial_axes)]

  # Map activation tensor axis to the covariance tensor axis.
  kernel_axis = {
      **{
          None: None,
          batch_axis: 0,
          channel_axis: -1,
      },
      **{
          spatial_axis: idx + 1 for idx, spatial_axis in enumerate(spatial_axes)
      }
  }[axis]

  var1 = _concat_covariance([k.var1 for k in kernels], kernel_axis, marginal)
  var2 = _concat_covariance([k.var2 for k in kernels], kernel_axis, marginal)
  nngp = _concat_covariance([k.nngp for k in kernels], kernel_axis, cross)
  ntk = _concat_covariance([k.ntk for k in kernels], kernel_axis, cross)
  kers = (var1, nngp, var2, ntk)

  return Kernel(*(
      kers + (is_gaussian, is_height_width, marginal, cross, None, None,
              kernels[0].x1_is_x2, kernels[0].is_input)))


def _flip_height_width(kernels):
  """Flips the order of spatial axes in the covariance matrices.

  Args:
    kernels: a `Kernel` object.

  Returns:
    A `Kernel` object with `height` and `width` axes order flipped in
    all covariance matrices. For example, if `kernels.nngp` has shape
    `[batch_size_1, batch_size_2, height, height, width, width]`, then
    `_flip_height_width(kernels).nngp` has shape
    `[batch_size_1, batch_size_2, width, width, height, height]`.
  """
  var1, nngp, var2, ntk, is_height_width, marginal, cross = (
      kernels.var1, kernels.nngp, kernels.var2, kernels.ntk,
      kernels.is_height_width, kernels.marginal, kernels.cross)

  def flip_5or6d(mat):
    return np.moveaxis(mat, (-2, -1), (-4, -3))

  if marginal == M.OVER_PIXELS:
    var1 = np.transpose(var1, (0, 2, 1))
    var2 = np.transpose(var2, (0, 2, 1)) if var2 is not None else var2
  elif marginal in [M.OVER_POINTS, M.NO]:
    var1 = flip_5or6d(var1)
    var2 = flip_5or6d(var2) if var2 is not None else var2
  else:
    raise NotImplementedError(
        "Only implemented for `OVER_PIXELS`, `OVER_POINTS` and `NO`;"
        " supplied {}".format(marginal))

  if cross == M.OVER_PIXELS:
    nngp = np.moveaxis(nngp, -1, -2)
    ntk = np.moveaxis(ntk, -1, -2) if _is_array(ntk) else ntk
  elif cross in [M.OVER_POINTS, M.NO]:
    nngp = flip_5or6d(nngp)
    ntk = flip_5or6d(ntk) if _is_array(ntk) else ntk

  return kernels._replace(var1=var1, nngp=nngp, var2=var2, ntk=ntk,
                          is_height_width=not is_height_width)


def _concat_covariance(mats, axis, marginalisation):
  """Compute the covariance of concatenated activations with given covariances.

  Args:
    mats: a list of `np.ndarrray` covariance tensors of the same shape.
    axis: an `int` along which the covariances (not activations) are
      concatenated. `None` corresponds to sum, `-1` to averaging.
    marginalisation: a single `Kernel.Marginalisation` of all covariance
      matrices.

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
    mat = sum(mats) / n_mats

  # Simple concatenation along the axis if the axis is not duplicated.
  elif ((axis != 0 and marginalisation == M.OVER_PIXELS) or
        (axis == 0 and mat_ndim % 2)):
    concat_axis = axis + 1 - mat_ndim % 2
    mat = np.concatenate(mats, concat_axis)

  # 2D concatenation with insertion of 0-blocks if the axis is present twice.
  elif axis == 0 or marginalisation in (M.NO, M.OVER_POINTS):
    rows = []
    pad_axis = max(0, 2 * axis - mat_ndim % 2)
    for i, mat in enumerate(mats):
      pads = [(0, 0)] * mat_ndim
      pads[pad_axis] = (
          sum(mats[j].shape[pad_axis] for j in range(i)),
          sum(mats[j].shape[pad_axis] for j in range(i + 1, n_mats))
      )
      rows.append(np.pad(mat, pads))
    mat = np.concatenate(rows, pad_axis + 1)

  else:
    raise NotImplementedError('Asked to concatenate along axis %d given '
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

  _set_covariances_req_attr(kernel_fn, kernel_fns)
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

  _set_covariances_req_attr(kernel_fn, kernel_fns)
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
  if not _is_array(x):
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


def _conv_nngp_5or6d_double_conv(mat, filter_shape, strides, padding):
  """Compute covariance of the CNN outputs given inputs with covariance `nngp`.

  Uses 2D convolution and works on any hardware platform.

  Args:
    mat: a 5D or 6D `np.ndarray` containing sample-(sample-)pixel-pixel
      covariances. Has shape
      `[batch_size_1, (batch_size_2,) height, height, width, width]`.
    filter_shape: tuple of positive integers, the convolutional filters spatial
      shape (e.g. `(3, 3)` for a 2D convolution).
    strides: tuple of positive integers, the CNN strides (e.g. `(1, 1)` for a
      2D convolution).
    padding: a `Padding` enum, e.g. `Padding.CIRCULAR`.

  Returns:
    a 5D or 6D `np.ndarray` containing sample-(sample-)pixel-pixel covariances
    of CNN outputs. Has shape
    `[batch_size_1, (batch_size_2,) new_width, new_width,
      new_height, new_height]`.
  """
  if padding == Padding.CIRCULAR:
    pixel_axes = tuple(range(mat.ndim)[-4:])
    mat = _same_pad_for_filter_shape(
        mat,
        _double_tuple(filter_shape),
        _double_tuple(strides),
        pixel_axes,
        'wrap'
    )
    padding = Padding.VALID

  data_dim, X, Y = mat.shape[:-4], mat.shape[-3], mat.shape[-1]
  filter_x, filter_y = filter_shape
  stride_x, stride_y = strides

  ker_y = np.diag(np.full((filter_y,), 1. / filter_y, mat.dtype))
  ker_y = np.reshape(ker_y, (filter_y, filter_y, 1, 1))

  channel_axis = _CONV_QAB_DIMENSION_NUMBERS[0].index('C')
  mat = lax.conv_general_dilated(
      np.expand_dims(mat.reshape((-1, Y, Y)), channel_axis),
      ker_y, (stride_y, stride_y), padding.name,
      dimension_numbers=_CONV_QAB_DIMENSION_NUMBERS)
  out_Y = mat.shape[-2]
  mat = mat.reshape(data_dim + (X, X, out_Y, out_Y))

  ker_x = np.diag(np.full((filter_x,), 1. / filter_x, mat.dtype))
  ker_x = np.reshape(ker_x, (filter_x, filter_x, 1, 1))

  mat = np.moveaxis(mat, (-2, -1), (-4, -3))
  mat = lax.conv_general_dilated(
      np.expand_dims(mat.reshape((-1, X, X)), channel_axis),
      ker_x, (stride_x, stride_x), padding.name,
      dimension_numbers=_CONV_QAB_DIMENSION_NUMBERS)
  out_X = mat.shape[-2]
  mat = mat.reshape(data_dim + (out_Y, out_Y, out_X, out_X))

  return mat


def _conv_nngp_4d(nngp, filter_shape, strides, padding):
  """Compute covariance of the CNN outputs given inputs with covariance `nngp`.

  Uses 2D convolution and works on any platform, but only works with
    sample-sample-(same pixel) covariances.

  Args:
    nngp: a 4D `np.ndarray` containing sample-sample-(same pixel) covariances.
      Has shape `[batch_size_1, batch_size_2, height, width]`.
    filter_shape: tuple of positive integers, the convolutional filters spatial
      shape (e.g. `(3, 3)` for a 2D convolution).
    strides: tuple of positive integers, the CNN strides (e.g. `(1, 1)` for a
      2D convolution).
    padding: a `Padding` enum, e.g. `Padding.CIRCULAR`.

  Returns:
    a 4D `np.ndarray` containing sample-sample-pixel-pixel covariances of CNN
      outputs. Has shape `[batch_size_1, batch_size_2, new_height, new_width]`.
  """
  if padding == Padding.CIRCULAR:
    nngp = _same_pad_for_filter_shape(nngp, filter_shape, strides,
                                      (2, 3), 'wrap')
    padding = Padding.VALID

  ker_nngp = np.full(filter_shape + (1, 1), 1. / np.prod(filter_shape),
                     nngp.dtype)

  channel_axis = _CONV_QAB_DIMENSION_NUMBERS[0].index('C')
  batch1, batch2, X, Y = nngp.shape
  nngp = np.reshape(nngp, (-1, X, Y))
  nngp = np.expand_dims(nngp, channel_axis)
  nngp = lax.conv_general_dilated(nngp, ker_nngp, strides, padding.name,
                                  dimension_numbers=_CONV_QAB_DIMENSION_NUMBERS)
  nngp = np.squeeze(nngp, channel_axis)
  nngp = nngp.reshape((batch1, batch2,) + nngp.shape[1:])
  return nngp


def _conv_var_3d(var1, filter_shape, strides, padding):
  """Compute variances of the CNN outputs given inputs with variances `var1`.

  Args:
    var1: a 3D `np.ndarray` containing sample-pixel variances.
      Has shape `[batch_size, height, width]`.
    filter_shape: tuple of positive integers, the convolutional filters spatial
      shape (e.g. `(3, 3)` for a 2D convolution).
    strides: tuple of positive integers, the CNN strides (e.g. `(1, 1)` for a
      2D convolution).
    padding: a `Padding` enum, e.g. `Padding.CIRCULAR`.

  Returns:
    a 3D `np.ndarray` containing sample-pixel variances of CNN layer outputs.
      Has shape `[batch_size, new_height, new_width]`.
  """
  if var1 is None:
    return var1

  if padding == Padding.CIRCULAR:
    var1 = _same_pad_for_filter_shape(var1, filter_shape, strides,
                                      (1, 2), 'wrap')
    padding = Padding.VALID

  channel_axis = _CONV_QAB_DIMENSION_NUMBERS[0].index('C')
  var1 = np.expand_dims(var1, channel_axis)
  ker_var1 = np.full(filter_shape + (1, 1), 1. / np.prod(filter_shape),
                     var1.dtype)
  var1 = lax.conv_general_dilated(var1, ker_var1, strides, padding.name,
                                  dimension_numbers=_CONV_QAB_DIMENSION_NUMBERS)
  var1 = np.squeeze(var1, channel_axis)
  return var1


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

  lhs_spec, rhs_spec, out_spec = dimension_numbers

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

  def apply_fn(params, inputs, **kwargs):
    W, b = params

    if parameterization == 'ntk':
      norm = W_std / np.sqrt(input_total_dim(inputs.shape))
      b_rescale = b_std
    elif parameterization == 'standard':
      norm = 1.
      b_rescale = 1.

    apply_padding = padding
    if padding == Padding.CIRCULAR:
      apply_padding = Padding.VALID
      non_spatial_axes = (dimension_numbers[0].index('N'),
                          dimension_numbers[0].index('C'))
      spatial_axes = tuple(i for i in range(inputs.ndim)
                           if i not in non_spatial_axes)
      inputs = _same_pad_for_filter_shape(inputs, filter_shape, strides,
                                          spatial_axes, 'wrap')

    return norm * lax.conv_general_dilated(
        inputs,
        W,
        strides,
        apply_padding.name,
        dimension_numbers=dimension_numbers) + b_rescale * b

  def kernel_fn(kernels):
    """Compute the transformed kernels after a conv layer."""
    var1, nngp, var2, ntk, is_height_width, marginal, cross = (
        kernels.var1, kernels.nngp, kernels.var2, kernels.ntk,
        kernels.is_height_width, kernels.marginal, kernels.cross)

    if cross > M.OVER_PIXELS and not is_height_width:
      filter_shape_nngp = filter_shape[::-1]
      strides_nngp = strides[::-1]
    else:
      filter_shape_nngp = filter_shape
      strides_nngp = strides

    if cross == M.OVER_PIXELS:

      def conv_nngp_unscaled(x):
        if _is_array(x):
          x = _conv_nngp_4d(x, filter_shape_nngp, strides_nngp, padding)
        return x
    elif cross in [M.OVER_POINTS, M.NO]:

      def conv_nngp_unscaled(x):
        if _is_array(x):
          x = _conv_nngp_5or6d_double_conv(x, filter_shape_nngp,
                                           strides_nngp, padding)
        return x

      is_height_width = not is_height_width
    else:
      raise NotImplementedError(
          "Only implemented for `OVER_PIXELS`, `OVER_POINTS` and `NO`;"
          " supplied {}".format(cross))

    def conv_nngp(x):
      x = conv_nngp_unscaled(x)
      return _affine(x, W_std, b_std)

    if marginal == M.OVER_PIXELS:
      def conv_var(x):
        x = _conv_var_3d(x, filter_shape_nngp, strides_nngp, padding)
        x = _affine(x, W_std, b_std)
        return x
    elif marginal in [M.OVER_POINTS, M.NO]:
      def conv_var(x):
        if _is_array(x):
          x = _conv_nngp_5or6d_double_conv(x, filter_shape_nngp,
                                           strides_nngp, padding)
        x = _affine(x, W_std, b_std)
        return x
    else:
      raise NotImplementedError(
          "Only implemented for `OVER_PIXELS`, `OVER_POINTS` and `NO`;"
          " supplied {}".format(marginal))

    var1 = conv_var(var1)
    var2 = conv_var(var2)

    if parameterization == 'ntk':
      nngp = conv_nngp(nngp)
      ntk = conv_nngp(ntk) + nngp - b_std**2 if ntk is not None else ntk
    elif parameterization == 'standard':
      nngp_unscaled = conv_nngp_unscaled(nngp)
      if ntk is not None:
        ntk = (
            input_total_dim(kernels.shape1) * nngp_unscaled + 1. +
            W_std**2 * conv_nngp_unscaled(ntk))
      nngp = _affine(nngp_unscaled, W_std, b_std)

    return kernels._replace(
        var1=var1, nngp=nngp, var2=var2, ntk=ntk, is_gaussian=True,
        is_height_width=is_height_width, marginal=marginal, cross=cross,
        is_input=False)

  setattr(kernel_fn, _COVARIANCES_REQ, {'marginal': M.OVER_PIXELS,
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
  return GeneralConv(('NHWC', 'HWIO', 'NHWC'),
                     out_chan,
                     filter_shape,
                     strides,
                     padding,
                     W_std,
                     W_init,
                     b_std,
                     b_init,
                     parameterization)


def _average_pool_nngp_5or6d(mat, window_shape, strides, padding,
                             normalize_edges):
  """Get covariances of average pooling outputs given inputs covariances `mat`.

  Args:
    mat: a 5D or 6D `np.ndarray` containing sample-(sample-)pixel-pixel
      covariances. Has shape
      `[batch_size_1, (batch_size_2,) height, height, width, width]`.
    window_shape: tuple of two positive integers, the pooling spatial shape
      (e.g. `(3, 3)`).
    strides: tuple of two positive integers, the pooling strides, e.g. `(1, 1)`.
    padding: a `Padding` enum, e.g. `Padding.CIRCULAR`.
    normalize_edges: `True` to normalize output by the effective receptive
      field, `False` to normalize by the window size. Only has effect at the
      edges when `SAME` padding is used. Set to `True` to retain correspondence
      to `ostax.AvgPool`.

  Returns:
    a 5D or 6D `np.ndarray` containing sample-(sample-)pixel-pixel covariances
    of the average pooling outputs. Has shape
    `[batch_size_1, (batch_size_2,) new_height, new_height,
      new_width, new_width]`.
  """
  if not _is_array(mat):
    return mat

  if padding == Padding.CIRCULAR:
    pixel_axes = tuple(range(mat.ndim)[-4:])
    mat = _same_pad_for_filter_shape(mat, _double_tuple(window_shape),
                                     _double_tuple(strides), pixel_axes, 'wrap')
    padding = Padding.VALID

  window_shape = (1,) * (mat.ndim - 4) + _double_tuple(window_shape)
  strides = (1,) * (mat.ndim - 4) + _double_tuple(strides)

  nngp_out = lax.reduce_window(mat, 0., lax.add, window_shape, strides,
                               padding.name)

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
def AvgPool(window_shape,
            strides=None,
            padding=Padding.VALID.name,
            spec='NHWC',
            normalize_edges=True):
  """Layer construction function for a 2D average pooling layer.

  Based on `jax.experimental.stax.AvgPool`. Has a similar API apart from:

  Args:
    padding: in addition to `VALID` and `SAME' padding, supports `CIRCULAR`,
      not available in `jax.experimental.stax.GeneralConv`.
    spec: an optional `string`, specifying the dimension order of
      the input, e.g. `NCHW` or `NHWC`.
    normalize_edges: `True` to normalize output by the effective receptive
      field, `False` to normalize by the window size. Only has effect at the
      edges when `SAME` padding is used. Set to `True` to retain correspondence
      to `ostax.AvgPool`.
  """
  strides = strides or (1,) * len(window_shape)
  padding = Padding(padding)

  if padding == Padding.CIRCULAR:
    init_fn, _ = ostax.AvgPool(window_shape, strides, Padding.SAME.name,
                               spec)
    _, apply_fn_0 = ostax.AvgPool(window_shape, strides, Padding.VALID.name,
                                  spec)

    def apply_fn(params, inputs, **kwargs):
      non_spatial_axes = (spec.index('N'), spec.index('C'))
      spatial_axes = tuple(i for i in range(inputs.ndim)
                           if i not in non_spatial_axes)
      inputs = _same_pad_for_filter_shape(inputs, window_shape, strides,
                                          spatial_axes, 'wrap')
      res = apply_fn_0(params, inputs, **kwargs)
      return res

  elif normalize_edges:
    init_fn, apply_fn = ostax.AvgPool(window_shape, strides, padding.name, spec)

  else:
    def rescaler(*args, **kwargs):
      del args, kwargs  # Unused.
      return lambda outputs, _inputs, _spec: outputs / np.prod(window_shape)

    avgPool = ostax._pooling_layer(lax.add, 0., rescaler)
    init_fn, apply_fn = avgPool(window_shape, strides, padding.name, spec)

  def kernel_fn(kernels):
    """Kernel transformation."""
    var1, nngp, var2, ntk, is_gaussian, is_height_width, marginal, cross = (
        kernels.var1, kernels.nngp, kernels.var2, kernels.ntk,
        kernels.is_gaussian, kernels.is_height_width, kernels.marginal,
        kernels.cross)

    if is_height_width:
      window_shape_nngp = window_shape
      strides_nngp = strides
    else:
      window_shape_nngp = window_shape[::-1]
      strides_nngp = strides[::-1]

    nngp = _average_pool_nngp_5or6d(nngp, window_shape_nngp, strides_nngp,
                                    padding, normalize_edges)
    ntk = _average_pool_nngp_5or6d(ntk, window_shape_nngp, strides_nngp,
                                   padding, normalize_edges)
    var1 = _average_pool_nngp_5or6d(var1, window_shape_nngp, strides_nngp,
                                    padding, normalize_edges)
    if var2 is not None:
      var2 = _average_pool_nngp_5or6d(var2, window_shape_nngp, strides_nngp,
                                      padding, normalize_edges)

    return kernels._replace(
        var1=var1, nngp=nngp, var2=var2, ntk=ntk, is_gaussian=is_gaussian,
        is_height_width=is_height_width, marginal=marginal, cross=cross)

  setattr(kernel_fn, _COVARIANCES_REQ, {'marginal': M.OVER_POINTS,
                                        'cross': M.NO,
                                        'spec': spec})
  return init_fn, apply_fn, kernel_fn


@_layer
def GlobalAvgPool(spec='NHWC'):
  """Layer construction function for a global average pooling layer.

  Pools over and removes (`keepdims=False`) all spatial dimensions, making the
    batch dimension (`N`) leading.

  Args:
    spec: an optional `string`, specifying the dimension order of
      the input, e.g. `NCHW` or `NHWC`.
  """
  non_spatial_axes = (spec.index('N'), spec.index('C'))

  def init_fn(rng, input_shape):
    output_shape = tuple(input_shape[i] for i in non_spatial_axes)
    return output_shape, ()

  def apply_fn(params, inputs, **kwargs):
    spatial_axes = tuple(i for i in range(inputs.ndim)
                       if i not in non_spatial_axes)
    out = np.mean(inputs, axis=spatial_axes, keepdims=True)
    out = np.moveaxis(out, non_spatial_axes, (0, 1))
    out = np.squeeze(out, range(inputs.ndim)[len(non_spatial_axes):])
    return out

  def kernel_fn(kernels):
    var1, nngp, var2, ntk, is_gaussian, marginal, cross = (
        kernels.var1, kernels.nngp, kernels.var2, kernels.ntk,
        kernels.is_gaussian, kernels.marginal, kernels.cross)

    def _average_pool(ker_mat):
      spatial_axes = tuple(range(ker_mat.ndim)[-4:])
      return np.mean(ker_mat, axis=spatial_axes)

    nngp = _average_pool(nngp)
    ntk = _average_pool(ntk) if _is_array(ntk) else ntk
    var1 = _average_pool(var1)
    if var2 is not None:
      var2 = _average_pool(var2)

    return kernels._replace(
        var1=var1, nngp=nngp, var2=var2, ntk=ntk, is_gaussian=is_gaussian,
        is_height_width=True, marginal=M.OVER_ALL, cross=M.OVER_ALL)

  setattr(kernel_fn, _COVARIANCES_REQ, {'marginal': M.OVER_POINTS,
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
  batch_axis = spec.index('N') if spec else 0

  def get_output_shape(input_shape):
    return (
        input_shape[batch_axis],
        functools.reduce(op.mul,
            input_shape[:batch_axis] + input_shape[batch_axis + 1:],
            1))

  def init_fn(rng, input_shape):
    output_shape = get_output_shape(input_shape)
    return output_shape, ()

  def apply_fn(params, inputs, **kwargs):
    output_shape = get_output_shape(inputs.shape)
    inputs = np.moveaxis(inputs, batch_axis, 0)
    return np.reshape(inputs, output_shape)

  def kernel_fn(kernels):
    """Compute kernels."""
    var1, nngp, var2, ntk, is_gaussian, marginal, cross = (
        kernels.var1, kernels.nngp, kernels.var2, kernels.ntk,
        kernels.is_gaussian, kernels.marginal, kernels.cross)

    if nngp.ndim == 2:
      return kernels

    def trace(x):
      count = x.shape[-4] * x.shape[-2]
      y = np.trace(x, axis1=-2, axis2=-1)
      z = np.trace(y, axis1=-2, axis2=-1)
      return z / count

    if marginal == M.OVER_PIXELS:
      var1 = np.mean(var1, axis=(1, 2))
      var2 = var2 if var2 is None else np.mean(var2, axis=(1, 2))

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
      nngp = np.mean(nngp, axis=(2, 3))
      ntk =  np.mean(ntk, axis=(2, 3)) if _is_array(ntk) else ntk

    elif cross in [M.OVER_POINTS, M.NO]:
      nngp = trace(nngp)
      ntk = trace(ntk) if _is_array(ntk) else ntk

    elif cross != M.OVER_ALL:
      raise NotImplementedError(
          "Only implemented for , `OVER_ALL`, `OVER_PIXELS`, `OVER_POINTS` and "
          "`NO`; supplied {}".format(cross))

    return kernels._replace(
        var1=var1, nngp=nngp, var2=var2, ntk=ntk, is_gaussian=is_gaussian,
        is_height_width=True, marginal=M.OVER_ALL, cross=M.OVER_ALL)

  setattr(kernel_fn, _COVARIANCES_REQ, {'marginal': M.OVER_ALL,
                                        'cross': M.OVER_ALL,
                                        'spec': spec})
  return init_fn, apply_fn, kernel_fn


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
                        spec='NHWC'):
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
    spec: a string specifying ordering of the input dimensions, e.g.,
      `'NHWC'` for `[batch_size, height, width, channels] or `'NCHW'` for
      `[batch_size, channels, height, width]`.

  Warnings:
    Currently only works with image data.

  Raises:
    NotImplementedError: If `fixed` is `False`, call to `kernel_fn` will result
    in an error as there is no known analytic expression for the kernel.
  """
  # TODO(jaschasd) incorporate parameterization keyword argument for attention
  # (the details of this will require some thought. I believe it will be a no-op
  # for 1/n scaling (ie fixed scaling), but will change the NTK kernel for
  # 1/sqrt(n) scaling. Will also need to decide whether the 1/sqrt(n) scaling
  # should be absorbed into one or both of the key and value weight
  # initializations

  # TODO(romann): optimize different `spec` values to minimize transpositions.

  OV_gain = W_out_std * W_value_std
  QK_gain = W_query_std * W_key_std
  QK_prod_scaling = float(n_chan_key if fixed else n_chan_key**0.5)
  batch_axis, channel_axis = spec.index('N'), spec.index('C')

  def init_fn(rng, input_shape):
    n_chan_in = input_shape[channel_axis]
    output_shape = (input_shape[:channel_axis] + (n_chan_out,) +
                    input_shape[channel_axis + 1:])

    rng_Q, rng_K, rng_V, rng_O, rng_b = random.split(rng, 5)
    key_matrices = W_key_init(rng_K, shape=(n_heads, n_chan_in, n_chan_key))
    val_matrices = W_value_init(rng_V, shape=(n_heads, n_chan_in, n_chan_val))
    W_out = W_out_init(rng_O, shape=(n_chan_val * n_heads, n_chan_out))
    b = b_init(rng_b, shape=(n_chan_out,))

    if fixed:
      query_matrices = None
      warnings.warn("Fixed attention used -> `W_query_init` ignored, tying"
                    " the weights (see docstring for more details).")
    else:
      query_matrices = W_query_init(rng_Q,
                                    shape=(n_heads, n_chan_in, n_chan_key))

    return output_shape, (query_matrices, key_matrices, val_matrices, W_out, b)

  def apply_fn(params, inputs, **kwargs):
    query_matrices, key_matrices, val_matrices, W_out, b = params
    n = inputs.shape[batch_axis]
    n_chan_in = inputs.shape[channel_axis]
    spatial_shape = tuple(s for i, s in enumerate(inputs.shape)
                          if i not in (batch_axis, channel_axis))

    inputs = np.moveaxis(inputs, (batch_axis, channel_axis), (0, -1))
    inputs = inputs.reshape((n, -1, n_chan_in))

    def _inputs_dot(matrices, std):
      ret = np.dot(inputs, std * matrices / np.sqrt(n_chan_in))
      return np.moveaxis(ret, 2, 0)

    keys = _inputs_dot(key_matrices, W_key_std)
    values = _inputs_dot(val_matrices, W_value_std)
    if fixed:
      queries = keys * W_query_std / W_key_std
    else:
      queries = _inputs_dot(query_matrices, W_query_std)

    G_mat  = np.matmul(queries, np.moveaxis(keys, -1, -2))
    G_mat /= QK_prod_scaling
    G_mat = ostax.softmax(G_mat, axis=-1)

    heads = np.matmul(G_mat, values)
    heads = np.moveaxis(heads, 0, -1)
    heads = np.reshape(heads, heads.shape[:-2] + (-1,))

    ret = np.matmul(heads, W_out_std * W_out / np.sqrt(n_chan_val * n_heads))
    ret = np.reshape(ret, (n,) + spatial_shape + (n_chan_out,)) + b_std * b
    ret = np.moveaxis(ret, (0, -1), (batch_axis, channel_axis))
    return ret

  def kernel_fn(kernels):
    var1, nngp, var2, ntk, is_height_width, marginal, cross = (
        kernels.var1, kernels.nngp, kernels.var2, kernels.ntk,
        kernels.is_height_width, kernels.marginal, kernels.cross)

    if not fixed:
      # TODO(jirihron): implement the approximation and raise a warning
      raise NotImplementedError("No known closed form expression.")

    def _get_G_softmax(mat):
      if marginal == M.NO:
        mat = np.moveaxis(np.diagonal(mat, axis1=0, axis2=1), -1, 0)
      axes = range(mat.ndim)
      return ostax.softmax(QK_gain * mat, axis=(axes[-3], axes[-1]))

    def _transform_kernel(mat, G1, G2=None):
      if not _is_array(mat):
        return mat

      G2 = G1 if G2 is None else G2
      if mat.ndim == 5:
        pattern = 'xacbd,xcedf,xgehf->xagbh'
      else:
        pattern = 'xacbd,xycedf,ygehf->xyagbh'
      return _affine(np.einsum(pattern, G1, mat, G2), OV_gain, b_std)

    G1 = _get_G_softmax(var1)
    G2 = _get_G_softmax(var2) if var2 is not None else G1

    var1 = _transform_kernel(var1, G1)
    var2 = _transform_kernel(var2, G2) if var2 is not None else var2
    nngp = _transform_kernel(nngp, G1, G2)
    ntk = (_transform_kernel(ntk, G1, G2) + 2 * (nngp - b_std**2)
           if ntk is not None else ntk)

    return kernels._replace(
        var1=var1, nngp=nngp, var2=var2, ntk=ntk, is_gaussian=True,
        is_height_width=is_height_width, marginal=marginal, cross=cross)

  setattr(kernel_fn, _COVARIANCES_REQ, {'marginal': M.OVER_POINTS,
                                        'cross': M.NO,
                                        'spec': spec})
  return init_fn, apply_fn, kernel_fn


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
  axis = (axis,) if isinstance(axis, int) else tuple(axis)
  _spec = spec if spec else 'NHWC'

  def init_fn(rng, input_shape):
    return input_shape, ()

  def apply_fn(params, inputs, **kwargs):
    mean = np.mean(inputs, axis=axis, keepdims=True)
    var = np.var(inputs, axis=axis, keepdims=True)
    return (inputs - mean) / np.sqrt(eps + var)

  def kernel_fn(kernels):
    var1, nngp, var2, ntk, is_height_width, marginal, cross = (
        kernels.var1, kernels.nngp, kernels.var2, kernels.ntk,
        kernels.is_height_width, kernels.marginal, kernels.cross)
    _axis = axis

    if marginal != M.OVER_ALL or cross != M.OVER_ALL:
      _axis = list(set(np.arange(len(_spec))[list(_axis)]))

      channel_axis = _spec.index('C')
      if channel_axis not in _axis:
        raise ValueError("Normalisation over channels (axis %d) necessary for "
                         "convergence to an asymptotic kernel; "
                         "got axis=%s" % (channel_axis, _axis))

      batch_axis = _spec.index('N')
      if batch_axis in _axis:
        raise ValueError("Normalisation over batch (axis %d) not supported for "
                         "convergence to an asymptotic kernel; "
                         "got axis=%s" % (batch_axis, _axis))

      _axis.remove(channel_axis)

      kernel_axis = []
      h_axis, w_axis = sorted((_spec.index('H'), _spec.index('W')))
      if h_axis in _axis:
        kernel_axis += [1] if is_height_width else [2]
      if w_axis in _axis:
        kernel_axis += [2] if is_height_width else [1]

    else:
      __spec = [c for c in _spec if c in ('N', 'C')]
      _axis = list(set(np.arange(len(__spec))[list(_axis)]))
      if _axis != [__spec.index('C')]:
        raise ValueError("Normalisation over features necessary for convergence"
                         " to an asymptotic kernel; axis={}".format(_axis))
      kernel_axis = []

    prod11, prod12, prod22 = _get_normalising_prod(
        eps + var1, var2 if var2 is None else eps + var2,
        marginal, axis=kernel_axis)
    nngp /= np.sqrt(prod12)
    if _is_array(ntk):
      ntk /= np.sqrt(prod12)

    var1 /= np.sqrt(prod11)
    if var2 is not None:
      var2 /= np.sqrt(prod22)

    return kernels._replace(
        var1=var1, nngp=nngp, var2=var2, ntk=ntk,
        is_height_width=is_height_width, marginal=marginal, cross=cross)

  if len(axis) > 1:
    setattr(kernel_fn, _COVARIANCES_REQ, {'marginal': M.OVER_PIXELS,
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

  init_fn, apply_fn = ostax.Dropout(rate, mode=mode)
  kernel_fn_test = lambda kernels: kernels

  def kernel_fn_train(kernels):
    """kernel_fn for `train` mode. """
    var1, nngp, var2, ntk, marginal, cross, x1_is_x2, is_input = (
        kernels.var1, kernels.nngp, kernels.var2, kernels.ntk,
        kernels.marginal, kernels.cross, kernels.x1_is_x2,
        kernels.is_input)

    if is_input:
      raise ValueError('Dropout cannot be applied to the input layer.')
    factor = 1./rate
    var1 *= factor
    if var2 is not None:
      var2 *= factor
    new_factor = np.where(x1_is_x2, factor, 1.)
    nngp = _diag_mul(nngp, new_factor)
    ntk = _diag_mul(ntk, new_factor) if _is_array(ntk) else ntk
    # TODO(xlc): under which condition could we leave `is_gaussian` unchanged?
    return kernels._replace(var1=var1, nngp=nngp, var2=var2, ntk=ntk,
                            is_gaussian=False)

  kernel_fn = kernel_fn_test if mode == 'test' else kernel_fn_train

  return init_fn, apply_fn, kernel_fn
