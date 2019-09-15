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

1) Instead of `(init_fun, apply_fun)` tuple, layer constructors return a triple
  `(init_fun, apply_fun, ker_fun)`, where the added `ker_fun` maps an
  `Kernel` to a new `Kernel`, and represents the change in the
  analytic NTK and NNGP kernels (fields of `Kernel`). These functions
  are chained / stacked together within the `serial` or `parallel` combinators,
  similarly to `init_fun` and `apply_fun`.

2) In layers with random weights, NTK parameterization is used
  (https://arxiv.org/abs/1806.07572, page 3).

3) Individual methods may have some new or missing functionality. For instance,
  `CIRCULAR` padding is supported, but certain `dimension_numbers` aren't yet.

Example:
  ```python
  >>> import jax.numpy as np
  >>> import jax.scipy as sp
  >>>
  >>> x_train = np.ones((20, 32, 32, 3))
  >>> y_train = np.ones((20, 10))
  >>> x_test = np.ones((10, 32, 32, 3))
  >>>
  >>> init_fun, apply_fun, ker_fun = serial(
  >>>     Conv(128, (3, 3)),
  >>>     Relu(),
  >>>     Conv(256, (3, 3)),
  >>>     Relu(),
  >>>     Conv(512, (3, 3)),
  >>>     Flatten(),
  >>>     Dense(10)
  >>> )
  >>>
  >>> K_train_train = ker_fun(x_train)
  >>> K_test_train = ker_fun(x_test, x_train)
  >>>
  >>> # NNGP prediction
  >>> y_test_nngp = np.matmul(
  >>>     K_test_train.nngp,
  >>>     sp.linalg.solve(K_train_train.nngp, y_train, sym_pos=True)
  >>> )
  >>> # NTK prediction
  >>> y_test_ntk = np.matmul(
  >>>     K_test_train.ntk,
  >>>     sp.linalg.solve(K_train_train.ntk, y_train, sym_pos=True)
  >>> )
  >>> # TODO(romann): improve this example.
  ```
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from functools import wraps
import warnings
import enum
from jax import lax
from jax import random
from jax.experimental import stax
from jax.lib import xla_bridge
import jax.numpy as np
from jax.scipy.special import erf
from neural_tangents.utils.kernel import Kernel


_CONV_DIMENSION_NUMBERS = ('NHWC', 'HWIO', 'NHWC')
_CONV_QAB_DIMENSION_NUMBERS = ('NCHW', 'HWIO', 'NCHW')
_USE_POOLING = 'use_pooling'


class Padding(enum.Enum):
  CIRCULAR = 'CIRCULAR'
  SAME = 'SAME'
  VALID = 'VALID'


def _is_array(x):
  return isinstance(x, np.ndarray) and hasattr(x, 'shape') and x.shape


def _set_pooling_attr(combinator_ker_fun, ker_funs):
  """Labels whether `combinator_ker_fun` uses pooling based on `ker_funs`.

  Specifically, sets it's `_USE_POOLING` attribute to `True` if any of the
    `ker_funs` has it set to `True`, otherwise to `False` if any of them has it
    set to `False`, and does not set the attribute otherwise.

  Args:
    combinator_ker_fun: a 'ker_fun` of a `serial` or `parallel` combinator.
    ker_funs: list of 'ker_fun`s fed to the `ker_funs` (e.g. a list of
      convolutional layers and nonlinearities to be chained together with the
      `serial` combinator).

  Returns:
    `ker_funs` with the `_USE_POOLING` attribute set to `True`
  """
  for f in ker_funs:
    if hasattr(f, _USE_POOLING):
      if getattr(f, _USE_POOLING):
        setattr(combinator_ker_fun, _USE_POOLING, True)
        break

      setattr(combinator_ker_fun, _USE_POOLING, False)

  return combinator_ker_fun


def _randn(stddev=1e-2):
  """`stax.randn` for implicitly-typed results."""
  def init(rng, shape):
    return stddev * random.normal(rng, shape)
  return init


def _double_tuple(x):
  return tuple(v for v in x for _ in range(2))


def _get_variance(x):
  return np.sum(x**2, axis=-1, keepdims=False) / x.shape[-1]


def _batch_uncentered_covariance(x1, x2):
  """Batch uncentered covariance of `x1` and `x2` with batch inner dimensions.

  Args:
    x1: a (2+k)D (k >= 0) `np.ndarray` of shape
      `[batch_size_1, <k inner dimensions>, n_observations]`.
    x2: an optional `np.ndarray` that has the same shape as `a` apart from
      possibly different leading (`batch_size_2`) dimension. `None` means
      `x1 == x2`.

  Returns:
    an (2+k)D `np.ndarray` empirical uncentered batch covariance with shape
      `[batch_size_1, batch_size_2, <k inner dimensions>]`.
  """
  if x2 is None:
    x2 = x1

  prod = np.matmul(np.moveaxis(x1, 0, -2), np.moveaxis(x2, 0, -1))
  prod = np.moveaxis(prod, (-2, -1), (0, 1)) / x1.shape[-1]
  return prod


def _inputs_to_kernel(x1, x2, use_pooling, compute_ntk):
  """Transforms (batches of) inputs to a `Kernel`.

  This is a private method. Docstring and example are for internal reference.

   The kernel contains the empirical covariances between different inputs and
     their entries (pixels) necessary to compute the covariance of the Gaussian
     Process corresponding to an infinite Bayesian or gradient-flow-trained
     neural network.

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
    use_pooling: a boolean, indicating whether pooling will be used somewhere in
      the model. If so, more covariance entries need to be tracked. Is set
      automatically based on the network topology. Specifically, is set to
      `False` if a `serial` or `parallel` networks contain a `Flatten` layer
      and no pooling layers (`AvgPool` or `GlobalAvgPool`). Has no effect for
      non-convolutional models.
    compute_ntk: a boolean, `True` to compute both NTK and NNGP kernels,
        `False` to only compute NNGP.

    Example:
      ```python
          >>> x = np.ones((10, 32, 16, 3))
          >>> _inputs_to_kernel(x, None, use_pooling=True,
          >>>                   compute_ntk=True).ntk.shape
          (10, 10, 32, 32, 16, 16)
          >>> _inputs_to_kernel(x, None, use_pooling=False,
          >>>                   compute_ntk=True).ntk.shape
          (10, 10, 32, 16)
          >>> x1 = np.ones((10, 128))
          >>> x2 = np.ones((20, 128))
          >>> _inputs_to_kernel(x, None, use_pooling=True,
          >>>                   compute_ntk=False).nngp.shape
          (10, 20)
          >>> _inputs_to_kernel(x, None, use_pooling=False,
          >>>                   compute_ntk=False).nngp.shape
          (10, 20)
          >>> _inputs_to_kernel(x, None, use_pooling=False,
          >>>                   compute_ntk=False).ntk
          None
      ```

  Returns:
    a `Kernel` object.
  """
  x1 = x1.astype(xla_bridge.canonicalize_dtype(np.float64))
  var1 = _get_variance(x1)

  if x2 is None:
    x2 = x1
    var2 = None
  else:
    if x1.shape[1:] != x2.shape[1:]:
      raise ValueError('`x1` and `x2` are expected to be batches of'
                       ' inputs with the same shape (apart from the batch size),'
                       ' got %s and %s.' %
                       (str(x1.shape), str(x2.shape)))

    x2 = x2.astype(xla_bridge.canonicalize_dtype(np.float64))
    var2 = _get_variance(x2)

  if use_pooling and x1.ndim == 4:
    x2 = np.expand_dims(x2, -1)
    nngp = np.dot(x1, x2) / x1.shape[-1]
    nngp = np.transpose(np.squeeze(nngp, -1), (0, 3, 1, 4, 2, 5))

  elif x1.ndim == 4 or x1.ndim == 2:
    nngp = _batch_uncentered_covariance(x1, x2)

  else:
    raise ValueError('Inputs must be 2D or 4D `np.ndarray`s of shape '
                     '`[batch_size, n_features]` or '
                     '`[batch_size, height, width, channels]`, '
                     'got %s.' % str(x1.shape))

  ntk = 0. if compute_ntk else None
  is_gaussian = False
  is_height_width = True
  return Kernel(var1, nngp, var2, ntk, is_gaussian, is_height_width)


def _preprocess_ker_fun(ker_fun):
  def new_ker_fun(x1_or_kernel,
                  x2=None,
                  compute_nngp=True,
                  compute_ntk=True):
    """Returns the `Kernel` resulting from applying `ker_fun` to given inputs.

    Inputs can be either a pair of `np.ndarray`s, or a `Kernel'. If `n_samples`
      is positive, `ker_fun` is estimated by Monte Carlo sampling of random
      networks defined by `(init_fun, apply_fun)`.

    Args:
      x1_or_kernel: either a `np.ndarray` with shape
        `[batch_size_1] + input_shape`, or a `Kernel`.
      x2: an optional `np.ndarray` with shape `[batch_size_2] + input_shape`.
        `None` means `x2 == x1` or `x1_or_kernel is Kernel`.
      compute_nngp: a boolean, `True` to compute NNGP kernel.
      compute_ntk: a boolean, `True` to compute NTK kernel.

    Returns:
      A `Kernel`.
    """
    if (isinstance(x1_or_kernel, Kernel) or
        (isinstance(x1_or_kernel, list) and
         all(isinstance(k, Kernel) for k in x1_or_kernel))):
      kernel = x1_or_kernel

    elif isinstance(x1_or_kernel, np.ndarray):
      if x2 is None or isinstance(x2, np.ndarray):
        if not compute_nngp:
          if compute_ntk:
            raise ValueError('NNGP has to be computed to compute NTK. Please '
                             'set `compute_nngp=True`.')
          else:
            return Kernel(None, None, None, None, None, None)

        use_pooling = getattr(ker_fun, _USE_POOLING, True)
        kernel = _inputs_to_kernel(x1_or_kernel, x2, use_pooling, compute_ntk)

      else:
        raise TypeError('`x2` to a kernel propagation function '
                        'should be `None` or a `np.ndarray`, got %s.'
                        % type(x2))

    else:
      raise TypeError('Inputs to a kernel propagation function should be '
                      'a `Kernel`, '
                      'a `list` of `Kernel`s, '
                      'or a (tuple of) `np.ndarray`(s), got %s.'
                      % type(x1_or_kernel))

    return ker_fun(kernel)

  if hasattr(ker_fun, _USE_POOLING):
    setattr(new_ker_fun, _USE_POOLING, getattr(ker_fun, _USE_POOLING))

  return new_ker_fun


def _layer(layer):
  """A convenience decorator to be added to all public layers like `Relu` etc.

  Makes the `ker_fun` of the layer work with both input `np.ndarray`s (when the
    layer is the first one applied to inputs), and with `Kernel` for
    intermediary layers. Also adds optional arguments to make the `ker_fun` call
    the empirical Monte Carlo kernel estimation instead of computing it
    analytically (by default), as well as specifying the batching strategy.

  Args:
    layer: A layer function returning a triple `(init_fun, apply_fun, ker_fun)`.

  Returns:
    A function with the same signature as `layer` with `ker_fun` now accepting
      `np.ndarray`s as inputs if needed, and optional `n_samples=0`, `key=None`,
      `compute_ntk=True` arguments to let the user indicate that they want the
      kernel to be computed by Monte Carlo sampling.
  """
  @wraps(layer)
  def layer_fun(*args, **kwargs):
    init_fun, apply_fun, ker_fun = layer(*args, **kwargs)
    ker_fun = _preprocess_ker_fun(ker_fun)
    return init_fun, apply_fun, ker_fun
  return layer_fun


def _elementwise(fun, **fun_kwargs):
  init_fun, apply_fun = stax.elementwise(fun, **fun_kwargs)
  ker_fun = lambda kernels: _transform_kernels(kernels, fun, **fun_kwargs)
  return init_fun, apply_fun, ker_fun


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


def _get_var_prod(var1, nngp, var2):
  if var2 is not None:
    if var1.shape[1:] != var2.shape[1:]:
      raise ValueError(var1.shape, var2.shape)
  else:
    var2 = var1

  if nngp.ndim == 2 or nngp.ndim == 4:
    prod = np.expand_dims(var1, 1) * np.expand_dims(var2, 0)

  elif nngp.ndim == 6:
    # conv + pooling.
    var2 = np.reshape(var2,
                      (1, var2.shape[0], 1, var2.shape[1], 1, var2.shape[2]))
    var1 = np.reshape(var1,
                      (var1.shape[0], 1, var1.shape[1], 1, var1.shape[2], 1))
    prod = var1 * var2

  else:
    raise ValueError('NNGP should be either a rank-2, -4 or -6 array, got %d.'
                     % nngp.ndim)

  return prod


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


def _arcsin(x, do_backprop):
  if do_backprop:
    # https://github.com/google/jax/issues/654
    x = np.where(np.abs(x) >= 1, np.sign(x), x)
  else:
    x = np.clip(x, -1, 1)
  return np.arcsin(x)


def _transform_kernels_ab_relu(kernels, a, b, do_backprop, do_stabilize):
  """Compute new kernels after an `ABRelu` layer.

  See https://arxiv.org/pdf/1711.09090.pdf for the leaky ReLU derivation.
  """
  var1, nngp, var2, ntk, _, is_height_width = kernels

  if do_stabilize:
    factor = np.max([np.max(np.abs(nngp)), 1e-12])
    nngp /= factor
    var1 /= factor
    if var2 is not None:
      var2 /= factor

  prod = _get_var_prod(var1, nngp, var2)
  cosines = nngp / np.sqrt(prod)
  angles = _arccos(cosines, do_backprop)
  dot_sigma = (a**2 + b**2 - (a - b)**2 * angles / np.pi) / 2
  if ntk is not None:
    ntk *= dot_sigma

  nngp = ((a - b)**2 * _sqrt(prod - nngp**2, do_backprop)
         / (2 * np.pi) + dot_sigma * nngp)
  if do_stabilize:
    nngp *= factor

  var1 *= (a**2 + b**2) / 2
  if var2 is not None:
    var2 *= (a**2 + b**2) / 2

  if do_stabilize:
    var1 *= factor
    var2 *= factor

  return Kernel(var1, nngp, var2, ntk, a == b, is_height_width)


def _transform_kernels_erf(kernels, do_backprop):
  """Compute new kernels after an `Erf` layer."""
  var1, nngp, var2, ntk, _, is_height_width = kernels
  _var1_denom = 1 + 2 * var1
  _var2_denom = None if var2 is None else 1 + 2 * var2
  prod = _get_var_prod(_var1_denom, nngp, _var2_denom)

  dot_sigma = 4 / (np.pi * np.sqrt(prod - 4 * nngp ** 2))
  if ntk is not None:
    ntk *= dot_sigma

  nngp = _arcsin(2 * nngp / np.sqrt(prod), do_backprop) * 2 / np.pi

  var1 = np.arcsin(2 * var1 / _var1_denom) * 2 / np.pi
  if var2 is not None:
    var2 = np.arcsin(2 * var2 / _var2_denom) * 2 / np.pi

  return Kernel(var1, nngp, var2, ntk, False, is_height_width)


def _transform_kernels(kernels, fun, **fun_kwargs):
  """Apply transformation to kernels.

  Args:
    kernels: a `Kernel` object.
    fun: nonlinearity function, can only be Relu, Erf or Identity.
  Returns:
    The transformed kernel.
  """
  is_gaussian = kernels.is_gaussian
  if not is_gaussian:
    raise ValueError('An affine layer (i.e. dense or convolution) '
                     'has to be applied before a nonlinearity layer.')
  if fun is _ab_relu:
    return _transform_kernels_ab_relu(kernels, **fun_kwargs)
  if fun is _erf:
    return _transform_kernels_erf(kernels, **fun_kwargs)
  # TODO(xlc): Monte Carlo approximation to the integral (suggested by schsam.)
  raise NotImplementedError('Analaytic kernel for activiation {} is not '
                            'implmented'.format(fun))


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
def Dense(out_dim, W_std, b_std, W_init=_randn(1.0), b_init=_randn(1.0)):
  """Layer constructor function for a dense (fully-connected) layer.

  Based on `jax.experimental.stax.Dense`. Has a similar API.
  """
  init_fun, _ = stax.Dense(out_dim, W_init, b_init)

  def apply_fun(params, inputs, **kwargs):
    W, b = params
    norm = W_std / np.sqrt(inputs.shape[-1])
    return norm * np.dot(inputs, W) + b_std * b

  def ker_fun(kernels):
    """Compute the transformed kernels after a dense layer."""
    var1, nngp, var2, ntk, _, _ = kernels

    def fc(x):
      return _affine(x, W_std, b_std)

    var1, nngp, var2, ntk = map(fc, (var1, nngp, var2, ntk))
    if ntk is not None:
      ntk += nngp - b_std**2

    return Kernel(var1, nngp, var2, ntk, True, True)

  return init_fun, apply_fun, ker_fun


@_layer
def serial(*layers):
  """Combinator for composing layers in serial.

  Based on `jax.experimental.stax.serial`.

  Args:
    *layers: a sequence of layers, each an (init_fun, apply_fun, ker_fun) tuple.

  Returns:
    A new layer, meaning an `(init_fun, apply_fun, ker_fun)` tuple, representing
      the serial composition of the given sequence of layers.
  """
  init_funs, apply_funs, ker_funs = zip(*layers)
  init_fun, apply_fun = stax.serial(*zip(init_funs, apply_funs))

  def ker_fun(kernels):
    for f in ker_funs:
      kernels = f(kernels)
    return kernels

  _set_pooling_attr(ker_fun, ker_funs)
  return init_fun, apply_fun, ker_fun


@_layer
def Identity():
  """Layer construction function for an identity layer.

  Based on `jax.experimental.stax.Identity`.
  """
  init_fun, apply_fun = stax.Identity
  ker_fun = lambda kernels: kernels
  return init_fun, apply_fun, ker_fun


@_layer
def FanOut(num):
  """Layer construction function for a fan-out layer.

  Based on `jax.experimental.stax.FanOut`.
  """
  init_fun, apply_fun = stax.FanOut(num)
  ker_fun = lambda kernels: [kernels] * num
  return init_fun, apply_fun, ker_fun


@_layer
def FanInSum():
  """Layer construction function for a fan-in sum layer.

  Based on `jax.experimental.stax.FanInSum`.
  """
  init_fun, apply_fun = stax.FanInSum
  def ker_fun(kernels):
    is_gaussian = all(ker.is_gaussian for ker in kernels)
    if not is_gaussian:
      raise NotImplementedError('`FanInSum` layer is only implemented for the '
                                'case if all input layers guaranteed to be mean'
                                '-zero gaussian, i.e. having all `is_gaussian'
                                'set to `True`.')

    # If kernels have different height/width order, transpose some of them.
    n_kernels = len(kernels)
    n_height_width = sum(ker.is_height_width for ker in kernels)

    if n_height_width == n_kernels:
      is_height_width = True

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

    kers = tuple(None if all(ker[i] is None for ker in kernels) else
                 sum(ker[i] for ker in kernels) for i in range(4))
    return Kernel(*(kers + (is_gaussian, is_height_width)))

  return init_fun, apply_fun, ker_fun


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
  var1, nngp, var2, ntk, is_gaussian, is_height_width = kernels
  var1 = np.transpose(var1, (0, 2, 1))
  var2 = np.transpose(var2, (0, 2, 1)) if var2 is not None else var2
  nngp = np.transpose(nngp, (0, 1, 4, 5, 2, 3))
  ntk = np.transpose(ntk, (0, 1, 4, 5, 2, 3)) if ntk is not None else ntk
  return Kernel(var1, nngp, var2, ntk, is_gaussian, not is_height_width)


@_layer
def parallel(*layers):
  """Combinator for composing layers in parallel.

  The layer resulting from this combinator is often used with the `FanOut` and
    `FanInSum` layers. Based on `jax.experimental.stax.parallel`.

  Args:
    *layers: a sequence of layers, each an `(init_fun, apply_fun, ker_fun)`
      triple.

  Returns:
    A new layer, meaning an `(init_fun, apply_fun, ker_fun)` triples,
      representing the parallel composition of the given sequence of layers. In
      particular, the returned layer takes a sequence of inputs and returns a
      sequence of outputs with the same length as the argument `layers`.
  """
  init_funs, apply_funs, ker_funs = zip(*layers)
  init_fun, apply_fun = stax.parallel(*zip(init_funs, apply_funs))

  def ker_fun(kernels):
    return [f(ker) for ker, f in zip(kernels, ker_funs)]

  _set_pooling_attr(ker_fun, ker_funs)
  return init_fun, apply_fun, ker_fun


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
                                  Padding.SAME.value)

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


def _conv_nngp_6d_double_conv(nngp, filter_shape, strides, padding):
  """Compute covariances of the CNN outputs given inputs with covariances `nngp`.

  Uses 2D convolution and works on any hardware platform.

  Args:
    nngp: a 6D `np.ndarray` containing sample-sample-pixel-pixel covariances.
      Has shape `[batch_size_1, batch_size_2, height, height, width, width]`.
    filter_shape: tuple of positive integers, the convolutional filters spatial
      shape (e.g. `(3, 3)` for a 2D convolution).
    strides: tuple of positive integers, the CNN strides (e.g. `(1, 1)` for a
      2D convolution).
    padding: a `Padding` enum, e.g. `Padding.CIRCULAR`.

  Returns:
    a 6D `np.ndarray` containing sample-sample-pixel-pixel covariances of CNN
      outputs. Has shape `[batch_size_1, batch_size_2, new_width, new_width,
                           new_height, new_height]`.
  """
  if padding == Padding.CIRCULAR:
    pixel_axes = tuple(range(2, nngp.ndim))
    nngp = _same_pad_for_filter_shape(
        nngp,
        _double_tuple(filter_shape),
        _double_tuple(strides),
        pixel_axes,
        'wrap'
    )
    padding = Padding.VALID

  batch_size_1, batch_size_2, X, _, Y, _ = nngp.shape
  filter_x, filter_y = filter_shape
  stride_x, stride_y = strides

  ker_y = np.diag(np.full((filter_y,), 1. / filter_y, nngp.dtype))
  ker_y = np.reshape(ker_y, (filter_y, filter_y, 1, 1))

  channel_axis = _CONV_QAB_DIMENSION_NUMBERS[0].index('C')
  nngp = lax.conv_general_dilated(
      np.expand_dims(nngp.reshape((-1, Y, Y)), channel_axis),
      ker_y, (stride_y, stride_y), padding.value,
      dimension_numbers=_CONV_QAB_DIMENSION_NUMBERS)
  out_Y = nngp.shape[-2]
  nngp = nngp.reshape((batch_size_1, batch_size_2, X, X, out_Y, out_Y))

  ker_x = np.diag(np.full((filter_x,), 1. / filter_x, nngp.dtype))
  ker_x = np.reshape(ker_x, (filter_x, filter_x, 1, 1))

  nngp = np.transpose(nngp, (0, 1, 4, 5, 2, 3))
  nngp = lax.conv_general_dilated(
      np.expand_dims(nngp.reshape((-1, X, X)), channel_axis),
      ker_x, (stride_x, stride_x), padding.value,
      dimension_numbers=_CONV_QAB_DIMENSION_NUMBERS)
  out_X = nngp.shape[-2]
  nngp = nngp.reshape((batch_size_1, batch_size_2, out_Y, out_Y, out_X, out_X))

  return nngp


def _conv_nngp_4d(nngp, filter_shape, strides, padding):
  """Compute covariances of the CNN outputs given inputs with covariances `nngp`.

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
    nngp = _same_pad_for_filter_shape(nngp, filter_shape, strides, (2, 3), 'wrap')
    padding = Padding.VALID

  ker_nngp = np.full(filter_shape + (1, 1), 1. / np.prod(filter_shape),
                    nngp.dtype)

  channel_axis = _CONV_QAB_DIMENSION_NUMBERS[0].index('C')
  batch_1, batch_b, X, Y = nngp.shape
  nngp = np.reshape(nngp, (-1, X, Y))
  nngp = np.expand_dims(nngp, channel_axis)
  nngp = lax.conv_general_dilated(nngp, ker_nngp, strides, padding.value,
                                 dimension_numbers=_CONV_QAB_DIMENSION_NUMBERS)
  nngp = np.squeeze(nngp, channel_axis)
  nngp = nngp.reshape((batch_1, batch_b,) + nngp.shape[1:])
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
    var1 = _same_pad_for_filter_shape(var1, filter_shape, strides, (1, 2), 'wrap')
    padding = Padding.VALID

  channel_axis = _CONV_QAB_DIMENSION_NUMBERS[0].index('C')
  var1 = np.expand_dims(var1, channel_axis)
  ker__var1 = np.full(filter_shape + (1, 1), 1. / np.prod(filter_shape),
                    var1.dtype)
  var1 = lax.conv_general_dilated(var1, ker__var1, strides, padding.value,
                                 dimension_numbers=_CONV_QAB_DIMENSION_NUMBERS)
  var1 = np.squeeze(var1, channel_axis)
  return var1


@_layer
def _GeneralConv(dimension_numbers, out_chan, filter_shape,
                 strides=None, padding=Padding.VALID.value,
                 W_std=1.0, W_init=_randn(1.0),
                 b_std=0.0, b_init=_randn(1.0)):
  """Layer construction function for a general convolution layer.

  Based on `jax.experimental.stax.GeneralConv`. Has a similar API apart from:

  Args:
    padding: in addition to `VALID` and `SAME' padding, supports `CIRCULAR`,
      not available in `jax.experimental.stax.GeneralConv`.
  """
  if dimension_numbers != _CONV_DIMENSION_NUMBERS:
    raise NotImplementedError('Dimension numbers %s not implemented.'
                              % str(dimension_numbers))

  lhs_spec, rhs_spec, out_spec = dimension_numbers

  one = (1,) * len(filter_shape)
  strides = strides or one

  padding = Padding(padding)
  init_padding = padding
  if padding == Padding.CIRCULAR:
    init_padding = Padding.SAME

  init_fun, _ = stax.GeneralConv(dimension_numbers, out_chan, filter_shape,
                                 strides, init_padding.value, W_init, b_init)

  def apply_fun(params, inputs, **kwargs):
    W, b = params
    norm = inputs.shape[lhs_spec.index('C')]
    norm *= np.prod(filter_shape)
    apply_padding = padding
    if padding == Padding.CIRCULAR:
      apply_padding = Padding.VALID
      inputs = _same_pad_for_filter_shape(inputs, filter_shape, strides, (1, 2),
                                          'wrap')
    norm = W_std / np.sqrt(norm)

    return norm * lax.conv_general_dilated(
        inputs, W, strides, apply_padding.value,
        dimension_numbers=dimension_numbers) + b_std * b

  def ker_fun(kernels):
    """Compute the transformed kernels after a conv layer."""
    # var1: batch_1 * height * width
    # var2: batch_2 * height * width
    # nngp, ntk: batch_1 * batch_2 * height * height * width * width (pooling)
    #  or batch_1 * batch_2 * height * width (flattening)
    var1, nngp, var2, ntk, _, is_height_width = kernels

    if nngp.ndim == 4:
      def conv_var(x):
        x = _conv_var_3d(x, filter_shape, strides, padding)
        x = _affine(x, W_std, b_std)
        return x

      def conv_nngp(x):
        if _is_array(x):
          x = _conv_nngp_4d(x, filter_shape, strides, padding)
        x = _affine(x, W_std, b_std)
        return x

    elif nngp.ndim == 6:
      if not is_height_width:
        filter_shape_nngp = filter_shape[::-1]
        strides_nngp = strides[::-1]
      else:
        filter_shape_nngp = filter_shape
        strides_nngp = strides

      def conv_var(x):
        x = _conv_var_3d(x, filter_shape_nngp, strides_nngp, padding)
        if x is not None:
          x = np.transpose(x, (0, 2, 1))
        x = _affine(x, W_std, b_std)
        return x

      def conv_nngp(x):
        if _is_array(x):
          x = _conv_nngp_6d_double_conv(x, filter_shape_nngp, strides_nngp,
                                        padding)
        x = _affine(x, W_std, b_std)
        return x

      is_height_width = not is_height_width

    else:
      raise ValueError('`nngp` array must be either 4d or 6d, got %d.'
                       % nngp.ndim)

    var1 = conv_var(var1)
    var2 = conv_var(var2)
    nngp = conv_nngp(nngp)
    ntk = conv_nngp(ntk) + nngp - b_std**2 if ntk is not None else ntk
    return Kernel(var1, nngp, var2, ntk, True, is_height_width)

  return init_fun, apply_fun, ker_fun


def Conv(out_chan, filter_shape,
         strides=None, padding=Padding.VALID.value,
         W_std=1.0, W_init=_randn(1.0),
         b_std=0.0, b_init=_randn(1.0)):
  """Layer construction function for a convolution layer.

  Based on `jax.experimental.stax.Conv`. Has a similar API apart from:

  Args:
    padding: in addition to `VALID` and `SAME' padding, supports `CIRCULAR`,
      not available in `jax.experimental.stax.GeneralConv`.
  """
  return _GeneralConv(_CONV_DIMENSION_NUMBERS, out_chan, filter_shape,
                      strides, padding, W_std, W_init, b_std, b_init)


def _average_pool_nngp_6d(nngp, window_shape, strides, padding):
  """Get covariances of average pooling outputs given inputs covariances `nngp`.

  Args:
    nngp: a 6D `np.ndarray` containing sample-sample-pixel-pixel covariances.
      Has shape `[batch_size_1, batch_size_2, height, height, width, width]`.
    window_shape: tuple of two positive integers, the pooling spatial shape
      (e.g. `(3, 3)`).
    strides: tuple of two positive integers, the pooling strides, e.g. `(1, 1)`.
    padding: a `Padding` enum, e.g. `Padding.CIRCULAR`.

  Returns:
    a 6D `np.ndarray` containing sample-sample-pixel-pixel covariances of the
      average pooling outputs. Has shape `[batch_size_1, new_height, new_width,
                                          batch_size_2, new_height, new_width]`.
  """
  if not _is_array(nngp):
    return nngp

  if padding == Padding.CIRCULAR:
    pixel_axes = tuple(range(2, nngp.ndim))
    nngp = _same_pad_for_filter_shape(nngp, _double_tuple(window_shape),
                                     _double_tuple(strides), pixel_axes, 'wrap')
    padding = Padding.VALID

  window_shape = _double_tuple((1,) + window_shape)
  strides = _double_tuple((1,) + strides)

  nngp_out = lax.reduce_window(nngp, 0., lax.add, window_shape, strides,
                              padding.value)

  if padding == Padding.SAME:
    # `SAME` padding in `jax.experimental.stax.AvgPool` normalizes by actual
    # window size, which is smaller at the edges.
    one = np.ones(nngp.shape, nngp.dtype)
    window_sizes = lax.reduce_window(one, 0., lax.add, window_shape, strides,
                                     padding.value)
    nngp_out /= window_sizes
  else:
    nngp_out /= np.prod(window_shape)

  return nngp_out


def _diagonal_nngp_6d(nngp):
  """Get all sample-pixel variances from the covariances `nngp`.

  Args:
    nngp: a 6D `np.ndarray` containing sample-sample-pixel--pixel covariances.
      Has shape `[batch_size, batch_size, height, height, width, width]`.

  Returns:
    a 3D `np.ndarray` with sample-pixel variances on the diagonal of `nngp`. Has
    shape `[batch_size, height, width]`.
  """
  batch_idx = np.reshape(np.arange(nngp.shape[0]), (-1, 1, 1))
  X_idx = np.reshape(np.arange(nngp.shape[2]), (1, -1, 1))
  Y_idx = np.reshape(np.arange(nngp.shape[4]), (1, 1, -1))
  var1 = nngp[batch_idx, batch_idx, X_idx, X_idx, Y_idx, Y_idx]
  return var1


@_layer
def AvgPool(window_shape, strides=None, padding=Padding.VALID.value):
  """Layer construction function for a 2D average pooling layer.

  Based on `jax.experimental.stax.AvgPool`. Has a similar API apart from:

  Args:
    padding: in addition to `VALID` and `SAME' padding, supports `CIRCULAR`,
      not available in `jax.experimental.stax.GeneralConv`.
  """
  strides = strides or (1,) * len(window_shape)
  padding = Padding(padding)

  if padding == Padding.CIRCULAR:
    init_fun, _ = stax.AvgPool(window_shape, strides, Padding.SAME.value)
    _, apply_fun_0 = stax.AvgPool(window_shape, strides, Padding.VALID.value)

    def apply_fun(params, inputs, **kwargs):
      inputs = _same_pad_for_filter_shape(inputs, window_shape, strides, (1, 2),
                                          'wrap')
      res = apply_fun_0(params, inputs, **kwargs)
      return res
  else:
    init_fun, apply_fun = stax.AvgPool(window_shape, strides, padding.value)

  def ker_fun(kernels):
    """Kernel transformation."""
    var1, nngp, var2, ntk, is_gaussian, is_height_width = kernels

    if not is_height_width:
      window_shape_nngp = window_shape[::-1]
      strides_nngp = strides[::-1]
    else:
      window_shape_nngp = window_shape
      strides_nngp = strides

    nngp = _average_pool_nngp_6d(nngp, window_shape_nngp, strides_nngp, padding)
    ntk = _average_pool_nngp_6d(ntk, window_shape_nngp, strides_nngp, padding)

    if var2 is None:
      var1 = _diagonal_nngp_6d(nngp)
    else:
      # TODO(romann)
      warnings.warn('Pooling for different inputs `x1` and `x2` is not '
                    'implemented and will only work if there are no '
                    'nonlinearities in the network anywhere after the pooling '
                    'layer. `var1` and `var2` will have wrong values. '
                    'This will be fixed soon.')

    return Kernel(var1, nngp, var2, ntk, is_gaussian, is_height_width)

  setattr(ker_fun, _USE_POOLING, True)
  return init_fun, apply_fun, ker_fun


@_layer
def GlobalAvgPool():
  """Layer construction function for a global average pooling layer.

  Pools over and removes (`keepdims=False`) all inner dimensions (from 1 to -2),
    e.g. appropriate for `NHWC`, `NWHC`, `CHWN`, `CWHN` inputs.
  """
  def init_fun(rng, input_shape):
    output_shape = input_shape[0], input_shape[-1]
    return output_shape, ()

  def apply_fun(params, inputs, **kwargs):
    pixel_axes = tuple(range(1, inputs.ndim - 1))
    return np.mean(inputs, axis=pixel_axes)

  def ker_fun(kernels):
    var1, nngp, var2, ntk, is_gaussian, _ = kernels

    pixel_axes = tuple(range(2, nngp.ndim))
    nngp = np.mean(nngp, axis=pixel_axes)
    ntk = np.mean(ntk, axis=pixel_axes) if _is_array(ntk) else ntk

    if var2 is None:
      var1 = np.diagonal(nngp)
    else:
      # TODO(romann)
      warnings.warn('Pooling for different inputs `x1` and `x2` is not '
                    'implemented and will only work if there are no '
                    'nonlinearities in the network anywhere after the pooling '
                    'layer. `var1` and `var2` will have wrong values. '
                    'This will be fixed soon.')

    return Kernel(var1, nngp, var2, ntk, is_gaussian, True)

  setattr(ker_fun, _USE_POOLING, True)
  return init_fun, apply_fun, ker_fun


@_layer
def Flatten():
  """Layer construction function for flattening all but the leading dim.

  Based on `jax.experimental.stax.Flatten`. Has a similar API.
  """
  init_fun, apply_fun = stax.Flatten
  def ker_fun(kernels):
    """Compute kernels."""
    var1, nngp, var2, ntk, is_gaussian, _ = kernels
    if nngp.ndim == 2:
      return kernels

    var1 = np.mean(var1, axis=(1, 2))
    var2 = var2 if var2 is None else np.mean(var2, axis=(1, 2))

    if nngp.ndim == 4:
      nngp = np.mean(nngp, axis=(2, 3))
      if _is_array(ntk):
        ntk = np.mean(ntk, axis=(2, 3))
      return Kernel(var1, nngp, var2, ntk, is_gaussian, True)

    if nngp.ndim == 6:
      def trace(x):
        count = x.shape[2] * x.shape[4]
        y = np.trace(x, axis1=4, axis2=5)
        z = np.trace(y, axis1=2, axis2=3)
        return z / count
      nngp = trace(nngp)
      if _is_array(ntk):
        ntk = trace(ntk)
      return Kernel(var1, nngp, var2, ntk, is_gaussian, True)

    raise ValueError('`nngp` array must be 2d or 6d.')

  setattr(ker_fun, _USE_POOLING, False)
  return init_fun, apply_fun, ker_fun
