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

"""Layer combinators."""

import operator as op
from typing import Any, Callable, Dict, List
import warnings

import frozendict
from jax import random, lax
import jax.example_libraries.stax as ostax
from .requirements import Diagonal, get_req, layer, requires
from ..utils.kernel import Kernel
from ..utils.typing import InternalLayer, Layer, LayerKernelFn, NTTree, NTTrees, Shapes


@layer
def serial(*layers: Layer) -> InternalLayer:
  """Combinator for composing layers in serial.

  Based on :obj:`jax.example_libraries.stax.serial`.

  Args:
    *layers:
      a sequence of layers, each an `(init_fn, apply_fn, kernel_fn)` triple.

  See Also:
    :obj:`~neural_tangents.stax.repeat` for compiled repeated composition.

  Returns:
    A new layer, meaning an `(init_fn, apply_fn, kernel_fn)` triple,
    representing the serial composition of the given sequence of layers.
  """
  init_fns, apply_fns, kernel_fns = zip(*layers)
  init_fn, apply_fn = ostax.serial(*zip(init_fns, apply_fns))

  @requires(**_get_input_req_attr(kernel_fns, fold=op.rshift))
  def kernel_fn(k: NTTree[Kernel], **kwargs) -> NTTree[Kernel]:
    # TODO(xlc): if we drop `x1_is_x2` and use `rng` instead, need split key
    # inside kernel functions here and parallel below.
    for f in kernel_fns:
      k = f(k, **kwargs)
    return k

  return init_fn, apply_fn, kernel_fn


@layer
def repeat(layer: Layer, n: int) -> InternalLayer:
  """Compose `layer` in a compiled loop `n` times.

  Equivalent to `serial(*([layer] * n))`, but allows faster compilation time
  for large `n` (but same runtime).

  .. warning::
    `apply_fn` of the `layer` is assumed to keep the activation (`x`) shape
    unchanged.

  .. warning::
    `kernel_fn` of the `layer` is assumed to keep the
    :class:`~neural_tangents.Kernel` metadata unchanged. This is most notably
    not satisfied in :obj:`~neural_tangents.stax.Conv` and other convolutional
    layers which flip the `is_reversed` attribute with each application. A
    workaround is to either use `serial(*([layer] * n))`, or to use
    `repeat(serial(layer, layer), n // 2)` instead of `repeat(layer, n)` for an
    even `n`, i.e. to use two (or, generally, any even number of) convolutions
    per `layer` instead of one (or, generally, any odd number), such that
    `layer` does not alter the `is_reversed` attribute. Similar caution should
    be applied to other :class:`~neural_tangents.Kernel` attributes.

  See Also:
    `RepeatTest` in `tests/stax/combinators_test.py` for examples and
    :obj:`~neural_tangents.stax.serial` for unrolled composition.

  Example:
    >>> from neural_tangents import stax
    >>> #
    >>> layer = stax.serial(stax.Dense(128), stax.Relu())
    >>> depth = 100
    >>> #
    >>> # Unrolled loop:
    >>> nn_unrolled = stax.serial(*([layer] * depth))
    >>> #
    >>> # Compiled loop:
    >>> nn_compiled = stax.repeat(layer, depth)
    >>> # `nn_unrolled` and `nn_compiled` perform the same computation, but
    >>> # `nn_compiled` compiles faster and with smaller memory footprint.

  Args:
    layer:
      layer to be repeated. Outputs must have the same shape and other metadata
      as inputs.

    n:
      number of times to repeat a layer (depth).

  Returns:
    A new layer, meaning an `(init_fn, apply_fn, kernel_fn)` triple,
    representing the repeated composition of `layer` `n` times.
  """
  init_fn, apply_fn, kernel_fn = layer

  def init_fn_repeat(rng, input_shape):
    out_shape, _ = init_fn(rng, input_shape)
    if out_shape != input_shape:
      raise ValueError(
          f'`init_fn` produces a different output shape {out_shape} than the '
          f'input shape {input_shape}. Please use the `serial(*([layer] * n)`) '
          f'construction in this setting.'
      )

    def init_fn_scan(rng, params):
      rng, layer_rng = random.split(rng)
      out_shape, params = init_fn(layer_rng, input_shape)
      return rng, params

    _, params = lax.scan(init_fn_scan, rng, None, n)
    return out_shape, params

  def apply_fn_repeat(params, inputs, **kwargs):
    def apply_fn_scan(x, params):
      return apply_fn(params, x, **kwargs), None

    outputs, _ = lax.scan(apply_fn_scan, inputs, params, n)
    return outputs

  @requires(**get_req(kernel_fn))
  def kernel_fn_repeat(k: NTTree[Kernel], **kwargs) -> NTTree[Kernel]:
    if n > 0:
      k = kernel_fn(k, **kwargs)

      def kernel_fn_scan(k, _):
        k = kernel_fn(k, **kwargs)
        return k, None

      k, _ = lax.scan(kernel_fn_scan, k, None, n - 1)

    return k

  return init_fn_repeat, apply_fn_repeat, kernel_fn_repeat


@layer
def parallel(*layers: Layer) -> InternalLayer:
  """Combinator for composing layers in parallel.

  The layer resulting from this combinator is often used with the
  :obj:`~neural_tangents.stax.FanOut`, :obj:`~neural_tangents.stax.FanInSum`,
  and :obj:`~neural_tangents.stax.FanInConcat` layers. Based on
  :obj:`jax.example_libraries.stax.parallel`.

  Args:
    *layers:
      a sequence of layers, each with a `(init_fn, apply_fn, kernel_fn)` triple.

  Returns:
    A new layer, meaning an `(init_fn, apply_fn, kernel_fn)` triples,
    representing the parallel composition of the given sequence of layers. In
    particular, the returned layer takes a sequence of inputs and returns a
    sequence of outputs with the same length as the argument `layers`.
  """
  init_fns, apply_fns, kernel_fns = zip(*layers)
  init_fn_stax, apply_fn_stax = ostax.parallel(*zip(init_fns, apply_fns))

  def init_fn(rng: random.KeyArray, input_shape: Shapes):
    return type(input_shape)(init_fn_stax(rng, input_shape))

  def apply_fn(params, inputs, **kwargs):
    return type(inputs)(apply_fn_stax(params, inputs, **kwargs))

  @requires(**_get_input_req_attr(kernel_fns, fold=op.and_))
  def kernel_fn(ks: NTTrees[Kernel], **kwargs) -> NTTrees[Kernel]:
    return type(ks)(f(k, **kwargs) for k, f in zip(ks, kernel_fns))

  return init_fn, apply_fn, kernel_fn


# INTERNAL UTILITIES


def _get_input_req_attr(
    kernel_fns: List[LayerKernelFn],
    fold: Callable[[Diagonal, Diagonal], Diagonal]) -> Dict[str, Any]:
  """Gets requirements of the combined layer based on individual requirements.

  Specifically, gets the requirements / allowances to the inputs to a `serial`
  or `parallel` sequence of layers based on requirements of each layer, setting
  requirements / allowances to the most / least demanding among all layers.

  Args:
    kernel_fns:
      list of `kernel_fn`s fed to the `kernel_fns` (e.g. a list of
      convolutional layers and nonlinearities to be chained together with the
      `serial` combinator) or evaluated in parallel (`parallel` combinator).
    fold:
      binary associative operator to combine allowances of consecutive
      individual `kernel_fn`s. Can be only `operator.rshift` (`>>`), i.e.
      composition (corresponding to `serial`) or `operator.and_`, (`&`), i.e.
      `AND` (corresponding to `parallel`).

  Returns:
    A `dict` with combined requirements / allowances.
  """
  req = {}

  for f in kernel_fns:
    req_f = get_req(f, default=frozendict.frozendict())

    for k, v in req_f.items():
      if k == 'use_dropout':
        if k in req and req[k] != v:
          raise ValueError('`use_dropout` is a single whole-network attribute '
                           'and cannot be set to different values.')
        req[k] = v

      elif k in ('batch_axis', 'channel_axis'):
        if k not in req:
          req[k] = v
        else:
          if fold is op.and_:
            if k in req and req[k] != v:
              if (req[k] >= 0 and v >= 0) or (req[k] < 0 and v < 0):
                warnings.warn(f'For `kernel_fn`, `{k}` parameters must match in'
                              f' all parallel branches, got {req[k]} and {v}. '
                              f'This WILL lead to [silent] errors if '
                              f'`kernel_fn` is called.')
              else:
                warnings.warn(f'Got potentially mismatching `{k}` values in '
                              f'parallel branches: {req[k]} and {v}.')

          elif fold is not op.rshift:
            raise ValueError(fold)

      elif k in ('diagonal_batch', 'diagonal_spatial'):
        if k in req:
          req[k] = fold(req[k], v)
        else:
          req[k] = v

      else:
        raise NotImplementedError(k)

  return req
