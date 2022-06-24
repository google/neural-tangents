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
from jax import random
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

