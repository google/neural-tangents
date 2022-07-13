# Copyright 2020 Google LLC
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

"""Common Type Definitions."""

from typing import Any, Dict, Generator, List, Optional, Sequence, TYPE_CHECKING, Tuple, TypeVar, Union

from jax import random
import jax.numpy as np
from .kernel import Kernel
from typing_extensions import Protocol


PyTree = Any
"""A PyTree, see `JAX docs`_ for details.

.. _JAX docs: https://jax.readthedocs.io/en/latest/pytrees.html
"""


Axes = Union[int, Sequence[int]]
"""Axes specification, can be integers (`axis=-1`) or sequences (`axis=(1, 3)`).
"""


T = TypeVar('T')

if TYPE_CHECKING:
  NTTree = Union[List['NTTree[T]'], Tuple['NTTree[T]', ...], T]
  NTTrees = Union[List['NTTree[T]'], Tuple['NTTree[T]', ...]]
else:
  # Can't use recursive types with `sphinx-autodoc-typehints`.
  NTTree = Union[List[T], Tuple[T, ...], T]
  """Neural Tangents Tree.

  Trees of kernels and arrays naturally emerge in certain neural
  network computations (for example, when neural networks have nested parallel
  layers).

  Mimicking JAX, we use a lightweight tree structure called an :class:`NTTree`.
  :class:`NTTree` has internal nodes that are either lists or tuples and leaves
  which are either :class:`jax.numpy.ndarray` or
  :class:`~neural_tangents.Kernel` objects.
  """

  NTTrees = Union[List[T], Tuple[T, ...]]
  """A list or tuple of :class:`NTTree` s.
  """


Shapes = NTTree[Tuple[int, ...]]
"""A shape - a tuple of integers, or an :class:`NTTree` of such tuples.
"""


# Layer Definition.


class InitFn(Protocol):
  """A type alias for initialization functions.

  Initialization functions construct parameters for neural networks given a
  random key and an input shape. Specifically, they produce a tuple giving the
  output shape and a PyTree of parameters.
  """

  def __call__(
      self,
      rng: random.KeyArray,
      input_shape: Shapes,
      **kwargs
  ) -> Tuple[Shapes, PyTree]:
    ...


class ApplyFn(Protocol):
  """A type alias for apply functions.

  Apply functions do computations with finite-width neural networks. They are
  functions that take a PyTree of parameters and an array of inputs and produce
  an array of outputs.
  """

  def __call__(
      self,
      params: PyTree,
      inputs: NTTree[np.ndarray],
      *args,
      **kwargs
  ) -> NTTree[np.ndarray]:
    ...


class MaskFn(Protocol):
  """A type alias for a masking functions.

  Forward-propagate a mask in a layer of a finite-width network.
  """

  def __call__(
      self,
      mask: Union[np.ndarray, Sequence[np.ndarray]],
      input_shape: Shapes,
  ) -> Union[np.ndarray, Sequence[np.ndarray]]:
    ...


KernelOrInput = Union[NTTree[Kernel], NTTree[np.ndarray]]


Get = Union[Tuple[str, ...], str, None]


class LayerKernelFn(Protocol):
  """A type alias for pure kernel functions.

  A pure kernel function takes a PyTree of Kernel object(s) and produces a
  PyTree of Kernel object(s). These functions are used to define new layer
  types.
  """

  def __call__(
      self,
      k: NTTree[Kernel]
  ) -> NTTree[Kernel]:
    ...


class AnalyticKernelFn(Protocol):
  """A type alias for analytic kernel functions.

  A kernel function that computes an analytic kernel. Takes either a
  :class:`~neural_tangents.Kernel` or :class:`jax.numpy.ndarray` inputs and a
  `get` argument that specifies what quantities should be computed by the
  kernel. Returns either a :class:`~neural_tangents.Kernel` object or
  :class:`jax.numpy.ndarray`-s for kernels specified by `get`.
  """

  def __call__(
      self,
      x1: KernelOrInput,
      x2: Optional[NTTree[np.ndarray]] = None,
      get: Get = None,
      **kwargs
  ) -> Union[NTTree[Kernel], NTTree[np.ndarray]]:
    ...


class EmpiricalGetKernelFn(Protocol):
  """A type alias for empirical kernel functions accepting a `get` argument.

  A kernel function that produces an empirical kernel from a single
  instantiation of a neural network specified by its parameters.

  Equivalent to `EmpiricalKernelFn`, but accepts a `get` argument, which can be
  for example `get=("nngp", "ntk")`, to compute both kernels together.
  """

  def __call__(
      self,
      x1: NTTree[np.ndarray],
      x2: Optional[NTTree[np.ndarray]],
      get: Get,
      params: PyTree,
      **kwargs
  ) -> NTTree[np.ndarray]:
    ...


class EmpiricalKernelFn(Protocol):
  """A type alias for empirical kernel functions computing either NTK or NNGP.

  A kernel function that produces an empirical kernel from a single
  instantiation of a neural network specified by its parameters.

  Equivalent to `EmpiricalGetKernelFn` with `get="nngp"` or `get="ntk"`.
  """

  def __call__(
      self,
      x1: NTTree[np.ndarray],
      x2: Optional[NTTree[np.ndarray]],
      params: PyTree,
      **kwargs
  ) -> NTTree[np.ndarray]:
    ...


class MonteCarloKernelFn(Protocol):
  """A type alias for Monte Carlo kernel functions.

  A kernel function that produces an estimate of an `AnalyticKernel`
  by monte carlo sampling given a `PRNGKey`.
  """

  def __call__(
      self,
      x1: NTTree[np.ndarray],
      x2: Optional[NTTree[np.ndarray]],
      get: Get = None,
      **kwargs
  ) -> Union[NTTree[np.ndarray], Generator[NTTree[np.ndarray], None, None]]:
    ...


KernelFn = Union[
    AnalyticKernelFn,
    EmpiricalKernelFn,
    EmpiricalGetKernelFn,
    MonteCarloKernelFn,
]


InternalLayer = Tuple[InitFn, ApplyFn, LayerKernelFn]
InternalLayerMasked = Tuple[InitFn, ApplyFn, LayerKernelFn, MaskFn]


Layer = Tuple[InitFn, ApplyFn, AnalyticKernelFn]


Kernels = Union[List[Kernel], Tuple[Kernel, ...]]
"""Kernel inputs/outputs of `FanOut`, `FanInSum`, etc.
"""


_VMapAxis = Optional[PyTree]
"""A `PyTree` of integers.
"""

VMapAxisTriple = Tuple[_VMapAxis, _VMapAxis, Dict[str, _VMapAxis]]
VMapAxes = Union[_VMapAxis, VMapAxisTriple]
"""Specifies `(input, output, kwargs)` axes for `vmap` in empirical NTK.
"""
