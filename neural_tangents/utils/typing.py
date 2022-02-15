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

from typing import Tuple, Callable, Union, List, Any, Optional, Sequence, Generator, TypeVar, Dict
import jax.numpy as np
from neural_tangents.utils.kernel import Kernel


# Missing JAX Types.
PyTree = Any


"""A type alias for PRNGKeys.

  See https://jax.readthedocs.io/en/latest/jax.random.html#jax.random.PRNGKey
  for details.
"""
PRNGKey = np.ndarray


"""A type alias for axes specification.

  Axes can be specified as integers (`axis=-1`) or sequences (`axis=(1, 3)`).
"""
Axes = Union[int, Sequence[int]]


"""Neural Tangents Trees.

Trees of kernels and arrays naturally emerge in certain neural
network computations computations (for example, when neural networks have nested
parallel layers).

Mimicking JAX, we use a lightweight tree structure called an NTTree. NTTrees
have internal nodes that are either Lists or Tuples and leaves which are either
array or kernel objects.
"""
T = TypeVar('T')
NTTree = Union[List[T], Tuple[T, ...], T]


Shapes = NTTree[Tuple[int, ...]]


# Layer Definition.
"""A type alias for initialization functions.

Initialization functions construct parameters for neural networks given a
random key and an input shape. Specifically, they produce a tuple giving the
output shape and a PyTree of parameters.
"""
InitFn = Callable[[PRNGKey, Shapes], Tuple[Shapes, PyTree]]


"""A type alias for apply functions.

Apply functions do computations with finite-width neural networks. They are
functions that take a PyTree of parameters and an array of inputs and produce
an array of outputs.
"""
ApplyFn = Callable[[PyTree, NTTree[np.ndarray]], NTTree[np.ndarray]]


KernelOrInput = Union[NTTree[Kernel], NTTree[np.ndarray]]


Get = Union[Tuple[str, ...], str, None]

"""A type alias for pure kernel functions.

A pure kernel function takes a PyTree of Kernel object(s) and produces a
PyTree of Kernel object(s). These functions are used to define new layer
types.
"""
LayerKernelFn = Callable[[NTTree[Kernel]], NTTree[Kernel]]


"""A type alias for analytic kernel functions.

A kernel function that computes an analytic kernel. Takes either a kernel
or np.ndarray inputs and a `get` argument that specifies what quantities
should be computed by the kernel. Returns either a kernel object or
np.ndarrays for kernels specified by `get`.
"""
AnalyticKernelFn = Callable[[KernelOrInput, Optional[NTTree[np.ndarray]], Get],
                            Union[NTTree[Kernel], NTTree[np.ndarray]]]


"""A type alias for empirical kernel functions.

A kernel function that produces an empirical kernel from a single
instantiation of a neural network specified by its parameters.
"""
EmpiricalKernelFn = Callable[[NTTree[np.ndarray],
                              Optional[NTTree[np.ndarray]],
                              Get,
                              PyTree],
                             NTTree[np.ndarray]]


"""A type alias for Monte Carlo kernel functions.

A kernel function that produces an estimate of an `AnalyticKernel`
by monte carlo sampling given a `PRNGKey`.
"""
MonteCarloKernelFn = Callable[
    [NTTree[np.ndarray], Optional[NTTree[np.ndarray]], Get],
    Union[NTTree[np.ndarray],
          Generator[NTTree[np.ndarray], None, None]]]


KernelFn = Union[AnalyticKernelFn, EmpiricalKernelFn, MonteCarloKernelFn]


InternalLayer = Union[Tuple[InitFn, ApplyFn, LayerKernelFn],
                      Tuple[InitFn, ApplyFn, LayerKernelFn, Callable]]


Layer = Tuple[InitFn, ApplyFn, AnalyticKernelFn]


"""A type alias for kernel inputs/outputs of `FanOut`, `FanInSum`, etc.
"""
Kernels = Union[List[Kernel], Tuple[Kernel, ...]]


"""Specifies `(input, output, kwargs)` axes for `vmap` in empirical NTK.
"""
_VMapAxis = Optional[NTTree[int]]
VMapAxes = Tuple[_VMapAxis, _VMapAxis, Dict[str, _VMapAxis]]
