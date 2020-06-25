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

from typing import Tuple, Callable, Union, List, Any, Optional, Sequence, \
  Generator
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


# Layer Definition.
"""A type alias for initialization functions.

Initialization functions construct parameters for neural networks given a
random key and an input shape. Specifically, they produce a tuple giving the
output shape and a PyTree of parameters.
"""
InitFn = Callable[[PRNGKey, Tuple[int, ...]], Tuple[Tuple[int, ...], PyTree]]


"""A type alias for apply functions.

Apply functions do computations with finite-width neural networks. They are
functions that take a PyTree of parameters and an array of inputs and produce
an array of outputs.
"""
ApplyFn = Callable[..., np.ndarray]


Shapes = Union[Tuple[int, ...], List[Tuple[int, ...]]]


Kernels = Union[Kernel, List[Kernel]]


KernelOrInput = Union[Kernel, np.ndarray]


Get = Union[Tuple[str, ...], str, None]


"""A type alias for pure kernel functions.

A pure kernel function takes a (list of) Kernel object(s) and produces a
(list of) Kernel object(s). These functions are used to define new layer
types.
"""
LayerKernelFn = Callable[[Kernels], Kernels]


"""A type alias for analytic kernel functions.

A kernel function that computes an analytic kernel. Takes either a kernel
or np.ndarray inputs and a `get` argument that specifies what quantities
should be computed by the kernel. Returns either a kernel object or
np.ndarrays for kernels specified by `get`.
"""
AnalyticKernelFn = Callable[[KernelOrInput, Optional[np.ndarray], Get],
                            Union[Kernel, np.ndarray, Tuple[np.ndarray, ...]]]


"""A type alias for empirical kernel functions.

A kernel function that produces an empirical kernel from a single
instantiation of a neural network specified by its parameters.
"""
EmpiricalKernelFn = Callable[[np.ndarray, Optional[np.ndarray], PyTree, Get],
                             Union[np.ndarray, Tuple[np.ndarray, ...]]]


"""A type alias for Monte Carlo kernel functions.

A kernel function that produces an estimate of an `AnalyticKernel`
by monte carlo sampling given a `PRNGKey`.
"""
MonteCarloKernelFn = Callable[
    [np.ndarray, Optional[np.ndarray], Get],
    Union[Union[np.ndarray, Tuple[np.ndarray, ...]],
          Generator[Union[np.ndarray, Tuple[np.ndarray, ...]], None, None]]]


KernelFn = Union[AnalyticKernelFn, EmpiricalKernelFn, MonteCarloKernelFn]


InternalLayer = Union[Tuple[InitFn, ApplyFn, LayerKernelFn],
                      Tuple[InitFn, ApplyFn, LayerKernelFn, Callable]]


Layer = Tuple[InitFn, ApplyFn, AnalyticKernelFn]
