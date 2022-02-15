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

"""Closed-form NNGP and NTK library.

This library contains layer constructors mimicking those in
`jax.example_libraries.stax` with similar API apart apart from:

1) Instead of `(init_fn, apply_fn)` tuple, layer constructors return a triple
`(init_fn, apply_fn, kernel_fn)`, where the added `kernel_fn` maps a
`Kernel` to a new `Kernel`, and represents the change in the
analytic NTK and NNGP kernels (`Kernel.nngp`, `Kernel.ntk`). These functions
are chained / stacked together within the `serial` or `parallel`
combinators, similarly to `init_fn` and `apply_fn`.

2) In layers with random weights, NTK parameterization is used by default
(https://arxiv.org/abs/1806.07572, page 3). Standard parameterization
(https://arxiv.org/abs/2001.07301) can be specified for `Conv` and `Dense`
layers by a keyword argument `parameterization`.

3) Some functionality may be missing (e.g. `BatchNorm`), and some may be
present only in our library (e.g. `CIRCULAR` padding, `LayerNorm`,
`GlobalAvgPool`, `GlobalSelfAttention`, flexible batch and channel axes etc.).

Example:
  >>>  from jax import random
  >>>  import neural_tangents as nt
  >>>  from neural_tangents import stax
  >>>
  >>>  key1, key2 = random.split(random.PRNGKey(1), 2)
  >>>  x_train = random.normal(key1, (20, 32, 32, 3))
  >>>  y_train = random.uniform(key1, (20, 10))
  >>>  x_test = random.normal(key2, (5, 32, 32, 3))
  >>>
  >>>  init_fn, apply_fn, kernel_fn = stax.serial(
  >>>      stax.Conv(128, (3, 3)),
  >>>      stax.Relu(),
  >>>      stax.Conv(256, (3, 3)),
  >>>      stax.Relu(),
  >>>      stax.Conv(512, (3, 3)),
  >>>      stax.Flatten(),
  >>>      stax.Dense(10)
  >>>  )
  >>>
  >>>  predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, x_train,
  >>>                                                        y_train)
  >>>
  >>>  # (5, 10) np.ndarray NNGP test prediction
  >>>  y_test_nngp = predict_fn(x_test=x_test, get='nngp')
  >>>
  >>>  # (5, 10) np.ndarray NTK prediction
  >>>  y_test_ntk = predict_fn(x_test=x_test, get='ntk')
"""


# Layer combinators, combining multiple layers into a single layer.
from ._src.stax.combinators import (
    parallel,
    serial,
)


# Elementwise nonlinearities.
from ._src.stax.elementwise import (
    ABRelu,
    Abs,
    Cos,
    Elementwise,
    ElementwiseNumerical,
    Erf,
    Exp,
    ExpNormalized,
    Gaussian,
    Gelu,
    Hermite,
    LeakyRelu,
    Rbf,
    Relu,
    Sigmoid_like,
    Sign,
    Sin,
)


# Linear layers.
from ._src.stax.linear import (
    Aggregate,
    AvgPool,
    Conv,
    ConvLocal,
    ConvTranspose,
    Dense,
    Identity,
    DotGeneral,
    Dropout,
    Flatten,
    GlobalAvgPool,
    GlobalSelfAttention,
    GlobalSumPool,
    ImageResize,
    LayerNorm,
    SumPool,
)


# Branching layers.
from ._src.stax.branching import (
    FanInConcat,
    FanInProd,
    FanInSum,
    FanOut,
)


# Enums to specify layer behavior.
from ._src.stax.linear import (
    Padding,
    PositionalEmbedding,
)


# Decorators and classes for constructing your own layers.
from ._src.stax.requirements import (
    layer,
    supports_masking,
    requires,
    Bool,
    Diagonal
)
