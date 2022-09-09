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
# ==============================================================================

"""Public Neural Tangents modules and functions."""


__version__ = '0.6.1'

from . import experimental
from . import predict
from . import stax
from ._src.batching import batch
from ._src.empirical import empirical_kernel_fn
from ._src.empirical import empirical_nngp_fn
from ._src.empirical import empirical_ntk_fn
from ._src.empirical import empirical_ntk_vp_fn
from ._src.empirical import linearize
from ._src.empirical import NtkImplementation
from ._src.empirical import taylor_expand
from ._src.monte_carlo import monte_carlo_kernel_fn
from ._src.utils.kernel import Kernel
