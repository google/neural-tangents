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

"""Methods to compute Monte Carlo NNGP and NTK estimates.

The library has a public method `get_ker_fun_monte_carlo` that allow to compute
  Monte Carlo estimates of NNGP and NTK kernels of arbitrary functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from jax import random
from functools import partial
from neural_tangents.utils import batch
from neural_tangents.utils import empirical
from neural_tangents.utils.kernel import Kernel


def _get_ker_fun_sample_once(ker_fun,
                             init_fun,
                             batch_size=0,
                             device_count=0,
                             store_on_device=True):

  @partial(batch.batch, batch_size=batch_size,
           device_count=device_count,
           store_on_device=store_on_device)
  def ker_fun_sample_once(x1, x2, key):
    _, params = init_fun(key, x1.shape)

    def ker_fun_params(x1, x2):
      return ker_fun(x1, x2, params)

    return ker_fun_params(x1, x2)

  return ker_fun_sample_once


def _get_ker_fun_sample_many(ker_fun_sample_once,
                             compute_nngp=True,
                             compute_ntk=True):
  def get_sampled_kernel(x1, x2, key, n_samples):
    if x2 is not None:
      assert x1.shape[1:] == x2.shape[1:]

    if key.shape == (2,):
      key = random.split(key, n_samples)
    elif n_samples is not None:
      raise ValueError('Got set `n_samples=%d` and %d RNG keys.' %
                       (n_samples, key.shape[0]))

    ker_sampled = Kernel(var1=None,
                         nngp=0. if compute_nngp else None,
                         var2=None,
                         ntk=0. if compute_ntk else None,
                         is_gaussian=None,
                         is_height_width=None,
                         marginal=None,
                         cross=None)
    for subkey in key:
      ker_sampled += ker_fun_sample_once(x1, x2, subkey)

    return ker_sampled / len(key)

  return get_sampled_kernel


def get_ker_fun_monte_carlo(init_fun,
                            apply_fun,
                            compute_nngp=True,
                            compute_ntk=True,
                            batch_size=0,
                            device_count=-1,
                            store_on_device=True):
  """Return a Monte Carlo sampler of NTK and NNGP kernels of a given function.

  Args:
    init_fun: a function initializing parameters of the neural network. From
      `jax.experimental.stax`: "takes an rng key and an input shape and returns
      an `(output_shape, params)` pair".
    apply_fun: a function computing the output of the neural network.
      From `jax.experimental.stax`: "takes params, inputs, and an rng key and
      applies the layer".
    compute_nngp: a boolean, `True` to compute NNGP kernel.
    compute_ntk: a boolean, `True` to compute NTK kernel.
    batch_size: an integer making the kernel computed in batches of `x1` and
      `x2` of this size. `0` means computing the whole kernel. Must divide
      `x1.shape[0]` and `x2.shape[0]`.
    device_count: an integer making the kernel be computed in parallel across
      this number of devices (e.g. GPUs or TPU cores). `-1` means use all
      available devices. `0` means compute on a single device sequentially. If
      not `0`, must divide `x1.shape[0]`.
    store_on_device: a boolean, indicating whether to store the resulting
      kernel on the device (e.g. GPU or TPU), or in the CPU RAM, where larger
      kernels may fit.

  Returns:
    A function of signature `ker_fun(x1, x2, key, n_samples)` to sample an
      empirical `Kernel`.
  """
  ker_fun = empirical.get_ker_fun_empirical(apply_fun,
                                            compute_nngp,
                                            compute_ntk)

  ker_fun_sample_once = _get_ker_fun_sample_once(ker_fun,
                                                 init_fun,
                                                 batch_size,
                                                 device_count,
                                                 store_on_device)

  ker_fun_sample_many = _get_ker_fun_sample_many(ker_fun_sample_once,
                                                 compute_nngp,
                                                 compute_ntk)
  return ker_fun_sample_many
