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

"""General-purpose internal utilities."""

from jax.api import vmap
from jax.lib import xla_bridge
import jax.numpy as np
from collections import namedtuple
from functools import wraps
import inspect
import types


def stub_out_pmap(batch, count):
  # If we are using GPU or CPU stub out pmap with vmap to simulate multi-core.
  if count > 1:
    class xla_bridge_stub(object):
      def device_count(self):
        return count

    platform = xla_bridge.get_backend().platform
    if platform == 'gpu' or platform == 'cpu':
      # TODO(romann): investigate why vmap is extremely slow in
      # `utils/monte_carlo_test.py`, `test_monte_carlo_vs_analytic`.
      # Example: http://sponge/e081c176-e77f-428c-846d-bafbfd86a46c
      batch.pmap = vmap
      batch.xla_bridge = xla_bridge_stub()


def assert_close_matrices(self, expected, actual, rtol):
  self.assertEqual(expected.shape, actual.shape)
  relative_error = (np.linalg.norm(actual - expected) /
                    np.maximum(np.linalg.norm(expected), 1e-12))
  if relative_error > rtol or np.isnan(relative_error):
    self.fail(self.failureException(float(relative_error), expected, actual))
  else:
    print('PASSED with %f relative error.' % relative_error)


def canonicalize_get(get):
  if not get:
    # NOTE(schsam): It seems slightly nicer to not support the empty-tuple
    # case. Happy to add support later, if there's a use-case.
    raise ValueError('"get" must be non-empty.')

  get_is_not_tuple = isinstance(get, str)
  if get_is_not_tuple:
    get = (get,)

  get = tuple(s.lower() for s in get)
  if len(set(get)) < len(get):
    raise ValueError(
        'All entries in "get" must be unique. Got {}'.format(get))
  return get_is_not_tuple, get


_KERNEL_NAMED_TUPLE_CACHE = {}
def named_tuple_factory(name, get):
  key = (name, get)
  if key in _KERNEL_NAMED_TUPLE_CACHE:
    return _KERNEL_NAMED_TUPLE_CACHE[key]
  else:
    _KERNEL_NAMED_TUPLE_CACHE[key] = namedtuple(name, get)
    return named_tuple_factory(name, get)


def get_namedtuple(name):
  def getter_decorator(fun):
    try:
      get_index = inspect.getargspec(fun).args.index('get')
    except:
      raise ValueError(
          '"get_namedtuple" functions must have a "get" argument.')

    @wraps(fun)
    def getter_fun(*args, **kwargs):
      if not args:
        raise ValueError(
            'A get_namedtuple function must have a "get" argument.')

      canonicalized_args = list(args)
      if 'get' in kwargs:
        get_is_not_tuple, get = canonicalize_get(kwargs['get'])
        kwargs['get'] = get
      else:
        get_is_not_tuple, get = canonicalize_get(args[get_index])
        canonicalized_args[get_index] = get

      fun_out = fun(*canonicalized_args, **kwargs)

      if get_is_not_tuple:
        if isinstance(fun_out, types.GeneratorType):
          return (output[get[0]] for output in fun_out)
        else:
          return fun_out[get[0]]

      ReturnType = named_tuple_factory(name, get)
      if isinstance(fun_out, types.GeneratorType):
        return (ReturnType(*tuple(output[g] for g in get))
                for output in fun_out)
      else:
        return ReturnType(*tuple(fun_out[g] for g in get))

    return getter_fun
  return getter_decorator
