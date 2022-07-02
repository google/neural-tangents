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

"""Utilities for testing."""

import dataclasses
import itertools
import logging
import os
from types import ModuleType
from typing import Callable, Dict, Optional, Sequence, Tuple

from absl import flags
from absl.testing import parameterized
import jax
from jax import config
from jax import dtypes as _dtypes
from jax import jit
from jax import vmap
import jax.numpy as np
import numpy as onp


flags.DEFINE_string(
    'nt_test_dut',
    '',
    help=
    'Describes the device under test in case special consideration is required.'
)

flags.DEFINE_integer(
    'nt_num_generated_cases',
    int(os.getenv('NT_NUM_GENERATED_CASES', '10')),
    help='Number of generated cases to test'
)

FLAGS = flags.FLAGS

# Utility functions forked from :obj:`jax._src.public_test_util`.


_python_scalar_dtypes: Dict[type, onp.dtype] = {
    bool: onp.dtype('bool'),
    int: onp.dtype('int64'),
    float: onp.dtype('float64'),
    complex: onp.dtype('complex128'),
}


def _dtype(x) -> onp.dtype:
  if hasattr(x, 'dtype'):
    return x.dtype
  elif type(x) in _python_scalar_dtypes:
    return onp.dtype(_python_scalar_dtypes[type(x)])
  else:
    return onp.asarray(x).dtype


def _is_sequence(x) -> bool:
  try:
    iter(x)
  except TypeError:
    return False
  else:
    return True


def device_under_test() -> str:
  return getattr(FLAGS, 'nt_test_dut', None) or jax.default_backend()


_DEFAULT_TOLERANCE: Dict[onp.dtype, float] = {
    onp.dtype(onp.bool_): 0,
    onp.dtype(onp.int32): 0,
    onp.dtype(onp.int64): 0,
    onp.dtype(onp.float16): 5e-3,
    onp.dtype(onp.float32): 5e-3,
    onp.dtype(onp.float64): 1e-5,
    onp.dtype(np.bfloat16): 5e-3
}


def _default_tolerance() -> Dict[onp.dtype, float]:
  if device_under_test() != 'tpu':
    return _DEFAULT_TOLERANCE
  tol = _DEFAULT_TOLERANCE.copy()
  tol[onp.dtype(onp.float32)] = 5e-2
  tol[onp.dtype(onp.complex64)] = 5e-2
  return tol


def _assert_numpy_allclose(
    a: onp.ndarray,
    b: onp.ndarray,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    err_msg: str = ''
):
  if a.dtype == b.dtype == _dtypes.float0:
    onp.testing.assert_array_equal(a, b, err_msg=err_msg)
    return
  a = a.astype(onp.float32) if a.dtype == _dtypes.bfloat16 else a
  b = b.astype(onp.float32) if b.dtype == _dtypes.bfloat16 else b
  kw = {}
  if atol: kw['atol'] = atol
  if rtol: kw['rtol'] = rtol
  with onp.errstate(invalid='ignore'):
    # TODO(phawkins): surprisingly, assert_allclose sometimes reports invalid
    # value errors. It should not do that.
    onp.testing.assert_allclose(a, b, **kw, err_msg=err_msg)


def _tolerance(dtype: onp.dtype, tol: Optional[float] = None) -> float:
  tol = {} if tol is None else tol
  if not isinstance(tol, dict):
    return tol
  tol = {onp.dtype(key): value for key, value in tol.items()}
  dtype = _dtypes.canonicalize_dtype(onp.dtype(dtype))
  return tol.get(dtype, _default_tolerance()[dtype])


_CACHED_INDICES: Dict[int, Sequence[int]] = {}


def _cases_from_list(xs):
  xs = list(xs)
  n = len(xs)
  if n < FLAGS.nt_num_generated_cases:
    return xs
  k = min(n, FLAGS.nt_num_generated_cases)
  # Random sampling for every parameterized test is expensive. Do it once and
  # cache the result.
  indices = _CACHED_INDICES.get(n)
  if indices is None:
    rng = onp.random.RandomState(42)
    _CACHED_INDICES[n] = indices = rng.permutation(n)
  return [xs[i] for i in indices[:k]]


def product(*kwargs_seqs, **testgrid):
  """Test case decorator to randomly subset a cartesian product of parameters.

  Forked from `absltest.parameterized.product`.

  Args:
    *kwargs_seqs: Each positional parameter is a sequence of keyword arg dicts;
      every test case generated will include exactly one kwargs dict from each
      positional parameter; these will then be merged to form an overall list
      of arguments for the test case.
    **testgrid: A mapping of parameter names and their possible values. Possible
      values should given as either a list or a tuple.

  Raises:
    NoTestsError: Raised when the decorator generates no tests.

  Returns:
     A test generator to be handled by TestGeneratorMetaclass.
  """

  for name, values in testgrid.items():
    assert isinstance(values, (list, tuple)), (
        'Values of {} must be given as list or tuple, found {}'.format(
            name, type(values)))

  prior_arg_names = set()
  for kwargs_seq in kwargs_seqs:
    assert ((isinstance(kwargs_seq, (list, tuple))) and
            all(isinstance(kwargs, dict) for kwargs in kwargs_seq)), (
                'Positional parameters must be a sequence of keyword arg'
                'dicts, found {}'
                .format(kwargs_seq))
    if kwargs_seq:
      arg_names = set(kwargs_seq[0])
      assert all(set(kwargs) == arg_names for kwargs in kwargs_seq), (
          'Keyword argument dicts within a single parameter must all have the '
          'same keys, found {}'.format(kwargs_seq))
      assert not (arg_names & prior_arg_names), (
          'Keyword argument dict sequences must all have distinct argument '
          'names, found duplicate(s) {}'
          .format(sorted(arg_names & prior_arg_names)))
      prior_arg_names |= arg_names

  assert not (prior_arg_names & set(testgrid)), (
      'Arguments supplied in kwargs dicts in positional parameters must not '
      'overlap with arguments supplied as named parameters; found duplicate '
      'argument(s) {}'.format(sorted(prior_arg_names & set(testgrid))))

  # Convert testgrid into a sequence of sequences of kwargs dicts and combine
  # with the positional parameters.
  # So foo=[1,2], bar=[3,4] --> [[{foo: 1}, {foo: 2}], [{bar: 3, bar: 4}]]
  testgrid = (tuple({k: v} for v in vs) for k, vs in testgrid.items())
  testgrid = tuple(kwargs_seqs) + tuple(testgrid)

  # Create all possible combinations of parameters as a cartesian product
  # of parameter values.
  testcases = [
      dict(itertools.chain.from_iterable(case.items()
                                         for case in cases))
      for cases in itertools.product(*testgrid)
  ]
  return parameters(testcases)


def parameters(testcases):
  """A decorator for parameterized tests randomly sampled from the list.

  Adapted from `absltest.parameterized.parameters`.

  Args:
    testcases: an iterable of test cases.

  Raises:
    NoTestsError: Raised when the decorator generates no tests.

  Returns:
     A test generator to be handled by TestGeneratorMetaclass.
  """
  subset = _cases_from_list(testcases)
  return parameterized.parameters(subset)


class NeuralTangentsTestCase(parameterized.TestCase):
  """Testing helper class forked from JaxTestCase."""

  def _assertAllClose(
      self,
      x,
      y,
      *,
      check_dtypes: bool = True,
      atol: Optional[float] = None,
      rtol: Optional[float] = None,
      canonicalize_dtypes: bool = True,
      err_msg: str = ''
  ):
    """Assert that x and y, either arrays or nested tuples/lists, are close."""
    if isinstance(x, dict):
      self.assertIsInstance(y, dict)
      self.assertEqual(set(x.keys()), set(y.keys()))
      for k in x.keys():
        self._assertAllClose(x[k], y[k], check_dtypes=check_dtypes, atol=atol,
                             rtol=rtol, canonicalize_dtypes=canonicalize_dtypes,
                             err_msg=err_msg)
    elif _is_sequence(x) and not hasattr(x, '__array__'):
      self.assertTrue(_is_sequence(y) and not hasattr(y, '__array__'))
      self.assertEqual(len(x), len(y))
      for x_elt, y_elt in zip(x, y):
        self._assertAllClose(x_elt, y_elt, check_dtypes=check_dtypes, atol=atol,
                             rtol=rtol, canonicalize_dtypes=canonicalize_dtypes,
                             err_msg=err_msg)
    elif hasattr(x, '__array__') or onp.isscalar(x):
      self.assertTrue(hasattr(y, '__array__') or onp.isscalar(y))
      if check_dtypes:
        self.assertDtypesMatch(x, y, canonicalize_dtypes=canonicalize_dtypes)
      x = onp.asarray(x)
      y = onp.asarray(y)
      self.assertArraysAllClose(x, y, check_dtypes=False, atol=atol, rtol=rtol,
                                err_msg=err_msg)
    elif x == y:
      return
    else:
      raise TypeError((type(x), type(y)))

  def assertArraysAllClose(
      self,
      x,
      y,
      *,
      check_dtypes: bool = True,
      atol: Optional[float] = None,
      rtol: Optional[float] = None,
      err_msg: str = ''
  ):
    """Assert that x and y are close (up to numerical tolerances)."""
    self.assertEqual(x.shape, y.shape)
    atol = max(_tolerance(_dtype(x), atol), _tolerance(_dtype(y), atol))
    rtol = max(_tolerance(_dtype(x), rtol), _tolerance(_dtype(y), rtol))
    _assert_numpy_allclose(x, y, atol=atol, rtol=rtol, err_msg=err_msg)

    if check_dtypes:
      self.assertDtypesMatch(x, y)

  def assertDtypesMatch(self, x, y, *, canonicalize_dtypes: bool = True):
    if not config.x64_enabled and canonicalize_dtypes:
      self.assertEqual(_dtypes.canonicalize_dtype(_dtype(x)),
                       _dtypes.canonicalize_dtype(_dtype(y)))
    else:
      self.assertEqual(_dtype(x), _dtype(y))

  def assertAllClose(
      self,
      x,
      y,
      *,
      check_dtypes: bool = True,
      atol: Optional[float] = None,
      rtol: Optional[float] = None,
      canonicalize_dtypes: bool = True,
      check_finite: bool = True,
      err_msg=''):
    if check_finite:
      def is_finite(x):
        self.assertTrue(np.all(np.isfinite(x)))

      jax.tree_map(is_finite, x)
      jax.tree_map(is_finite, y)

    def assert_close(x, y):
      self._assertAllClose(
          x, y,
          check_dtypes=check_dtypes,
          atol=atol,
          rtol=rtol,
          canonicalize_dtypes=canonicalize_dtypes,
          err_msg=err_msg,
      )

    if dataclasses.is_dataclass(x):
      self.assertIs(type(y), type(x))
      for field in dataclasses.fields(x):
        key = field.name
        x_value, y_value = getattr(x, key), getattr(y, key)
        is_pytree_node = field.metadata.get('pytree_node', True)
        if is_pytree_node:
          assert_close(x_value, y_value)
        else:
          self.assertEqual(x_value, y_value, key)
    else:
      assert_close(x, y)


# Neural Tangents specific utilities.


def _jit_vmap(f: Callable) -> Callable:
  return jit(vmap(f))


def update_test_tolerance(f32_tol: float = 5e-3, f64_tol: float = 1e-5):
  _DEFAULT_TOLERANCE[onp.dtype(onp.float32)] = f32_tol
  _DEFAULT_TOLERANCE[onp.dtype(onp.float64)] = f64_tol


def stub_out_pmap(batch: ModuleType, count: int):
  # If we are using GPU or CPU stub out pmap with vmap to simulate multi-core.
  if count > 0:
    class xla_bridge_stub:

      def device_count(self) -> int:
        return count

    platform = jax.default_backend()
    if platform == 'gpu' or platform == 'cpu':
      batch.pmap = _jit_vmap
      batch.xla_bridge = xla_bridge_stub()


def _log(
    relative_error: float,
    absolute_error: float,
    expected,
    actual,
    did_pass: bool
):
  msg = 'PASSED' if did_pass else 'FAILED'
  logging.info(f'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n'
               f'\n{msg} with {relative_error} relative error \n'
               f'and {absolute_error} absolute error: \n'
               f'---------------------------------------------\n'
               f'EXPECTED: \n'
               f'{expected}\n'
               f'---------------------------------------------\n'
               f'ACTUAL: \n'
               f'{actual}\n'
               f'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n'
               )


def assert_close_matrices(self, expected, actual, rtol, atol=0.1):
  def assert_close(expected, actual):
    self.assertEqual(expected.shape, actual.shape)
    relative_error = (
        np.linalg.norm(actual - expected) /
        np.maximum(np.linalg.norm(expected), 1e-12))

    absolute_error = np.mean(np.abs(actual - expected))

    if (np.isnan(relative_error) or
        relative_error > rtol or
        absolute_error > atol):
      _log(relative_error, absolute_error, expected, actual, False)
      self.fail(self.failureException('Relative ERROR: ',
                                      float(relative_error),
                                      'EXPECTED:' + ' ' * 50,
                                      expected,
                                      'ACTUAL:' + ' ' * 50,
                                      actual,
                                      ' ' * 50,
                                      'Absolute ERROR: ',
                                      float(absolute_error)))
    else:
      _log(relative_error, absolute_error, expected, actual, True)

  jax.tree_map(assert_close, expected, actual)


def skip_test(
    self,
    msg: str = 'Skipping large tests for speed.',
    platforms: Tuple[str, ...] = ('cpu',)
):
  if jax.default_backend() in platforms:
    raise parameterized.TestCase.skipTest(self, msg)


def mask(
    x: np.ndarray,
    mask_constant: Optional[float],
    mask_axis: Sequence[int],
    key: jax.random.KeyArray,
    p: float) -> np.ndarray:
  if mask_constant is not None:
    mask_shape = [1 if i in mask_axis else s
                  for i, s in enumerate(x.shape)]
    mask_mat = jax.random.bernoulli(key, p=p, shape=mask_shape)
    x = np.where(mask_mat, mask_constant, x)
    x = np.sort(x, 1)
  return x
