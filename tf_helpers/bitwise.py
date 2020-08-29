# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
The bitwise `or` and `and` operation has not been integrated as part of TF Numpy API,
so I will put the implementation here temporarily for the use of Neural Tangents `or`
on TF boolean arrays. The following code is from:

    https://github.com/tensorflow/tensorflow/blob/964eca57caaad0ee0d24785c9f681795236c26ec/tensorflow/python/ops/numpy_ops/np_math_ops.py
"""

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_utils


def _bin_op(tf_fun, a, b, promote=True):
  if promote:
    a, b = np_array_ops._promote_dtype(a, b)  # pylint: disable=protected-access
  else:
    a = np_array_ops.array(a)
    b = np_array_ops.array(b)
  return np_utils.tensor_to_ndarray(tf_fun(a.data, b.data))


def _bitwise_binary_op(tf_fn, x1, x2):  # pylint: disable=missing-function-docstring

  def f(x1, x2):
    is_bool = (x1.dtype == dtypes.bool)
    if is_bool:
      assert x2.dtype == dtypes.bool
      x1 = math_ops.cast(x1, dtypes.int8)
      x2 = math_ops.cast(x2, dtypes.int8)
    r = tf_fn(x1, x2)
    if is_bool:
      r = math_ops.cast(r, dtypes.bool)
    return r

  return _bin_op(f, x1, x2)


def bitwise_or(x1, x2):
  return _bitwise_binary_op(bitwise_ops.bitwise_or, x1, x2)

def bitwise_and(x1, x2):
  return _bitwise_binary_op(bitwise_ops.bitwise_and, x1, x2)
