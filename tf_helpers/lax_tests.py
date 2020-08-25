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


from tensorflow import nn
import tensorflow as tf
import lax
from tensorflow.python.platform import test
from absl.testing import parameterized
import itertools
import numpy as onp
from tensorflow.python.ops import numpy_ops as tfnp
from jax import numpy as jnp
import jax
import sys

class TFConvGeneralTest(tf.test.TestCase, parameterized.TestCase):


  @parameterized.parameters(
    {"lhs_np": onp.ones((5, 3)), "rhs_np": onp.ones((3, 2)),
      "dims": (((1,), (0,)), ((), ()))},
    {"lhs_np": onp.ones((5, 3)), "rhs_np": onp.ones((5, 3)),
      "dims": (((0, 1), (0, 1)), ((), ()))},
    {"lhs_np": onp.ones((5, 3, 2)), "rhs_np": onp.ones((2, 3, 2)),
      "dims": (((1, 2), (1, 0)), ((), ()))},
    {"lhs_np": onp.ones((6, 5, 3)), "rhs_np": onp.ones((6, 3, 2)),
      "dims": (((2,), (1,)), ((0,), (0,)))},
    {"lhs_np": onp.ones((6, 3, 5)), "rhs_np": onp.ones((6, 3, 2)),
      "dims": (((1,), (1,)), ((0,), (0,)))},
    {"lhs_np": onp.ones((5, 3, 2, 2)), "rhs_np": onp.ones((5, 2, 2, 6)),
      "dims": (((2, 3), (1, 2)), ((0,), (0,)))},
    {"lhs_np": onp.ones((2, 2, 5, 3)), "rhs_np": onp.ones((2, 2, 3, 2)),
      "dims": (((3,), (2,)), ((0, 1), (0, 1)))},
    {"lhs_np": onp.ones((2, 2, 5, 2)), "rhs_np": onp.ones((2, 2, 3, 2)),
      "dims": (((3,), (1,)), ((0,), (0,)))},
    {"lhs_np": onp.ones((2, 2, 5, 3, 3)), "rhs_np": onp.ones((2, 3, 2, 3, 2)),
      "dims": (((4,), (1,)), ((0,), (0,)))},
  )
  def test_tf_dot_general(self, lhs_np, rhs_np, dims):
    ans = jax.lax.dot_general(lhs_np, rhs_np, dims)
    result = lax.dot_general(lhs_np, rhs_np, dims)
    self.assertAllClose(result, tfnp.array(ans))


  @parameterized.named_parameters([
      ("_lhs_shape={}_rhs_shape={}_strides={}_padding={}"
       "_lhs_dilation={}_rhs_dilation={}"
       "_feature_group_count={}_batch_group_count={}_dims={}"
       "_perms={}".format(lhs_shape, rhs_shape,
           strides, padding, lhs_dilation, rhs_dilation,
           feature_group_count, batch_group_count, ",".join(dimension_numbers), perms),
           lhs_shape, rhs_shape, strides, padding, lhs_dilation, rhs_dilation,
           feature_group_count, batch_group_count, dimension_numbers, perms)
      for batch_group_count, feature_group_count in [(1, 1)]
      for lhs_shape, rhs_shape in [
          ((b * batch_group_count, i * feature_group_count, 9, w),
           (j * feature_group_count * batch_group_count, i, 4, 5))
          for w in [0, 10]
          for b, i, j in itertools.product([2, 3], repeat=3)]
      for strides in [(1, 1), (2, 1)]
      for padding in ['SAME']
      for lhs_dilation, rhs_dilation in [
        (None, (1, 1))
      ]
      for dimension_numbers, perms in [
        (("NHWC", "HWIO", "NHWC"), ([0, 2, 3, 1], [2, 3, 1, 0]))
      ]])
  def testConvGeneralDilated(self, lhs_shape, rhs_shape, strides,
                             padding, lhs_dilation, rhs_dilation,
                             feature_group_count, batch_group_count,
                             dimension_numbers, perms):
    tf.print("dimension_numbers: {}".format(dimension_numbers), output_stream=sys.stdout)
    lhs_perm, rhs_perm = perms  # permute to compatible shapes

    lhs_tf = tfnp.transpose(tfnp.ones(lhs_shape), lhs_perm)
    rhs_tf = tfnp.transpose(tfnp.ones(rhs_shape), rhs_perm)

    lhs_jax = jnp.transpose(jnp.ones(lhs_shape), lhs_perm)
    rhs_jax = jnp.transpose(jnp.ones(rhs_shape), rhs_perm)

    jax_conv = jax.lax.conv_general_dilated(lhs_jax, rhs_jax, strides, padding, lhs_dilation,
      rhs_dilation, dimension_numbers, feature_group_count, batch_group_count)

    tf_conv = lax.conv_general_dilated(lhs_tf, rhs_tf, strides, padding, jax_conv.shape, lhs_dilation,
      rhs_dilation, dimension_numbers, feature_group_count, batch_group_count)

    self.assertAllEqual(tf_conv, tfnp.asarray(jax_conv))


if __name__ == "__main__":
  test.main()
