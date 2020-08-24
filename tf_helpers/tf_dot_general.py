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
Construct an equivalent general dot operation as that in JAX -
    <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.dot_general.html>

Although there is an implementation in TF XLA, avoid directly using XLA when
possible.

Zhibo Zhang, 2020.06.30
"""

import tensorflow as tf
from tensorflow.python.ops import numpy_ops as tf_np
import string


def _minus(a, b):
  return [x for x in a if x not in b]


def compose_output_rep(lhs_rep, rhs_rep, lhs_contraction, rhs_contraction,
                        lhs_batch, rhs_batch):
  """ Compose the output string representation.

  e.g., ij, jk, (((1,), (0,)), ((), ())) -> ik
        aij, ajk, (((2,), (1,)), ((0,), (0,))) -> aik

  Args:
    lhs_rep: A string representation for the left-hand side input array
    rhs_rep: A string representation for the right-hand side input array
    lhs_contraction: Sequence[int] (the contraction dimensions of lhs)
    rhs_contraction: Sequence[int] (the contraction dimensions of rhs)
    lhs_batch: Sequence[int] (the batch dimensions of lhs)
    rhs_batch: Sequence[int] (the batch dimensions of rhs)

  Returns:
    A string representation of the result array.
  """
  output_rep = []
  for dim in lhs_batch:
    output_rep.append(lhs_rep[dim])

  for i in _minus(range(len(lhs_rep)), lhs_batch + lhs_contraction):
    output_rep.append(lhs_rep[i])
  for i in _minus(range(len(rhs_rep)), rhs_batch + rhs_contraction):
    output_rep.append(rhs_rep[i])
  return ''.join(output_rep)


def non_batched_matmul(lhs, rhs, lhs_contraction, rhs_contraction):
  """ Compute the non-batched matrix multiplication.

  If it is the general non-batched/single-batched matrix multiplication,
  use the highly optimized kernel `tf.tensordot` to handle it.

  Args:
    lhs: an array (the left-hand side matrix/vector to be multiplied)
    rhs: an array (the right-hand side matrix/vector to be multiplied)
    lhs_contraction: Sequence[int] (the contraction dimensions of lhs)
    rhs_contraction: Sequence[int] (the contraction dimensions of rhs)

  Returns:
    An array that contains the result.
  """
  return tf.tensordot(lhs, rhs, axes=(list(lhs_contraction), list(rhs_contraction)))


def tf_dot_general(lhs, rhs, dimension_numbers):
  """ The general dot operation for TensorFlow.

  An equivalent general dot operation as that in JAX -
     <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.dot_general.html>
  Although there is an implementation in TF XLA, avoid directly using XLA when
  possible.

  e.g., non-batched: ij,jk->ik
        batched: ijk,ikl->ijl

  Args:
    lhs: an array (the left-hand side matrix/vector to be multiplied)
    rhs: an array (the right-hand side matrix/vector to be multiplied)
    dimension_numbers: (Tuple[Tuple[Sequence[int], Sequence[int]],
      Tuple[Sequence[int], Sequence[int]]]) – a tuple of tuples of the form
      ((lhs_contracting_dims, rhs_contracting_dims), (lhs_batch_dims, rhs_batch_dims))

  Returns:
    An array that contains the result.
  """
  char_list = list(string.ascii_lowercase)
  char_list = char_list[8:] + char_list[:8]
  lhs_rank, rhs_rank = len(lhs.shape), len(rhs.shape)
  lhs_rep = char_list[:lhs_rank]
  rhs_rep = char_list[lhs_rank:lhs_rank+rhs_rank]
  contraction, batch = dimension_numbers
  lhs_contraction, rhs_contraction = contraction
  if len(lhs_contraction) != len(rhs_contraction):
    raise ValueError("The input matrices are required to have the same number "
                     "of contraction dimensions, but got: lhs {}, rhs: {}".format(
                     len(lhs_contraction), len(rhs_contraction)))
  lhs_batch, rhs_batch = batch
  if len(lhs_batch) != len(rhs_batch):
    raise ValueError("The input matrices are required to have the same number "
                     "of batch dimensions, but got: lhs {}, rhs: {}".format(
                     len(lhs_batch), len(rhs_batch)))

  if len(lhs_batch) == 0 and len(rhs_batch) == 0:
    return non_batched_matmul(lhs, rhs, lhs_contraction, rhs_contraction)

  if (lhs_rank == rhs_rank == 3 and lhs_batch == (0,) and rhs_batch == (0,)
      and lhs_contraction == (2,) and rhs_contraction == (1,)):
    return tf.linalg.matmul(lhs, rhs)

  for i in range(len(lhs_contraction)):
    rhs_rep[rhs_contraction[i]] = lhs_rep[lhs_contraction[i]]
  for i in range(len(lhs_batch)):
    rhs_rep[rhs_batch[i]] = lhs_rep[lhs_batch[i]]

  output_rep = compose_output_rep(lhs_rep, rhs_rep, lhs_contraction,
                                  rhs_contraction, lhs_batch, rhs_batch)
  equation = ''.join(lhs_rep) + ',' + ''.join(rhs_rep) + "->" + output_rep
  return tf.einsum(equation, lhs, rhs)
