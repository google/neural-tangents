"""Structured derivatives rules."""

from .dataclasses import dataclass, field
import functools
from typing import Callable, Optional, Tuple, Dict, List, Union, Any

from . import utils
import jax
from jax import lax
from jax.core import JaxprEqn, ShapedArray, Primitive, Jaxpr, Var, AbstractValue, Literal
from jax._src import dispatch as jax_dispatch
from jax.interpreters import ad
import jax.numpy as np
import numpy as onp


# pytype: disable=wrong-keyword-args


@dataclass
class Structure:
  """Describes structure present in a primitive derivative dy/dw.

  # TODO(romann): make this a python dataclass.

  Attributes:
    out_trace:
      axes of the primitive `y` output along which the primitive Jacobian
      `dy/dw` is constant-block diagonal along the respective axes in the input
      `in_trace`.

    in_trace:
      axes of the primitive `y` inputs along which the primitive Jacobian
      `dy/dw` is constant-block diagonal along the respective axes in the output
      `out_trace`.

    in_trace_idxs:
      indices of input variables to which `in_trace` axes are applied. Other
      variables are considered untouched.

    out_diagonal:
      axes of the primitive `y` output along which the primitive Jacobian
      `dy/dw` is (not constant) block diagonal along the respective axes in the
      input `in_diagonal`.

    in_diagonal:
      axes of the primitive `y` inputs along which the primitive Jacobian
      `dy/dw` is (not constant) block diagonal along the respective axes in the
      output `out_diagonal`. Each entry in the `in_diagonal` tuple is a tuple of
      length equal to the number of input variables; each entry in the tuple is
      either an integer axis number correspomnding to the respective input
      variable, or `None`, meaning that the respective variable is considered
      untouched.

    out_broadcast:
      axes of the primitive `y` output along which the primitive Jacobian
      `dy/dw` is block-tiled.

    out_broadcast_idxs:
      indices of input variables that need to be squeezed along the
      `out_broadcast` axes in order for the primitive `y` to return the slice
      that is being tiled along `out_broadcast` in the full output.

    in_broadcast:
      axes of the primitive `y` inputs along which the primitive Jacobian
      `dy/dw` is block-tiled.

    in_broadcast_idx:
      indices of input variables that need to be squeezed along the
      `in_broadcast` axes in order for the primitive Jacobian `dy/dw` to return
      the slice that is being tiled along `in_broadcast` in the full output.
  """
  out_trace: Tuple[int, ...] = field(False, default_factory=tuple)
  in_trace: Tuple[int, ...] = field(False, default_factory=tuple)
  in_trace_idxs: Tuple[int, ...] = field(False, default_factory=tuple)

  out_diagonal: Tuple[int, ...] = field(False, default_factory=tuple)
  in_diagonal: Tuple[Tuple[Optional[int], ...], ...] = field(
      False, default_factory=tuple)

  out_broadcast: Tuple[int, ...] = field(False, default_factory=tuple)
  out_broadcast_idxs: Tuple[int, ...] = field(False, default_factory=tuple)

  in_broadcast: Tuple[int, ...] = field(False, default_factory=tuple)
  in_broadcast_idx: int = field(False, default_factory=int)

  def __and__(self, other):
    """Defines interaction with structure of the other primitive dy2/dw."""
    assert len(self.in_trace) == len(self.out_trace), (self, other)
    assert len(other.in_trace) == len(other.out_trace), (self, other)

    in_trace_idxs = self.in_trace_idxs
    in_trace = tuple(i for i in self.in_trace if i in other.in_trace)

    out_trace = tuple(self.out_trace[i] for i in range(len(self.out_trace))
                      if self.in_trace[i] in other.in_trace
                      )

    assert len(in_trace) == len(out_trace), (self, other)

    out_diagonal = tuple(i for i in self.out_diagonal
                         if i in other.out_diagonal)
    in_diagonal = tuple(i for ix, i in enumerate(self.in_diagonal)
                        if self.out_diagonal[ix] in other.out_diagonal)

    out_broadcast = tuple(i for i in self.out_broadcast
                          if i in other.out_broadcast)

    in_broadcast = tuple(i for i in self.out_broadcast
                         if i in other.out_broadcast)

    return Structure(
        out_trace=out_trace,
        in_trace=in_trace,
        in_trace_idxs=in_trace_idxs,
        out_diagonal=out_diagonal,
        in_diagonal=in_diagonal,
        out_broadcast=out_broadcast,
        out_broadcast_idxs=self.out_broadcast_idxs,
        in_broadcast=in_broadcast,
        in_broadcast_idx=self.in_broadcast_idx,
    )


STRUCTURE_RULES: Dict[Optional[Primitive], Callable[..., Structure]] = {}
JACOBIAN_RULES: Dict[Optional[Primitive], Callable[..., np.ndarray]] = {}
EQN_PARAMS_RULES: Dict[Optional[Primitive], Callable[..., Dict[str, Any]]] = {}


def get_structure(
    eqn: Optional[JaxprEqn],
    invals: List[Union[ShapedArray, AbstractValue]],
    idx: int,
    _s_rules: bool
) -> Structure:
  if any(i is AbstractValue for i in invals):
    raise TypeError(invals)

  if eqn is None:
    # Identity function
    primitive = None
    cts_in = invals[0]
    assert idx == 0

  else:
    if len(eqn.outvars) != 1:
      raise NotImplementedError(eqn)
    cts_in = eqn.outvars[0].aval

    primitive = eqn.primitive
    assert len(invals) == len(eqn.invars)
    assert 0 <= idx < len(eqn.invars)

  if not isinstance(cts_in, ShapedArray):
    raise TypeError(cts_in)

  if primitive in STRUCTURE_RULES and _s_rules:
    structure = STRUCTURE_RULES[primitive](eqn, idx, invals, cts_in)

  else:
    # No simplification rule found.
    structure = Structure()

  # TODO(romann): can we avoid special-casing `reshape`s?
  if primitive == lax.reshape_p:
    cts_in = ShapedArray(invals[idx].shape, invals[idx].dtype)

  # Check that number of trace output and input axes match.
  assert len(structure.in_trace) == len(structure.out_trace)

  # Check that input and output traced sizes are the same.
  out_trace_size = utils.size_at(cts_in, structure.out_trace)
  in_trace_size = utils.size_at(invals[idx], structure.in_trace)
  assert in_trace_size == out_trace_size

  # Check that number of input/output diagonal axes match.
  assert len(structure.out_diagonal) == len(structure.in_diagonal)

  # Check for each output diagonal axis there's only input axes of correct
  # size or `None`. Inval axis should be not `None`.
  for out_d, in_d in zip(structure.out_diagonal, structure.in_diagonal):
    assert len(in_d) == len(invals)
    assert in_d[idx] is not None
    for ix, i in enumerate(in_d):
      if i is not None:
        assert invals[ix].shape[i] == cts_in.shape[out_d]

  return structure


def get_structure_cache(
    jaxpr: Jaxpr,
    _s_rules: bool
) -> Dict[Var, Structure]:
  """Associates a least common structure to each input variable of the `jaxpr`.

  Args:
    jaxpr: Jaxpr to build cache for.
    _s_rules: whether to use structure rules or not.

  Returns:
    A dictionary mapping input variables to the least common structure of all
    primitives it is present in as a direct input.
  """
  invar_to_structure: Dict[Var, Structure] = {}

  for var in jaxpr.invars:
    if var in jaxpr.outvars:
      if isinstance(var, Literal):
        raise TypeError(var)

      # Identity function
      structure = get_id_structure(var.aval, _s_rules)

      if var in invar_to_structure:
        invar_to_structure[var] &= structure
      else:
        invar_to_structure[var] = structure

  for eqn in jaxpr.eqns:
    for i_eqn, var in enumerate(eqn.invars):
      if var in jaxpr.invars:
        if isinstance(var, Literal):
          raise TypeError(var)

        structure = get_structure(
            eqn=eqn,
            invals=[v.aval for v in eqn.invars],
            idx=i_eqn,
            _s_rules=_s_rules
        )

        if var in invar_to_structure:
          invar_to_structure[var] &= structure
        else:
          invar_to_structure[var] = structure

  return invar_to_structure


def get_id_structure(
    inval: AbstractValue,
    _s_rules: bool
) -> Structure:
  if not isinstance(inval, ShapedArray):
    raise TypeError(inval)

  eqn = None
  idx = 0
  invals = [inval]
  return get_structure(eqn, invals, idx, _s_rules)


# UTILS


def _eye_like(out_shaped: ShapedArray, in_shaped: ShapedArray) -> np.ndarray:
  assert out_shaped.size == in_shaped.size, (out_shaped, in_shaped)
  eye = np.eye(out_shaped.size, dtype=out_shaped.dtype)
  eye = eye.reshape(out_shaped.shape + in_shaped.shape)
  return eye


# BINARY PRIMITIVES


def _dot_general_s(
    eqn: JaxprEqn,
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> Structure:
  contracting_dims, batch_dims = eqn.params['dimension_numbers']
  self, other = invals[idx], invals[1 if idx == 0 else 0]

  self_c_dims = contracting_dims[idx]

  self_b_dims = batch_dims[idx]

  in_trace = tuple(i for i in range(self.ndim) if
                   (i not in self_c_dims) and (i not in self_b_dims))
  out_trace = tuple(
      utils.axis_after_dot(i, self_c_dims, self_b_dims,
                           lhs_ndim=None if idx == 0 else other.ndim)
      for i in in_trace
  )

  return Structure(
      out_trace=out_trace,
      in_trace=in_trace,
      in_trace_idxs=(idx,),
      in_diagonal=tuple(zip(*batch_dims)),
      out_diagonal=tuple(range(len(self_b_dims))),
  )

def _dot_general_j(
    eqn: JaxprEqn,
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> np.ndarray:
  contracting_dims, batch_dims = eqn.params['dimension_numbers']

  lhs_c_dims, rhs_c_dims = contracting_dims
  lhs_b_dims, rhs_b_dims = batch_dims

  lhs, rhs = invals

  if idx == 0:
    self = lhs
    self_c_dims, self_b_dims = lhs_c_dims, lhs_b_dims

    other = rhs
    other_c_dims, other_b_dims = rhs_c_dims, rhs_b_dims

  else:
    self = rhs
    self_c_dims, self_b_dims = rhs_c_dims, rhs_b_dims

    other = lhs
    other_c_dims, other_b_dims = lhs_c_dims, lhs_b_dims

  self_ncb_dims = tuple(i for i in range(self.ndim)
                        if i not in self_c_dims + self_b_dims)
  self_nc_dims = tuple(i for i in range(self.ndim)
                       if i not in self_c_dims)

  j = np.moveaxis(
      other,
      other_b_dims + tuple(d[1]
                           for d in sorted(zip(self_c_dims, other_c_dims))),
      tuple(range(len(other_b_dims))) + tuple(range(-len(other_c_dims), 0))
  )

  self_ncb_out = tuple(utils.axis_after_dot(
      i,
      self_c_dims,
      self_b_dims,
      other.ndim if idx == 1 else None
  ) for i in self_ncb_dims)

  self_nc_in = tuple(cts_in.ndim + i for i in self_nc_dims)
  j = np.expand_dims(j, self_ncb_out + self_nc_in)

  self_ncb_size = utils.size_at(self, self_ncb_dims)
  self_ncb_in = tuple(i + cts_in.ndim for i in self_ncb_dims)
  shape = [1 for _ in range(j.ndim)]
  for i_out, i_in in zip(self_ncb_out, self_ncb_in):
    shape[i_out] = shape[i_in] = self.shape[i_in - cts_in.ndim]

  eye = np.eye(self_ncb_size, dtype=np.bool_)
  eye = eye.reshape(shape)
  j = np.where(eye, j, np.zeros((), j.dtype))

  for out_b, (self_b, other_b) in enumerate(zip(self_b_dims, other_b_dims)):
    b_size = other.shape[other_b]
    eye = np.eye(b_size, dtype=np.bool_)
    shape = [1 for _ in range(j.ndim)]
    shape[out_b] = shape[cts_in.ndim + self_b] = b_size
    eye = eye.reshape(shape)
    j = np.where(eye, j, np.zeros((), j.dtype))

  return j

STRUCTURE_RULES[lax.dot_general_p] = _dot_general_s
JACOBIAN_RULES[lax.dot_general_p] = _dot_general_j


def _conv_general_dilated_s(
    eqn: JaxprEqn,
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> Structure:
  if idx != 1:
    raise NotImplementedError(eqn, idx)

  lhs_spec, rhs_spec, out_spec = eqn.params['dimension_numbers']
  batch_group_count = eqn.params['batch_group_count']
  feature_group_count = eqn.params['feature_group_count']
  lhs, rhs = invals

  if (rhs.shape[rhs_spec[0]] == feature_group_count and
      rhs.shape[rhs_spec[1]] == 1):
    assert lhs.shape[lhs_spec[1]] == feature_group_count
    return Structure(
        in_trace=(),
        in_trace_idxs=(),
        out_trace=(),
        in_diagonal=((lhs_spec[1], rhs_spec[0]),),
        out_diagonal=(out_spec[1],)
    )

  elif (lhs.shape[lhs_spec[0]] == batch_group_count and
        rhs.shape[rhs_spec[0]] == batch_group_count):
    return Structure(
        in_trace=(),
        in_trace_idxs=(),
        out_trace=(),
        in_diagonal=((lhs_spec[0], rhs_spec[0]),),
        out_diagonal=(out_spec[1],)
    )

  elif batch_group_count == feature_group_count == 1:
    return Structure(
        in_trace=(rhs_spec[0],),
        in_trace_idxs=(idx,),
        out_trace=(out_spec[1],),
        out_diagonal=(),
        in_diagonal=()
    )

  return Structure()

def _conv_general_dilated_j(
    eqn: JaxprEqn,
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> np.ndarray:
  if idx != 1:
    raise NotImplementedError(eqn, idx)

  lhs = invals[1 if idx == 0 else 0]
  rhs = invals[idx]
  ndim = cts_in.ndim

  lhs_spec, rhs_spec, out_spec = eqn.params['dimension_numbers']
  precision = eqn.params['precision']

  n_groups_f = eqn.params['feature_group_count']
  n_groups_b = eqn.params['batch_group_count']

  n_channels_in = lhs.shape[lhs_spec[1]]
  n_batch_in = lhs.shape[lhs_spec[0]]
  group_size_out = rhs.shape[rhs_spec[0]] // (n_groups_f * n_groups_b)
  group_size_in = n_channels_in // n_groups_f
  batch_size_in = n_batch_in // n_groups_b

  if isinstance(precision, tuple):
    if precision[0] == precision[1]:
      precision = precision[0]
    else:
      raise NotImplementedError(precision)

  filter_shape = tuple(rhs.shape[i] for i in range(ndim) if i in rhs_spec[2:])

  j = lax.conv_general_dilated_patches(
      lhs=lhs,
      filter_shape=filter_shape,
      window_strides=eqn.params['window_strides'],
      padding=eqn.params['padding'],
      lhs_dilation=eqn.params['lhs_dilation'],
      rhs_dilation=eqn.params['rhs_dilation'],
      dimension_numbers=eqn.params['dimension_numbers'],
      precision=precision,
      preferred_element_type=eqn.params['preferred_element_type']
  )

  if n_groups_b > 1:
    j = np.moveaxis(j, (out_spec[0], out_spec[1]), (-1, -2))
    j = j.reshape(j.shape[:-2] +
                  (n_channels_in, *filter_shape, n_groups_b, batch_size_in))
    j = np.moveaxis(j, (-1, -2), (-2, -1))

  else:
    j = np.moveaxis(j, out_spec[1], -1)
    rhs_shape = (n_groups_f, group_size_in) + filter_shape

    j = j.reshape(j.shape[:ndim - 1] + rhs_shape)
    j = np.moveaxis(j, (ndim - 1, ndim), (-1, -2))

  j = np.vectorize(np.diag, signature='(k)->(k,k)')(j)

  if n_groups_b > 1:
    j = np.moveaxis(
        j,
        tuple(range(ndim - 2, j.ndim)),
        [ndim + rhs_spec[1]] +
        [ndim + i for i in sorted(rhs_spec[2:])] +
        [out_spec[0], out_spec[1], ndim + rhs_spec[0]]
    )

  else:
    j = np.moveaxis(
        j,
        tuple(range(ndim - 1, j.ndim)),
        [ndim + i for i in sorted(rhs_spec[2:])] +
        [ndim + rhs_spec[1], out_spec[1], ndim + rhs_spec[0]]
    )

  eye = np.eye(group_size_out, dtype=lhs.dtype)
  eye = np.expand_dims(
      eye,
      [i for i in range(j.ndim) if i not in (out_spec[1], ndim + rhs_spec[0])]
  )
  j = np.kron(j, eye)
  return j

def _conv_general_dilated_e(
    params: Dict[str, Any],
    idx: int,
    trimmed_invals: List[ShapedArray],
    trimmed_cts_in: ShapedArray
) -> Dict[str, Any]:
  # `conv_general_dilated` has `lhs_shape` and `rhs_shape` arguments that are
  # for some reason not inferred from the `lhs` and `rhs` themselves.
  # TODO(romann): ask JAX why these are there.
  dn = params['dimension_numbers']

  if (params['feature_group_count'] == params['lhs_shape'][dn[0][1]] and
      params['feature_group_count'] == params['rhs_shape'][dn[1][0]]):
    params['feature_group_count'] = 1

  if (params['batch_group_count'] == params['rhs_shape'][dn[1][0]] and
      params['batch_group_count'] == params['lhs_shape'][dn[0][0]]):
    params['batch_group_count'] = 1

  lhs, rhs = trimmed_invals
  params['lhs_shape'] = lhs.shape
  params['rhs_shape'] = rhs.shape

  return params

STRUCTURE_RULES[lax.conv_general_dilated_p] = _conv_general_dilated_s
JACOBIAN_RULES[lax.conv_general_dilated_p] = _conv_general_dilated_j
EQN_PARAMS_RULES[lax.conv_general_dilated_p] = _conv_general_dilated_e


def _add_s(
    eqn: JaxprEqn,
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> Structure:
  inval = invals[idx]

  other = invals[1 if idx == 0 else 0]

  if other.ndim == 0:
    # Adding a scalar
    out_trace = tuple(range(inval.ndim))
    out_broadcast = ()

  elif inval.ndim == 0:
    # This array is a scalar
    out_broadcast = tuple(range(other.ndim))
    out_trace = ()

  elif other.ndim == inval.ndim:
    # Adding a broadcastable array.
    out_trace = ()
    out_broadcast = ()

    for i in range(inval.ndim):
      if other.shape[i] in (inval.shape[i], 1):
        # Other array is broadcasted.
        out_trace += (i,)

      elif inval.shape[i] == 1:
        # This array is broadcasted
        out_broadcast += (i,)

      else:
        raise ValueError(inval.shape, other.shape)

  else:
    raise ValueError(inval.ndim, other.ndim)

  return Structure(
      out_trace=out_trace,
      in_trace=out_trace,
      in_trace_idxs=(0, 1),
      out_diagonal=(),
      in_diagonal=(),
      out_broadcast=out_broadcast,
      out_broadcast_idxs=(1 if idx == 0 else 0,)
  )

def _add_j(
    eqn: JaxprEqn,
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray,
    is_sub: bool
) -> np.ndarray:
  j = np.eye(utils.size_at(invals[idx]), dtype=invals[idx].dtype)
  j = j.reshape(invals[idx].shape * 2)
  j = np.broadcast_to(j, cts_in.shape + invals[idx].shape)
  if is_sub and idx == 1:
    j = -j
  return j

STRUCTURE_RULES[lax.add_p] = _add_s
JACOBIAN_RULES[lax.add_p] = functools.partial(_add_j, is_sub=False)

STRUCTURE_RULES[ad.add_jaxvals_p] = _add_s
JACOBIAN_RULES[ad.add_jaxvals_p] = functools.partial(_add_j, is_sub=False)

STRUCTURE_RULES[lax.sub_p] = _add_s
JACOBIAN_RULES[lax.sub_p] = functools.partial(_add_j, is_sub=True)


def _mul_s(
    eqn: JaxprEqn,
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> Structure:
  inval = invals[idx]
  ndim = inval.ndim
  other = invals[1 if idx == 0 else 0]

  out_diagonal = ()
  in_diagonal = ()

  if other.ndim == 0:
    # Multiplication by a scalar
    out_trace = tuple(range(ndim))

  else:
    # Multiplication by a broadcastable array.
    out_trace = ()
    for i in range(ndim):
      if other.shape[i] == 1:
        # Axis `i` is multiplied by a scalar.
        out_trace += (i,)

      else:

        if other.shape[i] == inval.shape[i]:
          out_diagonal += (i,)
          in_diagonal += ((i, i),)

        elif inval.shape[i] == 1:
          # This array is broadcasted
          pass

        else:
          raise ValueError(inval.shape, other.shape)

  in_trace = out_trace
  return Structure(
      out_trace=out_trace,
      in_trace=in_trace,
      in_trace_idxs=(idx,),
      out_diagonal=out_diagonal,
      in_diagonal=in_diagonal,
  )

def _mul_j(
    eqn: JaxprEqn,
    idx: int,
    invals: List[Union[ShapedArray, np.ndarray]],
    cts_in: ShapedArray,
    is_div: bool
) -> np.ndarray:
  if is_div and idx != 0:
    raise ValueError(eqn, idx)

  inval = invals[idx]
  if inval.size == 0:
    return np.zeros(cts_in.shape + inval.shape, inval.dtype)

  other = invals[1 if idx == 0 else 0]
  if is_div:
    other = np.ones((), other.dtype) / other

  if inval.ndim == 0:
    return other

  if other.ndim == 0:
    other = np.broadcast_to(other, inval.shape)

  assert other.ndim == inval.ndim == cts_in.ndim

  j = np.broadcast_to(other, cts_in.shape).reshape((-1,))
  j = np.diag(j)
  j = j.reshape(cts_in.shape * 2)

  sum_axes = ()
  for i in range(inval.ndim):
    if inval.shape[i] == 1:
      sum_axes += (cts_in.ndim + i,)

  j = np.sum(j, axis=sum_axes, keepdims=True)
  return j

STRUCTURE_RULES[lax.mul_p] = _mul_s
JACOBIAN_RULES[lax.mul_p] = functools.partial(_mul_j, is_div=False)

STRUCTURE_RULES[lax.div_p] = _mul_s
JACOBIAN_RULES[lax.div_p] = functools.partial(_mul_j, is_div=True)


# N-ARY PRIMITIVES


def _concatenate_s(
    eqn: JaxprEqn,
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> Structure:
  dimension = eqn.params['dimension']

  out_trace = tuple(i for i in range(cts_in.ndim) if i != dimension)
  in_trace = out_trace

  return Structure(
      out_trace=out_trace,
      in_trace=in_trace,
      in_trace_idxs=tuple(range(len(invals))),
  )

def _concatenate_j(
    eqn: JaxprEqn,
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> np.ndarray:
  dimension = eqn.params['dimension']

  js = []
  inval = invals[idx]
  for i in range(len(invals)):
    inval_i = invals[i]
    inval_i_shape = tuple(inval_i.shape[k] if k == dimension else
                          inval.shape[k] for k in range(inval.ndim))

    if i == idx:
      j = np.eye(inval.size, dtype=inval.dtype)
    else:
      inval_i_size = onp.prod(inval_i_shape)
      j = np.zeros((inval_i_size, inval.size), inval.dtype)

    j = j.reshape(inval_i_shape + inval.shape)
    js.append(j)

  j = lax.concatenate(js, dimension)
  j = j.reshape(cts_in.shape + inval.shape)
  return j

STRUCTURE_RULES[lax.concatenate_p] = _concatenate_s
JACOBIAN_RULES[lax.concatenate_p] = _concatenate_j


# UNARY PRIMITIVES


def _rev_s(
    eqn: JaxprEqn,
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> Structure:
  dimensions = eqn.params['dimensions']
  in_trace = out_trace = tuple(i for i in range(invals[idx].ndim)
                               if i not in dimensions)

  return Structure(
      out_trace=out_trace,
      in_trace=in_trace,
      in_trace_idxs=(idx,),
      out_diagonal=(),
      in_diagonal=(),
  )

def _rev_j(
    eqn: JaxprEqn,
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> np.ndarray:
  inval = invals[idx]
  j = _eye_like(cts_in, inval)
  j = lax.rev(j, eqn.params['dimensions'])
  return j

STRUCTURE_RULES[lax.rev_p] = _rev_s
JACOBIAN_RULES[lax.rev_p] = _rev_j


def _broadcast_in_dim_s(
    eqn: JaxprEqn,
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> Structure:
  broadcast_dimensions = eqn.params['broadcast_dimensions']

  out_trace = broadcast_dimensions
  in_trace = tuple(range(invals[idx].ndim))

  out_broadcast = tuple(i for i in range(cts_in.ndim)
                        if i not in broadcast_dimensions)

  return Structure(
      out_trace=out_trace,
      in_trace=in_trace,
      in_trace_idxs=(idx,),
      out_diagonal=(),
      in_diagonal=(),
      out_broadcast=out_broadcast,
  )

def _broadcast_in_dim_j(
    eqn: JaxprEqn,
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> np.ndarray:
  inval = invals[idx]
  j = np.eye(inval.size, dtype=inval.dtype)
  j = j.reshape(inval.shape * 2)
  j = lax.broadcast_in_dim(
      j,
      cts_in.shape + inval.shape,
      broadcast_dimensions=eqn.params['broadcast_dimensions'] +
      tuple(range(cts_in.ndim, cts_in.ndim + inval.ndim)))
  return j

def _broadcast_in_dim_e(
    params: Dict[str, Any],
    idx: int,
    trimmed_invals: List[ShapedArray],
    trimmed_cts_in: ShapedArray
) -> Dict[str, Any]:
  # `broadcast_in_dim` is the only primitive JVP where we need to change
  # equation parameters in response to tweaking the inputs/cotangents
  # shapes.
  params['shape'] = trimmed_cts_in.shape
  return params

STRUCTURE_RULES[lax.broadcast_in_dim_p] = _broadcast_in_dim_s
JACOBIAN_RULES[lax.broadcast_in_dim_p] = _broadcast_in_dim_j
EQN_PARAMS_RULES[lax.broadcast_in_dim_p] = _broadcast_in_dim_e


def _reduce_sum_s(
    eqn: JaxprEqn,
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> Structure:
  axes = eqn.params['axes']

  out_trace = tuple(range(cts_in.ndim))
  in_trace = tuple(i for i in range(invals[idx].ndim) if i not in axes)

  return Structure(
      out_trace=out_trace,
      in_trace=in_trace,
      in_trace_idxs=(idx,),
      out_diagonal=(),
      in_diagonal=(),
  )

def _reduce_sum_j(
    eqn: JaxprEqn,
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> np.ndarray:
  inval = invals[idx]
  j = np.eye(cts_in.size, dtype=inval.dtype)
  j = j.reshape(cts_in.shape * 2)
  j = np.expand_dims(j, tuple(a + cts_in.ndim for a in  eqn.params['axes']))
  j = np.broadcast_to(j, cts_in.shape + inval.shape)
  return j

STRUCTURE_RULES[lax.reduce_sum_p] = _reduce_sum_s
JACOBIAN_RULES[lax.reduce_sum_p] = _reduce_sum_j


def _reduce_window_sum_s(
    eqn: JaxprEqn,
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> Structure:
  out_trace = ()
  for i in range(cts_in.ndim):
    if (eqn.params['base_dilation'][i] == 1 and
        eqn.params['padding'][i] == (0, 0) and
        eqn.params['window_dilation'][i] == 1 and
        eqn.params['window_dimensions'][i] == 1 and
        eqn.params['window_strides'][i] == 1):
      out_trace += (i,)

  in_trace = out_trace
  return Structure(
      out_trace=out_trace,
      in_trace=in_trace,
      in_trace_idxs=(idx,),
  )

STRUCTURE_RULES[lax.reduce_window_sum_p] = _reduce_window_sum_s


def _pad_s(
    eqn: JaxprEqn,
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> Structure:
  padding_config = eqn.params['padding_config']

  out_trace = tuple(i for i in range(cts_in.ndim)
                    if padding_config[i] == (0, 0, 0))
  in_trace = out_trace

  return Structure(
      out_trace=out_trace,
      in_trace=in_trace,
      in_trace_idxs=(idx,),
      out_diagonal=(),
      in_diagonal=(),
  )

def _pad_j(
    eqn: JaxprEqn,
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> np.ndarray:
  padding_config = eqn.params['padding_config']

  inval = invals[idx]
  j = np.eye(inval.size, dtype=inval.dtype)
  j = j.reshape(inval.shape * 2)
  for _ in range(inval.ndim):
    padding_config += ((0, 0, 0),)

  j = lax.pad(j, np.zeros((), j.dtype), padding_config)
  return j

STRUCTURE_RULES[lax.pad_p] = _pad_s
JACOBIAN_RULES[lax.pad_p] = _pad_j


def _reshape_s(
    eqn: JaxprEqn,
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> Structure:
  out_trace = tuple(range(invals[idx].ndim))
  if eqn.params['dimensions'] is None:
    in_trace = out_trace
  else:
    in_trace = tuple(eqn.params['dimensions'].index(i) for i in out_trace)

  return Structure(
      out_trace=out_trace,
      in_trace=in_trace,
      in_trace_idxs=(idx,),
      out_diagonal=(),
      in_diagonal=(),
  )

def _reshape_j(
    eqn: JaxprEqn,
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> np.ndarray:
  inval = invals[idx]
  j = _eye_like(inval, inval)
  j = j.reshape(inval.shape * 2)

  inval_dims = tuple(i + inval.ndim for i in range(inval.ndim))
  if eqn.params['dimensions'] is not None:
    j = lax.transpose(j, eqn.params['dimensions'] + inval_dims)
  j = j.reshape(inval.shape + inval.shape)
  return j

def _reshape_e(
    params: Dict[str, Any],
    idx: int,
    trimmed_invals: List[ShapedArray],
    trimmed_cts_in: ShapedArray
) -> Dict[str, Any]:
  # Hack for more efficient `reshape` structure rule.
  params['new_sizes'] = trimmed_invals[idx].shape
  return params

STRUCTURE_RULES[lax.reshape_p] = _reshape_s
JACOBIAN_RULES[lax.reshape_p] = _reshape_j
EQN_PARAMS_RULES[lax.reshape_p] = _reshape_e


def _eye_s(
    eqn: Optional[JaxprEqn],
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> Structure:
  """Use this for elementwise-linear in `p` primitives `y(p, x)`.

  Precisely, require that `y(p, x)_k(i) = g(x)(p_i)` for some function `g(x)`
  and an index bijection `k: i -> j`.

  Note: multiplication doesn't satisfy this, since `y(p, x)_i = g(p_i, x_i)`.

  In this case the derivative matrix `dy/dp` is a constant-diagonal matrix, and
  all input-output axes can be collapsed.
  """
  out_trace = tuple(range(cts_in.ndim))
  in_trace = tuple(range(invals[idx].ndim))
  return Structure(
      out_trace=out_trace,
      in_trace=in_trace,
      in_trace_idxs=(idx,),
      out_diagonal=(),
      in_diagonal=(),
  )

def _eye_j(
    eqn: Optional[JaxprEqn],
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> np.ndarray:
  j = _eye_like(cts_in, invals[idx])
  return j


# Identity
STRUCTURE_RULES[None] = _eye_s
JACOBIAN_RULES[None] = _eye_j


def _neg_j(
    eqn: JaxprEqn,
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> np.ndarray:
  j = _eye_like(cts_in, invals[idx])
  return -j

STRUCTURE_RULES[lax.neg_p] = _eye_s
JACOBIAN_RULES[lax.neg_p] = _neg_j


def _zeros_like_j(
    eqn: JaxprEqn,
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> np.ndarray:
  return np.zeros(cts_in.shape + invals[idx].shape, cts_in.dtype)

STRUCTURE_RULES[jax.ad.zeros_like_p] = _eye_s
JACOBIAN_RULES[jax.ad.zeros_like_p] = _zeros_like_j


def _transpose_s(
    eqn: JaxprEqn,
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> Structure:
  in_trace = tuple(range(cts_in.ndim))
  out_trace = tuple(eqn.params['permutation'].index(i) for i in in_trace)

  return Structure(
      out_trace=out_trace,
      in_trace=in_trace,
      in_trace_idxs=(idx,),
      out_diagonal=(),
      in_diagonal=(),
  )

def _transpose_j(
    eqn: JaxprEqn,
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> np.ndarray:
  j = _eye_like(cts_in, invals[idx])
  inval = invals[idx]
  j = j.reshape(inval.shape * 2)

  inval_dims = tuple(i + cts_in.ndim for i in range(cts_in.ndim))
  j = lax.transpose(j, eqn.params['permutation'] + inval_dims)
  j = j.reshape(cts_in.shape + invals[idx].shape)
  return j

STRUCTURE_RULES[lax.transpose_p] = _transpose_s
JACOBIAN_RULES[lax.transpose_p] = _transpose_j


def _squeeze_s(
    eqn: JaxprEqn,
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> Structure:
  out_trace = tuple(range(cts_in.ndim))
  in_trace = tuple(i for i in range(invals[idx].ndim)
                   if i not in eqn.params['dimensions'])
  return Structure(
      out_trace=out_trace,
      in_trace=in_trace,
      in_trace_idxs=(idx,),
      out_diagonal=(),
      in_diagonal=(),
  )

STRUCTURE_RULES[lax.squeeze_p] = _squeeze_s
JACOBIAN_RULES[lax.squeeze_p] = _eye_j


STRUCTURE_RULES[lax.convert_element_type_p] = _eye_s
JACOBIAN_RULES[lax.convert_element_type_p] = _eye_j


device_put_p = jax_dispatch.device_put_p
STRUCTURE_RULES[device_put_p] = _eye_s
JACOBIAN_RULES[device_put_p] = _eye_j


copy_p = jax.lax.copy_p
STRUCTURE_RULES[copy_p] = _eye_s
JACOBIAN_RULES[copy_p] = _eye_j
