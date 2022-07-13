import enum
from typing import Optional, Callable, Sequence, Tuple
import frozendict
import string
import functools
import operator as op

from jax import lax
from jax import random
from jax._src.util import prod
from jax import numpy as np
import jax.example_libraries.stax as ostax
from jax import eval_shape, ShapedArray

from neural_tangents._src.utils import dataclasses
from neural_tangents._src.utils.typing import Axes
from neural_tangents._src.stax.requirements import _set_req, get_req, _fuse_requirements, _DEFAULT_INPUT_REQ
from neural_tangents._src.stax.combinators import _get_input_req_attr
from neural_tangents._src.stax.linear import _pool_kernel, Padding, _get_dimension_numbers, AggregateImplementation
from neural_tangents._src.stax.linear import _Pooling as Pooling

from experimental.sketching import TensorSRHT, PolyTensorSketch
from experimental.poly_fitting import kappa0_coeffs, kappa1_coeffs, kappa0, kappa1, relu_ntk_coeffs
""" Implementation for NTK Sketching and Random Features """


@dataclasses.dataclass
class Features:
  nngp_feat: np.ndarray
  ntk_feat: np.ndarray

  batch_axis: int = 0
  channel_axis: int = -1

  replace = ...  # type: Callable[..., 'Features']


class ReluFeaturesImplementation(enum.Enum):
  """Method for ReLU NNGP/NTK features approximation."""
  RANDFEAT = 'RANDFEAT'
  POLYSKETCH = 'POLYSKETCH'
  PSRF = 'PSRF'
  POLY = 'POLY'
  EXACT = 'EXACT'


def requires(**static_reqs):

  def req(feature_fn):
    _set_req(feature_fn, frozendict.frozendict(static_reqs))
    return feature_fn

  return req


def layer(layer_fn):

  def new_layer_fns(*args, **kwargs):
    init_fn, feature_fn = layer_fn(*args, **kwargs)
    init_fn = _preprocess_init_fn(init_fn)
    feature_fn = _preprocess_feature_fn(feature_fn)
    return init_fn, feature_fn

  return new_layer_fns


def _preprocess_init_fn(init_fn):

  def init_fn_any(rng, input_shape_any, **kwargs):
    if _is_single_shape(input_shape_any):
      # Add a dummy shape for ntk_feat
      dummy_shape = (-1,) + (0,) * (len(input_shape_any) - 1)
      input_shape = (input_shape_any, dummy_shape, '')
      return init_fn(rng, input_shape, **kwargs)
    else:
      return init_fn(rng, input_shape_any, **kwargs)

  return init_fn_any


def _is_single_shape(input_shape):
  if all(isinstance(n, int) for n in input_shape):
    return True
  elif len(input_shape) == 3 and all(
      _is_single_shape(s) for s in input_shape[:2]):
    return False
  raise ValueError(input_shape)


# For flexible `feature_fn` with both input `np.ndarray` and with `Feature`.
# Followed https://github.com/google/neural-tangents/blob/main/neural_tangents/_src/stax/requirements.py
def _preprocess_feature_fn(feature_fn):

  def feature_fn_feature(feature, input, **kwargs):
    return feature_fn(feature, input, **kwargs)

  def feature_fn_x(x, input, **kwargs):
    feature_fn_reqs = get_req(feature_fn)
    reqs = _fuse_requirements(feature_fn_reqs, _DEFAULT_INPUT_REQ, **kwargs)
    feature = _inputs_to_features(x, **reqs)
    return feature_fn(feature, input, **kwargs)

  def feature_fn_any(x_or_feature, input, **kwargs):
    if isinstance(x_or_feature, Features):
      return feature_fn_feature(x_or_feature, input, **kwargs)
    return feature_fn_x(x_or_feature, input, **kwargs)

  _set_req(feature_fn_any, get_req(feature_fn))
  return feature_fn_any


def _inputs_to_features(x: np.ndarray,
                        batch_axis: int = 0,
                        channel_axis: int = -1,
                        **kwargs) -> Features:
  """Transforms (batches of) inputs to a `Features`."""
  # Followed the same initialization of Neural Tangents library.
  if channel_axis is None:
    x = np.moveaxis(x, batch_axis, 0).reshape((x.shape[batch_axis], -1))
    batch_axis, channel_axis = 0, 1
  else:
    channel_axis %= x.ndim

  nngp_feat = x / x.shape[channel_axis]**0.5
  ntk_feat = np.zeros(x.shape[:channel_axis] + (0,) +
                      x.shape[channel_axis + 1:],
                      dtype=x.dtype)
  return Features(nngp_feat=nngp_feat,
                  ntk_feat=ntk_feat,
                  batch_axis=batch_axis,
                  channel_axis=channel_axis)  # pytype:disable=wrong-keyword-args


# Modified the serial process of feature map blocks.
# Followed https://github.com/google/neural-tangents/blob/main/neural_tangents/stax.py
@layer
def serial(*layers):

  init_fns, feature_fns = zip(*layers)
  init_fn, _ = ostax.serial(*zip(init_fns, init_fns))

  @requires(**_get_input_req_attr(feature_fns, fold=op.rshift))
  def feature_fn(features: Features, inputs, **kwargs) -> Features:
    if not (len(init_fns) == len(feature_fns) == len(inputs)):
      raise ValueError('Length of inputs should be same as that of layers.')
    for feature_fn_, input_ in zip(feature_fns, inputs):
      features = feature_fn_(features, input_, **kwargs)
    return features

  return init_fn, feature_fn


@layer
def DenseFeatures(out_dim: int,
                  W_std: float = 1.,
                  b_std: Optional[float] = None,
                  batch_axis: int = 0,
                  channel_axis: int = -1,
                  parameterization: str = 'ntk'):

  parameterization = parameterization.lower()

  if parameterization != 'ntk':
    raise NotImplementedError(f'Parameterization ({parameterization}) is '
                              ' not implemented yet.')

  def init_fn(rng, input_shape):
    nngp_feat_shape, ntk_feat_shape = input_shape[0], input_shape[1]
    _channel_axis = channel_axis % len(nngp_feat_shape)

    nngp_feat_dim = nngp_feat_shape[_channel_axis] + (1 if b_std is not None
                                                      else 0)
    new_nngp_feat_shape = nngp_feat_shape[:_channel_axis] + (
        nngp_feat_dim,) + nngp_feat_shape[_channel_axis + 1:]

    if prod(ntk_feat_shape) == 0:
      new_ntk_feat_shape = new_nngp_feat_shape
    else:
      ntk_feat_dim = nngp_feat_dim + ntk_feat_shape[_channel_axis]
      new_ntk_feat_shape = ntk_feat_shape[:_channel_axis] + (
          ntk_feat_dim,) + ntk_feat_shape[_channel_axis + 1:]

    return (new_nngp_feat_shape, new_ntk_feat_shape, input_shape[2] + 'D'), ()

  @requires(batch_axis=batch_axis, channel_axis=channel_axis)
  def feature_fn(f: Features, input, **kwargs):
    nngp_feat = f.nngp_feat
    ntk_feat = f.ntk_feat

    _channel_axis = channel_axis % nngp_feat.ndim

    if b_std is not None:  # concatenate bias vector in nngp_feat
      biases = b_std * np.ones(nngp_feat.shape[:_channel_axis] +
                               (1,) + nngp_feat.shape[_channel_axis + 1:],
                               dtype=nngp_feat.dtype)
      nngp_feat = np.concatenate((W_std * nngp_feat, biases),
                                 axis=_channel_axis)
      ntk_feat = W_std * ntk_feat
    else:
      nngp_feat *= W_std
      ntk_feat *= W_std

    ntk_feat = np.concatenate((ntk_feat, nngp_feat), axis=_channel_axis)
    return f.replace(nngp_feat=nngp_feat, ntk_feat=ntk_feat)

  return init_fn, feature_fn


@layer
def ReluFeatures(method: str = 'RANDFEAT',
                 feature_dim0: int = 1,
                 feature_dim1: int = 1,
                 sketch_dim: int = 1,
                 poly_degree: int = 8,
                 poly_sketch_dim: int = 1,
                 generate_rand_mtx: bool = True,
                 batch_axis: int = 0,
                 channel_axis: int = -1):

  method = ReluFeaturesImplementation(method.upper())

  def init_fn(rng, input_shape):
    nngp_feat_shape, ntk_feat_shape = input_shape[0], input_shape[1]

    net_shape = input_shape[2]
    relu_layers_count = net_shape.count('R')
    new_net_shape = net_shape + 'R'

    ndim = len(nngp_feat_shape)
    _channel_axis = channel_axis % ndim

    if method == ReluFeaturesImplementation.RANDFEAT:
      new_nngp_feat_shape = nngp_feat_shape[:_channel_axis] + (
          feature_dim1,) + nngp_feat_shape[_channel_axis + 1:]
      new_ntk_feat_shape = ntk_feat_shape[:_channel_axis] + (
          sketch_dim,) + ntk_feat_shape[_channel_axis + 1:]

      rng1, rng2, rng3 = random.split(rng, 3)
      if generate_rand_mtx:
        # Random vectors for random features of arc-cosine kernel of order 0.
        W0 = random.normal(rng1, (nngp_feat_shape[_channel_axis], feature_dim0))
        # Random vectors for random features of arc-cosine kernel of order 1.
        W1 = random.normal(rng2, (nngp_feat_shape[_channel_axis], feature_dim1))
      else:
        # if `generate_rand_mtx` is False, return random seeds and shapes instead of np.ndarray.
        W0 = (rng1, (nngp_feat_shape[_channel_axis], feature_dim0))
        W1 = (rng2, (nngp_feat_shape[_channel_axis], feature_dim1))

      # TensorSRHT of degree 2 for approximating tensor product.
      tensorsrht = TensorSRHT(rng=rng3,
                              input_dim1=ntk_feat_shape[_channel_axis],
                              input_dim2=feature_dim0,
                              sketch_dim=sketch_dim).init_sketches()  # pytype:disable=wrong-keyword-args

      return (new_nngp_feat_shape, new_ntk_feat_shape,
              new_net_shape), (W0, W1, tensorsrht)

    elif method == ReluFeaturesImplementation.POLYSKETCH:
      new_nngp_feat_shape = nngp_feat_shape[:_channel_axis] + (
          poly_sketch_dim,) + nngp_feat_shape[_channel_axis + 1:]
      new_ntk_feat_shape = ntk_feat_shape[:_channel_axis] + (
          sketch_dim,) + ntk_feat_shape[_channel_axis + 1:]

      rng1, rng2 = random.split(rng, 2)

      new_nngp_feat_shape = nngp_feat_shape[:_channel_axis] + (
          poly_sketch_dim,) + nngp_feat_shape[_channel_axis + 1:]

      kappa1_coeff = kappa1_coeffs(poly_degree, relu_layers_count)
      kappa0_coeff = kappa0_coeffs(poly_degree, relu_layers_count)

      # PolySketch expansion for nngp features.
      if relu_layers_count == 0:
        pts_input_dim = nngp_feat_shape[_channel_axis]
      else:
        pts_input_dim = int(nngp_feat_shape[_channel_axis] / 2 + 0.5)
      polysketch = PolyTensorSketch(rng=rng1,
                                    input_dim=pts_input_dim,
                                    sketch_dim=poly_sketch_dim,
                                    degree=poly_degree).init_sketches()  # pytype:disable=wrong-keyword-args

      # TensorSRHT of degree 2 for approximating tensor product.
      if relu_layers_count == 0:
        ts_input_dim = ntk_feat_shape[_channel_axis]
      else:
        ts_input_dim = int(ntk_feat_shape[_channel_axis] / 2 + 0.5)
      tensorsrht = TensorSRHT(rng=rng2,
                              input_dim1=ts_input_dim,
                              input_dim2=poly_degree *
                              (polysketch.sketch_dim // 4 - 1) + 1,
                              sketch_dim=sketch_dim).init_sketches()  # pytype:disable=wrong-keyword-args

      return (new_nngp_feat_shape, new_ntk_feat_shape,
              new_net_shape), (polysketch, tensorsrht, kappa0_coeff,
                               kappa1_coeff)

    elif method == ReluFeaturesImplementation.PSRF:
      new_nngp_feat_shape = nngp_feat_shape[:_channel_axis] + (
          poly_sketch_dim,) + nngp_feat_shape[_channel_axis + 1:]
      new_ntk_feat_shape = ntk_feat_shape[:_channel_axis] + (
          sketch_dim,) + ntk_feat_shape[_channel_axis + 1:]

      rng1, rng2, rng3 = random.split(rng, 3)

      kappa1_coeff = kappa1_coeffs(poly_degree, relu_layers_count)

      # PolySketch expansion for nngp features.
      if relu_layers_count == 0:
        pts_input_dim = nngp_feat_shape[_channel_axis]
      else:
        pts_input_dim = int(nngp_feat_shape[_channel_axis] / 2 + 0.5)
      polysketch = PolyTensorSketch(rng=rng1,
                                    input_dim=pts_input_dim,
                                    sketch_dim=poly_sketch_dim,
                                    degree=poly_degree).init_sketches()  # pytype:disable=wrong-keyword-args

      # TensorSRHT of degree 2 for approximating tensor product.
      if relu_layers_count == 0:
        ts_input_dim = ntk_feat_shape[_channel_axis]
      else:
        ts_input_dim = int(ntk_feat_shape[_channel_axis] / 2 + 0.5)
      tensorsrht = TensorSRHT(rng=rng2,
                              input_dim1=ts_input_dim,
                              input_dim2=feature_dim0,
                              sketch_dim=sketch_dim).init_sketches()  # pytype:disable=wrong-keyword-args

      # Random vectors for random features of arc-cosine kernel of order 0.
      if relu_layers_count == 0:
        W0 = random.normal(rng3,
                           (nngp_feat_shape[_channel_axis], feature_dim0 // 2))
      else:
        W0 = random.normal(
            rng3,
            (int(nngp_feat_shape[_channel_axis] / 2 + 0.5), feature_dim0 // 2))

      return (new_nngp_feat_shape, new_ntk_feat_shape,
              new_net_shape), (W0, polysketch, tensorsrht, kappa1_coeff)

    elif method == ReluFeaturesImplementation.POLY:
      # This only uses the polynomial approximation without sketching.
      feat_dim = prod(
          tuple(nngp_feat_shape[i]
                for i in range(ndim)
                if i not in [_channel_axis]))

      new_nngp_feat_shape = nngp_feat_shape[:_channel_axis] + (
          feat_dim,) + nngp_feat_shape[_channel_axis + 1:]
      new_ntk_feat_shape = ntk_feat_shape[:_channel_axis] + (
          feat_dim,) + ntk_feat_shape[_channel_axis + 1:]

      kappa1_coeff = kappa1_coeffs(poly_degree, relu_layers_count)
      kappa0_coeff = kappa0_coeffs(poly_degree, relu_layers_count)

      return (new_nngp_feat_shape, new_ntk_feat_shape,
              new_net_shape), (kappa0_coeff, kappa1_coeff)

    elif method == ReluFeaturesImplementation.EXACT:
      # The exact feature map computation is for debug.
      feat_dim = prod(
          tuple(nngp_feat_shape[i]
                for i in range(ndim)
                if i not in [_channel_axis]))

      new_nngp_feat_shape = nngp_feat_shape[:_channel_axis] + (
          feat_dim,) + nngp_feat_shape[_channel_axis + 1:]
      new_ntk_feat_shape = ntk_feat_shape[:_channel_axis] + (
          feat_dim,) + ntk_feat_shape[_channel_axis + 1:]

      return (new_nngp_feat_shape, new_ntk_feat_shape, new_net_shape), ()

    else:
      raise NotImplementedError(f'Invalid method name: {method}')

  @requires(batch_axis=batch_axis, channel_axis=channel_axis)
  def feature_fn(f: Features, input, **kwargs) -> Features:
    ndim = len(f.nngp_feat.shape)
    _channel_axis = channel_axis % ndim
    spatial_axes = tuple(
        f.nngp_feat.shape[i] for i in range(ndim) if i != _channel_axis)

    def _convert_to_original(x):
      return np.moveaxis(x.reshape(spatial_axes + (-1,)), -1, _channel_axis)

    def _convert_to_2d(x):
      feat_dim = x.shape[_channel_axis]
      return np.moveaxis(x, _channel_axis, -1).reshape(-1, feat_dim)

    nngp_feat_2d = _convert_to_2d(f.nngp_feat)
    if prod(f.ntk_feat.shape) != 0:
      ntk_feat_2d = _convert_to_2d(f.ntk_feat)

    if method == ReluFeaturesImplementation.RANDFEAT:  # Random Features approach.
      if generate_rand_mtx:
        W0: np.ndarray = input[0]
        W1: np.ndarray = input[1]
      else:
        W0 = random.normal(input[0][0], shape=input[0][1])
        W1 = random.normal(input[1][0], shape=input[1][1])
      tensorsrht: TensorSRHT = input[2]

      kappa0_feat = (nngp_feat_2d @ W0 > 0) / W0.shape[-1]**0.5
      del W0
      nngp_feat = (np.maximum(nngp_feat_2d @ W1, 0) / W1.shape[-1]**0.5)
      del W1
      ntk_feat = tensorsrht.sketch(ntk_feat_2d, kappa0_feat, real_output=True)

      nngp_feat = _convert_to_original(nngp_feat)
      ntk_feat = _convert_to_original(ntk_feat)

    elif method == ReluFeaturesImplementation.POLYSKETCH:
      polysketch: PolyTensorSketch = input[0]
      tensorsrht: TensorSRHT = input[1]
      kappa0_coeff: np.ndarray = input[2]
      kappa1_coeff: np.ndarray = input[3]

      norms = np.linalg.norm(nngp_feat_2d, axis=-1, keepdims=True)
      norms = np.maximum(norms, 1e-12)

      nngp_feat_2d /= norms
      ntk_feat_2d /= norms

      # Apply PolySketch to approximate feature maps of kappa0 & kappa1 kernels.
      polysketch_feats = polysketch.sketch(nngp_feat_2d)
      kappa1_feat = polysketch.expand_feats(polysketch_feats, kappa1_coeff)
      kappa0_feat = polysketch.expand_feats(polysketch_feats, kappa0_coeff)
      del polysketch_feats

      # Apply SRHT to kappa1_feat so that dimension of nngp_feat is poly_sketch_dim//2.
      nngp_feat = polysketch.standardsrht(kappa1_feat)

      # Apply TensorSRHT to ntk_feat_2d and kappa0_feat to approximate their tensor product.
      ntk_feat = tensorsrht.sketch(ntk_feat_2d, kappa0_feat)

      nngp_feat *= norms
      ntk_feat *= norms

      nngp_feat = _convert_to_original(nngp_feat)
      ntk_feat = _convert_to_original(ntk_feat)

    elif method == ReluFeaturesImplementation.PSRF:  # Combination of PolySketch and Random Features.
      W0: np.ndarray = input[0]
      polysketch: PolyTensorSketch = input[1]
      tensorsrht: TensorSRHT = input[2]
      kappa1_coeff: np.ndarray = input[3]

      norms = np.linalg.norm(nngp_feat_2d, axis=-1, keepdims=True)
      norms = np.maximum(norms, 1e-12)

      nngp_feat_2d /= norms
      ntk_feat_2d /= norms

      # Apply PolySketch to approximate feature maps of kappa1 kernels.
      polysketch_feats = polysketch.sketch(nngp_feat_2d)
      kappa1_feat = polysketch.expand_feats(polysketch_feats, kappa1_coeff)
      del polysketch_feats

      nngp_feat = polysketch.standardsrht(kappa1_feat)

      nngp_proj = nngp_feat_2d @ W0
      kappa0_feat = np.concatenate(
          ((nngp_proj > 0), (nngp_proj <= 0)), axis=1) / W0.shape[-1]**0.5
      del W0

      # Apply TensorSRHT to ntk_feat_2d and kappa0_feat to approximate their tensor product.
      ntk_feat = tensorsrht.sketch(ntk_feat_2d, kappa0_feat)

      nngp_feat *= norms
      ntk_feat *= norms

      nngp_feat = _convert_to_original(nngp_feat)
      ntk_feat = _convert_to_original(ntk_feat)

    elif method == ReluFeaturesImplementation.POLY:  # Polynomial approximation without sketching.
      kappa0_coeff: np.ndarray = input[0]
      kappa1_coeff: np.ndarray = input[1]

      norms = np.linalg.norm(nngp_feat_2d, axis=-1, keepdims=True)
      norms = np.maximum(norms, 1e-12)

      nngp_feat_2d /= norms
      ntk_feat_2d /= norms

      gram_nngp = np.dot(nngp_feat_2d, nngp_feat_2d.T)
      nngp_feat = _cholesky(np.polyval(kappa1_coeff[::-1], gram_nngp))

      ntk = ntk_feat_2d @ ntk_feat_2d.T
      kappa0_mat = np.polyval(kappa0_coeff[::-1], gram_nngp)
      ntk_feat = _cholesky(ntk * kappa0_mat)

      nngp_feat *= norms
      ntk_feat *= norms

      nngp_feat = _convert_to_original(nngp_feat)
      ntk_feat = _convert_to_original(ntk_feat)

    elif method == ReluFeaturesImplementation.EXACT:  # Exact feature map computations via Cholesky decomposition.
      nngp_feat = _convert_to_original(
          _cholesky(kappa1(nngp_feat_2d, is_x_matrix=True)))

      if prod(f.ntk_feat.shape) != 0:
        ntk = ntk_feat_2d @ ntk_feat_2d.T
        kappa0_mat = kappa0(nngp_feat_2d, is_x_matrix=True)
        ntk_feat = _convert_to_original(_cholesky(ntk * kappa0_mat))
      else:
        ntk_feat = f.ntk_feat

    else:
      raise NotImplementedError(f'Invalid method name: {method}')

    if method != ReluFeaturesImplementation.RANDFEAT:
      ntk_feat /= 2.0**0.5
      nngp_feat /= 2.0**0.5

    return f.replace(nngp_feat=nngp_feat, ntk_feat=ntk_feat)

  return init_fn, feature_fn


def _cholesky(mat):
  return np.linalg.cholesky(mat + 1e-8 * np.eye(mat.shape[0]))


@layer
def ReluNTKFeatures(
    num_layers: int,
    poly_degree: int = 16,
    poly_sketch_dim: int = 1024,
    W_std: float = 1.,
):

  def init_fn(rng, input_shape):
    input_dim = input_shape[0][-1]

    # PolySketch expansion for nngp/ntk features.
    polysketch = PolyTensorSketch(rng=rng,
                                  input_dim=input_dim,
                                  sketch_dim=poly_sketch_dim,
                                  degree=poly_degree).init_sketches()  # pytype:disable=wrong-keyword-args

    nngp_coeffs, ntk_coeffs = relu_ntk_coeffs(poly_degree, num_layers)

    return (), (polysketch, nngp_coeffs, ntk_coeffs)

  def feature_fn(f, input=None, **kwargs):
    input_shape = f.nngp_feat.shape[:-1]

    polysketch: PolyTensorSketch = input[0]
    nngp_coeffs: np.ndarray = input[1]
    ntk_coeffs: np.ndarray = input[2]

    polysketch_feats = polysketch.sketch(f.nngp_feat)
    nngp_feat = polysketch.expand_feats(polysketch_feats, nngp_coeffs)
    ntk_feat = polysketch.expand_feats(polysketch_feats, ntk_coeffs)

    # Apply SRHT to features so that dimensions are poly_sketch_dim//2.
    nngp_feat = polysketch.standardsrht(nngp_feat).reshape(input_shape + (-1,))
    ntk_feat = polysketch.standardsrht(ntk_feat).reshape(input_shape + (-1,))

    # Convert complex features to real ones.
    ntk_feat = np.concatenate((ntk_feat.real, ntk_feat.imag), axis=-1)
    nngp_feat = np.concatenate((nngp_feat.real, nngp_feat.imag), axis=-1)

    return f.replace(nngp_feat=nngp_feat, ntk_feat=ntk_feat)

  return init_fn, feature_fn


@layer
def ConvFeatures(out_chan: int,
                 filter_shape: Sequence[int],
                 strides: Optional[Sequence[int]] = None,
                 padding: str = 'SAME',
                 W_std: float = 1.0,
                 b_std: Optional[float] = None,
                 dimension_numbers: Optional[Tuple[str, str, str]] = None,
                 parameterization: str = 'ntk'):

  parameterization = parameterization.lower()

  if dimension_numbers is None:
    dimension_numbers = _get_dimension_numbers(len(filter_shape), False)

  lhs_spec, rhs_spec, out_spec = dimension_numbers

  channel_axis = lhs_spec.index('C')

  patch_size = prod(filter_shape)

  if parameterization != 'ntk':
    raise NotImplementedError(f'Parameterization ({parameterization}) is '
                              ' not implemented yet.')

  def init_fn(rng, input_shape):
    nngp_feat_shape, ntk_feat_shape = input_shape[0], input_shape[1]

    nngp_feat_dim = nngp_feat_shape[channel_axis] * patch_size
    if b_std is not None:
      nngp_feat_dim += 1

    nngp_feat_dim = nngp_feat_shape[channel_axis] * patch_size + (
        1 if b_std is not None else 0)
    ntk_feat_dim = nngp_feat_dim + ntk_feat_shape[channel_axis] * patch_size

    new_nngp_feat_shape = nngp_feat_shape[:channel_axis] + (
        nngp_feat_dim,) + nngp_feat_shape[channel_axis + 1:]
    new_ntk_feat_shape = ntk_feat_shape[:channel_axis] + (
        ntk_feat_dim,) + ntk_feat_shape[channel_axis + 1:]

    return (new_nngp_feat_shape, new_ntk_feat_shape, input_shape[2] + 'C'), ()

  @requires(batch_axis=lhs_spec.index('N'), channel_axis=lhs_spec.index('C'))
  def feature_fn(f: Features, input, **kwargs):

    nngp_feat = f.nngp_feat

    _channel_axis = channel_axis % nngp_feat.ndim

    nngp_feat = _concat_shifted_features_2d(
        nngp_feat, filter_shape, dimension_numbers) * W_std / patch_size**0.5

    if b_std is not None:
      biases = b_std * np.ones(nngp_feat.shape[:_channel_axis] +
                               (1,) + nngp_feat.shape[_channel_axis + 1:],
                               dtype=nngp_feat.dtype)
      nngp_feat = np.concatenate((nngp_feat, biases), axis=_channel_axis)

    if prod(f.ntk_feat.shape) == 0:  # if ntk_feat is empty skip feature concat
      ntk_feat = nngp_feat
    else:
      ntk_feat = _concat_shifted_features_2d(
          f.ntk_feat, filter_shape, dimension_numbers) * W_std / patch_size**0.5
      ntk_feat = np.concatenate((ntk_feat, nngp_feat), axis=_channel_axis)

    return f.replace(nngp_feat=nngp_feat,
                     ntk_feat=ntk_feat,
                     batch_axis=out_spec.index('N'),
                     channel_axis=out_spec.index('C'))

  return init_fn, feature_fn


def _concat_shifted_features_2d(X: np.ndarray,
                                filter_shape: Sequence[int],
                                dimension_numbers: Optional[Tuple[str, str,
                                                                  str]] = None):
  return lax.conv_general_dilated_patches(X,
                                          filter_shape=filter_shape,
                                          window_strides=(1, 1),
                                          padding='SAME',
                                          dimension_numbers=dimension_numbers)


@layer
def AvgPoolFeatures(window_shape: Sequence[int],
                    strides: Optional[Sequence[int]] = None,
                    padding: str = 'VALID',
                    normalize_edges: bool = False,
                    batch_axis: int = 0,
                    channel_axis: int = -1):

  if window_shape[0] != strides[0] or window_shape[1] != strides[1]:
    raise NotImplementedError('window_shape should be equal to strides.')

  channel_axis %= 4
  spec = ''.join(
      c for c in string.ascii_uppercase if c not in ('N', 'C'))[:len(strides)]
  for a in sorted((batch_axis, channel_axis % (2 + len(strides)))):
    if a == batch_axis:
      spec = spec[:a] + 'N' + spec[a:]
    else:
      spec = spec[:a] + 'C' + spec[a:]

  _kernel_window_shape = lambda x_: tuple(
      [x_[0] if s == 'A' else x_[0] if s == 'B' else 1 for s in spec])
  window_shape_kernel = _kernel_window_shape(window_shape)
  strides_kernel = _kernel_window_shape(strides)

  pooling = lambda x: _pool_kernel(x, Pooling.AVG,
                                   window_shape_kernel, strides_kernel,
                                   Padding(padding), normalize_edges, 0)

  def init_fn(rng, input_shape):
    nngp_feat_shape, ntk_feat_shape = input_shape[0], input_shape[1]

    new_nngp_feat_shape = eval_shape(pooling,
                                     ShapedArray(nngp_feat_shape,
                                                 np.float32)).shape
    new_ntk_feat_shape = eval_shape(pooling,
                                    ShapedArray(ntk_feat_shape,
                                                np.float32)).shape

    return (new_nngp_feat_shape, new_ntk_feat_shape, input_shape[2] + 'A'), ()

  @requires(batch_axis=batch_axis, channel_axis=channel_axis)
  def feature_fn(f: Features, input, **kwargs):
    nngp_feat = f.nngp_feat
    ntk_feat = f.ntk_feat

    nngp_feat = pooling(nngp_feat)

    if prod(f.ntk_feat.shape) == 0:  # check if ntk_feat is empty
      ntk_feat = nngp_feat
    else:
      ntk_feat = pooling(ntk_feat)

    return f.replace(nngp_feat=nngp_feat, ntk_feat=ntk_feat)

  return init_fn, feature_fn


@layer
def GlobalAvgPoolFeatures(batch_axis: int = 0, channel_axis: int = -1):

  def init_fn(rng, input_shape):
    nngp_feat_shape, ntk_feat_shape = input_shape[0], input_shape[1]
    ndim = len(nngp_feat_shape)
    non_spatial_axes = (batch_axis % ndim, channel_axis % ndim)
    _get_output_shape = lambda _shape: tuple(_shape[i]
                                             for i in range(ndim)
                                             if i in non_spatial_axes)
    new_nngp_feat_shape = _get_output_shape(nngp_feat_shape)
    new_ntk_feat_shape = _get_output_shape(ntk_feat_shape)

    return (new_nngp_feat_shape, new_ntk_feat_shape, input_shape[2]), ()

  @requires(batch_axis=batch_axis, channel_axis=channel_axis)
  def feature_fn(f: Features, input, **kwargs):
    nngp_feat = f.nngp_feat
    ntk_feat = f.ntk_feat

    ndim = len(nngp_feat.shape)
    non_spatial_axes = (batch_axis % ndim, channel_axis % ndim)
    spatial_axes = tuple(set(range(ndim)) - set(non_spatial_axes))

    nngp_feat = np.mean(nngp_feat, axis=spatial_axes)
    ntk_feat = np.mean(ntk_feat, axis=spatial_axes)

    batch_first = batch_axis % ndim < channel_axis % ndim
    return f.replace(nngp_feat=nngp_feat,
                     ntk_feat=ntk_feat,
                     batch_axis=0 if batch_first else 1,
                     channel_axis=1 if batch_first else 0)

  return init_fn, feature_fn


@layer
def FlattenFeatures(batch_axis: int = 0, batch_axis_out: int = 0):

  if batch_axis_out in (0, -2):
    batch_axis_out = 0
    channel_axis_out = 1
  elif batch_axis_out in (1, -1):
    batch_axis_out = 1
    channel_axis_out = 0
  else:
    raise ValueError(f'`batch_axis_out` must be 0 or 1, got {batch_axis_out}.')

  def get_output_shape(input_shape):
    batch_size = input_shape[batch_axis]
    channel_size = functools.reduce(
        op.mul, input_shape[:batch_axis] +
        input_shape[(batch_axis + 1) or len(input_shape):], 1)
    if batch_axis_out == 0:
      return batch_size, channel_size
    return channel_size, batch_size

  def init_fn(rng, input_shape):
    nngp_feat_shape, ntk_feat_shape = input_shape[0], input_shape[1]
    new_nngp_feat_shape = get_output_shape(nngp_feat_shape)
    new_ntk_feat_shape = get_output_shape(ntk_feat_shape)

    return (new_nngp_feat_shape, new_ntk_feat_shape, input_shape[2] + 'F'), ()

  @requires(batch_axis=batch_axis, channel_axis=None)
  def feature_fn(f: Features, input, **kwargs):
    nngp_feat = f.nngp_feat

    batch_size = nngp_feat.shape[batch_axis]
    nngp_feat_dim = prod(
        nngp_feat.shape) / batch_size / f.nngp_feat.shape[f.channel_axis]
    nngp_feat = nngp_feat.reshape(batch_size, -1) / nngp_feat_dim**0.5

    if prod(f.ntk_feat.shape) != 0:  # check if ntk_feat is not empty
      ntk_feat_dim = prod(
          f.ntk_feat.shape) / batch_size / f.ntk_feat.shape[f.channel_axis]
      ntk_feat = f.ntk_feat.reshape(batch_size, -1) / ntk_feat_dim**0.5
    else:
      ntk_feat = f.ntk_feat.reshape(batch_size, -1)

    return f.replace(nngp_feat=nngp_feat,
                     ntk_feat=ntk_feat,
                     batch_axis=batch_axis_out,
                     channel_axis=channel_axis_out)

  return init_fn, feature_fn


@layer
def LayerNormFeatures(axis: Axes = -1,
                      eps: float = 1e-12,
                      batch_axis: int = 0,
                      channel_axis: int = -1):

  def init_fn(rng, input_shape):
    return input_shape, ()

  @requires(batch_axis=batch_axis, channel_axis=channel_axis)
  def feature_fn(f: Features, input, **kwargs):
    norms = np.linalg.norm(f.nngp_feat, keepdims=True, axis=channel_axis)
    norms = np.maximum(norms, eps)

    nngp_feat = f.nngp_feat / norms
    ntk_feat = f.ntk_feat / norms if prod(f.ntk_feat.shape) != 0 else f.ntk_feat
    return f.replace(nngp_feat=nngp_feat, ntk_feat=ntk_feat)

  return init_fn, feature_fn


@layer
def AggregateFeatures(
    aggregate_axis: Optional[Axes] = None,
    batch_axis: int = 0,
    channel_axis: int = -1,
    to_dense: Optional[Callable[[np.ndarray], np.ndarray]] = lambda p: p,
    implementation: str = AggregateImplementation.DENSE.value):

  def init_fn(rng, input_shape):
    return input_shape, ()

  @requires(batch_axis=batch_axis, channel_axis=channel_axis)
  def feature_fn(f: Features, input=None, pattern=None, **kwargs):
    if pattern is None:
      raise NotImplementedError('`pattern=None` is not implemented.')

    nngp_feat = f.nngp_feat
    ntk_feat = f.ntk_feat

    pattern_T = np.swapaxes(pattern, 1, 2)
    nngp_feat = np.einsum("bnm,bmc->bnc", pattern_T, nngp_feat)

    if prod(f.ntk_feat.shape) != 0:  # check if ntk_feat is not empty
      ntk_feat = np.einsum("bnm,bmc->bnc", pattern_T, ntk_feat)
    else:
      ntk_feat = nngp_feat

    return f.replace(nngp_feat=nngp_feat, ntk_feat=ntk_feat)

  return init_fn, feature_fn
