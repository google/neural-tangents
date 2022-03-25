from typing import Optional, Callable
from jax import random
from jax import numpy as np
from jax.numpy.linalg import cholesky
import jax.example_libraries.stax as ostax
from jax.numpy import linalg as LA

from neural_tangents import stax
from neural_tangents._src.utils import dataclasses
from neural_tangents._src.stax.linear import _pool_kernel, Padding
from neural_tangents._src.stax.linear import _Pooling as Pooling

from sketching import TensorSRHT2, PolyTensorSketch, CmplxTensorSRHT
from poly_fitting import kappa0_coeffs, kappa1_coeffs
""" Implementation for NTK Sketching and Random Features """


def _prod(tuple_):
  prod = 1
  for x in tuple_:
    prod = prod * x
  return prod


def poly_expansion(x, coeffs):
  y = np.ones_like(x)
  results = np.zeros_like(x)
  for c in coeffs:
    results += c*y
    y = y*x
  return results


@dataclasses.dataclass
class Features:
  nngp_feat: Optional[np.ndarray] = None
  ntk_feat: Optional[np.ndarray] = None
  norms: Optional[np.ndarray] = None

  batch_axis: int = 0
  channel_axis: int = -1

  replace = ...  # type: Callable[..., 'Features']


def _inputs_to_features(x: np.ndarray,
                        batch_axis: int = 0,
                        channel_axis: int = -1,
                        **kwargs) -> Features:
  """Transforms (batches of) inputs to a `Features`."""

  # Followed the same initialization of Neural Tangents library.
  nngp_feat = x / x.shape[channel_axis]**0.5
  norms = LA.norm(nngp_feat, axis=channel_axis)
  norms = np.where(norms>0, norms, 1.0)
  nngp_feat = (nngp_feat.T / norms).T

  ntk_feat = np.array([0.0], dtype=nngp_feat.dtype)

  return Features(nngp_feat=nngp_feat,
                  ntk_feat=ntk_feat,
                  norms=norms,
                  batch_axis=batch_axis,
                  channel_axis=channel_axis)  # pytype:disable=wrong-keyword-args


# For flexible `feature_fn` with both input `np.ndarray` and with `Feature`.
# Followed https://github.com/google/neural-tangents/blob/main/neural_tangents/_src/stax/requirements.py
def _preprocess_feature_fn(feature_fn):

  def feature_fn_feature(feature, input, **kwargs):
    return feature_fn(feature, input, **kwargs)

  def feature_fn_x(x, input, **kwargs):
    feature = _inputs_to_features(x, **kwargs)
    return feature_fn(feature, input, **kwargs)

  def feature_fn_any(x_or_feature, input=None, **kwargs):
    if isinstance(x_or_feature, Features):
      return feature_fn_feature(x_or_feature, input, **kwargs)
    return feature_fn_x(x_or_feature, input, **kwargs)

  return feature_fn_any


def _is_sinlge_shape(input_shape):
  if all(isinstance(n, int) for n in input_shape):
    return True
  return False


def _is_defaut_feature(feat):
  return feat.ndim == 1


def _preprocess_init_fn(init_fn):

  def init_fn_any(rng, input_shape_any, **kwargs):
    if _is_sinlge_shape(input_shape_any):
      input_shape = (input_shape_any, (-1, 0))
      return init_fn(rng, input_shape, **kwargs)
    else:
      return init_fn(rng, input_shape_any, **kwargs)

  return init_fn_any


def layer(layer_fn):

  def new_layer_fns(*args, **kwargs):
    init_fn, feature_fn = layer_fn(*args, **kwargs)
    feature_fn = _preprocess_feature_fn(feature_fn)
    init_fn = _preprocess_init_fn(init_fn)
    return init_fn, feature_fn

  return new_layer_fns


# Modified the serial process of feature map blocks.
# Followed https://github.com/google/neural-tangents/blob/main/neural_tangents/stax.py
@layer
def serial(*layers):
  init_fns, feature_fns = zip(*layers)
  init_fn, _ = ostax.serial(*zip(init_fns, init_fns))

  def feature_fn(k, inputs, **kwargs):
    for f, input_ in zip(feature_fns, inputs):
      k = f(k, input_, **kwargs)
    return k

  return init_fn, feature_fn


@layer
def DenseFeatures(out_dim: int,
                  W_std: float = 1.,
                  b_std: float = 0.,
                  batch_axis: int = 0,
                  channel_axis: int = -1):

  if b_std != 0.0:
    raise NotImplementedError('Non-zero b_std is not implemented yet .'
                              ' Please set b_std to be `0`.')

  def init_fn(rng, input_shape):
    nngp_feat_shape, ntk_feat_shape = input_shape[0], input_shape[1]
    new_ntk_feat_shape = nngp_feat_shape[:-1] + (nngp_feat_shape[-1] +
                                                 ntk_feat_shape[-1],)
    
    if len(input_shape) > 2:
        return (nngp_feat_shape, new_ntk_feat_shape, input_shape[2]+'D'), ()
    else:
        return (nngp_feat_shape, new_ntk_feat_shape, 'D'), ()

  def feature_fn(f: Features, input, **kwargs):
    nngp_feat, ntk_feat = f.nngp_feat, f.ntk_feat
    nngp_feat *= W_std
    ntk_feat *= W_std

    if _is_defaut_feature(ntk_feat):  # check if ntk_feat is empty
      ntk_feat = nngp_feat
    else:
      ntk_feat = np.concatenate((ntk_feat, nngp_feat), axis=channel_axis)

    return f.replace(nngp_feat=nngp_feat, ntk_feat=ntk_feat)

  return init_fn, feature_fn


@layer
def ReluFeatures(
    feature_dim0: int = 1,
    feature_dim1: int = 1,
    sketch_dim: int = 1,
    poly_degree: int = 4,
    poly_sketch_dim: int = 1,
    method: str = 'rf',
    top_layer: bool = False
):

    method = method.lower()
    assert method in ['rf', 'ps', 'exact', 'psrf']
    
    def init_fn(rng, input_shape):
        nngp_feat_shape, ntk_feat_shape = input_shape[0], input_shape[1]
        new_nngp_feat_shape = nngp_feat_shape[:-1] + (feature_dim1,)
        new_ntk_feat_shape = ntk_feat_shape[:-1] + (sketch_dim,)
        net_shape = input_shape[2]
        layer_count = len(net_shape)//2+1
        
        if method == 'rf':
            rng1, rng2, rng3 = random.split(rng, 3)
            # Random vectors for random features of arc-cosine kernel of order 0.
            W0 = random.normal(rng1, (nngp_feat_shape[-1], feature_dim0))
            # Random vectors for random features of arc-cosine kernel of order 1.
            W1 = random.normal(rng2, (nngp_feat_shape[-1], feature_dim1))
            # TensorSRHT of degree 2 for approximating tensor product.
            ts2 = TensorSRHT2(rng=rng3,
                              input_dim1=ntk_feat_shape[-1],
                              input_dim2=feature_dim0,
                              sketch_dim=sketch_dim).init_sketches()  # pytype:disable=wrong-keyword-args
            return (new_nngp_feat_shape, new_ntk_feat_shape, net_shape+'R'), (W0, W1, ts2)

        elif method == 'psrf':
            new_nngp_feat_shape = nngp_feat_shape[:-1] + (poly_sketch_dim,)
            rng1, rng2, rng3 = random.split(rng, 3)
            
            kappa1_coeff = kappa1_coeffs(poly_degree,layer_count-1)
            
            # PolySketch expansion for nngp features.
            polysketch = PolyTensorSketch(rng1, nngp_feat_shape[-1]//(1+(layer_count>1)),  
                                          poly_sketch_dim, poly_degree).init_sketch()
            # TensorSRHT of degree 2 for approximating tensor product.
            tensorsrht = CmplxTensorSRHT(input_dim1=ntk_feat_shape[-1]//(1+(layer_count>1)), 
                                         input_dim2=feature_dim0, 
                                         sketch_dim=sketch_dim , rng=rng2).init_sketches()

            # Random vectors for random features of arc-cosine kernel of order 0.
            # W0 = random.choice(rng3, 2, shape=(nngp_feat_shape[-1], feature_dim0//2)) * 2 - 1
            if layer_count ==1:
                W0 = random.normal(rng3, (2*nngp_feat_shape[-1], feature_dim0//2))
            else:
                W0 = random.normal(rng3, (nngp_feat_shape[-1], feature_dim0//2),dtype='float32')

            return (new_nngp_feat_shape, new_ntk_feat_shape, net_shape+'R'), (W0, polysketch, tensorsrht, (kappa1_coeff, layer_count))

        elif method == 'ps':
            new_nngp_feat_shape = nngp_feat_shape[:-1] + (poly_sketch_dim,)
            rng1, rng2, rng3 = random.split(rng, 3)
            
            kappa1_coeff = kappa1_coeffs(poly_degree,layer_count-1) 
            kappa0_coeff = kappa0_coeffs(poly_degree,layer_count-1)

            # PolySketch expansion for nngp features.
            polysketch = PolyTensorSketch(rng1, nngp_feat_shape[-1]//(1+(layer_count>1)), 
                                          poly_sketch_dim, poly_degree).init_sketch()
            # TensorSRHT of degree 2 for approximating tensor product.
            tensorsrht = CmplxTensorSRHT(input_dim1=ntk_feat_shape[-1]//(1+(layer_count>1)), 
                                         input_dim2=poly_degree*(polysketch.sketch_dim//4-1)+1, 
                                         sketch_dim=sketch_dim , rng=rng2).init_sketches()

            return (new_nngp_feat_shape, new_ntk_feat_shape, net_shape+'R'), (polysketch, tensorsrht,(kappa0_coeff,kappa1_coeff, layer_count))
            raise NotImplementedError
          
        elif method == 'exact':
            # The exact feature map computation is for debug.
            new_nngp_feat_shape = nngp_feat_shape[:-1] + (_prod(
                nngp_feat_shape[:-1]),)
            new_ntk_feat_shape = ntk_feat_shape[:-1] + (_prod(ntk_feat_shape[:-1]),)
            
            kappa1_coeff = kappa1_coeffs(poly_degree,layer_count-1) 
            kappa0_coeff = kappa0_coeffs(poly_degree,layer_count-1)
            
            return (new_nngp_feat_shape, new_ntk_feat_shape, net_shape+'R'), (kappa0_coeff, kappa1_coeff, layer_count)
    
    def feature_fn(f: Features, input=None, **kwargs) -> Features:
    
        input_shape = f.nngp_feat.shape[:-1]
        nngp_feat_dim = f.nngp_feat.shape[-1]
        ntk_feat_dim = f.ntk_feat.shape[-1]
          
        nngp_feat_2d = f.nngp_feat.reshape(-1, nngp_feat_dim)
        ntk_feat_2d = f.ntk_feat.reshape(-1, ntk_feat_dim)
          
        if method == 'rf':  # Random Features approach.
            W0: np.ndarray = input[0]
            W1: np.ndarray = input[1]
            ts2: TensorSRHT2 = input[2]
            
            kappa0_feat = (nngp_feat_2d @ W0 > 0) / np.sqrt(W0.shape[-1])
            del W0
            nngp_feat = (np.maximum(nngp_feat_2d @ W1, 0) /
                         np.sqrt(W1.shape[-1])).reshape(input_shape + (-1,))
            del W1
            ntk_feat = ts2.sketch(ntk_feat_2d,
                                  kappa0_feat).reshape(input_shape + (-1,))
        
        elif method == 'psrf':  # Combination of Poly Sketch and Random Features.
            W0: np.ndarray = input[0]
            polysketch: PolyTensorSketch = input[1]
            tensorsrht: CmplxTensorSRHT = input[2]
            kappa1_coeff: np.ndarray = input[3][0]
            layer_count = input[3][1]
            
            polysketch_feats = polysketch.sketch(nngp_feat_2d)
            kappa1_feat = polysketch.expand_feats(polysketch_feats, kappa1_coeff)
            del polysketch_feats
            nngp_feat = polysketch.standardsrht(kappa1_feat, polysketch.rand_inds, 
                                                polysketch.rand_signs).reshape(input_shape + (-1,))
            # nngp_feat = np.concatenate((nngp_feat.real, nngp_feat.imag), axis=1)

            nngp_proj = np.dot(np.concatenate((nngp_feat_2d.real, nngp_feat_2d.imag), axis=1) , W0)
            
            kappa0_feat = np.concatenate(((nngp_proj > 0), (nngp_proj <= 0)), axis=1) / np.sqrt(W0.shape[-1])
            del W0
            ntk_feat = tensorsrht.sketch(ntk_feat_2d, kappa0_feat).reshape(input_shape + (-1,))
            if top_layer:
                ntk_feat = (1 / 2**(layer_count/2))*np.concatenate((ntk_feat.real, ntk_feat.imag), axis=1)
                nngp_feat = (1 / 2**(layer_count/2))*np.concatenate((nngp_feat.real, nngp_feat.imag), axis=1)

          
        elif method == 'ps':
            polysketch: PolyTensorSketch = input[0]
            tensorsrht: CmplxTensorSRHT = input[1]
            kappa0_coeff: np.ndarray = input[2][0]
            kappa1_coeff: np.ndarray = input[2][1]
            layer_count = input[2][2]
            
            polysketch_feats = polysketch.sketch(nngp_feat_2d)
            kappa1_feat = polysketch.expand_feats(polysketch_feats, kappa1_coeff)
            nngp_feat = polysketch.standardsrht(kappa1_feat, polysketch.rand_inds, 
                                                polysketch.rand_signs).reshape(input_shape + (-1,))
            
            kappa0_feat = polysketch.expand_feats(polysketch_feats, kappa0_coeff)
            del polysketch_feats
            ntk_feat = tensorsrht.sketch(ntk_feat_2d, kappa0_feat).reshape(input_shape + (-1,))
            
            if top_layer:
                ntk_feat = (1 / 2**(layer_count/2))*np.concatenate((ntk_feat.real, ntk_feat.imag), axis=1)
                nngp_feat = (1 / 2**(layer_count/2))*np.concatenate((nngp_feat.real, nngp_feat.imag), axis=1)

          
        elif method == 'exact':
            
            kappa0_coeff: np.ndarray = input[0]
            kappa1_coeff: np.ndarray = input[1]
            layer_count = input[2]
            
            gram_nngp = np.dot(nngp_feat_2d, nngp_feat_2d.T)
            nngp_feat = cholesky(poly_expansion(gram_nngp, kappa1_coeff)).reshape(input_shape + (-1,))
            
            
            ntk = ntk_feat_2d @ ntk_feat_2d.T
            kappa0_mat = poly_expansion(gram_nngp, kappa0_coeff)
            # kappa0_mat = kappa0(nngp_feat_2d)
            ntk_feat = cholesky(ntk * kappa0_mat).reshape(input_shape + (-1,))
            
            if top_layer:
                ntk_feat = (1 / 2**(layer_count/2))*ntk_feat
                nngp_feat = (1 / 2**(layer_count/2))*nngp_feat
          
        else:
          raise NotImplementedError
        
        if top_layer:
            ntk_feat = (ntk_feat.T * f.norms).T
            nngp_feat = (nngp_feat.T * f.norms).T
          
          
        return f.replace(nngp_feat=nngp_feat, ntk_feat=ntk_feat)
    
    return init_fn, feature_fn


def _conv_feat(X, filter_size):
  N, H, W, C = X.shape
  out = np.zeros((N, H, W, C * filter_size))
  out = out.at[:, :, :, :C].set(X)
  j = 1
  for i in range(1, min((filter_size + 1) // 2, W)):
    out = out.at[:, :, :-i, j * C:(j + 1) * C].set(X[:, :, i:])
    j += 1
    out = out.at[:, :, i:, j * C:(j + 1) * C].set(X[:, :, :-i])
    j += 1
  return out


def _conv2d_feat(X, filter_size):
  return _conv_feat(np.moveaxis(_conv_feat(X, filter_size), 1, 2), filter_size)


@layer
def ConvFeatures(out_dim: int,
                 filter_size: int,
                 W_std: float = 1.0,
                 b_std: float = 0.,
                 channel_axis: int = -1):

  if b_std != 0.0:
    raise NotImplementedError('Non-zero b_std is not implemented yet .'
                              ' Please set b_std to be `0`.')

  def init_fn(rng, input_shape):
    nngp_feat_shape, ntk_feat_shape = input_shape[0], input_shape[1]
    new_nngp_feat_shape = nngp_feat_shape[:-1] + (nngp_feat_shape[-1] *
                                                  filter_size**2,)
    new_ntk_feat_shape = nngp_feat_shape[:-1] + (
        (nngp_feat_shape[-1] + ntk_feat_shape[-1]) * filter_size**2,)
    return (new_nngp_feat_shape, new_ntk_feat_shape), ()

  def feature_fn(f, input, **kwargs):
    nngp_feat, ntk_feat = f.nngp_feat, f.ntk_feat

    nngp_feat = _conv2d_feat(nngp_feat, filter_size) / filter_size * W_std

    if _is_defaut_feature(ntk_feat):  # check if ntk_feat is empty
      ntk_feat = nngp_feat
    else:
      ntk_feat = _conv2d_feat(ntk_feat, filter_size) / filter_size * W_std
      ntk_feat = np.concatenate((ntk_feat, nngp_feat), axis=channel_axis)

    return f.replace(nngp_feat=nngp_feat, ntk_feat=ntk_feat)

  return init_fn, feature_fn


@layer
def AvgPoolFeatures(window_size: int,
                    stride_size: int = 2,
                    padding: str = stax.Padding.VALID.name,
                    normalize_edges: bool = False,
                    batch_axis: int = 0,
                    channel_axis: int = -1):

  def init_fn(rng, input_shape):
    nngp_feat_shape, ntk_feat_shape = input_shape[0], input_shape[1]

    new_nngp_feat_shape = nngp_feat_shape[:1] + (
        nngp_feat_shape[1] // window_size,
        nngp_feat_shape[2] // window_size) + nngp_feat_shape[-1:]
    new_ntk_feat_shape = ntk_feat_shape[:1] + (
        ntk_feat_shape[1] // window_size,
        ntk_feat_shape[2] // window_size) + ntk_feat_shape[-1:]
    return (new_nngp_feat_shape, new_ntk_feat_shape), ()

  def feature_fn(f, input=None, **kwargs):
    window_shape_kernel = (1,) + (window_size,) * 2 + (1,)
    strides_kernel = (1,) + (window_size,) * 2 + (1,)
    pooling = lambda x: _pool_kernel(x, Pooling.AVG,
                                     window_shape_kernel, strides_kernel,
                                     Padding(padding), normalize_edges, 0)
    nngp_feat = pooling(f.nngp_feat)
    ntk_feat = pooling(f.ntk_feat)

    return f.replace(nngp_feat=nngp_feat, ntk_feat=ntk_feat)

  return init_fn, feature_fn


@layer
def FlattenFeatures(batch_axis: int = 0, batch_axis_out: int = 0):

  def init_fn(rng, input_shape):
    nngp_feat_shape, ntk_feat_shape = input_shape[0], input_shape[1]
    new_nngp_feat_shape = nngp_feat_shape[:1] + (_prod(nngp_feat_shape[1:]),)
    new_ntk_feat_shape = ntk_feat_shape[:1] + (_prod(ntk_feat_shape[1:]),)
    return (new_nngp_feat_shape, new_ntk_feat_shape), ()

  def feature_fn(f, input=None, **kwargs):
    batch_size = f.nngp_feat.shape[0]
    nngp_feat = f.nngp_feat.reshape(batch_size, -1) / np.sqrt(
        _prod(f.nngp_feat.shape[1:-1]))
    if _is_defaut_feature(f.ntk_feat):  # check if ntk_feat is empty
      ntk_feat = f.ntk_feat
    else:
      ntk_feat = f.ntk_feat.reshape(batch_size, -1) / np.sqrt(
          _prod(f.ntk_feat.shape[1:-1]))

    return f.replace(nngp_feat=nngp_feat, ntk_feat=ntk_feat)

  return init_fn, feature_fn
