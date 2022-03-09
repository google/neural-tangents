from typing import Optional, Callable
from jax import random
from jax import numpy as np
from jax.numpy.linalg import cholesky
import jax.example_libraries.stax as ostax

from neural_tangents import stax
from neural_tangents._src.utils import dataclasses
from neural_tangents._src.stax.linear import _pool_kernel, Padding
from neural_tangents._src.stax.linear import _Pooling as Pooling

from experimental.sketching import TensorSRHT2
""" Implementation for NTK Sketching and Random Features """


def _prod(tuple_):
  prod = 1
  for x in tuple_:
    prod = prod * x
  return prod


# Arc-cosine kernel functions is for debugging.
def _arccos(x):
  return np.arccos(np.clip(x, -1, 1))


def _sqrt(x):
  return np.sqrt(np.maximum(x, 1e-20))


def kappa0(x):
  xxt = x @ x.T
  prod = np.outer(np.linalg.norm(x, axis=-1)**2, np.linalg.norm(x, axis=-1)**2)
  return (1 - _arccos(xxt / _sqrt(prod)) / np.pi) / 2


def kappa1(x):
  xxt = x @ x.T
  prod = np.outer(np.linalg.norm(x, axis=-1)**2, np.linalg.norm(x, axis=-1)**2)
  return (_sqrt(prod - xxt**2) +
          (np.pi - _arccos(xxt / _sqrt(prod))) * xxt) / np.pi / 2


@dataclasses.dataclass
class Features:
  nngp_feat: Optional[np.ndarray] = None
  ntk_feat: Optional[np.ndarray] = None

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
  ntk_feat = np.empty((), dtype=nngp_feat.dtype)

  return Features(nngp_feat=nngp_feat,
                  ntk_feat=ntk_feat,
                  batch_axis=batch_axis,
                  channel_axis=channel_axis)  # pytype:disable=wrong-keyword-args


# Modified the serial process of feature map blocks.
# Followed https://github.com/google/neural-tangents/blob/main/neural_tangents/stax.py
def serial(*layers):
  init_fns, apply_fns, feature_fns = zip(*layers)
  init_fn, apply_fn = ostax.serial(*zip(init_fns, apply_fns))

  # import time
  def feature_fn(k, inputs, **kwargs):
    for f, input_ in zip(feature_fns, inputs):
      # print(f)
      # tic = time.time()
      k = f(k, input_, **kwargs)
      # print(f"toc: {time.time() - tic:.2f} sec")
    return k

  return init_fn, apply_fn, feature_fn


def DenseFeatures(out_dim: int,
                  W_std: float = 1.,
                  b_std: float = 1.,
                  parameterization: str = 'ntk',
                  batch_axis: int = 0,
                  channel_axis: int = -1):

  def init_fn(rng, input_shape):
    nngp_feat_shape, ntk_feat_shape = input_shape[0], input_shape[1]
    new_ntk_feat_shape = nngp_feat_shape[:-1] + (nngp_feat_shape[-1] +
                                                 ntk_feat_shape[-1],)
    return (nngp_feat_shape, new_ntk_feat_shape), ()

  def apply_fn(**kwargs):
    return None

  def kernel_fn(f: Features, input, **kwargs):
    nngp_feat, ntk_feat = f.nngp_feat, f.ntk_feat
    nngp_feat *= W_std
    ntk_feat *= W_std

    if ntk_feat.ndim == 0:  # check if ntk_feat is empty
      ntk_feat = nngp_feat
    else:
      ntk_feat = np.concatenate((ntk_feat, nngp_feat), axis=channel_axis)

    return f.replace(nngp_feat=nngp_feat, ntk_feat=ntk_feat)

  return init_fn, apply_fn, kernel_fn


def ReluFeatures(
    feature_dim0: int = 1,
    feature_dim1: int = 1,
    sketch_dim: int = 1,
    poly_degree0: int = 4,
    poly_degree1: int = 4,
    poly_sketch_dim0: int = 1,
    poly_sketch_dim1: int = 1,
    method: str = 'rf',
):

  method = method.lower()
  assert method in ['rf', 'ps', 'exact']

  def init_fn(rng, input_shape):
    nngp_feat_shape, ntk_feat_shape = input_shape[0], input_shape[1]
    new_nngp_feat_shape = nngp_feat_shape[:-1] + (feature_dim1,)
    new_ntk_feat_shape = ntk_feat_shape[:-1] + (sketch_dim,)

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
      return (new_nngp_feat_shape, new_ntk_feat_shape), (W0, W1, ts2)

    elif method == 'ps':
      # rng1, rng2, rng3 = random.split(rng, 3)
      # # PolySketch algorithm for arc-cosine kernel of order 0.
      # ps0 = PolyTensorSRHT(rng1, nngp_feat_shape[-1], poly_sketch_dim0,
      #                      poly_degree0)
      # # PolySketch algorithm for arc-cosine kernel of order 1.
      # ps1 = PolyTensorSRHT(rng2, nngp_feat_shape[-1], poly_sketch_dim1,
      #                      poly_degree1)
      # # TensorSRHT of degree 2 for approximating tensor product.
      # ts2 = TensorSRHT2(rng3, ntk_feat_shape[-1], feature_dim0, sketch_dim)
      # return (new_nngp_feat_shape, new_ntk_feat_shape), (ps0, ps1, ts2)
      raise NotImplementedError

    elif method == 'exact':
      # The exact feature map computation is for debug.
      new_nngp_feat_shape = nngp_feat_shape[:-1] + (_prod(
          nngp_feat_shape[:-1]),)
      new_ntk_feat_shape = ntk_feat_shape[:-1] + (_prod(ntk_feat_shape[:-1]),)
      return (new_nngp_feat_shape, new_ntk_feat_shape), ()

  def apply_fn(**kwargs):
    return None

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
      nngp_feat = (np.maximum(nngp_feat_2d @ W1, 0) /
                   np.sqrt(W1.shape[-1])).reshape(input_shape + (-1,))
      ntk_feat = ts2.sketch(ntk_feat_2d,
                            kappa0_feat).reshape(input_shape + (-1,))

    elif method == 'ps':
      # ps0: PolyTensorSRHT = input[0]
      # ps1: PolyTensorSRHT = input[1]
      # ts2: TensorSRHT2 = input[2]
      raise NotImplementedError

    elif method == 'exact':  # Exact feature extraction via Cholesky decomposition.
      nngp_feat = cholesky(kappa1(nngp_feat_2d)).reshape(input_shape + (-1,))

      ntk = ntk_feat_2d @ ntk_feat_2d.T
      kappa0_mat = kappa0(nngp_feat_2d)
      ntk_feat = cholesky(ntk * kappa0_mat).reshape(input_shape + (-1,))

    else:
      raise NotImplementedError

    return f.replace(nngp_feat=nngp_feat, ntk_feat=ntk_feat)

  return init_fn, apply_fn, feature_fn


def conv_feat(X, filter_size):
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


def conv2d_feat(X, filter_size):
  return conv_feat(np.moveaxis(conv_feat(X, filter_size), 1, 2), filter_size)


def ConvFeatures(out_dim: int,
                 filter_size: int,
                 W_std: float = 1.0,
                 b_std: float = 0.,
                 channel_axis: int = -1):

  def init_fn(rng, input_shape):
    nngp_feat_shape, ntk_feat_shape = input_shape[0], input_shape[1]
    new_nngp_feat_shape = nngp_feat_shape[:-1] + (nngp_feat_shape[-1] *
                                                  filter_size**2,)
    new_ntk_feat_shape = nngp_feat_shape[:-1] + (
        (nngp_feat_shape[-1] + ntk_feat_shape[-1]) * filter_size**2,)
    return (new_nngp_feat_shape, new_ntk_feat_shape), ()

  def apply_fn(**kwargs):
    return None

  def feature_fn(f, input, **kwargs):
    nngp_feat, ntk_feat = f.nngp_feat, f.ntk_feat

    nngp_feat = conv2d_feat(nngp_feat, filter_size) / filter_size * W_std

    if ntk_feat.ndim == 0:  # check if ntk_feat is empty
      ntk_feat = nngp_feat
    else:
      ntk_feat = conv2d_feat(ntk_feat, filter_size) / filter_size * W_std
      ntk_feat = np.concatenate((ntk_feat, nngp_feat), axis=channel_axis)

    return f.replace(nngp_feat=nngp_feat, ntk_feat=ntk_feat)

  return init_fn, apply_fn, feature_fn


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

  def apply_fn(**kwargs):
    return None

  def feature_fn(f, input=None, **kwargs):
    window_shape_kernel = (1,) + (window_size,) * 2 + (1,)
    strides_kernel = (1,) + (window_size,) * 2 + (1,)
    pooling = lambda x: _pool_kernel(x, Pooling.AVG,
                                     window_shape_kernel, strides_kernel,
                                     Padding(padding), normalize_edges, 0)
    nngp_feat = pooling(f.nngp_feat)
    ntk_feat = pooling(f.ntk_feat)

    return f.replace(nngp_feat=nngp_feat, ntk_feat=ntk_feat)

  return init_fn, apply_fn, feature_fn


def FlattenFeatures(batch_axis: int = 0, batch_axis_out: int = 0):

  def init_fn(rng, input_shape):
    nngp_feat_shape, ntk_feat_shape = input_shape[0], input_shape[1]
    new_nngp_feat_shape = nngp_feat_shape[:1] + (_prod(nngp_feat_shape[1:]),)
    new_ntk_feat_shape = ntk_feat_shape[:1] + (_prod(ntk_feat_shape[1:]),)
    return (new_nngp_feat_shape, new_ntk_feat_shape), ()

  def apply_fn(**kwargs):
    return None

  def feature_fn(f, input=None, **kwargs):
    batch_size = f.nngp_feat.shape[0]
    nngp_feat = f.nngp_feat.reshape(batch_size, -1) / np.sqrt(
        _prod(f.nngp_feat.shape[1:-1]))
    ntk_feat = f.ntk_feat.reshape(batch_size, -1) / np.sqrt(
        _prod(f.ntk_feat.shape[1:-1]))

    return f.replace(nngp_feat=nngp_feat, ntk_feat=ntk_feat)

  return init_fn, apply_fn, feature_fn
