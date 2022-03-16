import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
sys.path.append("./")
import functools
from numpy.linalg import norm
from jax.config import config
from jax import jit
# Enable float64 for JAX
config.update("jax_enable_x64", True)

import jax.numpy as np
from jax import random

from neural_tangents import stax
from experimental.features import ReluFeatures, ConvFeatures, AvgPoolFeatures, serial, FlattenFeatures, DenseFeatures, _inputs_to_features

layer_factor = {5: [2, 1, 1], 7: [2, 2, 2], 10: [3, 3, 3]}
width = 1


def MyrtleNetwork(depth, W_std=np.sqrt(2.0), b_std=0.):
  activation_fn = stax.Relu()
  conv = functools.partial(stax.Conv, W_std=W_std, b_std=b_std, padding='SAME')

  layers = []
  layers += [conv(width, (3, 3)), activation_fn] * layer_factor[depth][0]
  layers += [stax.AvgPool((2, 2), strides=(2, 2))]
  layers += [conv(width, (3, 3)), activation_fn] * layer_factor[depth][1]
  layers += [stax.AvgPool((2, 2), strides=(2, 2))]
  layers += [conv(width, (3, 3)), activation_fn] * layer_factor[depth][2]
  layers += [stax.AvgPool((2, 2), strides=(2, 2))] * 3

  layers += [stax.Flatten(), stax.Dense(1, W_std, b_std)]

  return stax.serial(*layers)


def MyrtleNetworkFeatures(depth, W_std=np.sqrt(2.0), b_std=0., **relu_args):

  conv_fn = functools.partial(ConvFeatures, W_std=W_std, b_std=b_std)

  layers = []
  layers += [conv_fn(width, filter_size=3),
             ReluFeatures(**relu_args)] * layer_factor[depth][0]
  layers += [AvgPoolFeatures(2, 2)]
  layers += [
      ConvFeatures(width, filter_size=3, W_std=W_std),
      ReluFeatures(**relu_args)
  ] * layer_factor[depth][1]
  layers += [AvgPoolFeatures(2, 2)]
  layers += [
      ConvFeatures(width, filter_size=3, W_std=W_std),
      ReluFeatures(**relu_args)
  ] * layer_factor[depth][2]
  layers += [AvgPoolFeatures(2, 2)] * 3
  layers += [FlattenFeatures(), DenseFeatures(1, W_std, b_std)]

  return serial(*layers)


key = random.PRNGKey(0)

N, H, W, C = 4, 32, 32, 3
key1, key2 = random.split(key)
x = random.normal(key1, shape=(N, H, W, C))

_, _, kernel_fn = MyrtleNetwork(5)
kernel_fn = jit(kernel_fn)

print("================= Result of Neural Tangent Library =================")

nt_kernel = kernel_fn(x)
print("K_nngp (exact):")
print(nt_kernel.nngp)
print()

print("K_ntk (exact):")
print(nt_kernel.ntk)
print()

print("================= CNTK Random Features =================")
kappa0_feat_dim = 1000
kappa1_feat_dim = 1000
sketch_dim = 1000

relufeat_arg = {
    'method': 'rf',
    'feature_dim0': kappa0_feat_dim,
    'feature_dim1': kappa1_feat_dim,
    'sketch_dim': sketch_dim,
}

init_fn, feature_fn = MyrtleNetworkFeatures(5, **relufeat_arg)
feature_fn = jit(feature_fn)

init_nngp_feat_shape = x.shape
init_ntk_feat_shape = (-1, 0)
init_feat_shape = (init_nngp_feat_shape, init_ntk_feat_shape)

inputs_shape, feat_fn_inputs = init_fn(key2, init_feat_shape)

f0 = _inputs_to_features(x)
feats = feature_fn(f0, feat_fn_inputs)

print("K_nngp (approx):")
print(feats.nngp_feat @ feats.nngp_feat.T)
print()

print("K_ntk (approx):")
print(feats.ntk_feat @ feats.ntk_feat.T)
print()

print(
    f"|| K_nngp - f_nngp @ f_nngp.T ||_fro = {norm(nt_kernel.nngp - feats.nngp_feat @ feats.nngp_feat.T)}"
)
print(
    f"|| K_ntk  -  f_ntk @ f_ntk.T  ||_fro = {norm(nt_kernel.ntk - feats.ntk_feat @ feats.ntk_feat.T)}"
)
