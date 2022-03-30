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
from experimental.features import ReluFeatures, ConvFeatures, AvgPoolFeatures, serial, FlattenFeatures, DenseFeatures

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


def MyrtleNetworkFeatures(depth, W_std=np.sqrt(2.0), **relu_args):

  conv_fn = functools.partial(ConvFeatures, W_std=W_std)

  layers = []
  layers += [conv_fn(width, filter_shape=(3, 3)),
             ReluFeatures(**relu_args)] * layer_factor[depth][0]
  layers += [AvgPoolFeatures((2, 2), strides=(2, 2))]
  layers += [
      ConvFeatures(width, filter_shape=(3, 3), W_std=W_std),
      ReluFeatures(**relu_args)
  ] * layer_factor[depth][1]
  layers += [AvgPoolFeatures((2, 2), strides=(2, 2))]
  layers += [
      ConvFeatures(width, filter_shape=(3, 3), W_std=W_std),
      ReluFeatures(**relu_args)
  ] * layer_factor[depth][2]
  layers += [AvgPoolFeatures((2, 2), strides=(2, 2))] * 3
  layers += [FlattenFeatures(), DenseFeatures(1, W_std)]

  return serial(*layers)


key = random.PRNGKey(0)

N, H, W, C = 4, 32, 32, 3
depth = 5
no_jitting = False

key1, key2 = random.split(key)
x = random.normal(key1, shape=(N, H, W, C))

print("================= Result of Neural Tangent Library =================")

_, _, kernel_fn = MyrtleNetwork(depth)
kernel_fn = kernel_fn if no_jitting else jit(kernel_fn)

nt_kernel = kernel_fn(x)
print("K_nngp (exact):")
print(nt_kernel.nngp)
print()

print("K_ntk (exact):")
print(nt_kernel.ntk)
print()


def test_myrtle_network_approx(relufeat_arg, init_fn=None, feature_fn=None):

  print(f"ReluFeatures params:")
  for name_, value_ in relufeat_arg.items():
    print(f"{name_:<12} : {value_}")
  print()

  if init_fn is None or feature_fn is None:
    init_fn, feature_fn = MyrtleNetworkFeatures(depth, **relufeat_arg)

  # Initialize random vectors and sketching algorithms
  _, feat_fn_inputs = init_fn(key2, x.shape)

  # Transform input vectors to NNGP/NTK feature map
  feature_fn = feature_fn if no_jitting else jit(feature_fn)
  feats = feature_fn(x, feat_fn_inputs)

  # PolySketch returns complex features. Convert complex features to real ones.
  if np.iscomplexobj(feats.nngp_feat) or np.iscomplexobj(feats.ntk_feat):
    nngp_feat = np.concatenate((feats.nngp_feat.real, feats.nngp_feat.imag), axis=-1)
    ntk_feat = np.concatenate((feats.ntk_feat.real, feats.ntk_feat.imag), axis=-1)
    feats = feats.replace(nngp_feat=nngp_feat, ntk_feat=ntk_feat)

  print(f"f_nngp shape: {feats.nngp_feat.shape}")
  print(f"f_ntk shape: {feats.ntk_feat.shape}")

  print("K_nngp:")
  print(feats.nngp_feat @ feats.nngp_feat.T)
  print()

  print("K_ntk:")
  print(feats.ntk_feat @ feats.ntk_feat.T)
  print()

  print(
      f"|| K_nngp - f_nngp @ f_nngp.T ||_fro = {np.linalg.norm(nt_kernel.nngp - feats.nngp_feat @ feats.nngp_feat.T)}"
  )
  print(
      f"|| K_ntk  -  f_ntk @ f_ntk.T  ||_fro = {np.linalg.norm(nt_kernel.ntk - feats.ntk_feat @ feats.ntk_feat.T)}"
  )
  print()



print("=================== Result of CNTK Random Features ===================")
kappa0_feat_dim = 1024
kappa1_feat_dim = 1024
sketch_dim = 1024

test_myrtle_network_approx({
    'method': 'RANDFEAT',
    'feature_dim0': kappa0_feat_dim,
    'feature_dim1': kappa1_feat_dim,
    'sketch_dim': sketch_dim,
})

print("==================== Result of CNTK wih PolySketch ====================")
poly_degree = 16
poly_sketch_dim = 4096
sketch_dim = 4096

test_myrtle_network_approx({
  'method': 'POLYSKETCH',
  'sketch_dim': sketch_dim,
  'poly_degree': poly_degree,
  'poly_sketch_dim': poly_sketch_dim
})

print("=============== Result of PolySketch + Random Features ===============")
kappa0_feat_dim = 2048
sketch_dim = 4096
poly_degree = 4
poly_sketch_dim = 4096

test_myrtle_network_approx({
  'method': 'PSRF',
  'feature_dim0': kappa0_feat_dim,
  'sketch_dim': sketch_dim,
  'poly_degree': poly_degree,
  'poly_sketch_dim': poly_sketch_dim
})

print("====== (Debug) Exact NTK Feature Maps via Cholesky Decomposition ======")

test_myrtle_network_approx({'method': 'EXACT'})