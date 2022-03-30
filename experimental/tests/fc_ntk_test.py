from jax import numpy as np
from jax import random
from jax.config import config
from jax import jit
import sys
sys.path.append("./")

config.update("jax_enable_x64", True)
from neural_tangents import stax

from experimental.features import DenseFeatures, ReluFeatures, serial, ReluNTKFeatures



seed = 1
n, d = 6, 5
no_jitting = False

key1, key2 = random.split(random.PRNGKey(seed))
x = random.normal(key1, (n, d))

width = 512  # this does not matter the output
W_std = 1.234  # std of Gaussian random weights

print("================== Result of Neural Tangent Library ==================")

init_fn, _, kernel_fn = stax.serial(stax.Dense(width, W_std=W_std), stax.Relu(),
                                    stax.Dense(width, W_std=W_std), stax.Relu(),
                                    stax.Dense(width, W_std=W_std), stax.Relu(),
                                    stax.Dense(1, W_std=W_std))

kernel_fn = kernel_fn if no_jitting else jit(kernel_fn)
nt_kernel = kernel_fn(x, None)

print("K_nngp :")
print(nt_kernel.nngp)
print()

print("K_ntk :")
print(nt_kernel.ntk)
print()


def test_fc_relu_ntk_approx(relufeat_arg, init_fn=None, feature_fn=None):
    
  print(f"ReluFeatures params:")
  for name_, value_ in relufeat_arg.items():
    print(f"{name_:<12} : {value_}")
  print()

  if init_fn is None or feature_fn is None:
    init_fn, feature_fn = serial(
      DenseFeatures(width, W_std=W_std), ReluFeatures(**relufeat_arg),
      DenseFeatures(width, W_std=W_std), ReluFeatures(**relufeat_arg),
      DenseFeatures(width, W_std=W_std), ReluFeatures(**relufeat_arg),
      DenseFeatures(1, W_std=W_std))

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



print("==================== Result of NTK Random Features ====================")

kappa0_feat_dim = 4096
kappa1_feat_dim = 4096
sketch_dim = 4096

test_fc_relu_ntk_approx({
  'method': 'RANDFEAT',
  'feature_dim0': kappa0_feat_dim,
  'feature_dim1': kappa1_feat_dim,
  'sketch_dim': sketch_dim,
})

print("==================== Result of NTK wih PolySketch ====================")

poly_degree = 4
poly_sketch_dim = 4096
sketch_dim = 4096

test_fc_relu_ntk_approx({
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

test_fc_relu_ntk_approx({
  'method': 'PSRF',
  'feature_dim0': kappa0_feat_dim,
  'sketch_dim': sketch_dim,
  'poly_degree': poly_degree,
  'poly_sketch_dim': poly_sketch_dim
})

print("=========== Result of ReLU-NTK Sketch (one-pass sketching) ===========")

relufeat_arg = {
  'num_layers': 3,
  'poly_degree': 32,
  'poly_sketch_dim': 4096,
  'W_std': W_std,
}

init_fn, feature_fn = ReluNTKFeatures(**relufeat_arg)
test_fc_relu_ntk_approx(relufeat_arg, init_fn, feature_fn)

print("======= (Debug) NTK Feature Maps with Polynomial Approximation =======")
print("\t(*No Sketching algorithm is applied.)")

test_fc_relu_ntk_approx({'method': 'POLY', 'poly_degree': 16})

print("====== (Debug) Exact NTK Feature Maps via Cholesky Decomposition ======")

test_fc_relu_ntk_approx({'method': 'EXACT'})