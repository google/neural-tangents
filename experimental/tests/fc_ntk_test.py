from numpy.linalg import norm
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

key1, key2 = random.split(random.PRNGKey(seed))
x = random.normal(key1, (n, d))

width = 512  # this does not matter the output
W_std = 1.234  # std of Gaussian random weights

print("================== Result of Neural Tangent Library ==================")

init_fn, _, kernel_fn = stax.serial(stax.Dense(width, W_std=W_std), stax.Relu(),
                                    stax.Dense(width, W_std=W_std), stax.Relu(),
                                    stax.Dense(width, W_std=W_std), stax.Relu(),
                                    stax.Dense(1, W_std=W_std))

nt_kernel = kernel_fn(x, None)

print("K_nngp :")
print(nt_kernel.nngp)
print()

print("K_ntk :")
print(nt_kernel.ntk)
print()


def eval_features(f_):
  print(f"f_nngp shape: {f_.nngp_feat.shape}")
  print(f"f_ntk shape: {f_.ntk_feat.shape}")

  print("K_nngp:")
  print(f_.nngp_feat @ f_.nngp_feat.T)
  print()

  print("K_ntk:")
  print(f_.ntk_feat @ f_.ntk_feat.T)
  print()

  print(
      f"|| K_nngp - f_nngp @ f_nngp.T ||_fro = {norm(nt_kernel.nngp - f_.nngp_feat @ f_.nngp_feat.T)}"
  )
  print(
      f"|| K_ntk  -  f_ntk @ f_ntk.T  ||_fro = {norm(nt_kernel.ntk - f_.ntk_feat @ f_.ntk_feat.T)}"
  )
  print()


print("==================== Result of NTK Random Features ====================")

kappa0_feat_dim = 4096
kappa1_feat_dim = 4096
sketch_dim = 4096

relufeat_arg = {
  'method': 'rf',
  'feature_dim0': kappa0_feat_dim,
  'feature_dim1': kappa1_feat_dim,
  'sketch_dim': sketch_dim,
}

print(f"ReluFeatures params:")
for name_, value_ in relufeat_arg.items():
  print(f"{name_:<12} : {value_}")
print()

init_fn, feature_fn = serial(
  DenseFeatures(width, W_std=W_std), ReluFeatures(**relufeat_arg),
  DenseFeatures(width, W_std=W_std), ReluFeatures(**relufeat_arg),
  DenseFeatures(width, W_std=W_std), ReluFeatures(**relufeat_arg),
  DenseFeatures(1, W_std=W_std))

# Initialize random vectors and sketching algorithms
feat_shape, feat_fn_inputs = init_fn(key2, x.shape)

# Transform input vectors to NNGP/NTK feature map
feats = feature_fn(x, feat_fn_inputs)
eval_features(feats)

print("==================== Result of NTK wih PolySketch ====================")

poly_degree = 4
poly_sketch_dim = 4096
sketch_dim = 4096

relufeat_arg = {
  'method': 'ps',
  'sketch_dim': sketch_dim,
  'poly_degree': poly_degree,
  'poly_sketch_dim': poly_sketch_dim
}

print(f"ReluFeatures params:")
for name_, value_ in relufeat_arg.items():
  print(f"{name_:<12} : {value_}")
print()

init_fn, feature_fn = serial(
  DenseFeatures(width, W_std=W_std), ReluFeatures(**relufeat_arg),
  DenseFeatures(width, W_std=W_std), ReluFeatures(**relufeat_arg),
  DenseFeatures(width, W_std=W_std), ReluFeatures(**relufeat_arg, top_layer=True),
  DenseFeatures(1, W_std=W_std))

# Initialize random vectors and sketching algorithms
feat_shape, feat_fn_inputs = init_fn(key2, x.shape)

# Transform input vectors to NNGP/NTK feature map
feats = jit(feature_fn)(x, feat_fn_inputs)
eval_features(feats)

print("=============== Result of PolySketch + Random Features ===============")

kappa0_feat_dim = 2048
sketch_dim = 4096
poly_degree = 4
poly_sketch_dim = 4096

relufeat_arg = {
  'method': 'psrf',
  'feature_dim0': kappa0_feat_dim,
  'sketch_dim': sketch_dim,
  'poly_degree': poly_degree,
  'poly_sketch_dim': poly_sketch_dim
}

print(f"ReluFeatures params:")
for name_, value_ in relufeat_arg.items():
  print(f"{name_:<12} : {value_}")
print()

init_fn, feature_fn = serial(
  DenseFeatures(width, W_std=W_std), ReluFeatures(**relufeat_arg),
  DenseFeatures(width, W_std=W_std), ReluFeatures(**relufeat_arg),
  DenseFeatures(width, W_std=W_std), ReluFeatures(**relufeat_arg, top_layer=True),
  DenseFeatures(1, W_std=W_std))

# Initialize random vectors and sketching algorithms
feat_shape, feat_fn_inputs = init_fn(key2, x.shape)

# Transform input vectors to NNGP/NTK feature map
feats = jit(feature_fn)(x, feat_fn_inputs)
eval_features(feats)

print("=========== Result of ReLU-NTK Sketch (one-pass sketching) ===========")

relufeat_arg = {
  'num_layers': 3,
  'poly_degree': 32,
  'poly_sketch_dim': 4096,
  'W_std': W_std,
}

print(f"ReluFeatures params:")
for name_, value_ in relufeat_arg.items():
  print(f"{name_:<12} : {value_}")
print()

init_fn, feature_fn = ReluNTKFeatures(**relufeat_arg)
_, feat_fn_inputs = init_fn(key2, x.shape)

# Transform input vectors to NNGP/NTK feature map
feats = jit(feature_fn)(x, feat_fn_inputs)
eval_features(feats)


print("======= (Debug) NTK Feature Maps with Polynomial Approximation =======")
print("\t(*No Sketching algorithm is applied.)")

relufeat_arg = {'method': 'poly', 'poly_degree': 64}

print(f"ReluFeatures params:")
for name_, value_ in relufeat_arg.items():
  print(f"{name_:<12} : {value_}")
print()

init_fn, feature_fn = serial(
  DenseFeatures(width, W_std=W_std), ReluFeatures(**relufeat_arg),
  DenseFeatures(width, W_std=W_std), ReluFeatures(**relufeat_arg),
  DenseFeatures(width, W_std=W_std), ReluFeatures(**relufeat_arg),
  DenseFeatures(1, W_std=W_std))

# Initialize random vectors and sketching algorithms
feat_shape, feat_fn_inputs = init_fn(key2, x.shape)

feats = jit(feature_fn)(x, feat_fn_inputs)
eval_features(feats)


print("====== (Debug) Exact NTK Feature Maps via Cholesky Decomposition ======")

relufeat_arg = {'method': 'exact'}

print(f"ReluFeatures params:")
for name_, value_ in relufeat_arg.items():
  print(f"{name_:<12} : {value_}")
print()

init_fn, feature_fn = serial(
  DenseFeatures(width, W_std=W_std), ReluFeatures(**relufeat_arg),
  DenseFeatures(width, W_std=W_std), ReluFeatures(**relufeat_arg),
  DenseFeatures(width, W_std=W_std), ReluFeatures(**relufeat_arg),
  DenseFeatures(1, W_std=W_std))

# Initialize random vectors and sketching algorithms
feat_shape, feat_fn_inputs = init_fn(key2, x.shape)

feats = jit(feature_fn)(x, feat_fn_inputs)
eval_features(feats)