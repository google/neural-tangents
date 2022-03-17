from numpy.linalg import norm
from jax import random
from jax.config import config
from jax import jit
import sys
sys.path.append("./")

config.update("jax_enable_x64", True)
from neural_tangents import stax

from experimental.features import DenseFeatures, ReluFeatures, serial

seed = 1
n, d = 6, 4

key1, key2 = random.split(random.PRNGKey(seed))
x = random.normal(key1, (n, d))

width = 512  # this does not matter the output

print("================= Result of Neural Tangent Library =================")

init_fn, _, kernel_fn = stax.serial(stax.Dense(width), stax.Relu(),
                                    stax.Dense(width), stax.Relu(),
                                    stax.Dense(1))

nt_kernel = kernel_fn(x, None)

print("K_nngp :")
print(nt_kernel.nngp)
print()

print("K_ntk :")
print(nt_kernel.ntk)
print()

print("================= Result of NTK Random Features =================")

kappa0_feat_dim = 10000
kappa1_feat_dim = 10000
sketch_dim = 20000

relufeat_arg = {
    'method': 'rf',
    'feature_dim0': kappa0_feat_dim,
    'feature_dim1': kappa1_feat_dim,
    'sketch_dim': sketch_dim,
}

init_fn, features_fn = serial(DenseFeatures(width),
                              ReluFeatures(**relufeat_arg),
                              DenseFeatures(width),
                              ReluFeatures(**relufeat_arg), DenseFeatures(1))

# Initialize random vectors and sketching algorithms
feat_shape, feat_fn_inputs = init_fn(key2, x.shape)

# Transform input vectors to NNGP/NTK feature map
feats = jit(features_fn)(x, feat_fn_inputs)

print(f"f_nngp shape: {feat_shape[0]}")
print("K_nngp (approx):")
print(feats.nngp_feat @ feats.nngp_feat.T)
print()

print(f"f_ntk shape: {feat_shape[1]}")
print("K_ntk (approx):")
print(feats.ntk_feat @ feats.ntk_feat.T)
print()

print(
    f"|| K_nngp - f_nngp @ f_nngp.T ||_fro = {norm(nt_kernel.nngp - feats.nngp_feat @ feats.nngp_feat.T)}"
)
print(
    f"|| K_ntk  -  f_ntk @ f_ntk.T  ||_fro = {norm(nt_kernel.ntk - feats.ntk_feat @ feats.ntk_feat.T)}"
)
print()

print("================= (Debug) Exact NTK Feature Maps =================")

relufeat_arg = {'method': 'exact'}

init_fn, features_fn = serial(DenseFeatures(width),
                              ReluFeatures(**relufeat_arg),
                              DenseFeatures(width),
                              ReluFeatures(**relufeat_arg), DenseFeatures(1))

feats = jit(features_fn)(x, feat_fn_inputs)

print("K_nngp :")
print(feats.nngp_feat @ feats.nngp_feat.T)
print()

print("K_ntk :")
print(feats.ntk_feat @ feats.ntk_feat.T)
print()

print(
    f"|| K_nngp - f_nngp @ f_nngp.T ||_fro = {norm(nt_kernel.nngp - feats.nngp_feat @ feats.nngp_feat.T)}"
)
print(
    f"|| K_ntk  -  f_ntk @ f_ntk.T  ||_fro = {norm(nt_kernel.ntk - feats.ntk_feat @ feats.ntk_feat.T)}"
)
