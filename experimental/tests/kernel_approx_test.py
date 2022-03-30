import jax
from jax import numpy as np
from jax import random
from jax.config import config
from jax import jit
import sys

sys.path.append("./")

config.update("jax_enable_x64", True)
import neural_tangents as nt
from neural_tangents._src import empirical
from neural_tangents import stax

from experimental.features import DenseFeatures, ReluFeatures, serial, ReluNTKFeatures


def _generate_fc_relu_ntk(width, depth, W_std):
  layers = []
  layers += [stax.Dense(width, W_std=W_std), stax.Relu()] * depth
  layers += [stax.Dense(output_dim, W_std=W_std)]
  init_fn, apply_f, kernel_fn = stax.serial(*layers)
  return init_fn, apply_f, kernel_fn


# This is re-implementation of neural_tangents.empirical_ntk_fn.
# The result is same with "nt.empirical_ntk_fn(apply_fn)(x, None, params)"
def _get_grad(x, output_dim, params, apply_fn):

  f_output = empirical._get_f_params(apply_fn, x, None, None, {})
  jac_f_output = jax.jacobian(f_output)
  jacobian = jac_f_output(params)

  grad_all = []
  for jac_ in jacobian:
    if len(jac_) > 0:
      for j_ in jac_:
        if j_ is None or np.linalg.norm(j_) < 1e-10:
          continue
        grad_all.append(j_.reshape(n, -1))

  grad_all = np.hstack(grad_all)
  return grad_all / np.sqrt(output_dim)


def _get_grad_feat_dim(input_dim, width, output_dim, depth):
  dim_1 = input_dim * width
  dim_2 = np.asarray([width**2 for _ in range(depth - 1)]).sum()
  dim_3 = width * output_dim
  return (dim_1 + dim_2 + dim_3) * output_dim - dim_3


def fc_relu_ntk_sketching(relufeat_arg,
                          rng,
                          init_fn=None,
                          feature_fn=None,
                          W_std=1.,
                          depth=-1,
                          no_jitting=False):

  if init_fn is None or feature_fn is None:
    layers = []
    layers += [
        DenseFeatures(1, W_std=W_std),
        ReluFeatures(**relufeat_arg),
    ] * depth
    layers += [DenseFeatures(1, W_std=W_std)]
    init_fn, feature_fn = serial(*layers)

  # Initialize random vectors and sketching algorithms
  _, feat_fn_inputs = init_fn(rng, x.shape)

  # Transform input vectors to NNGP/NTK feature map
  feature_fn = feature_fn if no_jitting else jit(feature_fn)
  feats = feature_fn(x, feat_fn_inputs)

  # PolySketch returns complex features. Convert complex features to real ones.
  if np.iscomplexobj(feats.ntk_feat):
    return np.concatenate((feats.ntk_feat.real, feats.ntk_feat.imag), axis=-1)
  return feats.ntk_feat


seed = 1
n, d = 1000, 28 * 28
no_jitting = False

key1, key2, key3 = random.split(random.PRNGKey(seed), 3)
x = random.normal(key1, (n, d))

width = 4
depth = 3
W_std = 1.234
output_dim = 2

init_fn, apply_fn, kernel_fn = _generate_fc_relu_ntk(width, depth, W_std)

kernel_fn = kernel_fn if no_jitting else jit(kernel_fn)
nt_kernel = kernel_fn(x, None)

# Sanity check of grad_feat.
_, params = init_fn(key2, x.shape)
grad_feat = _get_grad(x, output_dim, params, apply_fn)
assert np.linalg.norm(
    nt.empirical_ntk_fn(apply_fn)(x, None, params) -
    grad_feat @ grad_feat.T) <= 1e-12

# Store Frobenius-norm of the exact NTK for estimating relative errors.
ntk_norm = np.linalg.norm(nt_kernel.ntk)

width_all = np.arange(2, 16)
grad_feat_dims_all = []

print("empirical_ntk_fn results:")
for width in width_all:
  init_fn, apply_fn, _ = _generate_fc_relu_ntk(width, depth, W_std)
  _, params = init_fn(key2, x.shape)
  grad_feat = _get_grad(x, output_dim, params, apply_fn)
  rel_err = np.linalg.norm(grad_feat @ grad_feat.T - nt_kernel.ntk) / ntk_norm
  grad_feat_dims_all.append(grad_feat.shape[1])
  print(
      f"feat_dim : {grad_feat.shape[1]} (width : {width}), relative err : {rel_err}"
  )

print()
print("ReluNTKFeatures results:")
relufeat_arg = {
    'num_layers': depth,
    'poly_degree': 16,
    'W_std': W_std,
}

for feat_dim in grad_feat_dims_all:
  relufeat_arg['poly_sketch_dim'] = feat_dim
  init_fn, feature_fn = ReluNTKFeatures(**relufeat_arg)
  ntk_feat = fc_relu_ntk_sketching(relufeat_arg,
                                   key3,
                                   init_fn=init_fn,
                                   feature_fn=feature_fn)

  rel_err = np.linalg.norm(ntk_feat @ ntk_feat.T - nt_kernel.ntk) / ntk_norm
  print(f"feat_dim : {ntk_feat.shape[1]}, err : {rel_err}")