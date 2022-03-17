# Efficient Feature Map of Neural Tangent Kernels via Sketching and Random Features

Implementations developed in [[1]](#1-scaling-neural-tangent-kernels-via-sketching-and-random-features). The library is written for users familar with [JAX](https://github.com/google/jax) and [Neural Tangents](https://github.com/google/neural-tangents) library. The codes are compatible with NT v0.5.0.

[PyTorch](https://pytorch.org/) Implementations can be found in [here](https://github.com/insuhan/ntk-sketch-rf).


## Examples

### Fully-connected NTK approximation via Random Features:

```python
from jax import random
from experimental.features import DenseFeatures, ReluFeatures, serial

relufeat_arg = {
    'feature_dim0': 128,
    'feature_dim1': 128,
    'sketch_dim': 256,
    'method': 'rf',
}

init_fn, feature_fn = serial(
    DenseFeatures(512), ReluFeatures(**relufeat_arg),
    DenseFeatures(512), ReluFeatures(**relufeat_arg),
    DenseFeatures(1)
)

key1, key2 = random.split(random.PRNGKey(1))
x = random.normal(key1, (5, 4))

_, feat_fn_inputs = init_fn(key2, x.shape)
feats = feature_fn(x, feat_fn_inputs)
# feats.nngp_feat is a feature map of NNGP kernel
# feats.ntk_feat is a feature map of NTK
```
For more details of fully connected NTK features, please check `test_fc_ntk.py`.

### Convolutional NTK approximation via Random Features:
```python
from experimental.features import ConvFeatures, AvgPoolFeatures, FlattenFeatures

init_fn, feature_fn = serial(
    ConvFeatures(512, filter_size=3), ReluFeatures(**relufeat_arg),
    AvgPoolFeatures(2, 2), FlattenFeatures()
)

n, H, W, C = 5, 8, 8, 3
key1, key2 = random.split(random.PRNGKey(1))
x = random.normal(key1, shape=(n, H, W, C))

_, feat_fn_inputs = init_fn(key2, x.shape)
feats = feature_fn(x, feat_fn_inputs)
# feats.nngp_feat is a feature map of NNGP kernel
# feats.ntk_feat is a feature map of NTK
```
For more complex CNTK features, please check `test_myrtle_networks.py`.

# Modules

All modules return a pair of functions `(init_fn, feature_fn)`. Instead of kernel function `kernel_fn` in [Neural Tangents](https://github.com/google/neural-tangents) library, we replace it with the feature map function `feature_fn`. We do not return `apply_fn` functions.

- `init_fn` takes (1) random seed and (2) input shape. It returns (1) a pair of shapes of both NNGP and NTK features and (2) parameters used for approximating the features (e.g., random vectors for Random Features approach).
- `feature_fn` takes (1) feature structure `features.Feature` and (2) parameters used for feature approximation (initialized by `init_fn`). It returns `features.Feature` including approximate features of the corresponding module.


## [`features.DenseFeatures`](https://github.com/insuhan/ntk-sketching-neural-tangents/blob/ea23f8575a61f39c88aa57723408c175dbba0045/features.py#L88)
`features.DenseFeatures` provides features for fully-connected dense layer and corresponds to `stax.Dense` module in [Neural Tangents](https://github.com/google/neural-tangents). We assume that the input is a tabular dataset (i.e., a n-by-d matrix). Its `feature_fn` updates the NTK features by concatenating NNGP features and NTK features. This is because `stax.Dense` updates a new NTK kernel matrix `(N x D)` by adding the previous NNGP and NTK kernel matrices. The features of dense layer are exact and no approximations are applied. 
```python
import numpy as np
from neural_tangents import stax

width = 1
x = random.normal(key1, shape=(3, 2))
_, _, kernel_fn = stax.Dense(width)
nt_kernel = kernel_fn(x)

_, feat_fn = DenseFeatures(width)
feat = feat_fn(x, ())

assert np.linalg.norm(nt_kernel.ntk - feat.ntk_feat @ feat.ntk_feat.T) <= 1e-12
assert np.linalg.norm(nt_kernel.nngp - feat.nngp_feat @ feat.nngp_feat.T) <= 1e-12
```

## [`features.ReluFeatures`](https://github.com/insuhan/ntk-sketching-neural-tangents/blob/ea23f8575a61f39c88aa57723408c175dbba0045/features.py#L119)
`features.ReluFeatures` is a key module of the NTK approximation. We implement feature approximations based on (1) Random Features of arc-cosine kernels [[2]](#2) and (2) Polynomial Sketch [[3]](#3). Parameters used for feature approximation are intialized in `init_fn`.  We support tabular and image datasets. For tabular dataset, the input features are of form `N x D` matrix and the approximations are applied to the d-dimensional vectors.

For image dataset, the inputs are 4-D tensors with shape `N x H x W x D` where N is batch size, H is image height, W is image width and D is the feature dimension. We reshape the image features into 2-D tensor with shape `NHW x D` and apply proper feature approximations. Then, the resulting features reshape to 4-D tensor with shape `N x H x W x D'` where `D'` is the output dimension of the feature approximation.

To use the Random Features approach, set the parameter `method` to `rf` (default `rf`), e.g.,
```python
x = random.normal(key1, shape=(3, 32))

init_fn, feat_fn = serial(
    DenseFeatures(1),
    ReluFeatures(method='rf', feature_dim0=10, feature_dim1=20, sketch_dim=30)
)

_, params = init_fn(key1, x.shape)

out_feat = feat_fn(x, params)

assert out_feat.nngp_feat.shape == (3, 20)
assert out_feat.ntk_feat.shape == (3, 30)
```

To use the exact feature map (based on Cholesky decomposition), set the parameter `method` to `exact`, e.g.,
```python
init_fn, feat_fn = serial(DenseFeatures(1), ReluFeatures(method='exact'))
_, params = init_fn(key1, x.shape)
out_feat = feat_fn(x, params)

assert out_feat.nngp_feat.shape == (3, 3)
assert out_feat.ntk_feat.shape == (3, 3)
```
(This is for debugging. The dimension of the exact feature map is equal to the number of inputs, i.e., `N` for tabular dataset, `NHW` for image dataset).


## [`features.ConvFeatures`](https://github.com/insuhan/ntk-sketching-neural-tangents/blob/447cf2f6add6cf9f8374df4ea8530bf73d156c1b/features.py#L236)

`features.ConvFeatures` is similar to `features.DenseFeatures` as it updates the NTK feature of the next layer by concatenting NNGP and NTK features of the previous one.  But, it additionlly requires the kernel pooling operations. Precisely, [[4]](#4) studied that the NNGP/NTK kernel matrices require to compute the trace of submatrix of size `stride_size`. This can be seen as convolution with an identity matrix with size `stride_size`. However, in the feature side, this can be done via concatenating shifted features thus the resulting feature dimension becomes `stride_size` times larger. Moreover, since image datasets are 2-D matrices, the kernel pooling should be applied along with two axes hence the output feature has the shape `N x H x W x (d * s**2)` where `s` is the stride size and `d` is the input feature dimension.


## [`features.AvgPoolFeatures`](https://github.com/insuhan/ntk-sketching-neural-tangents/blob/447cf2f6add6cf9f8374df4ea8530bf73d156c1b/features.py#L269)

`features.AvgPoolFeatures` operates the average pooling on features of both NNGP and NTK. It calls [`_pool_kernel`](https://github.com/google/neural-tangents/blob/dd7eabb718c9e3c6640c47ca2379d93db6194214/neural_tangents/_src/stax/linear.py#L3143) function in [Neural Tangents](https://github.com/google/neural-tangents) as a subroutine.

## [`features.FlattenFeatures`](https://github.com/insuhan/ntk-sketching-neural-tangents/blob/447cf2f6add6cf9f8374df4ea8530bf73d156c1b/features.py#L304)

`features.FlattenFeatures` makes the features 2-D tensors. Similar to [`Flatten`](https://github.com/google/neural-tangents/blob/dd7eabb718c9e3c6640c47ca2379d93db6194214/neural_tangents/_src/stax/linear.py#L1641) module in [Neural Tangents](https://github.com/google/neural-tangents), the flattened features recale by the square-root of the number of elements. For example, if `nngp_feat` has the shape `N x H x W x C`, it returns a `N x HWC` matrix where all entries are divided by `(H*W*C)**0.5`.


## References
#### [1] [Scaling Neural Tangent Kernels via Sketching and Random Features](https://arxiv.org/pdf/2106.07880.pdf)
#### [2] [Kernel methods for deep learning](https://cseweb.ucsd.edu/~saul/papers/nips09_kernel.pdf)
#### [3] [Oblivious Sketching of High-Degree Polynomial Kernels](https://arxiv.org/pdf/1909.01410.pdf)
#### [4] [On Exact Computation with an Infinitely Wide Neural Net](https://arxiv.org/pdf/1904.11955.pdf)

