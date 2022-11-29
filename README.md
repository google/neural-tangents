# **Stand with Ukraine!** 🇺🇦

Freedom of thought is fundamental to all of science. Right now, our freedom is being suppressed with carpet bombing of civilians in Ukraine. **Don't be against the war - fight against the war! [supportukrainenow.org](https://supportukrainenow.org/)**.

### News

We're at NeurIPS! Come say hi at our poster **[Fast Neural Kernel Embeddings for General Activations](https://neurips.cc/virtual/2022/poster/52791)** at **#806 Hall J, Tue 29 Nov 11 a.m. CST — 1 p.m. CST**.

# Neural Tangents
[**ICLR 2020 Video**](https://iclr.cc/virtual_2020/poster_SklD9yrFPS.html)
| [**Paper**](https://arxiv.org/abs/1912.02803)
| [**Quickstart**](#colab-notebooks)
| [**Install guide**](#installation)
| [**Reference docs**](https://neural-tangents.readthedocs.io/en/latest/)
| [**Release notes**](https://github.com/google/neural-tangents/releases)

[![PyPI](https://img.shields.io/pypi/v/neural-tangents)](https://pypi.org/project/neural-tangents/) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/neural-tangents)](https://pypi.org/project/neural-tangents/)
[![Linux](https://github.com/google/neural-tangents/actions/workflows/linux.yml/badge.svg)](https://github.com/google/neural-tangents/actions/workflows/linux.yml)
[![macOS](https://github.com/google/neural-tangents/actions/workflows/macos.yml/badge.svg)](https://github.com/google/neural-tangents/actions/workflows/macos.yml)
[![Pytype](https://github.com/google/neural-tangents/actions/workflows/pytype.yml/badge.svg)](https://github.com/google/neural-tangents/actions/workflows/pytype.yml)
[![Coverage](https://codecov.io/gh/google/neural-tangents/branch/main/graph/badge.svg)](https://codecov.io/gh/google/neural-tangents)
[![Readthedocs](https://readthedocs.org/projects/neural-tangents/badge/?version=latest)](https://neural-tangents.readthedocs.io/en/latest/?badge=latest)

[//]: # ([![PyPI - License]&#40;https://img.shields.io/pypi/l/neural_tangents&#41;]&#40;https://github.com/google/neural-tangents/blob/main/LICENSE&#41;)


## Overview

Neural Tangents is a high-level neural network API for specifying complex, hierarchical, neural networks of both finite and _infinite_ width. Neural Tangents allows researchers to define, train, and evaluate infinite networks as easily as finite ones.

Infinite (in width or channel count) neural networks are Gaussian Processes (GPs) with a kernel function determined by their architecture. See [References](#references) for details and nuances of this correspondence. Also see [this listing](https://github.com/google/neural-tangents/wiki/Overparameterized-Neural-Networks:-Theory-and-Empirics) of papers written by the creators of Neural Tangents which study the infinite width limit of neural networks.

Neural Tangents allows you to construct a neural network model from common building blocks like convolutions, pooling, residual connections, nonlinearities, and more, and obtain not only the finite model, but also the kernel function of the respective GP.

The library is written in python using [JAX](https://github.com/google/jax) and leveraging [XLA](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/g3doc/index.md) to run out-of-the-box on CPU, GPU, or TPU. Kernel computation is highly optimized for speed and memory efficiency, and can be automatically distributed over multiple accelerators with near-perfect scaling.

Neural Tangents is a work in progress.
We happily welcome contributions!




## Contents
* [Colab Notebooks](#colab-notebooks)
* [Installation](#installation)
* [5-Minute intro](#5-minute-intro)
* [Package description](#package-description)
* [Technical gotchas](#technical-gotchas)
* [Training dynamics of wide but finite networks](#training-dynamics-of-wide-but-finite-networks)
* [Performance](#performance)
* [Papers](#papers)
* [Citation](#citation)
* [References](#references)

## Colab Notebooks

An easy way to get started with Neural Tangents is by playing around with the following interactive notebooks in Colaboratory. They demo the major features of Neural Tangents and show how it can be used in research.

- [Neural Tangents Cookbook](https://colab.research.google.com/github/google/neural-tangents/blob/main/notebooks/neural_tangents_cookbook.ipynb)
- [Weight Space Linearization](https://colab.research.google.com/github/google/neural-tangents/blob/main/notebooks/weight_space_linearization.ipynb)
- [Function Space Linearization](https://colab.research.google.com/github/google/neural-tangents/blob/main/notebooks/function_space_linearization.ipynb)
- [Neural Network Phase Diagram](https://colab.research.google.com/github/google/neural-tangents/blob/main/notebooks/phase_diagram.ipynb)
- [Performance Benchmark](https://colab.research.google.com/github/google/neural-tangents/blob/main/notebooks/myrtle_kernel_with_neural_tangents.ipynb): simple benchmark for Myrtle kernels used in [[16]](#16-neural-kernels-without-tangents). Also see [Performance](#myrtle-network)
- [**New**] Empirical NTK:
  - [Fully-connected network](https://colab.research.google.com/github/google/neural-tangents/blob/main/notebooks/empirical_ntk_fcn.ipynb)
  - [FLAX ResNet18](https://colab.research.google.com/github/google/neural-tangents/blob/main/notebooks/empirical_ntk_resnet.ipynb)
  - [Experimental: Tensorflow ResNet50](https://colab.research.google.com/github/google/neural-tangents/blob/main/notebooks/experimental/empirical_ntk_resnet_tf.ipynb)
- [**New**] [Automatic NNGP/NTK of elementwise nonlinearities](https://colab.research.google.com/github/google/neural-tangents/blob/main/notebooks/elementwise.ipynb)


## Installation

To use GPU, first follow [JAX's](https://www.github.com/google/jax/) GPU installation instructions. Otherwise, install JAX on CPU by running

```
pip install jax jaxlib --upgrade
```

Once JAX is installed install Neural Tangents by running

```
pip install neural-tangents
```
or, to use the bleeding-edge version from GitHub source,

```
git clone https://github.com/google/neural-tangents; cd neural-tangents
pip install -e .
```

You can now run the examples and tests by calling:

```
pip install .[testing]
set -e; for f in examples/*.py; do python $f; done  # Run examples
set -e; for f in tests/*.py; do python $f; done  # Run tests
```


## 5-Minute intro

<b>See this [Colab](https://colab.research.google.com/github/google/neural-tangents/blob/main/notebooks/neural_tangents_cookbook.ipynb) for a detailed tutorial. Below is a very quick introduction.</b>

Our library closely follows JAX's API for specifying neural networks,  [`stax`](https://github.com/google/jax/blob/main/jax/example_libraries/stax.py). In `stax` a network is defined by a pair of functions `(init_fn, apply_fn)` initializing the trainable parameters and computing the outputs of the network respectively. Below is an example of defining a 3-layer network and computing its outputs `y` given inputs `x`.

```python
from jax import random
from jax.example_libraries import stax

init_fn, apply_fn = stax.serial(
    stax.Dense(512), stax.Relu,
    stax.Dense(512), stax.Relu,
    stax.Dense(1)
)

key = random.PRNGKey(1)
x = random.normal(key, (10, 100))
_, params = init_fn(key, input_shape=x.shape)

y = apply_fn(params, x)  # (10, 1) np.ndarray outputs of the neural network
```

Neural Tangents is designed to serve as a drop-in replacement for `stax`, extending the `(init_fn, apply_fn)` tuple to a triple `(init_fn, apply_fn, kernel_fn)`, where `kernel_fn` is the kernel function of the infinite network (GP) of the given architecture. Below is an example of computing the covariances of the GP between two batches of inputs `x1` and `x2`.

```python
from jax import random
from neural_tangents import stax

init_fn, apply_fn, kernel_fn = stax.serial(
    stax.Dense(512), stax.Relu(),
    stax.Dense(512), stax.Relu(),
    stax.Dense(1)
)

key1, key2 = random.split(random.PRNGKey(1))
x1 = random.normal(key1, (10, 100))
x2 = random.normal(key2, (20, 100))

kernel = kernel_fn(x1, x2, 'nngp')
```

Note that `kernel_fn` can compute _two_ covariance matrices corresponding to the Neural Network Gaussian Process (NNGP) and Neural Tangent (NT) kernels respectively. The NNGP kernel corresponds to the _Bayesian_ infinite neural network [[1-5]](#5-deep-neural-networks-as-gaussian-processes). The NTK corresponds to the _(continuous) gradient descent trained_ infinite network [[10]](#10-neural-tangent-kernel-convergence-and-generalization-in-neural-networks). In the above example, we compute the NNGP kernel, but we could compute the NTK or both:

```python
# Get kernel of a single type
nngp = kernel_fn(x1, x2, 'nngp') # (10, 20) np.ndarray
ntk = kernel_fn(x1, x2, 'ntk') # (10, 20) np.ndarray

# Get kernels as a namedtuple
both = kernel_fn(x1, x2, ('nngp', 'ntk'))
both.nngp == nngp  # True
both.ntk == ntk  # True

# Unpack the kernels namedtuple
nngp, ntk = kernel_fn(x1, x2, ('nngp', 'ntk'))
```

Additionally, if no third-argument is specified then the `kernel_fn` will return a `Kernel` namedtuple that contains additional metadata. This can be useful for composing applications of `kernel_fn` as follows:

```python
kernel = kernel_fn(x1, x2)
kernel = kernel_fn(kernel)
print(kernel.nngp)
```

Doing inference with infinite networks trained on MSE loss reduces to classical GP inference, for which we also provide convenient tools:

```python
import neural_tangents as nt

x_train, x_test = x1, x2
y_train = random.uniform(key1, shape=(10, 1))  # training targets

predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, x_train,
                                                      y_train)

y_test_nngp = predict_fn(x_test=x_test, get='nngp')
# (20, 1) np.ndarray test predictions of an infinite Bayesian network

y_test_ntk = predict_fn(x_test=x_test, get='ntk')
# (20, 1) np.ndarray test predictions of an infinite continuous
# gradient descent trained network at convergence (t = inf)

# Get predictions as a namedtuple
both = predict_fn(x_test=x_test, get=('nngp', 'ntk'))
both.nngp == y_test_nngp  # True
both.ntk == y_test_ntk  # True

# Unpack the predictions namedtuple
y_test_nngp, y_test_ntk = predict_fn(x_test=x_test, get=('nngp', 'ntk'))
```


### Infinitely WideResnet

We can define a more complex, (infinitely) Wide Residual Network [[14]](#14-wide-residual-networks) using the same `nt.stax` building blocks:

```python
from neural_tangents import stax

def WideResnetBlock(channels, strides=(1, 1), channel_mismatch=False):
  Main = stax.serial(
      stax.Relu(), stax.Conv(channels, (3, 3), strides, padding='SAME'),
      stax.Relu(), stax.Conv(channels, (3, 3), padding='SAME'))
  Shortcut = stax.Identity() if not channel_mismatch else stax.Conv(
      channels, (3, 3), strides, padding='SAME')
  return stax.serial(stax.FanOut(2),
                     stax.parallel(Main, Shortcut),
                     stax.FanInSum())

def WideResnetGroup(n, channels, strides=(1, 1)):
  blocks = []
  blocks += [WideResnetBlock(channels, strides, channel_mismatch=True)]
  for _ in range(n - 1):
    blocks += [WideResnetBlock(channels, (1, 1))]
  return stax.serial(*blocks)

def WideResnet(block_size, k, num_classes):
  return stax.serial(
      stax.Conv(16, (3, 3), padding='SAME'),
      WideResnetGroup(block_size, int(16 * k)),
      WideResnetGroup(block_size, int(32 * k), (2, 2)),
      WideResnetGroup(block_size, int(64 * k), (2, 2)),
      stax.AvgPool((8, 8)),
      stax.Flatten(),
      stax.Dense(num_classes, 1., 0.))

init_fn, apply_fn, kernel_fn = WideResnet(block_size=4, k=1, num_classes=10)
```


## Package description

The `neural_tangents` (`nt`) package contains the following modules and functions:

* `stax` - primitives to construct neural networks like `Conv`, `Relu`, `serial`, `parallel` etc.

* `predict` - predictions with infinite networks:

  * `predict.gradient_descent_mse` - inference with a single infinite width / linearized network trained on MSE loss with continuous gradient descent for an arbitrary finite or infinite (`t=None`) time. Computed in closed form.

  * `predict.gradient_descent` - inference with a single infinite width / linearized network trained on arbitrary loss with continuous (momentum) gradient descent for an arbitrary finite time. Computed using an ODE solver.

  * `predict.gradient_descent_mse_ensemble` - inference with an infinite ensemble of infinite width networks, either fully Bayesian (`get='nngp'`) or inference with MSE loss using continuous gradient descent (`get='ntk'`). Finite-time Bayesian inference (e.g. `t=1., get='nngp'`) is interpreted as gradient descent on the top layer only [[11]](#11-wide-neural-networks-of-any-depth-evolve-as-linear-models-under-gradient-descent), since it converges to exact Gaussian process inference with NNGP (`t=None, get='nngp'`). Computed in closed form.

  * `predict.gp_inference` - exact closed form Gaussian process inference using NNGP (`get='nngp'`), NTK (`get='ntk'`), or both (`get=('nngp', 'ntk')`). Equivalent to `predict.gradient_descent_mse_ensemble` with `t=None` (infinite training time), but has a slightly different API (accepting precomputed kernel matrix `k_train_train` instead of `kernel_fn` and `x_train`).

* `monte_carlo_kernel_fn` - compute a Monte Carlo kernel estimate  of _any_ `(init_fn, apply_fn)`, not necessarily specified via `nt.stax`, enabling the kernel computation of infinite networks without closed-form expressions.

* Tools to investigate training dynamics of _wide but finite_ neural networks, like `linearize`, `taylor_expand`, `empirical.kernel_fn` and more. See [Training dynamics of wide but finite networks](#training-dynamics-of-wide-but-finite-networks) for details.


## Technical gotchas


### [`nt.stax`](https://github.com/google/neural-tangents/blob/main/neural_tangents/stax.py) vs [`jax.example_libraries.stax`](https://github.com/google/jax/blob/main/jax/example_libraries/stax.py)
We remark the following differences between our library and the JAX one.

* All `nt.stax` layers are instantiated with a function call, i.e. `nt.stax.Relu()` vs `jax.example_libraries.stax.Relu`.
* All layers with trainable parameters use the _NTK parameterization_ by default (see [[10]](#10-neural-tangent-kernel-convergence-and-generalization-in-neural-networks), Remark 1). However, Dense and Conv layers also support the _standard parameterization_ via a `parameterization` keyword argument (see [[15]](#15-on-the-infinite-width-limit-of-neural-networks-with-a-standard-parameterization)).
* `nt.stax` and `jax.example_libraries.stax` may have different layers and options available (for example `nt.stax` layers support `CIRCULAR` padding, have `LayerNorm`, but no `BatchNorm`.).


### CPU and TPU performance

For CNNs w/ pooling, our CPU and TPU performance is suboptimal due to low core
utilization (10-20%, looks like an XLA:CPU issue), and excessive padding
respectively. We will look into improving performance, but recommend NVIDIA GPUs
in the meantime. See [Performance](#performance).


## Training dynamics of wide but finite networks

The kernel of an infinite network `kernel_fn(x1, x2).ntk` combined with  `nt.predict.gradient_descent_mse` together allow to analytically track the outputs of an infinitely wide neural network trained on MSE loss throughout training. Here we discuss the implications for _wide but finite_ neural networks and present tools to study their evolution in _weight space_ (trainable parameters of the network) and _function space_ (outputs of the network).

### Weight space

Continuous gradient descent in an infinite network has been shown in [[11]](#11-wide-neural-networks-of-any-depth-evolve-as-linear-models-under-gradient-descent) to correspond to training a _linear_ (in trainable parameters) model, which makes linearized neural networks an important subject of study for understanding the behavior of parameters in wide models.

For this, we provide two convenient functions:

* `nt.linearize`, and
* `nt.taylor_expand`,

which allow us to linearize or get an arbitrary-order Taylor expansion of any function `apply_fn(params, x)` around some initial parameters `params_0` as `apply_fn_lin = nt.linearize(apply_fn, params_0)`.

One can use `apply_fn_lin(params, x)` exactly as you would any other function
(including as an input to JAX optimizers). This makes it easy to compare the
training trajectory of neural networks with that of its linearization.
Prior theory and experiments have examined the linearization of neural
networks from inputs to logits or pre-activations, rather than from inputs to
post-activations which are substantially more nonlinear.

#### Example:

```python
import jax.numpy as np
import neural_tangents as nt

def apply_fn(params, x):
  W, b = params
  return np.dot(x, W) + b

W_0 = np.array([[1., 0.], [0., 1.]])
b_0 = np.zeros((2,))

apply_fn_lin = nt.linearize(apply_fn, (W_0, b_0))
W = np.array([[1.5, 0.2], [0.1, 0.9]])
b = b_0 + 0.2

x = np.array([[0.3, 0.2], [0.4, 0.5], [1.2, 0.2]])
logits = apply_fn_lin((W, b), x)  # (3, 2) np.ndarray
```

### Function space:

Outputs of a linearized model evolve identically to those of an infinite one [[11]](#11-wide-neural-networks-of-any-depth-evolve-as-linear-models-under-gradient-descent) but with a different kernel - specifically, the Neural Tangent Kernel [[10]](#10-neural-tangent-kernel-convergence-and-generalization-in-neural-networks) evaluated on the specific `apply_fn` of the finite network given specific `params_0` that the network is initialized with. For this we provide the `nt.empirical_kernel_fn` function that accepts any `apply_fn` and returns a `kernel_fn(x1, x2, get, params)` that allows to compute the empirical NTK and/or NNGP (based on `get`) kernels on specific `params`.

#### Example:

```python
import jax.random as random
import jax.numpy as np
import neural_tangents as nt


def apply_fn(params, x):
  W, b = params
  return np.dot(x, W) + b


W_0 = np.array([[1., 0.], [0., 1.]])
b_0 = np.zeros((2,))
params = (W_0, b_0)

key1, key2 = random.split(random.PRNGKey(1), 2)
x_train = random.normal(key1, (3, 2))
x_test = random.normal(key2, (4, 2))
y_train = random.uniform(key1, shape=(3, 2))

kernel_fn = nt.empirical_kernel_fn(apply_fn)
ntk_train_train = kernel_fn(x_train, None, 'ntk', params)
ntk_test_train = kernel_fn(x_test, x_train, 'ntk', params)
mse_predictor = nt.predict.gradient_descent_mse(ntk_train_train, y_train)

t = 5.
y_train_0 = apply_fn(params, x_train)
y_test_0 = apply_fn(params, x_test)
y_train_t, y_test_t = mse_predictor(t, y_train_0, y_test_0, ntk_test_train)
# (3, 2) and (4, 2) np.ndarray train and test outputs after `t` units of time
# training with continuous gradient descent
```

### What to Expect

The success or failure of the linear approximation is highly architecture
dependent. However, some rules of thumb that we've observed are:

1. Convergence as the network size increases.

   * For fully-connected networks one generally observes very strong
     agreement by the time the layer-width is 512 (RMSE of about 0.05 at the
     end of training).

   * For convolutional networks one generally observes reasonable
     agreement by the time the number of channels is 512.

2. Convergence at small learning rates.

With a new model it is therefore advisable to start with a very large model on
a small dataset using a small learning rate.


## Performance

In the table below we measure time to compute a single NTK
entry in a 21-layer CNN (`3x3` filters, no strides, `SAME` padding, `ReLU`) on inputs of shape `3x32x32`. Precisely:

```python
layers = []
for _ in range(21):
  layers += [stax.Conv(1, (3, 3), (1, 1), 'SAME'), stax.Relu()]
```


### CNN with pooling

Top layer is `stax.GlobalAvgPool()`:

```
_, _, kernel_fn = stax.serial(*(layers + [stax.GlobalAvgPool()]))
```

| Platform                    | Precision | Milliseconds / NTK entry | Max batch size (`NxN`) |
|-----------------------------|-----------|--------------------------|------------------------|
| CPU, >56 cores, >700 Gb RAM | 32        |  112.90                  | >= 128                 |
| CPU, >56 cores, >700 Gb RAM | 64        |  258.55                  |    95 (fastest - 72)   |
| TPU v2                      | 32/16     |  3.2550                  |    16                  |
| TPU v3                      | 32/16     |  2.3022                  |    24                  |
| NVIDIA P100                 | 32        |  5.9433                  |    26                  |
| NVIDIA P100                 | 64        |  11.349                  |    18                  |
| NVIDIA V100                 | 32        |  2.7001                  |    26                  |
| NVIDIA V100                 | 64        |  6.2058                  |    18                  |


### CNN without pooling

Top layer is `stax.Flatten()`:

```
_, _, kernel_fn = stax.serial(*(layers + [stax.Flatten()]))
```

| Platform                    | Precision | Milliseconds / NTK entry | Max batch size (`NxN`)            |
|-----------------------------|-----------|--------------------------|-----------------------------------|
| CPU, >56 cores, >700 Gb RAM | 32        |  0.12013                 |  2048 <= N < 4096 (fastest - 512) |
| CPU, >56 cores, >700 Gb RAM | 64        |  0.3414                  |  2048 <= N < 4096 (fastest - 256) |
| TPU v2                      | 32/16     |  0.0015722               |  512  <= N < 1024                 |
| TPU v3                      | 32/16     |  0.0010647               |  512  <= N < 1024                 |
| NVIDIA P100                 | 32        |  0.015171                |  512  <= N < 1024                 |
| NVIDIA P100                 | 64        |  0.019894                |  512  <= N < 1024                 |
| NVIDIA V100                 | 32        |  0.0046510               |  512  <= N < 1024                 |
| NVIDIA V100                 | 64        |  0.010822                |  512  <= N < 1024                 |




Tested using version `0.2.1`. All GPU results are per single accelerator.
Note that runtime is proportional to the depth of your network.
If your performance differs significantly,
please [file a bug](https://github.com/google/neural-tangents/issues/new)!



### Myrtle network

Colab notebook [Performance Benchmark](https://colab.research.google.com/github/google/neural-tangents/blob/main/notebooks/myrtle_kernel_with_neural_tangents.ipynb)
demonstrates how one would construct and benchmark kernels. To demonstrate
flexibility, we took architecture from [[16]](#16-neural-kernels-without-tangents)
as an example. With `NVIDIA V100` 64-bit precision, `nt` took 316/330/508 GPU-hours on full 60k CIFAR-10 dataset for Myrtle-5/7/10 kernels.

## Papers

Neural Tangents has been used in the following papers (newest first):

1. [Characterizing the Spectrum of the NTK via a Power Series Expansion](https://arxiv.org/abs/2211.07844)
2. [Evolution of Neural Tangent Kernels under Benign and Adversarial Training](https://arxiv.org/abs/2210.12030)
3. [Efficient Dataset Distillation Using Random Feature Approximation](https://arxiv.org/abs/2210.12067)
4. [Bidirectional Learning for Offline Infinite-width Model-based Optimization](https://arxiv.org/abs/2209.07507)
5. [Joint Embedding Self-Supervised Learning in the Kernel Regime](https://arxiv.org/abs/2209.14884)
6. [What Can the Neural Tangent Kernel Tell Us About Adversarial Robustness?](https://arxiv.org/abs/2210.05577)
7. [Few-shot Backdoor Attacks via Neural Tangent Kernels](https://arxiv.org/abs/2210.05929)
8. [Fast Neural Kernel Embeddings for General Activations](https://arxiv.org/abs/2209.04121)
9. [Neural Tangent Kernel: A Survey](https://arxiv.org/abs/2208.13614)
10. [Cognitive analyses of machine learning systems](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2022/EECS-2022-209.pdf)
11. [Gaussian process surrogate models for neural networks](https://arxiv.org/abs/2208.06028)
12. [Can we achieve robustness from data alone?](https://arxiv.org/abs/2207.11727)
13. [Synergy and Symmetry in Deep Learning: Interactions between the Data, Model, and Inference Algorithm](https://arxiv.org/abs/2207.04612)
14. [Bounding generalization error with input compression: An empirical study with infinite-width networks](https://arxiv.org/abs/2207.09408)
15. [Graph Neural Network Bandits](https://arxiv.org/abs/2207.06456)
16. [Making Look-Ahead Active Learning Strategies Feasible with Neural Tangent Kernels](https://arxiv.org/abs/2206.12569)
17. [A Fast, Well-Founded Approximation to the Empirical Neural Tangent Kernel](https://arxiv.org/abs/2206.12543)
18. [Limitations of the NTK for Understanding Generalization in Deep Learning](https://arxiv.org/abs/2206.10012)
19. [Wide Bayesian neural networks have a simple weight posterior: theory and accelerated sampling](https://arxiv.org/abs/2206.07673)
20. [Faster and easier: cross-validation and model robustness checks](https://dspace.mit.edu/handle/1721.1/143247)
21. [Lightweight and Accurate Cardinality Estimation by Neural Network Gaussian Process](https://dl.acm.org/doi/abs/10.1145/3514221.3526156)
22. [Infinite Recommendation Networks: A Data-Centric Approach
    ](https://arxiv.org/abs/2206.02626)
23. [Why So Pessimistic? Estimating Uncertainties for Offline RL through Ensembles, and Why Their Independence Matters](https://arxiv.org/abs/2205.13703)
24. [On the Interpretability of Regularisation for Neural Networks Through Model Gradient Similarity](https://arxiv.org/abs/2205.12642)
25. [Generative Adversarial Method Based on Neural Tangent Kernels](https://arxiv.org/abs/2204.04090)
26. [Generalization Through The Lens Of Leave-One-Out Error](https://arxiv.org/abs/2203.03443)
27. [Fast rates for noisy interpolation require rethinking the effects of inductive bias](https://arxiv.org/abs/2203.03597)
28. [A duality connecting neural network and cosmological dynamics](https://arxiv.org/abs/2202.11104)
29. [Representation Learning and Deep Generative Modeling in Dynamical Systems](https://tel.archives-ouvertes.fr/tel-03591720/document)
30. [Do autoencoders need a bottleneck for anomaly detection?](https://www.researchgate.net/profile/Bang-Xiang-Yong/publication/358445830_Do_autoencoders_need_a_bottleneck_for_anomaly_detection/links/6202e7d96adc0779cd52574a/Do-autoencoders-need-a-bottleneck-for-anomaly-detection.pdf)
31. [Finding Dynamics Preserving Adversarial Winning Tickets](https://arxiv.org/abs/2202.06488)
32. [Learning Representation from Neural Fisher Kernel with Low-rank Approximation](https://arxiv.org/abs/2202.01944)
33. [MIT 6.S088 Modern Machine Learning: Simple Methods that Work](https://web.mit.edu/modernml/course/)
34. [A Neural Tangent Kernel Perspective on Function-Space Regularization in Neural Networks](https://hudsonchen.github.io/papers/A_Neural_Tangent_Kernel_Perspective_on_Function_Space_Regularization_in_Neural_Networks.pdf)
35. [Eigenspace Restructuring: a Principle of Space and Frequency in Neural Networks](https://arxiv.org/abs/2112.05611)
36. [Functional Regularization for Reinforcement Learning via Learned Fourier Features](https://arxiv.org/abs/2112.03257)
37. [A Structured Dictionary Perspective on Implicit Neural Representations](https://arxiv.org/abs/2112.01917)
38. [Critical initialization of wide and deep neural networks through partial Jacobians: general theory and applications to LayerNorm](https://arxiv.org/abs/2111.12143)
39. [Asymptotics of representation learning in finite Bayesian neural networks](https://arxiv.org/abs/2106.00651)
40. [On the Equivalence between Neural Network and Support Vector Machine](https://arxiv.org/abs/2111.06063)
41. [An Empirical Study of Neural Kernel Bandits](https://arxiv.org/abs/2111.03543)
42. [Neural Networks as Kernel Learners: The Silent Alignment Effect](https://arxiv.org/abs/2111.00034)
43. [Understanding Deep Learning via Analyzing Dynamics of Gradient Descent](https://dataspace.princeton.edu/handle/88435/dsp01xp68kk34b)
44. [Neural Scene Representations for View Synthesis](https://digitalassets.lib.berkeley.edu/techreports/ucb/incoming/EECS-2020-223.pdf)
45. [Neural Tangent Kernel Eigenvalues Accurately Predict Generalization](https://arxiv.org/abs/2110.03922)
46. [Uniform Generalization Bounds for Overparameterized Neural Networks](https://arxiv.org/abs/2109.06099)
47. [Data Summarization via Bilevel Optimization](https://arxiv.org/abs/2109.12534)
48. [Neural Tangent Generalization Attacks](http://proceedings.mlr.press/v139/yuan21b.html)
49. [Dataset Distillation with Infinitely Wide Convolutional Networks](https://arxiv.org/abs/2107.13034)
50. [Neural Contextual Bandits without Regret](https://arxiv.org/abs/2107.03144)
51. [Epistemic Neural Networks](https://arxiv.org/abs/2107.08924)
52. [Uncertainty-aware Cardinality Estimation by Neural Network Gaussian Process](https://arxiv.org/abs/2107.08706)
53. [Scale Mixtures of Neural Network Gaussian Processes](https://arxiv.org/abs/2107.01408)
54. [Provably efficient machine learning for quantum many-body problems](https://arxiv.org/abs/2106.12627)
55. [Wide Mean-Field Variational Bayesian Neural Networks Ignore the Data](https://arxiv.org/abs/2106.07052)
56. [Spectral bias and task-model alignment explain generalization in kernel regression and infinitely wide neural networks](https://www.nature.com/articles/s41467-021-23103-1)
57. [Bridging Multi-Task Learning and Meta-Learning: Towards Efficient Training and Effective Adaptation](https://arxiv.org/abs/2106.09017)
58. [Wide Mean-Field Variational Bayesian Neural Networks Ignore the Data](https://arxiv.org/abs/2106.07052)
59. [What can linearized neural networks actually say about generalization?](https://arxiv.org/abs/2106.06770)
60. [Measuring the sensitivity of Gaussian processes to kernel choice](https://arxiv.org/abs/2106.06510)
61. [A Neural Tangent Kernel Perspective of GANs](https://arxiv.org/abs/2106.05566)
62. [On the Power of Shallow Learning](https://arxiv.org/abs/2106.03186)
63. [Learning Curves for SGD on Structured Features](https://arxiv.org/abs/2106.02713)
64. [Out-of-Distribution Generalization in Kernel Regression](https://arxiv.org/abs/2106.02261)
65. [Rapid Feature Evolution Accelerates Learning in Neural Networks](https://arxiv.org/abs/2105.14301)
66. [Scalable and Flexible Deep Bayesian Optimization with Auxiliary Information for Scientific Problems](https://arxiv.org/abs/2104.11667)
67. [Random Features for the Neural Tangent Kernel](https://arxiv.org/abs/2104.01351)
68. [Multi-Level Fine-Tuning: Closing Generalization Gaps in Approximation of Solution Maps under a Limited Budget for Training Data](https://arxiv.org/abs/2102.07169)
69. [Explaining Neural Scaling Laws](https://arxiv.org/abs/2102.06701)
70. [Correlated Weights in Infinite Limits of Deep Convolutional Neural Networks](https://arxiv.org/abs/2101.04097)
71. [Dataset Meta-Learning from Kernel Ridge-Regression](https://arxiv.org/abs/2011.00050)
72. [Deep learning versus kernel learning: an empirical study of loss landscape geometry and the time evolution of the Neural Tangent Kernel](https://arxiv.org/abs/2010.15110)
73. [Stable ResNet](https://arxiv.org/abs/2010.12859)
74. [Label-Aware Neural Tangent Kernel: Toward Better Generalization and Local Elasticity](https://arxiv.org/abs/2010.11775)
75. [Semi-supervised Batch Active Learning via Bilevel Optimization](https://arxiv.org/abs/2010.09654)
76. [Temperature check: theory and practice for training models with softmax-cross-entropy losses](https://arxiv.org/abs/2010.07344)
77. [Experimental Design for Overparameterized Learning with Application to Single Shot Deep Active Learning](https://arxiv.org/abs/2009.12820)
78. [How Neural Networks Extrapolate: From Feedforward to Graph Neural Networks](https://arxiv.org/abs/2009.11848)
79. [Exploring the Uncertainty Properties of Neural Networks’ Implicit Priors in the Infinite-Width Limit](http://www.gatsby.ucl.ac.uk/~balaji/udl2020/accepted-papers/UDL2020-paper-115.pdf)
80. [Cold Posteriors and Aleatoric Uncertainty](https://arxiv.org/abs/2008.00029)
81. [Asymptotics of Wide Convolutional Neural Networks](https://arxiv.org/abs/2008.08675)
82. [Finite Versus Infinite Neural Networks: an Empirical Study](https://arxiv.org/abs/2007.15801)
83. [Bayesian Deep Ensembles via the Neural Tangent Kernel](https://arxiv.org/abs/2007.05864)
84. [The Surprising Simplicity of the Early-Time Learning Dynamics of Neural Networks](https://arxiv.org/abs/2006.14599)
85. [When Do Neural Networks Outperform Kernel Methods?](https://arxiv.org/abs/2006.13409)
86. [Statistical Mechanics of Generalization in Kernel Regression](https://arxiv.org/abs/2006.13198)
87. [Exact posterior distributions of wide Bayesian neural networks](https://arxiv.org/abs/2006.10541)
88. [Infinite attention: NNGP and NTK for deep attention networks](https://arxiv.org/abs/2006.10540)
89. [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](https://arxiv.org/abs/2006.10739)
90. [Finding trainable sparse networks through Neural Tangent Transfer](https://arxiv.org/abs/2006.08228)
91. [Coresets via Bilevel Optimization for Continual Learning and Streaming](https://arxiv.org/abs/2006.03875)
92. [On the Neural Tangent Kernel of Deep Networks with Orthogonal Initialization](https://arxiv.org/abs/2004.05867)
93. [The large learning rate phase of deep learning: the catapult mechanism](https://arxiv.org/abs/2003.02218)
94. [Spectrum Dependent Learning Curves in Kernel Regression and Wide Neural Networks](https://arxiv.org/abs/2002.02561)
95. [Taylorized Training: Towards Better Approximation of Neural Network Training at Finite Width](https://arxiv.org/abs/2002.04010)
96. [On the Infinite Width Limit of Neural Networks with a Standard Parameterization](https://arxiv.org/abs/2001.07301)
97. [Disentangling Trainability and Generalization in Deep Learning](https://arxiv.org/abs/1912.13053)
98. [Information in Infinite Ensembles of Infinitely-Wide Neural Networks](https://arxiv.org/abs/1911.09189)
99. [Training Dynamics of Deep Networks using Stochastic Gradient Descent via Neural Tangent Kernel](https://arxiv.org/abs/1905.13654)
100. [Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent](https://arxiv.org/abs/1902.06720)
101. [Bayesian Deep Convolutional Networks with Many Channels are Gaussian Processes](https://arxiv.org/abs/1810.05148)


Please let us know if you make use of the code in a publication, and we'll add it
to the list!


## Citation

If you use the code in a publication, please cite our papers:

```bibtex
# Infinite width NTK/NNGP:
@inproceedings{neuraltangents2020,
    title={Neural Tangents: Fast and Easy Infinite Neural Networks in Python},
    author={Roman Novak and Lechao Xiao and Jiri Hron and Jaehoon Lee and Alexander A. Alemi and Jascha Sohl-Dickstein and Samuel S. Schoenholz},
    booktitle={International Conference on Learning Representations},
    year={2020},
    pdf={https://arxiv.org/abs/1912.02803},
    url={https://github.com/google/neural-tangents}
}

# Finite width, empirical NTK/NNGP:
@inproceedings{novak2022fast,
    title={Fast Finite Width Neural Tangent Kernel},
    author={Roman Novak and Jascha Sohl-Dickstein and Samuel S. Schoenholz},
    booktitle={International Conference on Machine Learning},
    year={2022},
    pdf={https://arxiv.org/abs/2206.08720},
    url={https://github.com/google/neural-tangents}
}

# Attention and variable-length inputs:
@inproceedings{hron2020infinite,
    title={Infinite attention: NNGP and NTK for deep attention networks},
    author={Jiri Hron and Yasaman Bahri and Jascha Sohl-Dickstein and Roman Novak},
    booktitle={International Conference on Machine Learning},
    year={2020},
    pdf={https://arxiv.org/abs/2006.10540},
    url={https://github.com/google/neural-tangents}
}

# Infinite-width "standard" parameterization:
@misc{sohl2020on,
    title={On the infinite width limit of neural networks with a standard parameterization},
    author={Jascha Sohl-Dickstein and Roman Novak and Samuel S. Schoenholz and Jaehoon Lee},
    publisher = {arXiv},
    year={2020},
    pdf={https://arxiv.org/abs/2001.07301},
    url={https://github.com/google/neural-tangents}
}

# Elementwise nonlinearities and sketching:
@inproceedings{han2022fast,
    title={Fast Neural Kernel Embeddings for General Activations},
    author={Insu Han and Amir Zandieh and Jaehoon Lee and Roman Novak and Lechao Xiao and Amin Karbasi},
    booktitle = {Advances in Neural Information Processing Systems},
    year={2022},
    pdf={https://arxiv.org/abs/2209.04121},
    url={https://github.com/google/neural-tangents}
}
```



## References

###### [1] [Priors for Infinite Networks](https://www.cs.toronto.edu/~radford/pin.abstract.html)
###### [2] [Exponential expressivity in deep neural networks through transient chaos](https://arxiv.org/abs/1606.05340)
###### [3] [Toward deeper understanding of neural networks: The power of initialization and a dual view on expressivity](http://papers.nips.cc/paper/6427-toward-deeper-understanding-of-neural-networks-the-power-of-initialization-and-a-dual-view-on-expressivity)
###### [4] [Deep Information Propagation](https://arxiv.org/abs/1611.01232)
###### [5] [Deep Neural Networks as Gaussian Processes](https://arxiv.org/abs/1711.00165)
###### [6] [Gaussian Process Behaviour in Wide Deep Neural Networks](https://arxiv.org/abs/1804.11271)
###### [7] [Dynamical Isometry and a Mean Field Theory of CNNs: How to Train 10,000-Layer Vanilla Convolutional Neural Networks.](https://arxiv.org/abs/1806.05393)
###### [8] [Bayesian Deep Convolutional Networks with Many Channels are Gaussian Processes](https://arxiv.org/abs/1810.05148)
###### [9] [Deep Convolutional Networks as shallow Gaussian Processes](https://arxiv.org/abs/1808.05587)
###### [10] [Neural Tangent Kernel: Convergence and Generalization in Neural Networks](https://arxiv.org/abs/1806.07572)
###### [11] [Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent](https://arxiv.org/abs/1902.06720)
###### [12] [Scaling Limits of Wide Neural Networks with Weight Sharing: Gaussian Process Behavior, Gradient Independence, and Neural Tangent Kernel Derivation](https://arxiv.org/abs/1902.04760)
###### [13] [Mean Field Residual Networks: On the Edge of Chaos](https://arxiv.org/abs/1712.08969)
###### [14] [Wide Residual Networks](https://arxiv.org/abs/1605.07146)
###### [15] [On the Infinite Width Limit of Neural Networks with a Standard Parameterization](https://arxiv.org/abs/2001.07301)
###### [16] [Neural Kernels Without Tangents](https://arxiv.org/abs/2003.02237)
