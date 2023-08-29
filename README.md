# **Stand with Ukraine!** 🇺🇦

Freedom of thought is fundamental to all of science. Right now, our freedom is being suppressed with bombing of civilians in Ukraine. **Don't be against the war - fight against the war! [supportukrainenow.org](https://supportukrainenow.org/)**.

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

Neural Tangents is a high-level neural network API for specifying complex, hierarchical, neural networks of both finite and _infinite_ width. Neural Tangents allows researchers to define, train, and evaluate infinite networks as easily as finite ones. The library has been used in [>100 papers](https://scholar.google.com/scholar?oi=bibs&hl=en&cites=4030630874639258770,4161931758707925692,2891750348147928089,8612471018033907356,10117604240015578443,4178323439418493877).

Infinite (in width or channel count) neural networks are Gaussian Processes (GPs) with a kernel function determined by their architecture. See [this listing](https://github.com/google/neural-tangents/wiki/Overparameterized-Neural-Networks:-Theory-and-Empirics) of papers written by the creators of Neural Tangents which study the infinite width limit of neural networks.

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
* [Citation](#citation)

## Colab Notebooks

An easy way to get started with Neural Tangents is by playing around with the following interactive notebooks in Colaboratory. They demo the major features of Neural Tangents and show how it can be used in research.

- [Neural Tangents Cookbook](https://colab.research.google.com/github/google/neural-tangents/blob/main/notebooks/neural_tangents_cookbook.ipynb)
- [Weight Space Linearization](https://colab.research.google.com/github/google/neural-tangents/blob/main/notebooks/weight_space_linearization.ipynb)
- [Function Space Linearization](https://colab.research.google.com/github/google/neural-tangents/blob/main/notebooks/function_space_linearization.ipynb)
- [Neural Network Phase Diagram](https://colab.research.google.com/github/google/neural-tangents/blob/main/notebooks/phase_diagram.ipynb)
- [Performance Benchmark](https://colab.research.google.com/github/google/neural-tangents/blob/main/notebooks/myrtle_kernel_with_neural_tangents.ipynb): simple benchmark for [Myrtle kernels](https://arxiv.org/abs/2003.02237). See also [Performance](#myrtle-network)
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

y = apply_fn(params, x)  # (10, 1) jnp.ndarray outputs of the neural network
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

Note that `kernel_fn` can compute _two_ covariance matrices corresponding to the [Neural Network Gaussian Process (NNGP)](https://en.wikipedia.org/wiki/Neural_network_Gaussian_process) and [Neural Tangent (NT)](https://en.wikipedia.org/wiki/Neural_tangent_kernel) kernels respectively. The NNGP kernel corresponds to the _Bayesian_ infinite neural network. The NTK corresponds to the _(continuous) gradient descent trained_ infinite network. In the above example, we compute the NNGP kernel, but we could compute the NTK or both:

```python
# Get kernel of a single type
nngp = kernel_fn(x1, x2, 'nngp') # (10, 20) jnp.ndarray
ntk = kernel_fn(x1, x2, 'ntk') # (10, 20) jnp.ndarray

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
# (20, 1) jnp.ndarray test predictions of an infinite Bayesian network

y_test_ntk = predict_fn(x_test=x_test, get='ntk')
# (20, 1) jnp.ndarray test predictions of an infinite continuous
# gradient descent trained network at convergence (t = inf)

# Get predictions as a namedtuple
both = predict_fn(x_test=x_test, get=('nngp', 'ntk'))
both.nngp == y_test_nngp  # True
both.ntk == y_test_ntk  # True

# Unpack the predictions namedtuple
y_test_nngp, y_test_ntk = predict_fn(x_test=x_test, get=('nngp', 'ntk'))
```


### Infinitely WideResnet

We can define a more complex, (infinitely) [Wide Residual Network](https://arxiv.org/abs/1605.07146) using the same `nt.stax` building blocks:

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

  * `predict.gradient_descent_mse_ensemble` - inference with an infinite ensemble of infinite width networks, either fully Bayesian (`get='nngp'`) or inference with MSE loss using continuous gradient descent (`get='ntk'`). Finite-time Bayesian inference (e.g. `t=1., get='nngp'`) is interpreted as [gradient descent on the top layer only](https://arxiv.org/abs/1902.06720), since it converges to exact Gaussian process inference with NNGP (`t=None, get='nngp'`). Computed in closed form.

  * `predict.gp_inference` - exact closed form Gaussian process inference using NNGP (`get='nngp'`), NTK (`get='ntk'`), or both (`get=('nngp', 'ntk')`). Equivalent to `predict.gradient_descent_mse_ensemble` with `t=None` (infinite training time), but has a slightly different API (accepting precomputed kernel matrix `k_train_train` instead of `kernel_fn` and `x_train`).

* `monte_carlo_kernel_fn` - compute a Monte Carlo kernel estimate  of _any_ `(init_fn, apply_fn)`, not necessarily specified via `nt.stax`, enabling the kernel computation of infinite networks without closed-form expressions.

* Tools to investigate training dynamics of _wide but finite_ neural networks, like `linearize`, `taylor_expand`, `empirical_kernel_fn` and more. See [Training dynamics of wide but finite networks](#training-dynamics-of-wide-but-finite-networks) for details.


## Technical gotchas


### [`nt.stax`](https://github.com/google/neural-tangents/blob/main/neural_tangents/stax.py) vs [`jax.example_libraries.stax`](https://github.com/google/jax/blob/main/jax/example_libraries/stax.py)
We remark the following differences between our library and the JAX one.

* All `nt.stax` layers are instantiated with a function call, i.e. `nt.stax.Relu()` vs `jax.example_libraries.stax.Relu`.
* All layers with trainable parameters use the [_NTK parameterization_](https://arxiv.org/abs/1806.07572) by default. However, `Dense` and `Conv` layers also support the [_standard parameterization_](https://arxiv.org/abs/2001.07301) via a `parameterization` keyword argument.
* `nt.stax` and `jax.example_libraries.stax` may have different layers and options available (for example `nt.stax` layers support `CIRCULAR` padding, have `LayerNorm`, but no `BatchNorm`.).


### CPU and TPU performance

For CNNs w/ pooling, our CPU and TPU performance is suboptimal due to low core
utilization (10-20%, looks like an XLA:CPU issue), and excessive padding
respectively. We will look into improving performance, but recommend NVIDIA GPUs
in the meantime. See [Performance](#performance).


## Training dynamics of wide but finite networks

The kernel of an infinite network `kernel_fn(x1, x2).ntk` combined with  `nt.predict.gradient_descent_mse` together allow to analytically track the outputs of an infinitely wide neural network trained on MSE loss throughout training. Here we discuss the implications for _wide but finite_ neural networks and present tools to study their evolution in _weight space_ (trainable parameters of the network) and _function space_ (outputs of the network).

### Weight space

Continuous gradient descent in an infinite network [has been shown in](https://arxiv.org/abs/1902.06720) to correspond to training a _linear_ (in trainable parameters) model, which makes linearized neural networks an important subject of study for understanding the behavior of parameters in wide models.

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
import jax.numpy as jnp
import neural_tangents as nt

def apply_fn(params, x):
  W, b = params
  return jnp.dot(x, W) + b

W_0 = jnp.array([[1., 0.], [0., 1.]])
b_0 = jnp.zeros((2,))

apply_fn_lin = nt.linearize(apply_fn, (W_0, b_0))
W = jnp.array([[1.5, 0.2], [0.1, 0.9]])
b = b_0 + 0.2

x = jnp.array([[0.3, 0.2], [0.4, 0.5], [1.2, 0.2]])
logits = apply_fn_lin((W, b), x)  # (3, 2) jnp.ndarray
```

### Function space:

Outputs of a linearized model [evolve identically to those of an infinite one](https://arxiv.org/abs/1902.06720) but with a different kernel - precisely, the [Neural Tangent Kernel](https://arxiv.org/1806.07572) evaluated on the specific `apply_fn` of the finite network given specific `params_0` that the network is initialized with. For this we provide the `nt.empirical_kernel_fn` function that accepts any `apply_fn` and returns a `kernel_fn(x1, x2, get, params)` that allows to compute the empirical NTK and/or NNGP (based on `get`) kernels on specific `params`.

#### Example:

```python
import jax.random as random
import jax.numpy as jnp
import neural_tangents as nt


def apply_fn(params, x):
  W, b = params
  return jnp.dot(x, W) + b


W_0 = jnp.array([[1., 0.], [0., 1.]])
b_0 = jnp.zeros((2,))
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
# (3, 2) and (4, 2) jnp.ndarray train and test outputs after `t` units of time
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

With a new model it is therefore advisable to start with large width on a small dataset using a small learning rate.


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
flexibility, we took the [Myrtle architecture](https://arxiv.org/2003.02237)
as an example. With `NVIDIA V100` 64-bit precision, `nt` took 316/330/508 GPU-hours on full 60k CIFAR-10 dataset for Myrtle-5/7/10 kernels.


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
