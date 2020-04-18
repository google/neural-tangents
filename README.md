# Neural Tangents [[arXiv](https://arxiv.org/abs/1912.02803)]
[**Quickstart**](#colab-notebooks)
| [**Install guide**](#installation)
| [**Reference docs**](https://neural-tangents.readthedocs.io/en/latest/)

[![PyPI](https://img.shields.io/pypi/v/neural-tangents)](https://pypi.org/project/neural-tangents/) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/neural-tangents)](https://pypi.org/project/neural-tangents/)
[![Build Status](https://travis-ci.org/google/neural-tangents.svg?branch=master)](https://travis-ci.org/google/neural-tangents)
[![Readthedocs](https://readthedocs.org/projects/neural-tangents/badge/?version=latest)](https://neural-tangents.readthedocs.io/en/latest/?badge=latest)
[![PyPI - License](https://img.shields.io/pypi/l/neural_tangents)](https://github.com/google/neural-tangents/blob/master/LICENSE)

**News:**

* Neural Tangents just got faster! >4X speedup in computing analytic
kernels for CNN architectures with pooling, starting from version 0.2.1. See our
[Performance](#performance).

* We will be at [ICLR 2020](https://iclr.cc/), stay tuned for our live session
time slots.

## Overview

Neural Tangents is a high-level neural network API for specifying complex, hierarchical, neural networks of both finite and _infinite_ width. Neural Tangents allows researchers to define, train, and evaluate infinite networks as easily as finite ones.

Infinite (in width or channel count) neural networks are Gaussian Processes (GPs) with a kernel function determined by their architecture (see [References](#references) for details and nuances of this correspondence).

Neural Tangents allows you to construct a neural network model with the usual building blocks like convolutions, pooling, residual connections, nonlinearities etc. and obtain not only the finite model, but also the kernel function of the respective GP.

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

- [Neural Tangents Cookbook](https://colab.sandbox.google.com/github/google/neural-tangents/blob/master/notebooks/neural_tangents_cookbook.ipynb)
- [Weight Space Linearization](https://colab.research.google.com/github/google/neural-tangents/blob/master/notebooks/weight_space_linearization.ipynb)
- [Function Space Linearization](https://colab.research.google.com/github/google/neural-tangents/blob/master/notebooks/function_space_linearization.ipynb)
- [Neural Network Phase Diagram](https://colab.sandbox.google.com/github/google/neural-tangents/blob/master/notebooks/phase_diagram.ipynb)

## Installation

To use GPU, first follow [JAX's](https://www.github.com/google/jax/) GPU installation instructions. Otherwise, install JAX on CPU by running

```
pip install jaxlib jax --upgrade
```

Once JAX is installed install Neural Tangents by running

```
pip install neural-tangents
```
or, to use the bleeding-edge version from GitHub source,

```
git clone https://github.com/google/neural-tangents
pip install -e neural-tangents
```

You can now run the examples (using [`tensorflow_datasets`](https://github.com/tensorflow/datasets)) by calling:

```
# Note that Tensorflow does not work with Python 3.8
# https://github.com/tensorflow/tensorflow/issues/33374
pip install tensorflow "tensorflow-datasets>=2.0.0"

python neural-tangents/examples/infinite_fcn.py
python neural-tangents/examples/weight_space.py
python neural-tangents/examples/function_space.py
```

Finally, you can run tests by calling:

```
# NOTE: a few tests will fail without
# pip install tensorflow "tensorflow-datasets>=2.0.0"

for f in neural-tangents/neural_tangents/tests/*.py; do python $f; done
```

## 5-Minute intro

<b>See this [Colab](https://colab.sandbox.google.com/github/google/neural-tangents/blob/master/notebooks/neural_tangents_cookbook.ipynb) for a detailed tutorial. Below is a very quick introduction.</b>

Our library closely follows JAX's API for specifying neural networks,  [`stax`](https://github.com/google/jax/blob/master/jax/experimental/stax.py). In `stax` a network is defined by a pair of functions `(init_fn, apply_fn)` initializing the trainable parameters and computing the outputs of the network respectively. Below is an example of defining a 3-layer network and computing it's outputs `y` given inputs `x`.

```python
from jax import random
from jax.experimental import stax

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
Note that `kernel_fn` can compute _two_ covariance matrices corresponding to the Neural Network Gaussian Process (NNGP) and Neural Tangent (NT) kernels respectively. The NNGP kernel corresponds to the _Bayesian_ infinite neural network [[1-5]](#5-deep-neural-networks-as-gaussian-processes-iclr-2018-jaehoon-lee-yasaman-bahri-roman-novak-samuel-s-schoenholz-jeffrey-pennington-jascha-sohl-dickstein). The NTK corresponds to the _(continuous) gradient descent trained_ infinite network [[10]](#10-neural-tangent-kernel-convergence-and-generalization-in-neural-networks-neurips-2018-arthur-jacot-franck-gabriel-clément-hongler). In the above example, we compute the NNGP kernel but we could compute the NTK or both:

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

y_test_nngp = nt.predict.gp_inference(kernel_fn, x_train, y_train, x_test,
                                      get='nngp')
# (20, 1) np.ndarray test predictions of an infinite Bayesian network

y_test_ntk = nt.predict.gp_inference(kernel_fn, x_train, y_train, x_test,
                                     get='ntk')
# (20, 1) np.ndarray test predictions of an infinite continuous
# gradient descent trained network at convergence (t = inf)
```


### Infinitely WideResnet

We can define a more compex, (infinitely) Wide Residual Network [[14]](#14-wide-residual-networks-bmvc-2018-sergey-zagoruyko-nikos-komodakis) using the same `nt.stax` building blocks:

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

The `neural_tangents` (`nt`) package contains the following modules and methods:

* `stax` - primitives to construct neural networks like `Conv`, `Relu`, `serial`, `parallel` etc.

* `predict` - predictions with infinite networks:

  * `predict.gp_inference` - either fully Bayesian inference (`get='nngp'`) or inference with a network trained to full convergence (infinite time) on MSE loss using continuous gradient descent (`get='ntk'`).

  * `predict.gradient_descent_mse` - inference with a network trained on MSE loss with continuous gradient descent for an arbitrary finite time.

  * `predict.gradient_descent` - inference with a network trained on arbitrary loss with continuous gradient descent for an arbitrary finite time (using an ODE solver).

  * `predict.momentum` - inference with a network trained on arbitrary loss with continuous momentum gradient descent for an arbitrary finite time (using an ODE solver).

* `monte_carlo_kernel_fn` - compute a Monte Carlo kernel estimate  of _any_ `(init_fn, apply_fn)`, not necessarily specified `nt.stax`, enabling the kernel computation of infinite networks without closed-form expressions.

* Tools to investigate training dynamics of _wide but finite_ neural networks, like `linearize`, `taylor_expand`, `empirical_kernel_fn` and more. See [Training dynamics of wide but finite networks](#training-dynamics-of-wide-but-finite-networks) for details.


## Technical gotchas


### 64-bit precision
To enable 64-bit precision, set the respective JAX flag _before_ importing `neural_tangents` (see the JAX [guide](https://colab.research.google.com/github/google/jax/blob/master/notebooks/Common_Gotchas_in_JAX.ipynb#scrollTo=YTktlwTTMgFl)), for example:

```python
from jax.config import config
config.update("jax_enable_x64", True)
import neural_tangents as nt  # 64-bit precision enabled
```


### [`nt.stax`](https://github.com/google/neural-tangents/blob/master/neural_tangents/stax.py) vs [`jax.experimental.stax`](https://github.com/google/jax/blob/master/jax/experimental/stax.py)
We remark the following differences between our library and the JAX one.

* All `nt.stax` layers are instantiated with a function call, i.e. `nt.stax.Relu()` vs `jax.experimental.stax.Relu`.
* All layers with trainable parameters use the _NTK parameterization_ by default (see [[10]](#10-neural-tangent-kernel-convergence-and-generalization-in-neural-networks-neurips-2018-arthur-jacot-franck-gabriel-clément-hongler), Remark 1). However, Dense and Conv layers also support the _standard parameterization_ via a `parameterization` keyword argument (see [[15]](#15-on-the-infinite-width-limit-of-neural-networks-with-a-standard-parameterization-arxiv-2020-jascha-sohl-dickstein-roman-novak-samuel-s-schoenholz-jaehoon-lee)).
* `nt.stax` and `jax.experimental.stax` may have different layers and options available (for example `nt.stax` layers support `CIRCULAR` padding, have `LayerNorm`, but no `BatchNorm`.).


### CPU and TPU performance

For CNNs w/ pooling, our CPU and TPU performance is suboptimal due to low core
utilization (10-20%, looks like an XLA:CPU issue), and excessive padding
respectively. We will look into improving performance, but recommend NVIDIA GPUs
in the meantime. See [Performance](#performance).


## Training dynamics of wide but finite networks

The kernel of an infinite network `kernel_fn(x1, x2).ntk` combined with  `nt.predict.gradient_descent_mse` together allow to analytically track the outputs of an infinitely wide neural network trained on MSE loss througout training. Here we discuss the implications for _wide but finite_ neural networks and present tools to study their evolution in _weight space_ (trainable parameters of the network) and _function space_ (outputs of the network).

### Weight space

Continuous gradient descent in an infinite network has been shown in [[11]](#11-wide-neural-networks-of-any-depth-evolve-as-linear-models-under-gradient-descent-neurips-2019-jaehoon-lee-lechao-xiao-samuel-s-schoenholz-yasaman-bahri-roman-novak-jascha-sohl-dickstein-jeffrey-pennington) to correspond to training a _linear_ (in trainable parameters) model, which makes linearized neural networks an important subject of study for understanding the behavior of parameters in wide models.

For this, we provide two convenient methods:

* `nt.linearize`, and
* `nt.taylor_expand`,

which allow to linearize or get an arbitrary-order Taylor expansion of any function `apply_fn(params, x)` around some initial parameters `params_0` as `apply_fn_lin = nt.linearize(apply_fn, params_0)`.

One can use `apply_fn_lin(params, x)` exactly as you would any other function
(including as an input to JAX optimizers). This makes it easy to compare the
training trajectory of neural networks with that of its linearization.
Previous theory and experiments have examined the linearization of neural
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

Outputs of a linearized model evolve identically to those of an infinite one [[11]](#11-wide-neural-networks-of-any-depth-evolve-as-linear-models-under-gradient-descent-neurips-2019-jaehoon-lee-lechao-xiao-samuel-s-schoenholz-yasaman-bahri-roman-novak-jascha-sohl-dickstein-jeffrey-pennington) but with a different kernel - specifically, the Neural Tangent Kernel [[10]](#10-neural-tangent-kernel-convergence-and-generalization-in-neural-networks-neurips-2018-arthur-jacot-franck-gabriel-clément-hongler) evaluated on the specific `apply_fn` of the finite network given specific `params_0` that the network is initialized with. For this we provide the `nt.empirical_kernel_fn` function that accepts any `apply_fn` and returns a `kernel_fn(x1, x2, params)` that allows to compute the empirical NTK and NNGP kernels on specific `params`.

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
ntk_train_train = kernel_fn(x_train, x_train, params, 'ntk')
ntk_test_train = kernel_fn(x_test, x_train, params, 'ntk')
mse_predictor = nt.predict.gradient_descent_mse(
    ntk_train_train, y_train, ntk_test_train)

t = 5.
y_train_0 = apply_fn(params, x_train)
y_test_0 = apply_fn(params, x_test)
y_train_t, y_test_t = mse_predictor(t, y_train_0, y_test_0)
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

   * For convolutional networks one generally observes reasonable agreement
     agreement by the time the number of channels is 512.

2. Convergence at small learning rates.

With a new model it is therefore advisable to start with a very large model on
a small dataset using a small learning rate.


## Performance

In the table below we measure time to compute a single NTK
entry in a 21-layer CNN (`3x3` filters, no strides, `SAME` padding, `ReLU`)
followed by `stax.GlobalAvgPool` on inputs of shape `3x32x32`. Precisely:

```python
layers = []
for _ in range(21):
  layers += [stax.Conv(1, (3, 3), (1, 1), 'SAME'), stax.Relu()]
layers += [stax.GlobalAvgPool()]
_, _, kernel_fn = stax.serial(*layers)
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



Tested using version 0.2.1. All GPU results are per single accelerator.
Note that runtime is proportional to the depth of your network.
If your performance differs significantly,
please [file a bug](https://github.com/google/neural-tangents/issues/new)!


## Papers

Neural Tangents has been used in the following papers:

* [The large learning rate phase of deep learning: the catapult mechanism.](https://arxiv.org/abs/2003.02218) \
Aitor Lewkowycz, Yasaman Bahri, Ethan Dyer, Jascha Sohl-Dickstein, Guy Gur-Ari

* [Spectrum Dependent Learning Curves in Kernel Regression and Wide Neural Networks.
](https://arxiv.org/abs/2002.02561) \
Blake Bordelon, Abdulkadir Canatar, Cengiz Pehlevan

* [Taylorized Training: Towards Better Approximation of Neural Network Training at Finite Width.](https://arxiv.org/abs/2002.04010) \
   Yu Bai, Ben Krause, Huan Wang, Caiming Xiong, Richard Socher

* [On the Infinite Width Limit of Neural Networks with a Standard Parameterization.](https://arxiv.org/pdf/2001.07301.pdf) \
Jascha Sohl-Dickstein, Roman Novak, Samuel S. Schoenholz, Jaehoon Lee

* [Disentangling Trainability and Generalization in Deep Learning.](https://arxiv.org/abs/1912.13053) \
Lechao Xiao, Jeffrey Pennington, Samuel S. Schoenholz

* [Information in Infinite Ensembles of Infinitely-Wide Neural Networks.](https://arxiv.org/abs/1911.09189) \
Ravid Shwartz-Ziv, Alexander A. Alemi

* [Training Dynamics of Deep Networks using Stochastic Gradient Descent via Neural Tangent Kernel.](https://arxiv.org/abs/1905.13654) \
Soufiane Hayou, Arnaud Doucet, Judith Rousseau

* [Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient
Descent.](https://arxiv.org/abs/1902.06720) \
Jaehoon Lee*, Lechao Xiao*, Samuel S. Schoenholz, Yasaman Bahri, Roman Novak, Jascha
Sohl-Dickstein, Jeffrey Pennington

Please let us know if you make use of the code in a publication and we'll add it
to the list!


## Citation

If you use the code in a publication, please cite our ICLR 2020 paper:

```
@inproceedings{neuraltangents2020,
    title={Neural Tangents: Fast and Easy Infinite Neural Networks in Python},
    author={Roman Novak and Lechao Xiao and Jiri Hron and Jaehoon Lee and Alexander A. Alemi and Jascha Sohl-Dickstein and Samuel S. Schoenholz},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://github.com/google/neural-tangents}
}
```



## References

##### [1] [Priors for Infinite Networks.](https://www.cs.toronto.edu/~radford/pin.abstract.html) Radford M. Neal

##### [2] [Exponential expressivity in deep neural networks through transient chaos.](https://arxiv.org/abs/1606.05340) *NeurIPS 2016.* Ben Poole, Subhaneil Lahiri, Maithra Raghu, Jascha Sohl-Dickstein, Surya Ganguli

##### [3] [Toward deeper understanding of neural networks: The power of initialization and a dual view on expressivity.](http://papers.nips.cc/paper/6427-toward-deeper-understanding-of-neural-networks-the-power-of-initialization-and-a-dual-view-on-expressivity) *NeurIPS 2016.* Amit Daniely, Roy Frostig, Yoram Singer

##### [4] [Deep Information Propagation.](https://arxiv.org/abs/1611.01232) *ICLR 2017.* Samuel S. Schoenholz, Justin Gilmer, Surya Ganguli, Jascha Sohl-Dickstein

##### [5] [Deep Neural Networks as Gaussian Processes.](https://arxiv.org/abs/1806.07572) *ICLR 2018.* Jaehoon Lee*, Yasaman Bahri*, Roman Novak, Samuel S. Schoenholz, Jeffrey Pennington, Jascha Sohl-Dickstein

##### [6] [Gaussian Process Behaviour in Wide Deep Neural Networks.](https://arxiv.org/abs/1804.11271) *ICLR 2018.* Alexander G. de G. Matthews, Mark Rowland, Jiri Hron, Richard E. Turner, Zoubin Ghahramani

##### [7] [Dynamical Isometry and a Mean Field Theory of CNNs: How to Train 10,000-Layer Vanilla Convolutional Neural Networks.](https://arxiv.org/abs/1806.05393) *ICML 2018.* Lechao Xiao, Yasaman Bahri, Jascha Sohl-Dickstein, Samuel S. Schoenholz, Jeffrey Pennington

##### [8] [Bayesian Deep Convolutional Networks with Many Channels are Gaussian Processes.](https://arxiv.org/abs/1810.05148) *ICLR 2019.* Roman Novak*, Lechao Xiao*, Jaehoon Lee, Yasaman Bahri, Greg Yang, Jiri Hron, Daniel A. Abolafia, Jeffrey Pennington, Jascha Sohl-Dickstein

##### [9] [Deep Convolutional Networks as shallow Gaussian Processes.](https://arxiv.org/abs/1808.05587) *ICLR 2019.* Adrià Garriga-Alonso, Carl Edward Rasmussen, Laurence Aitchison

##### [10] [Neural Tangent Kernel: Convergence and Generalization in Neural Networks.](https://arxiv.org/abs/1806.07572) *NeurIPS 2018.* Arthur Jacot, Franck Gabriel, Clément Hongler

##### [11] [Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient Descent.](https://arxiv.org/abs/1902.06720) *NeurIPS 2019.* Jaehoon Lee*, Lechao Xiao*, Samuel S. Schoenholz, Yasaman Bahri, Roman Novak, Jascha Sohl-Dickstein, Jeffrey Pennington

##### [12] [Scaling Limits of Wide Neural Networks with Weight Sharing: Gaussian Process Behavior, Gradient Independence, and Neural Tangent Kernel Derivation.](https://arxiv.org/abs/1902.04760) *arXiv 2019.* Greg Yang

##### [13] [Mean Field Residual Networks: On the Edge of Chaos.](https://arxiv.org/abs/1712.08969) *NeurIPS 2017.* Greg Yang, Samuel S. Schoenholz

##### [14] [Wide Residual Networks.](https://arxiv.org/abs/1605.07146) *BMVC 2018.* Sergey Zagoruyko, Nikos Komodakis

##### [15] [On the Infinite Width Limit of Neural Networks with a Standard Parameterization.](https://arxiv.org/pdf/2001.07301.pdf) *arXiv 2020.* Jascha Sohl-Dickstein, Roman Novak, Samuel S. Schoenholz, Jaehoon Lee
