# Neural Tangents
 Easy and Fast Infinite Neural Networks in Python

## Overview

Neural Tangents is a high-level neural network API for specifying complex, hierarchical, neural networks of both finite and _infinite_ width. Neural Tangents allows researchers to define, train, and evaluate infinite networks as easily as finite ones.

Infinite (in width or channel count) neural networks are Gaussian Processes (GPs) with a kernel function determined by their architecture (see [References](References) for details and nuances of this correspondence).

Neural Tangents allows you to construct a neural network model with the usual building blocks like convolutions, pooling, residual connections, nonlinearities etc. and obtain not only the finite model, but also the kernel function of the respective GP.

The library is written in python using [JAX](https://github.com/google/jax) and leveraging [XLA](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/g3doc/index.md) to run out-of-the-box on CPU, GPU, or TPU. Kernel computation is highly optimized for speed and memory efficiency, and can be automatically distributed over multiple accelerators with near-perfect scaling.

Neural Tangents is a work in progress.
We happily welcome contributions!

## 5-Minute Intro

<b>See this [Colab](https://colab.sandbox.google.com/github/google/neural-tangents/blob/master/notebooks/neural_tangents_cookbook.ipynb) for a detailed tutorial. Below is a very quick introduction.</b>

Our library closely follows JAX's API for specifying neural networks,  [`stax`](https://github.com/google/jax/blob/master/jax/experimental/stax.py). In `stax` a network is defined by a pair of functions `(init_fun, apply_fun)` initializing the trainable parameters and computing the outputs of the network respectively. Below is an example of defining a 3-layer network and computing it's outputs `y` given inputs `x`.

```python
from jax import random
from jax.experimental import stax

init_fun, apply_fun = stax.serial(
    stax.Dense(512), stax.Relu,
    stax.Dense(512), stax.Relu,
    stax.Dense(1)
)

key = random.PRNGKey(1)
x = random.normal(key, (10, 100))
_, params = init_fun(key, input_shape=x.shape)

y = apply_fun(params, x)  # (10, 1) np.ndarray outputs of the neural network
```

Neural Tangents is designed to serve as a drop-in replacement for `stax`, extending the `(init_fun, apply_fun)` tuple to a triple `(init_fun, apply_fun, ker_fun)`, where `ker_fun` is the kernel function of the infinite network (GP) of the given architecture. Below is an example of computing the covariances of the GP between two batches of inputs `x1` and `x2`.

```python
from jax import random
from neural_tangents import stax

init_fun, apply_fun, ker_fun = stax.serial(
    stax.Dense(512), stax.Relu(),
    stax.Dense(512), stax.Relu(),
    stax.Dense(1)
)

key1, key2 = random.split(random.PRNGKey(1))
x1 = random.normal(key1, (10, 100))
x2 = random.normal(key2, (20, 100))

kernel = ker_fun(x1, x2)
```
Note that `kernel` contains _two_ covariance matrices: `kernel.nngp` and `kernel.ntk`. `kernel.nngp` corresponds to the _Bayesian_ infinite neural network, and is commonly referred to as "NNGP" (Neural Network Gaussian Process, [[1]](1)). `kernel.ntk` corresponds to the _(continuous) gradient descent trained_ infinite network, and is commonly referred to as "NTK" (Neural Tangent Kernel [[5]](5)). These matrices can be accessed as follows:

```python
nngp = kernel.nngp  # (10, 20) np.ndarray
ntk = kernel.ntk  # (10, 20) np.ndarray
```

Doing inference with infinite networks trained on MSE loss reduces to classical GP inference, for which we also provide convenient tools:

```python
from neural_tangents import predict

x_train, x_test = x1, x2
y_train = random.uniform(key1, shape=(10, 1))  # training targets

y_test_nngp = predict.gp_inference(ker_fun, x_train, y_train, x_test, mode='NNGP')
# (20, 1) np.ndarray test predictions of an infinite Bayesian network

y_test_ntk = predict.gp_inference(ker_fun, x_train, y_train, x_test, mode='NTK')
# (20, 1) np.ndarray test predictions of an infinite continuous gradient descent trained network at convergence (t = inf)
```


### Infinitely WideResnet

We can define a more compex, (infinitely) Wide Residual Network [[8](8)] using the same `neural_tangents.stax` building blocks:

```python
from neural_tangents import stax

def WideResnetBlock(channels, strides=(1, 1), channel_mismatch=False):
  Main = stax.serial(
      stax.Relu(), stax.Conv(channels, (3, 3), strides, padding='SAME'),
      stax.Relu(), stax.Conv(channels, (3, 3), padding='SAME'))
  Shortcut = stax.Identity() if not channel_mismatch else stax.Conv(
      channels, (3, 3), strides, padding='SAME')
  return stax.serial(stax.FanOut(2), stax.parallel(Main, Shortcut), stax.FanInSum())

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

init_fun, apply_fun, ker_fun = WideResnet(block_size=4, k=1, num_classes=10)
```


## Package description

The `neural_tangents` package contains two modules:

* `stax` - primitives to construct neural networks like `Conv`, `Relu`, `serial`, `parallel` etc.

* `predict` - predictions with infinite networks:

  * `predict.gp_inference` - either fully Bayesian inference (`mode="NNGP"`) or inference with a network trained to full convergence (infinite time) on MSE loss using continuous gradient descent (`mode="NTK"`).

  * `predict.gradient_descent_mse` - inference with a network trained on MSE loss with continuous gradient descent for an arbitrary finite time.
  
  * `predict.gradient_descent` - inference with a network trained on arbitrary loss with continuous gradient descent for an arbitrary finite time (using an ODE solver).
  
  * `predict.momentum` - inference with a network trained on arbitrary loss with continuous momentum gradient descent for an arbitrary finite time (using an ODE solver).

* `api` - various methods useful for working with infinite networks, including (but not limited to!):
    * `batch` - makes any kernel function `ker_fun` compute the kernel in batches over inputs, in parallel over available GPUs or TPU cores.
    
    * `get_ker_fun_monte_carlo` - compute a Monte Carlo kernel estimate  of _any_ `(init_fun, apply_fun)`, not necessarily specified `neural_tangents.stax`, enabling the kernel computation of infinite networks without closed-form expressions.
  
  * Tools to investigate training dynamics of _wide but finite_ neural networks, like `linearize`, `taylor_expand`, `get_ker_fun_empirical` and more. See [Training Dynamics of Wide but Finite Networks](wide) for details.
    



## <a name="wide"></a>Training Dynamics of Wide but Finite Networks

The kernel of an infinite network `ker_fun(x1, x2).ntk` combined with  `neural_tangents.predict.gradient_descent_mse` together allow to analytically track the outputs of an infinitely wide neural network trained on MSE loss througout training. Here we discuss the implications for _wide but finite_ neural networks and present tools to study their evolution in _weight space_ (trainable parameters of the network) and _function space_ (outputs of the network).

### Weight Space

Continuous gradient descent in an infinite network has been shown in [[6]](6) to correspond to training a _linear_ (in trainable parameters) model, which makes linearized neural networks an important subject of study for understanding the behavior of parameters in wide models.

For this, we provide two convenient methods:

* `neural_tangents.api.linearize`, and
* `neural_tangents.api.taylor_expand`,

which allow to linearize or get an arbitrary-order Taylor expansion of any function `apply_fun(params, x)` around some initial parameters `params_0` as `apply_fun_lin = linearize(apply_fun, params_0)`.

One can use `apply_fun_lin(params, x)` exactly as you would any other function
(including as an input to JAX optimizers). This makes it easy to compare the
training trajectory of neural networks with that of its linearization.
Previous theory and experiments have examined the linearization of neural 
networks from inputs to logits or pre-activations, rather than from inputs to
post-activations which are substantially more nonlinear.

#### Example:

```python
import jax.numpy as np
from neural_tangents.api import linearize

def apply_fun(params, x):
  W, b = params
  return np.dot(x, W) + b

W_0 = np.array([[1., 0.], [0., 1.]])
b_0 = np.zeros((2,))

apply_fun_lin = linearize(apply_fun, (W_0, b_0))
W = np.array([[1.5, 0.2], [0.1, 0.9]])
b = b_0 + 0.2

x = np.array([[0.3, 0.2], [0.4, 0.5], [1.2, 0.2]])
logits = apply_fun_lin((W, b), x)  # (3, 2) np.ndarray
```

### Function Space:

Outputs of a linearized model evolve identically to those of an infinite one [[6]](6) but with a different kernel - specifically, the Neural Tangent Kernel [[5]](5) evaluated on the specific `apply_fun` of the finite network given specific `params_0` that the network is initialized with. For this we provide the `neural_tangents.api.get_ker_fun_empirical` function that accepts any `apply_fun` and returns a `ker_fun(x1, x2, params)` that allows to compute the empirical NTK and NNGP kernels on specific `params`.

#### Example:

```python
import jax.numpy as np
from neural_tangents.api import get_ker_fun_empirical
from neural_tangents import predict

def apply_fun(params, x):
  W, b = params
  return np.dot(x, W) + b

W_0 = np.array([[1., 0.], [0., 1.]])
b_0 = np.zeros((2,))
params = (W_0, b_0)

key1, key2 = random.split(random.PRNGKey(1), 2)
x_train = random.normal(key1, (3, 2))
x_test = random.normal(key2, (4, 2))
y_train = random.uniform(key1, shape=(3, 2))

ker_fun = get_ker_fun_empirical(apply_fun)
ntk_train_train = ker_fun(x_train, x_train, params).ntk
ntk_test_train = ker_fun(x_test, x_train, params).ntk
mse_predictor = predict.gradient_descent_mse(ntk_train_train, y_train, ntk_test_train)

t = 5.
y_train_0 = apply_fun(params, x_train)
y_test_0 = apply_fun(params, x_test)
y_train_t, y_test_t = mse_predictor(t, y_train_0, y_test_0)
# (3, 2) and (4, 2) np.ndarray train and test outputs after `t` units of time training with continuous gradient descent
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

With a new model it is therefore adviseable to start with a very large model on
a small dataset using a small learning rate.


## Getting Started

To install Neural Tangents, first follow [JAX's](https://www.github.com/google/jax/)
installation instructions. With JAX installed, using Neural Tangents should be
as easy as:

```
git clone https://github.com/google/neural-tangents
pip install -e neural-tangents
```

You can then run the examples by calling:

```
pip install tensorflow-datasets

python neural-tangents/examples/weight_space.py
python neural-tangents/examples/function_space.py
```

Finally, you can run tests by calling:

```
for f in neural-tangents/neural_tangents/tests/*.py; do python $f; done
```

If you would prefer, you can get started without installing by checking out our
colab examples:

- [Neural Tangents Cookbook](https://colab.sandbox.google.com/github/google/neural-tangents/blob/master/notebooks/neural_tangents_cookbook.ipynb)
- [Weight Space Linearization](https://colab.research.google.com/github/google/neural-tangents/blob/master/notebooks/weight_space_linearization.ipynb)
- [Function Space Linearization](https://colab.research.google.com/github/google/neural-tangents/blob/master/notebooks/function_space_linearization.ipynb)


## <a name="Papers"></a>Papers

Neural tangents has been used in the following papers:

* [Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient
Descent.](https://arxiv.org/abs/1902.06720) \
Jaehoon Lee*, Lechao Xiao*, Samuel S. Schoenholz, Yasaman Bahri, Roman Novak, Jascha
Sohl-Dickstein, Jeffrey Pennington

* [Training Dynamics of Deep Networks using Stochastic Gradient Descent via Neural Tangent Kernel.](https://arxiv.org/abs/1905.13654) \
Soufiane Hayou, Arnaud Doucet, Judith Rousseau

Please let us know if you make use of the code in a publication and we'll add it
to the list!


## Citation

If you use the code in a publication, please cite the repo using the .bib,

```
Coming soon.
```



## <a name="References"></a> References

<a name="1"></a>[1] [Deep Neural Networks as Gaussian Processes.](https://arxiv.org/abs/1806.07572)
*ICLR 2018.* \
Jaehoon Lee*, Yasaman Bahri*, Roman Novak, Samuel S. Schoenholz, Jeffrey Pennington, Jascha Sohl-Dickstein

<a name="2"></a>[2] [Gaussian Process Behaviour in Wide Deep Neural Networks.](https://arxiv.org/abs/1804.11271)
*ICLR 2018.* \
Alexander G. de G. Matthews, Mark Rowland, Jiri Hron, Richard E. Turner, Zoubin Ghahramani

<a name="3"></a>[3] [Bayesian Deep Convolutional Networks with Many Channels are Gaussian Processes.](https://arxiv.org/abs/1810.05148)
*ICLR 2019.* \
Roman Novak*, Lechao Xiao*, Jaehoon Lee, Yasaman Bahri, Greg Yang, Jiri Hron, Daniel A. Abolafia, Jeffrey Pennington, Jascha Sohl-Dickstein

<a name="4"></a>[4] [Deep Convolutional Networks as shallow Gaussian Processes.](https://arxiv.org/abs/1808.05587)
*ICLR 2019.* \
Adrià Garriga-Alonso, Carl Edward Rasmussen, Laurence Aitchison

<a name="5"></a>[5] [Neural Tangent Kernel: Convergence and Generalization in Neural Networks.](https://arxiv.org/abs/1806.07572)
*NeurIPS 2018.* \
Arthur Jacot, Franck Gabriel, Clément Hongler

<a name="6"></a>[6] [Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient
Descent.](https://arxiv.org/abs/1902.06720)
*NeurIPS 2019.* \
Jaehoon Lee*, Lechao Xiao*, Samuel S. Schoenholz, Yasaman Bahri, Roman Novak, Jascha
Sohl-Dickstein, Jeffrey Pennington

<a name="7"></a>[7] [Scaling Limits of Wide Neural Networks with Weight Sharing: Gaussian Process Behavior, Gradient Independence, and Neural Tangent Kernel Derivation.](https://arxiv.org/abs/1902.04760)
*arXiv 2019.* \
Greg Yang

<a name="WRN"></a>[8] [Wide Residual Networks.](https://arxiv.org/abs/1605.07146)
*BMVC 2018.* \
Sergey Zagoruyko, Nikos Komodakis
