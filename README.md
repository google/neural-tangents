# Neural Tangents

## Investigating linearized learning dynamics

Neural Tangents is a set of tools that can be used to probe the linearized
training dynamics of neural networks. There are two, dual,
perspectives that are explored here: linearization of training in weight space,
and linearization of training in function space.

The code is written using JAX and we adhere to JAX's overall aesthetic.
To that end, the code is functional and revolves around a few function
transformations. All of the transformations defined here can be applied to any
JAX function whose signature is `f(params, x)`. One can therefore use this code
to look at linear learning for functions that are not neural networks.

Neural Tangents is a research project written by Sam Schoenholz, Jaehoon Lee,
Roman Novak, Lechao Xiao, Yasaman Bahri, and Jascha Sohl-Dickstein.
We happily welcome contributions!

## Overview

The bulk of Neural Tangents is contained in `tangents.py` and involves
linearization in weight space and function space, described separately below. We
also include a small file, `layers.py` which contains stax densely connected and
convolutional layers in the NTK parameterization.

### Weight Space:

In weight space, we view functions as maps from pairs of parameters and inputs
to outputs. As such, we keep track of inputs (training and test points) as
well as the parameters of the network. In this case we take the first order
taylor series of a function about some initial parameters,

$$
f_{\text{lin}}(\theta, x) = f(\theta_0, x) + J(\theta_0, x)(\theta - \theta_0)
$$

where $$J_{ij}(\theta, x) = \partial_{\theta_i} f_j(\theta, x)$$ is the Jacobian
of the realization function, $$F:\mathbb R^P\to\mathcal F$$ that associates
parameters with specific realizations of the neural network. The Jacobian
evaluated on a single input will have shape `[output_dim, parameters]`. Often we
will compute the Jacobian over a batch of data in which case it will have shape
`[datapoints, output_dim, parameters]`. This is implemented by the function
`linearize(f, params_0)` which converts a function `f(params, x)` into its
linearization, `f_lin(params, x)`, about some initial parameters, `params_0`.

One can use `f_lin(params, x)` exactly as you would any other function
(including as an input to JAX optimizers). This makes it easy to compare the
training trajectory of neural networks with that of its linearization.

#### Example:

```python
import jax.numpy as np
import neural_tangents as tangents

def f(params, X):
  W, b = params
  return np.dot(X, W) + b

W_0 = np.array([[1, 0], [0, 1]])
b_0 = np.zeros((2,))

f_lin = tangents.linearize(f, (W_0, b_0))
W = np.array([[1.5, 0.2], [0.1, 0.9]])
b = b_0 + 0.2

logits = f_lin((W, b), x)
```

### Function Space:

Instead of tracking the function parameters and inputs, we can instead look at
the values that the function takes on training and test points. In this case one
can describe the evolution of the function values during training using an
object called the Neural Tangent Kernel,

$$G_\theta(x_1, x_2) = J(\theta, x_1) J(\theta, x_2)^T.$$

Since the Jacobian has shape `[output_dim, parameters]`, the NTK will have shape
`[output_dim, output_dim]`. As in the case of the Jacobian we will usually
compute the NTK over two batches of inputs, $$X_1$$ and $$X_2$$, of size $$N_1$$ and $$N_2$$ respectively. In this case we generalize the NTK to have shape
`[N_1, output_dim, N_2, output_dim]` though we will typically deal with a
flattened version whose shape is `[N_1 * output_dim, N_2 * output_dim]`. We
compute the NTK using the function `ntk(f, batch_size)` which returns a function that will compute the NTK for different datasets and parameters. Once the NTK is computed, there are solutions to certain learning algorithms in the small
learning-rate limit.

1.  With an MSE loss under gradient descent, there is an analytic solution to
    the dynamics which we implement using the `analytic_mse_predictor(G_DD,
    labels, G_TD)` function.

2.  Under gradient descent with an arbitrary loss, the dynamics can be solved
    using an ODE solver. This is implemented using the
    `gradient_descent_predictor(G_DD, y_train, loss, G_TD)` function. We choose
    to use the scipy.ode solver.

3.  Using the SGD with momentum and an arbitrary loss, the dynamics can be
    solved using an ODE solver augmented to include momentum variables. This is
    implemented using the `momentum_predictor(G_DD, y_train, loss,
    learning_rate, G_TD, momentum)` function.

#### Example:

```python
import jax.numpy as np
import neural_tangents as tangents

def f(params, X):
  W, b = params
  return np.dot(X, W) + b

W_0 = np.array([[1, 0], [0, 1]])
b_0 = np.zeros((2,))
params = (W_0, b_0)

G = tangents.ntk(f)
G_dd = G(params, X, X)
mse_predictor = tangents.analytic_mse_predictor(G_dd, Y)

fX = f(params, X)

fX = mse_predictor(fX, train_time)
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

### Theory

The theory underlying this toolkit was laid out in:

[1]
[Neural Tangent Kernel: Convergence and Generalization in Neural Networks.](https://arxiv.org/abs/1806.07572)
*NeurIPS 2018.* \
Arthur Jacot, Franck Gabriel, Clément Hongler

[2] [Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient
Descent.](https://arxiv.org/abs/1902.06720) \
Jaehoon Lee*, Lechao Xiao*, Samuel S. Schoenholz, Yasaman Bahri, Jascha
Sohl-Dickstein, Jeffrey Pennington

## Getting Started

Installing Neural Tangents should be as easy as:

```
git clone https://github.com/google/neural-tangents
pip install -e neural-tangents
```

You can then run the examples by calling:

```
python neural-tangents/examples/weight_space.py
```

Finally, you can run tests by calling:

```
python neural-tangents/neural_tangents/tangents_test.py
```

If you would prefer, you can get started without installing by checking out our
colab examples:

- [Weight Space Linearization](https://colab.research.google.com/github/google/neural-tangents/notebooks/weight_space_linearization.ipynb)
- [Function Space Linearization](https://colab.research.google.com/github/google/neural-tangents/notebooks/function_space_linearization.ipynb)


## Papers

Neural tangents has been used in the following papers:

[1] [Wide Neural Networks of Any Depth Evolve as Linear Models Under Gradient
Descent.](https://arxiv.org/abs/1902.06720) \
Jaehoon Lee*, Lechao Xiao*, Samuel S. Schoenholz, Yasaman Bahri, Jascha
Sohl-Dickstein, Jeffrey Pennington

If you use the code in a publication, please cite the repo using the .bib,

```
@software{neuraltangents2019,
  author = {Samuel S. Schoenholz and Jaehoon Lee and Roman Novak and Lechao Xiao and Yasaman Bahri and Jascha Sohl-Dickstein},
  title = {Neural Tangents},
  url = {http://github.com/google/neural-tangents},
  version = {0.0.1},
  year = {2019},
}
```

Please let us know if you make use of the code in a publication and we'll add it
to the list!


