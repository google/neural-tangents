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
