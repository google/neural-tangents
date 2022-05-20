:github_url: https://github.com/google/neural-tangents/tree/main/docs/stax.rst

.. default-role:: code

`nt.stax` -- infinite NNGP and NTK
===========================================

.. automodule:: neural_tangents.stax


Combinators
--------------------------------------
Layers to combine multiple other layers into one.

.. autosummary::
 :toctree: _autosummary

    serial
    parallel


Branching
--------------------------------------
Layers to split outputs into many, or combine many into ones.

.. autosummary::
 :toctree: _autosummary

    FanOut
    FanInConcat
    FanInProd
    FanInSum


Linear parametric
--------------------------------------
Linear layers with trainable parameters.

.. autosummary::
 :toctree: _autosummary

    Dense
    Conv
    ConvLocal
    ConvTranspose
    GlobalSelfAttention


Linear nonparametric
--------------------------------------
Linear layers without any trainable parameters.

.. autosummary::
 :toctree: _autosummary

    Aggregate
    AvgPool
    Identity
    DotGeneral
    Dropout
    Flatten
    GlobalAvgPool
    GlobalSumPool
    ImageResize
    LayerNorm
    SumPool


Elementwise nonlinear
--------------------------------------
Pointwise nonlinear layers.

.. autosummary::
 :toctree: _autosummary

    ABRelu
    Abs
    Cos
    Elementwise
    ElementwiseNumerical
    Erf
    Exp
    ExpNormalized
    Gaussian
    Gelu
    Hermite
    LeakyRelu
    Rbf
    Relu
    Sigmoid_like
    Sign
    Sin


Helper enums
--------------------------------------
Enums for specifying layer properties. Strings can be used in their place.

.. autosummary::
 :toctree: _autosummary

    Padding
    PositionalEmbedding


For developers
--------------------------------------
Classes and decorators helpful for constructing your own layers.

.. autosummary::
 :toctree: _autosummary

    layer
    supports_masking
    requires
    Bool
    Diagonal
