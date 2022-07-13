:github_url: https://github.com/google/neural-tangents/tree/main/docs/stax.rst



`nt.stax` -- infinite NNGP and NTK
===========================================


.. automodule:: neural_tangents.stax


Combinators
--------------------------------------
Layers to combine multiple other layers into one.

.. autosummary::
 :toctree: _autosummary

    parallel
    serial


Branching
--------------------------------------
Layers to split outputs into many, or combine many into ones.

.. autosummary::
 :toctree: _autosummary

    FanInConcat
    FanInProd
    FanInSum
    FanOut


Linear parametric
--------------------------------------
Linear layers with trainable parameters.

.. autosummary::
 :toctree: _autosummary

    Conv
    ConvLocal
    ConvTranspose
    Dense
    GlobalSelfAttention


Linear nonparametric
--------------------------------------
Linear layers without any trainable parameters.

.. autosummary::
 :toctree: _autosummary

    Aggregate
    AvgPool
    DotGeneral
    Dropout
    Flatten
    GlobalAvgPool
    GlobalSumPool
    Identity
    ImageResize
    Index
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


Helper classes
--------------------------------------
Utility classes for specifying layer properties. For enums, strings can be passed in their place.

.. autosummary::
 :toctree: _autosummary

    AggregateImplementation
    AttentionMechanism
    Padding
    PositionalEmbedding
    Slice


For developers
--------------------------------------
Classes and decorators helpful for constructing your own layers.

.. autosummary::
 :toctree: _autosummary

    Bool
    Diagonal
    layer
    requires
    supports_masking
