:github_url: https://github.com/google/neural-tangents/tree/main/docs/empirical.rst



`nt.empirical` -- finite NNGP and NTK
======================================

.. automodule:: neural_tangents._src.empirical
.. currentmodule:: neural_tangents

Kernel functions
--------------------------------------
Finite-width NNGP and/or NTK kernel functions.

.. autosummary::
 :toctree: _autosummary

    empirical_kernel_fn
    empirical_nngp_fn
    empirical_ntk_fn

NTK implementation
--------------------------------------
An :class:`enum.IntEnum` specifying NTK implementation method.

.. autoclass:: NtkImplementation

NTK-vector products
--------------------------------------
A function to compute NTK-vector products without instantiating the NTK.

.. autosummary::
 :toctree: _autosummary

    empirical_ntk_vp_fn

Linearization and Taylor expansion
--------------------------------------
Decorators to Taylor-expand around function parameters.

.. autosummary::
 :toctree: _autosummary

    linearize
    taylor_expand
