:github_url: https://github.com/google/neural-tangents/tree/main/docs/empirical.rst

.. default-role:: code

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


Linearization and Taylor expansion
--------------------------------------
Decorators to Taylor-expand around function parameters.

.. autosummary::
 :toctree: _autosummary

    linearize
    taylor_expand
