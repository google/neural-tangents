from jax import numpy as np
from jax.lax import fori_loop
from jaxopt import OSQP


def kappa0(x, is_x_matrix=True):
  if is_x_matrix:
    xxt = x @ x.T
    xnormsq = np.linalg.norm(x, axis=-1)**2
    prod = np.outer(xnormsq, xnormsq)
    return (1 - _arccos(xxt / _sqrt(prod)) / np.pi)
  else:  # vector input
    return (1 - _arccos(x) / np.pi)


def kappa1(x, is_x_matrix=True):
  if is_x_matrix:
    xxt = x @ x.T
    xnormsq = np.linalg.norm(x, axis=-1)**2
    prod = np.outer(xnormsq, xnormsq)
    return (_sqrt(prod - xxt**2) +
            (np.pi - _arccos(xxt / _sqrt(prod))) * xxt) / np.pi
  else:  # vector input
    return (_sqrt(1 - x**2) + (np.pi - _arccos(x)) * x) / np.pi


def _arccos(x):
  return np.arccos(np.clip(x, -1, 1))


def _sqrt(x):
  return np.maximum(x, 1e-20)**0.5


def poly_fitting_qp(xvals: np.ndarray,
                    fvals: np.ndarray,
                    weights: np.ndarray,
                    degree: int,
                    eq_last_point: bool = False):
  """ Computes polynomial coefficients that fitting input observations. 
    For a dot-product kernel (e.g., kappa0 or kappa1), coefficients of its 
    Taylor series expansion are always nonnegative.  Moreover, the kernel 
    function is a monotone increasing function. This can be solved by 
    Quadratic Programming (QP) under inequality constraints.
    """
  nx = len(xvals)
  x_powers = np.ones((nx, degree + 1), dtype=xvals.dtype)
  for i in range(degree):
    x_powers = x_powers.at[:, i + 1].set(x_powers[:, i] * xvals)

  y_weighted = fvals * weights
  x_powers_weighted = x_powers.T * weights

  dx_powers = x_powers[:-1, :] - x_powers[1:, :]

  # OSQP algorithm for solving min_x x'*Q*x + c'*x such that A*x=b, G*x<= h
  P = x_powers_weighted @ x_powers_weighted.T
  Q = .5 * (P.T + P + 1e-5 * np.eye(P.shape[0], dtype=xvals.dtype)
           )  # make sure Q is symmetric
  c = -x_powers_weighted @ y_weighted
  G = np.concatenate((dx_powers, -np.eye(degree + 1)), axis=0)
  h = np.zeros(nx + degree, dtype=xvals.dtype)

  if eq_last_point:
    A = x_powers[-1, :][None, :]
    b = fvals[-1:]
    return OSQP().run(params_obj=(Q, c), params_eq=(A, b),
                      params_ineq=(G, h)).params.primal
  else:
    return OSQP().run(params_obj=(Q, c), params_ineq=(G, h)).params.primal


def kappa0_coeffs(degree: int, num_layers: int):

  # A lower bound of kappa0^{(num_layers)} reduces to alpha_ from -1
  alpha_ = fori_loop(0, num_layers, lambda i, x_:
                     (x_ + kappa1(x_, is_x_matrix=False)) / 2., -1.)

  # Points for polynomial fitting contain (1) equi-spaced ones from [alpha_,1]
  # and (2) non-equi-spaced ones from [0,1]. For (2), cosine function is used
  # where more points are around 1.
  num_points = 20 * num_layers + 8 * degree
  x_eq = np.linspace(alpha_, 1., num=201)
  x_noneq = np.cos((2 * np.arange(num_points) + 1) * np.pi / (4 * num_points))
  xvals = np.sort(np.concatenate((x_eq, x_noneq)))
  fvals = kappa0(xvals, is_x_matrix=False)

  # For kappa0, we set all weights to be one.
  weights = np.ones(len(fvals), dtype=xvals.dtype)

  # Coefficients can be obtained by solving QP with OSQP jaxopt.
  coeffs = poly_fitting_qp(xvals, fvals, weights, degree)
  return np.where(coeffs < 1e-5, 0.0, coeffs)


def kappa1_coeffs(degree: int, num_layers: int):

  # A lower bound of kappa1^{(num_layers)} reduces to alpha_ from -1
  alpha_ = fori_loop(
      0, num_layers, lambda i, x_:
      (2. * x_ + kappa1(x_, is_x_matrix=False)) / 3., -1.)

  # Points for polynomial fitting contain (1) equi-spaced ones from [alpha_,1]
  # and (2) non-equi-spaced ones from [0,1]. For (2), cosine function is used
  # where more points are around 1.
  num_points = 15 * num_layers + 5 * degree
  x_eq = np.linspace(alpha_, 1., num=201)
  x_noneq = np.cos(
      (2. * np.arange(num_points) + 1.) * np.pi / (4. * num_points))
  xvals = np.sort(np.concatenate((x_eq, x_noneq)))
  fvals = kappa1(xvals, is_x_matrix=False)

  # For kappa1, we set all weights to be one.
  weights = np.ones(len(fvals), dtype=xvals.dtype)

  # For kappa1, we consider an equality condition for the last point
  # (close to 1) because the slope around 1 is much sharper.
  coeffs = poly_fitting_qp(xvals, fvals, weights, degree, eq_last_point=True)
  return np.where(coeffs < 1e-5, 0.0, coeffs)


def relu_ntk_coeffs(degree: int, num_layers: int):

  num_points = 20 * num_layers + 8 * degree
  x_eq = np.linspace(-1, 1., num=201)
  x_noneq = np.cos((2 * np.arange(num_points) + 1) * np.pi / (4 * num_points))
  x = np.sort(np.concatenate((x_eq, x_noneq)))

  kappa1s = {}
  kappa1s[0] = x
  for i in range(num_layers):
    kappa1s[i + 1] = kappa1(kappa1s[i], is_x_matrix=False)

  weights = np.linspace(0.0, 1.0, num=len(x)) + 2 / num_layers
  nngp_coeffs = poly_fitting_qp(x, kappa1s[num_layers], weights, degree)
  nngp_coeffs = np.where(nngp_coeffs < 1e-5, 0.0, nngp_coeffs)

  ntk = np.zeros(len(x), dtype=x.dtype)
  for i in range(num_layers + 1):
    z = kappa1s[i]
    for j in range(i, num_layers):
      z *= kappa0(kappa1s[j], is_x_matrix=False)
    ntk += z
  ntk_coeffs = poly_fitting_qp(x, ntk, weights, degree)
  ntk_coeffs = np.where(ntk_coeffs < 1e-5, 0.0, ntk_coeffs)

  return nngp_coeffs, ntk_coeffs
