#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 15:41:18 2022

@author: amir
"""

import numpy as onp
import quadprog
from matplotlib import pyplot as plt
from jax import numpy as np
from jax.numpy import linalg as LA
from sketching import standardsrht

def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
    qp_G = .5 * (P + P.T +1e-5*onp.eye(P.shape[0]))   # make sure P is symmetric
    qp_a = -q
    if A is not None:
        qp_C = -onp.vstack([A, G]).T
        qp_b = -onp.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]

def ntk_poly_coeffs(L,degree):
    n=15*L+5*degree
    Y = onp.zeros((201+n,L+1))
    Y[:,0] = onp.sort(onp.concatenate((onp.linspace(-1.0, 1.0, num=201), onp.cos((2*onp.arange(n)+1)*onp.pi / (4*n))), axis=0))

    grid_len = Y.shape[0]

    for i in range(L):
        Y[:,i+1] = (onp.sqrt(1-Y[:,i]**2) + Y[:,i]*(onp.pi - onp.arccos(Y[:,i])))/onp.pi
    
    y = onp.zeros(grid_len)
    for i in range(L+1):
        z = Y[:,i]
        for j in range(i,L):
            z = z*(onp.pi - onp.arccos(Y[:,j]))/onp.pi
        y = y + z
    
    Z = onp.zeros((grid_len,degree+1))
    Z[:,0] = onp.ones(grid_len)
    for i in range(degree):
        Z[:,i+1] = Z[:,i] * Y[:,0]
    
    
    weight_ = onp.linspace(0.0, 1.0, num=grid_len) + 2/L
    w = y * weight_
    U = Z.T * weight_

    coeffs = quadprog_solve_qp(onp.dot(U, U.T), -onp.dot(U,w) , onp.concatenate((Z[0:grid_len-1,:]-Z[1:grid_len,:], -onp.eye(degree+1)),axis=0), onp.zeros(degree+grid_len))
    coeffs[coeffs < 1e-5] = 0

    return coeffs

def poly_ntk_sketch(depth, polysketch, X):
    degree = polysketch.degree
    n = X.shape[0]
    
    ntk_coeff = ntk_poly_coeffs(depth, degree)
    
    norm_x = LA.norm(X, axis=1)
    normalizer = np.where(norm_x>0, norm_x, 1.0)
    x_normlzd = ((X.T / normalizer).T)
    
    polysketch_feats = polysketch.sketch(x_normlzd)
    
    sktch_dim = polysketch_feats[0].shape[1]
    
    Z = np.zeros((len(polysketch.rand_signs),n), dtype=np.complex64)
    for i in range(degree):
        Z = Z.at[sktch_dim*i:sktch_dim*(i+1),:].set(np.sqrt( ntk_coeff[i+1] ) * 
                                                  polysketch_feats[degree-i-1].T)
    
    Z = standardsrht(Z.T, polysketch.rand_inds, polysketch.rand_signs)
    Z = (Z.T * normalizer).T
    
    return np.concatenate(( np.sqrt(ntk_coeff[0]) * normalizer.reshape((n,1)), np.concatenate((Z.real, Z.imag), 1)), 1)

