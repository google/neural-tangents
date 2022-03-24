#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 16:56:26 2022

@author: amir
"""

import numpy as onp
import quadprog


from matplotlib import pyplot as plt



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



def kappa1_coeffs(degree,h):
    alpha_ = -1.0
    for i in range(h):
        alpha_ = (2.0*alpha_ + (onp.sqrt(1-alpha_**2) + alpha_*(onp.pi - onp.arccos(alpha_)))/onp.pi)/3.0
    
    n=15*h+5*degree
    x = onp.sort(onp.concatenate((onp.linspace(alpha_, 1.0, num=201), onp.cos((2*onp.arange(n)+1)*onp.pi / (4*n))), axis=0))
    y = (onp.sqrt(1-x**2) + x*(onp.pi - onp.arccos(x)))/onp.pi
    grid_len = len(x)
    
    Z = onp.zeros((grid_len,degree+1))
    Z[:,0] = onp.ones(grid_len)
    for i in range(degree):
        Z[:,i+1] = Z[:,i] * x
    
    w = y
    U = Z.T

    beta_ = quadprog_solve_qp(onp.dot(U, U.T), -onp.dot(U,w) , onp.concatenate((Z[0:grid_len-1,:]-Z[1:grid_len,:], -onp.eye(degree+1)),axis=0), onp.zeros(degree+grid_len), Z[grid_len-1,:][onp.newaxis,:],y[grid_len-1])
    beta_[beta_ < 1e-5] = 0
    
    return beta_



def kappa0_coeffs(degree,h):
    alpha_ = -1.0
    for i in range(h):
        alpha_ = (1.0*alpha_ + (onp.sqrt(1-alpha_**2) + alpha_*(onp.pi - onp.arccos(alpha_)))/onp.pi)/2.0
    
    n=20*h+8*degree
    x = onp.sort(onp.concatenate((onp.linspace(alpha_, 1.0, num=201), onp.cos((2*onp.arange(n)+1)*onp.pi / (4*n))), axis=0))
    y = (onp.pi - onp.arccos(x))/onp.pi    
    grid_len = len(x)

    
    Z = onp.zeros((grid_len,degree+1))
    Z[:,0] = onp.ones(grid_len)
    for i in range(degree):
        Z[:,i+1] = Z[:,i] * x
    
    w = y 
    U = Z.T 

    beta_ = quadprog_solve_qp(onp.dot(U, U.T), -onp.dot(U,w) , onp.concatenate((Z[0:grid_len-1,:]-Z[1:grid_len,:], -onp.eye(degree+1)),axis=0), onp.zeros(degree+grid_len))#, Z[200,:][np.newaxis,:],y[200])
    beta_[beta_ < 1e-5] = 0
    
    return beta_