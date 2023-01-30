#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
- Author(s) : Nima Hosseini Dashtbayaz
- Date: November 14th, 2022
- Title of ource code: Pinns
- Commit SHA: 3ff52aad08cb5315c4037824724240f43b336ab7
- Type: Python
- Repo URL: https://github.com/nimahsn/pinns.git
"""


import numpy as np
import tensorflow as tf
from scipy.optimize import fsolve

def kawaharaCosEqnsPos(U, a1, alpha, beta, sigma, N): #defining the equation
    kawaharaCosEqnsPos=np.zeros(N+2,dtype='float64') #first set all the equations equal to 0=0
    
    
    V=U[0] #Vector U contains the unknown coefficients and the unknown speed, we have N+2 unknowns
    a=U[1::]
    
    ### for the coefficients ###
    
    for k in range(N+1):
        sum1=0.#set the sums for the nonlinear term equal to 0 when solving for every coefficient
        sum2=0.
        for n in range(k,N+1):
            sum1=sum1+a[n]*a[n-k] 
        for n in range(0,k):
            sum2=sum2+a[n]*a[k-n] 
        kawaharaCosEqnsPos[k]=((V*a[k] + 1./2.*sigma*sum1 + 1./2.*sigma*sum2 - alpha*k**2*a[k] + beta*k**4*a[k]))
        
    kawaharaCosEqnsPos[N+1]=-a1+a[1] #for the last equation, linearize to obtain an equation for speed
    return kawaharaCosEqnsPos

def simulate_kdv(n_samples, phi_function, boundary_function, length, time, random_seed = 42, dtype=tf.float32) -> tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]:
    """
    Simulate the heat equation in 1D with a given initial condition and Dirichlet boundary conditions.
    Args:
        n_samples (int): number of samples to generate
        phi_function (function): Function that returns the initial condition of the heat equation on u.
        boundary_function (function): Function that returns the boundary condition of the heat equation on u.
        length (float): Length of the domain.
        time (float): Time frame of the simulation.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        dtype (tf.dtype, optional): Data type of the samples. Defaults to tf.float32.

    Returns:
        tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]: Samples of the heat equation. Returns a tuple of tensors (equation_samples, initial_samples, boundary_samples).
    """

    r = np.random.RandomState(random_seed)
    t = r.uniform(0, time, (n_samples, 1))
    x = r.uniform(0, length, (n_samples, 1))
    tx_eqn = np.concatenate((t, x), axis = 1)

    t_init = np.zeros((n_samples, 1))
    x_init = r.uniform(0, length, (n_samples, 1))
    tx_init = np.concatenate((t_init, x_init), axis = 1)

    t_boundary = r.uniform(0, time, (n_samples, 1))
    x_boundary = np.ones((n_samples//2, 1))*length
    x_boundary = np.append(x_boundary, np.zeros((n_samples - n_samples//2, 1)), axis=0)
    tx_boundary = np.concatenate((t_boundary, x_boundary), axis = 1)

    tx_eqn = tf.convert_to_tensor(tx_eqn, dtype = dtype)
    tx_init = tf.convert_to_tensor(tx_init, dtype = dtype)
    tx_boundary = tf.convert_to_tensor(tx_boundary, dtype = dtype)

    y_eqn = tf.zeros((n_samples, 1))
    y_phi = phi_function(tx_init)
    y_boundary = boundary_function(tx_boundary)

    # y_eqn = tf.convert_to_tensor(y_eqn, dtype = dtype)
    # y_phi = tf.convert_to_tensor(y_phi, dtype = dtype)
    # y_boundary = tf.convert_to_tensor(y_boundary, dtype = dtype)

    return (tx_eqn, y_eqn), (tx_init, y_phi), (tx_boundary, y_boundary)

def simulate_travelling_kawahara(n_samples, length, time, random_seed = 42, dtype=tf.float32) -> tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]:
    """
    Simulate the heat equation in 1D with a given initial condition and Dirichlet boundary conditions.
    Args:
        n_samples (int): number of samples to generate
        phi_function (function): Function that returns the initial condition of the heat equation on u.
        boundary_function (function): Function that returns the boundary condition of the heat equation on u.
        length (float): Length of the domain.
        time (float): Time frame of the simulation.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        dtype (tf.dtype, optional): Data type of the samples. Defaults to tf.float32.

    Returns:
        tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]: Samples of the heat equation. Returns a tuple of tensors (equation_samples, initial_samples, boundary_samples).
    """

    r = np.random.RandomState(random_seed)
    t = r.uniform(0, time, (n_samples, 1))
    x = r.uniform(0, length, (n_samples, 1))
    tx_eqn = np.concatenate((t, x), axis = 1)

    t_init = np.zeros((n_samples, 1))
    x_init = r.uniform(0, length, (n_samples, 1))
    tx_init = np.concatenate((t_init, x_init), axis = 1)

    t_boundary = r.uniform(0, time, (n_samples, 1))
    x_boundary = np.ones((n_samples//2, 1))*length
    x_boundary = np.append(x_boundary, np.zeros((n_samples - n_samples//2, 1)), axis=0)
    tx_boundary = np.concatenate((t_boundary, x_boundary), axis = 1)

    tx_eqn = tf.convert_to_tensor(tx_eqn, dtype = dtype)
    tx_init = tf.convert_to_tensor(tx_init, dtype = dtype)
    tx_boundary = tf.convert_to_tensor(tx_boundary, dtype = dtype)

    thirdAlpha = 1.
    fifthBeta =1/4
    nonlinSigma = 1.
    c = thirdAlpha - fifthBeta
    # L = (np.pi)*2
    # spacext = tx[:, 1:2]
    conSteps =1500 # number of continuation steps
    a1=1.0e-6 # beginning amplitude
    aF=1.0e-2 # ending amplitude
    aS = np.linspace(a1,aF,conSteps) # vector of free parameter a1 (amplitudes)
    velocities=np.zeros(conSteps) # Tracks all the velocities for bifurcation branch
    NN=21 # number of modes at which the Fourier series is truncated 
    uguess = np.zeros(NN+2)

    uguess[0] = c # uguess is our initial guess vector, it has zeros everywhere, except for first two elements: c and a1
    uguess[1] = a1
    V=c
    for k in range(conSteps):


        solution = fsolve(kawaharaCosEqnsPos, uguess, args=(aS[k], thirdAlpha, fifthBeta, nonlinSigma, NN),xtol=1.e-8) #notice amplitude changes with every iteration
        soln = solution[1::] #all the As (excludes speed)

        V = solution[0]
        uguess = np.concatenate((V,solution[1],aS[k],solution[3::]),axis=None) #update initial guess
        

    # generating the solution in real space made of cosines
        # phi = soln[0]*np.cos(0.*(tx[:, 1:2]-V*tx[:, 0:1]))
        phi_init = soln[0]*np.cos(0.*(tx_init[:, 1:2]-V*tx_init[:, 0:1]))
        phi_boundary = soln[0]*np.cos(0.*(tx_boundary[:, 1:2]-V*tx_boundary[:, 0:1]))
        # phix = -0.*soln[0]*np.sin(0.*(tx[:, 1:2]-V*tx[:, 0:1]))
        ii = 0.
        for aii in soln[1:]:
            ii = ii+1.
            # phi = phi + aii*np.cos(ii*(tx[:, 1:2]-V*tx[:, 0:1]))
            phi_init = phi_init + aii*np.cos(ii*(tx_init[:, 1:2]-V*tx_init[:, 0:1]))
            phi_boundary = phi_boundary + aii*np.cos(ii*(tx_boundary[:, 1:2]-V*tx_boundary[:, 0:1]))
            # phix = phix - (ii)*aii*np.sin(ii*(tx[:, 1:2]-V*tx[:, 0:1]))





        velocities[k]=solution[0]

    y_eqn = tf.zeros((n_samples, 1))
    y_phi = phi_init
    y_boundary = phi_boundary

    # y_eqn = tf.convert_to_tensor(y_eqn, dtype = dtype)
    # y_phi = tf.convert_to_tensor(y_phi, dtype = dtype)
    # y_boundary = tf.convert_to_tensor(y_boundary, dtype = dtype)

    return (tx_eqn, y_eqn), (tx_init, y_phi), (tx_boundary, y_boundary)