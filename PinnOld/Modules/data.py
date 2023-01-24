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