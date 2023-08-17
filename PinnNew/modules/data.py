"""
Module for the data generation of the heat, wave, schrodinger, burgers, and poisson equations.
"""
import tensorflow as tf
from typing import Callable, Tuple
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


def simulate_burgers(n_samples, init_function=None, boundary_function=None, n_init=None, n_bndry=None, random_seed=42, dtype=tf.float32) \
    -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
    """
    Simulate the burgers equation

    Args:
        n_samples (int): number of samples to generate
        init_function (function, optional): Function that returns the initial condition of the burgers equation. \
            If None, sin(pi*x) is used. Defaults to None.
        boundary_function (function, optional): Function that returns the boundary condition of the burgers equation. \
            If None, 0 is used. Defaults to None.
        boundary_samples (int, optional): number of boundary samples to generate. If None, then boundary_samples = n_samples. \
            Defaults to None.
        n_init (int, optional): number of initial samples to generate. If None, then n_init = n_samples. Defaults to None.
        n_bndry (int, optional): number of boundary samples to generate. If None, then n_bndry = n_samples. Defaults to None.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        dtype (tf.dtype, optional): Data type of the samples. Defaults to tf.float32.

    returns:
        tf.Tensor: Samples of the burgers equation. If training = True, returns a tuple of tensors (equation_samples, initial_samples, \
            n_samples).
    """

    if n_init is None:
        n_init = n_samples
    if n_bndry is None:
        n_bndry = n_samples

    if init_function is None:
        def init_function(tx):
            return -tf.sin(np.pi*tx[:, 1:])

    if boundary_function is None:
        def boundary_function(tx):
            return tf.zeros_like(tx[:, 1:])

    assert n_bndry % 2 == 0, "n_bndry must be even"

    tx_samples = tf.random.uniform((n_samples, 1), 0, 1, dtype=dtype, seed=random_seed)
    tx_samples = tf.concat([tx_samples, tf.random.uniform((n_samples, 1), -1, 1, seed=random_seed, dtype=dtype)], axis=1)
    y_samples = tf.zeros((n_samples, 1), dtype=dtype)

    tx_init = tf.zeros((n_init, 1), dtype=dtype)
    tx_init = tf.concat([tx_init, tf.random.uniform((n_init, 1), -1, 1, seed=random_seed, dtype=dtype)], axis=1)
    y_init = init_function(tx_init)

    tx_boundary = tf.random.uniform((n_bndry, 1), 0, 1, dtype=dtype, seed=random_seed)
    ones = tf.ones((n_bndry//2, 1), dtype=dtype)
    ones = tf.concat([ones, -tf.ones((n_bndry//2, 1), dtype=dtype)], axis=0)
    tx_boundary = tf.concat([tx_boundary, ones], axis=1)
    tx_boundary = tf.random.shuffle(tx_boundary, seed=random_seed)
    y_boundary = boundary_function(tx_boundary)
    
    return (tx_samples, y_samples), (tx_init, y_init), (tx_boundary, y_boundary)


def simulate_wave(n_samples, phi_function, psi_function, boundary_function, x_start=0.0, length=1.0, time=1.0, n_init=None, n_bndry=None, \
    random_seed=42, dtype = tf.float32) -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor, tf.Tensor], 
                                                 Tuple[tf.Tensor, tf.Tensor]]:
    """
    Simulate the wave equation in 1D or 2D with a given initial condition and Dirichlet boundary conditions.
    Args:
        n_samples (int): number of samples to generate
        dimension (int): dimension of the wave equation. Either 1 or 2.
        phi_function (function): Function that returns the initial condition of the wave equation on u.
        psi_function (function): Function that returns the initial condition of the wave equation on u_t.
        boundary_function_start (function): Function that returns the boundary condition of the wave equation on u at the start of the domain.
        boundary_function_end (function): Function that returns the boundary condition of the wave equation on u at the end of the domain.
        x_start (float, optional): Start of the domain. Defaults to 0.
        length (float, optional): Length of the domain. Defaults to 1.
        time (float, optional): Time frame of the simulation. Defaults to 1.
        n_init (int, optional): number of initial samples to generate. If None, then n_init = n_samples. Defaults to None.
        n_bndry (int, optional): number of boundary samples to generate. If None, then n_bndry = n_samples. Defaults to None.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        dtype (tf.dtype, optional): Data type of the samples. Defaults to tf.float32.

    Returns:
        tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]: Samples of the wave equation. \
            Returns a tuple of tensors (equation_samples, initial_samples, boundary_samples).
    
    """
    if n_init is None:
        n_init = n_samples
    if n_bndry is None:
        n_bndry = n_samples
    
    assert n_bndry % 2 == 0, "n_bndry must be even"
    
    t = tf.random.uniform((n_samples, 1), 0, time, dtype=dtype, seed=random_seed)
    x = tf.random.uniform((n_samples, 1), x_start, x_start + length, dtype=dtype, seed=random_seed)
    tx_eqn = tf.concat((t, x), axis=1)
    y_eqn = tf.zeros((n_samples, 1), dtype=dtype)

    t_init = tf.zeros((n_init, 1))
    x_init = tf.random.uniform((n_init, 1), x_start, x_start + length, dtype=dtype, seed=random_seed)
    tx_init = tf.concat((t_init, x_init), axis=1)

    t_boundary = tf.random.uniform((n_bndry, 1), 0, time, dtype=dtype, seed=random_seed)
    x_boundary = tf.ones((n_bndry//2, 1), dtype=dtype) * (x_start + length)
    x_boundary = tf.concat([x_boundary, tf.ones((n_bndry//2, 1), dtype=dtype) * x_start], axis=0)
    x_boundary = tf.random.shuffle(x_boundary, seed=random_seed)
    tx_boundary = tf.concat([t_boundary, x_boundary], axis=1)

    y_phi = phi_function(tx_init)
    y_psi = psi_function(tx_init)
    y_boundary = boundary_function(tx_boundary)

    return (tx_eqn, y_eqn), (tx_init, y_phi, y_psi), (tx_boundary, y_boundary)

def simulate_kdv(n_samples, phi_function, boundary_function, length, time, xstart,random_seed = 42, dtype=tf.float32) -> tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]:
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
    x = r.uniform(xstart, length, (n_samples, 1))
    tx_eqn = np.concatenate((t, x), axis = 1)

    t_init = np.zeros((n_samples, 1))
    x_init = r.uniform(xstart, length, (n_samples, 1))
    tx_init = np.concatenate((t_init, x_init), axis = 1)

    t_boundary = r.uniform(0, time, (n_samples, 1))
    x_boundary = np.ones((n_samples//2, 1))*length
    x_boundary = np.append(x_boundary, np.ones((n_samples - n_samples//2, 1))*xstart, axis=0)
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

def simulate_KP(n_samples, phi_function, boundary_function, time, xstart, xlength, ystart, ylength, normalize = False, random_seed = 42, dtype=tf.float32) -> tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]:


    r = np.random.RandomState(random_seed)
    t = r.uniform(0, time, (n_samples, 1))
    x = r.uniform(xstart, xlength, (n_samples, 1))
    y = r.uniform(ystart, ylength, (n_samples, 1))
    txy_eqn = np.concatenate((t, x, y), axis = 1)
    


    t_init = np.zeros((n_samples, 1))
    x_init = r.uniform(xstart, xlength, (n_samples, 1))
    y_init = r.uniform(ystart, ylength, (n_samples, 1))
    txy_init = np.concatenate((t_init, x_init, y_init), axis = 1)

    t_boundary = r.uniform(0, time, (n_samples, 1))
    x_boundary = np.ones((n_samples//2, 1))*xlength
    x_boundary = np.append(x_boundary, np.ones((n_samples - n_samples//2, 1))*xstart, axis=0)
    y_bnd_right = np.ones((n_samples, 1))*ylength
    y_bnd_left = np.ones((n_samples, 1))*ystart
    txy_boundary = np.concatenate((t_boundary, x_boundary, y), axis = 1)
    txy_boundary_y_left = np.concatenate((t_boundary, x, y_bnd_left), axis = 1)
    txy_boundary_y_right = np.concatenate((t_boundary, x, y_bnd_right), axis = 1)

    if normalize:
        u_phi = phi_function(txy_init)
        txy_eqn = np.concatenate((t/time, x/xlength, y/ylength), axis = 1)
        txy_init = np.concatenate((t_init/time, x_init/xlength, y_init/ylength), axis = 1)
        txy_boundary = np.concatenate((t_boundary/time, x_boundary/xlength, y/ylength), axis = 1)
        txy_boundary_y_left = np.concatenate((t_boundary/time, x/xlength, y_bnd_left/ylength), axis = 1)
        txy_boundary_y_right = np.concatenate((t_boundary, x/xlength, y_bnd_right/ylength), axis = 1)
    else: 
        u_phi = phi_function(txy_init)
        


    txy_eqn = tf.convert_to_tensor(txy_eqn, dtype = dtype)
    txy_init = tf.convert_to_tensor(txy_init, dtype = dtype)
    txy_boundary = tf.convert_to_tensor(txy_boundary, dtype = dtype)
    txy_boundary_y_left = tf.convert_to_tensor(txy_boundary_y_left, dtype = dtype)
    txy_boundary_y_right = tf.convert_to_tensor(txy_boundary_y_right, dtype = dtype)

    u_eqn = tf.zeros((n_samples, 1))
    
    u_boundary = boundary_function(txy_boundary)

    



    return (txy_eqn, u_eqn), (txy_init, u_phi), (txy_boundary, u_boundary), (txy_boundary_y_left, txy_boundary_y_right)
# def simulate_kdv(n_samples, init_function, bnd_fcn, xstart, length, time, compute_periodic = False, solver_function = None, nx = 256, nt = 201, shuffle_bnd = False, n_init=None, n_bndry=None,random_seed = 42, dtype=tf.float32) -> tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]:
#     """
#     Simulate the KdV equation in 1D with a given initial condition and Dirichlet boundary conditions.
#     Args:
#         n_samples (int): number of samples to generate
#         phi_function (function): Function that returns the initial condition of the heat equation on u.
#         boundary_function (function): Function that returns the boundary condition of the heat equation on u.
#         length (float): End of the domain.
#         time (float): Time frame of the simulation.
#         random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
#         dtype (tf.dtype, optional): Data type of the samples. Defaults to tf.float32.

#     Returns:
#         tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]: Samples of the heat equation. Returns a tuple of tensors (equation_samples, initial_samples, boundary_samples).
#     """
    
#     if n_init is None:
#         n_init = n_samples
#     if n_bndry is None:
#         n_bndry = n_samples
#     assert n_bndry % 2 == 0, "n_bndry must be even"

#     r = np.random.RandomState(random_seed)
#     t = r.uniform(0, time, (n_samples, 1))
#     x = r.uniform(xstart, length, (n_samples, 1))
#     tx_eqn = np.concatenate((t, x), axis = 1) 
#     if compute_periodic==True:
#         x_flat = np.linspace(xstart, length, nx)
#         t_flat = np.linspace(0, time, nt)
#         x_flat = tf.convert_to_tensor(x_flat, dtype = dtype)
#         t_flat = tf.convert_to_tensor(t_flat, dtype = dtype)
#         t_, x_ = tf.meshgrid(t_flat, x_flat)    
#         tx_periodic = tf.concat((tf.reshape(t_, (-1, 1)), tf.reshape(x_, (-1, 1))), axis=1)
#         y_phi = solver_function(tx_periodic,nx,nt)
#         y_phi_matrix = tf.reshape(y_phi,x_.shape)
#         y_boundary = tf.reshape(y_phi_matrix[0, :], (-1, 1)) # on ne side of the boundary, this is nt long
    
#     t_init = tf.zeros((n_init, 1), dtype=dtype)
#     x_init = tf.random.uniform((n_init, 1), xstart, length, dtype=dtype, seed=random_seed)
#     tx_init = tf.concat((t_init, x_init), axis=1)

#     t_boundary = tf.random.uniform((n_bndry, 1), 0, time, dtype=dtype, seed=random_seed)
#     x_boundary = tf.ones((n_bndry//2, 1), dtype=dtype) * xstart
#     x_boundary = tf.concat([x_boundary, tf.ones((n_bndry//2, 1) , dtype=dtype) * length], axis=0)
#     x_boundary = tf.random.shuffle(x_boundary, seed=random_seed) 
#     x_boundary_start = tf.cast(tf.reshape([xstart] * n_bndry, (-1, 1)), dtype = dtype)
#     x_boundary_end = tf.cast(tf.reshape([length] * n_bndry, (-1, 1)), dtype = dtype)
#     tx_boundary_start = tf.concat((t_boundary, x_boundary_start), axis=1)
#     tx_boundary_end = tf.concat((t_boundary, x_boundary_end), axis=1)
#     tx_boundary = tf.concat((t_boundary, x_boundary), axis=1)

#     # Are these 3 lines necessary?
#     tx_eqn = tf.convert_to_tensor(tx_eqn, dtype = dtype)
#     tx_init = tf.convert_to_tensor(tx_init, dtype = dtype)

    
#     #sample points
#     samples_indices = tf.random.shuffle(tf.range(tf.shape(tx_eqn)[0], dtype=tf.int32), seed=random_seed)[:n_samples]
#     boundary_indices = tf.random.shuffle(tf.range(tf.shape(tx_boundary_start)[0], dtype=tf.int32), seed=random_seed)[:n_bndry]
#     init_indices = tf.random.shuffle(tf.range(tf.shape(tx_init)[0], dtype=tf.int32), seed=random_seed)[:n_init]

#     y_eqn = tf.zeros((n_samples, 1), dtype=dtype)
#     y_init = init_function(tx_init)
#     y_boundary = bnd_fcn(tx_boundary)
    
#     # shuffle 
#     tx_eqn = tf.gather(tx_eqn, samples_indices)
#     tx_boundary_start = tf.gather(tx_boundary_start, boundary_indices)
#     tx_boundary_end = tf.gather(tx_boundary_end, boundary_indices)
#     y_boundary = tf.gather(y_boundary, boundary_indices)
#     tx_init = tf.gather(tx_init, init_indices)
#     y_init = tf.gather(y_init, init_indices)
        

#     return (tx_eqn, y_eqn), (tx_init, y_init), (tx_boundary_start, y_boundary), (tx_boundary_end, y_boundary), (tx_boundary, y_boundary)

def simulate_heat(n_samples, phi_function, boundary_function, length, time, n_init=None, n_bndry=None, random_seed=2, dtype=tf.float32) \
    -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
    """
    Simulate the heat equation in 1D with a given initial condition and Dirichlet boundary conditions.
    Args:
        n_samples (int): number of samples to generate
        phi_function (function): Function that returns the initial condition of the heat equation on u.
        boundary_function (function): Function that returns the boundary condition of the heat equation on u.
        length (float): Length of the domain.
        time (float): Time frame of the simulation.
        n_init (int, optional): number of initial samples to generate. If None, then n_init = n_samples. Defaults to None.
        n_bndry (int, optional): number of boundary samples to generate. If None, then n_bndry = n_samples. Defaults to None.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        dtype (tf.dtype, optional): Data type of the samples. Defaults to tf.float32.

    Returns:
        tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]: Samples of the heat equation. \
            Returns a tuple of tensors (equation_samples, initial_samples, boundary_samples).
    """
    if n_init is None:
        n_init = n_samples
    if n_bndry is None:
        n_bndry = n_samples
    assert n_bndry % 2 == 0, "n_bndry must be even"

    t = tf.random.uniform((n_samples, 1), 0, time, dtype=dtype, seed=random_seed)
    x = tf.random.uniform((n_samples, 1), 0, length, dtype=dtype, seed=random_seed)
    tx_eqn = tf.concat((t, x), axis=1)

    t_init = tf.zeros((n_init, 1), dtype=dtype)
    x_init = tf.random.uniform((n_init, 1), 0, length, dtype=dtype, seed=random_seed)
    tx_init = tf.concat((t_init, x_init), axis=1)

    t_boundary = tf.random.uniform((n_bndry, 1), 0, time, dtype=dtype, seed=random_seed)
    x_boundary = tf.ones((n_bndry//2, 1), dtype=dtype) * length
    x_boundary = tf.concat([x_boundary, tf.zeros((n_bndry//2, 1))], axis=0)
    x_boundary = tf.random.shuffle(x_boundary, seed=random_seed)
    tx_boundary = tf.concat([t_boundary, x_boundary], axis=1)

    y_eqn = tf.zeros((n_samples, 1), dtype=dtype)
    y_phi = phi_function(tx_init)
    y_boundary = boundary_function(tx_boundary)

    return (tx_eqn, y_eqn), (tx_init, y_phi), (tx_boundary, y_boundary)


def simulate_poisson(n_samples, rhs_function, boundary_function, x_start: float = 0.0, length: float = 1.0, n_bndry=None, random_seed=42, \
    dtype=tf.float32) -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
    """
    Simulate the Poisson equation in 1D with a given right hand side and Dirichlet boundary conditions.
    Args:
        n_samples (int): number of samples to generate
        rhs_function (function): Function that returns the right hand side of the Poisson equation.
        boundary_function (function): Function that returns the boundary condition of the Poisson equation on u.
        boundary_start (float, optional): Start of the boundary. Defaults to 0.0.
        length (float, optional): Length of the domain. Defaults to 1.0.
        n_bndry (int, optional): number of boundary samples to generate. If None, then n_bndry = n_samples. Defaults to None.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        dtype (tf.dtype, optional): Data type of the samples. Defaults to tf.float32.
    """
    if n_bndry is None:
        n_bndry = n_samples
    assert n_bndry % 2 == 0, "n_bndry must be even"
    
    x_eqn = tf.random.uniform((n_samples, 1), x_start, x_start + length, dtype=dtype, seed=random_seed)
    rhs_eqn = rhs_function(x_eqn)

    x_boundary = tf.ones((n_bndry//2, 1), dtype=dtype) * x_start
    x_boundary = tf.concat([x_boundary, tf.ones((n_bndry//2, 1), dtype=dtype) * (x_start + length)], axis=0)
    x_boundary = tf.random.shuffle(x_boundary, seed=random_seed)
    u_boundary = boundary_function(x_boundary)

    return (x_eqn, rhs_eqn), (x_boundary, u_boundary)

def simulate_advection(n_samples, boundary_function: Callable = None, x_start: float = 0.0, length: float = 1, n_bndry=None, \
     random_seed=42, dtype=tf.float32):
    """
    Simulate the steady advection diffusion equation in 1D with a given boundary conditions.
    Args:
        n_samples (int): number of samples to generate
        boundary_function (function): Function that returns the boundary condition of the advection diffusion equation on u.\
            If None, the boundary condition is set to zero on start and one on end. Defaults to None.
        x_start (float, optional): Start of the boundary. Defaults to 0.0.
        length (float, optional): Length of the domain. Defaults to 1.0.
        n_bndry (int, optional): number of boundary samples to generate. If None, then n_bndry = n_samples. Defaults to None.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        dtype (tf.dtype, optional): Data type of the samples. Defaults to tf.float32.
    """
    if n_bndry is None:
        n_bndry = n_samples
    assert n_bndry % 2 == 0, "n_bndry must be even"

    if boundary_function is None:
        def boundary_function(x):
            return tf.where(x == x_start, 0.0, 1.0)
    
    x_eqn = tf.random.uniform((n_samples, 1), x_start, x_start + length, dtype=dtype, seed=random_seed)
    f_eqn = tf.zeros((n_samples, 1))

    x_boundary = tf.ones((n_bndry//2, 1), dtype=dtype) * x_start
    x_boundary = tf.concat([x_boundary, tf.ones((n_bndry//2, 1), dtype=dtype) * (x_start + length)], axis=0)
    x_boundary = tf.random.shuffle(x_boundary, seed=random_seed)
    u_boundary = boundary_function(x_boundary)

    return (x_eqn, f_eqn), (x_boundary, u_boundary)

def simulate_schrodinger(n_samples, init_function, x_start, length, time, n_init=None, n_bndry=None, random_seed=42, dtype=tf.float32):
    """
    Simulate the Schrodinger equation in 1D with a given initial condition.
    Args:
        n_samples (int): number of samples to generate
        init_function (function): Function that returns the initial condition of the Schrodinger equation.
        x_start (float): Start of the boundary.
        length (float): Length of the domain.
        time (float): Time of the simulation.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        n_init (int, optional): number of initial condition samples to generate. If None, then n_init = n_samples. Defaults to None.
        n_bndry (int, optional): number of boundary samples to generate. If None, then n_bndry = n_samples. Defaults to None.
        dtype (tf.dtype, optional): Data type of the samples. Defaults to tf.float32.

    Returns:
        Tuple[Tuple[tf.tensor, tf.tensor], Tuple[tf.tensor, tf.tensor], tf.tensor]: Tuple of tuples of tensors. \
            The first tuple contains the equation samples, the second tuple the initial condition samples and the third tensor the \
                boundary condition samples.
    """
    if n_init is None:
        n_init = n_samples
    if n_bndry is None:
        n_bndry = n_samples

    
    t = tf.random.uniform((n_samples, 1), 0, time, dtype=dtype, seed=random_seed)
    x = tf.random.uniform((n_samples, 1), x_start, x_start + length, dtype=dtype, seed=random_seed)
    tx_eqn = tf.concat((t, x), axis=1)
    y_eqn = tf.zeros((n_samples, 2), dtype=dtype)

    t_init = tf.zeros((n_init, 1), dtype=dtype)
    x_init = tf.random.uniform((n_init, 1), x_start, x_start + length, dtype=dtype, seed=random_seed)
    tx_init = tf.concat((t_init, x_init), axis=1)
    y_init = init_function(tx_init)

    t_boundary = tf.random.uniform((n_bndry, 1), 0, time, dtype=dtype, seed=random_seed)
    x_boundary_start = tf.ones((n_bndry, 1), dtype=dtype) * x_start
    x_boundary_end = tf.ones((n_bndry, 1), dtype=dtype) * (x_start + length)
    txx_boundary = tf.concat([t_boundary, x_boundary_start, x_boundary_end], axis=1)

    return (tx_eqn, y_eqn), (tx_init, y_init), txx_boundary

def simulate_reaction_diffusion(n_samples, n_init, n_boundary, solver_function, u0, nu, rho, x_start=0.0, length=2*np.pi, time=1.0,
                                time_steps=200, x_steps=256, interior_only = True, add_bnd = False, return_mesh=True, random_seed=42,
                                  dtype=tf.float32):
    """
    Simulate the reaction diffusion equation in 1D with dirichlet initial and boundary condition.
    Args:
        n_samples (int): number of samples to generate
        n_init (int): number of initial condition samples to generate
        n_boundary (int): number of boundary condition samples to generate
        solver_function (function): Function that returns the solution of the reaction diffusion equation.
        x_start (float, optional): Start of the boundary. Defaults to 0.0.
        length (float, optional): Length of the domain. Defaults to 2*np.pi.
        time (float, optional): Time of the simulation. Defaults to 1.0.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        dtype (tf.dtype, optional): Data type of the samples. Defaults to tf.float32.
    """

    dx = length / x_steps
    dt = time / time_steps
    x = np.arange(0, length, dx) # not inclusive of the last point
    t = np.linspace(0, time, time_steps)
    #convert to tf
    x = tf.convert_to_tensor(x, dtype=dtype)
    t = tf.convert_to_tensor(t, dtype=dtype)
    X, T = tf.meshgrid(x, t)
    U = solver_function(u0, nu, rho, x_steps, time_steps)
    # convert u to tf
    U = tf.convert_to_tensor(U, dtype=dtype)
    U = tf.reshape(U, X.shape)

    if not interior_only:
        tx_samples = tf.concat((tf.reshape(T, (-1, 1)), tf.reshape(X, (-1, 1))), axis=1)
        u_samples = tf.reshape(U, (-1, 1))
    else:
        X_no_bnd = X[1:, 1:]
        T_no_init = T[1:, 1:]
        U_no_bnd_init = U[1:, 1:]
        tx_samples = tf.concat((tf.reshape(T_no_init, (-1, 1)), tf.reshape(X_no_bnd, (-1, 1))), axis=1)
        u_samples = tf.reshape(U_no_bnd_init, (-1, 1))

    x_boundary_start = tf.reshape(X[:, 0], (-1, 1))
    x_boundary_end = tf.reshape(X[:, -1], (-1, 1))
    u_boundary_start = tf.reshape(U[:, 0], (-1, 1))
    u_boundary_end = tf.reshape(U[:, -1], (-1, 1))
    tx_boundary_start = tf.concat((t[:, None], x_boundary_start), axis=1)
    tx_boundary_end = tf.concat((t[:, None], x_boundary_end), axis=1)
    tx_boundary = tf.concat((tx_boundary_start, tx_boundary_end), axis=0)
    u_boundary = tf.concat((u_boundary_start, u_boundary_end), axis=0)

    t_init = tf.zeros((x_steps, 1), dtype=dtype)
    tx_init = tf.concat((t_init, x[:, None]), axis=1)
    u_init = tf.reshape(U[0, :], (-1, 1))

    #sample points
    samples_indices = tf.random.shuffle(tf.range(tf.shape(tx_samples)[0], dtype=tf.int32), seed=random_seed)[:n_samples]
    boundary_indices = tf.random.shuffle(tf.range(tf.shape(tx_boundary)[0], dtype=tf.int32), seed=random_seed)[:n_boundary]
    init_indices = tf.random.shuffle(tf.range(tf.shape(tx_init)[0], dtype=tf.int32), seed=random_seed)[:n_init]

    tx_samples = tf.gather(tx_samples, samples_indices)
    u_samples = tf.gather(u_samples, samples_indices)
    samples_residuals = tf.zeros_like(u_samples, dtype=dtype)
    tx_boundary = tf.gather(tx_boundary, boundary_indices)
    u_boundary = tf.gather(u_boundary, boundary_indices)
    tx_init = tf.gather(tx_init, init_indices)
    u_init = tf.gather(u_init, init_indices)

    if add_bnd:
        tx_samples = tf.concat((tx_samples, tx_init, tx_boundary), axis=0)
        u_samples = tf.concat((u_samples, u_init, u_boundary), axis=0)
        samples_residuals = tf.concat((samples_residuals, tf.zeros_like(u_init, dtype=dtype), tf.zeros_like(u_boundary, dtype=dtype)), axis=0)

    if return_mesh:
        return (tx_samples, u_samples, samples_residuals), (tx_init, u_init), (tx_boundary, u_boundary), (X, T, U)
    return (tx_samples, u_samples, samples_residuals), (tx_init, u_init), (tx_boundary, u_boundary)

def simulate_klein_gordon(n_colloc, n_init, n_bnd, rhs_function=None, init_function=None, bnd_function=None, init_ut_function=None, 
                          x_start=0.0, length=1.0, time=1.0, dtype=tf.float32, random_seed=42):
    """
    Simulate the Klein Gordon equation in 1D with dirichlet-neuman initial and dirichlet boundary condition.

    Args:
        n_colloc (int): number of collocation points to generate
        n_init (int): number of initial condition samples to generate
        n_bnd (int): number of boundary condition samples to generate
        rhs_function (function, optional): Function that returns the right hand side of the PDE. Defaults to None. If None, zero is used.
        init_function (function, optional): Function that returns the initial condition. Defaults to None. If None, zero is used.
        bnd_function (function, optional): Function that returns the boundary condition. Defaults to None. If None, zero is used.
        init_ut_function (function, optional): Function that returns the initial condition for the time derivative. Defaults to None. \
            If None, zero is used.
        x_start (float, optional): Start of the boundary. Defaults to 0.0.
        length (float, optional): Length of the domain. Defaults to 1.0.
        time (float, optional): Time of the simulation. Defaults to 1.0.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        dtype (tf.dtype, optional): Data type of the samples. Defaults to tf.float32.
    """

    tx_colloc = tf.random.uniform((n_colloc, 2), minval=[0.0, x_start], maxval=[time, x_start+length], dtype=dtype, seed=random_seed)
    if rhs_function is None:
        rhs = tf.zeros((n_colloc, 1), dtype=dtype)
    else:
        rhs = rhs_function(tx_colloc)

    tx_init = tf.random.uniform((n_init, 2), minval=[0.0, x_start], maxval=[0.0, x_start+length], dtype=dtype, seed=random_seed)
    if init_function is None:
        u_init = tf.zeros((n_init, 1), dtype=dtype)
    else:
        u_init = init_function(tx_init)
    if init_ut_function is None:
        ut_init = tf.zeros((n_init, 1), dtype=dtype)
    else:
        ut_init = init_ut_function(tx_init)
    
    tx_bnd = tf.random.uniform((n_bnd // 2, 2), minval=[0.0, x_start], maxval=[time, x_start], dtype=dtype, seed=random_seed)
    tx_bnd = tf.concat((tx_bnd, tf.random.uniform((n_bnd // 2, 2), minval=[0.0, x_start+length], maxval=[time, x_start+length], \
                                                  dtype=dtype, seed=random_seed)), axis=0)
    tx_bnd = tf.gather(tx_bnd, tf.random.shuffle(tf.range(tf.shape(tx_bnd)[0], dtype=tf.int32), seed=random_seed))
    if bnd_function is None:
        u_bnd = tf.zeros((n_bnd, 1), dtype=dtype)
    else:
        u_bnd = bnd_function(tx_bnd)

    return (tx_colloc, rhs), (tx_init, u_init, ut_init), (tx_bnd, u_bnd)

def simulate_transport(n_samples, n_init, n_boundary, solver_function, beta, u0='sin(x)', length=2*np.pi, time=1.0,
                                time_steps=100, x_steps=256, interior_only = True, add_bnd = False, return_mesh=True, random_seed=42,
                                  dtype=tf.float32):
    dx = length / x_steps
    dt = time / time_steps
    x = np.arange(0, length, dx) # not inclusive of the last point
    t = np.linspace(0, time, time_steps)
    #convert to tf
    x = tf.convert_to_tensor(x, dtype=dtype)
    t = tf.convert_to_tensor(t, dtype=dtype)
    X, T = tf.meshgrid(x, t)
    U = solver_function(u0, 0.0, beta, 0, x_steps, time_steps)
    # convert u to tf
    U = tf.convert_to_tensor(U, dtype=dtype)
    U = tf.reshape(U, X.shape)

    if not interior_only:
        tx_samples = tf.concat((tf.reshape(T, (-1, 1)), tf.reshape(X, (-1, 1))), axis=1)
        u_samples = tf.reshape(U, (-1, 1))
    else:
        X_no_bnd = X[1:, 1:]
        T_no_init = T[1:, 1:]
        U_no_bnd_init = U[1:, 1:]
        tx_samples = tf.concat((tf.reshape(T_no_init, (-1, 1)), tf.reshape(X_no_bnd, (-1, 1))), axis=1)
        u_samples = tf.reshape(U_no_bnd_init, (-1, 1))

    x_boundary_start = tf.reshape(X[:, 0], (-1, 1))
    x_boundary_end = tf.reshape([length] * time_steps, (-1, 1))
    u_boundary = tf.reshape(U[:, 0], (-1, 1)) # exact solution for start and end of the domain. Boundary condition is periodic, so the end is the same as the start
    tx_bnd_start = tf.concat((t[:, None], x_boundary_start), axis=1)
    tx_bnd_end = tf.concat((t[:, None], x_boundary_end), axis=1)

    t_init = tf.zeros((x_steps, 1), dtype=dtype)
    tx_init = tf.concat((t_init, x[:, None]), axis=1)
    u_init = tf.reshape(U[0, :], (-1, 1))

    #sample points
    samples_indices = tf.random.shuffle(tf.range(tf.shape(tx_samples)[0], dtype=tf.int32), seed=random_seed)[:n_samples]
    boundary_indices = tf.random.shuffle(tf.range(tf.shape(tx_bnd_start)[0], dtype=tf.int32), seed=random_seed)[:n_boundary]
    init_indices = tf.random.shuffle(tf.range(tf.shape(tx_init)[0], dtype=tf.int32), seed=random_seed)[:n_init]

    tx_samples = tf.gather(tx_samples, samples_indices)
    u_samples = tf.gather(u_samples, samples_indices)
    samples_residuals = tf.zeros_like(u_samples, dtype=dtype)
    tx_bnd_start = tf.gather(tx_bnd_start, boundary_indices)
    tx_bnd_end = tf.gather(tx_bnd_end, boundary_indices)
    u_boundary = tf.gather(u_boundary, boundary_indices)
    tx_init = tf.gather(tx_init, init_indices)
    u_init = tf.gather(u_init, init_indices)

    if add_bnd:
        tx_samples = tf.concat((tx_samples, tx_init, tx_bnd_start, tx_bnd_end), axis=0)
        u_samples = tf.concat((u_samples, u_init, u_boundary, u_boundary), axis=0)
        samples_residuals = tf.concat((samples_residuals, tf.zeros_like(u_init, dtype=dtype), 
                                       tf.zeros_like(u_boundary, dtype=dtype), tf.zeros_like(u_boundary, dtype=dtype)), axis=0)

    if return_mesh:
        return (tx_samples, u_samples, samples_residuals), (tx_init, u_init), (tx_bnd_start, tx_bnd_end, u_boundary), (X, T, U)
    return (tx_samples, u_samples, samples_residuals), (tx_init, u_init), (tx_bnd_start, tx_bnd_end, u_boundary)

def kawaharaCosEqnsPos(U, a1, alpha, beta, sigma, N): 
    kawaharaCosEqnsPos = np.zeros(N+2,dtype='float64')
    V=U[0] # Vector U contains the unknown coefficients and the unknown speed, we have N+2 unknowns
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
        
    kawaharaCosEqnsPos[N+1] = -a1 + a[1] 
    return kawaharaCosEqnsPos

def simulate_travel_kawahara(n_samples, x_start, length, time, thirdAlpha = 1., fifthBeta = 1./4., nonlinSigma = 1, aF = 0.001, conSteps = 1500 , moving_frame = False, random_seed = 42, dtype=tf.float32) -> tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]:
    """
    Simulate the Kawahara equation in 1D with a given.
    Args:
        n_samples (int): number of samples to generate
        x_start (float): start of the domain.
        length (float): end of the domain.
        time (float): Time frame of the simulation.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        dtype (tf.dtype, optional): Data type of the samples. Defaults to tf.float32.

    Returns:
        tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]: Samples of the heat equation. Returns a tuple of tensors (equation_samples, initial_samples, boundary_samples).
    """

  

    
    t_eqn = tf.random.uniform((n_samples, 1), 0, time, dtype=dtype, seed=random_seed)
    x_eqn = tf.random.uniform((n_samples, 1), x_start, length, dtype=dtype, seed=random_seed)
    # x = tf.sort(x, axis=0, direction='ASCENDING', name=None)

    
    
    t_init = tf.zeros((n_samples, 1), dtype=dtype)
    x_init = tf.random.uniform((n_samples, 1), x_start, length, seed=random_seed, dtype=dtype)
    tx_init = tf.concat([t_init, x_init], axis=1)

    t_boundary = tf.random.uniform((n_samples, 1), 0, time, dtype=dtype, seed=random_seed)
    bnd_start = tf.ones((n_samples, 1), dtype=dtype)*x_start
    bnd_end = tf.ones((n_samples, 1), dtype=dtype)*length
    x_boundary = tf.concat([tf.reshape([x_start]*(n_samples//2),(-1,1)), tf.reshape([length]*(n_samples//2),(-1,1))], axis=0)
    x_boundary = tf.random.shuffle(x_boundary, seed=random_seed)

    

    c = thirdAlpha - fifthBeta
    
    a1 = 1.0e-6 # beginning amplitude

    aS = np.linspace(a1,aF,conSteps) # vector of free parameter a1 (amplitudes)
    velocities = np.zeros(conSteps) # Tracks all the velocities for bifurcation branch
    NN = 21 # number of modes at which the Fourier series is truncated 
    uguess = np.zeros(NN+2)

    uguess[0] = c # uguess is our initial guess vector, it has zeros everywhere, except for first two elements: c and a1
    uguess[2] = a1
    V = c
    for k in range(conSteps):
        solution = fsolve(kawaharaCosEqnsPos, uguess, args=(aS[k], thirdAlpha, fifthBeta, nonlinSigma, NN),xtol=1.e-8) 
        soln = solution[1::] # all the As (excludes speed)
        V = solution[0]
        uguess = np.concatenate((V,solution[1],aS[k],solution[3::]),axis=None) #update initial guess
        velocities[k] = solution[0]
        
    phi_init = soln[0]*np.cos(0.*(tx_init[:, 1:2]-0*tx_init[:, 0:1]))
    # phi_boundary = soln[0]*np.cos(0.*(tx_boundary[:, 1:2]-0*tx_boundary[:, 0:1]))
    
    if moving_frame:
        c = solution[0]
    else:
        c = 0.
    nt = n_samples # we only care about sampling from this domain for t_boundary as x_boundary is just the left and right points
    # dt = time/(nt-1) # care needs to be taken to ensure scaling does not result too high/low spatial points time = xdomain/c
    # x_bnd_flat = np.arange(x_start,length,dt*c)
    # t_bnd_flat = np.linspace(0, time, nt)
    # t_bnd_mesh, x_bnd_mesh = tf.meshgrid(t_bnd_flat, x_bnd_flat) # doing mesh grids to ensure there is no difference between left and right, equivalently could only conisder either left or right bounds
    # tx_bnd = tf.concat((tf.reshape(t_bnd_mesh, (-1, 1)), tf.reshape(x_bnd_mesh, (-1, 1))), axis=1)
    
    
    # ind_bc = tf.range(n_samples)
    # ind_bc_shuffled = tf.random.shuffle(ind_bc, seed = random_seed)[:n_samples]
    # t = tf.cast(t_bnd_flat, dtype=dtype)
    # t_boundary = tf.gather(t, ind_bc_shuffled)
    # # t_boundary = t # ordering time
    # t_boundary = tf.reshape(t_boundary,(n_samples,1))
    tx_boundary_start = tf.concat([t_boundary, bnd_start], axis=1) 
    tx_boundary_end = tf.concat([t_boundary, bnd_end], axis=1) 
    tx_boundary = tf.concat([t_boundary, x_boundary], axis=1)
    
    # ind_eqn = tf.range(n_samples)
    # ind_eqn_shuffled = tf.random.shuffle(ind_eqn, seed = random_seed)[:n_samples]
    # ind_eqn_shuffled2 = tf.random.shuffle(ind_eqn, seed = random_seed+1)[:n_samples]
    # x_bnd_flat = tf.cast(x_bnd_flat, dtype=dtype)
    # print(len(x_bnd_flat))
    
    # x_eqn = tf.gather(x_bnd_flat, ind_eqn_shuffled)
    # x_eqn= tf.reshape(x_eqn,(n_samples,1))
    # t_eqn = tf.gather(t, ind_eqn_shuffled2)
    # t_eqn= tf.reshape(t_eqn,(n_samples,1))
    tx_eqn = tf.concat([t_eqn, x_eqn], axis=1) 
    u_bnd = soln[0]*np.cos(0.*(tx_boundary[:, 1:2]-c*tx_boundary[:, 0:1])) 
    u_exact = soln[0]*np.cos(0.*(tx_eqn[:, 1:2])-c*tx_eqn[:, 0:1]) 
    ii = 0.
    
    for aii in soln[1:]:
        ii = ii+1.
        phi_init = phi_init + aii*np.cos(ii*(tx_init[:, 1:2]-c*tx_init[:, 0:1]))
        u_exact +=  aii*np.cos(ii*(tx_eqn[:, 1:2]-c*tx_eqn[:, 0:1])) 
        u_bnd +=  aii*np.cos(ii*(tx_boundary[:, 1:2]-c*tx_boundary[:, 0:1])) 
        
    # u_bnd = tf.reshape(u_bnd, x_bnd_mesh.shape)
    # u_bnd_left = tf.cast(u_bnd[0,:], dtype = dtype)
    # y_boundary = tf.gather(u_bnd_left, ind_bc_shuffled)
    y_boundary = u_bnd

    
    plt.plot(tx_eqn[:,1:2],u_exact,'.')
    plt.plot(tx_boundary_start[:,1:2][:10],y_boundary[:10],'.')
    plt.plot(tx_boundary_end[:,1:2][:10],y_boundary[:10],'.')
    
    plt.show()

    y_eqn = tf.zeros((n_samples, 1))
    y_phi = phi_init

    return (tx_eqn, y_eqn, u_exact), (tx_init, y_phi), (tx_boundary_start, y_boundary), (tx_boundary_end, y_boundary), (tx_boundary, y_boundary), (solution, velocities)

def simulate_c_parametrization(n_samples, x_start, length, thirdAlpha = 1., fifthBeta = 1./4., nonlinSigma = 1, a1 = 1.0e-6, aF = 0.001, random_seed = 42, dtype=tf.float32) -> tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]:



    c_init_val = thirdAlpha-fifthBeta
    conSteps = 2000 # number of continuation steps


    aS = np.linspace(a1,aF,conSteps) # vector of free parameter a1 (amplitudes)
    velocities = np.zeros(conSteps) # Tracks all the velocities for bifurcation branch
    NN = 21 # number of modes at which the Fourier series is truncated 
    uguess = np.zeros(NN+2)

    uguess[0] = c_init_val # uguess is our initial guess vector, it has zeros everywhere, except for first two elements: c and a1
    uguess[2] = a1
    V = c_init_val
    for k in range(conSteps):
        solution = fsolve(kawaharaCosEqnsPos, uguess, args=(aS[k], thirdAlpha, fifthBeta, nonlinSigma, NN),xtol=1.e-8) 
        soln = solution[1::] # all the As (excludes speed)
        V = solution[0]
        uguess = np.concatenate((V,solution[1],aS[k],solution[3::]),axis=None) #update initial guess
        velocities[k] = solution[0]
        
        if k == 0:
            solution_init = solution
        else: 
            pass
    c_init_val = thirdAlpha-fifthBeta
    c_eqn = tf.random.uniform((n_samples, 1), c_init_val, velocities[-1], dtype=dtype, seed=random_seed)
    x_eqn = tf.random.uniform((n_samples, 1), x_start, length, dtype=dtype, seed=random_seed)
    
    c_init = tf.ones((n_samples, 1), dtype=dtype)*c_init_val
    x_init = tf.random.uniform((n_samples, 1), x_start, length, seed=random_seed, dtype=dtype)
    cx_init = tf.concat([c_init, x_init], axis=1)

    c_boundary = tf.random.uniform((n_samples, 1), c_init_val, velocities[-1], dtype=dtype, seed=random_seed)
    bnd_start = tf.ones((n_samples, 1), dtype=dtype)*x_start
    bnd_end = tf.ones((n_samples, 1), dtype=dtype)*length
    x_boundary = tf.concat([tf.reshape([x_start]*(n_samples//2),(-1,1)), tf.reshape([length]*(n_samples//2),(-1,1))], axis=0)
    x_boundary = tf.random.shuffle(x_boundary, seed=random_seed)


    cx_boundary_start = tf.concat([c_boundary, bnd_start], axis=1) 
    cx_boundary_end = tf.concat([c_boundary, bnd_end], axis=1) 
    cx_boundary = tf.concat([c_boundary, x_boundary], axis=1)

    cx_eqn = tf.concat([c_eqn, x_eqn], axis=1) 
    
    soln_init = solution_init[1::]
    phi_init = soln_init[0]*np.cos(0.*(cx_init[:, 1:2]))

    ii = 0.
    
    for aii in soln_init[1:]:
        ii = ii+1.
        phi_init = phi_init + aii*np.cos(ii*(cx_init[:, 1:2]))

    phi_init = phi_init-soln_init[0]
    
    plt.plot(cx_init[:,1:2],phi_init,'.')

    
    plt.show()
    y_eqn = tf.zeros((n_samples, 1))


    return (cx_eqn, y_eqn), (cx_init, phi_init), (cx_boundary_start, cx_boundary_end, cx_boundary), (velocities, aS)

def simulate_seq2seqAmplitude(n_samples, x_start, length, thirdAlpha = 1., fifthBeta = 1./4., nonlinSigma = 1, a1 = 1.0e-6, aF = 0.001, conSteps = 1500, compute_IC = False, random_seed = 42, dtype=tf.float32) -> tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]:

    
    a_eqn = tf.random.uniform((n_samples, 1), a1, aF, dtype=dtype, seed=random_seed)
    x_eqn = tf.random.uniform((n_samples, 1), x_start, length, dtype=dtype, seed=random_seed)
    
    a_init = tf.ones((n_samples, 1), dtype=dtype)*a1
    x_init = tf.random.uniform((n_samples, 1), x_start, length, seed=random_seed, dtype=dtype)
    ax_init = tf.concat([a_init, x_init], axis=1)

    a_boundary = tf.random.uniform((n_samples, 1),a1, aF, dtype=dtype, seed=random_seed)
    bnd_start = tf.ones((n_samples, 1), dtype=dtype)*x_start
    bnd_end = tf.ones((n_samples, 1), dtype=dtype)*length
    x_boundary = tf.concat([tf.reshape([x_start]*(n_samples//2),(-1,1)), tf.reshape([length]*(n_samples//2),(-1,1))], axis=0)
    x_boundary = tf.random.shuffle(x_boundary, seed=random_seed)


    ax_boundary_start = tf.concat([a_boundary, bnd_start], axis=1) 
    ax_boundary_end = tf.concat([a_boundary, bnd_end], axis=1) 
    ax_boundary = tf.concat([a_boundary, x_boundary], axis=1)


    ax_eqn = tf.concat([a_eqn, x_eqn], axis=1) 
    y_eqn = tf.zeros((n_samples, 1))
    
    
    aS = np.linspace(1e-6,a1,conSteps) # vector of free parameter a1 (amplitudes)
    velocities = np.zeros(conSteps)
    NN = 21
    uguess = np.zeros(NN+2)
    uguess[0] = thirdAlpha - fifthBeta 
    uguess[2] = 1e-6
    V = uguess[0]
    if compute_IC:
        for k in range(conSteps):
            solution = fsolve(kawaharaCosEqnsPos, uguess, args=(aS[k], thirdAlpha, fifthBeta, nonlinSigma, NN),xtol=1.e-8) 
            soln = solution[1::] # all the As (excludes speed)
            V = solution[0]
            uguess = np.concatenate((V,solution[1],aS[k],solution[3::]),axis=None) #update initial guess
            velocities[k] = solution[0]
            
        phi_init = soln[0]*np.cos(0.*(ax_init[:, 1:2]))
        for i ,ai in enumerate(soln[1:]):
            phi_init = phi_init + ai*np.cos((i+1)*(ax_init[:, 1:2]))

        phi_init = phi_init-soln[0]
        plt.plot(ax_init[:,1:], phi_init,'.')
        plt.show()


        return (ax_eqn, y_eqn), (ax_init, phi_init), (ax_boundary_start, ax_boundary_end, ax_boundary), solution
    else:
        return (ax_eqn, y_eqn), (ax_init), (ax_boundary_start, ax_boundary_end, ax_boundary)
        