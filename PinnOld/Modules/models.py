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

from typing import Dict
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time

LOSS_RESIDUAL = "loss_residual"
LOSS_INITIAL = "loss_initial"
LOSS_BOUNDARY = "loss_boundary"
MEAN_ABSOLUTE_ERROR = "mean_absolute_error"


def _create_history_dict():
    return {
        LOSS_RESIDUAL: [],
        LOSS_INITIAL: [],
        LOSS_BOUNDARY: [],
        MEAN_ABSOLUTE_ERROR: []
    }


def _add_to_history_dict(history_dict, loss_residual = None, loss_initial = None, loss_boundary = None, mean_absolute_error = None):
  if loss_residual is not None:
    history_dict[LOSS_RESIDUAL].append(loss_residual)
  if loss_initial is not None:
    history_dict[LOSS_INITIAL].append(loss_initial)
  if loss_boundary is not None:
    history_dict[LOSS_BOUNDARY].append(loss_boundary)
  if mean_absolute_error is not None:
    history_dict[MEAN_ABSOLUTE_ERROR].append(mean_absolute_error)

class KdvPinn(tf.keras.Model):
  """
  Keras PINN model for the Heat PDE.
  Attributes:
    network (keras.Model): The neural network used to approximate the solution.
    k (float): The thermal conductivity of the material.
  """

  def __init__(self, network: "tf.keras.Model", k: float = 6.0) -> None:
    """
    Args:
      network: A keras model representing the backbone neural network.
      k: thermal conductivity. Default is 1.
    """
    super().__init__()
    self.network = network
    self.k = k



  def fit(self, inputs, labels, epochs, optimizer, u_exact=None, progress_interval=500) -> dict[str, list[float]]:
    """
    Train the model with the given inputs and optimizer.

    Args:
      inputs: A list of tensors, where the first tensor is the equation data,
        the second tensor is the initial condition data, and the fourth tensor is the
        boundary condition data.
      labels: A list of tensors, where the first tensor is the phi initial condition labels,
        the second tensor is the boundary condition labels.
      epochs: The number of epochs to train for.
      optimizer : The optimizer to use for training.
      progress_interval: The number of epochs between each progress report.
    Returns:
      A dictionary containing the loss history for each of the three loss terms.
    """
    history = _create_history_dict()

    start_time = time.time()
    for epoch in range(epochs):
      with tf.GradientTape() as tape:
        u, residual, u_init, u_bndry = self.call(inputs) #question residual, not u right?

        loss_residual = tf.reduce_mean(tf.square(residual))
        loss_init = tf.reduce_mean(tf.square(u_init - labels[0]))
        loss_boundary = tf.reduce_mean(tf.square(u_bndry - labels[1]))
        loss = loss_residual + loss_init + loss_boundary

      grads = tape.gradient(loss, self.trainable_weights)
      optimizer.apply_gradients(zip(grads, self.trainable_weights))

      if u_exact is not None:
        abs_error = tf.reduce_mean(tf.abs(u - u_exact))
      else:
        abs_error = None

      _add_to_history_dict(history, loss_residual, loss_init, loss_boundary, abs_error)
      
      if epoch % progress_interval == 0:
        print(f"Epoch: {epoch} Loss: {loss.numpy():.4f} Total Elapsed Time: {time.time() - start_time:.2f}")
    
    return history

  
  @tf.function
  def input_gradient(self, tx):
    """
    Compute the first order derivative w.r.t. time and second order derivative w.r.t. space of the network output.

    Args:
      tx: input tensor of shape (n_inputs, 2)

    Returns:
      u_t: first derivative of u with respect to t
      u_xx: second derivative of u with respect to x
    """
    with tf.GradientTape() as ggg:
        ggg.watch(tx)
        with tf.GradientTape() as gg:
            gg.watch(tx)
            with tf.GradientTape() as g:
                g.watch(tx)
                u = self.network(tx)

            first_order = g.batch_jacobian(u, tx)
            du_dt = first_order[..., 0]
            du_dx = first_order[..., 1]

        d2u_dx2 = gg.batch_jacobian(du_dx, tx)[..., 1]
    d3u_dx3 = ggg.batch_jacobian(d2u_dx2, tx)[..., 1]

    return u, du_dt, du_dx, d3u_dx3
    

  
  def call(self, inputs):
    """
    Performs forward pass of the model, computing the PDE residual and the initial and boundary conditions.

    Args:
      inputs: A list of tensors, where the first tensor is the equation data,
        the second tensor is the initial condition data, and the third tensor is the
        boundary condition data.

    Returns:
        pde_residual: The PDE residual of shape (n_inputs, 1)
        u_init: The initial condition output of shape (n_inputs, 1)
        u_bndry: The boundary condition output of shape (n_inputs, 1)

    """

    tx_equation = inputs[0]
    tx_init = inputs[1]
    tx_bound = inputs[2]

    u, du_dt, du_dx, d3u_dx3 = self.input_gradient(tx_equation)

    
    # Calculate PDE residual
    pde_residual = du_dt + d3u_dx3 + du_dx * ((self.k) * u ) # add -self.c in bracket to move into the frame 

    n_i = tf.shape(tx_init)[0]
    tx_ib = tf.concat([tx_init, tx_bound], axis=0)
    u_ib = self.network(tx_ib)
    u_init = u_ib[:n_i]
    u_bndry = u_ib[n_i:]

    return u, pde_residual, u_init, u_bndry

  @staticmethod
  def build_network(layers, n_inputs=2, n_outputs=1, activation=tf.keras.activations.tanh, initialization=tf.keras.initializers.glorot_normal):
    """
    Builds a fully connected neural network with the specified number of layers and nodes per layer.

    Args:
        layers (list): List of integers specifying the number of nodes in each layer.
        n_inputs (int): Number of inputs to the network.
        n_outputs (int): Number of outputs from the network.
        activation (function): Activation function to use in each layer.
        initialization (function): Initialization function to use in each layer.
    returns:
        keras.Model: A keras model representing the neural network.
    """
    inputs = tf.keras.layers.Input((n_inputs))
    x = inputs
    for i in layers:
      x = tf.keras.layers.Dense(i, activation = activation, kernel_initializer=initialization)(x)
    
    outputs = tf.keras.layers.Dense(n_outputs, kernel_initializer=initialization)(x)
    return tf.keras.Model(inputs=[inputs], outputs = [outputs])


class travelKawaharaPinn(tf.keras.Model):
  """
  Keras PINN model for the Heat PDE.
  Attributes:
    network (keras.Model): The neural network used to approximate the solution.

  """

  def __init__(self, network: "tf.keras.Model", alpha: float = 1.0,beta: float = 1/4 ,sigma: float = 2.0) -> None:
    """
    Args:
      network: A keras model representing the backbone neural network.
      alpha, beta, sigma: PDE params
    """
    super().__init__()
    self.network = network
    self.sigma = sigma
    self.alpha = alpha
    self.beta = beta


  def fit(self, inputs, labels, epochs, optimizer, u_exact=None, progress_interval=500) -> dict[str, list[float]]:
    """
    Train the model with the given inputs and optimizer.

    Args:
      inputs: A list of tensors, where the first tensor is the equation data,
        the second tensor is the initial condition data, and the third tensor is the
        boundary condition data.
      labels: A list of tensors, where the first tensor is the phi initial condition labels,
        the second tensor is the boundary condition labels.
      epochs: The number of epochs to train for.
      optimizer : The optimizer to use for training.
      progress_interval: The number of epochs between each progress report.
    Returns:
      A dictionary containing the loss history for each of the three loss terms.
    """
    history = _create_history_dict()

    start_time = time.time()
    for epoch in range(epochs):
      with tf.GradientTape() as tape:
        u, residual, u_init, u_bndry = self.call(inputs) 

        loss_residual = tf.reduce_mean(tf.square(residual))
        loss_init = tf.reduce_mean(tf.square(u_init - labels[0]))
        loss_boundary = tf.reduce_mean(tf.square(u_bndry - labels[1]))
        loss = loss_residual + loss_init + loss_boundary 
        

      grads = tape.gradient(loss, self.trainable_weights)
      optimizer.apply_gradients(zip(grads, self.trainable_weights))

      if u_exact is not None:
        abs_error = tf.reduce_mean(tf.abs(u - u_exact))
      else:
        abs_error = None

      _add_to_history_dict(history, loss_residual, loss_init, loss_boundary, abs_error)
      
      if epoch % progress_interval == 0:
        print(f"Epoch: {epoch} Loss: {loss.numpy():.4f} Total Elapsed Time: {time.time() - start_time:.2f}")
    
    return history

  
  @tf.function
  def input_gradient(self, tx):
    """
    Compute derivatives

    Args:
      tx: input tensor of shape (n_inputs, 2)

    Returns:
      u_t, u_3x, u_5x, u
    """
    with tf.GradientTape() as g5:
      g5.watch(tx)
      with tf.GradientTape() as g4:
        g4.watch(tx)
        with tf.GradientTape() as ggg:
            ggg.watch(tx)
            with tf.GradientTape() as gg:
                gg.watch(tx)
                with tf.GradientTape() as g:
                    g.watch(tx)
                    u = self.network(tx)

                first_order = g.batch_jacobian(u, tx)
                du_dt = first_order[..., 0]
                du_dx = first_order[..., 1]

            d2u_dx2 = gg.batch_jacobian(du_dx, tx)[..., 1]
        d3u_dx3 = ggg.batch_jacobian(d2u_dx2, tx)[..., 1]
      d4u_dx4 = g4.batch_jacobian(d3u_dx3, tx)[..., 1]
    d5u_dx5 = g5.batch_jacobian(d4u_dx4, tx)[..., 1]

    return u, du_dt, du_dx, d3u_dx3, d5u_dx5
  
  def call(self, inputs):
    """
    Performs forward pass of the model, computing the PDE residual and the initial and boundary conditions.

    Args:
      inputs: A list of tensors, where the first tensor is the equation data,
        the second tensor is the initial condition data, and the third tensor is the
        boundary condition data.

    Returns:
        pde_residual: The PDE residual of shape (n_inputs, 1)
        u_init: The initial condition output of shape (n_inputs, 1)
        u_bndry: The boundary condition output of shape (n_inputs, 1)

    """

    tx_equation = inputs[0]
    tx_init = inputs[1]
    tx_bound = inputs[2]

    u, du_dt, du_dx, d3u_dx3, d5u_dx5 = self.input_gradient(tx_equation)



    
    # Calculate PDE residual
    pde_residual = + self.alpha * d3u_dx3 + (self.beta) * d5u_dx5 + (self.sigma * u + 0.7500009999995948)* du_dx
    # set du_dt - du_dt to zero if it doesn't work, set beta to 0

    n_i = tf.shape(tx_init)[0]
    tx_ib = tf.concat([tx_init, tx_bound], axis=0)
    u_ib = self.network(tx_ib)
    u_init = u_ib[:n_i]
    u_bndry = u_ib[n_i:]

    return u, pde_residual, u_init, u_bndry

  @staticmethod
  def build_network(layers, n_inputs=2, n_outputs=1, activation=tf.keras.activations.tanh, initialization=tf.keras.initializers.glorot_normal):
    """
    Builds a fully connected neural network with the specified number of layers and nodes per layer.

    Args:
        layers (list): List of integers specifying the number of nodes in each layer.
        n_inputs (int): Number of inputs to the network.
        n_outputs (int): Number of outputs from the network.
        activation (function): Activation function to use in each layer.
        initialization (function): Initialization function to use in each layer.
    returns:
        keras.Model: A keras model representing the neural network.
    """
    inputs = tf.keras.layers.Input((n_inputs))
    x = inputs
    for i in layers:
      x = tf.keras.layers.Dense(i, activation = activation, kernel_initializer=initialization)(x)
    
    outputs = tf.keras.layers.Dense(n_outputs, kernel_initializer=initialization)(x)
    return tf.keras.Model(inputs=[inputs], outputs = [outputs])

