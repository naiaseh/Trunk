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
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
from Modules.models import LOSS_RESIDUAL, LOSS_BOUNDARY, LOSS_INITIAL, MEAN_ABSOLUTE_ERROR

def plot_kdv_model(model,x_start, length, time, save_path = None) -> None:
    """
    Plot the model predictions for the heat equation.
    Args:
        model: A trained HeatPinn model.
        length: The length of the domain.
        time: The time frame of the simulation.
        save_path: The path to save the plot to.
    """
    num_test_samples = 1000
    t_flat = np.linspace(0, time, num_test_samples)
    x_flat = np.linspace(x_start, length, num_test_samples)
    t, x = np.meshgrid(t_flat, x_flat)
    tx = np.stack([t.flatten(), x.flatten()], axis=-1)
    u = model.predict(tx, batch_size=num_test_samples)
    u = u.reshape(t.shape)

    # plot u(t,x) distribution as a color-map
    fig = plt.figure(figsize=(7,4))
    gs = GridSpec(2, 5)
    plt.subplot(gs[0, :])
    plt.pcolormesh(t, x, u)
    plt.xlabel('t')
    plt.ylabel('x')
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.set_label('u(t,x)')
    cbar.mappable.set_clim(-1, 2)
    # plot u(t=const, x) cross-sections
    t_cross_sections = [0, time/4, time/2, 3*time/4, time]
    for i, t_cs in enumerate(t_cross_sections):
        plt.subplot(gs[1, i])
        tx = np.stack([np.full(t_flat.shape, t_cs), x_flat], axis=-1)
        u = model.predict(tx, batch_size=num_test_samples)
        plt.plot(x_flat, u)
        plt.title('t={}'.format(np.round(t_cs)))
        plt.xlabel('x')
        plt.ylabel('u(t,x)')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
    
    
def plot_training_loss(history, x_scale = "linear", y_scale = "linear", save_path=None):
    """
    Plot the training residual, initial, and boundary losses separately.
    Args:
        history: The history object returned by the model.fit() method.
        x_scale: The scale of the x-axis.
        y_scale: The scale of the y-axis.
    """
    plt.figure(figsize=(10, 5), dpi = 150)
    plt.xscale(x_scale)
    plt.yscale(y_scale)
    if len(history[LOSS_INITIAL]) > 0:
        plt.plot(history[LOSS_INITIAL], label='initial loss')
    if len(history[LOSS_BOUNDARY]) > 0:
        plt.plot(history[LOSS_BOUNDARY], label='boundary loss')
    if len(history[LOSS_RESIDUAL]) > 0:
        plt.plot(history[LOSS_RESIDUAL], label='residual loss')
    if len(history[MEAN_ABSOLUTE_ERROR]) > 0:
        plt.plot(history[MEAN_ABSOLUTE_ERROR], label='mean absolute error')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


