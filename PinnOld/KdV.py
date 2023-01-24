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

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
from Modules.data import simulate_kdv
from Modules.models import KdvPinn
from Modules.plots import plot_kdv_model, plot_training_loss

k=6
c=5
phi = 1
def phi_function(tx):
    return c / (2*tf.cosh(np.sqrt(c)*(tx[:, 1:2]-c*tx[:,0:1]-2*np.pi)/2)**2)+phi
    

def boundary_function(tx):
    return phi*tf.ones_like(tx[:, 1:])

x_start = 0
length = 8*np.pi
time = np.pi / 2
#(tx_samples, y_samples), (tx_init, y_init), (tx_boundary,y_boundary) = simulate_kdv(1000, f_init,boundary_function, x_start, length, time)
(tx_eqn, y_eqn), (tx_init, y_init), (tx_boundary, y_boundary) = simulate_kdv(1000, phi_function, boundary_function, length, time, random_seed = 42, dtype=tf.float32) 
network = KdvPinn.build_network([32, 64])
model = KdvPinn(network,k)
model.compile()


inputs = tf.stack([tx_eqn, tx_init, tx_boundary], axis=0)
outputs = tf.stack([y_init, y_boundary], axis=0)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

history = model.fit(inputs, outputs, 10000, optimizer, progress_interval=200)

plot_kdv_model(model.network, x_start, length, time)

plot_training_loss(history, y_scale='log')