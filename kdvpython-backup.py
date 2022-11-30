import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
from modules.data import simulate_kdv2
from modules.models import KdvPinn
from modules.plots import plot_kdv_model, plot_training_loss

k=6
c=5
def phi_function(tx):
    return c / (2*tf.cosh(np.sqrt(c)*(tx[:, 1:2]-c*tx[:,0:1]-2*np.pi)/2)**2)
    

def boundary_function(tx):
    return tf.zeros_like(tx[:, 1:])

x_start = 0
length = 8*np.pi
time = np.pi / 2
#(tx_samples, y_samples), (tx_init, y_init), (tx_boundary,y_boundary) = simulate_kdv(1000, f_init,boundary_function, x_start, length, time)
(tx_eqn, y_eqn), (tx_init, y_init), (tx_boundary, y_boundary) = simulate_kdv2(1000, phi_function, boundary_function, length, time, random_seed = 42, dtype=tf.float32) 
network = KdvPinn.build_network([32, 64])
model = KdvPinn(network,k)
model.compile()


inputs = tf.stack([tx_eqn, tx_init, tx_boundary], axis=0)
outputs = tf.stack([y_init, y_boundary], axis=0)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

history = model.fit(inputs, outputs, 10000, optimizer, progress_interval=200)

plot_kdv_model(model.network, x_start, length, time)