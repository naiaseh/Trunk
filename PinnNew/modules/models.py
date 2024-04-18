'''
This file contains the PINN models for the Advection, Burgers, Schrodinger, Poisson, Heat, and Wave equations.
'''

from typing import Tuple, List, Union, Callable
import tensorflow as tf
import numpy as np
from keras import backend as K
import sys
from sklearn.linear_model import LinearRegression

LOSS_TOTAL = "loss_total"
LOSS_BOUNDARY = "loss_boundary"
LOSS_INITIAL = "loss_initial"
LOSS_RESIDUAL = "loss_residual"
MEAN_ABSOLUTE_ERROR = "mean_absolute_error"
LOSS_DUDT = "loss_dudt"
LOSS_HAMIL = "loss_hamil"
LOSS_BOUNDARY_Y = "loss_boundary_y"
LOSS_RESIDUAL1 = "loss_residual1"
LOSS_RESIDUAL2 = "loss_residual2"
LOSS_INITIAL_U = "loss_initial_u"
LOSS_INITIAL_ETA = "loss_initial_eta"
LOSS_BOUNDARY_U = "loss_boundary_u"
LOSS_BOUNDARY_ETA = "loss_boundary_eta"


def create_history_dictionary() -> dict:
    """
    Creates a history dictionary.

    Returns:
        The history dictionary.
    """
    return {
        LOSS_TOTAL: [],
        LOSS_BOUNDARY: [],
        LOSS_INITIAL: [],
        LOSS_RESIDUAL: [],
        MEAN_ABSOLUTE_ERROR: []
    }
def create_history_dictionary_KP() -> dict:
    """
    Creates a history dictionary.

    Returns:
        The history dictionary.
    """
    return {
        LOSS_TOTAL: [],
        LOSS_BOUNDARY: [],
        LOSS_BOUNDARY_Y: [],
        LOSS_INITIAL: [],
        LOSS_RESIDUAL: [],
        MEAN_ABSOLUTE_ERROR: []
    }
def create_history_dictionary_Fourier() -> dict:
    """
    Creates a history dictionary.

    Returns:
        The history dictionary.
    """
    return {
        LOSS_TOTAL: [],
        LOSS_INITIAL: [],
        LOSS_RESIDUAL: [],
        MEAN_ABSOLUTE_ERROR: []
    }
    
def create_history_dictionary_FourierNoC() -> dict:
    """
    Creates a history dictionary.

    Returns:
        The history dictionary.
    """
    return {
        LOSS_TOTAL: [],
        LOSS_RESIDUAL: [],
        MEAN_ABSOLUTE_ERROR: []
    }

def create_history_dictionary_Kawahara_custom() -> dict:
    """
    Creates a history dictionary.

    Returns:
        The history dictionary.
    """
    return {
        LOSS_TOTAL: [],
        LOSS_BOUNDARY: [],
        LOSS_INITIAL: [],
        LOSS_RESIDUAL: [],
        LOSS_HAMIL: [],
        MEAN_ABSOLUTE_ERROR: []
    }

def create_history_dictionary_BloodFlow() -> dict:
    """
    Creates a history dictionary.

    Returns:
        The history dictionary.
    """
    return {
        LOSS_TOTAL: [],
        LOSS_BOUNDARY_U: [],
        LOSS_BOUNDARY_ETA: [],
        LOSS_INITIAL_U: [],
        LOSS_INITIAL_ETA: [],
        LOSS_RESIDUAL1: [],
        LOSS_RESIDUAL2: [],
    }



def create_dense_model(layers: List[Union[int, "tf.keras.layers.Layer"]], activation: "tf.keras.activations.Activation", \
    initializer: "tf.keras.initializers.Initializer", n_inputs: int, n_outputs: int, **kwargs) -> "tf.keras.Model":
    """
    Creates a dense model with the given layers, activation, and input and output sizes.

    Args:
        layers: The layers to use. Elements can be either an integer or a Layer instance. If an integer, a Dense layer with that many units will be used.
        activation: The activation function to use.
        initializer: The initializer to use.
        n_inputs: The number of inputs.
        n_outputs: The number of outputs.
        **kwargs: Additional arguments to pass to the Model constructor.

    Returns:
        The dense model.
    """
    inputs = tf.keras.Input(shape=(n_inputs,))
    x = inputs
    for layer in layers:
        if isinstance(layer, int):
            x = tf.keras.layers.Dense(layer, activation=activation, kernel_initializer=initializer)(x)
        else:
            x = layer(x)
    outputs = tf.keras.layers.Dense(n_outputs, kernel_initializer=initializer)(x) #notice the only difference is no activation
    return tf.keras.Model(inputs=inputs, outputs=outputs, **kwargs)

def create_dense_model_Normalized(layers: List[Union[int, "tf.keras.layers.Layer"]], activation: "tf.keras.activations.Activation", \
    initializer: "tf.keras.initializers.Initializer", n_inputs: int, n_outputs: int, normalizer, **kwargs) -> "tf.keras.Model":

    
    inputs = tf.keras.Input(shape=(n_inputs,))
    x = normalizer(inputs)
    for layer in layers:
        if isinstance(layer, int):
            x = tf.keras.layers.Dense(layer, activation=activation, kernel_initializer=initializer)(x)
        else:
            x = layer(x)
    outputs = tf.keras.layers.Dense(n_outputs, kernel_initializer=initializer)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, **kwargs)


class AdvectionPinn(tf.keras.Model):
    """
    A PINN for the advection equation.

    Attributes:
        backbone: The backbone model.
        v: The velocity of the advection.
        k: The diffusion coefficient.
        loss_boundary_tracker: The boundary loss tracker.
        loss_residual_tracker: The residual loss tracker.
        mae_tracker: The mean absolute error tracker.
        loss_boundary_weight: The weight of the boundary loss.
        loss_residual_weight: The weight of the residual loss.

    """

    def __init__(self, backbone, v: float, k: float, loss_residual_weight: float = 1.0, loss_boundary_weight: float = 1.0, **kwargs):
        """
        Initializes the model.

        Args:
            backbone: The backbone model.
            v: The velocity of the advection.
            k: The diffusion coefficient.
            loss_residual_weight: The weight of the residual loss.
            loss_boundary_weight: The weight of the boundary loss.
            **kwargs: Additional arguments to pass to the Model constructor.
        """
        super().__init__(**kwargs)
        self.backbone = backbone
        self.v = v
        self.k = k
        self.loss_boundary_tracker = tf.keras.metrics.Mean(name=LOSS_BOUNDARY)
        self.loss_residual_tracker = tf.keras.metrics.Mean(name=LOSS_RESIDUAL)
        self.loss_total_tracker = tf.keras.metrics.Mean(name=LOSS_TOTAL)
        self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name=MEAN_ABSOLUTE_ERROR)
        self._loss_residual_weight = tf.Variable(loss_residual_weight, trainable=False, name="loss_residual_weight", dtype=tf.keras.backend.floatx())
        self._loss_boundary_weight = tf.Variable(loss_boundary_weight, trainable=False, name="loss_boundary_weight", dtype=tf.keras.backend.floatx())
        self.res_loss = tf.keras.losses.MeanSquaredError()
        self.bnd_loss = tf.keras.losses.MeanSquaredError()

    def set_loss_weights(self, loss_residual_weight: float, loss_boundary_weight: float):
        """
        Sets the loss weights.

        Args:
            loss_residual_weight: The weight of the residual loss.
            loss_boundary_weight: The weight of the boundary loss.
        """
        self._loss_residual_weight.assign(loss_residual_weight)
        self._loss_boundary_weight.assign(loss_boundary_weight)

    @tf.function
    def call(self, inputs: "tf.Tensor", training: bool = False) -> "tf.Tensor":
        """
        Calls the model on the inputs.

        Args:
            inputs: The inputs to the model. First input is the samples, second input is the boundary.
            training: Whether or not the model is being called in training mode.

        Returns:
            The output of the model.
        """

        #compute the derivatives
        inputs_residuals = inputs[0]
        inputs_bnd = inputs[1]

        with tf.GradientTape(watch_accessed_variables=False) as tape2:
            tape2.watch(inputs_residuals)
            with tf.GradientTape(watch_accessed_variables=False) as tape1:
                tape1.watch(inputs_residuals)
                u_samples = self.backbone(inputs_residuals, training=training)
            u_x = tape1.gradient(u_samples, inputs_residuals)
        u_xx = tape2.gradient(u_x, inputs_residuals)

        #compute the lhs
        lhs_samples = self.v * u_x - self.k * u_xx

        #compute the boundary
        u_bnd = self.backbone(inputs_bnd, training=training)

        return u_samples, lhs_samples, u_bnd
    
    def fit_custom(self, inputs: List['tf.Tensor'], outputs: List['tf.Tensor'], epochs: int, print_every: int = 1000):
        '''
        Custom alternative to tensorflow fit function, mainly to allow inputs with different sizes. Training is done in full batches.

        Args:
            data: The data to train on. Should be a list of tensors: [tx_colloc, tx_bnd]
            outputs: The outputs to train on. Should be a list of tensors: [u_colloc, residual, u_bnd]. u_colloc is only used for the MAE metric.
            epochs: The number of epochs to train for.
            print_every: How often to print the metrics. Defaults to 1000.
        '''
        history = create_history_dictionary()
        
        for epoch in range(epochs):
            metrs = self.train_step([inputs, outputs])
            for key, value in metrs.items():
                history[key].append(value.numpy())

            if epoch % print_every == 0:
                tf.print(f"Epoch {epoch}, Loss Residual: {metrs['loss_residual']:0.4f}, Loss Boundary: {metrs['loss_boundary']:0.4f}, MAE: {metrs['mean_absolute_error']:0.4f}")
                
            #reset metrics
            for m in self.metrics:
                m.reset_states()

        return history

    def train_step(self, data: Tuple["tf.Tensor", "tf.Tensor"]) -> "tf.Tensor":
        """
        Performs a training step on the given data.

        Args:
            data: The data to train on. First data is the inputs,second data is the outputs.\
                In inputs, first tensor is the residual samples, second tensor is the boundary samples.\
                    In outputs, first tensor is the exact u for the residual samples, second tensor is the \
                        exact rhs for the residual samples, and third tensor is the exact u for the boundary samples.

        Returns:
            The loss.
        """
        x, y = data

        # compute residual loss with samples
        with tf.GradientTape() as tape:
            u_samples, lhs_samples, u_bnd = self(x, training=True)
            loss_residual = self.res_loss(y[1], lhs_samples)
            loss_boundary = self.bnd_loss(y[2], u_bnd)
            loss = self._loss_residual_weight * loss_residual + self._loss_boundary_weight * loss_boundary

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.loss_total_tracker.update_state(loss)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_boundary_tracker.update_state(loss_boundary)
        self.mae_tracker.update_state(y[0], u_samples)


        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        '''
        Performs a test step on the given data.
        '''
        x, y = data

        u_samples, lhs_samples, u_bnd = self(x, training=False)
        loss_residual = self.res_loss(y[1], lhs_samples)
        loss_boundary = self.bnd_loss(y[2], u_bnd)
        loss = self._loss_residual_weight * loss_residual + self._loss_boundary_weight * loss_boundary

        self.loss_total_tracker.update_state(loss)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_boundary_tracker.update_state(loss_boundary)
        self.mae_tracker.update_state(y[0], u_samples)

        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        '''
        Returns the metrics of the model.
        '''
        return [self.loss_total_tracker, self.loss_residual_tracker, self.loss_boundary_tracker, self.mae_tracker]


class PoissonPinn(tf.keras.Model):
    """
    A PINN for the Poisson's equation.
    
    Attributes:
        backbone: The backbone model.
        loss_boundary_tracker: The boundary loss tracker.
        loss_residual_tracker: The residual loss tracker.
        mae_tracker: The mean absolute error tracker.
        _loss_residual_weight: The weight of the residual loss.
        _loss_boundary_weight: The weight of the boundary loss.
    """

    def __init__(self, backbone, loss_residual_weight: float = 1.0, loss_boundary_weight: float = 1.0, **kwargs):
        """
        Initializes the model.

        Args:
            backbone: The backbone model.
            loss_residual_weight: The weight of the residual loss.
            loss_boundary_weight: The weight of the boundary loss.
            **kwargs: Additional arguments to pass to the Model constructor.
        """
        super().__init__(**kwargs)
        self.backbone = backbone
        self.loss_total_tracker = tf.keras.metrics.Mean(name=LOSS_TOTAL)
        self.loss_boundary_tracker = tf.keras.metrics.Mean(name=LOSS_BOUNDARY)
        self.loss_residual_tracker = tf.keras.metrics.Mean(name=LOSS_RESIDUAL)
        self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name=MEAN_ABSOLUTE_ERROR)
        self._loss_residual_weight = tf.Variable(loss_residual_weight, trainable=False, name="loss_residual_weight", dtype=tf.keras.backend.floatx())
        self._loss_boundary_weight = tf.Variable(loss_boundary_weight, trainable=False, name="loss_boundary_weight", dtype=tf.keras.backend.floatx())
        self.res_loss = tf.keras.losses.MeanSquaredError()
        self.bnd_loss = tf.keras.losses.MeanSquaredError()

    def set_loss_weights(self, loss_residual_weight: float, loss_boundary_weight: float):
        """
        Sets the loss weights.

        Args:
            loss_residual_weight: The weight of the residual loss.
            loss_boundary_weight: The weight of the boundary loss.
        """
        self._loss_residual_weight.assign(loss_residual_weight)
        self._loss_boundary_weight.assign(loss_boundary_weight)

    @tf.function
    def call(self, inputs, training=False):
        """
        Calls the model on the inputs.

        Args:
            inputs: The inputs to the model. First input is the samples, second input is the boundary.
            training: Whether or not the model is being called in training mode.

        Returns:
            The output of the model.
        """

        #compute the derivatives
        inputs_residuals = inputs[0]
        inputs_bnd = inputs[1]

        with tf.GradientTape(watch_accessed_variables=False) as tape2:
            tape2.watch(inputs_residuals)
            with tf.GradientTape(watch_accessed_variables=False) as tape1:
                tape1.watch(inputs_residuals)
                u_samples = self.backbone(inputs_residuals, training=training)
            u_x = tape1.batch_jacobian(u_samples, inputs_residuals)[:, :, 0]
        u_xx = tape2.batch_jacobian(u_x, inputs_residuals)[:, :, 0]

        #compute the lhs
        lhs_samples = u_xx

        #compute the boundary
        u_bnd = self.backbone(inputs_bnd, training=training)

        return u_samples, lhs_samples, u_bnd

    def train_step(self, data):
        """
        Performs a training step on the given data.

        Args:
            data: The data to train on. First data is the inputs,second data is the outputs.\
                In inputs, first tensor is the residual samples, second tensor is the boundary samples.\
                    In outputs, first tensor is the exact u for the residual samples, second tensor is the \
                        exact rhs for the residual samples, and third tensor is the exact u for the boundary samples.

        Returns:
            The loss.
        """
        inputs, outputs = data
        u_exact, rhs_exact, u_bnd_exact = outputs

        # compute residual loss with samples
        with tf.GradientTape() as tape:
            u_samples, lhs_samples, u_bnd = self(inputs, training=True)
            # loss_residual = tf.losses.mean_squared_error(rhs_exact, lhs_samples)
            loss_residual = self.res_loss(rhs_exact, lhs_samples)
            loss_boundary = self.bnd_loss(u_bnd_exact, u_bnd)
            loss = self._loss_residual_weight * loss_residual + self._loss_boundary_weight * loss_boundary

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.loss_total_tracker.update_state(loss)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_boundary_tracker.update_state(loss_boundary)
        self.mae_tracker.update_state(u_exact, u_samples)

        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        """
        Performs a test step on the given data.
        """

        inputs, outputs = data
        u_exact, rhs_exact, u_bnd_exact = outputs

        # compute residual loss with samples
        u_samples, lhs_samples, u_bnd = self(inputs, training=False)
        loss_residual = self.res_loss(rhs_exact, lhs_samples)
        loss_boundary = self.bnd_loss(u_bnd_exact, u_bnd)
        loss = self._loss_residual_weight * loss_residual + self._loss_boundary_weight * loss_boundary

        self.loss_total_tracker.update_state(loss)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_boundary_tracker.update_state(loss_boundary)
        self.mae_tracker.update_state(u_exact, u_samples)

        return {m.name: m.result() for m in self.metrics}
    
    def fit_custom(self, inputs: List['tf.Tensor'], outputs: List['tf.Tensor'], epochs: int, print_every: int = 1000):
        '''
        Custom alternative to tensorflow fit function, mainly to allow inputs with different sizes. Training is done in full batches.

        Args:
            data: The data to train on. Should be a list of tensors: [tx_colloc, tx_bnd]
            outputs: The outputs to train on. Should be a list of tensors: [u_colloc, residual, u_bnd]. u_colloc is only used for the MAE metric.
            epochs: The number of epochs to train for.
            print_every: How often to print the metrics. Defaults to 1000.
        '''
        history = create_history_dictionary()
        
        for epoch in range(epochs):
            metrs = self.train_step([inputs, outputs])
            for key, value in metrs.items():
                history[key].append(value.numpy())

            if epoch % print_every == 0:
                tf.print(f"Epoch {epoch}, Loss Residual: {metrs['loss_residual']:0.4f}, Loss Boundary: {metrs['loss_boundary']:0.4f}, MAE: {metrs['mean_absolute_error']:0.4f}")
                
            #reset metrics
            for m in self.metrics:
                m.reset_states()

        return history

    @property
    def metrics(self):
        '''
        Returns the metrics of the model.
        '''
        return [self.loss_total_tracker, self.loss_residual_tracker, self.loss_boundary_tracker, self.mae_tracker]

class SchrodingerPinn(tf.keras.Model):
    """
    A PINN for the Schrodinger's equation.
    
    Attributes:
        backbone: The backbone model.
        k: The planck constant.
        loss_boundary_tracker: The boundary loss tracker.
        loss_initial_tracker: The initial loss tracker.
        loss_residual_tracker: The residual loss tracker.
        mae_tracker: The mean absolute error tracker.
        _loss_residual_weight: The weight of the residual loss.
        _loss_initial_weight: The weight of the initial loss.
        _loss_boundary_weight: The weight of the boundary loss.
    """

    def __init__(self, backbone, k: float = 0.5, loss_residual_weight: float = 1.0, loss_initial_weight: float = 1.0, \
        loss_boundary_weight: float = 1.0, **kwargs):
        """
        Initializes the model.

        Args:
            backbone: The backbone model.
            k: The planck constant.
            loss_residual_weight: The weight of the residual loss.
            loss_initial_weight: The weight of the initial loss.
            loss_boundary_weight: The weight of the boundary loss.
            **kwargs: Additional arguments to pass to the Model constructor.
        """
        super().__init__(**kwargs)
        self.backbone = backbone
        self.k = k
        self.loss_total_tracker = tf.keras.metrics.Mean(name=LOSS_TOTAL)
        self.loss_boundary_tracker = tf.keras.metrics.Mean(name=LOSS_BOUNDARY)
        self.loss_initial_tracker = tf.keras.metrics.Mean(name=LOSS_INITIAL)
        self.loss_residual_tracker = tf.keras.metrics.Mean(name=LOSS_RESIDUAL)
        self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name=MEAN_ABSOLUTE_ERROR)
        self._loss_residual_weight = tf.Variable(loss_residual_weight, trainable=False, name="loss_residual_weight", dtype=tf.keras.backend.floatx())
        self._loss_boundary_weight = tf.Variable(loss_boundary_weight, trainable=False, name="loss_boundary_weight", dtype=tf.keras.backend.floatx())
        self._loss_initial_weight = tf.Variable(loss_initial_weight, trainable=False, name="loss_initial_weight", dtype=tf.keras.backend.floatx())
        self.res_loss = tf.keras.losses.MeanSquaredError()
        self.bnd_loss = tf.keras.losses.MeanSquaredError()
        self.init_loss = tf.keras.losses.MeanSquaredError()

    def set_loss_weights(self, loss_residual_weight: float, loss_initial_weight: float, loss_boundary_weight: float):
        """
        Sets the loss weights.

        Args:
            loss_residual_weight: The weight of the residual loss.
            loss_initial_weight: The weight of the initial loss.
            loss_boundary_weight: The weight of the boundary loss.
        """
        self._loss_residual_weight.assign(loss_residual_weight)
        self._loss_boundary_weight.assign(loss_boundary_weight)
        self._loss_initial_weight.assign(loss_initial_weight)

    @tf.function
    def call(self, inputs, training=False):
        """
        Calls the model on the inputs.

        Args:
            inputs: The inputs to the model. First input is the samples, second input is the initial, \
                third input is the boundary start, fourth input is the boundary end.
            training: Whether or not the model is being called in training mode.

        Returns:
            The output of the model. First output is the solution for residual samples, second is the lhs residual, \
                third is solution for initial samples, fourth is solution for boundary samples, and fifth is dh/dx for boundary samples.
        """
        inputs_residuals = inputs[0]
        inputs_initial = inputs[1]
        inputs_bnd_start = inputs[2]
        inputs_bnd_end = inputs[3]

        with tf.GradientTape(watch_accessed_variables=False) as tape2:
            tape2.watch(inputs_residuals)
            with tf.GradientTape(watch_accessed_variables=False) as tape1:
                tape1.watch(inputs_residuals)
                h_samples = self.backbone(inputs_residuals, training=training)

            first_order = tape1.batch_jacobian(h_samples, inputs_residuals) # output is n_sample x 2 * 2
            dh_dt = first_order[:, :, 0]
            dh_dx = first_order[:, :, 1]
        d2h_dx2 = tape2.batch_jacobian(dh_dx, inputs_residuals)[:, :, 1]

        norm2_h = h_samples[:, 0:1] ** 2 + h_samples[:, 1:2] ** 2
        real_residual = -dh_dt[:, 1:2] + self.k * d2h_dx2[:, 0:1] + norm2_h * h_samples[:, 0:1]
        imag_residual = dh_dt[:, 0:1] + self.k * d2h_dx2[:, 1:2] + norm2_h * h_samples[:, 1:2]

        lhs_samples = tf.concat([real_residual, imag_residual], axis=1)

        h_initial = self.backbone(inputs_initial, training=training)

        inputs_bnd = tf.concat([inputs_bnd_start, inputs_bnd_end], axis=0)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(inputs_bnd)
            h_bnd = self.backbone(inputs_bnd, training=training)
        dh_dx_bnd = tape.batch_jacobian(h_bnd, inputs_bnd)[:, :, 1]

        h_bnd_start = h_bnd[0:tf.shape(inputs_bnd_start)[0]]
        h_bnd_end = h_bnd[tf.shape(inputs_bnd_start)[0]:]
        dh_dx_start = dh_dx_bnd[0:tf.shape(inputs_bnd_start)[0]]
        dh_dx_end = dh_dx_bnd[tf.shape(inputs_bnd_start)[0]:]

        return h_samples, lhs_samples, h_initial, h_bnd_start, h_bnd_end, dh_dx_start, dh_dx_end

    @tf.function
    def train_step(self, data):
        """
        Performs a training step.

        Args:
            data: The data to train on. First input is the samples, second input is the initial, third input is the boundary. \
                First output is the solution for residual samples, second is the lhs residual, third is solution for initial samples. \
                    The boundary samples must have 3 columns, where the first column is the t value, the second column is the x value for the start \
                        of the boundary, and the third column is the x value for the end of the boundary.

        Returns:
            The loss of the step.
        """
        inputs, outputs = data
        tx_samples, tx_initial, txx_bnd = inputs
        tx_bnd_start = tf.concat([txx_bnd[:, 0:1], txx_bnd[:, 1:2]], axis=1)
        tx_bnd_end = tf.concat([txx_bnd[:, 0:1], txx_bnd[:, 2:3]], axis=1)
        h_samples_exact, rhs_samples_exact, h_initial_exact = outputs

        with tf.GradientTape() as tape:
            h_samples, lhs_samples, h_initial, h_bnd_start, h_bnd_end, dh_dx_start, dh_dx_end = \
                self([tx_samples, tx_initial, tx_bnd_start, tx_bnd_end], training=True)
            loss_residual = self.res_loss(rhs_samples_exact, lhs_samples)
            loss_initial = self.init_loss(h_initial_exact, h_initial)
            loss_boundary_h = self.bnd_loss(h_bnd_start, h_bnd_end)
            loss_boundary_dh_dx = self.bnd_loss(dh_dx_start, dh_dx_end)
            loss_boundary = loss_boundary_h + loss_boundary_dh_dx
            loss = self._loss_residual_weight * loss_residual + self._loss_initial_weight * loss_initial + \
                self._loss_boundary_weight * loss_boundary

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_total_tracker.update_state(loss)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_initial_tracker.update_state(loss_initial)
        self.loss_boundary_tracker.update_state(loss_boundary)
        self.mae_tracker.update_state(h_samples_exact, h_samples)

        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):

        inputs, outputs = data
        tx_samples, tx_initial, txx_bnd = inputs
        tx_bnd_start = tf.concat([txx_bnd[:, 0:1], txx_bnd[:, 1:2]], axis=1)
        tx_bnd_end = tf.concat([txx_bnd[:, 0:1], txx_bnd[:, 2:3]], axis=1)
        h_samples_exact, rhs_samples_exact, h_initial_exact = outputs

        h_samples, lhs_samples, h_initial, h_bnd_start, h_bnd_end, dh_dx_start, dh_dx_end = \
            self([tx_samples, tx_initial, tx_bnd_start, tx_bnd_end], training=False)
        loss_residual = self.res_loss(rhs_samples_exact, lhs_samples)
        loss_initial = self.init_loss(h_initial_exact, h_initial)
        loss_boundary_h = self.bnd_loss(h_bnd_start, h_bnd_end)
        loss_boundary_dh_dx = self.bnd_loss(dh_dx_start, dh_dx_end)
        loss_boundary = loss_boundary_h + loss_boundary_dh_dx
        loss = self._loss_residual_weight * loss_residual + self._loss_initial_weight * loss_initial + \
            self._loss_boundary_weight * loss_boundary

        self.loss_total_tracker.update_state(loss)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_initial_tracker.update_state(loss_initial)
        self.loss_boundary_tracker.update_state(loss_boundary)
        self.mae_tracker.update_state(h_samples_exact, h_samples)

        return {m.name: m.result() for m in self.metrics}
    
    def fit_custom(self, inputs: List['tf.Tensor'], outputs: List['tf.Tensor'], epochs: int, print_every: int = 1000):
        '''
        Custom alternative to tensorflow fit function, mainly to allow inputs with different sizes. Training is done in full batches.

        Args:
            data: The data to train on. Should be a list of tensors: [tx_colloc, tx_init, tx_bnd]
            outputs: The outputs to train on. Should be a list of tensors: [u_colloc, residual, u_init, u_bnd]. u_colloc is only used for the MAE metric.
            epochs: The number of epochs to train for.
            print_every: How often to print the metrics. Defaults to 1000.
        '''
        history = create_history_dictionary()
        
        for epoch in range(epochs):
            metrs = self.train_step([inputs, outputs])
            for key, value in metrs.items():
                history[key].append(value.numpy())

            if epoch % print_every == 0:
                tf.print(f"Epoch {epoch}, Loss Residual: {metrs['loss_residual']:0.4f}, Loss Initial: {metrs['loss_initial']:0.4f}, Loss Boundary: {metrs['loss_boundary']:0.4f}, MAE: {metrs['mean_absolute_error']:0.4f}")
                
            #reset metrics
            for m in self.metrics:
                m.reset_states()

        return history

    @property
    def metrics(self):
        """
        Returns the metrics of the model.
        """
        return [self.loss_total_tracker, self.loss_residual_tracker, self.loss_initial_tracker, self.loss_boundary_tracker, self.mae_tracker]


class BurgersPinn(tf.keras.Model):
    """
    A model that solves the Burgers' equation.
    """
    def __init__(self, backbone, nu: float, loss_residual_weight=1.0, loss_initial_weight=1.0, \
        loss_boundary_weight=1.0, **kwargs):
        """
        Initializes the model.

        Args:
            backbone: The backbone model.
            nu: The viscosity of the fluid.
            loss_residual_weight: The weight of the residual loss.
            loss_initial_weight: The weight of the initial loss.
            loss_boundary_weight: The weight of the boundary loss.
            **kwargs: Additional arguments.
        """
        super(BurgersPinn, self).__init__(**kwargs)

        self.backbone = backbone
        self.nu = nu
        self.loss_total_tracker = tf.keras.metrics.Mean(name=LOSS_TOTAL)
        self.loss_residual_tracker = tf.keras.metrics.Mean(name=LOSS_RESIDUAL)
        self.loss_initial_tracker = tf.keras.metrics.Mean(name=LOSS_INITIAL)
        self.loss_boundary_tracker = tf.keras.metrics.Mean(name=LOSS_BOUNDARY)
        self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name=MEAN_ABSOLUTE_ERROR)
        self._loss_residual_weight = tf.Variable(loss_residual_weight, trainable=False, name="loss_residual_weight", dtype=tf.keras.backend.floatx())
        self._loss_boundary_weight = tf.Variable(loss_boundary_weight, trainable=False, name="loss_boundary_weight", dtype=tf.keras.backend.floatx())
        self._loss_initial_weight = tf.Variable(loss_initial_weight, trainable=False, name="loss_initial_weight", dtype=tf.keras.backend.floatx())
        self.res_loss = tf.keras.losses.MeanSquaredError()
        self.init_loss = tf.keras.losses.MeanSquaredError()
        self.bnd_loss = tf.keras.losses.MeanSquaredError()

    def set_loss_weights(self, loss_residual_weight: float, loss_initial_weight: float, loss_boundary_weight: float):
        """
        Sets the loss weights.

        Args:
            loss_residual_weight: The weight of the residual loss.
            loss_initial_weight: The weight of the initial loss.
            loss_boundary_weight: The weight of the boundary loss.
        """
        self._loss_residual_weight.assign(loss_residual_weight)
        self._loss_boundary_weight.assign(loss_boundary_weight)
        self._loss_initial_weight.assign(loss_initial_weight)

    @tf.function
    def call(self, inputs, training=False):
        """
        Performs a forward pass.

        Args:
            inputs: The inputs to the model. First input is the samples, second input is the initial, \
                and third input is the boundary data.
            training: Whether or not the model is training.

        Returns:
            The solution for the residual samples, the lhs residual, the solution for the initial samples, \
                and the solution for the boundary samples.
        """

        tx_samples = inputs[0]
        tx_init = inputs[1]
        tx_bnd = inputs[2]

        with tf.GradientTape(watch_accessed_variables=False) as tape2:
            tape2.watch(tx_samples)

            with tf.GradientTape(watch_accessed_variables=False) as tape1:
                tape1.watch(tx_samples)
                u_samples = self.backbone(tx_samples, training=training)

            first_order = tape1.batch_jacobian(u_samples, tx_samples)
            du_dt = first_order[..., 0]
            du_dx = first_order[..., 1]

        d2u_dx2 = tape2.batch_jacobian(du_dx, tx_samples)[..., 1]

        lhs_samples = du_dt + u_samples * du_dx - self.nu * d2u_dx2
        tx_ib = tf.concat([tx_init, tx_bnd], axis=0)
        u_ib = self.backbone(tx_ib, training=training)
        u_initial = u_ib[:tf.shape(tx_init)[0]]
        u_bnd = u_ib[tf.shape(tx_init)[0]:]

        return u_samples, lhs_samples, u_initial, u_bnd

    @tf.function
    def train_step(self, data):
        """
        Performs a training step.

        Args:
            data: The data to train on. First input is the samples, second input is the initial, \
                and third input is the boundary data. First output is the exact solution for the samples, \
                second output is the exact rhs for the samples, third output is the exact solution for the initial, \
                and fourth output is the exact solution for the boundary.

        Returns:
            The metrics for the training step.
        """

        inputs, outputs = data
        u_samples_exact, rhs_samples_exact, u_initial_exact, u_bnd_exact = outputs

        with tf.GradientTape() as tape:
            u_samples, lhs_samples, u_initial, u_bnd = self(inputs, training=True)

            loss_residual = self.res_loss(rhs_samples_exact, lhs_samples)
            loss_initial = self.init_loss(u_initial_exact, u_initial)
            loss_boundary = self.bnd_loss(u_bnd_exact, u_bnd)
            loss = self._loss_residual_weight * loss_residual + self._loss_initial_weight * loss_initial + \
                self._loss_boundary_weight * loss_boundary

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_total_tracker.update_state(loss)
        self.mae_tracker.update_state(u_samples_exact, u_samples)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_initial_tracker.update_state(loss_initial)
        self.loss_boundary_tracker.update_state(loss_boundary)

        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        """
        Performs a test step.
        """

        inputs, outputs = data
        u_samples_exact, rhs_samples_exact, u_initial_exact, u_bnd_exact = outputs

        u_samples, lhs_samples, u_initial, u_bnd = self(inputs, training=False)

        loss_residual = self.res_loss(rhs_samples_exact, lhs_samples)
        loss_initial = self.init_loss(u_initial_exact, u_initial)
        loss_boundary = self.bnd_loss(u_bnd_exact, u_bnd)
        loss = self._loss_residual_weight * loss_residual + self._loss_initial_weight * loss_initial + \
            self._loss_boundary_weight * loss_boundary

        self.loss_total_tracker.update_state(loss)
        self.mae_tracker.update_state(u_samples_exact, u_samples)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_initial_tracker.update_state(loss_initial)
        self.loss_boundary_tracker.update_state(loss_boundary)

        return {m.name: m.result() for m in self.metrics}

    def fit_custom(self, inputs: List['tf.Tensor'], outputs: List['tf.Tensor'], epochs: int, print_every: int = 1000):
        '''
        Custom alternative to tensorflow fit function, mainly to allow inputs with different sizes. Training is done in full batches.

        Args:
            data: The data to train on. Should be a list of tensors: [tx_colloc, tx_init, tx_bnd]
            outputs: The outputs to train on. Should be a list of tensors: [u_colloc, residual, u_init, u_bnd]. u_colloc is only used for the MAE metric.
            epochs: The number of epochs to train for.
            print_every: How often to print the metrics. Defaults to 1000.
        '''
        history = create_history_dictionary()
        
        for epoch in range(epochs):
            metrs = self.train_step([inputs, outputs])
            for key, value in metrs.items():
                history[key].append(value.numpy())

            if epoch % print_every == 0:
                tf.print(f"Epoch {epoch}, Loss Residual: {metrs['loss_residual']:0.4f}, Loss Initial: {metrs['loss_initial']:0.4f}, Loss Boundary: {metrs['loss_boundary']:0.4f}, MAE: {metrs['mean_absolute_error']:0.4f}")
                
            #reset metrics
            for m in self.metrics:
                m.reset_states()

        return history        

    @property
    def metrics(self):
        """
        Returns the metrics of the model.
        """
        return [self.loss_total_tracker, self.loss_residual_tracker, self.loss_initial_tracker, self.loss_boundary_tracker, self.mae_tracker]

class KdVPinn(tf.keras.Model):
    """
    A model that solves the KdV equation.
    """
    def __init__(self, backbone, k: float = 6.0, c: float = 0., beta: float=1., periodic_BC = False, loss_residual_weight=1.0, loss_initial_weight=1.0, \
        loss_boundary_weight=1.0, **kwargs):
        """
        Initializes the model.

        Args:
            backbone: The backbone model.
            k: The heat diffusivity constant.
            loss_residual_weight: The weight of the residual loss.
            loss_initial_weight: The weight of the initial loss.
            loss_boundary_weight: The weight of the boundary loss.
            **kwargs: Additional keyword arguments.
        """
        super(KdVPinn, self).__init__(**kwargs)

        self.backbone = backbone
        self.k = k
        self.c = c
        self.beta = beta
        self.periodic_BC = periodic_BC
        self.loss_total_tracker = tf.keras.metrics.Mean(name=LOSS_TOTAL)
        self.loss_residual_tracker = tf.keras.metrics.Mean(name=LOSS_RESIDUAL)
        self.loss_initial_tracker = tf.keras.metrics.Mean(name=LOSS_INITIAL)
        self.loss_boundary_tracker = tf.keras.metrics.Mean(name=LOSS_BOUNDARY)
        self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name=MEAN_ABSOLUTE_ERROR)
        self._loss_residual_weight = tf.Variable(loss_residual_weight, trainable=False, name="loss_residual_weight", dtype=tf.keras.backend.floatx())
        self._loss_boundary_weight = tf.Variable(loss_boundary_weight, trainable=False, name="loss_boundary_weight", dtype=tf.keras.backend.floatx())
        self._loss_initial_weight = tf.Variable(loss_initial_weight, trainable=False, name="loss_initial_weight", dtype=tf.keras.backend.floatx())
        self.res_loss = tf.keras.losses.MeanSquaredError()
        self.init_loss = tf.keras.losses.MeanSquaredError()
        self.bnd_loss = tf.keras.losses.MeanSquaredError()
        self.grad_res = []
        self.grad_bcs = []

    def set_loss_weights(self, loss_residual_weight: float, loss_initial_weight: float, loss_boundary_weight: float):
        """
        Sets the loss weights.

        Args:
            loss_residual_weight: The weight of the residual loss.
            loss_initial_weight: The weight of the initial loss.
            loss_boundary_weight: The weight of the boundary loss.
        """
        self._loss_residual_weight.assign(loss_residual_weight)
        self._loss_boundary_weight.assign(loss_boundary_weight)
        self._loss_initial_weight.assign(loss_initial_weight)


    @tf.function
    def call(self, inputs, training=False):
        """
        Performs a forward pass.

        Args:
            inputs: The inputs to the model. First input is the samples, second input is the initial, \
                and third input is the boundary data.
            training: Whether or not the model is training.

        Returns:
            The solution for the residual samples, the lhs residual, the solution for the initial samples, \
                and the solution for the boundary samples.
        """

        tx_samples = inputs[0]
        tx_init = inputs[1]
        tx_bnd_start = inputs[2]
        tx_bnd_end = inputs[3]
        tx_bnd = inputs[4]
        with tf.GradientTape(watch_accessed_variables=False) as tape3:
            tape3.watch(tx_samples)
            with tf.GradientTape(watch_accessed_variables=False) as tape2:
                tape2.watch(tx_samples)

                with tf.GradientTape(watch_accessed_variables=False) as tape1:
                    tape1.watch(tx_samples)
                    u_samples = self.backbone(tx_samples, training=training)

                first_order = tape1.batch_jacobian(u_samples, tx_samples)
                du_dt = first_order[..., 0]
                du_dx = first_order[..., 1]
            d2u_dx2 = tape2.batch_jacobian(du_dx, tx_samples)[..., 1]
        d3u_dx3 = tape3.batch_jacobian(d2u_dx2, tx_samples)[..., 1]

        lhs_samples = du_dt + (self.k * u_samples) * du_dx  + self.beta * d3u_dx3

        tx_ib = tf.concat([tx_init, tx_bnd], axis=0)
        u_ib = self.backbone(tx_ib, training=training)
        u_initial = u_ib[:tf.shape(tx_init)[0]]
        u_bnd = u_ib[tf.shape(tx_init)[0]:]
 
        u_bnd_start_pred = self.backbone(tx_bnd_start, training=training)
        u_bnd_end_pred = self.backbone(tx_bnd_end, training=training)

        return u_samples, lhs_samples, u_initial, u_bnd_start_pred, u_bnd_end_pred, u_bnd

    @tf.function
    def train_step(self, data):
        """
        Performs a training step.

        Args:
            data: The data to train on. First input is the samples, second input is the initial, \
                and third input is the boundary data. First output is the exact solution for the samples, \
                second output is the exact rhs for the samples, third output is the exact solution for the initial, \
                and fourth output is the exact solution for the boundary.
        """

        inputs, outputs = data
        u_samples_exact, rhs_samples_exact, u_initial_exact, u_bnd_exact = outputs

        with tf.GradientTape() as tape:
            u_samples, lhs_samples, u_initial, u_bnd_start_pred, u_bnd_end_pred, u_bnd = self(inputs, training=True)

            loss_residual = self.res_loss(rhs_samples_exact, lhs_samples)
            loss_initial = self.init_loss(u_initial_exact, u_initial)
            if self.periodic_BC == False:
                loss_boundary = self.bnd_loss(u_bnd_exact, u_bnd)
            else: 
                loss_boundary = self.bnd_loss(u_bnd_start_pred, u_bnd_end_pred)
            loss = self._loss_residual_weight * loss_residual + self._loss_initial_weight * loss_initial + \
                self._loss_boundary_weight * loss_boundary

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_total_tracker.update_state(loss)
        self.mae_tracker.update_state(u_samples_exact, u_samples)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_initial_tracker.update_state(loss_initial)
        self.loss_boundary_tracker.update_state(loss_boundary)

        self.grad_res = []
        self.grad_bcs = []

        for i in range(len(self.backbone.layers) - 1):
            self.grad_res.append(tf.gradients(loss_boundary, self.backbone.weights[i])[0])
            self.grad_bcs.append(tf.gradients(loss_residual, self.backbone.weights[i])[0])

        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        """
        Performs a test step.

        Args:
            data: The data to train on. First input is the samples, second input is the initial, \
                and third input is the boundary data. First output is the exact solution for the samples, \
                second output is the exact rhs for the samples, third output is the exact solution for the initial, \
                and fourth output is the exact solution for the boundary.
        """

        inputs, outputs = data
        u_samples_exact, rhs_samples_exact, u_initial_exact, u_bnd_exact= outputs
        u_samples, lhs_samples, u_initial, u_bnd_start, u_bnd_end, u_bnd = self(inputs, training=False)

        loss_residual = self.res_loss(rhs_samples_exact, lhs_samples)
        loss_initial = self.init_loss(u_initial_exact, u_initial)
        if self.periodic_BC == False:
            loss_boundary = self.bnd_loss(u_bnd_exact, u_bnd)
        else: 
            loss_boundary = self.bnd_loss(u_bnd_start, u_bnd_end)
        loss = self._loss_residual_weight * loss_residual + self._loss_initial_weight * loss_initial + \
            self._loss_boundary_weight * loss_boundary

        self.loss_total_tracker.update_state(loss)
        self.mae_tracker.update_state(u_samples_exact, u_samples)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_initial_tracker.update_state(loss_initial)
        self.loss_boundary_tracker.update_state(loss_boundary)

        return {m.name: m.result() for m in self.metrics}
    
    def fit_custom(self, inputs: List['tf.Tensor'], outputs: List['tf.Tensor'], epochs: int, print_every: int = 1000):
        '''
        Custom alternative to tensorflow fit function, mainly to allow inputs with different sizes. Training is done in full batches.

        Args:
            data: The data to train on. Should be a list of tensors: [tx_colloc, tx_init, tx_bnd]
            outputs: The outputs to train on. Should be a list of tensors: [u_colloc, residual, u_init, u_bnd]. u_colloc is only used for the MAE metric.
            epochs: The number of epochs to train for.
            print_every: How often to print the metrics. Defaults to 1000.
        '''
        history = create_history_dictionary()
        
        for epoch in range(epochs):
            metrs = self.train_step([inputs, outputs])
            for key, value in metrs.items():
                history[key].append(value.numpy())

            if epoch % print_every == 0:
                tf.print(f"Epoch {epoch}, Loss Residual: {metrs['loss_residual']:0.10f}, Loss Initial: {metrs['loss_initial']:0.10f}, Loss Boundary: {metrs['loss_boundary']:0.10f}, MAE: {metrs['mean_absolute_error']:0.10f}")
                
            #reset metrics
            for m in self.metrics:
                m.reset_states()

        return history

    @property
    def metrics(self):
        """
        Returns the metrics of the model.
        """
        return [self.loss_total_tracker, self.loss_residual_tracker, self.loss_initial_tracker, self.loss_boundary_tracker, self.mae_tracker]

class KdVBurgersPinn(tf.keras.Model):
    """
    A model that solves the KdV-B equation.
    """
    def __init__(self, backbone, gamma: float = 1.0, alpha: float = 1., beta: float = 1., c: float = 1., loss_residual_weight=1.0, loss_initial_weight=1.0, \
        loss_boundary_weight=1.0, **kwargs):
        """
        Initializes the model.

        Args:
            backbone: The backbone model.
            gamma: nonlinear term coefficient
            alpha: diffusive term coefficient
            beta: dispersive term coefficient
        """
        super(KdVBurgersPinn, self).__init__(**kwargs)

        self.backbone = backbone
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.c = c

        self.loss_total_tracker = tf.keras.metrics.Mean(name=LOSS_TOTAL)
        self.loss_residual_tracker = tf.keras.metrics.Mean(name=LOSS_RESIDUAL)
        self.loss_initial_tracker = tf.keras.metrics.Mean(name=LOSS_INITIAL)
        self.loss_boundary_tracker = tf.keras.metrics.Mean(name=LOSS_BOUNDARY)
        self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name=MEAN_ABSOLUTE_ERROR)
        self._loss_residual_weight = tf.Variable(loss_residual_weight, trainable=False, name="loss_residual_weight", dtype=tf.keras.backend.floatx())
        self._loss_boundary_weight = tf.Variable(loss_boundary_weight, trainable=False, name="loss_boundary_weight", dtype=tf.keras.backend.floatx())
        self._loss_initial_weight = tf.Variable(loss_initial_weight, trainable=False, name="loss_initial_weight", dtype=tf.keras.backend.floatx())
        self.res_loss = tf.keras.losses.MeanSquaredError()
        self.init_loss = tf.keras.losses.MeanSquaredError()
        self.bnd_loss = tf.keras.losses.MeanSquaredError()

    def set_loss_weights(self, loss_residual_weight: float, loss_initial_weight: float, loss_boundary_weight: float):
        """
        Sets the loss weights.

        Args:
            loss_residual_weight: The weight of the residual loss.
            loss_initial_weight: The weight of the initial loss.
            loss_boundary_weight: The weight of the boundary loss.
        """
        self._loss_residual_weight.assign(loss_residual_weight)
        self._loss_boundary_weight.assign(loss_boundary_weight)
        self._loss_initial_weight.assign(loss_initial_weight)


    @tf.function
    def call(self, inputs, training=False):
        """
        Performs a forward pass.

        Args:
            inputs: The inputs to the model. First input is the samples, second input is the initial, \
                and third input is the boundary data.
            training: Whether or not the model is training.

        Returns:
            The solution for the residual samples, the lhs residual, the solution for the initial samples, \
                and the solution for the boundary samples.
        """

        tx_samples = inputs[0]
        tx_init = inputs[1]
        tx_bnd = inputs[2]

        with tf.GradientTape(watch_accessed_variables=False) as tape3:
            tape3.watch(tx_samples)
            with tf.GradientTape(watch_accessed_variables=False) as tape2:
                tape2.watch(tx_samples)

                with tf.GradientTape(watch_accessed_variables=False) as tape1:
                    tape1.watch(tx_samples)
                    u_samples = self.backbone(tx_samples, training=training)

                first_order = tape1.batch_jacobian(u_samples, tx_samples)
                du_dt = first_order[..., 0]
                du_dx = first_order[..., 1]
            d2u_dx2 = tape2.batch_jacobian(du_dx, tx_samples)[..., 1]
        d3u_dx3 = tape3.batch_jacobian(d2u_dx2, tx_samples)[..., 1]

        # lhs_samples = du_dt + (self.gamma * u_samples) * du_dx - self.alpha * d2u_dx2 + self.beta * d3u_dx3
        lhs_samples = du_dt + self.c * du_dx + (self.gamma * u_samples) * du_dx - self.alpha * d2u_dx2 + self.beta * d3u_dx3
        

        tx_ib = tf.concat([tx_init, tx_bnd], axis=0)
        u_ib = self.backbone(tx_ib, training=training)
        u_initial = u_ib[:tf.shape(tx_init)[0]]
        u_bnd = u_ib[tf.shape(tx_init)[0]:]
 

        return u_samples, lhs_samples, u_initial, u_bnd

    @tf.function
    def train_step(self, data):
        """
        Performs a training step.

        Args:
            data: The data to train on. First input is the samples, second input is the initial, \
                and third input is the boundary data. First output is the exact solution for the samples, \
                second output is the exact rhs for the samples, third output is the exact solution for the initial, \
                and fourth output is the exact solution for the boundary.
        """

        inputs, outputs = data
        u_samples_exact, lhs_samples_exact, u_initial_exact, u_bnd_exact = outputs

        with tf.GradientTape() as tape:
            u_samples, lhs_samples, u_initial, u_bnd = self(inputs, training=True)

            loss_residual = self.res_loss(lhs_samples_exact, lhs_samples)
            loss_initial = self.init_loss(u_initial_exact, u_initial)
            loss_boundary = self.bnd_loss(u_bnd_exact, u_bnd)
            loss = self._loss_residual_weight * loss_residual + self._loss_initial_weight * loss_initial + \
                self._loss_boundary_weight * loss_boundary

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_total_tracker.update_state(loss)
        self.mae_tracker.update_state(u_samples_exact, u_samples)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_initial_tracker.update_state(loss_initial)
        self.loss_boundary_tracker.update_state(loss_boundary)

        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        """
        Performs a test step.

        Args:
            data: The data to train on. First input is the samples, second input is the initial, \
                and third input is the boundary data. First output is the exact solution for the samples, \
                second output is the exact rhs for the samples, third output is the exact solution for the initial, \
                and fourth output is the exact solution for the boundary.
        """

        inputs, outputs = data
        u_samples_exact, lhs_samples_exact, u_initial_exact, u_bnd_exact= outputs
        u_samples, lhs_samples, u_initial, u_bnd = self(inputs, training=False)

        loss_residual = self.res_loss(lhs_samples_exact, lhs_samples)
        loss_initial = self.init_loss(u_initial_exact, u_initial)
        loss_boundary = self.bnd_loss(u_bnd_exact, u_bnd)
        loss = self._loss_residual_weight * loss_residual + self._loss_initial_weight * loss_initial + \
            self._loss_boundary_weight * loss_boundary

        self.loss_total_tracker.update_state(loss)
        self.mae_tracker.update_state(u_samples_exact, u_samples)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_initial_tracker.update_state(loss_initial)
        self.loss_boundary_tracker.update_state(loss_boundary)

        return {m.name: m.result() for m in self.metrics}
    
    def fit_custom(self, inputs: List['tf.Tensor'], outputs: List['tf.Tensor'], epochs: int, print_every: int = 1000):
        '''
        Custom alternative to tensorflow fit function, mainly to allow inputs with different sizes. Training is done in full batches.

        Args:
            data: The data to train on. Should be a list of tensors: [tx_colloc, tx_init, tx_bnd]
            outputs: The outputs to train on. Should be a list of tensors: [u_colloc, residual, u_init, u_bnd]. u_colloc is only used for the MAE metric.
            epochs: The number of epochs to train for.
            print_every: How often to print the metrics. Defaults to 1000.
        '''
        history = create_history_dictionary()
        
        for epoch in range(epochs):
            metrs = self.train_step([inputs, outputs])
            for key, value in metrs.items():
                history[key].append(value.numpy())

            if epoch % print_every == 0:
                tf.print(f"Epoch {epoch}, Loss Residual: {metrs['loss_residual']:0.10f}, Loss Initial: {metrs['loss_initial']:0.10f}, Loss Boundary: {metrs['loss_boundary']:0.10f}, MAE: {metrs['mean_absolute_error']:0.10f}")
                
            #reset metrics
            for m in self.metrics:
                m.reset_states()

        return history

    @property
    def metrics(self):
        """
        Returns the metrics of the model.
        """
        return [self.loss_total_tracker, self.loss_residual_tracker, self.loss_initial_tracker, self.loss_boundary_tracker, self.mae_tracker]
# here
class mKdVPinn(tf.keras.Model):
    """
    A model that solves the KdV equation.
    """
    def __init__(self, backbone, d_0: float = 1.0, d_1: float = 1., d_2: float=1., loss_residual_weight=1.0, loss_initial_weight=1.0, \
        loss_boundary_weight=1.0, **kwargs):

        super(mKdVPinn, self).__init__(**kwargs)

        self.backbone = backbone
        self.d_0 = d_0
        self.d_1 = d_1
        self.d_2 = d_2
        self.loss_total_tracker = tf.keras.metrics.Mean(name=LOSS_TOTAL)
        self.loss_residual_tracker = tf.keras.metrics.Mean(name=LOSS_RESIDUAL)
        self.loss_initial_tracker = tf.keras.metrics.Mean(name=LOSS_INITIAL)
        self.loss_boundary_tracker = tf.keras.metrics.Mean(name=LOSS_BOUNDARY)
        self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name=MEAN_ABSOLUTE_ERROR)
        self._loss_residual_weight = tf.Variable(loss_residual_weight, trainable=False, name="loss_residual_weight", dtype=tf.keras.backend.floatx())
        self._loss_boundary_weight = tf.Variable(loss_boundary_weight, trainable=False, name="loss_boundary_weight", dtype=tf.keras.backend.floatx())
        self._loss_initial_weight = tf.Variable(loss_initial_weight, trainable=False, name="loss_initial_weight", dtype=tf.keras.backend.floatx())
        self.res_loss = tf.keras.losses.MeanSquaredError()
        self.init_loss = tf.keras.losses.MeanSquaredError()
        self.bnd_loss = tf.keras.losses.MeanSquaredError()
        self.grad_res = []
        self.grad_bcs = []

    def set_loss_weights(self, loss_residual_weight: float, loss_initial_weight: float, loss_boundary_weight: float):
        """
        Sets the loss weights.

        Args:
            loss_residual_weight: The weight of the residual loss.
            loss_initial_weight: The weight of the initial loss.
            loss_boundary_weight: The weight of the boundary loss.
        """
        self._loss_residual_weight.assign(loss_residual_weight)
        self._loss_boundary_weight.assign(loss_boundary_weight)
        self._loss_initial_weight.assign(loss_initial_weight)


    @tf.function
    def call(self, inputs, training=False):
        """
        Performs a forward pass.

        Args:
            inputs: The inputs to the model. First input is the samples, second input is the initial, \
                and third input is the boundary data.
            training: Whether or not the model is training.

        Returns:
            The solution for the residual samples, the lhs residual, the solution for the initial samples, \
                and the solution for the boundary samples.
        """

        xt_samples = inputs[0]
        xt_init = inputs[1]
        xt_bnd = inputs[2]
        with tf.GradientTape(watch_accessed_variables=False) as tape3:
            tape3.watch(xt_samples)
            with tf.GradientTape(watch_accessed_variables=False) as tape2:
                tape2.watch(xt_samples)

                with tf.GradientTape(watch_accessed_variables=False) as tape1:
                    tape1.watch(xt_samples)
                    u_samples = self.backbone(xt_samples, training=training)

                first_order = tape1.batch_jacobian(u_samples, xt_samples)
                du_dx = first_order[..., 0]
                du_dt = first_order[..., 1]
            d2u_dt2 = tape2.batch_jacobian(du_dt, xt_samples)[..., 1]
        d3u_dt3 = tape3.batch_jacobian(d2u_dt2, xt_samples)[..., 1]

        lhs_samples = du_dx + self.d_0 * du_dt + self.d_1 * u_samples * du_dt  + self.d_2 * d3u_dt3


        u_initial = self.backbone(xt_init, training=training)
        u_bnd = self.backbone(xt_bnd, training=training)


        return u_samples, lhs_samples, u_initial, u_bnd

    @tf.function
    def train_step(self, data):
        """
        Performs a training step.

        Args:
            data: The data to train on. First input is the samples, second input is the initial, \
                and third input is the boundary data. First output is the exact solution for the samples, \
                second output is the exact rhs for the samples, third output is the exact solution for the initial, \
                and fourth output is the exact solution for the boundary.
        """

        inputs, outputs = data
        u_samples_exact, rhs_samples_exact, u_initial_exact, u_bnd_exact = outputs

        with tf.GradientTape() as tape:
            u_samples, lhs_samples, u_initial, u_bnd = self(inputs, training=True)

            loss_residual = self.res_loss(rhs_samples_exact, lhs_samples)
            loss_initial = self.init_loss(u_initial_exact, u_initial)
            loss_boundary = self.bnd_loss(u_bnd_exact, u_bnd)
            loss = self._loss_residual_weight * loss_residual + self._loss_initial_weight * loss_initial + \
                self._loss_boundary_weight * loss_boundary

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_total_tracker.update_state(loss)
        self.mae_tracker.update_state(u_samples_exact, u_samples)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_initial_tracker.update_state(loss_initial)
        self.loss_boundary_tracker.update_state(loss_boundary)

        self.grad_res = []
        self.grad_bcs = []

        for i in range(len(self.backbone.layers) - 1):
            self.grad_res.append(tf.gradients(loss_boundary, self.backbone.weights[i])[0])
            self.grad_bcs.append(tf.gradients(loss_residual, self.backbone.weights[i])[0])

        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        """
        Performs a test step.

        Args:
            data: The data to train on. First input is the samples, second input is the initial, \
                and third input is the boundary data. First output is the exact solution for the samples, \
                second output is the exact rhs for the samples, third output is the exact solution for the initial, \
                and fourth output is the exact solution for the boundary.
        """

        inputs, outputs = data
        u_samples_exact, rhs_samples_exact, u_initial_exact, u_bnd_exact= outputs
        u_samples, lhs_samples, u_initial, u_bnd = self(inputs, training=False)

        loss_residual = self.res_loss(rhs_samples_exact, lhs_samples)
        loss_initial = self.init_loss(u_initial_exact, u_initial)
        loss_boundary = self.bnd_loss(u_bnd_exact, u_bnd)
        loss = self._loss_residual_weight * loss_residual + self._loss_initial_weight * loss_initial + \
            self._loss_boundary_weight * loss_boundary

        self.loss_total_tracker.update_state(loss)
        self.mae_tracker.update_state(u_samples_exact, u_samples)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_initial_tracker.update_state(loss_initial)
        self.loss_boundary_tracker.update_state(loss_boundary)

        return {m.name: m.result() for m in self.metrics}
    
    def fit_custom(self, inputs: List['tf.Tensor'], outputs: List['tf.Tensor'], epochs: int, print_every: int = 1000):
        '''
        Custom alternative to tensorflow fit function, mainly to allow inputs with different sizes. Training is done in full batches.

        Args:
            data: The data to train on. Should be a list of tensors: [tx_colloc, tx_init, tx_bnd]
            outputs: The outputs to train on. Should be a list of tensors: [u_colloc, residual, u_init, u_bnd]. u_colloc is only used for the MAE metric.
            epochs: The number of epochs to train for.
            print_every: How often to print the metrics. Defaults to 1000.
        '''
        history = create_history_dictionary()
        
        for epoch in range(epochs):
            metrs = self.train_step([inputs, outputs])
            for key, value in metrs.items():
                history[key].append(value.numpy())

            if epoch % print_every == 0:
                tf.print(f"Epoch {epoch}, Loss Residual: {metrs['loss_residual']:0.10f}, Loss Initial: {metrs['loss_initial']:0.10f}, Loss Boundary: {metrs['loss_boundary']:0.10f}, MAE: {metrs['mean_absolute_error']:0.10f}")
                
            #reset metrics
            for m in self.metrics:
                m.reset_states()

        return history

    @property
    def metrics(self):
        """
        Returns the metrics of the model.
        """
        return [self.loss_total_tracker, self.loss_residual_tracker, self.loss_initial_tracker, self.loss_boundary_tracker, self.mae_tracker]



class KPPinn(tf.keras.Model):

    def __init__(self, backbone, k: float = 6.0, c: float = 0., sig_sq: float = 3.,  loss_residual_weight=1.0, loss_initial_weight=1.0, \
        loss_boundary_weight=1.0, PBC_x = False, PBC_y = True, **kwargs):

        super(KPPinn, self).__init__(**kwargs)

        self.backbone = backbone
        self.k = k
        self.c = c
        self.sig_sq = sig_sq
        self.PBC_x = PBC_x
        self.PBC_y = PBC_y

        self.loss_total_tracker = tf.keras.metrics.Mean(name=LOSS_TOTAL)
        self.loss_residual_tracker = tf.keras.metrics.Mean(name=LOSS_RESIDUAL)
        self.loss_initial_tracker = tf.keras.metrics.Mean(name=LOSS_INITIAL)
        self.loss_boundary_tracker = tf.keras.metrics.Mean(name=LOSS_BOUNDARY)
        self.loss_boundary_y_tracker = tf.keras.metrics.Mean(name=LOSS_BOUNDARY_Y)
        self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name=MEAN_ABSOLUTE_ERROR)
        self._loss_residual_weight = tf.Variable(loss_residual_weight, trainable=False, name="loss_residual_weight", dtype=tf.keras.backend.floatx())
        self._loss_boundary_weight = tf.Variable(loss_boundary_weight, trainable=False, name="loss_boundary_weight", dtype=tf.keras.backend.floatx())
        self._loss_initial_weight = tf.Variable(loss_initial_weight, trainable=False, name="loss_initial_weight", dtype=tf.keras.backend.floatx())
        self.res_loss = tf.keras.losses.MeanSquaredError()
        self.init_loss = tf.keras.losses.MeanSquaredError()
        self.bnd_loss = tf.keras.losses.MeanSquaredError()
        self.bnd_y_loss = tf.keras.losses.MeanSquaredError()

    def set_loss_weights(self, loss_residual_weight: float, loss_initial_weight: float, loss_boundary_weight: float):
        """
        Sets the loss weights.

        Args:
            loss_residual_weight: The weight of the residual loss.
            loss_initial_weight: The weight of the initial loss.
            loss_boundary_weight: The weight of the boundary loss.
        """
        self._loss_residual_weight.assign(loss_residual_weight)
        self._loss_boundary_weight.assign(loss_boundary_weight)
        self._loss_initial_weight.assign(loss_initial_weight)


    @tf.function
    def call(self, inputs, training=False):

        txy_samples = inputs[0]
        txy_init = inputs[1]
        txy_bnd_x = inputs[2]
        txy_x_left = inputs[3]
        txy_x_right = inputs[4]
        txy_bnd_y = inputs[5]
        txy_y_left = inputs[6]
        txy_y_right = inputs[7]
        with tf.GradientTape(watch_accessed_variables=False) as tape4:
            tape4.watch(txy_samples)
            with tf.GradientTape(watch_accessed_variables=False) as tape3:
                tape3.watch(txy_samples)
                with tf.GradientTape(watch_accessed_variables=False, persistent = True) as tape2:
                    tape2.watch(txy_samples)

                    with tf.GradientTape(watch_accessed_variables=False) as tape1:
                        tape1.watch(txy_samples)
                        u_samples = self.backbone(txy_samples, training=training)

                    first_order = tape1.batch_jacobian(u_samples, txy_samples)
                    du_dt = first_order[..., 0]
                    du_dx = first_order[..., 1]
                    du_dy = first_order[..., 2]

                ddu_dtdx = tape2.batch_jacobian(du_dt, txy_samples)[..., 1]
                d2u_dx2 = tape2.batch_jacobian(du_dx, txy_samples)[..., 1]
                d2u_dy2 = tape2.batch_jacobian(du_dy, txy_samples)[..., 2]
            d3u_dx3 = tape3.batch_jacobian(d2u_dx2, txy_samples)[..., 1]
        d4u_dx4 = tape4.batch_jacobian(d3u_dx3, txy_samples)[..., 1]

        lhs_samples = ddu_dtdx + 6*((du_dx)**2 + u_samples * d2u_dx2) + d4u_dx4 + self.sig_sq * d2u_dy2
        # lhs_samples =  + 6*((du_dx)**2 + u_samples)
  
        tx_ib = tf.concat([txy_init, txy_bnd_x], axis=0)
        u_ib = self.backbone(tx_ib, training=training)
        u_initial = u_ib[:tf.shape(txy_init)[0]]
        u_bnd_x = u_ib[tf.shape(txy_init)[0]:]

        u_y_left = self.backbone(txy_y_left, training=training)
        u_y_right = self.backbone(txy_y_right, training=training)
        u_x_right = self.backbone(txy_x_right, training=training)
        u_x_left = self.backbone(txy_x_left, training=training)
        u_bnd_y = self.backbone(txy_bnd_y, training=training)


        return u_samples, lhs_samples, u_initial, u_bnd_x, u_x_left, u_x_right, u_bnd_y, u_y_left, u_y_right

    @tf.function
    def train_step(self, data):
        """
        Performs a training step.

        Args:
            data: The data to train on. First input is the samples, second input is the initial, \
                and third input is the boundary data. First output is the exact solution for the samples, \
                second output is the exact rhs for the samples, third output is the exact solution for the initial, \
                and fourth output is the exact solution for the boundary.
        """

        inputs, outputs = data
        u_samples_exact, rhs_samples_exact, u_initial_exact, u_bnd_x_exact, u_bnd_y_exact = outputs

        with tf.GradientTape() as tape:
            u_samples, lhs_samples, u_initial, u_bnd_x,  u_x_left, u_x_right, u_bnd_y, u_y_left, u_y_right = self(inputs, training=True)

            loss_residual = self.res_loss(rhs_samples_exact, lhs_samples)
            loss_initial = self.init_loss(u_initial_exact, u_initial)
            if self.PBC_x:
                loss_boundary = self.bnd_loss(u_x_right, u_x_left)
            else:
                loss_boundary = self.bnd_loss(u_bnd_x_exact, u_bnd_x)
            if self.PBC_y:
                loss_boundary_y = self.bnd_loss(u_y_left, u_y_right)
            else:
                loss_boundary_y = self.bnd_loss(u_bnd_y_exact, u_bnd_y)

            loss = self._loss_residual_weight * loss_residual + self._loss_initial_weight * loss_initial + \
                self._loss_boundary_weight * (loss_boundary + loss_boundary_y)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_total_tracker.update_state(loss)
        self.mae_tracker.update_state(u_samples_exact, u_samples)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_initial_tracker.update_state(loss_initial)
        self.loss_boundary_tracker.update_state(loss_boundary)
        self.loss_boundary_y_tracker.update_state(loss_boundary_y)

        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):

        inputs, outputs = data
        u_samples_exact, rhs_samples_exact, u_initial_exact, u_bnd_x_exact, u_bnd_y_exact= outputs
        u_samples, lhs_samples, u_initial, u_bnd_x,  u_x_left, u_x_right, u_bnd_y, u_y_left, u_y_right  = self(inputs, training=False)

        loss_residual = self.res_loss(rhs_samples_exact, lhs_samples)
        loss_initial = self.init_loss(u_initial_exact, u_initial)
        if self.PBC_x:
            loss_boundary = self.bnd_loss(u_x_right, u_x_left)
        else:
            loss_boundary = self.bnd_loss(u_bnd_x_exact, u_bnd_x)
        if self.PBC_y:
            loss_boundary_y = self.bnd_loss(u_y_left, u_y_right)
        else:
            loss_boundary_y = self.bnd_loss(u_bnd_y_exact, u_bnd_y)

        loss = self._loss_residual_weight * loss_residual + self._loss_initial_weight * loss_initial + \
            self._loss_boundary_weight * (loss_boundary + loss_boundary_y)

        self.loss_total_tracker.update_state(loss)
        self.mae_tracker.update_state(u_samples_exact, u_samples)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_initial_tracker.update_state(loss_initial)
        self.loss_boundary_tracker.update_state(loss_boundary)
        self.loss_boundary_y_tracker.update_state(loss_boundary_y)
        

        return {m.name: m.result() for m in self.metrics}
    
    def fit_custom(self, inputs: List['tf.Tensor'], outputs: List['tf.Tensor'], epochs: int, print_every: int = 1000):
        '''
        Custom alternative to tensorflow fit function, mainly to allow inputs with different sizes. Training is done in full batches.

        Args:
            data: The data to train on. Should be a list of tensors: [tx_colloc, tx_init, tx_bnd]
            outputs: The outputs to train on. Should be a list of tensors: [u_colloc, residual, u_init, u_bnd]. u_colloc is only used for the MAE metric.
            epochs: The number of epochs to train for.
            print_every: How often to print the metrics. Defaults to 1000.
        '''
        history = create_history_dictionary_KP()
        
        for epoch in range(epochs):
            metrs = self.train_step([inputs, outputs])
            for key, value in metrs.items():
                history[key].append(value.numpy())

            if epoch % print_every == 0:
                tf.print(f"Epoch {epoch}, Loss Residual: {metrs['loss_residual']:0.10f}, Loss Initial: {metrs['loss_initial']:0.10f}, Loss Boundary: {metrs['loss_boundary']:0.10f}, Loss Boundary Y: {metrs['loss_boundary_y']:0.10f}, MAE: {metrs['mean_absolute_error']:0.10f}")
                
            #reset metrics
            for m in self.metrics:
                m.reset_states()

        return history

    @property
    def metrics(self):
        """
        Returns the metrics of the model.
        """
        return [self.loss_total_tracker, self.loss_residual_tracker, self.loss_initial_tracker, self.loss_boundary_tracker, self.loss_boundary_y_tracker, self.mae_tracker]

class HeatPinn(tf.keras.Model):
    """
    A model that solves the heat equation.
    """
    def __init__(self, backbone, k: float = 1.0, loss_residual_weight=1.0, loss_initial_weight=1.0, \
        loss_boundary_weight=1.0, **kwargs):
        """
        Initializes the model.

        Args:
            backbone: The backbone model.
            k: The heat diffusivity constant.
            loss_residual_weight: The weight of the residual loss.
            loss_initial_weight: The weight of the initial loss.
            loss_boundary_weight: The weight of the boundary loss.
            **kwargs: Additional keyword arguments.
        """
        super(HeatPinn, self).__init__(**kwargs)

        self.backbone = backbone
        self.k = k
        self.loss_total_tracker = tf.keras.metrics.Mean(name=LOSS_TOTAL)
        self.loss_residual_tracker = tf.keras.metrics.Mean(name=LOSS_RESIDUAL)
        self.loss_initial_tracker = tf.keras.metrics.Mean(name=LOSS_INITIAL)
        self.loss_boundary_tracker = tf.keras.metrics.Mean(name=LOSS_BOUNDARY)
        self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name=MEAN_ABSOLUTE_ERROR)
        self._loss_residual_weight = tf.Variable(loss_residual_weight, trainable=False, name="loss_residual_weight", dtype=tf.keras.backend.floatx())
        self._loss_boundary_weight = tf.Variable(loss_boundary_weight, trainable=False, name="loss_boundary_weight", dtype=tf.keras.backend.floatx())
        self._loss_initial_weight = tf.Variable(loss_initial_weight, trainable=False, name="loss_initial_weight", dtype=tf.keras.backend.floatx())
        self.res_loss = tf.keras.losses.MeanSquaredError()
        self.init_loss = tf.keras.losses.MeanSquaredError()
        self.bnd_loss = tf.keras.losses.MeanSquaredError()

    def set_loss_weights(self, loss_residual_weight: float, loss_initial_weight: float, loss_boundary_weight: float):
        """
        Sets the loss weights.

        Args:
            loss_residual_weight: The weight of the residual loss.
            loss_initial_weight: The weight of the initial loss.
            loss_boundary_weight: The weight of the boundary loss.
        """
        self._loss_residual_weight.assign(loss_residual_weight)
        self._loss_boundary_weight.assign(loss_boundary_weight)
        self._loss_initial_weight.assign(loss_initial_weight)


    @tf.function
    def call(self, inputs, training=False):
        """
        Performs a forward pass.

        Args:
            inputs: The inputs to the model. First input is the samples, second input is the initial, \
                and third input is the boundary data.
            training: Whether or not the model is training.

        Returns:
            The solution for the residual samples, the lhs residual, the solution for the initial samples, \
                and the solution for the boundary samples.
        """

        tx_samples = inputs[0]
        tx_init = inputs[1]
        tx_bnd = inputs[2]

        with tf.GradientTape(watch_accessed_variables=False) as tape2:
            tape2.watch(tx_samples)

            with tf.GradientTape(watch_accessed_variables=False) as tape1:
                tape1.watch(tx_samples)
                u_samples = self.backbone(tx_samples, training=training)

            first_order = tape1.batch_jacobian(u_samples, tx_samples)
            du_dt = first_order[..., 0]
            du_dx = first_order[..., 1]
        d2u_dx2 = tape2.batch_jacobian(du_dx, tx_samples)[..., 1]
        lhs_samples = du_dt - self.k * d2u_dx2

        tx_ib = tf.concat([tx_init, tx_bnd], axis=0)
        u_ib = self.backbone(tx_ib, training=training)
        u_initial = u_ib[:tf.shape(tx_init)[0]]
        u_bnd = u_ib[tf.shape(tx_init)[0]:]

        return u_samples, lhs_samples, u_initial, u_bnd

    @tf.function
    def train_step(self, data):
        """
        Performs a training step.

        Args:
            data: The data to train on. First input is the samples, second input is the initial, \
                and third input is the boundary data. First output is the exact solution for the samples, \
                second output is the exact rhs for the samples, third output is the exact solution for the initial, \
                and fourth output is the exact solution for the boundary.
        """

        inputs, outputs = data
        u_samples_exact, rhs_samples_exact, u_initial_exact, u_bnd_exact = outputs

        with tf.GradientTape() as tape:
            u_samples, lhs_samples, u_initial, u_bnd = self(inputs, training=True)

            loss_residual = self.res_loss(rhs_samples_exact, lhs_samples)
            loss_initial = self.init_loss(u_initial_exact, u_initial)
            loss_boundary = self.bnd_loss(u_bnd_exact, u_bnd)
            loss = self._loss_residual_weight * loss_residual + self._loss_initial_weight * loss_initial + \
                self._loss_boundary_weight * loss_boundary

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_total_tracker.update_state(loss)
        self.mae_tracker.update_state(u_samples_exact, u_samples)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_initial_tracker.update_state(loss_initial)
        self.loss_boundary_tracker.update_state(loss_boundary)

        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        """
        Performs a test step.

        Args:
            data: The data to train on. First input is the samples, second input is the initial, \
                and third input is the boundary data. First output is the exact solution for the samples, \
                second output is the exact rhs for the samples, third output is the exact solution for the initial, \
                and fourth output is the exact solution for the boundary.
        """

        inputs, outputs = data
        u_samples_exact, rhs_samples_exact, u_initial_exact, u_bnd_exact = outputs

        u_samples, lhs_samples, u_initial, u_bnd = self(inputs, training=False)

        loss_residual = self.res_loss(rhs_samples_exact, lhs_samples)
        loss_initial = self.init_loss(u_initial_exact, u_initial)
        loss_boundary = self.bnd_loss(u_bnd_exact, u_bnd)
        loss = self._loss_residual_weight * loss_residual + self._loss_initial_weight * loss_initial + \
            self._loss_boundary_weight * loss_boundary

        self.loss_total_tracker.update_state(loss)
        self.mae_tracker.update_state(u_samples_exact, u_samples)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_initial_tracker.update_state(loss_initial)
        self.loss_boundary_tracker.update_state(loss_boundary)

        return {m.name: m.result() for m in self.metrics}
    
    def fit_custom(self, inputs: List['tf.Tensor'], outputs: List['tf.Tensor'], epochs: int, print_every: int = 1000):
        '''
        Custom alternative to tensorflow fit function, mainly to allow inputs with different sizes. Training is done in full batches.

        Args:
            data: The data to train on. Should be a list of tensors: [tx_colloc, tx_init, tx_bnd]
            outputs: The outputs to train on. Should be a list of tensors: [u_colloc, residual, u_init, u_bnd]. u_colloc is only used for the MAE metric.
            epochs: The number of epochs to train for.
            print_every: How often to print the metrics. Defaults to 1000.
        '''
        history = create_history_dictionary()
        
        for epoch in range(epochs):
            metrs = self.train_step([inputs, outputs])
            for key, value in metrs.items():
                history[key].append(value.numpy())

            if epoch % print_every == 0:
                tf.print(f"Epoch {epoch}, Loss Residual: {metrs['loss_residual']:0.4f}, Loss Initial: {metrs['loss_initial']:0.4f}, Loss Boundary: {metrs['loss_boundary']:0.4f}, MAE: {metrs['mean_absolute_error']:0.4f}")
                
            #reset metrics
            for m in self.metrics:
                m.reset_states()

        return history

    @property
    def metrics(self):
        """
        Returns the metrics of the model.
        """
        return [self.loss_total_tracker, self.loss_residual_tracker, self.loss_initial_tracker, self.loss_boundary_tracker, self.mae_tracker]

class WavePinn(tf.keras.Model):
    """
    A model that solves the wave equation.
    """

    def __init__(self, backbone: "tf.keras.Model", c: float, loss_residual_weight=1.0, loss_initial_weight=1.0, \
        loss_boundary_weight=1.0, **kwargs):
        """
        Initializes the model.

        Args:
            backbone: The backbone model.
            c: The wave speed.
            loss_residual_weight: The weight of the residual loss.
            loss_initial_weight: The weight of the initial loss.
            loss_boundary_weight: The weight of the boundary loss.
        """
        super().__init__(**kwargs)
        self.backbone = backbone
        self.c = c
        self.loss_total_tracker = tf.keras.metrics.Mean(name=LOSS_TOTAL)
        self.loss_residual_tracker = tf.keras.metrics.Mean(name=LOSS_RESIDUAL)
        self.loss_initial_tracker = tf.keras.metrics.Mean(name=LOSS_INITIAL)
        self.loss_boundary_tracker = tf.keras.metrics.Mean(name=LOSS_BOUNDARY)
        self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name=MEAN_ABSOLUTE_ERROR)
        self._loss_residual_weight = tf.Variable(loss_residual_weight, trainable=False, name="loss_residual_weight", dtype=tf.keras.backend.floatx())
        self._loss_boundary_weight = tf.Variable(loss_boundary_weight, trainable=False, name="loss_boundary_weight", dtype=tf.keras.backend.floatx())
        self._loss_initial_weight = tf.Variable(loss_initial_weight, trainable=False, name="loss_initial_weight", dtype=tf.keras.backend.floatx())
        self.res_loss = tf.keras.losses.MeanSquaredError()
        self.init_loss = tf.keras.losses.MeanSquaredError()
        self.bnd_loss = tf.keras.losses.MeanSquaredError()

    def set_loss_weights(self, loss_residual_weight: float, loss_initial_weight: float, loss_boundary_weight: float):
        """
        Sets the loss weights.

        Args:
            loss_residual_weight: The weight of the residual loss.
            loss_initial_weight: The weight of the initial loss.
            loss_boundary_weight: The weight of the boundary loss.
        """
        self._loss_residual_weight.assign(loss_residual_weight)
        self._loss_boundary_weight.assign(loss_boundary_weight)
        self._loss_initial_weight.assign(loss_initial_weight)

    @tf.function
    def call(self, inputs, training=False):
        """
        Performs a forward pass.

        Args:
            inputs: The inputs to the model. First input is the samples, second input is the initial, \
                and third input is the boundary data.
            training: Whether or not the model is training.

        Returns:
            The solution for the residual samples, the lhs residual, the solution for the initial samples, \
                and the solution for the boundary samples.
        """

        tx_samples = inputs[0]
        tx_init = inputs[1]
        tx_bnd = inputs[2]

        with tf.GradientTape(watch_accessed_variables=False) as tape2:
            tape2.watch(tx_samples)

            with tf.GradientTape(watch_accessed_variables=False) as tape1:
                tape1.watch(tx_samples)
                u_samples = self.backbone(tx_samples, training=training)

            first_order = tape1.batch_jacobian(u_samples, tx_samples)
        second_order = tape2.batch_jacobian(first_order, tx_samples)
        d2u_dt2 = second_order[..., 0, 0]
        d2u_dx2 = second_order[..., 1, 1]
        lhs_samples = d2u_dt2 - (self.c ** 2) * d2u_dx2

        u_bnd = self.backbone(tx_bnd, training=training)

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(tx_init)
            u_initial = self.backbone(tx_init, training=training)
        du_dt_init = tape.batch_jacobian(u_initial, tx_init)[..., 0]

        return u_samples, lhs_samples, u_initial, du_dt_init, u_bnd

    @tf.function
    def train_step(self, data):
        """
        Performs a training step.

        Args:
            data: The data to train on. First input is the samples, second input is the initial, \
                and third input is the boundary data. The outputs are the exact solutions for the samples, \
                the exact rhs for the samples, the exact solution for the initial, the exact derivative for the initial, \
                and the exact solution for the boundary.
        """

        inputs, outputs = data
        u_samples_exact, rhs_samples_exact, u_initial_exact, du_dt_init_exact, u_bnd_exact = outputs

        with tf.GradientTape() as tape:
            u_samples, lhs_samples, u_initial, du_dt_init, u_bnd = self(inputs, training=True)
            loss_residual = self.res_loss(rhs_samples_exact, lhs_samples)
            loss_initial_neumann = self.init_loss(du_dt_init_exact, du_dt_init)
            loss_initial_dirichlet = self.init_loss(u_initial_exact, u_initial)
            loss_initial = loss_initial_neumann + loss_initial_dirichlet
            loss_boundary = self.bnd_loss(u_bnd_exact, u_bnd)
            loss = self._loss_residual_weight * loss_residual + self._loss_initial_weight * loss_initial + \
                self._loss_boundary_weight * loss_boundary

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        self.loss_total_tracker.update_state(loss)
        self.mae_tracker.update_state(u_samples_exact, u_samples)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_initial_tracker.update_state(loss_initial)
        self.loss_boundary_tracker.update_state(loss_boundary)

        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        """
        Performs a test step.
        """
        inputs, outputs = data
        u_samples_exact, rhs_samples_exact, u_initial_exact, du_dt_init_exact, u_bnd_exact = outputs

        u_samples, lhs_samples, u_initial, du_dt_init, u_bnd = self(inputs, training=False)
        loss_residual = self.res_loss(rhs_samples_exact, lhs_samples)
        loss_initial_neumann = self.init_loss(du_dt_init_exact, du_dt_init)
        loss_initial_dirichlet = self.init_loss(u_initial_exact, u_initial)
        loss_initial = loss_initial_neumann + loss_initial_dirichlet
        loss_boundary = self.bnd_loss(u_bnd_exact, u_bnd)
        loss = self._loss_residual_weight * loss_residual + self._loss_initial_weight * loss_initial + \
            self._loss_boundary_weight * loss_boundary

        self.loss_total_tracker.update_state(loss)
        self.mae_tracker.update_state(u_samples_exact, u_samples)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_initial_tracker.update_state(loss_initial)
        self.loss_boundary_tracker.update_state(loss_boundary)

        return {m.name: m.result() for m in self.metrics}
    
    def fit_custom(self, inputs: List['tf.Tensor'], outputs: List['tf.Tensor'], epochs: int, print_every: int = 1000):
        '''
        Custom alternative to tensorflow fit function, mainly to allow inputs with different sizes. Training is done in full batches.

        Args:
            data: The data to train on. Should be a list of tensors: [tx_colloc, tx_init, tx_bnd]
            outputs: The outputs to train on. Should be a list of tensors: [u_colloc, residual, u_init, u_bnd]. u_colloc is only used for the MAE metric.
            epochs: The number of epochs to train for.
            print_every: How often to print the metrics. Defaults to 1000.
        '''
        history = create_history_dictionary()
        
        for epoch in range(epochs):
            metrs = self.train_step([inputs, outputs])
            for key, value in metrs.items():
                history[key].append(value.numpy())

            if epoch % print_every == 0:
                tf.print(f"Epoch {epoch}, Loss Residual: {metrs['loss_residual']:0.4f}, Loss Initial: {metrs['loss_initial']:0.4f}, Loss Boundary: {metrs['loss_boundary']:0.4f}, MAE: {metrs['mean_absolute_error']:0.4f}")
                
            #reset metrics
            for m in self.metrics:
                m.reset_states()

        return history

    @property
    def metrics(self):
        """
        Returns the metrics of the model.
        """
        return [self.loss_total_tracker, self.loss_residual_tracker, self.loss_initial_tracker, self.loss_boundary_tracker, self.mae_tracker]

class ReactionDiffusionPinn(tf.keras.Model):
    """
    A PINN for the reaction diffusion equation.

    Attributes:
        backbone: The backbone model.
        _nu: The diffusion coefficient.
        _R: The reaction function.
        _loss_residual_weight: The weight of the residual loss.
        _loss_initial_weight: The weight of the initial loss.
        _loss_boundary_weight: The weight of the boundary loss.
        loss_residual_tracker: The residual loss tracker.
        loss_initial_tracker: The initial loss tracker.
        loss_boundary_tracker: The boundary loss tracker.
        mae_tracker: The mean absolute error tracker.
    """

    def __init__(self, backbone: tf.keras.Model, nu: float, reaction_function: Callable[[tf.Tensor], tf.Tensor] = None,
                 loss_residual_weight: float = 1.0, loss_initial_weight: float = 1.0,
                 loss_boundary_weight: float = 1.0, **kwargs):
        """
        Initializes the model.

        Args:
            backbone: The backbone model.
            nu: The diffusion coefficient.
            reaction_function: The reaction function. If None, Fisher's equation with rho=1 is used. Defaults to None.
            loss_residual_weight: The weight of the residual loss
            loss_initial_weight: The weight of the initial loss.
            loss_boundary_weight: The weight of the boundary loss.
        """

        super().__init__(**kwargs)
        self.backbone = backbone
        self._nu = nu
        self._R = reaction_function if reaction_function is not None else \
            ReactionDiffusionPinn.get_fisher_reaction_function()
        self.loss_total_tracker = tf.keras.metrics.Mean(name=LOSS_TOTAL)
        self.loss_residual_tracker = tf.keras.metrics.Mean(name=LOSS_RESIDUAL)
        self.loss_initial_tracker = tf.keras.metrics.Mean(name=LOSS_INITIAL)
        self.loss_boundary_tracker = tf.keras.metrics.Mean(name=LOSS_BOUNDARY)
        self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name=MEAN_ABSOLUTE_ERROR)
        self._loss_residual_weight = tf.Variable(loss_residual_weight, trainable=False, name="loss_residual_weight",
                                                 dtype=tf.keras.backend.floatx())
        self._loss_boundary_weight = tf.Variable(loss_boundary_weight,
                                                 trainable=False, name="loss_boundary_weight",
                                                 dtype=tf.keras.backend.floatx())
        self._loss_initial_weight = tf.Variable(loss_initial_weight, trainable=False, name="loss_initial_weight",
                                                dtype=tf.keras.backend.floatx())
        self.res_loss = tf.keras.losses.MeanSquaredError()
        self.bnd_loss = tf.keras.losses.MeanSquaredError()
        self.init_loss = tf.keras.losses.MeanSquaredError()

    @staticmethod
    def get_fishers_reaction_function(rho: float = 1.0):
        """
        Returns the Fisher's reaction function.
        """
        @tf.function
        def reaction_function(u: tf.Tensor) -> tf.Tensor:
            return rho * u * (1 - u)
        return reaction_function

    @tf.function
    def call(self, inputs, training=False):
        """
        Performs a forward pass.

        Args:
            inputs: The inputs to the model. First input is the samples, second input is the initial, \
                and third input is the boundary data.
            training: Whether or not the model is training.

        Returns:
            The solution for the residual samples, the lhs residual, the solution for the initial samples, \
                and the solution for the boundary samples.
        """
        tx_samples = inputs[0]
        tx_init = inputs[1]
        tx_bnd = inputs[2]

        with tf.GradientTape(watch_accessed_variables=False) as tape2:
            tape2.watch(tx_samples)
            
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(tx_samples)
                u_samples = self.backbone(tx_samples, training=training)
            first_order = tape.batch_jacobian(u_samples, tx_samples)
            du_dt = first_order[..., 0]
            du_dx = first_order[..., 1]
        d2u_dx2 = tape2.batch_jacobian(du_dx, tx_samples)[..., 1]
        residual = du_dt - self._nu * d2u_dx2 - self._R(u_samples)

        tx_bi = tf.concat([tx_init, tx_bnd], axis=0)
        u_bi = self.backbone(tx_bi, training=training)
        u_init = u_bi[:tf.shape(tx_init)[0]]
        u_bnd = u_bi[tf.shape(tx_init)[0]:]

        return u_samples, residual, u_init, u_bnd

    @tf.function
    def train_step(self, data):
        """
        Performs a training step.

        Args:
            data: The data to train on.

        Returns:
            The metrics of the model.
        """
        x, y = data
        u_exact_colloc, residual_exact, u_initial_exact, u_bnd_exact = y

        with tf.GradientTape() as tape:
            u_colloc, residual, u_init, u_bnd = self(x, training=True)
            loss_res = self.res_loss(residual_exact, residual)
            loss_init = self.init_loss(u_initial_exact, u_init)
            loss_bnd = self.bnd_loss(u_bnd_exact, u_bnd)
            loss = self._loss_residual_weight * loss_res + self._loss_initial_weight * loss_init + \
                     self._loss_boundary_weight * loss_bnd

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_total_tracker.update_state(loss)
        self.mae_tracker.update_state(u_exact_colloc, u_colloc)
        self.loss_residual_tracker.update_state(loss_res)
        self.loss_initial_tracker.update_state(loss_init)
        self.loss_boundary_tracker.update_state(loss_bnd)

        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        """
        Performs a test step.

        Args:
            data: The data to test on.

        Returns:
            The metrics of the model.
        """
        x, y = data
        u_exact_colloc, residual_exact, u_initial_exact, u_bnd_exact = y

        u_colloc, residual, u_init, u_bnd = self(x, training=False)
        loss_res = self.res_loss(residual_exact, residual)
        loss_init = self.init_loss(u_initial_exact, u_init)
        loss_bnd = self.bnd_loss(u_bnd_exact, u_bnd)
        loss = self._loss_residual_weight * loss_res + self._loss_initial_weight * loss_init + \
                 self._loss_boundary_weight * loss_bnd

        self.loss_total_tracker.update_state(loss)
        self.mae_tracker.update_state(u_exact_colloc, u_colloc)
        self.loss_residual_tracker.update_state(loss_res)
        self.loss_initial_tracker.update_state(loss_init)
        self.loss_boundary_tracker.update_state(loss_bnd)

        return {m.name: m.result() for m in self.metrics}
    
    def fit_custom(self, inputs: List['tf.Tensor'], outputs: List['tf.Tensor'], epochs: int, print_every: int = 1000):
        '''
        Custom alternative to tensorflow fit function, mainly to allow inputs with different sizes. Training is done in full batches.

        Args:
            data: The data to train on. Should be a list of tensors: [tx_colloc, tx_init, tx_bnd]
            outputs: The outputs to train on. Should be a list of tensors: [u_colloc, residual, u_init, u_bnd]. u_colloc is only used for the MAE metric.
            epochs: The number of epochs to train for.
            print_every: How often to print the metrics. Defaults to 1000.
        '''
        history = create_history_dictionary()
        
        for epoch in range(epochs):
            metrs = self.train_step([inputs, outputs])
            for key, value in metrs.items():
                history[key].append(value.numpy())

            if epoch % print_every == 0:
                tf.print(f"Epoch {epoch}, Loss Residual: {metrs['loss_residual']:0.4f}, Loss Initial: {metrs['loss_initial']:0.4f}, Loss Boundary: {metrs['loss_boundary']:0.4f}, MAE: {metrs['mean_absolute_error']:0.4f}")
                
            #reset metrics
            for m in self.metrics:
                m.reset_states()

        return history
        
    @property
    def metrics(self):
        """
        Returns the metrics of the model.
        """
        return [self.loss_total_tracker, self.loss_residual_tracker, self.loss_initial_tracker, self.loss_boundary_tracker, self.mae_tracker]

class KleinGordonEquation(tf.keras.Model):
    '''
    Class for the Klein Gordon Equation.
    '''
    def __init__(self, backbone: "tf.keras.Model", alpha: float = -1.0, beta: float = 0.0, gamma: float = 1.0, k: int = 3, \
                 loss_residual_weight: float = 1.0, loss_initial_weight: float = 1.0, loss_boundary_weight: float = 1.0, \
                    *args, **kwargs):
        '''
        Args:
            backbone: The backbone model.
            alpha: The alpha parameter. Defaults to -1.0.
            beta: The beta parameter. Defaults to 0.
            gamma: The gamma parameter. Defaults to 1.
            k: The k parameter. 2 for quadratic and 3 for cubic nonlinearity.
        '''
        super().__init__(*args, **kwargs)
        self.backbone = backbone
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._k = k
        self._loss_residual_weight = tf.Variable(loss_residual_weight, trainable=False, dtype=tf.keras.backend.floatx(), \
                                                 name="loss_residual_weight")
        self._loss_initial_weight = tf.Variable(loss_initial_weight, trainable=False, dtype=tf.keras.backend.floatx(), \
                                                name="loss_initial_weight")
        self._loss_boundary_weight = tf.Variable(loss_boundary_weight, trainable=False, dtype=tf.keras.backend.floatx(), \
                                                 name="loss_boundary_weight")
        self.loss_total_tracker = tf.keras.metrics.Mean(name=LOSS_TOTAL)
        self.loss_residual_tracker = tf.keras.metrics.Mean(name=LOSS_RESIDUAL)
        self.loss_initial_tracker = tf.keras.metrics.Mean(name=LOSS_INITIAL)
        self.loss_boundary_tracker = tf.keras.metrics.Mean(name=LOSS_BOUNDARY)
        self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name=MEAN_ABSOLUTE_ERROR)
        self.res_loss = tf.keras.losses.MeanSquaredError()
        self.init_loss = tf.keras.losses.MeanSquaredError()
        self.bnd_loss = tf.keras.losses.MeanSquaredError()

    def set_loss_weights(self, loss_residual_weight: float, loss_initial_weight: float, loss_boundary_weight: float):
        """
        Sets the loss weights.

        Args:
            loss_residual_weight: The weight of the residual loss.
            loss_initial_weight: The weight of the initial loss.
            loss_boundary_weight: The weight of the boundary loss.
        """
        self._loss_residual_weight.assign(loss_residual_weight)
        self._loss_boundary_weight.assign(loss_boundary_weight)
        self._loss_initial_weight.assign(loss_initial_weight)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        """
        Performs a forward pass.

        Args:
            inputs: The inputs to the model. Should be a list of tensors: [tx_colloc, tx_init, tx_bnd]
            training: Whether the model is in training mode or not. Defaults to None.
            mask: The mask to apply. Defaults to None.

        Returns:
            The outputs of the model. Should be a list of tensors: [u_colloc, residual, u_init, u_t_init, u_bnd]
        """
        tx_colloc, tx_init, tx_bnd = inputs
        
        with tf.GradientTape(watch_accessed_variables=False) as tape2:
            tape2.watch(tx_colloc)
            
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(tx_colloc)
                u_colloc = self.backbone(tx_colloc, training=training)
            first_order = tape.batch_jacobian(u_colloc, tx_colloc)
        second_order = tape2.batch_jacobian(first_order, tx_colloc)
        u_tt = second_order[..., 0, 0]
        u_xx = second_order[..., 1, 1]
        residual = u_tt + self._alpha * u_xx + self._beta * u_colloc + self._gamma * u_colloc ** self._k

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(tx_init)
            u_init = self.backbone(tx_init, training=training)
        u_t_init = tape.batch_jacobian(u_init, tx_init)[..., 0]

        u_bnd = self.backbone(tx_bnd, training=training)

        return [u_colloc, residual, u_init, u_t_init, u_bnd]
    
    @tf.function
    def train_step(self, data):
        """
        Performs a training step.

        Args:
            data: The data to train on. Should be a list of tensors: [inputs, outputs]

        Returns:
            The metrics of the model.
        """
        inputs, outputs = data
        u_colloc, residual, u_init, u_t_init, u_bnd = outputs
        
        with tf.GradientTape() as tape:
            u_colloc_pred, residual_pred, u_init_pred, u_t_init_pred, u_bnd_pred = self(inputs, training=True)
            loss_residual = self.res_loss(residual, residual_pred)
            loss_initial = self.init_loss(u_init, u_init_pred) + self.init_loss(u_t_init, u_t_init_pred)
            loss_boundary = self.bnd_loss(u_bnd, u_bnd_pred)
            loss = self._loss_residual_weight * loss_residual + self._loss_initial_weight * loss_initial + self._loss_boundary_weight * loss_boundary
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        self.loss_total_tracker.update_state(loss)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_initial_tracker.update_state(loss_initial)
        self.loss_boundary_tracker.update_state(loss_boundary)
        self.mae_tracker.update_state(u_colloc, u_colloc_pred)
        
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        """
        Performs a test step.
        """
        
        inputs, outputs = data
        u_colloc, residual, u_init, u_t_init, u_bnd = outputs
        
        u_colloc_pred, residual_pred, u_init_pred, u_t_init_pred, u_bnd_pred = self(inputs, training=True)
        loss_residual = self.res_loss(residual, residual_pred)
        loss_initial = self.init_loss(u_init, u_init_pred) + self.init_loss(u_t_init, u_t_init_pred)
        loss_boundary = self.bnd_loss(u_bnd, u_bnd_pred)
        loss = self._loss_residual_weight * loss_residual + self._loss_initial_weight * loss_initial + self._loss_boundary_weight * loss_boundary
        
        self.loss_total_tracker.update_state(loss)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_initial_tracker.update_state(loss_initial)
        self.loss_boundary_tracker.update_state(loss_boundary)
        self.mae_tracker.update_state(u_colloc, u_colloc_pred)    

        return {m.name: m.result() for m in self.metrics}

    def fit_custom(self, inputs: List['tf.Tensor'], outputs: List['tf.Tensor'], epochs: int, print_every: int = 1000):
        '''
        Custom alternative to tensorflow fit function, mainly to allow inputs with different sizes. Training is done in full batches.

        Args:
            data: The data to train on. Should be a list of tensors: [tx_colloc, tx_init, tx_bnd]
            outputs: The outputs to train on. Should be a list of tensors: [u_colloc, residual, u_init, u_t_init, u_bnd]. u_colloc is only used for the MAE metric.
            epochs: The number of epochs to train for.
            print_every: How often to print the metrics. Defaults to 1000.
        '''
        history = create_history_dictionary()
        
        for epoch in range(epochs):
            metrs = self.train_step([inputs, outputs])
            for key, value in metrs.items():
                history[key].append(value.numpy())

            if epoch % print_every == 0:
                tf.print(f"Epoch {epoch}, Loss Residual: {metrs['loss_residual']:0.4f}, Loss Initial: {metrs['loss_initial']:0.4f}, Loss Boundary: {metrs['loss_boundary']:0.4f}, MAE: {metrs['mean_absolute_error']:0.4f}")
                
            #reset metrics
            for m in self.metrics:
                m.reset_states()

        return history
    
    @property
    def metrics(self):
        return [self.loss_total_tracker, self.loss_residual_tracker, self.loss_initial_tracker, self.loss_boundary_tracker, self.mae_tracker]
    
class TransportEquation(tf.keras.models.Model):
    """
    One dimensional Transport (Convection) Equation model.
    """

    def __init__(self, backbone, beta, loss_residual_weight=1., loss_initial_weight=1., loss_boundary_weight=1., **kwargs):
        """
        One dimensional Transport (Convection) Equation model.

        Args:
            backbone: The backbone of the model. Should be a tf.keras.Model.
            beta: The convection coefficient.
            loss_residual_weight: The weight of the residual loss. Defaults to 1.
            loss_initial_weight: The weight of the initial loss. Defaults to 1.
            loss_boundary_weight: The weight of the boundary loss. Defaults to 1.
            kwargs: Additional arguments to pass to the tf.keras.Model constructor.
        """
        super().__init__(**kwargs)
        self.backbone = backbone
        self._beta = beta
        self._loss_residual_weight = tf.Variable(loss_residual_weight, trainable=False, dtype=tf.keras.backend.floatx(), \
                                                 name="loss_residual_weight")
        self._loss_initial_weight = tf.Variable(loss_initial_weight, trainable=False, dtype=tf.keras.backend.floatx(), \
                                                name="loss_initial_weight")
        self._loss_boundary_weight = tf.Variable(loss_boundary_weight, trainable=False, dtype=tf.keras.backend.floatx(), \
                                                 name="loss_boundary_weight")
        self.loss_total_tracker = tf.keras.metrics.Mean(name=LOSS_TOTAL)
        self.loss_residual_tracker = tf.keras.metrics.Mean(name=LOSS_RESIDUAL)
        self.loss_initial_tracker = tf.keras.metrics.Mean(name=LOSS_INITIAL)
        self.loss_boundary_tracker = tf.keras.metrics.Mean(name=LOSS_BOUNDARY)
        self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name=MEAN_ABSOLUTE_ERROR)
        self.res_loss = tf.keras.losses.MeanSquaredError()
        self.init_loss = tf.keras.losses.MeanSquaredError()
        self.bnd_loss = tf.keras.losses.MeanSquaredError()

    def set_loss_weights(self, loss_residual_weight, loss_initial_weight, loss_boundary_weight):
        """
        Sets the loss weights.

        Args:
            loss_residual_weight: The weight of the residual loss. Defaults to 1.
            loss_initial_weight: The weight of the initial loss. Defaults to 1.
            loss_boundary_weight: The weight of the boundary loss. Defaults to 1.
        """
        self._loss_residual_weight.assign(loss_residual_weight)
        self._loss_initial_weight.assign(loss_initial_weight)
        self._loss_boundary_weight.assign(loss_boundary_weight)

    @tf.function
    def call(self, inputs, training=False):
        """
        Performs a forward pass through the model.

        Args:
            inputs: The inputs to the model. Should be a list of tensors: [tx_colloc, tx_init, tx_bnd]
            training: Whether to run in training mode. Defaults to False.

        Returns:
            A list of tensors: [u_colloc, residual, u_init, u_t_init, u_bnd]
        """
        tx_colloc, tx_init, tx_bnd_start, tx_bnd_end = inputs
        
        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
            tape.watch(tx_colloc)
            u_colloc = self.backbone(tx_colloc, training=training)
        first_order = tape.batch_jacobian(u_colloc, tx_colloc)
        u_t_colloc = first_order[..., 0]
        u_x_colloc = first_order[..., 1]
        residual = u_t_colloc + self._beta * u_x_colloc

        u_init = self.backbone(tx_init, training=training)

        tx_bnd = tf.concat([tx_bnd_start, tx_bnd_end], axis=0)
        u_bnd = self.backbone(tx_bnd, training=training)
        u_bnd_start = u_bnd[:tf.shape(tx_bnd_start)[0]]
        u_bnd_end = u_bnd[tf.shape(tx_bnd_start)[0]:]

        return u_colloc, residual, u_init, u_bnd_start, u_bnd_end
    
    @tf.function
    def train_step(self, data):
        """
        Performs a training step.

        Args:
            data: The data to train on. Should be a list of inputs and outputs: \
                [x, y], where x is a list of tensors: [tx_colloc, tx_init, tx_bnd] and y is \
                    a list of tensors: [u_colloc, residual, u_init, u_bnd]
        """

        x, y = data
        tx_colloc, tx_init, tx_bnd_start, tx_bnd_end = x
        u_colloc, residual, u_init = y
        with tf.GradientTape(persistent=False) as tape:
            u_colloc_pred, residual_pred, u_init_pred, u_bnd_start_pred, u_bnd_end_pred = self(
                [tx_colloc, tx_init, tx_bnd_start, tx_bnd_end], 
                training=True)
            loss_residual = self.res_loss(residual, residual_pred)
            loss_initial = self.init_loss(u_init, u_init_pred)
            loss_boundary = self.bnd_loss(u_bnd_start_pred, u_bnd_end_pred)
            loss_total = self._loss_residual_weight * loss_residual + self._loss_initial_weight * loss_initial \
                + self._loss_boundary_weight * loss_boundary
        gards = tape.gradient(loss_total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gards, self.trainable_variables))

        self.loss_initial_tracker.update_state(loss_initial)
        self.loss_boundary_tracker.update_state(loss_boundary)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_total_tracker.update_state(loss_total)
        self.mae_tracker.update_state(u_colloc, u_colloc_pred)

        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        """
        Performs a test step.

        Args:
            data: The data to test on. Should be a list of inputs and outputs: \
                [x, y], where x is a list of tensors: [tx_colloc, tx_init, tx_bnd] and y is \
                    a list of tensors: [u_colloc, residual, u_init, u_bnd]
        """

        x, y = data
        tx_colloc, tx_init, tx_bnd_start, tx_bnd_end = x
        u_colloc, residual, u_init = y
        u_colloc_pred, residual_pred, u_init_pred, u_bnd_start, u_bnd_end = self(
            [tx_colloc, tx_init, tx_bnd_start, tx_bnd_end])
        loss_residual = self.res_loss(residual, residual_pred)
        loss_initial = self.init_loss(u_init, u_init_pred)
        loss_boundary = self.bnd_loss(u_bnd_start, u_bnd_end)
        loss_total = self._loss_residual_weight * loss_residual + self._loss_initial_weight * loss_initial \
            + self._loss_boundary_weight * loss_boundary

        self.loss_initial_tracker.update_state(loss_initial)
        self.loss_boundary_tracker.update_state(loss_boundary)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_total_tracker.update_state(loss_total)
        self.mae_tracker.update_state(u_colloc, u_colloc_pred)

        return {m.name: m.result() for m in self.metrics}
    
    def fit_custom(self, inputs: List['tf.Tensor'], outputs: List['tf.Tensor'], epochs: int, print_every: int = 1000):
        """
        Custom alternative to tensorflow fit function, mainly to allow inputs with different sizes. Training is done in full batches.
        
        Args:
            inputs: The inputs to the model. Should be a list of tensors: [tx_colloc, tx_init, tx_bnd]
            outputs: The outputs to the model. Should be a list of tensors: [u_colloc, residual, u_init, u_bnd]
            epochs: The number of epochs to train for.
            print_every: The number of epochs between printing the loss. Defaults to 1000.

        Returns:
            A dictionary containing the history of the training.
        """

        history = create_history_dictionary()
        
        for epoch in range(epochs):
            metrs = self.train_step([inputs, outputs])
            for key, value in metrs.items():
                history[key].append(value.numpy())

            if epoch % print_every == 0:
                tf.print(f"Epoch {epoch}, Loss Residual: {metrs['loss_residual']:0.4f}, Loss Initial: {metrs['loss_initial']:0.4f}, Loss Boundary: {metrs['loss_boundary']:0.4f}, MAE: {metrs['mean_absolute_error']:0.4f}")
                
            #reset metrics
            for m in self.metrics:
                m.reset_states()

        return history

    @property
    def metrics(self):
        """
        Returns the metrics to track.
        """
        return [self.loss_total_tracker, self.loss_residual_tracker, self.loss_initial_tracker, \
            self.loss_boundary_tracker, self.mae_tracker]

class travelKawaharaPINN(tf.keras.models.Model):

    def __init__(self, backbone, c: float = 0. , alpha: float = 1.0, beta: float = 1/4 ,sigma: float = 1.0, loss_residual_weight=1., loss_initial_weight=1., loss_boundary_weight=1.,loss_hamil_weight=1., PBC = True, **kwargs):
        """
        Travelling Kawahara model.

        Args:
            backbone: The backbone of the model. Should be a tf.keras.Model.
            c: travelling wave speed (deprecated)
            loss_residual_weight: The weight of the residual loss. Defaults to 1.
            loss_initial_weight: The weight of the initial loss. Defaults to 1.
            loss_boundary_weight: The weight of the boundary loss. Defaults to 1.
            kwargs: Additional arguments to pass to the tf.keras.Model constructor.
        """
        super().__init__(**kwargs)
        self.backbone = backbone

        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta
        self.c = c
        self.PBC = PBC


        self._loss_residual_weight = tf.Variable(loss_residual_weight, trainable=False, dtype=tf.keras.backend.floatx(), \
                                                 name="loss_residual_weight")
        self._loss_initial_weight = tf.Variable(loss_initial_weight, trainable=False, dtype=tf.keras.backend.floatx(), \
                                                name="loss_initial_weight")
        self._loss_boundary_weight = tf.Variable(loss_boundary_weight, trainable=False, dtype=tf.keras.backend.floatx(), \
                                                 name="loss_boundary_weight")
        self._loss_hamil_weight = tf.Variable(loss_hamil_weight, trainable=False, dtype=tf.keras.backend.floatx(), \
                                                 name="loss_hamil_weight")
        # self._loss_dudt_weight = tf.Variable(loss_dudt_weight, trainable=False, dtype=tf.keras.backend.floatx(), \
                                                #  name="loss_dudt_weight")
        self.loss_total_tracker = tf.keras.metrics.Mean(name=LOSS_TOTAL)
        self.loss_residual_tracker = tf.keras.metrics.Mean(name=LOSS_RESIDUAL)
        self.loss_initial_tracker = tf.keras.metrics.Mean(name=LOSS_INITIAL)
        self.loss_boundary_tracker = tf.keras.metrics.Mean(name=LOSS_BOUNDARY)
        self.loss_hamil_tracker = tf.keras.metrics.Mean(name=LOSS_HAMIL)
        # self.loss_dudt_tracker = tf.keras.metrics.Mean(name=LOSS_DUDT)
        self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name=MEAN_ABSOLUTE_ERROR)
        self.res_loss = tf.keras.losses.MeanSquaredError()
        self.init_loss = tf.keras.losses.MeanSquaredError()
        self.bnd_loss = tf.keras.losses.MeanSquaredError()
        self.hamil_loss = tf.keras.losses.MeanSquaredError()
        # self.dudt_loss = tf.keras.losses.MeanSquaredError()

    def set_loss_weights(self, loss_residual_weight, loss_initial_weight, loss_boundary_weight, loss_hamil_weight):
        """
        Sets the loss weights.

        Args:
            loss_residual_weight: The weight of the residual loss. Defaults to 1.
            loss_initial_weight: The weight of the initial loss. Defaults to 1.
            loss_boundary_weight: The weight of the boundary loss. Defaults to 1.
            loss_dudt_weight: The weight of the du_dt loss. Defaults to 1.
        """
        self._loss_residual_weight.assign(loss_residual_weight)
        self._loss_initial_weight.assign(loss_initial_weight)
        self._loss_boundary_weight.assign(loss_boundary_weight)
        self._loss_hamil_weight.assign(loss_hamil_weight)
        # self._loss_dudt_weight.assign(loss_dudt_weight)

    @tf.function
    def call(self, inputs, training=False):
        """
        Performs a forward pass through the model.

        Args:
            inputs: The inputs to the model. Should be a list of tensors: [tx_colloc, tx_init, tx_bnd]
            training: Whether to run in training mode. Defaults to False.

        Returns:
            A list of tensors: [u_colloc, residual, u_init, u_bnd]
        """
        tx_colloc = inputs[0]
        tx_init = inputs[1]
        tx_bnd_start = inputs[2]
        tx_bnd_end = inputs[3]
        tx_bnd = inputs[4]
        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape5:
            with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape4:
                with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape3:
                    with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape2:
                        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
                            tape.watch(tx_colloc)
                            u_colloc = self.backbone(tx_colloc, training=training)
                        first_order = tape.batch_jacobian(u_colloc, tx_colloc)
                        u_t = first_order[..., 0]
                        u_x = first_order[..., 1]
                    u_xx = tape2.batch_jacobian(u_x, tx_colloc)[..., 1]
                u_xxx = tape3.batch_jacobian(u_xx, tx_colloc)[..., 1]
            u_4x = tape4.batch_jacobian(u_xxx, tx_colloc)[..., 1]
        u_5x = tape5.batch_jacobian(u_4x, tx_colloc)[..., 1]
        
        
        
        # residual = + self.alpha * u_xxx + (self.beta) * u_5x + (self.sigma * 2 * u_colloc + self.c)* u_x 
        # residual = u_xxx + u_x + tf.math.cos(tx_colloc[:,1:])
        # residual = + self.alpha * u_xxx - self.c * u_x 
        # amplitude = tf.math.abs(tf.reduce_max(u_colloc)-tf. reduce_min(u_colloc))
        residual = + self.alpha * u_xxx + (self.beta) * u_5x + (self.sigma * 2 * u_colloc) * u_x + (self.c * u_x - u_t)
        u_init = self.backbone(tx_init, training=training)
        # residual = self.backbone(tx_colloc, training = training) 
        u_bnd_start = self.backbone(tx_bnd_start, training=training)
        u_bnd_end = self.backbone(tx_bnd_end, training=training)
        u_bnd = self.backbone(tx_bnd, training=training)
        u_hamil_integrand = -self.alpha/2. * tf.math.square(u_x) +self.beta/2. * tf.math.square(u_xx) + self.sigma/3. * tf.math.square(u_colloc) * u_colloc
        hamil_term = tf.reduce_sum(u_hamil_integrand, 0)
        
        # integrand = -self.alpha/2 * u_x**2 + self.beta/2 * u_xx**2 + self.sigma/3 * u_colloc**3
        
        # integral = 0 
        # for i in range(1,len(tx_colloc[:,1:]-1)):
        #     integral += (integrand[i-1]+integrand[i])/2 * dx_hamil
        


        return u_colloc, residual, u_init, u_bnd_start, u_bnd_end, u_bnd, hamil_term
    
    @tf.function
    def train_step(self, data):
        """
        Performs a training step.

        Args:
            data: The data to train on. Should be a list of inputs and outputs: \
                [x, y], where x is a list of tensors: [tx_colloc, tx_init, tx_bnd] and y is \
                    a list of tensors: [u_colloc, residual, u_init, u_bnd_start]
        """

        x, y = data
        u_colloc, residual, u_init, y_boundary = y
        with tf.GradientTape(persistent=False) as tape:
            u_colloc_pred, residual_pred, u_init_pred, u_bnd_start_pred, u_bnd_end_pred, u_bnd_pred, hamil_term = self(
                x, 
                training=True)
            loss_residual = self.res_loss(residual, residual_pred)
            loss_initial = self.init_loss(u_init, u_init_pred)
            if self.PBC == True:
                loss_boundary = self.bnd_loss(u_bnd_start_pred, u_bnd_end_pred)
            else:
                loss_boundary = self.bnd_loss(u_bnd_pred, y_boundary)
            loss_hamil = self.hamil_loss(-0.002, hamil_term)
            # loss_dudt = self.dudt_loss(dudt, dudt_pred) # it is one value
            loss_total = self._loss_residual_weight * loss_residual + self._loss_initial_weight * loss_initial \
                + self._loss_boundary_weight * loss_boundary + self._loss_hamil_weight * loss_hamil 
        gards = tape.gradient(loss_total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gards, self.trainable_variables))

        self.loss_initial_tracker.update_state(loss_initial)
        self.loss_boundary_tracker.update_state(loss_boundary)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_hamil_tracker.update_state(loss_hamil)
        # self.loss_dudt_tracker.update_state(loss_dudt)
        self.loss_total_tracker.update_state(loss_total)
        self.mae_tracker.update_state(u_colloc, u_colloc_pred)

        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        """
        Performs a test step.

        Args:
            data: The data to test on. Should be a list of inputs and outputs: \
                [x, y], where x is a list of tensors: [tx_colloc, tx_init, tx_bnd] and y is \
                    a list of tensors: [u_colloc, residual, u_init, u_bnd_start]
        """

        x, y = data

        u_colloc, residual, u_init, y_boundary = y
        u_colloc_pred, residual_pred, u_init_pred, u_bnd_start_pred, u_bnd_end_pred, u_bnd_pred, hamil_term = self(
            x)
        loss_residual = self.res_loss(residual, residual_pred)
        loss_initial = self.init_loss(u_init, u_init_pred)
        if self.PBC == True:
            loss_boundary = self.bnd_loss(u_bnd_start_pred, u_bnd_end_pred)
        else:
            loss_boundary = self.bnd_loss(u_bnd_pred, y_boundary)
        loss_hamil = self.hamil_loss(-0.002, hamil_term)
        # loss_dudt = self.dudt_loss(dudt, dudt_pred)
        loss_total = self._loss_residual_weight * loss_residual + self._loss_initial_weight * loss_initial \
            + self._loss_boundary_weight * loss_boundary + self._loss_hamil_weight * loss_hamil 

        self.loss_initial_tracker.update_state(loss_initial)
        self.loss_boundary_tracker.update_state(loss_boundary)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_hamil_tracker.update_state(loss_hamil)
        # self.loss_dudt_tracker.update_state(loss_dudt)
        self.loss_total_tracker.update_state(loss_total)
        self.mae_tracker.update_state(u_colloc, u_colloc_pred)

        return {m.name: m.result() for m in self.metrics}
    
    def fit_custom(self, inputs: List['tf.Tensor'], outputs: List['tf.Tensor'], epochs: int, print_every: int = 1000):
        """
        Custom alternative to tensorflow fit function, mainly to allow inputs with different sizes. Training is done in full batches.
        
        Args:
            inputs: The inputs to the model. Should be a list of tensors: [tx_colloc, tx_init, tx_bnd]
            outputs: The outputs to the model. Should be a list of tensors: [u_colloc, residual, u_init, u_bnd]
            epochs: The number of epochs to train for.
            print_every: The number of epochs between printing the loss. Defaults to 1000.

        Returns:
            A dictionary containing the history of the training.
        """

        history = create_history_dictionary_Kawahara_custom()
        
        for epoch in range(epochs):
            metrs = self.train_step([inputs, outputs])
            for key, value in metrs.items():
                history[key].append(value.numpy())

            if epoch % print_every == 0:
                tf.print(f"Epoch {epoch}, Loss Residual: {metrs['loss_residual']:0.10f}, Loss Initial: {metrs['loss_initial']:0.10f}, Loss Boundary: {metrs['loss_boundary']:0.10f}, Loss Hamil: {metrs['loss_hamil']:0.4f}, MAE: {metrs['mean_absolute_error']:0.4f}")
                
            #reset metrics
            for m in self.metrics:
                m.reset_states()

        return history

    @property
    def metrics(self):
        """
        Returns the metrics to track.
        """
        return [self.loss_total_tracker, self.loss_residual_tracker, self.loss_initial_tracker, \
            self.loss_boundary_tracker, self.loss_hamil_tracker, self.mae_tracker]

class cParametrizationPINN(tf.keras.models.Model):

    def __init__(self, backbone, alpha: float = 1.0, beta: float = 1/4 ,sigma: float = 1.0, loss_residual_weight=1., loss_initial_weight=1., loss_boundary_weight=1., **kwargs):
        """
        Travelling Kawahara model.

        Args:
            backbone: The backbone of the model. Should be a tf.keras.Model.
            c: travelling wave speed (deprecated)
            loss_residual_weight: The weight of the residual loss. Defaults to 1.
            loss_initial_weight: The weight of the initial loss. Defaults to 1.
            loss_boundary_weight: The weight of the boundary loss. Defaults to 1.
            kwargs: Additional arguments to pass to the tf.keras.Model constructor.
        """
        super().__init__(**kwargs)
        self.backbone = backbone

        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta

        self._loss_residual_weight = tf.Variable(loss_residual_weight, trainable=False, dtype=tf.keras.backend.floatx(), \
                                                 name="loss_residual_weight")
        self._loss_initial_weight = tf.Variable(loss_initial_weight, trainable=False, dtype=tf.keras.backend.floatx(), \
                                                name="loss_initial_weight")
        self._loss_boundary_weight = tf.Variable(loss_boundary_weight, trainable=False, dtype=tf.keras.backend.floatx(), \
                                                 name="loss_boundary_weight")

        self.loss_total_tracker = tf.keras.metrics.Mean(name=LOSS_TOTAL)
        self.loss_residual_tracker = tf.keras.metrics.Mean(name=LOSS_RESIDUAL)
        self.loss_initial_tracker = tf.keras.metrics.Mean(name=LOSS_INITIAL)
        self.loss_boundary_tracker = tf.keras.metrics.Mean(name=LOSS_BOUNDARY)

        # self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name=MEAN_ABSOLUTE_ERROR)
        self.res_loss = tf.keras.losses.MeanSquaredError()
        self.init_loss = tf.keras.losses.MeanSquaredError()
        self.bnd_loss = tf.keras.losses.MeanSquaredError()

    def set_loss_weights(self, loss_residual_weight, loss_initial_weight, loss_boundary_weight):

        self._loss_residual_weight.assign(loss_residual_weight)
        self._loss_initial_weight.assign(loss_initial_weight)
        self._loss_boundary_weight.assign(loss_boundary_weight)

    @tf.function
    def call(self, inputs, training=False):
        """
        Performs a forward pass through the model.

        Args:
            inputs: The inputs to the model. Should be a list of tensors: [cx_colloc, cx_init, cx_bnd_start, cx_bnd_end]
            training: Whether to run in training mode. Defaults to False.

        Returns:
            A list of tensors: [u_colloc, residual, u_init, u_bnd]
        """
        cx_colloc = inputs[0]
        cx_init = inputs[1]
        cx_bnd_start = inputs[2]
        cx_bnd_end = inputs[3]

        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape5:
            with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape4:
                with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape3:
                    with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape2:
                        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
                            tape.watch(cx_colloc)
                            u_colloc = self.backbone(cx_colloc, training=training)
                        first_order = tape.batch_jacobian(u_colloc, cx_colloc)
                        u_c = first_order[..., 0]
                        u_x = first_order[..., 1]
                    u_xx = tape2.batch_jacobian(u_x, cx_colloc)[..., 1]
                u_xxx = tape3.batch_jacobian(u_xx, cx_colloc)[..., 1]
            u_4x = tape4.batch_jacobian(u_xxx, cx_colloc)[..., 1]
        u_5x = tape5.batch_jacobian(u_4x, cx_colloc)[..., 1]
        
        
        
        amplitude = tf.math.abs(tf.reduce_max(u_colloc)-tf.reduce_min(u_colloc))
        
        residual = + self.alpha * u_xxx + (self.beta) * u_5x + (self.sigma * amplitude * 2 * u_colloc) * u_x + (cx_colloc[:,:1]* u_x)
        u_init = self.backbone(cx_init, training=training)
        u_bnd_start = self.backbone(cx_bnd_start, training=training)
        u_bnd_end = self.backbone(cx_bnd_end, training=training)

        


        return residual, u_init, u_bnd_start, u_bnd_end, amplitude
    
    @tf.function
    def train_step(self, data):
        x, y = data
        residual, u_init = y
        with tf.GradientTape(persistent=False) as tape:
            residual_pred, u_init_pred, u_bnd_start_pred, u_bnd_end_pred, amplitude= self(
                x, 
                training=True)

            loss_residual = self.res_loss(residual, residual_pred)
            loss_initial = self.init_loss(u_init, u_init_pred)
            loss_boundary = self.bnd_loss(u_bnd_start_pred, u_bnd_end_pred)
            loss_total = self._loss_residual_weight * loss_residual + self._loss_initial_weight * loss_initial \
                + self._loss_boundary_weight * loss_boundary 
        gards = tape.gradient(loss_total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gards, self.trainable_variables))
        self.loss_initial_tracker.update_state(loss_initial)
        self.loss_boundary_tracker.update_state(loss_boundary)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_total_tracker.update_state(loss_total)
        # self.mae_tracker.update_state(u_colloc, u_colloc_pred)

        return {m.name: m.result() for m in self.metrics}, amplitude
    
    def test_step(self, data):


        x, y = data
        residual, u_init = y
        residual_pred, u_init_pred, u_bnd_start_pred, u_bnd_end_pred,_ = self(
            x)
        print('ampl test',_)
        loss_residual = self.res_loss(residual, residual_pred)
        loss_initial = self.init_loss(u_init, u_init_pred)
        loss_boundary = self.bnd_loss(u_bnd_start_pred, u_bnd_end_pred)
        loss_total = self._loss_residual_weight * loss_residual + self._loss_initial_weight * loss_initial \
            + self._loss_boundary_weight * loss_boundary 

        self.loss_initial_tracker.update_state(loss_initial)
        self.loss_boundary_tracker.update_state(loss_boundary)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_total_tracker.update_state(loss_total)
        # self.mae_tracker.update_state(u_colloc, u_colloc_pred)

        return {m.name: m.result() for m in self.metrics}
    
    def fit_custom(self, inputs: List['tf.Tensor'], outputs: List['tf.Tensor'], epochs: int, print_every: int = 1000):
        """
        Custom alternative to tensorflow fit function, mainly to allow inputs with different sizes. Training is done in full batches.
        
        Args:
            inputs: The inputs to the model. Should be a list of tensors: [tx_colloc, tx_init, tx_bnd]
            outputs: The outputs to the model. Should be a list of tensors: [u_colloc, residual, u_init, u_bnd]
            epochs: The number of epochs to train for.
            print_every: The number of epochs between printing the loss. Defaults to 1000.

        Returns:
            A dictionary containing the history of the training.
        """

        history = create_history_dictionary()
        
        for epoch in range(epochs):
            metrs, amplitude = self.train_step([inputs, outputs])
            for key, value in metrs.items():
                history[key].append(value.numpy())

            if epoch % print_every == 0:
                tf.print(f"Epoch {epoch}, Loss Residual: {metrs['loss_residual']:0.4f}, Loss Initial: {metrs['loss_initial']:0.4f}, Loss Boundary: {metrs['loss_boundary']:0.4f}",'ampl:',amplitude)
                
            #reset metrics
            for m in self.metrics:
                m.reset_states()

        return history

    @property
    def metrics(self):
        """
        Returns the metrics to track.
        """
        return [self.loss_total_tracker, self.loss_residual_tracker, self.loss_initial_tracker, \
            self.loss_boundary_tracker]
        
class seq2seqAmplitudePINN(tf.keras.models.Model):

    def __init__(self, backbone, alpha: float = 1.0, beta: float = 1/4 ,sigma: float = 1.0, loss_residual_weight=1., loss_initial_weight=1., loss_boundary_weight=1., **kwargs):
 
        super().__init__(**kwargs)
        self.backbone = backbone

        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta

        self._loss_residual_weight = tf.Variable(loss_residual_weight, trainable=False, dtype=tf.keras.backend.floatx(), \
                                                 name="loss_residual_weight")
        self._loss_initial_weight = tf.Variable(loss_initial_weight, trainable=False, dtype=tf.keras.backend.floatx(), \
                                                name="loss_initial_weight")
        self._loss_boundary_weight = tf.Variable(loss_boundary_weight, trainable=False, dtype=tf.keras.backend.floatx(), \
                                                 name="loss_boundary_weight")

        self.loss_total_tracker = tf.keras.metrics.Mean(name=LOSS_TOTAL)
        self.loss_residual_tracker = tf.keras.metrics.Mean(name=LOSS_RESIDUAL)
        self.loss_initial_tracker = tf.keras.metrics.Mean(name=LOSS_INITIAL)
        self.loss_boundary_tracker = tf.keras.metrics.Mean(name=LOSS_BOUNDARY)

        # self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name=MEAN_ABSOLUTE_ERROR)
        self.res_loss = tf.keras.losses.MeanSquaredError()
        self.init_loss = tf.keras.losses.MeanSquaredError()
        self.bnd_loss = tf.keras.losses.MeanSquaredError()

    def set_loss_weights(self, loss_residual_weight, loss_initial_weight, loss_boundary_weight):

        self._loss_residual_weight.assign(loss_residual_weight)
        self._loss_initial_weight.assign(loss_initial_weight)
        self._loss_boundary_weight.assign(loss_boundary_weight)

    @tf.function
    def call(self, inputs, training=False):

        ax_colloc = inputs[0]
        ax_init = inputs[1]
        ax_bnd_start = inputs[2]
        ax_bnd_end = inputs[3]

        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape5:
            with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape4:
                with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape3:
                    with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape2:
                        with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
                            tape.watch(ax_colloc)
                            u_colloc = self.backbone(ax_colloc, training=training)
                        first_order = tape.batch_jacobian(u_colloc, ax_colloc)
                        u_a = first_order[..., 0]
                        u_x = first_order[..., 1]
                    u_xx = tape2.batch_jacobian(u_x, ax_colloc)[..., 1]
                u_xxx = tape3.batch_jacobian(u_xx, ax_colloc)[..., 1]
            u_4x = tape4.batch_jacobian(u_xxx, ax_colloc)[..., 1]
        u_5x = tape5.batch_jacobian(u_4x, ax_colloc)[..., 1]
        
        
        
        
        residual = + self.alpha * u_xxx + (self.beta) * u_5x + (self.sigma  * 2 * u_colloc) * u_x  -  ax_colloc[:,:1] * u_x
        u_init = self.backbone(ax_init, training=training)
        u_bnd_start = self.backbone(ax_bnd_start, training=training)
        u_bnd_end = self.backbone(ax_bnd_end, training=training)

        


        return residual, u_init, u_bnd_start, u_bnd_end
    
    @tf.function
    def train_step(self, data):
        x, y = data
        residual, u_init = y
        with tf.GradientTape(persistent=False) as tape:
            residual_pred, u_init_pred, u_bnd_start_pred, u_bnd_end_pred = self(
                x, 
                training=True)

            loss_residual = self.res_loss(residual, residual_pred)
            loss_initial = self.init_loss(u_init, u_init_pred)
            loss_boundary = self.bnd_loss(u_bnd_start_pred, u_bnd_end_pred)
            loss_total = self._loss_residual_weight * loss_residual + self._loss_initial_weight * loss_initial \
                + self._loss_boundary_weight * loss_boundary 
        gards = tape.gradient(loss_total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gards, self.trainable_variables))
        self.loss_initial_tracker.update_state(loss_initial)
        self.loss_boundary_tracker.update_state(loss_boundary)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_total_tracker.update_state(loss_total)
        # self.mae_tracker.update_state(u_colloc, u_colloc_pred)

        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):


        x, y = data
        residual, u_init = y
        residual_pred, u_init_pred, u_bnd_start_pred, u_bnd_end_pred = self(
            x)
        loss_residual = self.res_loss(residual, residual_pred)
        loss_initial = self.init_loss(u_init, u_init_pred)
        loss_boundary = self.bnd_loss(u_bnd_start_pred, u_bnd_end_pred)
        loss_total = self._loss_residual_weight * loss_residual + self._loss_initial_weight * loss_initial \
            + self._loss_boundary_weight * loss_boundary 

        self.loss_initial_tracker.update_state(loss_initial)
        self.loss_boundary_tracker.update_state(loss_boundary)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_total_tracker.update_state(loss_total)
        # self.mae_tracker.update_state(u_colloc, u_colloc_pred)

        return {m.name: m.result() for m in self.metrics}
    
    def fit_custom(self, inputs: List['tf.Tensor'], outputs: List['tf.Tensor'], epochs: int, print_every: int = 1000):
        """
        Custom alternative to tensorflow fit function, mainly to allow inputs with different sizes. Training is done in full batches.
        
        Args:
            inputs: The inputs to the model. Should be a list of tensors: [tx_colloc, tx_init, tx_bnd]
            outputs: The outputs to the model. Should be a list of tensors: [u_colloc, residual, u_init, u_bnd]
            epochs: The number of epochs to train for.
            print_every: The number of epochs between printing the loss. Defaults to 1000.

        Returns:
            A dictionary containing the history of the training.
        """

        history = create_history_dictionary()
        
        for epoch in range(epochs):
            metrs = self.train_step([inputs, outputs])
            for key, value in metrs.items():
                history[key].append(value.numpy())

            if epoch % print_every == 0:
                tf.print(f"Epoch {epoch}, Loss Residual: {metrs['loss_residual']:0.4f}, Loss Initial: {metrs['loss_initial']:0.4f}, Loss Boundary: {metrs['loss_boundary']:0.4f}")
                
            #reset metrics
            for m in self.metrics:
                m.reset_states()

        return history

    @property
    def metrics(self):
        """
        Returns the metrics to track.
        """
        return [self.loss_total_tracker, self.loss_residual_tracker, self.loss_initial_tracker, \
            self.loss_boundary_tracker]
        
class FourierKawaharaPINN_noCInput(tf.keras.models.Model):

    def __init__(self, backbone, alpha: float = 1.0, beta: float = 1/4 ,sigma: float = 1.0, c = 1., loss_residual_weight=1., loss_initial_weight=1., **kwargs):
 
        super().__init__(**kwargs)
        self.backbone = backbone

        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta
        self.c = c

        self._loss_residual_weight = tf.Variable(loss_residual_weight, trainable=False, dtype=tf.keras.backend.floatx(), \
                                                 name="loss_residual_weight")
        self._loss_initial_weight = tf.Variable(loss_initial_weight, trainable=False, dtype=tf.keras.backend.floatx(), \
                                                name="loss_initial_weight")


        self.loss_total_tracker = tf.keras.metrics.Mean(name=LOSS_TOTAL)
        self.loss_residual_tracker = tf.keras.metrics.Mean(name=LOSS_RESIDUAL)
        self.loss_initial_tracker = tf.keras.metrics.Mean(name=LOSS_INITIAL)

        # self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name=MEAN_ABSOLUTE_ERROR)
        self.res_loss = tf.keras.losses.MeanSquaredError()
        self.init_loss = tf.keras.losses.MeanSquaredError()

    def set_loss_weights(self, loss_residual_weight, loss_initial_weight, loss_boundary_weight):

        self._loss_residual_weight.assign(loss_residual_weight)
        self._loss_initial_weight.assign(loss_initial_weight)
        
    def calculate_nonlinearity(self, u_colloc):
        length = 21
        sum1_net = np.zeros((21,1))
        sum2_net = np.zeros((21,1))
        u_colloc = u_colloc.numpy()
        for k in range(length):
            sum1 = 0.
            sum2 = 0.
            for n in range(k,length):
                sum1 = sum1+u_colloc[n]*u_colloc[n-k] 
            for n in range(0,k):
                sum2 = sum2+u_colloc[n]*u_colloc[k-n] 
            sum1_net[k] = sum1
            sum2_net[k] = sum2
        sum1_net =  tf.convert_to_tensor(sum1_net, dtype=tf.float32)
        sum2_net = tf.convert_to_tensor(sum2_net, dtype=tf.float32)
        return (sum1_net, sum2_net)

    @tf.function
    def call(self, inputs, training=False):
        tf.config.run_functions_eagerly(True)
        ck_colloc = inputs[0]
        ck_init_c = inputs[1]
    
        u_colloc = self.backbone(ck_colloc, training=training)
        (sum1_net, sum2_net)= self.calculate_nonlinearity(u_colloc)

        residual = ck_colloc[:,:1] * u_colloc + self.sigma/2 * (sum1_net + sum2_net) - self.alpha * ck_colloc[:,1:]**2 * u_colloc + self.beta * ck_colloc[:,1:]**4 * u_colloc
        u_init_c = self.backbone(ck_init_c, training=training)

        return residual, u_init_c    
    @tf.function
    def train_step(self, data):
        x, y = data
        residual, u_init_c = y
        with tf.GradientTape(persistent=False) as tape:
            residual_pred, u_init_c_pred  = self(
                x, 
                training=True)

            loss_residual = self.res_loss(residual, residual_pred)
            loss_initial = self.init_loss(u_init_c, u_init_c_pred)
            loss_total = self._loss_residual_weight * loss_residual + self._loss_initial_weight * loss_initial 
        gards = tape.gradient(loss_total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gards, self.trainable_variables))
        self.loss_initial_tracker.update_state(loss_initial)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_total_tracker.update_state(loss_total)
        # self.mae_tracker.update_state(u_colloc, u_colloc_pred)

        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):


        x, y = data
        residual, u_init_c= y
        residual_pred, u_init_c_pred = self(
            x)
        loss_residual = self.res_loss(residual, residual_pred)
        loss_initial = self.init_loss(u_init_c, u_init_c_pred)
        loss_total = self._loss_residual_weight * loss_residual + self._loss_initial_weight * loss_initial 

        self.loss_initial_tracker.update_state(loss_initial)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_total_tracker.update_state(loss_total)
        # self.mae_tracker.update_state(u_colloc, u_colloc_pred)

        return {m.name: m.result() for m in self.metrics}
    
    def fit_custom(self, inputs: List['tf.Tensor'], outputs: List['tf.Tensor'], epochs: int, print_every: int = 1000):
        """
        Custom alternative to tensorflow fit function, mainly to allow inputs with different sizes. Training is done in full batches.
        
        Args:
            inputs: The inputs to the model. Should be a list of tensors: [tx_colloc, tx_init, tx_bnd]
            outputs: The outputs to the model. Should be a list of tensors: [u_colloc, residual, u_init, u_bnd]
            epochs: The number of epochs to train for.
            print_every: The number of epochs between printing the loss. Defaults to 1000.

        Returns:
            A dictionary containing the history of the training.
        """

        history = create_history_dictionary_Fourier()
        
        for epoch in range(epochs):
            metrs = self.train_step([inputs, outputs])
            for key, value in metrs.items():
                history[key].append(value.numpy())

            if epoch % print_every == 0:
                tf.print(f"Epoch {epoch}, Loss Residual: {metrs['loss_residual']:0.4f}, Loss Initial: {metrs['loss_initial']:0.4f}")
                
            #reset metrics
            for m in self.metrics:
                m.reset_states()

        return history

    @property
    def metrics(self):
        """
        Returns the metrics to track.
        """
        return [self.loss_total_tracker, self.loss_residual_tracker, self.loss_initial_tracker]
    
class FourierKawaharaPINN(tf.keras.models.Model):

    def __init__(self, backbone, alpha: float = 1.0, beta: float = 1/4 ,sigma: float = 1.0, loss_residual_weight=1., loss_initial_weight=1., **kwargs):
 
        super().__init__(**kwargs)
        self.backbone = backbone

        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta

        self._loss_residual_weight = tf.Variable(loss_residual_weight, trainable=False, dtype=tf.keras.backend.floatx(), \
                                                 name="loss_residual_weight")
        self._loss_initial_weight = tf.Variable(loss_initial_weight, trainable=False, dtype=tf.keras.backend.floatx(), \
                                                name="loss_initial_weight")


        self.loss_total_tracker = tf.keras.metrics.Mean(name=LOSS_TOTAL)
        self.loss_residual_tracker = tf.keras.metrics.Mean(name=LOSS_RESIDUAL)
        self.loss_initial_tracker = tf.keras.metrics.Mean(name=LOSS_INITIAL)

        # self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name=MEAN_ABSOLUTE_ERROR)
        self.res_loss = tf.keras.losses.MeanSquaredError()
        self.init_loss = tf.keras.losses.MeanSquaredError()

    def set_loss_weights(self, loss_residual_weight, loss_initial_weight, loss_boundary_weight):

        self._loss_residual_weight.assign(loss_residual_weight)
        self._loss_initial_weight.assign(loss_initial_weight)
        
    def calculate_nonlinearity(self, u_colloc):
        length = 21
        sum1_net = np.zeros((21,1))
        sum2_net = np.zeros((21,1))
        u_colloc = u_colloc.numpy()
        for k in range(length):
            sum1 = 0.
            sum2 = 0.
            for n in range(k,length):
                sum1 = sum1+u_colloc[n]*u_colloc[n-k] 
            for n in range(0,k):
                sum2 = sum2+u_colloc[n]*u_colloc[k-n] 
            sum1_net[k] = sum1
            sum2_net[k] = sum2
        sum1_net =  tf.convert_to_tensor(sum1_net, dtype=tf.float32)
        sum2_net = tf.convert_to_tensor(sum2_net, dtype=tf.float32)
        return (sum1_net, sum2_net)

    @tf.function
    def call(self, inputs, training=False):
        tf.config.run_functions_eagerly(True)
        ck_colloc = inputs[0]
        ck_init_c = inputs[1]
    
        u_colloc = self.backbone(ck_colloc, training=training)
        (sum1_net, sum2_net)= self.calculate_nonlinearity(u_colloc)

        residual = self.c * u_colloc + self.sigma/2 * (sum1_net + sum2_net) - self.alpha * ck_colloc[:,1:]**2 * u_colloc + self.beta * ck_colloc[:,1:]**4 * u_colloc
        u_init_c = self.backbone(ck_init_c, training=training)

        return residual, u_init_c    
    @tf.function
    def train_step(self, data):
        x, y = data
        residual, u_init_c = y
        with tf.GradientTape(persistent=False) as tape:
            residual_pred, u_init_c_pred  = self(
                x, 
                training=True)

            loss_residual = self.res_loss(residual, residual_pred)
            loss_initial = self.init_loss(u_init_c, u_init_c_pred)
            loss_total = self._loss_residual_weight * loss_residual + self._loss_initial_weight * loss_initial 
        gards = tape.gradient(loss_total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gards, self.trainable_variables))
        self.loss_initial_tracker.update_state(loss_initial)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_total_tracker.update_state(loss_total)
        # self.mae_tracker.update_state(u_colloc, u_colloc_pred)

        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):


        x, y = data
        residual, u_init_c= y
        residual_pred, u_init_c_pred = self(
            x)
        loss_residual = self.res_loss(residual, residual_pred)
        loss_initial = self.init_loss(u_init_c, u_init_c_pred)
        loss_total = self._loss_residual_weight * loss_residual + self._loss_initial_weight * loss_initial 

        self.loss_initial_tracker.update_state(loss_initial)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_total_tracker.update_state(loss_total)
        # self.mae_tracker.update_state(u_colloc, u_colloc_pred)

        return {m.name: m.result() for m in self.metrics}
    
    def fit_custom(self, inputs: List['tf.Tensor'], outputs: List['tf.Tensor'], epochs: int, print_every: int = 1000):
        """
        Custom alternative to tensorflow fit function, mainly to allow inputs with different sizes. Training is done in full batches.
        
        Args:
            inputs: The inputs to the model. Should be a list of tensors: [tx_colloc, tx_init, tx_bnd]
            outputs: The outputs to the model. Should be a list of tensors: [u_colloc, residual, u_init, u_bnd]
            epochs: The number of epochs to train for.
            print_every: The number of epochs between printing the loss. Defaults to 1000.

        Returns:
            A dictionary containing the history of the training.
        """

        history = create_history_dictionary_Fourier()
        
        for epoch in range(epochs):
            metrs = self.train_step([inputs, outputs])
            for key, value in metrs.items():
                history[key].append(value.numpy())

            if epoch % print_every == 0:
                tf.print(f"Epoch {epoch}, Loss Residual: {metrs['loss_residual']:0.4f}, Loss Initial: {metrs['loss_initial']:0.4f}")
                
            #reset metrics
            for m in self.metrics:
                m.reset_states()

        return history

    @property
    def metrics(self):
        """
        Returns the metrics to track.
        """
        return [self.loss_total_tracker, self.loss_residual_tracker, self.loss_initial_tracker]
    
class FourierKawaharaPINN_noCInput(tf.keras.models.Model):

    def __init__(self, backbone, alpha: float = 1.0, beta: float = 1/4 ,sigma: float = 1.0, c = 1., loss_residual_weight=1., loss_initial_weight=1., **kwargs):
 
        super().__init__(**kwargs)
        self.backbone = backbone

        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta
        self.c = c

        self._loss_residual_weight = tf.Variable(loss_residual_weight, trainable=False, dtype=tf.keras.backend.floatx(), \
                                                 name="loss_residual_weight")
        # self._loss_initial_weight = tf.Variable(loss_initial_weight, trainable=False, dtype=tf.keras.backend.floatx(), \
        #                                         name="loss_initial_weight")


        self.loss_total_tracker = tf.keras.metrics.Mean(name=LOSS_TOTAL)
        self.loss_residual_tracker = tf.keras.metrics.Mean(name=LOSS_RESIDUAL)
        # self.loss_initial_tracker = tf.keras.metrics.Mean(name=LOSS_INITIAL)

        # self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name=MEAN_ABSOLUTE_ERROR)
        self.res_loss = tf.keras.losses.MeanSquaredError()
        # self.init_loss = tf.keras.losses.MeanSquaredError()

    def set_loss_weights(self, loss_residual_weight, loss_initial_weight, loss_boundary_weight):

        self._loss_residual_weight.assign(loss_residual_weight)
        # self._loss_initial_weight.assign(loss_initial_weight)
        
    def calculate_nonlinearity(self, u_colloc):
        length = 21
        sum1_net = np.zeros((21,1))
        sum2_net = np.zeros((21,1))
        u_colloc = u_colloc.numpy()
        for k in range(length):
            sum1 = 0.
            sum2 = 0.
            for n in range(k,length):
                sum1 = sum1+u_colloc[n]*u_colloc[n-k] 
            for n in range(0,k):
                sum2 = sum2+u_colloc[n]*u_colloc[k-n] 
            sum1_net[k] =2* sum1 # factor of 2 accounts for absolute difference
            sum2_net[k] = sum2
        sum1_net =  tf.convert_to_tensor(sum1_net, dtype=tf.float32)
        sum2_net = tf.convert_to_tensor(sum2_net, dtype=tf.float32)
        return (sum1_net, sum2_net)

    @tf.function
    def call(self, inputs, training=False):
        # tf.config.run_functions_eagerly(True)
        k_colloc = inputs[0]
    
        u_colloc = self.backbone(k_colloc, training=training)
        (sum1_net, sum2_net)= self.calculate_nonlinearity(u_colloc)

        residual = self.c * u_colloc + self.sigma/2 * (sum1_net + sum2_net) - self.alpha * k_colloc[:]**2 * u_colloc + self.beta * k_colloc[:]**4 * u_colloc

        return residual  
    @tf.function
    def train_step(self, data):
        x, y = data
        residual = y
        with tf.GradientTape(persistent=False) as tape:
            residual_pred  = self(
                x, 
                training=True)

            loss_residual = self.res_loss(residual, residual_pred)
            # loss_initial = self.init_loss(u_init_c, u_init_c_pred)
            loss_total = self._loss_residual_weight * loss_residual 
        gards = tape.gradient(loss_total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gards, self.trainable_variables))
        # self.loss_initial_tracker.update_state(loss_initial)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_total_tracker.update_state(loss_total)
        # self.mae_tracker.update_state(u_colloc, u_colloc_pred)

        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):


        x, y = data
        residual = y
        residual_pred = self(
            x)
        loss_residual = self.res_loss(residual, residual_pred)
        # loss_initial = self.init_loss(u_init_c, u_init_c_pred)
        loss_total = self._loss_residual_weight * loss_residual

        # self.loss_initial_tracker.update_state(loss_initial)
        self.loss_residual_tracker.update_state(loss_residual)
        self.loss_total_tracker.update_state(loss_total)
        # self.mae_tracker.update_state(u_colloc, u_colloc_pred)

        return {m.name: m.result() for m in self.metrics}
    
    def fit_custom(self, inputs: List['tf.Tensor'], outputs: List['tf.Tensor'], epochs: int, print_every: int = 1000):
        """
        Custom alternative to tensorflow fit function, mainly to allow inputs with different sizes. Training is done in full batches.
        
        Args:
            inputs: The inputs to the model. Should be a list of tensors: [tx_colloc, tx_init, tx_bnd]
            outputs: The outputs to the model. Should be a list of tensors: [u_colloc, residual, u_init, u_bnd]
            epochs: The number of epochs to train for.
            print_every: The number of epochs between printing the loss. Defaults to 1000.

        Returns:
            A dictionary containing the history of the training.
        """

        history = create_history_dictionary_FourierNoC()
        
        for epoch in range(epochs):
            metrs = self.train_step([inputs, outputs])
            for key, value in metrs.items():
                history[key].append(value.numpy())

            if epoch % print_every == 0:
                tf.print(f"Epoch {epoch}, Loss Residual: {metrs['loss_residual']:0.4f}")
                
            #reset metrics
            for m in self.metrics:
                m.reset_states()

        return history

    @property
    def metrics(self):
        """
        Returns the metrics to track.
        """
        return [self.loss_total_tracker, self.loss_residual_tracker]
    
class FourierFeatures(tf.keras.layers.Layer):
    """
    Fourier features layer.
    """

    def __init__(self, n_features, standard_dev = 1., **kwargs):
        """
        Fourier features layer.

        Args:
            n_features: The number of features to use.
            kwargs: Additional arguments to pass to the tf.keras.layers.Layer constructor.
        """
        super().__init__(**kwargs)
        self._n_features = n_features
        self.standard_dev = standard_dev

    def build(self, input_shape):
        """
        Builds the layer.

        Args:
            input_shape: The input shape.
        """
        
        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=self.standard_dev)
        self._w = self.add_weight(name="w", shape=(input_shape[-1], self._n_features), initializer=initializer, trainable=False)
        self._b = self.add_weight(name="b", shape=(self._n_features,), initializer="random_normal", trainable=False)

    def call(self, inputs, **kwargs):
        """
        Performs a forward pass.

        Args:
            inputs: The inputs to the layer.

        Returns:
            The outputs of the layer.
        """
        cos_feats = tf.math.cos(tf.matmul(inputs, self._w)) 
        sin_feats = tf.math.sin(tf.matmul(inputs, self._w)) 
        return tf.concat([cos_feats, sin_feats], axis=-1)


    def get_config(self):
        """
        Returns the configuration of the layer.
        """
        config = super().get_config()
        config.update({"n_features": self._n_features})
        return config
    
class BloodFlowPinn(tf.keras.Model):

    def __init__(self, backbone, r0: float = 1.0, beta_bar: float = 1., alpha_bar: float = 1., gamma: float = 1., kappa: float = 1. , loss_residual_weight=1.0, loss_initial_weight=1.0, \
        loss_boundary_weight=1.0, PBC_x = False, PBC_y = True, **kwargs):

        super(BloodFlowPinn, self).__init__(**kwargs)

        self.backbone = backbone
        self.kappa = kappa
        self.r0 = r0 
        self.alpha_bar = alpha_bar
        self.beta_bar = beta_bar
        self.gamma = gamma


        self.PBC_x = PBC_x
        self.PBC_y = PBC_y

        self.loss_total_tracker = tf.keras.metrics.Mean(name=LOSS_TOTAL)
        self.loss_residual1_tracker = tf.keras.metrics.Mean(name=LOSS_RESIDUAL1)
        self.loss_residual2_tracker = tf.keras.metrics.Mean(name=LOSS_RESIDUAL2)
    
        self.loss_initial_u_tracker = tf.keras.metrics.Mean(name=LOSS_INITIAL_U)
        self.loss_initial_eta_tracker = tf.keras.metrics.Mean(name=LOSS_INITIAL_ETA)
        self.loss_boundary_u_tracker = tf.keras.metrics.Mean(name=LOSS_BOUNDARY_U)
        self.loss_boundary_eta_tracker = tf.keras.metrics.Mean(name=LOSS_BOUNDARY_ETA)
  
       
        # self.mae_tracker = tf.keras.metrics.MeanAbsoluteError(name=MEAN_ABSOLUTE_ERROR)
        self._loss_residual_weight = tf.Variable(loss_residual_weight, trainable=False, name="loss_residual_weight", dtype=tf.keras.backend.floatx())
        self._loss_boundary_weight = tf.Variable(loss_boundary_weight, trainable=False, name="loss_boundary_weight", dtype=tf.keras.backend.floatx())
        self._loss_initial_weight = tf.Variable(loss_initial_weight, trainable=False, name="loss_initial_weight", dtype=tf.keras.backend.floatx())
        self.res1_loss = tf.keras.losses.MeanSquaredError()
        self.res2_loss = tf.keras.losses.MeanSquaredError()

        self.init_u_loss = tf.keras.losses.MeanSquaredError()
        self.init_eta_loss = tf.keras.losses.MeanSquaredError()
        self.bnd_u_loss = tf.keras.losses.MeanSquaredError()
        self.bnd_eta_loss = tf.keras.losses.MeanSquaredError()

    def set_loss_weights(self, loss_residual_weight: float, loss_initial_weight: float, loss_boundary_weight: float):
        """
        Sets the loss weights.

        Args:
            loss_residual_weight: The weight of the residual loss.
            loss_initial_weight: The weight of the initial loss.
            loss_boundary_weight: The weight of the boundary loss.
        """
        self._loss_residual_weight.assign(loss_residual_weight)
        self._loss_boundary_weight.assign(loss_boundary_weight)
        self._loss_initial_weight.assign(loss_initial_weight)


    @tf.function
    def call(self, inputs, training=False):

        tx_samples = inputs[0]
        tx_init = inputs[1]

        tx_x_bnd = inputs[2]



        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape4:
            tape4.watch(tx_samples)
            with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape3:
                tape3.watch(tx_samples)
                with tf.GradientTape(watch_accessed_variables=False, persistent=True)  as tape2:
                    tape2.watch(tx_samples)

                    with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape1:
                        tape1.watch(tx_samples)
                        u_eta = self.backbone(tx_samples, training=training)
                        u_samples = u_eta[...,0]
                        eta_samples = u_eta[...,1]

                    first_order_u = tape1.batch_jacobian(tf.reshape(u_samples,[-1,1]), tx_samples)
                    du_dt = first_order_u[..., 0]
                    du_dx = first_order_u[..., 1]

                    first_order_eta = tape1.batch_jacobian(tf.reshape(eta_samples,[-1,1]), tx_samples)
                    deta_dt = first_order_eta[..., 0]
                    deta_dx = first_order_eta[..., 1]


                d2u_dx2 = tape2.batch_jacobian(du_dx, tx_samples)[..., 1]
          
            dd2u_dx2dt = tape3.batch_jacobian(d2u_dx2, tx_samples)[..., 0]

 

   

        lhs1_samples = deta_dt + 1/2 * self.r0 * du_dx + 1/2 * eta_samples * du_dx + deta_dx * u_samples
        lhs2_samples = du_dt + self.beta_bar * deta_dx + u_samples * du_dx - (4*self.alpha_bar + self.r0) * self.r0/8 * dd2u_dx2dt + self.kappa * u_samples - self.gamma * self.beta_bar * self.r0/2 * d2u_dx2
  

  

        u_eta_initial = self.backbone(tx_init, training=training)
        u_initial = u_eta_initial[...,0]
        eta_initial = u_eta_initial[...,1]

        u_eta_bnd =  self.backbone(tx_x_bnd, training=training)
        u_bnd = u_eta_bnd[...,0]
        eta_bnd = u_eta_bnd[...,1]




        return lhs1_samples, lhs2_samples, u_initial, eta_initial, u_bnd, eta_bnd
    
    @tf.function
    def train_step(self, data):
        """
        Performs a training step.

        Args:
            data: The data to train on. First input is the samples, second input is the initial, \
                and third input is the boundary data. First output is the exact solution for the samples, \
                second output is the exact rhs for the samples, third output is the exact solution for the initial, \
                and fourth output is the exact solution for the boundary.
        """

        inputs, outputs = data
        rhs_samples_exact, u_initial_exact, eta_initial_exact, u_bnd_exact, eta_bnd_exact = outputs

        with tf.GradientTape() as tape:
            lhs1_samples, lhs2_samples, u_initial, eta_initial, u_bnd, eta_bnd = self(inputs, training=True)

            loss_residual1 = self.res1_loss(rhs_samples_exact, lhs1_samples)
            loss_residual2 = self.res2_loss(rhs_samples_exact, lhs2_samples)
            loss_initial_u = self.init_u_loss(u_initial_exact, u_initial)
            loss_initial_eta = self.init_eta_loss(eta_initial_exact, eta_initial)
            loss_bnd_u = self.bnd_u_loss(u_bnd_exact, u_bnd)
            loss_bnd_eta = self.bnd_eta_loss(eta_bnd_exact, eta_bnd)
            

            loss = self._loss_residual_weight * (loss_residual1 + loss_residual2) + self._loss_initial_weight * (loss_initial_u + \
                loss_initial_eta) + self._loss_boundary_weight * (loss_bnd_u+ loss_bnd_eta)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_total_tracker.update_state(loss)
        # self.mae_tracker.update_state(u_samples_exact, u_samples)
        self.loss_residual1_tracker.update_state(loss_residual1)
        self.loss_residual2_tracker.update_state(loss_residual2)
        self.loss_initial_u_tracker.update_state(loss_initial_u)
        self.loss_initial_eta_tracker.update_state(loss_initial_eta)
        self.loss_boundary_u_tracker.update_state(loss_bnd_u)
        self.loss_boundary_eta_tracker.update_state(loss_bnd_eta)

        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):

        inputs, outputs = data
        rhs_samples_exact, u_initial_exact, eta_initial_exact, u_bnd_exact, eta_bnd_exact = outputs
        lhs1_samples, lhs2_samples, u_initial, eta_initial, u_bnd, eta_bnd  = self(inputs, training=False)

        loss_residual1 = self.res1_loss(rhs_samples_exact, lhs1_samples)
        loss_residual2 = self.res2_loss(rhs_samples_exact, lhs2_samples)
        loss_initial_u = self.init_u_loss(u_initial_exact, u_initial)
        loss_initial_eta = self.init_eta_loss(eta_initial_exact, eta_initial)
        loss_bnd_u = self.bnd_u_loss(u_bnd_exact, u_bnd)
        loss_bnd_eta = self.bnd_eta_loss(eta_bnd_exact, eta_bnd)

        loss = self._loss_residual_weight * (loss_residual1 + loss_residual2) + self._loss_initial_weight * (loss_initial_u + \
                loss_initial_eta) + self._loss_boundary_weight * (loss_bnd_u+ loss_bnd_eta)
        self.loss_total_tracker.update_state(loss)
        # self.mae_tracker.update_state(u_samples_exact, u_samples)
        self.loss_residual1_tracker.update_state(loss_residual1)
        self.loss_residual2_tracker.update_state(loss_residual2)
        self.loss_initial_u_tracker.update_state(loss_initial_u)
        self.loss_initial_eta_tracker.update_state(loss_initial_eta)
        self.loss_boundary_u_tracker.update_state(loss_bnd_u)
        self.loss_boundary_eta_tracker.update_state(loss_bnd_eta)
        

        return {m.name: m.result() for m in self.metrics}
    
    def fit_custom(self, inputs: List['tf.Tensor'], outputs: List['tf.Tensor'], epochs: int, print_every: int = 1000):
        '''
        Custom alternative to tensorflow fit function, mainly to allow inputs with different sizes. Training is done in full batches.

        Args:
            data: The data to train on. Should be a list of tensors: [tx_colloc, tx_init, tx_bnd]
            outputs: The outputs to train on. Should be a list of tensors: [u_colloc, residual, u_init, u_bnd]. u_colloc is only used for the MAE metric.
            epochs: The number of epochs to train for.
            print_every: How often to print the metrics. Defaults to 1000.
        '''
        history = create_history_dictionary_BloodFlow()
        
        for epoch in range(epochs):
            metrs = self.train_step([inputs, outputs])
            for key, value in metrs.items():
                history[key].append(value.numpy())

            if epoch % print_every == 0:
                tf.print(f"Epoch {epoch}, Loss Residual u: {metrs['loss_residual1']:0.4f}, Loss Residual eta: {metrs['loss_residual2']:0.4f}, Loss Initial u: {metrs['loss_initial_u']:0.4f}, Loss Initial eta: {metrs['loss_initial_eta']:0.4f}, Loss Boundary u: {metrs['loss_boundary_u']:0.4f}, Loss Boundary eta: {metrs['loss_boundary_eta']:0.4f}")
                
            #reset metrics
            for m in self.metrics:
                m.reset_states()

        return history

    @property
    def metrics(self):
        """
        Returns the metrics of the model.
        """
        return [self.loss_total_tracker, self.loss_residual1_tracker, self.loss_residual2_tracker, self.loss_initial_u_tracker, self.loss_initial_eta_tracker, self.loss_boundary_u_tracker, self.loss_boundary_eta_tracker]
    
class  LinearProbe:
    """
    Linear probe for measuring layer representations. Adds a linear regression model on top of each layer in the original model.
    """

    def __init__(self, model=None, skip_layers=[]):
        """
        model: tf.keras model to compute linear probe on.
        skip_layers: list of layer names to skip when computing linear probe.
        """
        self.model = model
        self.skip_layers = skip_layers
        self.linear_models = []

    def fit(self, x, y):
        self.linear_models = []
        inp = self.model.input
        outputs = [layer.output for layer in self.model.layers if layer.name not in self.skip_layers]
        functor = tf.keras.backend.function([inp], outputs)
        layer_outs = functor(x)
        for i in range(len(layer_outs)):
            model = LinearRegression()
            model.fit(layer_outs[i], y)
            self.linear_models.append(model)
        return self

    def __call__(self, x):
        inp = self.model.input
        outputs = [layer.output for layer in self.model.layers if layer.name not in self.skip_layers]
        functor = tf.keras.backend.function([inp], outputs)
        layer_outs = functor(x)
        return [model.predict(layer_outs[i]) for i, model in enumerate(self.linear_models)]

    def predict(self, x):
        return self(x)