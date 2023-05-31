# CONTRIBUTORS: * Ericsson Chenebuah, Michel Nganbe and Alain Tchagang 
# Department of Mechanical Engineering, University of Ottawa, 75 Laurier Ave. East, Ottawa, ON, K1N 6N5 Canada
# Digital Technologies Research Centre, National Research Council of Canada, 1200 Montréal Road, Ottawa, ON, K1A 0R6 Canada
# * email: echen013@uottawa.ca 
# (June-2023)

# THIS CODE EXECUTES THE SEMI-SUPERVISORY FRAMEWORK FOR TARGET-LEARNING THE FORMATION ENERGY IN THE LATENT SPACE OF A VARIATIONAL AUTOENCODER.

import keras
from keras import layers
from keras.layers import Activation
import numpy as np
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.optimizers import Adam

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim), mean=0, stddev=1)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


latent_dim = 256

# 'X_samples' is the reshaped 94 X 8 X 3 RGB input image from 'input_image.py'

#encoder
encoder_inputs = keras.Input(shape=(X_samples.shape[1], X_samples.shape[2], X_samples.shape[3]))
x = layers.Conv2D(32, (3, 3), activation=layers.LeakyReLU(alpha=0.2), strides= 1, padding='same')(encoder_inputs)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(64, (3, 3), activation=layers.LeakyReLU(alpha=0.2), strides=2, padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Flatten()(x)
x = layers.Dense(1024, activation="sigmoid")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = layers.Lambda(Sampling(), output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# Regressor
reg_latent_inputs = Activation("relu")(z)
x = layers.Dense(256, activation="relu")(reg_latent_inputs)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(32, activation="relu")(x)
reg_outputs = layers.Dense(1, activation='linear', name='reg_output')(x)

# Decoder
latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
x = layers.Dense(1024, activation="sigmoid")(latent_inputs)
x = layers.Dense((X_samples.shape[1]/2 * X_samples.shape[2]/2 * 64), activation=layers.LeakyReLU(alpha=0.2))(x)
x = layers.BatchNormalization()(x)
x = layers.Reshape((int(X_samples.shape[1]/2), int(X_samples.shape[2]/2), 64))(x)
x = layers.Conv2DTranspose(32, (3, 3), activation=layers.LeakyReLU(alpha=0.2), strides=2, padding='same')(x)
x = layers.BatchNormalization()(x)
decoder_outputs = layers.Conv2DTranspose(3, (3, 3), activation='sigmoid', padding='same')(x)

# Models
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
reg_supervised = keras.Model(reg_latent_inputs, reg_outputs, name='reg')
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

# VAE compile
outputs = [decoder(encoder(encoder_inputs)[2]), reg_supervised(encoder(encoder_inputs)[2])]
vae = keras.Model(encoder_inputs, outputs, name='vae_mlp')

reconstruction_loss = (tf.reduce_mean(tf.reduce_sum(K.square(encoder_inputs - outputs[0]),axis=[1,2])))*X_samples.shape[1]*X_samples.shape[2]*X_samples.shape[3]
kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
vae_loss = reconstruction_loss + kl_loss
vae.add_loss(vae_loss)

vae.compile(optimizer=Adam(learning_rate=1e-4, decay=1e-3 / 200), 
            loss={'reg': 'mean_squared_error'}, metrics={'reg': 'mae'}  
            ) 

svae_history = vae.fit(X_samples, {'reg': y_Ef},  epochs=1500)

# Encoded latent vectors
_, _, z = encoder.predict(X_samples)
