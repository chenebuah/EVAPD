# CONTRIBUTORS: * Ericsson Chenebuah, Michel Nganbe and Alain Tchagang 
# Department of Mechanical Engineering, University of Ottawa, 75 Laurier Ave. East, Ottawa, ON, K1N 6N5 Canada
# Digital Technologies Research Centre, National Research Council of Canada, 1200 Montréal Road, Ottawa, ON, K1A 0R6 Canada
# * email: echen013@uottawa.ca 
# (June-2023)

# 2D Convolutional Neural Netowrks (Conv2D) for feedbacking 'GA-Similarity.py' fitness functions.
# The network architectures are the same for predicting energy above convex hull (Ehull) values and classifying ICSD labels. 
# However, modifications must be made at the output (final) dense layer and '.compile' to ensure the activation and loss functions correspond to the actual prediction/classification process. 

import keras
from keras import layers
from tensorflow.keras.optimizers import Adam

input = keras.Input(shape=(X_samples.shape[1], X_samples.shape[2], X_samples.shape[3]))

x = layers.Conv2D(8, (3, 3), activation='relu', strides= 1, padding='same')(input)
x = layers.Conv2D(16, (3, 3), activation='relu', strides=2, padding='same')(x)
x = layers.Conv2D(32, (3, 3), activation='relu', strides=2, padding='same')(x)

x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Dropout(0.2)(x)

x = layers.Flatten()(x)

x = layers.Dense(256, activation='relu')(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dense(16, activation='relu')(x)
x = layers.Dense(4, activation='relu')(x)

target = layers.Dense(1, activation='linear')(x) # If predicting continuous Ehull values
#target = layers.Dense(1, activation='sigmoid')(x) # If classifying ICSD labels  
    
opt = Adam(learning_rate=1e-3, decay=1e-3/200)
    
Ehull_model = keras.Model(input, target)
#ICSD_model = keras.Model(input, target)

Ehull_model.compile(loss="mean_squared_error", optimizer=opt, metrics=['mae'])
#ICSD_model.compile(loss="binary_cross_entropy", optimizer=opt, metrics=['accuracy'])
