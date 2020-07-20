import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import random

#Load Dataset
(X_train,y_train),(X_test,y_test) = keras.datasets.fashion_mnist.load_data()
plt.imshow(X_train[0],cmap='gray')
X_train = X_train
X_test = X_test



#encoder

from tensorflow.keras.layers import  Conv2D ,MaxPooling2D , UpSampling2D , Conv2DTranspose ,Input
from tensorflow.keras.models import Model
A = Input(shape=(28,28,1))
x = Conv2D(256, (3, 3), activation='relu', padding='same')(A)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(8, (1, 1), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(2, (1, 1), activation='relu', padding='same')(x)
B = Conv2D(1, (1, 1), activation='relu', padding='same')(x)

encoder = Model(A,B)
encoder.compile(loss = 'mean_squared_error',optimizer=tf.keras.optimizers.Adam(lr=0.1) ,metrics = ['accuracy'])
encoder.summary()


#decoder
A0  = tf.keras.layers.Input(shape=(7,7,1))
x = Conv2DTranspose(2, (1, 1), activation='relu', padding='same')(A0)
x = Conv2DTranspose(4, (3, 3), activation='relu', padding='same')(x)
x = Conv2DTranspose(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(x)
x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(x)
x = Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(x)
B0 = Conv2DTranspose(1, (3, 3), activation='tanh', padding='same')(x)

decoder = Model(A0,B0)
decoder.compile(loss = 'mean_squared_error',optimizer=tf.keras.optimizers.Adam(lr=0.1),metrics = ['accuracy'])
decoder.summary()


# Autoencoder
encoder_input = tf.keras.Input(shape=(28,28,1))
encoder_output = encoder(encoder_input)
decoder_output = decoder(encoder_output)

autoencoder = tf.keras.models.Model(encoder_input,decoder_output)
autoencoder.compile(loss = 'mean_squared_error',optimizer=tf.keras.optimizers.Adam(lr=0.1),metrics = ['accuracy'])
decoder.summary()


model_saver = tf.keras.callbacks.ModelCheckpoint("model_weights.h5",
                                                 monitor='val_loss', verbose=1,
                                                 save_best_only=True,
                                                 save_weights_only=False,
                                                 mode='auto')

history = autoencoder.fit(X_train.reshape(-1,28,28,1),
                X_train.reshape(-1,28,28,1),
                epochs = 20 ,
                batch_size = 32 ,
                validation_data=(X_test.reshape(-1,28,28,1),X_test.reshape(-1,28,28,1)),
                callbacks = [model_saver]
                )
code = encoder.predict(X_test[0:10].reshape(-1,28,28,1))
print(code.shape)
extract = decoder.predict(code)
print(extract.shape)