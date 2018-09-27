"""
This code is based on "AutoencoderでDeconvolutionとUpSamplingの比較をしてみました"
Link: https://qiita.com/MuAuan/items/69dda8bd4013007d7b18
"""


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Convolution2D, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.datasets import cifar10
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image

def arr_to_img(nparray, name = "image.jpg"):
    img = tf.keras.preprocessing.image.array_to_img(nparray)
    img.save(name)

def encoded(input_img, dim_factor):
    x = Convolution2D(64,(3,3), activation=tf.nn.relu, padding="same")(input_img)
    x = MaxPooling2D((2,2), padding="same")(x)
    x = Convolution2D(128, (3,3), activation=tf.nn.relu, padding="same")(x)
    x = MaxPooling2D((2,2), padding="same")(x)
    encoded = Convolution2D(8 * dim_factor, (3, 3), activation='relu', padding='same')(x)
    return encoded

def decoded(encoded, dim_factor):
    x = Convolution2D(8*dim_factor, (3,3), activation=tf.nn.relu, padding="same")(encoded)
    x = Conv2DTranspose(128, (9,9), activation=tf.nn.relu, padding="valid")(x)
    x = Convolution2D(128, (3,3), activation=tf.nn.relu, padding="same")(x)
    x = Conv2DTranspose(64, (17,17), activation=tf.nn.relu, padding="valid")(x)
    decoded = Convolution2D(3, (3,3), activation=tf.nn.sigmoid, padding="same")(x)
    return decoded

#-------------------------preprocess dataset-------------------------
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train[:50000]/255.
x_test  = x_test[:10000]/255.

#-------------------------set up model-------------------------
dim_factor = 1*2*3
input_img = Input(shape=(32,32,3))
encode = encoded(input_img, dim_factor)
decode = decoded(encode, dim_factor)
autoencoder = Model(inputs = input_img, outputs = decode)
autoencoder.summary()

#-------------------------set up training-------------------------
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
autoencoder.compile(optimizer=optimizer, loss="binary_crossentropy", metrics = ["accuracy"])


autoencoder.fit(x_train, x_train, epochs=10, shuffle=True, validation_data=(x_test, x_test))
decode_imgs = autoencoder.predict(x_test)

#-------------------------view raw image and generate image-------------------------
arr_to_img(x_test[10]*255, "raw1.jpg")
arr_to_img(decode_imgs[10]*255, "predict1.jpg")



