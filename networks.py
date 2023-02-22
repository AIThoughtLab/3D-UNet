# This class consists of several (not all implemented yet) neural networks.


import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Input, Conv3D, Dropout, MaxPooling3D, UpSampling3D, concatenate
from tensorflow.keras.models import Model

class Networks:
    def __init__(self, num_layers, num_neurons, filter_size, dropout_rate, input_shape=(144, 224, 224, 4)):
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.filter_size = filter_size
        self.dropout_rate = dropout_rate
        self.input_shape = input_shape

    def create_unet(self):
        inputs = Input(shape=self.input_shape)

        # Encoder
        conv_layers = []
        pool_layers = []
        x = inputs
        for i in range(self.num_layers):
            x = Conv3D(self.num_neurons * 2 ** i, self.filter_size, activation='relu', padding='same')(x)
            x = Dropout(rate=self.dropout_rate)(x) # Add dropout layer
            x = Conv3D(self.num_neurons * 2 ** i, self.filter_size, activation='relu', padding='same')(x)
            x = Dropout(rate=self.dropout_rate)(x) # Add dropout layer
            conv_layers.append(x)
            if i < self.num_layers - 1:
                x = MaxPooling3D((2, 2, 2))(x)
                pool_layers.append(x)

        # Decoder
        for i in range(self.num_layers - 1, -1, -1):
            if i < self.num_layers - 1:
                x = UpSampling3D((2, 2, 2))(x)
                x = concatenate([x, conv_layers[i]], axis=-1)
            x = Conv3D(self.num_neurons * 2 ** i, self.filter_size, activation='relu', padding='same')(x)
            x = Dropout(rate=self.dropout_rate)(x) # Add dropout layer
            x = Conv3D(self.num_neurons * 2 ** i, self.filter_size, activation='relu', padding='same')(x)
            x = Dropout(rate=self.dropout_rate)(x) # Add dropout layer

        # Output
        outputs = Conv3D(4, 1, activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs)
        return model
