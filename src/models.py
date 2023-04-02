import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.layers import Input, Lambda, ReLU, PReLU, LeakyReLU, GlobalMaxPool2D
from keras.regularizers import L1, L2, L1L2
from keras import backend as K

def Activation(activation):
    match activation:
        case 'ReLU':
            return ReLU()
        case 'LeakyReLU':
            return LeakyReLU()
        case 'PReLU':
            return PReLU()
        case _:
            return ReLU()

""" def cnn_block(v_cnn, filters, kernel=(3,3), strides=(1,1), padding='same', activation='ReLU'):
    return Sequential([
        Conv2D(filters, kernel, strides, padding=padding, use_bias=False),
        BatchNormalization(),
        ReLU(),
        MaxPool2D((2,2)),
    ]) """

#https://stackoverflow.com/questions/47727679/triplet-model-for-image-retrieval-from-the-keras-pretrained-network
def embeding_model(input_shape, v_cnn=1, kernel=(3,3), strides=(1,1), padding='same', embeding=256, activation='ReLU'):
    kernel = eval(kernel)
    strides = eval(strides)
    match v_cnn:
        case 1:
            return Sequential([
                Conv2D(32, (3,3), padding='same', activation='relu', input_shape=input_shape),
                MaxPool2D(pool_size=(2,2)),
                Dropout(0.4),
                Conv2D(64, (3,3), padding='same', activation='relu'),
                MaxPool2D(pool_size=(2,2)),
                Dropout(0.4),
                Flatten(),
                Dense(256, activation=None), # No activation on final dense layer
                Dropout(0.4),
                Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) # L2 normalize embeddings
            ])
        case 2:
            return Sequential([
                Conv2D(32, kernel, strides, padding=padding, use_bias=False, input_shape=input_shape),
                Activation(activation),
                Conv2D(32, kernel, strides, padding=padding, use_bias=False),
                Activation(activation),
                MaxPool2D((2,2), (2,2)),
                BatchNormalization(),

                Conv2D(64, kernel, strides, padding=padding, use_bias=False),
                Activation(activation),
                Conv2D(64, kernel, strides, padding=padding, use_bias=False),
                Activation(activation),
                MaxPool2D((2,2), (2,2)),
                BatchNormalization(),

                GlobalMaxPool2D(),

                Dropout(0.3),
                Dense(embeding, activation='sigmoid'),

                Lambda(lambda x: tf.math.l2_normalize(x, axis=1)), # L2 normalize embeddings
            ])
        case 3:
            return Sequential([
                Conv2D(32, kernel, strides, padding=padding, use_bias=False, input_shape=input_shape),
                Activation(activation),
                MaxPool2D((2,2), (2,2)),
                BatchNormalization(),

                Conv2D(64, kernel, strides, padding=padding, use_bias=False),
                Activation(activation),
                MaxPool2D((2,2), (2,2)),
                BatchNormalization(),

                Conv2D(128, kernel, strides, padding=padding, use_bias=False),
                Activation(activation),
                MaxPool2D((2,2), (2,2)),
                BatchNormalization(),

                GlobalMaxPool2D(),

                Dropout(0.3),
                Dense(embeding, activation='sigmoid'),

                Lambda(lambda x: tf.math.l2_normalize(x, axis=1)), # L2 normalize embeddings
            ])
 