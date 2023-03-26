import tensorflow as tf
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras.layers import Input, Lambda
from keras.regularizers import L1, L2, L1L2
from keras import backend as K

#https://stackoverflow.com/questions/47727679/triplet-model-for-image-retrieval-from-the-keras-pretrained-network
def embeding_model(input_shape ,version=1):
    match version:
        case 1:
            return Sequential([
                Conv2D(64, (2,2), use_bias=False, padding='same', input_shape=input_shape),
                BatchNormalization(),
                Activation("relu"),
                MaxPooling2D(pool_size=2),
                Dropout(0.3),
                Conv2D(32, (2,2), use_bias=False, padding='same'),
                BatchNormalization(),
                Activation("relu"),
                MaxPooling2D(pool_size=2),
                Dropout(0.3),
                Flatten(),
                Dense(128, use_bias=False, activation=None), # No activation on final dense layer
                BatchNormalization(),
                Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) # L2 normalize embeddings
            ])
        case 1.1:
            return Sequential([
                Conv2D(64, (3,3), use_bias=False, padding='same', input_shape=input_shape),
                BatchNormalization(),
                Activation("relu"),
                MaxPooling2D(pool_size=(2,2)),
                Dropout(0.5),

                Conv2D(128, (3,3), use_bias=False, padding='same'),
                BatchNormalization(),
                Activation("relu"),
                MaxPooling2D(pool_size=(2,2)),
                Dropout(0.5),

                Conv2D(256, (3,3), use_bias=False, padding='same'),
                BatchNormalization(),
                Activation("relu"),
                MaxPooling2D(pool_size=(2,2)),
                Dropout(0.5),

                Flatten(),
                Dense(1024, use_bias=False, activation=None), # No activation on final dense layer
                BatchNormalization(),
                Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) # L2 normalize embeddings
            ])
        case 2:
            return Sequential([
                Conv2D(32, (3,3), padding='same', activation='relu', input_shape=input_shape),
                MaxPooling2D(pool_size=(2,2)),
                Dropout(0.4),
                Conv2D(64, (3,3), padding='same', activation='relu'),
                MaxPooling2D(pool_size=(2,2)),
                Dropout(0.4),
                Conv2D(128, (3,3), padding='same', activation='relu'),
                MaxPooling2D(pool_size=(2,2)),
                Dropout(0.4),
                Conv2D(256, (3,3), padding='same', activation='relu'),
                MaxPooling2D(pool_size=(2,2)),
                Dropout(0.4),
                Flatten(),
                Dense(1024, activation=None), # No activation on final dense layer
                Dropout(0.4),
                Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) # L2 normalize embeddings
            ])
        case _:
            return Sequential([
                Conv2D(32, (3,3), padding='same', activation='relu', input_shape=input_shape),
                MaxPooling2D(pool_size=(2,2)),
                Dropout(0.4),
                Conv2D(64, (3,3), padding='same', activation='relu'),
                MaxPooling2D(pool_size=(2,2)),
                Dropout(0.4),
                Flatten(),
                Dense(256, activation=None), # No activation on final dense layer
                Dropout(0.4),
                Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) # L2 normalize embeddings
            ])