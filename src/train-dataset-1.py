#basic imports
import os
import glob
import random

#data processing
import librosa
import numpy as np
import pandas as pd

#modelling
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Input, Lambda
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model


import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from ann_visualizer.visualize import ann_viz

import tensorflow as tf


X_train, y_train = np.load('data/train.npy', allow_pickle=True)
#X_test, y_test = np.load('data/test.npy')

input_shape = (None, X_train[0].shape[1], X_train[0].shape[2], 1)

def identity_loss(y_true, y_pred):
    return K.mean(y_pred)

def triplet_loss(x, alpha = 0.2):
    # Triplet Loss function.
    anchor,positive,negative = x
    # distance between the anchor and the positive
    pos_dist = K.sum(K.square(anchor-positive),axis=1)
    # distance between the anchor and the negative
    neg_dist = K.sum(K.square(anchor-negative),axis=1)
    # compute loss
    basic_loss = pos_dist-neg_dist+alpha
    loss = K.maximum(basic_loss,0.0)
    return loss

def embedding_model():
  # Simple convolutional model 
  # used for the embedding model.
  model = Sequential()
  print(input_shape)
  model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(Flatten())
  model.summary()
  return model


def complete_model(base_model):
    # Create the complete model with three
    # embedding models and minimize the loss 
    # between their output embeddings
    input_1 = Input(input_shape)
    input_2 = Input(input_shape)
    input_3 = Input(input_shape)
        
    A = base_model(input_1)
    P = base_model(input_2)
    N = base_model(input_3)
   
    loss = Lambda(triplet_loss)([A, P, N]) 
    model = Model(inputs=[input_1, input_2, input_3], outputs=loss)
    model.compile(loss=identity_loss, optimizer=Adam(0.0001))
    return model

base_model = embedding_model()
model = complete_model(base_model)

early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
history = model.fit([X_train[0], X_train[1], X_train[2]], y_train, verbose=2, validation_split=0.3, epochs=30, batch_size=10, callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])

model.save_weights('model.hdf5')

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training and Validation Losses',size = 20)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

eval_model = Model(inputs=Input(input_shape), outputs=Lambda(triplet_loss)(base_model(Input(input_shape))) )
eval_model.load_weights('model.hdf5')

cos_sim = tf.keras.losses.cosine_similarity(
    eval_model(X_train[0][0]), eval_model(X_train[1][0])
).numpy().reshape(-1,1)

accuracy = tf.reduce_mean(tf.keras.metrics.binary_accuracy(1, -cos_sim, threshold=0))
print(accuracy)