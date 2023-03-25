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
X_test, y_test = np.load('data/test.npy', allow_pickle=True)

input_shape = (X_train[0].shape[1], X_train[0].shape[2], 1)
print(input_shape)

def triplet_loss(y_true, y_pred):
    margin = K.constant(1)
    return K.mean(K.maximum(K.constant(0), K.square(y_pred[:,0]) - 0.5*(K.square(y_pred[:,1])+K.square(y_pred[:,2])) + margin))

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def embeding_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 6), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (3, 6), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (3, 6), activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Flatten())
    return model

base_model = embeding_model()

anchor_input = Input(input_shape, name='anchor_input')
positive_input = Input(input_shape, name='positive_input')
negative_input = Input(input_shape, name='negative_input')

encoded_anchor = base_model(anchor_input)
encoded_positive = base_model(positive_input)
encoded_negative = base_model(negative_input)

positive_dist = Lambda(euclidean_distance, name='pos_dist')([encoded_anchor, encoded_positive])
negative_dist = Lambda(euclidean_distance, name='neg_dist')([encoded_anchor, encoded_negative])
tertiary_dist = Lambda(euclidean_distance, name='ter_dist')([encoded_positive, encoded_negative])

stacked_dists = Lambda(lambda vects: K.stack(vects, axis=1), name='stacked_dists')([positive_dist, negative_dist, tertiary_dist])

model = Model([anchor_input, positive_input, negative_input], stacked_dists, name='triple_siamese')

model.compile(loss=triplet_loss, optimizer=Adam(0.0001))

early_stopping = EarlyStopping(patience=3, restore_best_weights=True)
history = model.fit([X_train[0], X_train[1], X_train[2]], y_train, verbose=2, validation_split=0.2, batch_size=10, epochs=25)
model.save_weights('model.hdf5')

# summarize history for loss
""" plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training and Validation Losses',size = 20)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show() """

print(X_train[0].shape)
print(X_train[0][0].shape)

X_train_trm = base_model.predict(X_train[0])
#X_test_trm = base_model.predict(X_test[:sample_size].reshape(-1,28,28,1))

# TSNE to use dimensionality reduction to visulaise the resultant embeddings
tsne = TSNE()
train_tsne_embeds = tsne.fit_transform(X_train_trm)

""" def scatter(x, labels, subtitle=None):
    # Create a scatter plot of all the 
    # the embeddings of the model.
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))
    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0,alpha = 0.5, s=40,
                    c=palette[labels.astype(np.int)] )
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    
    

scatter(train_tsne_embeds, y_train[:sample_size]) """


eval_model = Model(inputs=anchor_input, outputs=encoded_anchor)
eval_model.load_weights('model.hdf5')

cos_sim = tf.keras.losses.cosine_similarity(
    eval_model(X_train[0]), eval_model(X_train[1])
).numpy().reshape(-1,1)
#https://stackoverflow.com/questions/71962592/evaluating-model-evaluate-with-a-triplet-loss-siamese-neural-network-model-t
accuracy = tf.reduce_mean(tf.keras.metrics.binary_accuracy(1, -cos_sim, threshold=0))
print(accuracy)

positive_similarity = tf.keras.losses.cosine_similarity(
    eval_model(X_test[0]), eval_model(X_test[1])
).numpy().mean()

negative_similarity = tf.keras.losses.cosine_similarity(
    eval_model(X_test[0]), eval_model(X_test[2])
).numpy().mean()

print(positive_similarity, negative_similarity)

positive_similarity = euclidean_distance(
    [eval_model(X_test[0]), eval_model(X_test[1])]
).numpy().mean()

negative_similarity = euclidean_distance(
    [eval_model(X_test[0]), eval_model(X_test[2])]
).numpy().mean()

print(positive_similarity, negative_similarity)
