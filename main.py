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


ROOT_DIR = "VCTK-Corpus"
WAV_DIR = os.path.join(ROOT_DIR, "wav48")

def extract_features(file_path, mode=1, max_pad_len=400):
    #read the audio file
    audio, sr = librosa.load(file_path, mono=True)
    librosa.get_duration(y=y, sr=sr)
    #reduce the shape
    audio = audio[::3]

    #extract the audio embeddings using MFCC
    """ if mode == 1:
        features = librosa.feature.melspectrogram(y=audio, sr = sr, n_mels = 128, fmax = None)
    else: """
    
    features = librosa.feature.mfcc(y=audio, sr=sr) 
    #as the audio embeddings length varies for different audio, we keep the maximum length as 400
    #pad them with zeros
    pad_width = max_pad_len - features.shape[1]
    features = np.pad(features, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return features

def create_speakers_dict(directory):
    speaker_dict = {}

    for speaker_id, speaker_dir in enumerate(sorted(glob.glob(os.path.join(directory, "*/")))):
        speaker_dict[speaker_id] = []
        
        for audio_path in glob.glob(os.path.join(speaker_dir, "*mic2.flac")):
            speaker_dict[speaker_id].append(audio_path)
                
    return speaker_dict

def reduce_dict(dict, n):
    return dict(zip(list(dict.keys())[:n], list(dict.values())[:n]))

def create_triplets(dict, keys = 10, n = 10, mode=1):
    anchors, positives, negatives = [], [], []
    labels = []

    dict_keys = list(dict.keys())[:keys]

    for speaker_id in dict_keys:
        for i in range(n):
            print(speaker_id, i)
            anchor_path, positive_path = random.sample(dict[speaker_id], 2)
            anchor_features = extract_features(anchor_path, mode=mode)
            positive_features = extract_features(positive_path, mode=mode)

            negative_speaker_id = random.choice([id for id in dict_keys if id != speaker_id])
            negative_path = random.choice(dict[negative_speaker_id])
            negative_features = extract_features(negative_path, mode=mode)

            anchors.append(anchor_features)
            positives.append(positive_features)
            negatives.append(negative_features)
            labels.append(speaker_id)

    #yield np.array(triples), np.array(labels)

    return [np.array([anchors, positives, negatives]), np.array(labels)]

# Define aspects of the model and create instances of both the 
# test and train batch generators and the complete model.

# Create a dictionary mapping speaker IDs to lists of audio file paths
speakers_dict = create_speakers_dict("VCTK-Corpus/wav48")
X, y = create_triplets(speakers_dict, keys=2, n=2)

""" # spliting of data
indices = list(range(X.shape[1]))
train_ind, test_ind = train_test_split(indices, test_size=0.2, random_state=42)
X_train = X[:,train_ind,:,:].shape
X_test = X[:,test_ind,:,:].shape

# spliting of labels
y_train, y_test = train_test_split(y, test_size=0.2, random_state=42) """

A_train, A_test, P_train, P_test, N_train, N_test, y_train, y_test = train_test_split(X[0], X[1], X[2], y, test_size=0.2, random_state=42)

#SHAPE = (triples[0].shape[0], triples[0].shape[1], 1)
input_shape = (X[0].shape[1], X[0].shape[2], 1)

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
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Flatten())
  model.add(Dense(1024, activation="sigmoid"))
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

#train_generator = triples
#test_generator = generate_triplets(test=True)
#batch = next(train_generator)

base_model = embedding_model()
model = complete_model(base_model)
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
exit()

# Fit the model using triplet images provided by the train batch generator.
# Save the trained weights.
""" history = model.fit(train_generator, epochs=20, verbose=2, steps_per_epoch=20, validation_steps=30)
model.save_weights('model.hdf5') """


#train.shape, test.shape
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
history = model.fit([A_test, P_train, N_train], y, verbose=2, validation_split=0.3, epochs=30, batch_size=20, callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])

model.save_weights('model.hdf5')

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training and Validation Losses',size = 20)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


prediction = base_model.predict(A_test)
print("prediction:", prediction.shape)
# TSNE to use dimensionality reduction to visulaise the resultant embeddings
tsne = TSNE()
#tsne_prediction = tsne.fit_transform(prediction)

def scatter(x, labels, subtitle=None):
    # Create a scatter plot of all the 
    # the embeddings of the model.
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))
    # We create a scatter plot.
    f = plt.figure(figsize=(6, 6))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0,alpha = 0.5, s=40,
                    c=palette[labels.astype(int)] )
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    plt.show()
    
scatter(prediction, y_test)

