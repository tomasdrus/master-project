#basic imports
import os
import glob
import random
import re
import math

import yaml
CONFIG = yaml.safe_load(open("config.yml"))['prepare']

#data processing
import librosa
import numpy as np

from sklearn.model_selection import train_test_split

def train_test_val_split_dict(dict, train_ratio=0.7, test_ratio=0.15):
    train_dict, val_dict, test_dict = {}, {}, {}
    
    for id in dict.keys():
        arr = random.sample(dict[id], 200)
        n = len(arr)
        train_size = int(n * train_ratio)
        test_size = int(n * test_ratio)

        train_dict[id] = arr[:train_size]
        test_dict[id] = arr[train_size:train_size+test_size]
        val_dict[id] = arr[train_size+test_size:]
        #print(f"{id} == {n}, {len(train_dict[id])}, {len(test_dict[id])}, {len(val_dict[id])}")

    return [train_dict, test_dict, val_dict]


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
            labels.append(0)

    return [np.array([anchors, positives, negatives]), np.array(labels)]


""" def create_triplets(X, y, n):
    anchors, positives, negatives = [], [], []

    keys = list(set(y))
    batch = math.floor(n / len(y))
    n = batch * len(y)
    labels = np.ones(n)

    for key in y:
        indicies = list(np.where(y == key)[0])
        for i in range(batch):
            anchor_i, positive_i = random.sample(indicies, 2)

            negative_key = random.choice([id for id in keys if id != key])
            negative_i = random.choice(list(np.where(y == negative_key)[0]))

            print(f'{i} => P{key+225} a-{anchor_i}, p-{positive_i} | P{negative_key+225} n-{negative_i}')

            anchors.append(X[anchor_i])
            positives.append(X[positive_i])
            negatives.append(X[negative_i])

    return [np.array(anchors), np.array(positives), np.array(negatives)], labels """

def train_test_split_dict(dict, test_ratio=.2):
    train_dict, test_dict = {}, {}
    
    for id in dict.keys():
        arr = random.sample(dict[id], 200)
        n = len(arr)
        test_size = int(n * test_ratio)

        train_dict[id] = arr[:test_size]
        test_dict[id] = arr[test_size:]

    return [train_dict, test_dict]

#speakers_dict = create_speakers_dict("VCTK-Corpus/wav48", mic=2)
#train_dict, test_dict = train_test_split_dict(speakers_dict)


#np.save('data/train.npy', create_triplets(train_dict, keys=10, n=20))
#np.save('data/test.npy', create_triplets(test_dict, keys=10, n=20))

#X_train, y_train = create_triplets(train_dict, keys=5, n=10)
#X_test, y_test = create_triplets(test_dict, keys=5, n=10)


# create triplets
def create_triplets(X, y, n):
    anchors, positives, negatives = [], [], []

    keys = list(set(y))
    batch = math.floor(n / len(keys))
    n = batch * len(keys)
    labels = np.ones(n)

    for key in keys:
        indicies = list(np.where(y == key)[0])
        for i in range(batch):
            anchor_i, positive_i = random.sample(indicies, 2)

            negative_key = random.choice([id for id in keys if id != key])
            negative_i = random.choice(list(np.where(y == negative_key)[0]))

            print(f'{i} => P{key+225} a-{anchor_i}, p-{positive_i} | P{negative_key} n-{negative_i}')

            anchors.append(X[anchor_i])
            positives.append(X[positive_i])
            negatives.append(X[negative_i])

    return [np.array(anchors), np.array(positives), np.array(negatives)], labels

# feature extraction
def extract_features(file_path, mode='spectogram'):
    y, sr = librosa.load(file_path, sr=22050, mono=True) #read the audio file
    y, _ = librosa.effects.trim(y) #trim silence from begin and end
    #print(librosa.get_duration(y=y, sr=sr))
    y = y[0 : (1 * sr)] #cut length to 3 seconds
    
    #y = y[::3] #reduce the shape

    #extract the audio embeddings using MFCC
    if mode == 'spectogram':
        features = librosa.feature.melspectrogram(y=y, sr = sr, n_mels=128, n_fft = 1024, fmax = None)
    else:
        features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20) 

    #pad shorter audio files with zeros
    print(features.shape)
    max_pad_len=44
    pad_width = max_pad_len - features.shape[1]
    features = np.pad(features, pad_width=((0, 0), (0, pad_width)), mode='constant')
    #exit()
    return features


""" def generate_triplets_unique(X, y, n):
    triplets = []
    for i in range(n):
        print(i)
        anchor, positive, negative = get_triplet(X, y)
        triplet = [anchor, positive, negative]

        unique = True
        for trip in triplets:
            if(np.allclose(trip, triplet)):
                unique = False
        
        if unique:
            triplets.append(triplet)
    
    print('before',n)
    print('after',len(triplets))

    anchors = [item[0] for item in triplets]
    positives = [item[1] for item in triplets]
    negatives = [item[2] for item in triplets]

    labels = np.ones(len(triplets))
    
    return [np.array(anchors), np.array(positives), np.array(negatives)], labels """


""" def get_triplet(X, y):
    keys = len(set(y))
    n = a = np.random.randint(keys)
    while n == a:
        n = np.random.randint(keys)
        print (n,a)
    a, p = get_feature(X, y, a), get_feature(X, y, a)
    n = get_feature(X, y, n)

    ap_cos_sim = cosine_similarity(a.flatten().reshape(1, -1), p.flatten().reshape(1, -1))[0][0]
    an_cos_sim = cosine_similarity(a.flatten().reshape(1, -1), n.flatten().reshape(1, -1))[0][0]
    ap_ssim = ssim(a, p, data_range=p.max() - p.min())
    an_ssim = ssim(a, n, data_range=n.max() - n.min())
    print('AP: ',round(ap_ssim,3),'AN: ',round(an_ssim,3), 'DIFF: ', ap_ssim - an_ssim)
    #print('AP',int(distance_matrix(a,p).sum()), int(distance_matrix(a,p).mean()), 'AN',int(distance_matrix(a,n).sum()), int(distance_matrix(a,n).mean()))
    return a, p, n """