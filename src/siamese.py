import yaml, time, os, argparse
import numpy as np
import pandas as pd

from munch import DefaultMunch

from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

import tensorflow as tf
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras.layers import Input, Lambda
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from ann_visualizer.visualize import ann_viz

# configuration
config = DefaultMunch.fromDict(yaml.safe_load(open("config.yml"))['siamese'])

# arguments (for grid search)
parser = argparse.ArgumentParser()
parser.add_argument('--batch', dest='batch_size', type=int)
parser.add_argument('--lr', dest='learning_rate', type=float)
parser.add_argument('--epochs', dest='epochs', type=int)
parser.add_argument('--patience', dest='patience', type=int)
parser.add_argument('--margin', dest='margin', type=int)

args = parser.parse_args()

def args_conf(name):
    if(hasattr(args, name) and getattr(args, name) is not None):
        return getattr(args, name)
    return getattr(config, name)


data = np.load('./data/data.npz')
triplets = np.load('./data/triplets.npz')

X_train, y_train, X_test, y_test = data['X_train'], data['y_train'], data['X_test'], data['y_test'] 
X_trip, y_trip = triplets['X_trip'], triplets['y_trip']

input_shape = (X_trip[0].shape[1], X_trip[0].shape[2], 1)
print(input_shape)

def triplet_loss(y_true, y_pred):
    margin = K.constant(args_conf('margin'))
    return K.mean(K.maximum(K.constant(0), K.square(y_pred[:,0]) - 0.5*(K.square(y_pred[:,1])+K.square(y_pred[:,2])) + margin))

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def embeding_model(batch_normalization=False):
    if(batch_normalization):
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
    else:
        return Sequential([
            Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=input_shape),
            MaxPooling2D(pool_size=2),
            Dropout(0.3),
            Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'),
            MaxPooling2D(pool_size=2),
            Dropout(0.3),
            Flatten(),
            Dense(128, activation=None), # No activation on final dense layer
            Lambda(lambda x: tf.math.l2_normalize(x, axis=1)) # L2 normalize embeddings
        ])
    

embeding_model = embeding_model(True)

anchor_input = Input(input_shape, name='anchor_input')
positive_input = Input(input_shape, name='positive_input')
negative_input = Input(input_shape, name='negative_input')

encoded_anchor = embeding_model(anchor_input)
encoded_positive = embeding_model(positive_input)
encoded_negative = embeding_model(negative_input)

positive_dist = Lambda(euclidean_distance, name='pos_dist')([encoded_anchor, encoded_positive])
negative_dist = Lambda(euclidean_distance, name='neg_dist')([encoded_anchor, encoded_negative])
tertiary_dist = Lambda(euclidean_distance, name='ter_dist')([encoded_positive, encoded_negative])

stacked_dists = Lambda(lambda vects: K.stack(vects, axis=1), name='stacked_dists')([positive_dist, negative_dist, tertiary_dist])

model = Model([anchor_input, positive_input, negative_input], stacked_dists, name='triple_siamese')

model.compile(loss=triplet_loss, optimizer=Adam(args_conf('learning_rate')))

early_stopping = EarlyStopping(patience=args_conf('patience'), restore_best_weights=True)
start_time = time.time()
history = model.fit([X_trip[0], X_trip[1], X_trip[2]], y_trip,
                    verbose=config.verbose, validation_split=args_conf('validation_split'),
                    batch_size=args_conf('batch_size'), epochs=args_conf('epochs'), callbacks=[early_stopping])
elapsed_time = time.time() - start_time
print(history.history['loss'][1])
model.save_weights('./weights/model.hdf5')


if(not os.path.exists(f"results/{config.result_name}.csv")):
    column_names = ["triplets", "length", "n_mfcc", "n_mels", "fmin", "fmax", "sr", "epochs", "batch", "lr", "loss", "val_loss", "time"]
    df = pd.DataFrame(columns=column_names)
else:
    df = pd.read_csv(f'results/{config.result_name}.csv', index_col=0)

settings = np.load('./data/settings.npy', allow_pickle=True).item()

df.loc[df.shape[0]] = {
    #'train_test': f'{y_train.shape[0]}/{y_test.shape[0]}',
    'triplets': X_trip[0].shape,
    'length': settings['length'],
    'n_mfcc': settings['n_mfcc'],
    'n_mels': settings['n_mels'],
    'fmin': settings['fmin'],
    'fmax': settings['fmax'],
    'sr': settings['sr'],
    'epochs': f'{len(history.history["loss"])}/{args_conf("epochs")}',
    'batch':args_conf('batch_size'),
    'lr':args_conf('learning_rate'),
    'loss':round(history.history['loss'][-1], 5),
    'val_loss':round(history.history['val_loss'][-1], 5),
    'time':round(elapsed_time, 2)}



print('\n',df,'\n')
#print('\n',df[['epochs', 'batch', 'lr', 'loss', 'val_loss', 'time']],'\n')
df.to_csv(f'results/{config.result_name}.csv')

# summarize history for loss
if(config.plot_history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Training and Validation Losses',size = 20)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()

if(config.create_embedings):
    X_train = embeding_model.predict(X_train)
    X_test = embeding_model.predict(X_test)

    np.savez('./data/embedings.npz', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

if(config.plot_scatter):
    print('Elements:', X_test.shape[0], 'enbedins size: ', X_test[1].shape)

    # TSNE to use dimensionality reduction to visulaise the resultant embeddings
    tsne = TSNE()
    #train_tsne_embeds = tsne.fit_transform(X_train)
    test_tsne_embeds = tsne.fit_transform(X_test)

    def scatter(data, labels, subtitle=None):
        palette = np.array(sns.color_palette("hls", len(set(labels))))
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(aspect='equal')
        ax.scatter(data[:,0], data[:,1], lw=0,alpha = 0.5, s=40,c=palette[labels] )
        plt.xlim(-25, 25)
        plt.ylim(-25, 25)
        ax.axis('off')
        ax.axis('tight')
        plt.show()

    scatter(test_tsne_embeds, y_test)

# plot speakers ids

""" # Create a Classifier that computes the class of a specific embedding. 
Classifier_input = Input((10,))
Classifier_output = Dense(10, activation='softmax')(Classifier_input)
Classifier_model = Model(Classifier_input, Classifier_output)

# convert the target labels to onehot encoded vectors.
Y_train_onehot = np_utils.to_categorical(y_train, 10)
Y_test_onehot = np_utils.to_categorical(y_test, 10)

Classifier_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
Classifier_model.fit(X_train,Y_train_onehot, validation_data=(X_test,Y_test_onehot),epochs=10)


def gini(x):
    # calculates the gini coeffiecent of 
    # an array. 
    mad = np.abs(np.subtract.outer(x, x)).mean()
    rmad = mad/np.mean(x)
    g = 0.5 * rmad
    return g

def DigitOrNumber(x):
  # Creates an embedding for an image and then calculates the 
  # equality of the softmax prediction distribution if it is below a certain threshold
  # then the image will be classified as a digit
  temp = embeding_model.predict(x)
  temp = Classifier_model.predict(temp)
  if gini(temp) < 0.87:
    print(np.argmax(temp))
  else:
    print('Input is not a Digit')
    
DigitOrNumber(X_test[20:21])
DigitOrNumber(X_test[500:501])
DigitOrNumber(X_test[1007:1008]) """
     
     


""" eval_model = Model(inputs=anchor_input, outputs=encoded_anchor)
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

print(positive_similarity, negative_similarity) """
