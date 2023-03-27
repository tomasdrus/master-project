import yaml, time, os, argparse
import numpy as np
import pandas as pd

from munch import DefaultMunch

from models import embeding_model

from sklearn.manifold import TSNE

from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.layers import Input, Lambda
from keras import backend as K
from keras.callbacks import EarlyStopping

import visualkeras

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from ann_visualizer.visualize import ann_viz

# configuration
config = DefaultMunch.fromDict(yaml.safe_load(open("config.yml"))['siamese'])

# parsing arguments
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

# load data
data = np.load('./data/data.npz')
triplets = np.load('./data/triplets.npz')
X_train, y_train, X_test, y_test = data['X_train'], data['y_train'], data['X_test'], data['y_test'] 
X_trip, y_trip = triplets['X_trip'], triplets['y_trip']

# siamese model
def triplet_loss(_, y_pred):
    margin = K.constant(args_conf('margin'))
    return K.mean(K.maximum(K.constant(0), K.square(y_pred[:,0]) - 0.5*(K.square(y_pred[:,1])+K.square(y_pred[:,2])) + margin))

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

input_shape = (X_trip[0].shape[1], X_trip[0].shape[2], 1)
embeding_model = embeding_model(input_shape, version=args_conf('version'))

visualkeras.layered_view(embeding_model, legend=True, to_file=f'img/model-v.{args_conf("version")}.png')#.show()

anch_input = Input(input_shape, name='anch_input')
pos_input = Input(input_shape, name='pos_input')
neg_input = Input(input_shape, name='neg_input')

anch_embeding = embeding_model(anch_input)
pos_embeding = embeding_model(pos_input)
beg_embeding = embeding_model(neg_input)

pos_dist = Lambda(euclidean_distance, name='pos_dist')([anch_embeding, pos_embeding])
neg_dist = Lambda(euclidean_distance, name='neg_dist')([anch_embeding, beg_embeding])
ter_dist = Lambda(euclidean_distance, name='ter_dist')([pos_embeding, beg_embeding])

stacked_dists = Lambda(lambda vects: K.stack(vects, axis=1), name='stacked_dists')([pos_dist, neg_dist, ter_dist])

model = Model([anch_input, pos_input, neg_input], stacked_dists, name='triple_siamese')

model.compile(loss=triplet_loss, optimizer=Adam(args_conf('learning_rate')))
#model.summary()
early_stopping = EarlyStopping(patience=args_conf('patience'), restore_best_weights=True)
start_time = time.time()
history = model.fit([X_trip[0], X_trip[1], X_trip[2]], y_trip,
                    verbose=config.verbose, validation_split=args_conf('validation_split'),
                    batch_size=args_conf('batch_size'), epochs=args_conf('epochs'), callbacks=[early_stopping])
elapsed_time = time.time() - start_time

print(history.history['loss'][1])
model.save_weights('./weights/model.hdf5')

# save results
if(not os.path.exists(f"results/{config.result_name}.csv")):
    column_names = ["ver", "data", "length", "epochs", "batch", "lr", "time", "loss", "val_loss", "LOSS", "ACC", "FAR", "FRR", "EER"]
    df = pd.DataFrame(columns=column_names)
else:
    df = pd.read_csv(f'results/{config.result_name}.csv', index_col=0)

settings = np.load('./data/settings.npy', allow_pickle=True).item()

df.loc[df.shape[0]] = {
    'ver': args_conf('version'),
    'data': X_trip[0].shape,
    'length': settings['length'],
    'epochs': f'{len(history.history["loss"])}/{args_conf("epochs")}',
    'batch':args_conf('batch_size'),
    'lr':args_conf('learning_rate'),
    'loss':round(history.history['loss'][-1], 5),
    'val_loss':round(history.history['val_loss'][-1], 5),
    'time':round(elapsed_time, 2)
    }


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

