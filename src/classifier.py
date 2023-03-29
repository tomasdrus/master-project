import yaml
import numpy as np
import pandas as pd

from helpers import *

from munch import DefaultMunch

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

from alive_progress import alive_bar, alive_it, config_handler
config_handler.set_global(theme='classic')

config = DefaultMunch.fromDict(yaml.safe_load(open("config.yml"))['classifier'])

def join_pair(x1, x2):
    return np.concatenate([x1, x2])

def generate_pairs(X, y, n):
    pairs, labels = [], []
    y_set = set(y)
    with alive_bar(n) as bar:
        while(len(pairs) < n):
            y1, y2 = np.random.choice(list(y_set), size=2, replace=False)
            pos_idx1, pos_idx2 = np.random.choice(np.where(y == y1)[0], size=2, replace=False)

            neg_idx1 = np.random.choice(np.where(y == y1)[0])
            neg_idx2 = np.random.choice(np.where(y == y2)[0])

            pairs.append(join_pair(X[pos_idx1], X[pos_idx2]))
            labels.append(1)
            pairs.append(join_pair(X[neg_idx1], X[neg_idx2]))
            labels.append(0)
            bar(2)

    pairs, labels = shuffle(np.array(pairs), np.array(labels))
    return pairs, labels 

embedings = np.load('data/embedings.npz')
X_train, y_train, X_test, y_test = embedings['X_train'], embedings['y_train'], embedings['X_test'], embedings['y_test'] 

X_train_pairs, y_train_pairs = generate_pairs(X_train, y_train, len(y_train) * 10)
X_test_pairs, y_test_pairs = generate_pairs(X_test, y_test, len(y_test) * 10)
input_shape = (X_train_pairs[0].shape)

# Define the neural network architecture
model = Sequential([
    Dense(256, input_shape=input_shape, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid'),
])

model.compile(loss='binary_crossentropy', optimizer=Adam(config.learning_rate), metrics=['accuracy'])
if(config.plot_summary):
    model.summary()

early_stopping = EarlyStopping(patience=config.patience, restore_best_weights=True)
history = model.fit(X_train_pairs, y_train_pairs, verbose=config.verbose,
                    validation_split=config.validation_split, batch_size=config.batch_size,
                    epochs=config.epochs, callbacks=[early_stopping])

if(config.plot_history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Training and Validation Losses',size = 20)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()

results = model.evaluate(X_test_pairs, y_test_pairs, verbose = 0)

print(f'\n ACCURACY {round(results[1], 5)}, LOSS {round(results[0], 5)}\n')

y_pred = model.predict(X_test_pairs)
treshold = 0.5
y_pred = [1 * (x[0]>=treshold) for x in y_pred]

def calculate_far_frr_eer(y_true, y_pred):
    y_true = np.array(y_true, dtype=bool)
    y_pred = np.array(y_pred, dtype=bool)

    false_accepts = np.logical_and(y_pred == 1, y_test_pairs == 0).sum()
    false_rejects = np.logical_and(y_pred == 0, y_test_pairs == 1).sum()
    true_accepts = np.logical_and(y_pred == 1, y_test_pairs == 1).sum()
    true_rejects = np.logical_and(y_pred == 0, y_test_pairs == 0).sum()

    FAR = false_accepts / (false_accepts + true_rejects)
    FRR = false_rejects / (false_rejects + true_accepts)
    EER = (FAR + FRR) / 2
    return FAR, FRR, EER

FAR, FRR, EER = calculate_far_frr_eer(y_test_pairs, y_pred)

print(f'FAR: {FAR} , FRR: {FRR} , EER: {EER} ')

df = pd.read_csv(f'results/{config.result_name}.csv', index_col=0)
df.loc[df.index[-1],'LOSS'] = round(results[0], 5)
df.loc[df.index[-1],'ACC'] = round(results[1], 5)
df.loc[df.index[-1],'FAR'] = round(FAR, 5)
df.loc[df.index[-1],'FRR'] = round(FRR, 5)
df.loc[df.index[-1],'EER'] = round(EER, 5)
df.to_csv(f'results/{config.result_name}.csv')

#print('\n',df,'\n')
print('\n',df[['train_test','data','epochs', 'batch', 'dense', 'loss', 'val_loss', 'time']],'\n')

if(config.plot_confusion):
    conf_mat = confusion_matrix(y_test_pairs, y_pred)
    sns.heatmap(conf_mat, square=True, annot=True, cmap='Blues', fmt='d', cbar=False)
    plt.show()