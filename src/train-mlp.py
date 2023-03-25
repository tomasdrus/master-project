import numpy as np

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

def make_pairs(X, y):
    pairs, labels = [], []

    numClasses = len(np.unique(y))
    indexes = [np.where(y == i)[0] for i in range(0, numClasses)]

    for anchor_i in range(len(X)):
        print(anchor_i)
        anchor = X[anchor_i]
        index = y[anchor_i]

        # positive pairs
        positive_i = np.random.choice(indexes[index])
        positive = X[positive_i]

        pairs.append(np.concatenate([anchor, positive]))
        labels.append([1])

        # negative pairs
        negative_i = np.where(y != index)[0]
        negative = X[np.random.choice(negative_i)]

        pairs.append(np.concatenate([anchor, negative]))
        labels.append([0])

    # shuffle arrays
    pairs, labels = shuffle(np.array(pairs), np.array(labels))
    return (pairs, labels)

embedings = np.load('data/embedings.npz')
X_train, y_train, X_test, y_test = embedings['X_train'], embedings['y_train'], embedings['X_test'], embedings['y_test'] 


X_train_pairs, y_train_pairs = make_pairs(X_train, y_train)
X_test_pairs, y_test_pairs = make_pairs(X_test, y_test)
input_shape = (X_train_pairs.shape[1], )

# Define the neural network architecture
model = Sequential([
    Dense(256, input_shape=input_shape, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid'),
])

model.compile(loss='binary_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])
model.summary()

early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
history = model.fit(X_train_pairs, y_train_pairs, verbose=2, validation_split=0.2, batch_size=64, epochs=200, callbacks=[early_stopping])

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Training and Validation Losses',size = 20)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

score, acc = model.evaluate(X_test_pairs, y_test_pairs, verbose = 0)
print('accuracy', acc)

y_pred = model.predict(X_test_pairs)
y_pred = [1 * (x[0]>=0.5) for x in y_pred]

conf_mat = confusion_matrix(y_test_pairs, y_pred)
sns.heatmap(conf_mat, square=True, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.show()