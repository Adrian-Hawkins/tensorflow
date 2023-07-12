import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight','Acceleration','Model Year','Origin']

dataset = pd.read_csv(url, names=column_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True)

# print(dataset.tail())
dataset = dataset.dropna()
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1)*1
dataset['Europe'] = (origin == 2)*1
dataset['Japan'] = (origin == 3)*1
# print(dataset.tail())

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# print(dataset.shape, train_dataset.shape, test_dataset.shape)
# print(train_dataset.describe().transpose())

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

def plot(feature, x=None, y=None):
    plt.figure(figsize=(10,8))
    plt.scatter(train_features[feature], train_labels, label='Data')
    if x is not None and y is not None:
        plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel(feature)
    plt.ylabel('MPG')
    plt.show()

# plot('Horsepower', x=train_features['Horsepower'], y=train_features['Horsepower']*0.1-10)  
# print(train_features.describe().transpose()[['mean', 'std']])
normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))
# print(normalizer)
# print(normalizer.mean.numpy())
first = np.array(train_features[:1])
# print(first)
# with np.printoptions(precision=2, suppress=True):
#     print('First example:', first)
#     print('Normalized:', normalizer(first).numpy())

feature = 'Horsepower'
# single_feature = np.array(train_features)
single_feature = np.array(train_features[feature])
single_feature = single_feature[:, np.newaxis]
# print(single_feature.shape, train_features.shape)

single_feature_normalizer = preprocessing.Normalization(axis=None)
single_feature_normalizer.adapt(single_feature)
print('Normalized shape after adaptation:', single_feature_normalizer(single_feature).shape)
# print(single_feature_normalizer.mean.numpy())


print('Input shape:', single_feature.shape)
print('Normalized shape:', single_feature_normalizer(single_feature).shape)
single_feature_model = tf.keras.Sequential([
    layers.Input(shape=(1,)),
    single_feature_normalizer,
    layers.Dense(units=1, activation='linear')
    # layers.Dense(units=1) # linear model that applies a linear transformation to the output
])
print(single_feature_model.summary())
loss = keras.losses.MeanAbsoluteError()  # MeanSquaredError() MeanAbsoluteError() MEAN ABSOLUTE WORKS BEST IT WOULD SEEM
optim = keras.optimizers.Adam(learning_rate=0.1)

single_feature_model.compile(optimizer=optim, loss=loss)

# history = single_feature_model.fit(
#     train_features[feature], train_labels,
#     epochs=100,
#     verbose=1,
#     validation_split=0.2
# )


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 25])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.show()

# plot_loss(history)
single_feature_model.evaluate(
    test_features[feature],
    test_labels,
    verbose=1
)

range_min = np.min(test_features[feature]) - 10
range_max = np.max(test_features[feature]) + 10
x = tf.linspace(range_min, range_max, 200)
y = single_feature_model.predict(x)

# plot(feature, x, y)

# Deep neural network
dnn_model = tf.keras.Sequential([
    layers.Input(shape=(1,)),
    single_feature_normalizer,
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    # layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

# loss = keras.losses.MeanAbsoluteError()  # MeanSquaredError()
# optim = keras.optimizers.Adam(learning_rate=0.1)

dnn_model.compile(loss=loss, optimizer=keras.optimizers.Adam(0.001))
print(dnn_model.summary())

history = dnn_model.fit(
    train_features[feature], 
    train_labels,
    validation_split=0.2,
    verbose=1,
    epochs=100
)
# plot_loss(history)
dnn_model.evaluate(
    test_features[feature],
    test_labels,
    verbose=1
)
x = tf.linspace(range_min, range_max, 200)
y = dnn_model.predict(x)
print(y)
# plot(feature, x, y)


