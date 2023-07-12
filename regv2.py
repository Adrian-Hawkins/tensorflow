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

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

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

plot('Horsepower', x=train_features['Horsepower'], y=train_features['Horsepower']*0.1-10)
normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))
first = np.array(train_features[:1])

feature = 'Horsepower'
single_feature = np.array(train_features[feature])
single_feature = single_feature[:, np.newaxis]

single_feature_normalizer = preprocessing.Normalization(axis=None)
single_feature_normalizer.adapt(single_feature)
print('Normalized shape after adaptation:', single_feature_normalizer(single_feature).shape)

print('Input shape:', single_feature.shape)
print('Normalized shape:', single_feature_normalizer(single_feature).shape)
single_feature_model = tf.keras.Sequential([
    layers.Input(shape=(1,)),
    single_feature_normalizer,
    layers.Dense(units=1, activation='linear')
])
print(single_feature_model.summary())