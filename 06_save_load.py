import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from tensorflow import keras
import numpy as np

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train, x_test = x_train/255.0, x_test/255.0

# Feed forward neural network
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

# config
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = ['accuracy']

# compile
model.compile(optimizer=optim, loss=loss, metrics=metrics)

# fit/train
model.fit(x_train, y_train, batch_size=64, epochs=5, shuffle=True, verbose=2)

# evaluate
print("Evaluate: ")
model.evaluate(x_test, y_test, verbose=2)

# 1) Save whole model
# SavedModel
# model.save("neural_net")
# # HDF5
# model.save("nn.h5")
new_model = keras.models.load_model("nn.h5")
# new_model.evaluate(x_test, y_test, verbose=2)
# print(new_model.predict(x_test))

# 2) Save weights only
model.save_weights("nn_weights.h5")
# init
model.load_weights("nn_weights.h5")

# 3) Save architecture only, to_json
json_string = model.to_json()

with open("nn_arch.json", "w") as f:
    f.write(json_string)

with open("nn_arch.json", "r") as f:
    json_string = f.read()

new_model = keras.models.model_from_json(json_string)
