import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow import keras
import numpy as np
# a
# |
# b
# |
# c

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

print(model.summary())

# Functional API
inputs = keras.Input(shape=(28,28))
flatten = keras.layers.Flatten()
dense1 = keras.layers.Dense(128, activation='relu')
dense2 = keras.layers.Dense(10)
dense2_2 = keras.layers.Dense(1)

x = flatten(inputs)
x = dense1(x)
outputs = dense2(x)
outputs2 = dense2_2(x)

model = keras.Model(inputs=inputs, outputs=[outputs,outputs2], name='functional_model')

print(model.summary())

# WTF?
# new_model = keras.models.Sequential()
# # for layer in model.layers:
# #     new_model.add(layer)
# inputs = keras.Input(shape=(28,28))
# x = new_model.layers[0](inputs)
# for layer in new_model.layers[1:-1]:
#     x = layer(x)
# outputs = x

# Models with multiple inputs and outputs
# Shared layers
# Extract and reuse nodes in the graph of layers
# (Model are callable like layers (put model into seuquential))

inputs = model.inputs
outputs = model.outputs

input0 = model.layers[0].input
output0 = model.layers[0].output

print(inputs)
print(outputs)
print(input0)
print(output0)

# Transfer learning
base_model = keras.applications.VGG16()
x = base_model.layers[-2].output
new_outputs = keras.layers.Dense(1)(x)

new_model = keras.Model(inputs=base_model.inputs, outputs=new_outputs)