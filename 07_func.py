import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import numpy as np

inputs = keras.Input(shape=(28,28))
flatten = keras.layers.Flatten()
dense1 = keras.layers.Dense(128, activation='relu')

dense2 = keras.layers.Dense(10, activation='softmax', name="category_output")
dense3 = keras.layers.Dense(1, activation='sigmoid', name="leftright_output")

x = flatten(inputs)
x = dense1(x)
outputs1 = dense2(x)
outputs2 = dense3(x)

model = keras.Model(inputs=inputs, outputs=[outputs1, outputs2], name="mnist_model")

print(model.summary())

loss1 = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
loss2 = keras.losses.BinaryCrossentropy(from_logits=False)
optim = keras.optimizers.Adam(lr=0.001)
metrics = ["accuracy"]

losses = {
    "category_output": loss1,
    "leftright_output": loss2,
}

model.compile(loss=losses, optimizer=optim, metrics=metrics)

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 0=left, 1=right
y_leftright = np.zeros(y_train.shape, dtype=np.uint8)
for idx, y in enumerate(y_train):
    if y > 5:
        y_leftright[idx] = 1

print(y_train.dtype, y_train[0:20])
print(y_leftright.dtype, y_leftright[0:20])

y= {"category_output": y_train,
    "leftright_output": y_leftright }

model.fit(x_train, y=y, epochs=5,
          batch_size=64, verbose=2)

predictions = model.predict(x_test)
print(len(predictions))

prediction_category = predictions[0]
prediction_lr = predictions[1]

pr_cat = prediction_category[0:20]
prediction_lr = prediction_lr[0:20]

labels_cat = np.argmax(pr_cat, axis=1)
labels_lr = np.array([1 if p >= 0.5 else 0 for p in prediction_lr])

print(y_test[0:20])
print(labels_cat)
print(labels_lr)