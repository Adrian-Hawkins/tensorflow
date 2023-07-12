import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt

cifar10 = keras.datasets.cifar10

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

print(train_images.shape)

train_images , test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck' ]

# def show():
#     plt.figure(figsize=(10,10))
#     for i in range(16):
#         plt.subplot(4,4,i+1)
#         plt.xticks([])
#         plt.yticks([])
#         plt.grid(True)
#         plt.imshow(train_images[i], cmap=plt.cm.binary)
#         plt.xlabel(class_names[train_labels[i][0]])
#     plt.show()

# show()

model = keras.Sequential([
    layers.Conv2D(32, (3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
    # keras.layers.Flatten(input_shape=(32,32,3)),
    # keras.layers.Dense(128, activation='relu'),
    # keras.layers.Dense(128, activation='relu'),
    # keras.layers.Dense(128, activation='relu'),
    # keras.layers.Dense(10, activation='softmax')
])
print(model.summary())
# import sys; sys.exit()

loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = ['accuracy']

model.compile(optimizer=optim, loss=loss, metrics=metrics)
batch_size = 64
epochs = 20

history = model.fit(
    train_images, 
    train_labels, 
    batch_size=batch_size, 
    epochs=epochs, 
    verbose=2, 
)
model.evaluate(test_images, test_labels, batch_size=batch_size, verbose=1)