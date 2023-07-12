import os
import math
import random
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BASE_DIR = 'lego/star-wars-images/'
names = ["YODA", "LUKE SKYWALKER", "R2-D2", "MACE WINDU", "GENERAL GRIEVOUS", "KYLO REN", "MANDALORIAN","MANDALORIAN-LADY", "BAD GUY 1", "BAD GUY 2", "BOW LADY", "HAN SOLO", "DARTH VADER", "BURNT ANAKIN", "EMPORER", "OBI WAN", "BOBA FETT"]

tf.random.set_seed(1)

if not os.path.isdir(BASE_DIR+'train/'):
    for name in names:
        os.makedirs(BASE_DIR+'train/'+name)
        os.makedirs(BASE_DIR+'val/'+name)
        os.makedirs(BASE_DIR+'test/'+name)

orig_folder = ["0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008", "0009", "0010", "0011", "0012", "0013", "0014", "0015", "0016", "0017"]
for folder_idx, folder in  enumerate(orig_folder):
    files = os.listdir(BASE_DIR+folder)
    number_of_images = len([name for name in files])
    n_train = int((number_of_images*0.6) + 0.5)
    n_valid = int((number_of_images*0.25) + 0.5)
    n_test = number_of_images - n_train - n_valid
    print(number_of_images, n_train, n_valid, n_test)
    for idx, file in enumerate(files):
        file_name = BASE_DIR+folder+'/'+file
        if idx < n_train:
            shutil.move(file_name, BASE_DIR + 'train/' + names[folder_idx])
        elif idx < n_train + n_valid:
            shutil.move(file_name, BASE_DIR + 'val/' + names[folder_idx])
        else:
            shutil.move(file_name, BASE_DIR + 'test/' + names[folder_idx])

train_gen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, 
    # rotation_range=20,
    # horizontal_flip=True,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # shear_range=0.2,
    # zoom_range=0.2
)
valid_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_batches = train_gen.flow_from_directory(
    BASE_DIR+'train', 
    target_size=(256, 256),
    class_mode='sparse',
    batch_size=4,
    shuffle=True,
    color_mode="rgb",
    classes=names
)

val_batches = valid_gen.flow_from_directory(
    BASE_DIR+'val', 
    target_size=(256, 256),
    class_mode='sparse',
    batch_size=4,
    shuffle=False,
    color_mode="rgb",
    classes=names
)

test_batches = test_gen.flow_from_directory(
    BASE_DIR+'test', 
    target_size=(256, 256),
    class_mode='sparse',
    batch_size=4,
    shuffle=False,
    color_mode="rgb",
    classes=names
)

train_batch = train_batches[0]
print(train_batch[0].shape)
print(train_batch[1])
test_batch = test_batches[0]
print(test_batch[0].shape)
print(test_batch[1])

def show(batch, pred_labels=None):
    plt.figure(figsize=(10, 10))
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(batch[0][i], cmap=plt.cm.binary)
        lbl = names[int(batch[1][i])]
        if pred_labels is not None:
            lbl += "/ Pred:" + names[int(pred_labels[i])]   
        plt.xlabel(lbl)
    plt.show()

# show(test_batch)
# show(train_batch)

model = keras.Sequential([
    layers.Conv2D(32, (3,3), strides=(1,1), padding='valid', activation='relu', input_shape=(256, 256,3)),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(17)
])

# print(model.summary())
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = ["accuracy"]

model.compile(loss=loss, optimizer=optim, metrics=metrics)

epochs = 30

# callbacks


early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=2
)
history = model.fit(
    train_batches,
    validation_data=val_batches,
    callbacks=[early_stopping],
    epochs=epochs,
    verbose=2
)

model.evaluate(test_batches, verbose=2)
model.save("lego_model.h5")

# preds = model.predict(test_batches)
# preds = tf.nn.softmax(preds)
# pred_labels = np.argmax(preds, axis=1)

# print(test_batch[0][1])
# # print(names[pred_labels[0:4]])
# for i in range(4):
#     print(names[int(test_batch[1][i])], names[int(pred_labels[i])])
# from PIL import Image

# # Load and preprocess the image
# image_path = "007.jpg"
# image = Image.open(image_path)
# image = image.resize((256, 256))
# image = np.array(image) / 255.0
# image = image[np.newaxis, ...]

# predictions = model.predict(image)
# probabilities = tf.nn.softmax(predictions)
# predicted_label_index = np.argmax(probabilities)
# predicted_label = names[predicted_label_index]
# print("Predicted label:", predicted_label)
