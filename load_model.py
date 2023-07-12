import tensorflow as tf
from tensorflow import keras
import numpy as np

new_model = keras.models.load_model("nn.h5")

print(new_model.summary())