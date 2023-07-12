import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model("lego_model.h5")

names = ["YODA", "LUKE SKYWALKER", "R2-D2", "MACE WINDU", "GENERAL GRIEVOUS", "KYLO REN", "MANDALORIAN",
         "MANDALORIAN-LADY", "BAD GUY 1", "BAD GUY 2", "BOW LADY", "HAN SOLO", "DARTH VADER", "BURNT ANAKIN",
         "EMPORER", "OBI WAN", "BOBA FETT"]

# Load and preprocess the image
image_path = "kyloren.jpeg"
image = Image.open(image_path)
image = image.resize((256, 256))
image = np.array(image) / 255.0
image = image[np.newaxis, ...]

predictions = model.predict(image)
probabilities = tf.nn.softmax(predictions)
predicted_label_index = np.argmax(probabilities)
predicted_label = names[predicted_label_index]
print("Predicted label:", predicted_label)
