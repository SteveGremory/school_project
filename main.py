from train_model import train_model
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas
import cv2
import os

# Create a new model if it doesn't exist
if os.path.exists("writtendigits.model"):
    model = tf.keras.models.load_model("writtendigits.model")
else:
    train_model()
    model = tf.keras.models.load_model("writtendigits.model")

# Read the data with the correct information
data = pandas.read_csv("testdata/numbers.csv")
correct = []

for image_number in data["num"]:
    if os.path.exists(f"./testdata/number_{image_number}.png"):

        image = cv2.imread(f"./testdata/number_{image_number}.png")[:, :, 0]
        image = np.invert(np.array([image]))

        predection = model.predict(image)

        if np.argmax(predection) == image_number:
            correct += [1]
        else:
            correct += [0]

        print("The predection is incorrect:")
        print(f"Number: {image_number}\t Predection: {np.argmax(predection)}")

# Display a graph of correct/incorrect numbers
# data to be plotted
x = np.arange(0, 9)
y = np.array(correct)

# plotting
plt.title("Model accuracy")
plt.xlabel("Predection")
plt.ylabel("Number")
plt.plot(x, y, color="green")

plt.show()
