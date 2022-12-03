from train_model import train_model
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas
import cv2
import os

# Create a new model if it doesn't exist
if os.path.exists("written_digits.model"):
    model = tf.keras.models.load_model("written_digits.model")
else:
    train_model()
    model = tf.keras.models.load_model("written_digits.model")

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
            print(
                f"The predection is correct:\nNumber: {image_number}\t Predection: {np.argmax(predection)}"
            )
        else:
            correct += [0]
            print(
                f"The predection is incorrect:\nNumber: {image_number}\t Predection: {np.argmax(predection)}"
            )

        # Show the current number
        # plt.imshow(image[0], cmap=plt.cm.binary)
        # plt.show()

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
